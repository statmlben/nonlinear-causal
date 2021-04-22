import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sliced import SlicedInverseRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
from scipy.linalg import sqrtm
import pycasso
from nonlinear_causal.variable_select import WLasso, SCAD, L0_IC

class _2SLS(object):
	"""Two-stage least square

	Parameters
	----------

	"""
	def __init__(self, normalize=True, sparse_reg=None):
		self.theta = None
		self.beta = None
		self.normalize = normalize
		self.fit_flag = False
		self.p_value = None
		self.sparse_reg = sparse_reg
		self.alpha = None
		self.p = None

	def fit_theta(self, LD_Z, cor_ZX):
		self.p = len(LD_Z)
		self.theta = np.dot(np.linalg.inv( LD_Z ), cor_ZX)
		if self.normalize:
			self.theta = normalize(self.theta.reshape(1, -1))[0]

	def fit_beta(self, LD_Z, cor_ZY, n2=None, lams=10**np.arange(-3,3,.1)):
		if self.sparse_reg == None:
			self.alpha = np.zeros(self.p)
			LD_X = self.theta.T.dot(LD_Z).dot(self.theta)
			if LD_X.ndim == 0:
				self.beta = self.theta.T.dot(cor_ZY) / LD_X
			else:
				self.beta = np.linalg.inv(LD_X).dot(self.theta.T).dot(cor_ZY)
		elif self.sparse_reg.fit_flag:
			self.beta = self.sparse_reg.coef_[-1]
			self.alpha = self.sparse_reg.coef_[:-1]
		else:
			p = len(LD_Z)
			LD_Z_aug = np.zeros((p+1,p+1))
			LD_Z_aug[:p,:p] = LD_Z
			cov_ZX = np.dot(self.theta, LD_Z)
			LD_Z_aug[-1,:p] = cov_ZX
			LD_Z_aug[:p,-1] = cov_ZX
			LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			cov_aug = np.hstack((cor_ZY, np.dot(self.theta, cor_ZY)))
			LD_Z_aug = LD_Z_aug + 1e-5*np.identity(p+1)
			pseudo_input = sqrtm(LD_Z_aug)
			pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
			ada_weight = np.ones(p+1, dtype=bool)
			ada_weight[-1] = False
			var_res = self.est_var_res(n2, LD_Z, cor_ZY)
			self.sparse_reg.ada_weight = ada_weight
			self.sparse_reg.var_res = var_res
			self.sparse_reg.fit(pseudo_input, pseudo_output)
			# self.sparse_reg = pycasso.Solver(pseudo_input, pseudo_output, penalty=self.reg, lambdas=lams)
			# self.sparse_reg.train()
			# ## select via BIC
			# var_eps = self.est_var_eps(n2, LD_Z, cor_ZY)
			# bic = self.bic(n2, LD_Z_aug, cov_aug, var_eps)
			# best_ind = np.argmin(bic)
			# self.beta = self.sparse_reg.coef()['beta'][best_ind][-1]
			# self.alpha = self.sparse_reg.coef()['beta'][best_ind][:-1]
			self.beta = self.sparse_reg.coef_[-1]
			self.alpha = self.sparse_reg.coef_[:-1]
		self.fit_flag = True

	def bic(self, n2, LD_Z2, cor_ZY2, var_eps):
		bic = []
		for i in range(len(self.sparse_reg.coef()['beta'])):
			beta_tmp = self.sparse_reg.coef()['beta'][i]
			df_tmp = self.sparse_reg.coef()['df'][i]
			error = ( n2 - 2*beta_tmp.dot(cor_ZY2) + beta_tmp.dot(LD_Z2).dot(beta_tmp) ) / n2 
			bic_tmp = error / var_eps + np.log(n2) / n2 * df_tmp
			bic.append(bic_tmp)
		return bic

	def est_var_res(self, n2, LD_Z2, cor_ZY2):
		alpha = np.linalg.inv(LD_Z2).dot(cor_ZY2)
		sigma_res_y = 1. - 2 * np.dot(alpha, cor_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return max(sigma_res_y, 0.) + np.finfo('float32').eps

	def fit(self, LD_Z, cor_ZX, cor_ZY, lams=10**np.arange(-3,3,.1)):
		self.fit_theta(LD_Z, cor_ZX)
		self.fit_beta(LD_Z, cor_ZY, lams)
	
	def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cor_ZY2, level):
		if self.fit_flag:
			var_eps = self.est_var_eps(n2, LD_Z2, cor_ZY2)
			ratio = n2 / n1
			var_x = np.mean( (X1 - np.dot(Z1, self.theta))**2 )
			var_beta = var_eps / self.theta.dot(LD_Z2).dot(self.theta.T) * n2
			# correction for variance
			var_beta = var_beta * (1. + ratio*var_x*(self.beta**2)/var_eps )
			delta_tmp = abs(norm.ppf((1. - level)/2)) * np.sqrt(var_beta) / np.sqrt(n2)
			beta_low = self.beta - delta_tmp
			beta_high = self.beta + delta_tmp
			return [beta_low, beta_high]
		else:
			raise NameError('CI can only be generated after fit!')
	
	def test_effect(self, n2, LD_Z2, cor_ZY2):
		if self.fit_flag:
			var_eps = self.est_var_res(n2, LD_Z2, cor_ZY2)
			if np.max(abs(self.alpha)) < np.finfo('float32').eps:
				var_beta = var_eps / self.theta.dot(LD_Z2).dot(self.theta.T) + np.finfo('float32').eps
				## we only test for absolute value
				Z = abs(self.beta) / np.sqrt(var_beta)
				self.p_value = 1. - norm.cdf(Z)  + norm.cdf(-Z)
			else:
				invalid_iv = np.where(abs(self.alpha) > np.finfo('float32').eps)[0]
				select_mat_inv = np.linalg.inv(LD_Z2[invalid_iv[:,None], invalid_iv])
				select_cov = LD_Z2[:,invalid_iv].dot(select_mat_inv).dot(LD_Z2[invalid_iv,:])
				select_var = self.theta.dot(select_cov).dot(self.theta.T)
				var_beta = var_eps / (self.theta.dot(LD_Z2).dot(self.theta.T) - select_var) + np.finfo('float32').eps
				Z = abs(self.beta) / np.sqrt(var_beta)
				self.p_value = 1. - norm.cdf(Z)  + norm.cdf(-Z)
			self.var_beta_ = var_beta
		else:
			raise NameError('Testing can only be conducted after fit!')

class _2SIR(object):
	"""Sliced inverse regression + least sqaure

	Parameters
	----------

	"""
	def __init__(self, theta=None, beta=None, sir=None, n_directions=1, n_slices='auto', data_in_slice=50, cond_mean=KNeighborsRegressor(n_neighbors=10), fit_link=True, reg='scad'):
		self.theta = theta
		self.beta = beta
		self.n_directions = n_directions
		self.n_slices = n_slices
		self.data_in_slice = data_in_slice
		self.fit_link = fit_link
		self.cond_mean = cond_mean
		self.sir = None
		self.rho = None
		self.fit_flag = False
		self.p_value = None
		self.sparse_reg = None
		self.reg = reg

	def fit_sir(self, Z, X):
		if self.n_slices == 'auto':
			n_slices = int(len(Z) / self.data_in_slice)
		else:
			n_slices = self.n_slices
		self.sir = SlicedInverseRegression(n_directions=self.n_directions, n_slices=n_slices)
		self.sir.fit(Z, X)
		self.theta = self.sir.directions_.flatten()
		if self.theta.shape[0] == 1:
			self.theta = self.theta.flatten()

	def fit_reg(self, LD_Z, cor_ZY):
		if self.reg == None:
			LD_X_sir = self.theta.T.dot(LD_Z).dot(self.theta)
			if LD_X_sir.ndim == 0:
				self.beta = np.dot(self.theta, cor_ZY) / LD_X_sir
			else:
				self.beta = np.linalg.inv(LD_X_sir).dot(np.dot(self.theta, cor_ZY))
		# elif self.sparse_reg.fit_flag:
		# 	self.beta = self.sparse_reg.beta[-1]
		else:
			p = len(LD_Z)
			LD_Z_aug = np.zeros((p+1,p+1))
			LD_Z_aug[:p,:p] = LD_Z
			cov_ZX = np.dot(self.theta, LD_Z)
			LD_Z_aug[-1,:p] = cov_ZX
			LD_Z_aug[:p,-1] = cov_ZX
			LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			cov_aug = np.hstack((cor_ZY, np.dot(self.theta, cor_ZY)))
			LD_Z_aug[np.diag_indices_from(LD_Z_aug)] = LD_Z_aug[np.diag_indices_from(LD_Z_aug)] + np.finfo('float32').eps
			pseudo_input = sqrtm(LD_Z_aug)
			pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
			self.sparse_reg = pycasso.Solver(pseudo_input, pseudo_output, penalty=self.reg, lambdas=lams)
			self.sparse_reg.train()
			## select via BIC
			var_eps = self.est_var_eps(n2, LD_Z, cor_ZY)
			bic = self.bic(n2, LD_Z_aug, cov_aug, var_eps)
			self.beta = self.sparse_reg.coef()['beta'][np.argmin(bic)][-1]
			# p = len(LD_Z)
			# LD_Z_aug = np.zeros((p+1,p+1))
			# LD_Z_aug[:p,:p] = LD_Z
			# cov_ZX = np.dot(self.theta, LD_Z)
			# LD_Z_aug[-1,:p] = cov_ZX
			# LD_Z_aug[:p,-1] = cov_ZX
			# LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			# cov_aug = np.hstack((cor_ZY, np.dot(self.theta, cor_ZY)))
			# self.sparse_reg.fit(LD_Z_aug, cov_aug)
			# self.beta = self.sparse_reg.beta[-1]
		if self.beta < 0.:
			self.beta = -self.beta
			self.theta = -self.theta
		self.fit_flag = True
	
	def bic(self, n2, LD_Z2, cor_ZY2, var_eps):
		bic = []
		for i in range(len(self.sparse_reg.coef()['beta'])):
			beta_tmp = self.sparse_reg.coef()['beta'][i]
			df_tmp = self.sparse_reg.coef()['df'][i]
			error = ( n2 - 2*beta_tmp.dot(cor_ZY2) + beta_tmp.dot(LD_Z2).dot(beta_tmp) ) / n2 / var_eps
			bic_tmp = error + np.log(n2) / n2 * df_tmp
			bic.append(bic_tmp)
		return bic
		
	def fit_air(self, Z, X):
		X_sir = self.sir.transform(Z).flatten()
		self.cond_mean.fit(X=X[:,None], y=X_sir)
		pred_mean = self.cond_mean.predict(X[:,None])
		LD_Z_sum = np.sum(Z[:, :, np.newaxis] * Z[:, np.newaxis, :], axis=0)
		cross_mean_Z = np.sum(Z * pred_mean[:,None], axis=0)
		self.rho = (self.theta.dot(LD_Z_sum).dot(self.theta)) / np.dot( self.theta, cross_mean_Z )

	def fit(self, Z, X, cor_ZY):
		## Stage 1: estimate theta based on SIR
		self.fit_sir(Z, X)
		## Stage 2: estimate beta via sparse regression
		self.fit_reg(Z, cor_ZY)
		## Estimate link function
		if self.fit_link:
			self.fit_air(Z, X)

	def link(self, X):
		if self.fit_link:
			return self.rho * self.cond_mean.predict(X)
		else:
			raise NameError('You must fit a link function before evaluate it!')

	def est_var_eps(self, n2, LD_Z2, cor_ZY2):
		alpha = np.linalg.inv(LD_Z2).dot(cor_ZY2)
		sigma_res_y = 1 - 2 * np.dot(alpha, cor_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return sigma_res_y

	# def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cor_ZY2, level=.95):
	# 	if self.fit_flag:
	# 		var_eps = self.est_var_eps(n2, LD_Z2, cor_ZY2)
	# 		ratio = n2 / n1
	# 		var_x = np.mean( (X1 - np.dot(Z1, self.theta))**2 )
	# 		var_beta = var_eps / self.theta.dot(LD_Z2).dot(self.theta.T) * n2
	# 		# correction for variance
	# 		var_beta = var_beta * (1. + ratio*var_x*(self.beta**2)/var_eps )
	# 		delta_tmp = abs(norm.ppf((1. - level)/2)) * np.sqrt(var_beta) / np.sqrt(n2)
	# 		beta_low = self.beta - delta_tmp
	# 		beta_high = self.beta + delta_tmp
	# 		return [beta_low, beta_high]
	# 	else:
	# 		raise NameError('CI can only be generated after fit!')

	def test_effect(self, n2, LD_Z2, cor_ZY2):
		if self.fit_flag:
			var_eps = self.est_var_eps(n2, LD_Z2, cor_ZY2)
			var_beta = var_eps / self.theta.dot(LD_Z2).dot(self.theta.T)
			Z = self.beta/np.sqrt(var_beta)
			self.p_value = 1. - norm.cdf(Z) + norm.cdf(-Z)
		else:
			raise NameError('Testing can only be conducted after fit!')


