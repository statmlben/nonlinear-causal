import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sliced import SlicedInverseRegression
from sklearn.neighbors import KNeighborsRegressor
from nonlinear_causal.CDLoop import elastCD
from scipy.stats import norm

class elasticSUM(object):
	"""Elastic Net based on summary statistics

	Parameters
	----------
	"""

	def __init__(self, lam=10., lam2=0., max_iter=1000, eps=1e-4, print_step=1, criterion='bic'):
		self.lam = lam
		self.lam2 = lam2
		self.beta = []
		self.max_iter = max_iter
		self.eps = eps
		self.print_step = print_step
		self.criterion = criterion
		self.fit_flag = False
	
	def fit(self, LD_X, cov):
		"""
		fit the linear coeff based on feature and summary statistics.

		Parameters
		----------
		"""
		if (type(self.lam) == int or float):
			self.beta = elastCD(LD_X, cov, self.lam, self.lam2, self.max_iter, self.eps, self.print_step)
		else:
			lam_lst, beta_path, df_lst = [], [], []
			for lam_tmp in self.lam:
				beta_tmp = elastCD(LD_X, cov, lam_tmp, self.lam2, self.max_iter, self.eps, self.print_step)
				df_tmp = np.sum(np.abs(beta_tmp) > self.eps)
				df_lst.append(df_tmp)
				beta_path.append(beta_tmp)
				lam_lst.append(lam_tmp)
			lam_lst, beta_path, df_lst = np.array(lam_lst), np.array(beta_path), np.array(df_lst)
			## compute criteria
		self.fit_flag = True

	def predict(self, X):
		return np.dot(X, self.beta)

class _2SLS(object):
	"""Two-stage least square

	Parameters
	----------

	"""
	def __init__(self, theta=None, beta=None, normalize=True, sparse_reg=elasticSUM()):
		self.theta = theta
		self.beta = beta
		self.normalize = normalize
		self.sparse_reg = sparse_reg
		self.fit_flag = False

	def fit_theta(self, LD_Z, cor_ZX):
		self.theta = np.dot(np.linalg.inv( LD_Z ), cor_ZX)
		if self.normalize:
			self.theta = normalize(self.theta.reshape(1, -1))[0]

	def fit_beta(self, LD_Z, cor_ZY):
		if self.sparse_reg == None:
			LD_X = self.theta.T.dot(LD_Z).dot(self.theta)
			if LD_X.ndim == 0:
				self.beta = self.theta.T.dot(cor_ZY) / LD_X
			else:
				self.beta = np.linalg.inv(LD_X).dot(self.theta.T).dot(cor_ZY)
		elif self.sparse_reg.fit_flag:
			self.beta = self.sparse_reg.beta[-1]
		else:
			p = len(LD_Z)
			LD_Z_aug = np.zeros((p+1,p+1))
			LD_Z_aug[:p,:p] = LD_Z
			cov_ZX = np.dot(self.theta, LD_Z)
			LD_Z_aug[-1,:p] = cov_ZX
			LD_Z_aug[:p,-1] = cov_ZX
			LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			cov_aug = np.hstack((cor_ZY, np.dot(self.theta, cor_ZY)))
			self.sparse_reg.fit(LD_Z_aug, cov_aug)
			self.beta = self.sparse_reg.beta[-1]
		self.fit_flag = True

	def est_var_eps(self, n2, LD_Z2, cor_ZY2):
		alpha = np.linalg.inv(LD_Z2).dot(cor_ZY2)
		sigma_res_y = 1 - 2 * np.dot(alpha, cor_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return sigma_res_y

	def fit(self, LD_Z, cor_ZX, cor_ZY):
		self.fit_theta(LD_Z, cor_ZX)
		self.fit_beta(LD_Z, cor_ZY)
	
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


class _2SIR(object):
	"""Sliced inverse regression + least sqaure

	Parameters
	----------

	"""
	def __init__(self, theta=None, beta=None, sir=None, n_directions=1, n_slices='auto', data_in_slice=50, cond_mean=KNeighborsRegressor(n_neighbors=10), fit_link=True, sparse_reg=elasticSUM()):
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
		self.sparse_reg = sparse_reg

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

	def fit_reg(self, Z, cor_ZY):
		X_sir = self.sir.transform(Z)
		if X_sir.shape[1] == 1:
			X_sir = X_sir.flatten()
		if self.sparse_reg == None:
			LD_X_sir = np.dot(X_sir.T, X_sir)
			if LD_X_sir.ndim == 0:
				self.beta = np.dot(self.theta, cor_ZY) / LD_X_sir
			else:
				self.beta = np.linalg.inv(LD_X_sir).dot(np.dot(self.theta, cor_ZY))
		elif self.sparse_reg.fit_flag:
			self.beta = self.sparse_reg.beta[-1]
		else:
			Z_aug = np.hstack((Z, X_sir[:,None]))
			cov_aug = np.hstack((cor_ZY, np.dot(self.theta, cor_ZY)))
			LD_Z_aug = np.dot(Z_aug.T, Z_aug)
			self.sparse_reg.fit(LD_Z_aug, cov_aug)
			self.beta = self.sparse_reg.beta[-1]
		if self.beta < 0.:
			self.beta = -self.beta
			self.theta = -self.theta
		self.fit_flag = True
		
	def fit_air(self, Z, X):
		X_sir = self.sir.transform(Z)
		self.cond_mean.fit(X=X[:,None], y=X_sir)
		pred_mean = self.cond_mean.predict(X=X[:,None])
		LD_Z_sum = np.sum(Z[:, :, np.newaxis] * Z[:, np.newaxis, :], axis=0)
		cross_mean_Z = np.sum(Z * pred_mean, axis=0)
		self.rho = (self.theta.dot(LD_Z_sum) * self.theta).sum(axis=1) / np.dot( self.theta, cross_mean_Z )

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
			return self.rho * self.cond_mean.predict(X=X)
		else:
			raise NameError('You must fit a link function before evaluate it!')

	def est_var_eps(self, n2, LD_Z2, cor_ZY2):
		alpha = np.linalg.inv(LD_Z2).dot(cor_ZY2)
		sigma_res_y = 1 - 2 * np.dot(alpha, cor_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return sigma_res_y

	def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cor_ZY2, level=.95):
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

	# def CI_beta(self, Z, alpha):
	# 	if self.fit_flag:
	# 		cov_Z = np.cov(Z,rowvar=False)
	# 		var_beta = 1. / self.theta.dot(cov_Z).dot(self.theta.T)[0][0]
	# 		delta_tmp = abs(norm.ppf((1. - alpha)/2)) * np.sqrt(var_beta) / np.sqrt(len(Z))
	# 		beta_low = self.beta - delta_tmp
	# 		beta_high = self.beta + delta_tmp
	# 		return [beta_low, beta_high]
	# 	else:
	# 		raise NameError('CI can only be generated after fit!')


