import numpy as np
# from sklearn.base import BaseEstimator
from sklearn.preprocessing import normalize
from sliced import SlicedInverseRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
from scipy.linalg import sqrtm
import pandas as pd

class _2SLS(object):
	"""
	Two-Stage least squares (2SLS)

	Two-Stage least squares (2SLS) regression is a statistical technique that is used in the analysis of structural equations.
	(stage 1) x = z^T \theta + \omega; 		(Stage 2) y = \beta x + z^T \alpha + \epsilon

	Note that data is expected to be centered, and y is normarlized as sd(y) = 1.

	Parameters
	----------
	normalize: bool, default=True
		Whether to normalize the resulting $\theta$ in Stage 1.
	
	fit_flag: bool, default=False
		A flag to indicate if the estimation is done.

	sparse_reg: class, default=None
		A sparse regression used in the Stage 2. If set to None, we will use OLS in for the Stage 2.
	
	Attributes
	----------
	p_value: float
		P-value for hypothesis testing H_0: \beta = 0; H_a: \beta \neq 0.
	
	theta: array of shape (n_features, )
		Estimated linear coefficients for the linear regression in Stage 1.

	beta: float
		The marginal causal effect $\beta$ in Stage 2.  
	
	alpha: array of shape (n_features, )
		Estimated linear coefficients for Invalid IVs in Stage 2.

	CI: array of shape (2, )
		Estimated confidence interval for marginal causal effect (\beta).
	
	Examples
    --------
    >>> import numpy as np
	>>> from nonlinear_causal._2S_TWAS import _2SLS
	>>> from sklearn.preprocessing import StandardScaler
	>>> from sklearn.model_selection import train_test_split
	>>> n, p = 1000, 50
	>>> Z = np.random.randn(n, p)
	>>> U, eps, gamma = np.random.randn(n), np.random.randn(n), np.random.randn(n)
	>>> theta0 = np.random.randn(p)
	>>> theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	>>> beta0 = .5
	>>> X = np.dot(Z, theta0) + U**2 + eps
	>>> y = beta0 * X + U + gamma
	>>> ## normalize Z, X, y
	>>>	center = StandardScaler(with_std=False)
	>>>	mean_X, mean_y = X.mean(), y.mean()
	>>>	Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
	>>>	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
	>>>	n1, n2 = len(Z1), len(Z2)
	>>>	LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	>>>	LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	>>> ## Define 2SLS 
	>>> LS = _2S_TWAS._2SLS(sparse_reg=None)
	>>> ## Estimate theta in Stage 1
	>>> LS.fit_theta(LD_Z1, cov_ZX1)
	>>> LS.theta
	>>> array([-0.20606854,  0.02530397, -0.00135975,  0.25715265, -0.11367783,
        0.10418381,  0.03810379,  0.14383798,  0.04221932,  0.00745648,
       -0.02993945,  0.1373113 , -0.02653782,  0.1544586 , -0.0245656 ,
       -0.13433774,  0.00501199, -0.00215122, -0.11677635, -0.21730994,
       -0.04971654, -0.02443733,  0.02816777, -0.10271307,  0.0776153 ,
        0.10138465, -0.3372472 ,  0.05636817,  0.29783612, -0.10146229,
        0.0390833 , -0.11371503, -0.01425523, -0.03003318,  0.15177592,
        0.1430128 ,  0.12335511, -0.09934032, -0.26117461,  0.13902241,
       -0.35279522,  0.38152773, -0.02832687, -0.01635542, -0.06796552,
       -0.03075916, -0.1368516 , -0.03330756,  0.0251337 ,  0.06097916])
	>>> ## Estimate beta in Stage 2
	>>> LS.fit_beta(LD_Z2, cov_ZY2, n2=n2)
	>>> LS.beta
	>>> 0.5514705289617268
	>>> ## p-value for infer if causal effect is zero
	>>> LS.test_effect(n2, LD_Z2, cov_ZY2)
	>>> LS.p_value
	>>> 1.016570334366972e-118
	>>> LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
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

	def fit_theta(self, LD_Z1, cov_ZX1):
		"""
		Fit the linear model in Stage 1.

		Parameters
		----------
		LD_Z1: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the first sample: LD_Z1 = np.dot(Z1.T, Z1)

		cov_ZX1: {array-like, float} of shape (n_features, )
			Cov(Z1, X1); cov_ZX1 = np.dot(Z1.T, X1)
		
		Returns
		-------
		self: returns an theta of self.
		"""

		self.p = len(LD_Z1)
		self.theta = np.dot(np.linalg.inv( LD_Z1 ), cov_ZX1)
		if self.normalize:
			self.theta = normalize(self.theta.reshape(1, -1))[0]

	def fit_beta(self, LD_Z2, cov_ZY2, n2):
		"""
		Fit the linear model in Stage 2 based on GWAS data.

		Parameters
		----------
		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		n2: int
			The number of sample in the second GWAS dataset.

		Returns
		-------
		self: returns \beta of self.
		"""

		eps64 = np.finfo('float64').eps
		if self.sparse_reg == None:
			self.alpha = np.zeros(self.p)
			LD_X = self.theta.T.dot(LD_Z2).dot(self.theta)
			if LD_X.ndim == 0:
				self.beta = self.theta.T.dot(cov_ZY2) / LD_X
			else:
				self.beta = np.linalg.inv(LD_X).dot(self.theta.T).dot(cov_ZY2)
		elif self.sparse_reg.fit_flag:
			self.beta = self.sparse_reg.coef_[-1]
			self.alpha = self.sparse_reg.coef_[:-1]
		else:
			p = len(LD_Z2)
			LD_Z_aug = np.zeros((p+1,p+1))
			LD_Z_aug[:p,:p] = LD_Z2
			cov_ZX = np.dot(self.theta, LD_Z2)
			LD_Z_aug[-1,:p] = cov_ZX
			LD_Z_aug[:p,-1] = cov_ZX
			LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			cov_aug = np.hstack((cov_ZY2, np.dot(self.theta, cov_ZY2)))
			LD_Z_aug = LD_Z_aug + np.finfo('float32').eps*np.identity(p+1)
			pseudo_input = sqrtm(LD_Z_aug)
			pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
			ada_weight = np.ones(p+1, dtype=bool)
			ada_weight[-1] = False
			var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
			self.var_res = var_res
			self.sparse_reg.ada_weight = ada_weight
			self.sparse_reg.var_res = var_res
			self.sparse_reg.fit(pseudo_input, pseudo_output)
			criterion_lst, mse_lst = [], []
			for model_tmp in self.sparse_reg.candidate_model_:
				model_tmp = np.array(model_tmp)
				LD_Z_aug_tmp = LD_Z_aug[model_tmp[:,None], model_tmp]
				cov_aug_tmp = cov_aug[model_tmp]
				coef_aug_tmp = np.linalg.inv(LD_Z_aug_tmp).dot(cov_aug_tmp)
				# pseudo_input_tmp = sqrtm(LD_Z_aug_tmp)
				# pseudo_output_tmp = np.linalg.inv(pseudo_input_tmp).dot(cov_aug_tmp)
				# clf_tmp = LinearRegression(fit_intercept=self.sparse_reg.fit_intercept)
				# clf_tmp.fit(pseudo_input_tmp, pseudo_output_tmp)
				mse_tmp = 1. - 2 * np.dot(coef_aug_tmp, cov_aug_tmp) / n2 + coef_aug_tmp.T.dot(LD_Z_aug_tmp).dot(coef_aug_tmp) / n2
				criterion_tmp = mse_tmp / (self.var_res + eps64) + len(model_tmp) * np.log(n2) / n2
				criterion_lst.append(criterion_tmp)
				mse_lst.append(mse_tmp)
			## fit the best model
			best_model = np.array(self.sparse_reg.candidate_model_[np.argmin(criterion_lst)])
			self.best_model_ = best_model
			LD_Z_aug_tmp = LD_Z_aug[best_model[:,None], best_model]
			cov_aug_tmp = cov_aug[best_model]
			coef_aug_best = np.linalg.inv(LD_Z_aug_tmp).dot(cov_aug_tmp)
			# pseudo_input_tmp = sqrtm(LD_Z_aug_tmp)
			# pseudo_output_tmp = np.linalg.inv(pseudo_input_tmp).dot(cov_aug_tmp)
			# clf_best = LinearRegression(fit_intercept=self.sparse_reg.fit_intercept)
			# clf_best.fit(pseudo_input_tmp, pseudo_output_tmp)
			self.alpha = np.zeros(p)
			# self.sparse_reg = pycasso.Solver(pseudo_input, pseudo_output, penalty=self.reg, lambdas=lams)
			# self.sparse_reg.train()
			# ## select via BIC
			# var_eps = self.est_var_eps(n2, LD_Z, cov_ZY)
			# bic = self.bic(n2, LD_Z_aug, cov_aug, var_eps)
			# best_ind = np.argmin(bic)
			# self.beta = self.sparse_reg.coef()['beta'][best_ind][-1]
			# self.alpha = self.sparse_reg.coef()['beta'][best_ind][:-1]
			self.beta = coef_aug_best[-1]
			self.alpha[best_model[:-1]] = coef_aug_best[:-1]
		self.fit_flag = True

	def selection_summary(self):
		"""
		A summary for the result of model selection of the sparse regression in Stage 2.

		Returns
		-------

		df: dataframe
			dataframe with columns: "candidate_model", "criteria", and "mse".

		"""
		d = {'candidate_model': self.candidate_model_, 'criteria': self.criterion_lst_, 'mse': self.mse_lst_}
		df = pd.DataFrame(data=d)
		# print(df)
		return df

	# def bic(self, n2, LD_Z2, cov_ZY2, var_eps):
	# 	"""
	# 	Return BIC for list of beta on `sparse_reg`

	# 	Parameters
	# 	----------
	# 	n2: int
	# 		The number of sample on the second dataset.

	# 	LD_Z2: {array-like, float} of shape (n_features, n_features)
	# 		LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

	# 	cov_ZY2: {array-like, float} of shape (n_features, )
	# 		Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

	# 	Returns
	# 	-------
	# 	self: returns an beta of self.
	# 	"""
	# 	bic = []
	# 	for i in range(len(self.sparse_reg.coef()['beta'])):
	# 		beta_tmp = self.sparse_reg.coef()['beta'][i]
	# 		df_tmp = self.sparse_reg.coef()['df'][i]
	# 		error = ( n2 - 2*beta_tmp.dot(cov_ZY2) + beta_tmp.dot(LD_Z2).dot(beta_tmp) ) / n2 
	# 		bic_tmp = error / var_eps + np.log(n2) / n2 * df_tmp
	# 		bic.append(bic_tmp)
	# 	return bic

	def est_var_res(self, n2, LD_Z2, cov_ZY2):
		"""
		Estimated variance for y regress on Z.

		Parameters
		----------
		n2: int
			The number of sample on the second dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		Returns
		-------
		The estimated variance y regress on Z.
		"""

		alpha = np.linalg.inv(LD_Z2).dot(cov_ZY2)
		sigma_res_y = 1. - 2 * np.dot(alpha, cov_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return max(sigma_res_y, 0.) + np.finfo('float64').eps

	def fit(self, LD_Z1, cov_ZX1, LD_Z2, cov_ZY2, n2):
		"""
		Fit the linear model in Stage 2 based on GWAS data.

		Parameters
		----------

		Parameters
		----------
		LD_Z1: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the first sample: LD_Z1 = np.dot(Z1.T, Z1)

		cov_ZX1: {array-like, float} of shape (n_features, )
			Cov(Z1, X1); cov_ZX1 = np.dot(Z1.T, X1)

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		n2: int
			The number of sample in the second GWAS dataset.

		Returns
		-------
		self: return \theta of self.

		self: returns \beta of self.
		"""
		self.fit_theta(LD_Z1, cov_ZX1)
		self.fit_beta(LD_Z2, cov_ZY2, n2)
		self.fit_flag = True
	
	def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=0.95):
		"""
		Estimated confidence interval (CI) for the causal effect \beta

		Parameters
		----------
		n1: int
			The number of sample on the first dataset.

		n2: int
			The number of sample on the second dataset.

		Z1: {array-like, float} of shape (n_sample, n_features)
			Samples of Z in the first dataset, where n_sample = n1 is the number of samples in the first dataset, and n_feature is the number of features.

		X1: {array-like, float} of shape (n_sample)
			Samples of X in the first dataset, where n_sample = n1 is the number of samples in the first dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		level: float, default=0.95
			The confidence level to compute, which must be between 0 and 1 inclusive.

		Returns:
		--------
		self: returns a confidence interval of self.
		"""
		if (level >= 1) or (level <= 0):
			raise NameError('Confidence level must be in (0,1)!')

		if self.fit_flag:
			# compute variance
			var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
			ratio = n2 / n1
			var_x = np.mean( (X1 - np.dot(Z1, self.theta))**2 )
			invalid_iv = np.where(abs(self.alpha) > np.finfo('float32').eps)[0]
			if len(invalid_iv) == 0:
				var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
				ratio = n2 / n1
				# var_x = np.mean( (X1 - np.dot(Z1, self.theta))**2 )
				var_beta = var_res / self.theta.dot(LD_Z2).dot(self.theta.T) * n2
				var_beta = var_beta * (1. + ratio*var_x*(self.beta**2)/var_res )
			else:
				# compute reduced covariance matrix 
				select_mat_inv = np.linalg.inv(LD_Z2[invalid_iv[:,None], invalid_iv])
				select_cov = LD_Z2[:,invalid_iv].dot(select_mat_inv).dot(LD_Z2[invalid_iv,:])
				reduced_cov = (LD_Z2 - select_cov) / n2
				# omega_x = np.linalg.inv( self.theta.dot(reduced_cov).dot(self.theta.T) )
				omega_x = 1. / self.theta.dot(reduced_cov).dot(self.theta.T)
				mid_mat = self.theta.dot(reduced_cov).dot(np.linalg.inv(np.dot(Z1.T, Z1) / n1)).dot(reduced_cov).dot(self.theta)
				var_beta = omega_x * var_res + ratio * self.beta**2 * omega_x**2 * var_x * mid_mat
			# CI
			delta_tmp = abs(norm.ppf((1. - level)/2)) * np.sqrt(var_beta) / np.sqrt(n2)
			beta_low = self.beta - delta_tmp
			beta_high = self.beta + delta_tmp
			self.CI = np.array([beta_low, beta_high])
		else:
			raise NameError('CI can only be generated after fit!')
	
	def test_effect(self, n2, LD_Z2, cov_ZY2):
		"""
		Causal inference for the marginal causal effect.

		Parameters
		----------
		n2: int
			The number of samples in the second dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		Returns
		-------
		self: returns p-value of self.
		"""

		if self.fit_flag:
			var_eps = self.est_var_res(n2, LD_Z2, cov_ZY2)
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
	"""Two-stage instrumental regression (2SIR): sliced inverse regression + least sqaure

	Two-stage instrumental regression (2SIR) is a statistical technique used in the analysis of structural equations.
	(stage 1) \phi(x) = z^T \theta + \omega; 		(Stage 2) y = \beta \phi(x) + z^T \alpha + \epsilon

	Note that data is expected to be centered, and y is normarlized as sd(y) = 1.

	Parameters
	----------
	n_directions: int, default=False
		A number of directions for sliced inverse regression (SIR). Currently, we only focus on 1-dimensional case.

	n_slices: int, default='auto'
		A number of slices for SIR, if `n_slices='auto'`, then it is determined by int(n_sample / data_in_slice).
	
	data_in_slice: int, default=100
		A number of samples in a slice for SIR. If data_in_slice is not None, then n_slices = int(n_sample / data_in_slice).

	cond_mean: class, default=KNeighborsRegressor(n_neighbors=10)
		A nonparameteric regression model for estimate link function.

	if_fit_link: bool, default=True
		Whether to calculate the link function $\phi$ for this model.

	fit_flag: bool, default=False
		A flag to indicate if the estimation is done.

	sparse_reg: class, default=None
		A sparse regression used in the Stage 2. If set to None, we will use OLS in for the Stage 2.
	
	Attributes
	----------
	p_value: float
		P-value for hypothesis testing H_0: \beta = 0; H_a: \beta \neq 0.
	
	theta: array of shape (n_features, )
		Estimated linear coefficients for the linear regression in Stage 1.

	beta: float
		The marginal causal effect $\beta$ in Stage 2.  
	
	alpha: array of shape (n_features, )
		Estimated linear coefficients for Invalid IVs in Stage 2.
	
	rho: float
		A correction ratio for the link estimation.

	CI: array of shape (2, )
		Estimated confidence interval for marginal causal effect (\beta).

	Examples
    --------
    >>> import numpy as np
	>>> from nonlinear_causal._2SCausal import _2SLS
	>>> from sklearn.preprocessing import StandardScaler
	>>> from sklearn.model_selection import train_test_split
	>>> n, p = 1000, 50
	>>> Z = np.random.randn(n, p)
	>>> U, eps, gamma = np.random.randn(n), np.random.randn(n), np.random.randn(n)
	>>> theta0 = np.random.randn(p)
	>>> theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	>>> beta0 = .5
	>>> X = np.dot(Z, theta0) + U**2 + eps
	>>> y = beta0 * X + U + gamma
	>>> ## normalize Z, X, y
	>>>	center = StandardScaler(with_std=False)
	>>>	mean_X, mean_y = X.mean(), y.mean()
	>>>	Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
	>>>	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
	>>>	n1, n2 = len(Z1), len(Z2)
	>>>	LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	>>>	LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	>>> ## Define 2SLS 
	>>> LS = _2SCausal._2SLS(sparse_reg=None)
	>>> ## Estimate theta in Stage 1
	>>> LS.fit_theta(LD_Z1, cov_ZX1)
	>>> LS.theta
	>>> array([-0.20606854,  0.02530397, -0.00135975,  0.25715265, -0.11367783,
        0.10418381,  0.03810379,  0.14383798,  0.04221932,  0.00745648,
       -0.02993945,  0.1373113 , -0.02653782,  0.1544586 , -0.0245656 ,
       -0.13433774,  0.00501199, -0.00215122, -0.11677635, -0.21730994,
       -0.04971654, -0.02443733,  0.02816777, -0.10271307,  0.0776153 ,
        0.10138465, -0.3372472 ,  0.05636817,  0.29783612, -0.10146229,
        0.0390833 , -0.11371503, -0.01425523, -0.03003318,  0.15177592,
        0.1430128 ,  0.12335511, -0.09934032, -0.26117461,  0.13902241,
       -0.35279522,  0.38152773, -0.02832687, -0.01635542, -0.06796552,
       -0.03075916, -0.1368516 , -0.03330756,  0.0251337 ,  0.06097916])
	>>> ## Estimate beta in Stage 2
	>>> LS.fit_beta(LD_Z2, cov_ZY2, n2=n2)
	>>> LS.beta
	>>> 0.5514705289617268
	>>> ## p-value for infer if causal effect is zero
	>>> LS.test_effect(n2, LD_Z2, cov_ZY2)
	>>> LS.p_value
	>>> 1.016570334366972e-118

	"""
	def __init__(self, n_directions=1, n_slices='auto', data_in_slice=100, cond_mean=KNeighborsRegressor(n_neighbors=10), if_fit_link=True, sparse_reg=None):
		self.theta = None
		self.beta = None
		self.n_directions = n_directions
		self.n_slices = n_slices
		self.data_in_slice = data_in_slice
		self.if_fit_link = if_fit_link
		self.cond_mean = cond_mean
		self.sparse_reg = sparse_reg
		self.sir = None
		self.rho = None
		self.fit_flag = False
		self.p_value = None
		self.alpha = None
		self.CI = []


	def fit_theta(self, Z1, X1):
		"""
		Estimate \theta in Stage 1 by using sliced inverse regression (SIR). 

		Parameters
		----------
		Z1: {array-like, float} of shape (n_sample, n_features)
			Samples of Z in the first dataset, where n_sample = n1 is the number of samples in the first dataset, and n_feature is the number of features.

		X1: {array-like, float} of shape (n_sample)
			Samples of X in the first dataset, where n_sample = n1 is the number of samples in the first dataset.
		
		Returns
		-------
		self: returns \theta of self.
		"""

		if self.n_slices == 'auto':
			n_slices = int(len(Z1) / self.data_in_slice)
		else:
			n_slices = self.n_slices
		self.sir = SlicedInverseRegression(n_directions=self.n_directions, n_slices=n_slices)
		self.sir.fit(Z1, X1)
		self.theta = self.sir.directions_.flatten()
		if self.theta.shape[0] == 1:
			self.theta = self.theta.flatten()

	def fit_beta(self, LD_Z2, cov_ZY2, n2):
		"""
		Fit the linear model in Stage 2 based on GWAS data.

		Parameters
		----------
		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		n2: int
			The number of samples in the second GWAS dataset, which will be used for model selection in sparse regerssion.

		Returns
		-------
		self: returns \beta of self.
		"""

		eps64 = np.finfo('float64').eps
		p = len(LD_Z2)
		if self.sparse_reg == None:
			self.alpha = np.zeros(p)
			LD_X_sir = self.theta.T.dot(LD_Z2).dot(self.theta)
			if LD_X_sir.ndim == 0:
				self.beta = np.dot(self.theta, cov_ZY2) / LD_X_sir
			else:
				self.beta = np.linalg.inv(LD_X_sir).dot(np.dot(self.theta, cov_ZY2))
		# elif self.sparse_reg.fit_flag:
		# 	self.beta = self.sparse_reg.beta[-1]
		else:
			p = len(LD_Z2)
			# compute aug info
			LD_Z_aug = np.zeros((p+1,p+1))
			LD_Z_aug[:p,:p] = LD_Z2
			cov_ZX = np.dot(self.theta, LD_Z2)
			LD_Z_aug[-1,:p] = cov_ZX
			LD_Z_aug[:p,-1] = cov_ZX
			LD_Z_aug[-1,-1] = cov_ZX.dot(self.theta)
			cov_aug = np.hstack((cov_ZY2, np.dot(self.theta, cov_ZY2)))
			LD_Z_aug = LD_Z_aug + np.finfo('float32').eps*np.identity(p+1)
			# generate pseudo input and output
			pseudo_input = sqrtm(LD_Z_aug)
			pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
			ada_weight = np.ones(p+1, dtype=bool)
			ada_weight[-1] = False
			var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
			# est residual variance
			self.var_res = var_res
			self.sparse_reg.ada_weight = ada_weight
			self.sparse_reg.var_res = var_res
			# fit model and find the candidate models
			self.sparse_reg.fit(pseudo_input, pseudo_output)
			# bic to select best model
			self.candidate_model_ = self.sparse_reg.candidate_model_
			criterion_lst, mse_lst = [], []
			for model_tmp in self.sparse_reg.candidate_model_:
				model_tmp = np.array(model_tmp)
				LD_Z_aug_tmp = LD_Z_aug[model_tmp[:,None], model_tmp]
				cov_aug_tmp = cov_aug[model_tmp]
				coef_aug_tmp = np.linalg.inv(LD_Z_aug_tmp).dot(cov_aug_tmp)
				mse_tmp = 1. - 2 * np.dot(coef_aug_tmp, cov_aug_tmp) / n2 + coef_aug_tmp.T.dot(LD_Z_aug_tmp).dot(coef_aug_tmp) / n2
				criterion_tmp = mse_tmp / (self.var_res + eps64) + len(model_tmp) * np.log(n2) / n2
				# criterion_tmp = n2 * np.log(mse_tmp) + np.log(n2) * len(model_tmp)
				criterion_lst.append(criterion_tmp)
				mse_lst.append(mse_tmp)
			self.criterion_lst_ = criterion_lst
			self.mse_lst_ = mse_lst
			## fit the best model
			best_model = np.array(self.sparse_reg.candidate_model_[np.argmin(criterion_lst)])
			self.best_model_ = best_model
			LD_Z_aug_tmp = LD_Z_aug[best_model[:,None], best_model]
			cov_aug_tmp = cov_aug[best_model]
			coef_aug_best = np.linalg.inv(LD_Z_aug_tmp).dot(cov_aug_tmp)
			self.alpha = np.zeros(p)
			self.beta = coef_aug_best[-1]
			self.alpha[best_model[:-1]] = coef_aug_best[:-1]
		if self.beta < 0.:
			self.beta = -self.beta
			self.theta = -self.theta
		self.fit_flag = True
	
	def selection_summary(self):
		"""
		A summary for the result of model selection of the sparse regression in Stage 2.

		Returns
		-------

		df: dataframe
			dataframe with columns: "candidate_model", "criteria", and "mse".

		"""
		d = {'candidate_model': self.candidate_model_, 'criteria': self.criterion_lst_, 'mse': self.mse_lst_}
		df = pd.DataFrame(data=d)
		# print(df)
		return df

	# def bic(self, n2, LD_Z2, cov_ZY2, var_eps):
	# 	bic = []
	# 	for i in range(len(self.sparse_reg.coef()['beta'])):
	# 		beta_tmp = self.sparse_reg.coef()['beta'][i]
	# 		df_tmp = self.sparse_reg.coef()['df'][i]
	# 		error = ( n2 - 2*beta_tmp.dot(cov_ZY2) + beta_tmp.dot(LD_Z2).dot(beta_tmp) ) / n2 / var_eps
	# 		bic_tmp = error + np.log(n2) / n2 * df_tmp
	# 		bic.append(bic_tmp)
	# 	return bic
		
	def fit_link(self, Z1, X1):
		"""
		Estimate nonlinear link (\phi) in Stage 1.

		Parameters
		----------
		Z1: {array-like, float} of shape (n_sample, n_features)
			Samples of Z in the first dataset, where n_sample = n1 is the number of samples in the first dataset, and n_feature is the number of features.

		X1: {array-like, float} of shape (n_sample)
			Samples of X in the first dataset, where n_sample = n1 is the number of samples in the first dataset.
		
		Returns
		-------
		self: returns cond_mean of self.
			Returns estimated conditional mean of z^T \theta conditional on X1

		self: returns rho of self.
			Return estimated correction ratio of conditional mean. 

		"""
		X_sir = self.sir.transform(Z1).flatten()
		self.cond_mean.fit(X1=X1[:,None], y=X_sir)
		pred_mean = self.cond_mean.predict(X1[:,None])
		LD_Z_sum = np.sum(Z1[:, :, np.newaxis] * Z1[:, np.newaxis, :], axis=0)
		cross_mean_Z = np.sum(Z1 * pred_mean[:,None], axis=0)
		self.rho = (self.theta.dot(LD_Z_sum).dot(self.theta)) / np.dot( self.theta, cross_mean_Z )
		self.fit_link = True

	def fit(self, Z1, X1, LD_Z2, cov_ZY2, n2):
		"""
		Fit \theta, \beta, (and \phi) in the causal model.

		Parameters
		----------
		Z1: {array-like, float} of shape (n_sample, n_features)
			Samples of Z in the first dataset, where n_sample = n1 is the number of samples in the first dataset, and n_feature is the number of features.

		X1: {array-like, float} of shape (n_sample)
			Samples of X in the first dataset, where n_sample = n1 is the number of samples in the first dataset.
		
		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		n2: int
			The number of samples in the second GWAS dataset, which will be used for model selection in sparse regerssion.

		Returns
		-------
		self: returns \theta of self.

		self: returns \beta of self.

		self: returns cond_mean of self.
			Returns estimated conditional mean of z^T \theta conditional on X1

		self: returns rho of self.
			Return estimated correction ratio of conditional mean. 

		"""
		## Stage 1: estimate theta based on SIR
		self.fit_theta(self, Z1, X1)
		## Stage 2: estimate beta via sparse regression
		self.fit_beta(self, LD_Z2, cov_ZY2, n2)
		## Estimate link function
		if self.if_fit_link:
			self.fit_link(Z1, X1)
		self.fit_flag = True

	def link(self, X):
		"""
		Values of the link function in Stage 1 on instances X.

		Parameters
		----------
		X1: {array-like, float} of shape (n_sample)
			Samples of ``X``, where ``n_sample`` is the number of samples.

		Returns
		-------
		link: {array-like, float} of shape (n_sample)
			Returns values of the link function on instances ``X``.
		"""

		if self.if_fit_link:
			return self.rho * self.cond_mean.predict(X)
		else:
			raise NameError('You must fit a link function before evaluate it!')

	def est_var_res(self, n2, LD_Z2, cov_ZY2):
		"""
		Estimated variance for y regress on Z.

		Parameters
		----------
		n2: int
			The number of sample on the second dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		Returns
		-------
		The estimated variance y regress on Z.
		"""
		alpha = np.linalg.inv(LD_Z2).dot(cov_ZY2)
		sigma_res_y = 1 - 2 * np.dot(alpha, cov_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
		return sigma_res_y

	def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, level=.95):
		"""
		Estimated confidence interval (CI) for the causal effect \beta

		Parameters
		----------
		n1: int
			The number of sample on the first dataset.

		n2: int
			The number of sample on the second dataset.

		Z1: {array-like, float} of shape (n_sample, n_features)
			Samples of Z in the first dataset, where n_sample = n1 is the number of samples in the first dataset, and n_feature is the number of features.

		X1: {array-like, float} of shape (n_sample)
			Samples of X in the first dataset, where n_sample = n1 is the number of samples in the first dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		B_sample: int, default=1000
			The number of bootstrap for estimate CI.

		level: float, default=0.95
			The confidence level to compute, which must be between 0 and 1 inclusive.

		Returns:
		--------
		self: returns a confidence interval of self.
		"""
		if not self.fit_flag:
			self.fit_theta(Z1, X1)
			self.fit_beta(LD_Z2, cov_ZY2)
		var_eps = self.est_var_res(n2, LD_Z2, cov_ZY2)
		## compute the variance of beta
		invalid_iv = np.where(abs(self.alpha) > np.finfo('float32').eps)[0]
		select_mat_inv = np.linalg.inv(LD_Z2[invalid_iv[:,None], invalid_iv])
		select_cov = LD_Z2[:,invalid_iv].dot(select_mat_inv).dot(LD_Z2[invalid_iv,:])
		mid_cov = (LD_Z2 - select_cov) / n2
		omega_x = 1. / (self.theta.dot(mid_cov).dot(self.theta.T))
		var_beta = var_eps * omega_x  + np.finfo('float64').eps
		## resampling
		zeta = np.sqrt(var_beta)*np.random.randn(B_sample)
		eta = []
		left_tmp = np.sqrt(n2/n1)*self.beta*omega_x*self.theta.dot(mid_cov)
		for i in range(B_sample):
			B_ind = np.random.choice(n1, n1)
			Z1_B, X1_B = Z1[B_ind], X1[B_ind]
			_2SIR_tmp = _2SIR(sparse_reg=None)
			_2SIR_tmp.fit_theta(Z1_B, X1_B)
			_2SIR_tmp.theta = np.sign(np.dot(self.theta, _2SIR_tmp.theta)) * _2SIR_tmp.theta
			# _2SIR_tmp.fit_reg(LD_Z2, cov_ZY2)
			xi_tmp = np.sqrt(n1)*( _2SIR_tmp.theta - self.theta )
			eta_tmp = left_tmp.dot(xi_tmp)
			eta.append(eta_tmp)
		eta = np.array(eta)
		err = np.abs(zeta - eta)
		delta = np.quantile(err, level) / np.sqrt(n2)
		beta_low = self.beta - delta
		beta_low = max(0., beta_low)
		beta_up = self.beta + delta
		self.CI = np.array([beta_low, beta_up])

	def test_effect(self, n2, LD_Z2, cov_ZY2):
		"""
		Causal inference for the marginal causal effect.

		Parameters
		----------
		n2: int
			The number of samples in the second dataset.

		LD_Z2: {array-like, float} of shape (n_features, n_features)
			LD matrix of Z based on the second sample: LD_Z2 = np.dot(Z2.T, Z2)

		cov_ZY2: {array-like, float} of shape (n_features, )
			Matrix product of Z2 and Y2; cov_ZX2 = np.dot(Z2.T, Y2)

		Returns
		-------
		self: returns p-value of self.
		"""
		if self.fit_flag:
			var_eps = self.est_var_res(n2, LD_Z2, cov_ZY2)
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


