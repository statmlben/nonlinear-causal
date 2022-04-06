## Partial functions and docs are adapted from sklearn.linear_model._base.LinearModel
## Author: Ben Dai

from sklearn.base import RegressorMixin
from sklearn.linear_model import Lasso, LinearRegression, LassoLarsIC, LassoCV
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model._coordinate_descent import LinearModelCV
import numpy as np
import pandas as pd


class WLasso(RegressorMixin, LinearModel):
	"""	
	Linear Model trained with Weighted L1 prior as regularizer (aka the weighted-Lasso)
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * sum_{j=1}^d weight_j * |w_j|
	Technically the Weighted Lasso model is optimizing the same objective function as
	the Lasso with X = X / ada_weight[None,:].

	Parameters
	===========
	alpha: float, default=1.0
		Constant that multiplies the L1 term. Defaults to 1.0.
		``alpha = 0`` is equivalent to an ordinary least square, solved
		by the :class:`LinearRegression` object. For numerical
		reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
		Given this, you should use the :class:`LinearRegression` object.

	ada_weight: ndarray of shape (n_features,)
		Weight that multiplies the L1 term for each coefficient. Defaults to 1.0.

	fit_intercept: bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).

	normalize: bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.

	precompute: 'auto', bool or array-like of shape (n_features, n_features),\
					default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.

	copy_X: bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.

	max_iter: int, default=1000
		The maximum number of iterations.

	tol: float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.

	warm_start: bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.

	positive: bool, default=False
		When set to ``True``, forces the coefficients to be positive.

	random_state: int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.

	selection: {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.

	Attributes
	===========
	coef_: ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).

	dual_gap_: float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.

	sparse_coef_: sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.

	intercept_: float or ndarray of shape (n_targets,)
		Independent term in decision function.

	n_iter_: int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.

	Examples
	==========
	>>> from nl_causal import sparse_reg
	>>> clf = sparse_reg.WLasso(alpha=0.1, ada_weight=[1.,0.])
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	>>> print(clf.coef_)
	[0.		, 0.99999998]
	>>> print(clf.intercept_)
	1.7881393254981504e-08
	>>> clf = sparse_reg.WLasso(alpha=0.1, ada_weight=[0.,1.])
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	>>> print(clf.coef_)
	[0.99999998 0.		]
	>>> print(clf.intercept_)
	1.7881393254981504e-08
	"""
	
	def __init__(self, alpha=1.0, *, ada_weight=1.0, fit_intercept=True,
				 precompute=False, copy_X=True, max_iter=1000,
				 tol=1e-4, warm_start=False, positive=False,
				 random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		# self.normalize = normalize
		self.precompute = precompute
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.tol = tol
		self.warm_start = warm_start
		self.positive = positive
		self.random_state = random_state
		self.selection = selection

	def fit(self, X, y, sample_weight=None):
		"""
		Fit linear model.

		Parameters
		-----------
		X: {array-like, sparse matrix} of shape (n_samples, n_features)
			Training data
		y: array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values. Will be cast to X's dtype if necessary
		sample_weight: array-like of shape (n_samples,), default=None
			Individual weights for each sample

		Returns
		--------
		self: returns an instance of self.
		"""
		X, y = np.array(X), np.array(y)
		n_feature = X.shape[1]
		lasso_tmp = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
						  precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
				 		  tol=self.tol, warm_start=self.warm_start, positive=self.positive,
				 		  random_state=self.random_state, selection=self.selection)
		ada_weight = self.ada_weight
		if (type(ada_weight) is float) or (type(ada_weight) is int):
			X = X / ada_weight
		else:
			if len(ada_weight) == n_feature:
				ada_weight = ada_weight + np.finfo('float32').eps
				X = X / ada_weight[None,:]
			else:
				raise NameError('The dimention for adaweight should be same as n_feature!')
		
		lasso_tmp.fit(X, y, sample_weight)
		self.coef_ = lasso_tmp.coef_ / ada_weight
		self.intercept_ = lasso_tmp.intercept_
		self.sparse_coef_ = lasso_tmp.sparse_coef_

class SCAD(RegressorMixin, LinearModel):
	"""
	Linear Model trained with Weighted SCAD prior as regularizer (aka the weighted-SCAD)
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * sum_{j=1}^d weight_j * SCAD(|w_j|)
	
	Parameters
	-----------
	alpha: float, default=1.0
		Constant that multiplies the SCAD penalty. Defaults to 1.0.
		``alpha = 0`` is equivalent to an ordinary least square, solved
		by the :class:`LinearRegression` object. For numerical
		reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
		Given this, you should use the :class:`LinearRegression` object.
	
	ada_weight: ndarray of shape (n_features,)
		Weight that multiplies the L1 term for each coefficient. Defaults to 1.0.
	
	fit_intercept: bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).
	
	normalize: bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.
	
	precompute: 'auto', bool or array-like of shape (n_features, n_features),\
				 default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.
	
	copy_X: bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.
	
	max_iter: int, default=1000
		The maximum number of iterations.
	
	tol: float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.
	
	warm_start: bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.
	
	positive: bool, default=False
		When set to ``True``, forces the coefficients to be positive.
	
	random_state: int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.
	
	selection: {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.

	Attributes
	-----------
	coef_: ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).
	
	dual_gap_: float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.
	
	sparse_coef_: sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.
	
	intercept_: float or ndarray of shape (n_targets,)
		Independent term in decision function.
	
	n_iter_: int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.
	
	Examples
	---------
	>>> from nl_causal import sparse_reg
	>>> clf = sparse_reg.SCAD(alpha=0.1)
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	>>> print(clf.coef_)
	[0.99999998 0.		]
	>>> print(clf.intercept_)
	1.7881393254981504e-08
	"""
	
	def __init__(self, alpha=1.0, *, ada_weight=1.0, fit_intercept=True,
				 precompute=False, copy_X=True, max_iter=1000,
				 tol=1e-4, warm_start=False, positive=False,
				 random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		# self.normalize = normalize
		self.precompute = precompute
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.tol = tol
		self.warm_start = warm_start
		self.positive = positive
		self.random_state = random_state
		self.selection = selection
		self.sol_path_ = []

	def fit(self, X, y, sample_weight=None):
		"""
		Fit linear model.

		Parameters
		-----------
		X: {array-like, sparse matrix} of shape (n_samples, n_features)
			Training data
		y: array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values. Will be cast to X's dtype if necessary
		sample_weight: array-like of shape (n_samples,), default=None
			Individual weights for each sample

		Returns
		--------
		self: returns an instance of self.
		"""
		Wlasso_tmp = WLasso(alpha=self.alpha, ada_weight=self.ada_weight, fit_intercept=self.fit_intercept, 
						   precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
						   tol=self.tol, warm_start=self.warm_start, positive=self.positive,
						   random_state=self.random_state, selection=self.selection)
		Wlasso_tmp.fit(X, y, sample_weight)
		self.coef_ = Wlasso_tmp.coef_
		self.intercept_ = Wlasso_tmp.intercept_
		coef_old = Wlasso_tmp.coef_
		for i in range(self.max_iter):
			# update the weights
			ada_weight = self.ada_weight*self.grad_SCAD_()
			# update coef
			Wlasso_tmp.ada_weight = ada_weight
			Wlasso_tmp.fit(X, y, sample_weight)
			self.coef_ = Wlasso_tmp.coef_
			self.intercept_ = Wlasso_tmp.intercept_
			self.sparse_coef_ = Wlasso_tmp.sparse_coef_
			self.sol_path_.append(list(Wlasso_tmp.coef_))
			if np.max(abs(coef_old - Wlasso_tmp.coef_)) < self.tol:
				break
			coef_old = Wlasso_tmp.coef_
	
	def grad_SCAD_(self, a=3.7):
		"""
		Compute first-order gradient of SCAD
		"""
		abs_coef = abs(self.coef_)
		return self.alpha*(abs_coef <= self.alpha) + np.maximum(a*self.alpha - abs_coef, 0.) / (a - 1) * (abs_coef > self.alpha)

class SCAD_IC(RegressorMixin, LinearModelCV):
	"""
	Linear Model Selection trained with L0 prior as regularizer
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * sum_{j=1}^d weight_j * SCAD(|w_j|)
	
	Parameters
	-----------
	alphas: float, default=1.0
		List of alphas where to compute the SCAD. default=np.arange(-3,3,.1)
	
	mask: ndarray of shape (n_features,); dtype = bool
		Indicator to count the variable in L0 term. default = 'full'
	
	fit_intercept: bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).
	
	normalize: bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.
	
	precompute: 'auto', bool or array-like of shape (n_features, n_features),\
				 default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.
	
	copy_X: bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.
	
	max_iter: int, default=1000
		The maximum number of iterations.
	
	tol: float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.
	
	warm_start: bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.
	
	positive: bool, default=False
		When set to ``True``, forces the coefficients to be positive.
	
	random_state: int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.
	
	selection: {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.
	
	Attributes
	-----------
	
	coef_: ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).
	
	dual_gap_: float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.
	
	sparse_coef_: sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.
	
	intercept_: float or ndarray of shape (n_targets,)
		Independent term in decision function.
	
	n_iter_: int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.
	
	Examples
	---------
	>>> from nl_causal import sparse_reg
	>>> clf = sparse_reg.SCAD_IC(alphas=[.001, .01, .1, 1.])
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	>>> print(clf.coef_)
	[1. 0.]
	>>> print(clf.intercept_)
	1.7881396363605973e-10
	>>> clf.selection_summary()
   		alpha 	model	  criteria		   mse
	0  	0.001   [0]  	3.663001e-01  2.131628e-20
	1  	0.010   [0]  	3.758041e-01  2.131628e-18
	2  	0.100   [0]  	1.326204e+00  2.131628e-16
	3  	1.000	[]  	3.002400e+15  6.666667e-01
	"""

	def __init__(self, alphas, *, criterion='bic', ada_weight=1.0, fit_intercept=True,
				precompute=False, copy_X=True, max_iter=1000, var_res = None,
				tol=1e-4, warm_start=False, positive=False,
				random_state=None, selection='cyclic'):
		self.alphas = alphas
		self.criterion = criterion
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		# self.normalize = normalize
		self.precompute = precompute
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.tol = tol
		self.warm_start = warm_start
		self.positive = positive
		self.random_state = random_state
		self.selection = selection
		self.var_res = var_res
		self.best_estimator = None
		self.fit_flag = False

	def fit(self, X, y, sample_weight=None):
		"""
		Fit linear model.

		Parameters
		-----------

		X: {array-like, sparse matrix} of shape (n_samples, n_features)
			Training data
		y: array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values. Will be cast to X's dtype if necessary
		sample_weight: array-like of shape (n_samples,), default=None
			Individual weights for each sample

		Returns
		--------
		self: returns an instance of self.
		"""
		X, y = np.array(X), np.array(y)
		n_sample, n_feature = X.shape
		eps64 = np.finfo('float64').eps
		self.ada_weight = self.ada_weight * np.ones(n_feature)
		scad_tmp = SCAD(ada_weight=self.ada_weight, fit_intercept=self.fit_intercept,
						precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
						tol=self.tol, warm_start=self.warm_start, positive=self.positive, 
						random_state=self.random_state, selection=self.selection)
		if self.var_res == None:
			clf_full = LinearRegression(fit_intercept=self.fit_intercept)
			clf_full.fit(X, y)
			var_res = np.mean(( y - clf_full.predict(X) )**2)
			self.var_res = var_res
		criterion_lst, mse_lst, model_lst = [], [], []
		for alpha_tmp in self.alphas:
			scad_tmp.alpha = alpha_tmp
			scad_tmp.fit(X, y, sample_weight)
			res = y - scad_tmp.predict(X)
			mse_tmp = np.mean(res**2)
			model_tmp = np.where( abs(scad_tmp.coef_) > eps64 )[0]
			df_tmp = len(model_tmp)
			if self.criterion == 'bic':
				criterion_tmp = mse_tmp / (self.var_res + eps64) + df_tmp * np.log(n_sample) / n_sample
			elif self.criterion == 'aic':
				criterion_tmp = mse_tmp / (self.var_res + eps64) + df_tmp * 2 / n_sample
			else:
				raise NameError('criteria should be aic or bic')
			criterion_lst.append(criterion_tmp)
			mse_lst.append(mse_tmp)
			model_lst.append(model_tmp)
		self.criterion_lst_ = criterion_lst
		self.mse_lst_ = mse_lst
		self.model_lst_ = model_lst
		## best model
		best_alpha = self.alphas[np.argmin(criterion_lst)]
		scad_tmp.alpha = best_alpha
		scad_tmp.fit(X, y, sample_weight)
		self.coef_ = scad_tmp.coef_
		self.intercept_ = scad_tmp.intercept_
		self.best_estimator = scad_tmp
		self.fit_flag = True
	
	def _get_estimator(self):
		return SCAD()

	def _is_multitask(self):
		return False

	def _more_tags(self):
		return {'multioutput': False}
	
	def selection_summary(self):
		"""
		A summary for the result of model selection of the sparse regression in Stage 2.

		Returns
		--------
		df: dataframe
			dataframe with columns: "candidate_model", "criteria", and "mse".
		"""
		d = {'alpha': self.alphas, 'model': self.model_lst_, 'criteria': self.criterion_lst_, 'mse': self.mse_lst_}
		df = pd.DataFrame(data=d)
		# print(df)
		return df
		
class L0_IC(LassoLarsIC):
	"""
	Linear Model Selection trained with L0 prior as regularizer
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2, s.t. ||w||_0 <= K

	Parameters
	----------
	
	Ks: range of int, default=range(1,10)
		Number of nonzero coef to be tuned.
	
	alphas: float, default=1.0
		List of alphas where to compute the SCAD. default=np.arange(-3,3,.1)
	
	mask: ndarray of shape (n_features,); dtype = bool
		Indicator to count the variable in L0 term. default = 'full'
	
	fit_intercept: bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).
	
	normalize: bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.
	
	precompute: 'auto', bool or array-like of shape (n_features, n_features),\
				 default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.
	
	copy_X: bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.
	
	max_iter: int, default=1000
		The maximum number of iterations.
	
	tol: float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.
	
	warm_start: bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.
	
	positive: bool, default=False
		When set to ``True``, forces the coefficients to be positive.
	
	random_state: int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.
	
	selection: {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.
	
	Attributes
	----------
	
	coef_: ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).
	
	dual_gap_: float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.
	
	sparse_coef_: sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.
	
	intercept_: float or ndarray of shape (n_targets,)
		Independent term in decision function.
	
	n_iter_: int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.
	
	Examples
	--------
	>>> from nl_causal import sparse_reg
	>>> clf = sparse_reg.L0_IC(Ks=[1,2])
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	>>> print(clf.coef_)
	[1. 0.]
	>>> print(clf.intercept_)
	2.220446049250313e-16
	>>> clf.selection_summary()
		model	 	criteria		   mse
	0	(0,)  		3.662041e-01  3.286920e-32
	1	()  		3.002400e+15  6.666667e-01
	"""

	def __init__(self, alphas, criterion='bic', *, Ks=range(10), ada_weight=True, fit_intercept=True,
				precompute=False, copy_X=True, max_iter=1000, verbose=False, eps=np.finfo(float).eps,
				tol=1e-4, warm_start=False, positive=False, var_res = None, refit = True, find_best=True,
				random_state=None, selection='cyclic'):
		self.criterion = criterion
		self.Ks = Ks
		self.alphas = alphas
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		# self.normalize = normalize
		self.precompute = precompute
		self.max_iter = max_iter
		self.verbose = verbose
		self.copy_X = copy_X
		self.tol = tol
		self.warm_start = warm_start
		self.positive = positive
		self.random_state = random_state
		self.selection = selection
		self.fit_flag = False
		self.eps = eps
		self.var_res = var_res
		self.refit = refit
		self.find_best = find_best
		self.criterion_lst_ = []
		self.mse_lst_ = []

	def fit(self, X, y, sample_weight=None):
		"""
		Fit linear model.

		Parameters
		----------
		X: {array-like, sparse matrix} of shape (n_samples, n_features)
			Training data
		y: array-like of shape (n_samples,) or (n_samples, n_targets)
			Target values. Will be cast to X's dtype if necessary
		sample_weight: array-like of shape (n_samples,), default=None
			Individual weights for each sample

		Returns
		-------
		self: returns an instance of self.
		"""
		X, y = np.array(X), np.array(y)
		n_sample, n_feature = X.shape
		eps64 = np.finfo('float64').eps
		pre_select = list(np.where(self.ada_weight==False)[0])
		self.ada_weight = self.ada_weight * np.ones(n_feature, dtype=bool)
		if all(self.ada_weight):
			scad_tmp = SCAD(fit_intercept=self.fit_intercept, precompute=self.precompute, 
							copy_X=self.copy_X, max_iter=self.max_iter,
							tol=self.tol, warm_start=self.warm_start, positive=self.positive,
							random_state=self.random_state, selection=self.selection)
		else:
			scad_tmp = SCAD(ada_weight=1.*self.ada_weight, fit_intercept=self.fit_intercept, 
							precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
							tol=self.tol, warm_start=self.warm_start, positive=self.positive,
							random_state=self.random_state, selection=self.selection)
		candidate_model = []
		for alpha_tmp in self.alphas:
			scad_tmp.alpha = alpha_tmp
			scad_tmp.fit(X, y, sample_weight)
			## we don't select the features with mask = False
			abs_coef = abs(scad_tmp.coef_) * self.ada_weight
			if sum(abs_coef > eps64) == 0:
				candidate_model.append(pre_select)
			else:
				nz_ind = np.argsort(abs_coef)[-sum(abs_coef > eps64):][::-1]
				if len(nz_ind) > 0:
					for K_tmp in self.Ks:
						if K_tmp > len(nz_ind):
							break
						nz_ind_tmp = list(set(nz_ind[:K_tmp]))
						nz_ind_tmp.extend(pre_select)
						candidate_model.append(nz_ind_tmp)
		candidate_model = set(map(tuple, candidate_model))
		candidate_model = list(candidate_model)
		self.candidate_model_ = candidate_model
		if self.find_best:
			## fit largest model to get variance
			if self.var_res == None:
				clf_full = LinearRegression(fit_intercept=self.fit_intercept)
				clf_full.fit(X, y)
				var_res = np.mean(( y - clf_full.predict(X) )**2)
				self.var_res = var_res
			## find the best model
			criterion_lst, mse_lst = [], []
			for model_tmp in candidate_model:
				if len(model_tmp) == 0:
				# if model_tmp == []:
					if self.fit_intercept:
						res = y - y.mean()
					else:
						res = y
				else:
					model_tmp = np.array(model_tmp)
					clf_tmp = LinearRegression(fit_intercept=self.fit_intercept)
					clf_tmp.fit(X[:,model_tmp], y, sample_weight)
					res = y - clf_tmp.predict(X[:,model_tmp])
				mse_tmp = np.mean(res**2)
				if self.criterion == 'bic':
					criterion_tmp = mse_tmp / (self.var_res + eps64) + len(model_tmp) * np.log(n_sample) / n_sample
				elif self.criterion == 'aic':
					criterion_tmp = mse_tmp / (self.var_res + eps64) + len(model_tmp) * 2 / n_sample
				else:
					raise NameError('criteria should be aic or bic')
				criterion_lst.append(criterion_tmp)
				mse_lst.append(mse_tmp)
			self.criterion_lst_ = criterion_lst
			self.mse_lst_ = mse_lst
			## best model
			if self.refit:
				best_model = np.array(candidate_model[np.argmin(criterion_lst)])
				if len(best_model) == 0:
					if self.fit_intercept:
						self.coef_ = np.zeros(n_feature)
						self.intercept_ = y.mean()
					else:
						self.coef_ = np.zeros(n_feature)
						self.intercept_ = 0.
				else:
					clf_best = LinearRegression(fit_intercept=self.fit_intercept)
					clf_best.fit(X[:,best_model], y)
					self.coef_ = np.zeros(n_feature)
					self.coef_[best_model] = clf_best.coef_
					self.intercept_ = clf_best.intercept_
					# self.dual_gap_ = clf_best.dual_gap_
				self.fit_flag = True

	def selection_summary(self):
		"""
		A summary for the result of model selection of the sparse regression in Stage 2.

		Returns
		--------

		df: dataframe
			dataframe with columns: "candidate_model", "criteria", and "mse".

		"""
		d = {'model': self.candidate_model_, 'criteria': self.criterion_lst_, 'mse': self.mse_lst_}
		df = pd.DataFrame(data=d)
		# print(df)
		return df
		