from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.linear_model._base import LinearModel
import numpy as np

class WLasso(RegressorMixin, LinearModel):
	"""Linear Model trained with Weighted L1 prior as regularizer (aka the weighted-Lasso)
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * sum_{j=1}^d weight_j * |w_j|
	Technically the Weighted Lasso model is optimizing the same objective function as
	the Lasso with X = X / ada_weight[None,:].
	Parameters
	----------
	alpha : float, default=1.0
		Constant that multiplies the L1 term. Defaults to 1.0.
		``alpha = 0`` is equivalent to an ordinary least square, solved
		by the :class:`LinearRegression` object. For numerical
		reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
		Given this, you should use the :class:`LinearRegression` object.
	ada_weight: ndarray of shape (n_features,)
		Weight that multiplies the L1 term for each coefficient. Defaults to 1.0.
	fit_intercept : bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).
	normalize : bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.
	precompute : 'auto', bool or array-like of shape (n_features, n_features),\
				 default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.
	copy_X : bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.
	max_iter : int, default=1000
		The maximum number of iterations.
	tol : float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.
	warm_start : bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.
	positive : bool, default=False
		When set to ``True``, forces the coefficients to be positive.
	random_state : int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.
	selection : {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.
	Attributes
	----------
	coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).
	dual_gap_ : float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.
	sparse_coef_ : sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.
	intercept_ : float or ndarray of shape (n_targets,)
		Independent term in decision function.
	n_iter_ : int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.
	Examples
	--------
	>>> from sklearn import linear_model
	>>> clf = linear_model.Lasso(alpha=0.1)
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	Lasso(alpha=0.1)
	>>> print(clf.coef_)
	[0.85 0.  ]
	>>> print(clf.intercept_)
	0.15...
	The algorithm used to fit the model is coordinate descent.
	To avoid unnecessary memory duplication the X argument of the fit method
	should be directly passed as a Fortran-contiguous numpy array.
	"""
	def __init__(self, alpha=1.0, *, ada_weight=1.0, fit_intercept=True, normalize=False,
				 precompute=False, copy_X=True, max_iter=1000,
				 tol=1e-4, warm_start=False, positive=False,
				 random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		self.normalize = normalize
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
		fit the linear coeff based on feature and summary statistics.

		Parameters
		----------
		"""
		n_sample, n_feature = X.shape
		lasso_tmp = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept, normalize=self.normalize,
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
	"""Linear Model trained with Weighted SCAD prior as regularizer (aka the weighted-SCAD)
	The optimization objective for Lasso is::
		(1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * sum_{j=1}^d weight_j * SCAD(|w_j|)
	Parameters
	----------
	alpha : float, default=1.0
		Constant that multiplies the SCAD penalty. Defaults to 1.0.
		``alpha = 0`` is equivalent to an ordinary least square, solved
		by the :class:`LinearRegression` object. For numerical
		reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
		Given this, you should use the :class:`LinearRegression` object.
	ada_weight: ndarray of shape (n_features,)
		Weight that multiplies the L1 term for each coefficient. Defaults to 1.0.
	fit_intercept : bool, default=True
		Whether to calculate the intercept for this model. If set
		to False, no intercept will be used in calculations
		(i.e. data is expected to be centered).
	normalize : bool, default=False
		This parameter is ignored when ``fit_intercept`` is set to False.
		If True, the regressors X will be normalized before regression by
		subtracting the mean and dividing by the l2-norm.
		If you wish to standardize, please use
		:class:`~sklearn.preprocessing.StandardScaler` before calling ``fit``
		on an estimator with ``normalize=False``.
	precompute : 'auto', bool or array-like of shape (n_features, n_features),\
				 default=False
		Whether to use a precomputed Gram matrix to speed up
		calculations. If set to ``'auto'`` let us decide. The Gram
		matrix can also be passed as argument. For sparse input
		this option is always ``True`` to preserve sparsity.
	copy_X : bool, default=True
		If ``True``, X will be copied; else, it may be overwritten.
	max_iter : int, default=1000
		The maximum number of iterations.
	tol : float, default=1e-4
		The tolerance for the optimization: if the updates are
		smaller than ``tol``, the optimization code checks the
		dual gap for optimality and continues until it is smaller
		than ``tol``.
	warm_start : bool, default=False
		When set to True, reuse the solution of the previous call to fit as
		initialization, otherwise, just erase the previous solution.
		See :term:`the Glossary <warm_start>`.
	positive : bool, default=False
		When set to ``True``, forces the coefficients to be positive.
	random_state : int, RandomState instance, default=None
		The seed of the pseudo random number generator that selects a random
		feature to update. Used when ``selection`` == 'random'.
		Pass an int for reproducible output across multiple function calls.
		See :term:`Glossary <random_state>`.
	selection : {'cyclic', 'random'}, default='cyclic'
		If set to 'random', a random coefficient is updated every iteration
		rather than looping over features sequentially by default. This
		(setting to 'random') often leads to significantly faster convergence
		especially when tol is higher than 1e-4.
	Attributes
	----------
	coef_ : ndarray of shape (n_features,) or (n_targets, n_features)
		Parameter vector (w in the cost function formula).
	dual_gap_ : float or ndarray of shape (n_targets,)
		Given param alpha, the dual gaps at the end of the optimization,
		same shape as each observation of y.
	sparse_coef_ : sparse matrix of shape (n_features, 1) or \
			(n_targets, n_features)
		Readonly property derived from ``coef_``.
	intercept_ : float or ndarray of shape (n_targets,)
		Independent term in decision function.
	n_iter_ : int or list of int
		Number of iterations run by the coordinate descent solver to reach
		the specified tolerance.
	Examples
	--------
	>>> from sklearn import linear_model
	>>> clf = linear_model.Lasso(alpha=0.1)
	>>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
	Lasso(alpha=0.1)
	>>> print(clf.coef_)
	[0.85 0.  ]
	>>> print(clf.intercept_)
	0.15...
	The algorithm used to fit the model is coordinate descent.
	To avoid unnecessary memory duplication the X argument of the fit method
	should be directly passed as a Fortran-contiguous numpy array.
	"""
	def __init__(self, alpha=1.0, *, ada_weight=1.0, fit_intercept=True, normalize=False,
				 precompute=False, copy_X=True, max_iter=1000,
				 tol=1e-4, warm_start=False, positive=False,
				 random_state=None, selection='cyclic'):
		self.alpha = alpha
		self.ada_weight = ada_weight
		self.fit_intercept = fit_intercept
		self.normalize = normalize
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
		fit the linear coeff based on feature and summary statistics.

		Parameters
		----------
		"""
		n_sample, n_feature = X.shape
		Wlasso_tmp = WLasso(alpha=self.alpha, ada_weight=self.ada_weight, fit_intercept=self.fit_intercept, 
						   normalize=self.normalize, precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
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
			if np.mean((coef_old - Wlasso_tmp.coef_)**2) < self.tol:
				break
			coef_old = Wlasso_tmp.coef_
	
	def grad_SCAD_(self, a=3.7):
		abs_coef = abs(self.coef_)
		return self.alpha*(abs_coef <= self.alpha) + np.maximum(a*self.alpha - abs_coef, 0.) / (a - 1) * (abs_coef > self.alpha)

class L0_IC(RegressorMixin, LinearModel):

	def __init__(self, criterion='bic', Ks=range(1,10), alphas=np.arange(-3,3,.1), mask='full', fit_intercept=True, normalize=False,
				precompute=False, copy_X=True, max_iter=1000,
				tol=1e-4, warm_start=False, positive=False,
				random_state=None, selection='cyclic'):
		self.criterion = criterion
		self.Ks = Ks
		self.alphas = alphas
		self.mask = mask
		self.fit_intercept = fit_intercept
		self.normalize = normalize
		self.precompute = precompute
		self.max_iter = max_iter
		self.copy_X = copy_X
		self.tol = tol
		self.warm_start = warm_start
		self.positive = positive
		self.random_state = random_state
		self.selection = selection

	def fit(self, X, y, sample_weight=None):
		n_sample, n_feature = X.shape
		eps64 = np.finfo('float64').eps
		pre_select = list(np.where(self.mask==False)[0])
		if self.mask == 'full':
			scad_tmp = SCAD(fit_intercept=self.fit_intercept, normalize=self.normalize, precompute=self.precompute, 
							copy_X=self.copy_X, max_iter=self.max_iter,
							tol=self.tol, warm_start=self.warm_start, positive=self.positive,
							random_state=self.random_state, selection=self.selection)
		else:
			scad_tmp = SCAD(ada_weight=1.*self.mask, fit_intercept=self.fit_intercept, 
							normalize=self.normalize, precompute=self.precompute, copy_X=self.copy_X, max_iter=self.max_iter,
							tol=self.tol, warm_start=self.warm_start, positive=self.positive,
							random_state=self.random_state, selection=self.selection)
		candidate_model = []
		for alpha_tmp in self.alphas:
			scad_tmp.alpha = alpha_tmp
			scad_tmp.fit(X, y, sample_weight)
			## we don't select the features with mask = False
			abs_coef = abs(scad_tmp.coef_) * self.mask
			nz_ind = np.argsort(abs_coef)[-sum(abs_coef > eps64):][::-1]
			for K_tmp in self.Ks:
				if K_tmp > len(nz_ind):
					break
				nz_ind_tmp = list(nz_ind[:K_tmp])
				nz_ind_tmp.extend(pre_select)
				candidate_model.append(nz_ind_tmp)
		candidate_model = set(map(tuple, candidate_model))
		candidate_model = list(candidate_model)
		
		## fit largest model to get variance
		clf_full = LinearRegression()
		clf_full.fit(X, y)
		var_res = np.mean(( y - clf_full.predict(X) )**2)
		## find the best model
		criterion_lst = []
		for model_tmp in candidate_model:
			model_tmp = np.array(model_tmp)
			clf_tmp = LinearRegression()
			clf_tmp.fit(X[:,model_tmp], y, sample_weight)
			res = y - clf_tmp.predict(X[:,model_tmp])
			if self.criterion == 'bic':
				criterion_tmp = np.mean(res**2) / (var_res**2 + eps64) + len(model_tmp) * np.log(n_sample) / n_sample
			elif self.criterion == 'aic':
				criterion_tmp = np.mean(res**2) / (var_res**2 + eps64) + len(model_tmp) * 2 / n_sample
			else:
				raise NameError('criteria should be aic or bic')
			criterion_lst.append(criterion_tmp)
		## best model
		best_model = np.array(candidate_model[np.argmin(criterion_lst)])
		clf_best = LinearRegression()
		clf_best.fit(X[:,best_model], y)
		self.coef_ = np.zeros(n_feature)
		self.coef_[best_model] = clf_best.coef_
		self.intercept_ = clf_best.intercept_
		
# ## test
import numpy as np
from sklearn.datasets import make_regression
X, y, true_beta = make_regression(1000, 100, coef=True)
## L0_IC
n, d = X.shape
mask = np.ones(d, dtype=bool)
mask[0] = False
tmp = L0_IC(Ks=range(1,d), mask=mask)
tmp.fit(X, y)


## weighted Lasso
ada_weight = np.ones(100)
ada_weight[:50] = 0.
tmp = SCAD(alpha=.1, ada_weight=ada_weight)
tmp.fit(X, y)
pred_y = tmp.predict(X)
np.mean((y - pred_y)**2)

# ## lasso
# clf = Lasso(alpha=.1)
# clf.fit(X, y)
# pred_y = clf.predict(X)
# np.mean((y - pred_y)**2)
