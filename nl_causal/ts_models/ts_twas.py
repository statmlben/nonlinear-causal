"""Two Stage Methods based GWAS dataset"""
# Author: Ben Dai <bendai@cuhk.edu.hk>

import numpy as np
# from sklearn.base import BaseEstimator
import sklearn.preprocessing as pps
from sliced import SlicedInverseRegression
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import norm
from scipy.linalg import sqrtm
import pandas as pd

class _2SLS(object):
    """
    Two-Stage least squares (2SLS) regression is a statistical technique that is used in the analysis of structural equations::
    
        (stage 1) x = z^T theta + omega; 		(Stage 2) y = beta x + z^T alpha + epsilon

    Note that data is expected to be centered, and y is normarlized as sd(y) = 1.

    Parameters
    ----------
    normalize: bool, default=True
        Whether to normalize the resulting `theta` in Stage 1.nl_causal/ts_models/ts_twas.py
    
    fit_flag: bool, default=False
        A flag to indicate if the estimation is done.

    sparse_reg: class, default=None
        A sparse regression used in the Stage 2. If set to None, we will use OLS in for the Stage 2.
    
    Attributes
    ----------
    p_value: float
        P-value for hypothesis testing:: 
            H_0: `beta` = 0;		 H_a: `beta` neq 0.
    
    theta: array of shape (n_features, )
        Estimated linear coefficients for the linear regression in Stage 1.

    beta: float
        The marginal causal effect `beta` in Stage 2.  
    
    alpha: array of shape (n_features, )
        Estimated linear coefficients for Invalid IVs in Stage 2.

    CI: array of shape (2, )
        Estimated confidence interval for marginal causal effect (`beta`).
    
    Examples
    --------
    >>> import numpy as np
    >>> from nl_causal import ts_models
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.model_selection import train_test_split
    >>> n, p = 1000, 50
    >>> np.random.seed(0)
    >>> Z = np.random.randn(n, p)
    >>> U, eps, gamma = np.random.randn(n), np.random.randn(n), np.random.randn(n)
    >>> theta0 = np.random.randn(p)
    >>> theta0 = theta0 / np.sqrt(np.sum(theta0**2))
    >>> beta0 = .5
    >>> X = np.dot(Z, theta0) + U**2 + eps
    >>> y = beta0 * X + U + gamma
    >>> center = StandardScaler(with_std=False)
    >>> mean_X, mean_y = X.mean(), y.mean()
    >>> y_scale = y.std()
    >>> y = y / y_scale
    >>> Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
    >>> Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
    >>> n1, n2 = len(Z1), len(Z2)
    >>> LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
    >>> LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
    >>> LS = ts_models._2SLS(sparse_reg=None)
    >>> LS.fit_theta(LD_Z1, cov_ZX1)
    >>> LS.theta
    array([ 0.07268042, -0.06393631, -0.34863568, -0.04453319, -0.10313442,
        0.09284717,  0.13514061,  0.24597997,  0.08304104,  0.02459302,
        -0.06349473,  0.21502581,  0.21177209, -0.08895919,  0.08082504,
        -0.13174606,  0.04644996,  0.23102817, -0.05635546,  0.08286319,
        0.20736079, -0.07574444, -0.20129764,  0.20311458, -0.15965619,
        -0.02148001,  0.01761156,  0.10617795,  0.02028776, -0.00221961,
        -0.11686226, -0.09116777,  0.08004126,  0.00663467, -0.13549927,
        -0.12674926,  0.09331474, -0.24688913, -0.18701941,  0.02714403,
        0.0854651 ,  0.30291367,  0.08926479,  0.023272  , -0.04798961,
        0.26668035, -0.16051689,  0.01169355, -0.08651508, -0.1342292 ])
    >>> LS.fit_beta(LD_Z2, cov_ZY2, n2=n2)
    >>> LS.beta
    0.47526913304288304
    >>> LS.test_effect(n2, LD_Z2, cov_ZY2)
    >>> LS.p_value
    1.2114293989960872e-59
    >>> LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
    array([0.38869162, 0.56184665])
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
        self.theta_norm = 1.

    def fit_theta(self, LD_Z1, cov_ZX1):
        """
        Fit the linear model in Stage 1.

        Parameters
        ----------
        LD_Z1: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the first sample: ``LD_Z1 = np.dot(Z1.T, Z1)``

        cov_ZX1: {array-like, float} of shape (n_features, )
            Cov(Z1, X1); ``cov_ZX1 = np.dot(Z1.T, X1)``
        
        Returns
        -------
        self: returns an theta of self.
        """

        self.p = len(LD_Z1)
        self.theta = np.dot(np.linalg.inv( LD_Z1 ), cov_ZX1)
        if self.normalize:
            self.theta_norm = np.sqrt(np.sum(self.theta**2))
            self.theta = pps.normalize(self.theta.reshape(1, -1))[0]

    def fit_beta(self, LD_Z2, cov_ZY2, n2, criterion='ebic'):
        """
        Fit the linear model in Stage 2 based on *GWAS* data.

        Parameters
        ----------
        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        n2: int
            The number of sample in the second *GWAS* dataset.

        Returns
        -------
        self: returns `beta` of self.
        """

        eps32 = np.finfo('float32').eps
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
            # LD_Z_aug = LD_Z_aug + np.finfo('float16').eps*np.identity(p+1)
            ## keep the covariance matrix p.d.
            # LD_Z_aug += np.linalg.norm(LD_Z_aug) * np.finfo('float32').eps*np.eye(p+1)
            # eig, eigv = np.linalg.eig(LD_Z_aug)
            pseudo_input = sqrtm(LD_Z_aug).real
            pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
            ada_weight = np.ones(p+1, dtype=bool)
            ada_weight[-1] = False
            
            self.sparse_reg.ada_weight = ada_weight
            self.sparse_reg.fit(pseudo_input, pseudo_output)
            candidate_model_lst, criterion_lst, mse_lst = [], [], []
            # print('var_res: %.3f' %self.var_res)
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
                if mse_tmp < 0:
                    ## when mse is negative, we tend to believe the model is wrong
                    continue
                # print('mse_tmp: %.3f' %mse_tmp)
                # if criterion == 'bic':
                #     var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
                #     criterion_tmp = mse_tmp / (var_res + eps64) + len(model_tmp) * np.log(n2) / n2
                # elif criterion == 'aic':
                #     var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
                #     criterion_tmp = mse_tmp / (var_res + eps64) + len(model_tmp) * 2 / n2
                if criterion == 'ebic':
                    criterion_tmp = np.log(mse_tmp) + len(model_tmp) * np.log(n2) / n2
                else:
                    raise NameError('criteria should be ebic')
                candidate_model_lst.append(model_tmp)
                criterion_lst.append(criterion_tmp)
                mse_lst.append(mse_tmp)
            self.candidate_model_ = candidate_model_lst
            self.criterion_lst_ = criterion_lst
            self.mse_lst_ = mse_lst
            ## fit the best model
            best_model = np.array(self.candidate_model_[np.argmin(criterion_lst)])
            self.best_model_ = best_model
            self.best_mse_ = mse_lst[np.argmin(criterion_lst)]
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
            dataframe with columns: ``candidate_model``, ``criteria``, and ``mse``.

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
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        Returns
        -------
        The estimated variance y regress on Z.
        """
        if self.sparse_reg == None:
            alpha = np.linalg.inv(LD_Z2).dot(cov_ZY2)
            sigma_res_y = 1. - 2 * np.dot(alpha, cov_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
        else:
            sigma_res_y = self.best_mse_
        if sigma_res_y < 0:
            raise Exception("Sorry, we get a negative est variance, the inference is suspended.") 
        return sigma_res_y + np.finfo('float64').eps

    def fit(self, LD_Z1, cov_ZX1, LD_Z2, cov_ZY2, n2):
        """
        Fit the linear model in Stage 2 based on **GWAS** data.

        Parameters
        ----------
        LD_Z1: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the first sample: ``LD_Z1 = np.dot(Z1.T, Z1)``

        cov_ZX1: {array-like, float} of shape (n_features, )
            Cov(Z1, X1); ``cov_ZX1 = np.dot(Z1.T, X1)``

        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        n2: int
            The number of sample in the second *GWAS* dataset.

        Returns
        -------
        self: return `theta` of self.

        self: returns `beta` of self.
        """
        self.fit_theta(LD_Z1, cov_ZX1)
        self.fit_beta(LD_Z2, cov_ZY2, n2)
        self.fit_flag = True
    
    def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=0.95):
        """
        Estimated confidence interval (CI) for the causal effect `beta`.

        Parameters
        ----------
        n1: int
            The number of sample on the first dataset.

        n2: int
            The number of sample on the second dataset.

        Z1: {array-like, float} of shape (n_sample, n_features)
            Samples of Z in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset, and n_feature is the number of features.

        X1: {array-like, float} of shape (n_sample)
            Samples of X in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset.

        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

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
            var_x = np.mean( (X1 - self.theta_norm*np.dot(Z1, self.theta))**2 )
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
            var_beta = max(np.finfo('float32').eps, var_beta)
            delta_tmp = abs(norm.ppf((1. - level)/2)) * np.sqrt(var_beta) / np.sqrt(n2)
            beta_low = self.beta - delta_tmp
            beta_high = self.beta + delta_tmp
            self.CI = np.array([beta_low, beta_high])
        else:
            raise NameError('CI can only be generated after fit!')
    
    def test_effect(self, n2, LD_Z2, cov_ZY2):
        """
        Causal inference for the marginal causal effect `beta`.
        P-value for hypothesis testing:: 
            H0: `beta` = 0; 		Ha: `beta` neq 0.

        Parameters
        ----------
        n2: int
            The number of samples in the second dataset.

        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

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

    Two-stage instrumental regression (2SIR) is a statistical technique used in the analysis of structural equations::
    
    (stage 1) phi(x) = z^T theta + omega; 		(Stage 2) y = beta phi(x) + z^T alpha + epsilon

    Note that data is expected to be centered, and y is normarlized as sd(y) = 1.

    Parameters
    ----------
    n_directions: int, default=False
        A number of directions for sliced inverse regression (SIR). Currently, we only focus on 1-dimensional case.

    n_slices: int, default='auto'
        A number of slices for SIR, if `n_slices='auto'`, then it is determined by ``int(n_sample/data_in_slice)``.
    
    data_in_slice: int, default=100
        A number of samples in a slice for SIR. If data_in_slice is not None, then ``n_slices=int(n_sample/data_in_slice)``.

    cond_mean: class, default=KNeighborsRegressor(n_neighbors=10)
        A nonparameteric regression model for estimate link function.

    if_fit_link: bool, default=True
        Whether to calculate the link function `phi` for this model.

    fit_flag: bool, default=False
        A flag to indicate if the estimation is done.

    sparse_reg: class, default=None
        A sparse regression used in the Stage 2. If set to None, we will use OLS in for the Stage 2.
    
    Attributes
    ----------
    p_value: float
        P-value for hypothesis testing:: 
            H0: `beta` = 0; 		Ha: `beta` neq 0.
    
    theta: array of shape (n_features, )
        Estimated linear coefficients for the linear regression in Stage 1.

    beta: float
        The marginal causal effect `beta` in Stage 2.  
    
    alpha: array of shape (n_features, )
        Estimated linear coefficients for Invalid IVs in Stage 2.
    
    rho: float
        A correction ratio for the link estimation.

    CI: array of shape (2, )
        Estimated confidence interval for marginal causal effect (`beta`).

    Examples
    --------
    >>> import numpy as np
    >>> from nl_causal import ts_models
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.model_selection import train_test_split
    >>> n, p = 2000, 50
    >>> np.random.seed(0)
    >>> Z = np.random.randn(n, p)
    >>> U, eps, gamma = np.random.randn(n), np.random.randn(n), np.random.randn(n)
    >>> theta0 = np.random.randn(p)
    >>> theta0 = theta0 / np.sqrt(np.sum(theta0**2))
    >>> beta0 = 1.
    >>>	X = 1. / (np.dot(Z, theta0) + U**2 + eps)
    >>> phi = 1. / X
    >>> y = beta0 * phi + U + gamma
    >>> ## normalize Z, X, y
    >>> center = StandardScaler(with_std=False)
    >>> mean_X, mean_y = X.mean(), y.mean()
    >>> Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
    >>> y_scale = y.std()
    >>> y = y / y_scale
    >>> Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
    >>> n1, n2 = len(Z1), len(Z2)
    >>> LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
    >>> LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
    >>> ## Define 2SLS
    >>> nl_twas = ts_models._2SIR(sparse_reg=None)
    >>> ## Estimate theta in Stage 1
    >>> nl_twas.fit_theta(Z1, X1)
    >>> nl_twas.theta
    >>> array([-0.31331769, -0.10041349, -0.0068858 ,  0.01292981, -0.06329046,
        -0.16037138, -0.16804225,  0.21417387,  0.03366815,  0.06368144,
        -0.24217008,  0.05969993,  0.10396408,  0.1464862 , -0.08197703,
        0.06766957,  0.0663083 , -0.11894489,  0.01675101,  0.29531242,
        0.1923396 , -0.02299256,  0.14921243, -0.01075526, -0.04526044,
        -0.03111288,  0.05537396, -0.02358006,  0.12615653,  0.03938541,
        -0.09100911,  0.12907855,  0.19518874, -0.23574715, -0.12036349,
        -0.04914323, -0.03463147,  0.01019404,  0.15832153, -0.0180269 ,
        0.05225932,  0.33307795,  0.1104155 , -0.21012056, -0.16505056,
        0.16029017,  0.04530822,  0.24969932,  0.13906269,  0.13336765])
    >>> ## Estimate beta in Stage 2
    >>> nl_twas.fit_beta(LD_Z2, cov_ZY2, n2)
    >>> nl_twas.beta*y_scale
    >>> 1.0294791446256897
    >>> ## p-value for infer if causal effect is zero
    >>> nl_twas.test_effect(n2, LD_Z2, cov_ZY2)
    >>> nl_twas.p_value
    >>> 5.582982509645985e-48
    >>> nl_twas.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, level=.95)
    >>> nl_twas.CI*y_scale
    >>> array([0.82237254, 1.23658575])

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
        Estimate `theta` in Stage 1 by using sliced inverse regression (SIR). 

        Parameters
        ----------
        Z1: {array-like, float} of shape (n_sample, n_features)
            Samples of Z in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset, and n_feature is the number of features.

        X1: {array-like, float} of shape (n_sample)
            Samples of X in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset.
        
        Returns
        -------
        self: returns `theta` of self.
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

    def fit_beta(self, LD_Z2, cov_ZY2, n2, criterion='ebic'):
        """
        Fit the linear model in Stage 2 based on *GWAS* data.

        Parameters
        ----------
        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        n2: int
            The number of samples in the second *GWAS* dataset, which will be used for model selection in sparse regerssion.

        Returns
        -------
        self: returns `beta` of self.
        """

        eps32 = np.finfo('float32').eps
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
            pseudo_input = sqrtm(LD_Z_aug).real
            pseudo_output = np.linalg.inv(pseudo_input).dot(cov_aug)
            ada_weight = np.ones(p+1, dtype=bool)
            ada_weight[-1] = False
            
            # est residual variance
            self.sparse_reg.ada_weight = ada_weight
            # fit model and find the candidate models
            self.sparse_reg.fit(pseudo_input, pseudo_output)
            # bic to select best model
            candidate_model_lst, criterion_lst, mse_lst = [], [], []
            for model_tmp in self.sparse_reg.candidate_model_:
                model_tmp = np.array(model_tmp)
                LD_Z_aug_tmp = LD_Z_aug[model_tmp[:,None], model_tmp]
                cov_aug_tmp = cov_aug[model_tmp]
                coef_aug_tmp = np.linalg.inv(LD_Z_aug_tmp).dot(cov_aug_tmp)
                mse_tmp = 1. - 2 * np.dot(coef_aug_tmp, cov_aug_tmp) / n2 + coef_aug_tmp.T.dot(LD_Z_aug_tmp).dot(coef_aug_tmp) / n2
                if mse_tmp < 0:
                    continue
                # if criterion == 'bic':
                #     var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
                #     criterion_tmp = mse_tmp / (var_res + eps64) + len(model_tmp) * np.log(n2) / n2
                # elif criterion == 'aic':
                #     var_res = self.est_var_res(n2, LD_Z2, cov_ZY2)
                #     criterion_tmp = mse_tmp / (var_res + eps64) + len(model_tmp) * 2 / n2
                if criterion == 'ebic':
                    criterion_tmp = np.log(mse_tmp) + len(model_tmp) * np.log(n2) / n2
                else:
                    raise NameError('criteria should be ebic')
                
                candidate_model_lst.append(model_tmp)
                criterion_lst.append(criterion_tmp)
                mse_lst.append(mse_tmp)
            self.candidate_model_ = candidate_model_lst
            self.criterion_lst_ = criterion_lst
            self.mse_lst_ = mse_lst
            ## fit the best model
            best_model = np.array(self.candidate_model_[np.argmin(criterion_lst)])
            self.best_model_ = best_model
            self.best_mse_ = mse_lst[np.argmin(criterion_lst)]
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
            dataframe with columns: ``candidate_model``, ``criteria``, and ``mse``.

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
        Estimate nonlinear link (`phi`) in Stage 1.

        Parameters
        ----------
        Z1: {array-like, float} of shape (n_sample, n_features)
            Samples of Z in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset, and ``n_feature`` is the number of features.

        X1: {array-like, float} of shape (n_sample)
            Samples of X in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset.
        
        Returns
        -------
        self: returns cond_mean of self.
            Returns estimated conditional mean of z^T `theta` conditional on X1

        self: returns rho of self.
            Return estimated correction ratio of conditional mean. 

        """
        X_sir = self.sir.transform(Z1).flatten()
        self.cond_mean.fit(X=X1[:,None], y=X_sir)
        pred_mean = self.cond_mean.predict(X1[:,None])
        LD_Z_sum = np.sum(Z1[:, :, np.newaxis] * Z1[:, np.newaxis, :], axis=0)
        cross_mean_Z = np.sum(Z1 * pred_mean[:,None], axis=0)
        self.rho = (self.theta.dot(LD_Z_sum).dot(self.theta)) / np.dot( self.theta, cross_mean_Z)
        self.if_fit_link = True

    def fit(self, Z1, X1, LD_Z2, cov_ZY2, n2):
        """
        Fit `theta`, `beta`, (and `phi`) in the causal model.

        Parameters
        ----------
        Z1: {array-like, float} of shape (n_sample, n_features)
            Samples of Z in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset, and n_feature is the number of features.

        X1: {array-like, float} of shape (n_sample)
            Samples of X in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset.
        
        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        n2: int
            The number of samples in the second *GWAS* dataset, which will be used for model selection in sparse regerssion.

        Returns
        -------
        self: returns `theta` of self.

        self: returns `beta` of self.

        self: returns cond_mean of self.
            Returns estimated conditional mean of z^T `theta` conditional on X1

        self: returns `rho` of self.
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
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        Returns
        -------
        The estimated variance y regress on Z.
        """
        if self.sparse_reg == None:
            alpha = np.dot(np.linalg.inv(LD_Z2), cov_ZY2)
            sigma_res_y = 1 - 2 * np.dot(alpha, cov_ZY2) / n2 + alpha.T.dot(LD_Z2).dot(alpha) / n2
        else:
            sigma_res_y = self.best_mse_
        
        if sigma_res_y < 0:
                raise Exception("Sorry, we get a negative est variance, the inference is suspended.") 
                # print('We get negative variance for eps: %.3f, and we correct it as eps.' %sigma_res_y)
        return max(sigma_res_y, 0.) + np.finfo('float64').eps
        

    def CI_beta(self, n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, boot_over='theta', level=.95):
        """
        Estimated confidence interval (CI) for the causal effect `beta`

        Parameters
        ----------
        n1: int
            The number of sample on the first dataset.

        n2: int
            The number of sample on the second dataset.

        Z1: {array-like, float} of shape (n_sample, n_features)
            Samples of Z in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset, and n_feature is the number of features.

        X1: {array-like, float} of shape (n_sample)
            Samples of X in the first dataset, where ``n_sample = n1`` is the number of samples in the first dataset.

        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

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
        if var_eps <= 0:
            print('We get negative variance for eps: %.3f, and we correct it as eps.' %var_eps)
            var_eps = np.finfo('float32').eps
        ## compute the variance of beta
        invalid_iv = np.where(abs(self.alpha) > np.finfo('float32').eps)[0]
        select_mat_inv = np.linalg.inv(LD_Z2[invalid_iv[:,None], invalid_iv])
        select_cov = LD_Z2[:,invalid_iv].dot(select_mat_inv).dot(LD_Z2[invalid_iv,:])
        mid_cov = (LD_Z2 - select_cov) / n2
        omega_x = 1. / ((self.theta.dot(mid_cov).dot(self.theta.T)) + np.finfo('float32').eps)
        var_beta = var_eps * omega_x  + np.finfo('float64').eps
        ## resampling
        if var_beta <= 0:
            print('We get negative variance for beta: %.3f, and we correct it as eps.' %var_beta)
            var_beta = np.finfo('float32').eps
        ## bootstrap over theta
        if boot_over == 'theta':
            zeta = np.sqrt(var_beta)*np.random.randn(B_sample)
            eta, beta_B = [], []
            left_full = np.sqrt(n2/n1)*self.beta*omega_x*self.theta.dot(mid_cov)
            score_full = np.sqrt(n1)*left_full.dot(self.theta)
            for i in range(B_sample):
                B_ind = np.random.choice(n1, n1)
                Z1_B, X1_B = Z1[B_ind], X1[B_ind]
                _2SIR_tmp = _2SIR(sparse_reg=self.sparse_reg, data_in_slice=self.data_in_slice)
                _2SIR_tmp.fit_theta(Z1_B, X1_B)
                _2SIR_tmp.theta = np.sign(np.dot(self.theta, _2SIR_tmp.theta)) * _2SIR_tmp.theta
                # _2SIR_tmp.fit_beta(LD_Z2, cov_ZY2, n2)
                # xi_tmp = np.sqrt(n1)*(_2SIR_tmp.theta - self.theta)
                left_tmp = np.sqrt(n2/n1)*self.beta*omega_x*_2SIR_tmp.theta.dot(mid_cov)
                score_tmp = np.sqrt(n1)*left_tmp.dot(_2SIR_tmp.theta)
                eta_tmp = score_tmp - score_full
                # eta_tmp = left_tmp.dot(xi_tmp)
                eta.append(eta_tmp / 2)
            eta = np.array(eta)
            err = np.abs(zeta - eta)
            delta = np.quantile(err, level) / np.sqrt(n2)
            beta_low = self.beta - delta
            beta_low = max(0., beta_low)
            beta_up = self.beta + delta
        ## bootstrap over beta
        elif boot_over == 'beta':
            zeta = np.sqrt(var_beta)*np.random.randn(B_sample)
            eta, beta_B = [], []
            left_full = np.sqrt(n2/n1)*self.beta*omega_x*self.theta.dot(mid_cov)
            score_full = np.sqrt(n1)*left_full.dot(self.theta)
            for i in range(B_sample):
                B_ind = np.random.choice(n1, n1)
                Z1_B, X1_B = Z1[B_ind], X1[B_ind]
                _2SIR_tmp = _2SIR(sparse_reg=self.sparse_reg, data_in_slice=self.data_in_slice)
                _2SIR_tmp.fit_theta(Z1_B, X1_B)
                _2SIR_tmp.theta = np.sign(np.dot(self.theta, _2SIR_tmp.theta)) * _2SIR_tmp.theta
                _2SIR_tmp.fit_beta(LD_Z2, cov_ZY2, n2)
                # xi_tmp = np.sqrt(n1)*(_2SIR_tmp.theta - self.theta)
                left_tmp = np.sqrt(n2/n1)*_2SIR_tmp.beta*omega_x*_2SIR_tmp.theta.dot(mid_cov)
                score_tmp = np.sqrt(n1)*left_tmp.dot(_2SIR_tmp.theta)
                eta_tmp = score_tmp - score_full
                # eta_tmp = left_tmp.dot(xi_tmp)
                eta.append(eta_tmp)
            eta = np.array(eta)
            err = np.abs(zeta - eta)
            delta = np.quantile(err, level) / np.sqrt(n2)
            beta_low = self.beta - delta
            beta_low = max(0., beta_low)
            beta_up = self.beta + delta
        else:
            print('boot_over must be beta or theta!')
        self.CI = np.array([beta_low, beta_up])

    def test_effect(self, n2, LD_Z2, cov_ZY2):
        """
        Causal inference for the marginal causal effect.

        Parameters
        ----------
        n2: int
            The number of samples in the second dataset.

        LD_Z2: {array-like, float} of shape (n_features, n_features)
            LD matrix of Z based on the second sample: ``LD_Z2 = np.dot(Z2.T, Z2)``

        cov_ZY2: {array-like, float} of shape (n_features, )
            Matrix product of Z2 and Y2; ``cov_ZX2 = np.dot(Z2.T, Y2)``

        Returns
        -------
        self: returns p-value of self.
        """
        
        if self.fit_flag:
            var_eps = self.est_var_res(n2, LD_Z2, cov_ZY2)
            if var_eps < 0:
                print('variance of eps is negative: %.3f, correct it as np.eps.' %var_eps)
                var_eps = np.finfo('float32').eps
            if np.max(abs(self.alpha)) < np.finfo('float32').eps:
                var_beta = var_eps / self.theta.dot(LD_Z2).dot(self.theta.T)
                ## we only test for absolute value
                if var_beta < 0:
                    print('variance of beta is negative: %.3f, correct it as np.eps.' %var_beta)
                    var_beta = np.finfo('float32').eps
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

