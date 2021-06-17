## Test invalid IVs
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
from sklearn.preprocessing import power_transform, quantile_transform
from scipy.linalg import sqrtm
from nonlinear_causal.variable_select import WLasso, SCAD, L0_IC, SCAD_IC
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression, LassoLarsIC, LassoCV


n, p = 10000, 50
# theta0 = np.random.randn(p)
# for beta0 in [.03, .05, .10]:
for beta0 in [.05]:
	bad_case, bad_select = 0, 0
	len_LS, len_RT_LS, len_SIR = [], [], []
	cover_LS, cover_RT_LS, cover_SIR = 0, 0, 0
	n_sim = 1000
	for i in range(n_sim):
		theta0 = np.ones(p)
		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
		alpha0 = np.zeros(p)
		alpha0[:5] = 1.
		# alpha0 = alpha0 / np.sqrt(np.sum(alpha0**2))
		Z, X, y, phi = sim(n, p, theta0, beta0, alpha0=alpha0, case='piecewise_linear', feat='AP-normal')
		if abs(X).max() > 1e+8:
			bad_case = bad_case + 1
			continue
		## normalize Z, X, y
		center = StandardScaler(with_std=False)
		mean_X, mean_y = X.mean(), y.mean()
		Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
		Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=.5, random_state=42)
		## scale y
		y_scale = y2.std()
		y1 = y1 / y_scale
		y2 = y2 / y_scale
		y1 = y1 - y2.mean()
		y2 = y2 - y2.mean()
		# summary data
		n1, n2 = len(Z1), len(Z2)
		LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
		LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
		# print('True beta: %.3f' %beta0)

		Ks = range(p)
		## solve by 2sls
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3),
						Ks=Ks, max_iter=10000, refit=False, find_best=False)
		LS = _2SCausal._2SLS(sparse_reg=reg_model)
		# LS = _2SCausal._2SLS(sparse_reg = SCAD_IC(fit_intercept=False, max_iter=10000))
		## Stage-1 fit theta
		LS.fit_theta(LD_Z1, cov_ZX1)
		## Stage-2 fit beta
		LS.fit_beta(LD_Z2, cov_ZY2, n2=n2)
		## generate CI for beta
		LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
		LS.CI[0] = max(LS.CI[0], 0.)
		len_tmp = (LS.CI[1] - LS.CI[0])*y_scale
		if ( (beta0 >= LS.CI[0]*y_scale) and (beta0 <= LS.CI[1]*y_scale) ):
			cover_LS = cover_LS + 1 / n_sim
		len_LS.append(len_tmp)
		print('est beta based on 2SLS: %.3f; CI: %s; len: %.3f' %(LS.beta*y_scale, LS.CI*y_scale, (LS.CI[1] - LS.CI[0])*y_scale))

		RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
		# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3), 
					Ks=Ks, max_iter=10000, refit=False, find_best=False)
		RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
		RT_LS = _2SCausal._2SLS(sparse_reg=reg_model)
		## Stage-1 fit theta
		RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
		## Stage-2 fit beta
		RT_LS.fit_beta(LD_Z2, cov_ZY2, n2=n2)
		## generate CI for beta
		RT_LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
		RT_LS.CI[0] = max(RT_LS.CI[0], 0.)
		len_tmp = (RT_LS.CI[1] - RT_LS.CI[0])*y_scale
		if ( (beta0 >= RT_LS.CI[0]*y_scale) and (beta0 <= RT_LS.CI[1]*y_scale) ):
			cover_RT_LS = cover_RT_LS + 1 / n_sim
		len_RT_LS.append(len_tmp)
		print('est beta based on RT-2SLS: %.3f; CI: %s; len: %.3f' %(RT_LS.beta*y_scale, RT_LS.CI*y_scale, (RT_LS.CI[1] - RT_LS.CI[0])*y_scale))

		## solve by SIR+LS
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3), 
				Ks=Ks, max_iter=10000, refit=False, find_best=False)
		echo = _2SCausal._2SIR(sparse_reg=reg_model)
		## Stage-1 fit theta
		echo.fit_sir(Z1, X1)
		## Stage-2 fit beta
		echo.fit_reg(LD_Z2, cov_ZY2, n2=n2)
		## generate CI for beta
		echo.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, level=.95)
		len_tmp = (echo.CI[1] - echo.CI[0])*y_scale		
		if ( (beta0 >= echo.CI[0]*y_scale) and (beta0 <= echo.CI[1]*y_scale) ):
			cover_SIR = cover_SIR + 1 / n_sim
		len_SIR.append(len_tmp)
		print('est beta based on 2SIR: %.3f; CI: %s; len: %.3f' %(echo.beta*y_scale, echo.CI*y_scale, (echo.CI[1] - echo.CI[0])*y_scale))

		if sorted(echo.best_model_) != sorted([0,1,2,3,4,50]):
			bad_select += 1
	
	print('2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_LS, np.mean(len_LS)))
	print('PT-2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_RT_LS, np.mean(len_RT_LS)))
	print('2SIR: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_SIR, np.mean(len_SIR)))