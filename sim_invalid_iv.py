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

# normal AR(.5)
# ==================================================
# 'linear': beta0: 0.000, n: 10000, p: 50, bad_select: 143
# Rejection: 2sls: 0.050; RT_2sls: 0.051; SIR: 0.051

# ==================================================
# 'log': beta0: 0.000, n: 10000, p: 50, bad_select: 116
# Rejection: 2sls: 0.054; RT_2sls: 0.058; SIR: 0.055

# ==================================================
# 'cube-root': beta0: 0.000, n: 10000, p: 50, bad_select: 138
# Rejection: 2sls: 0.055; RT_2sls: 0.056; SIR: 0.055

# ==================================================
# inverse: beta0: 0.000, n: 10000, p: 50, bad_select: 153
# Rejection: 2sls: 0.056; RT_2sls: 0.068; SIR: 0.057

# ==================================================
# PL: beta0: 0.000, n: 10000, p: 50, bad_select: 164
# Rejection: 2sls: 0.055; RT_2sls: 0.057; SIR: 0.058

n, p = 10000, 50
# theta0 = np.random.randn(p)
for beta0 in [.03, .05, .10]:
	bad_case, bad_select = 0, 0
# for beta0 in [.00]:
	p_value = []
	n_sim = 100
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
		LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
		LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
		# print('True beta: %.3f' %beta0)

		Ks = range(p)
		## solve by 2sls
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3),
						Ks=Ks, max_iter=10000, refit=False, find_best=False)
		LS = _2SCausal._2SLS(sparse_reg=reg_model)
		# LS = _2SCausal._2SLS(sparse_reg = SCAD_IC(fit_intercept=False, max_iter=10000))
		## Stage-1 fit theta
		LS.fit_theta(LD_Z1, cor_ZX1)
		## Stage-2 fit beta
		LS.fit_beta(LD_Z2, cor_ZY2, n2=n2)
		## generate CI for beta
		LS.test_effect(n2, LD_Z2, cor_ZY2)
		# print('alpha: %s' %(LS.alpha*y_scale))
		# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

		RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
		# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3), 
					Ks=Ks, max_iter=10000, refit=False, find_best=False)
		RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
		RT_LS = _2SCausal._2SLS(sparse_reg=reg_model)
		## Stage-1 fit theta
		RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
		## Stage-2 fit beta
		RT_LS.fit_beta(LD_Z2, cor_ZY2, n2=n2)
		## generate CI for beta
		RT_LS.test_effect(n2, LD_Z2, cor_ZY2)
		# print('est beta based on PT_LS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

		## solve by SIR+LS
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-1,3,.3), 
				Ks=Ks, max_iter=10000, refit=False, find_best=False)
		echo = _2SCausal._2SIR(sparse_reg=reg_model)
		## Stage-1 fit theta
		echo.fit_sir(Z1, X1)
		## Stage-2 fit beta
		echo.fit_reg(LD_Z2, cor_ZY2, n2=n2)
		## generate CI for beta
		echo.test_effect(n2, LD_Z2, cor_ZY2)
		# print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))
		if sorted(echo.best_model_) != sorted([0,1,2,3,4,50]):
			bad_select += 1
		p_value.append([LS.p_value, RT_LS.p_value, echo.p_value])
	p_value = np.array(p_value)

	n_sim = n_sim - bad_case
	print('='*50)
	print('beta0: %.3f, n: %d, p: %d, bad_select: %d'%(beta0, n, p, bad_select))
	print('Rejection: 2sls: %.3f; RT_2sls: %.3f; SIR: %.3f'
			%( len(p_value[p_value[:,0]<.05])/n_sim,
			len(p_value[p_value[:,1]<.05])/n_sim,
			len(p_value[p_value[:,2]<.05])/n_sim))