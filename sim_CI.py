## simulate two-stage dataset
# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
from sklearn.preprocessing import power_transform, quantile_transform

n, p = 5000, 10
for beta0 in [0.00]:
# for beta0 in [.00, .05, .10, .15]:
	beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
	cover_LS, cover_RT_LS, cover_SIR = 0, 0, 0
	n_sim = 500
	for i in range(n_sim):
		theta0 = np.random.randn(p)
		# theta0 = np.ones(p)
		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
		Z, X, y, phi = sim(n, p, theta0, beta0, case='linear', feat='normal')
		# if abs(X).max() > 1e+8:
		# 	continue
		## normalize Z, X, y
		center = StandardScaler(with_std=False)
		mean_X, mean_y = X.mean(), y.mean()
		Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
		y_scale = y.std()
		y = y / y_scale
		Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
		n1, n2 = len(Z1), len(Z2)
		# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
		LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
		LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
		# np.cov( np.dot(Z, theta0), X )
		# print('True beta: %.3f' %beta0)

		## solve by 2sls
		LS = _2SCausal._2SLS(sparse_reg=None)
		## Stage-1 fit theta 
		LS.fit_theta(LD_Z1, cov_ZX1)
		## Stage-2 fit beta
		LS.fit_beta(LD_Z2, cov_ZY2)
		## generate CI for beta
		LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
		if ( (beta0 >= LS.CI[0]*y_scale) and (beta0 <= LS.CI[1]*y_scale) ):
			cover_LS = cover_LS + 1 / n_sim
	
		# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

		## solve by RT-2SLS
		# RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
		# # RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
		# RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
		# RT_LS = _2SCausal._2SLS(sparse_reg=None)
		# ## Stage-1 fit theta
		# RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
		# ## Stage-2 fit beta
		# RT_LS.fit_beta(LD_Z2, cov_ZY2)
		# ## generate CI for beta
		# RT_LS.test_effect(n2, LD_Z2, cov_ZY2)

		# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

		## solve by SIR+LS
		echo = _2SCausal._2SIR(sparse_reg=None)
		## Stage-1 fit theta
		echo.fit_sir(Z1, X1)
		## Stage-2 fit beta
		echo.fit_reg(LD_Z2, cov_ZY2)
		## generate CI for beta
		echo.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, level=.95)	
		print('est beta based on 2SIR: %.3f; CI: %s' %(echo.beta*y_scale, echo.CI*y_scale))
		if ( (beta0 >= echo.CI[0]*y_scale) and (beta0 <= echo.CI[1]*y_scale) ):
			cover_SIR = cover_SIR + 1 / n_sim

		# beta_LS.append(LS.beta*y_scale)
		# beta_RT_LS.append(RT_LS.beta*y_scale)
		# beta_LS_SIR.append(echo.beta*y_scale)

	# d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
	# 	'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}
	# p_value = np.array(p_value)

	# print('#'*60)
	# print('simulation setting: n: %d, p: %d, beta0: %.3f' %(n,p, beta0))
	# ## estimation acc
	# print('est beta: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
	# 		%( np.mean(beta_LS), np.std(beta_LS), np.mean(beta_RT_LS), np.std(beta_RT_LS), 
	# 		np.mean(beta_LS_SIR), np.std(beta_LS_SIR)))

	print('beta0: %.3f; CI coverage: %.3f'%(beta0, cover_LS))			
	print('beta0: %.3f; CI coverage: %.3f'%(beta0, cover_LS))
