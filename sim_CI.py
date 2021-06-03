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
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor


n, p = 2000, 50
for beta0 in [.05]:
# for beta0 in [.00, 1/np.sqrt(n), .05, .10, .15]:
	beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
	len_LS, len_RT_LS, len_SIR = [], [], []
	cover_LS, cover_RT_LS, cover_SIR = 0, 0, 0
	n_sim = 1000
	for i in range(n_sim):
		theta0 = np.random.randn(p)
		# theta0 = np.ones(p)
		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
		Z, X, y, phi = sim(n, p, theta0, beta0, case='cube-root', feat='normal')
		if abs(X).max() > 1e+8:
			i = i - 1
			continue
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
		LS.CI[0] = max(LS.CI[0], 0.)
		len_tmp = (LS.CI[1] - LS.CI[0])*y_scale
		if ( (beta0 >= LS.CI[0]*y_scale) and (beta0 <= LS.CI[1]*y_scale) ):
			cover_LS = cover_LS + 1 / n_sim
		len_LS.append(len_tmp)
		print('est beta based on 2SLS: %.3f; CI: %s; len: %.3f' %(LS.beta*y_scale, LS.CI*y_scale, (LS.CI[1] - LS.CI[0])*y_scale))


		# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

		## solve by RT-2SLS
		RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
		# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
		RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
		RT_LS = _2SCausal._2SLS(sparse_reg=None)
		## Stage-1 fit theta
		RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
		## Stage-2 fit beta
		RT_LS.fit_beta(LD_Z2, cov_ZY2)
		## generate CI for beta
		RT_LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
		RT_LS.CI[0] = max(RT_LS.CI[0], 0.)
		len_tmp = (RT_LS.CI[1] - RT_LS.CI[0])*y_scale
		if ( (beta0 >= RT_LS.CI[0]*y_scale) and (beta0 <= RT_LS.CI[1]*y_scale) ):
			cover_RT_LS = cover_RT_LS + 1 / n_sim
		len_RT_LS.append(len_tmp)
		print('est beta based on RT-2SLS: %.3f; CI: %s; len: %.3f' %(RT_LS.beta*y_scale, RT_LS.CI*y_scale, (RT_LS.CI[1] - RT_LS.CI[0])*y_scale))


		# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

		## solve by SIR+LS
		echo = _2SCausal._2SIR(sparse_reg=None)
		# echo.cond_mean = KNeighborsRegressor(n_neighbors=20)
		echo.cond_mean = IsotonicRegression(increasing='auto',out_of_bounds='clip')
		## Stage-1 fit theta
		echo.fit_sir(Z1, X1)
		## Stage-2 fit beta
		echo.fit_reg(LD_Z2, cov_ZY2)
		echo.fit_air(Z1, X1)
		## generate CI for beta
		echo.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=1000, level=.95)
		len_tmp = (echo.CI[1] - echo.CI[0])*y_scale
		print('est beta based on 2SIR: %.3f; CI: %s; len: %.3f' %(echo.beta*y_scale, echo.CI*y_scale, (echo.CI[1] - echo.CI[0])*y_scale))
		if ( (beta0 >= echo.CI[0]*y_scale) and (beta0 <= echo.CI[1]*y_scale) ):
			cover_SIR = cover_SIR + 1 / n_sim
		len_SIR.append(len_tmp)
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

	print('2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_LS, np.mean(len_LS)))
	print('PT-2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_RT_LS, np.mean(len_RT_LS)))
	print('2SIR: beta0: %.3f; CI coverage: %.3f; CI len: %.3f'%(beta0, cover_SIR, np.mean(len_SIR)))

## n = 5000, p = 10; linear
# 2SLS: beta0: 1.000; CI coverage: 0.984; CI len: 0.221
# 2SIR: beta0: 1.000; CI coverage: 0.944; CI len: 0.175
# 2SIR_asym: beta0: 1.000; CI coverage: 0.990; CI len: 0.278

## n = 10000, p = 10; linear
# 2SLS: beta0: 1.000; CI coverage: 0.994; CI len: 0.156
# 2SIR: beta0: 1.000; CI coverage: 0.956; CI len: 0.123
# 2SIR_asym: beta0: 1.000; CI coverage: 0.988; CI len: 0.194
