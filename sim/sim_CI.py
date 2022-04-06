## simulate two-stage dataset
# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nl_causal import ts_models
from sklearn.preprocessing import power_transform, quantile_transform
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import KNeighborsRegressor

# simulation for CI
n, p = 5000, 50
for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear', 'quad']:
# for case in ['linear']:
	for beta0 in [.005]:
		beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
		len_LS, len_RT_LS, len_SIR = [], [], []
		cover_LS, cover_RT_LS, cover_SIR = [], [], []
		n_sim = 500
		for i in range(n_sim):
			theta0 = np.random.randn(p)
			theta0[:int(.3*p)] = 0.
			# theta0 = np.ones(p)
			theta0 = theta0 / np.sqrt(np.sum(theta0**2))
			Z, X, y, phi = sim(n, p, theta0, beta0, case=case, feat='normal')
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
			LS = ts_models._2SLS(sparse_reg=None)
			## Stage-1 fit theta 
			LS.fit_theta(LD_Z1, cov_ZX1)
			## Stage-2 fit beta
			LS.fit_beta(LD_Z2, cov_ZY2, n2)
			## generate CI for beta
			LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, level=.95)
			LS.CI[0] = max(LS.CI[0], 0.)
			len_tmp = (LS.CI[1] - LS.CI[0])*y_scale
			if ( (beta0 >= LS.CI[0]*y_scale) and (beta0 <= LS.CI[1]*y_scale) ):
				cover_LS.append(1)
			else:
				cover_LS.append(0)
			len_LS.append(len_tmp)
			# print('est beta based on 2SLS: %.3f; CI: %s; len: %.3f' %(LS.beta*y_scale, LS.CI*y_scale, (LS.CI[1] - LS.CI[0])*y_scale))
			# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

			## solve by RT-2SLS
			RT_X1 = power_transform(X1.reshape(-1,1), method='yeo-johnson').flatten()
			# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
			RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
			RT_LS = ts_models._2SLS(sparse_reg=None)
			## Stage-1 fit theta
			RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
			## Stage-2 fit beta
			RT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
			## generate CI for beta
			RT_LS.CI_beta(n1, n2, Z1, RT_X1, LD_Z2, cov_ZY2, level=.95)
			RT_LS.CI[0] = max(RT_LS.CI[0], 0.)
			len_tmp = (RT_LS.CI[1] - RT_LS.CI[0])*y_scale
			if ( (beta0 >= RT_LS.CI[0]*y_scale) and (beta0 <= RT_LS.CI[1]*y_scale) ):
				cover_RT_LS.append(1)
			else:
				cover_RT_LS.append(0)
			len_RT_LS.append(len_tmp)
			# print('est beta based on RT-2SLS: %.3f; CI: %s; len: %.3f' %(RT_LS.beta*y_scale, RT_LS.CI*y_scale, (RT_LS.CI[1] - RT_LS.CI[0])*y_scale))
			# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

			# solve by SIR+LS
			echo = ts_models._2SIR(sparse_reg=None)
			echo.cond_mean = KNeighborsRegressor(n_neighbors=20)
			## Stage-1 fit theta
			echo.fit_theta(Z1, X1)
			## Stage-2 fit beta
			echo.fit_beta(LD_Z2, cov_ZY2, n2)
			# echo.fit_air(Z1, X1)
			## generate CI for beta
			echo.CI_beta(n1, n2, Z1, X1, LD_Z2, cov_ZY2, B_sample=100, level=.95)
			len_tmp = (echo.CI[1] - echo.CI[0])*y_scale
			# print('est beta based on 2SIR: %.3f; CI: %s; len: %.3f' %(echo.beta*y_scale, echo.CI*y_scale, (echo.CI[1] - echo.CI[0])*y_scale))
			if ( (beta0 >= echo.CI[0]*y_scale) and (beta0 <= echo.CI[1]*y_scale) ):
				cover_SIR.append(1)
			else:
				cover_SIR.append(0)
			len_SIR.append(len_tmp)

			# beta_LS.append(LS.beta*y_scale)
			# beta_RT_LS.append(RT_LS.beta*y_scale)
			# beta_LS_SIR.append(echo.beta*y_scale)

		# d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
		# 	'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}
		# p_value = np.array(p_value)

		print('#'*40)
		print('simulation setting: case: %s n: %d, p: %d, beta0: %.3f' %(case, n,p, beta0))
		# ## estimation acc
		# print('est beta: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
		# 		%( np.mean(beta_LS), np.std(beta_LS), np.mean(beta_RT_LS), np.std(beta_RT_LS), 
		# 		np.mean(beta_LS_SIR), np.std(beta_LS_SIR)))

		print('2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f(%.3f)'
				%(beta0, np.mean(cover_LS), np.mean(len_LS), np.std(len_LS)))
		print('PT-2SLS: beta0: %.3f; CI coverage: %.3f; CI len: %.3f(%.3f)'
				%(beta0, np.mean(cover_RT_LS), np.mean(len_RT_LS), np.std(len_RT_LS)))
		print('2SIR: beta0: %.3f; CI coverage: %.3f; CI len: %.3f(%.3f)'
				%(beta0, np.mean(cover_SIR), np.mean(len_SIR), np.std(len_SIR)))
