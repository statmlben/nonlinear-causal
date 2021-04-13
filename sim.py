## simulate two-stage dataset
## With correction
############################################################
# simulation setting: n: 1000, p: 10, beta0: 10.000
# est beta: 2sls: 9.703(0.611); RT_2sls: 9.700(0.611); SIR: 9.646(0.620)
# coverage: 2sls: 0.990; RT_2sls: 0.990; SIR: 0.986
# CI length: 2sls: 3.673(0.185); RT_2sls: 3.673(0.185); SIR: 3.666(0.187)

############################################################
# simulation setting: n: 1000, p: 10, beta0: 100.000
# est beta: 2sls: 97.064(5.893); RT_2sls: 97.042(5.902); SIR: 96.557(5.981)
# coverage: 2sls: 0.994; RT_2sls: 0.994; SIR: 0.992
# CI length: 2sls: 37.032(1.886); RT_2sls: 37.028(1.886); SIR: 36.975(1.879)

# without correction
############################################################
# simulation setting: n: 1000, p: 10, beta0: 10.000
# est beta: 2sls: 9.731(0.608); RT_2sls: 9.729(0.607); SIR: 9.674(0.618)
# coverage: 2sls: 0.910; RT_2sls: 0.904; SIR: 0.886
# CI length: 2sls: 2.165(0.091); RT_2sls: 2.165(0.091); SIR: 2.165(0.091)

############################################################
# simulation setting: n: 1000, p: 10, beta0: 100.000
# est beta: 2sls: 97.653(5.843); RT_2sls: 97.640(5.858); SIR: 97.134(5.965)
# coverage: 2sls: 0.854; RT_2sls: 0.850; SIR: 0.814
# CI length: 2sls: 20.326(0.869); RT_2sls: 20.326(0.869); SIR: 20.333(0.872)


# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
n, p = 1000, 10

beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
# MSE_LS, MSE_RT_LS, MSE_LS_SIR = [], [], []
CI_LS, CI_RT_LS, CI_LS_SIR = [], [], []

n_sim = 500
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	Z, X, y = sim(n, p, theta0, beta0, case='linear', feat='normal', range=.01)
	## normalize Z, X, y
	center = StandardScaler(with_std=False)
	Z, X, y = center.fit_transform(Z), X - X.mean(), y - y.mean()
	y_scale = y.std()
	y = y / y_scale
	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.7, random_state=42)
	n1, n2 = len(Z1), len(Z2)
	# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
	LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
	LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
	# np.cov( np.dot(Z, theta0), X )
	print('True beta: %.3f' %beta0)

	## solve by 2sls
	LS = _2SCausal._2SLS(sparse_reg=None)
	## Stage-1 fit theta 
	LS.fit_theta(LD_Z1, cor_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cor_ZY2)
	## generate CI for beta
	LS_CI_tmp = LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cor_ZY2, level=.95)

	# One sample method
	# LS.fit(LD_Z, cor_ZX, cor_ZY)
	
	print('est beta based on OLS: %.3f' %LS.beta)

	## solve by RT-2SLS
	from sklearn.preprocessing import power_transform, quantile_transform
	RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
	# RT_X = quantile_transform(X.reshape(-1,1), n_quantiles=n/10, output_distribution='normal')
	RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
	RT_LS = _2SCausal._2SLS(sparse_reg=None)
	## Stage-1 fit theta
	RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
	## Stage-2 fit beta
	RT_LS.fit_beta(LD_Z2, cor_ZY2)
	## generate CI for beta
	RT_LS_CI_tmp = RT_LS.CI_beta(n1, n2, Z1, X1, LD_Z2, cor_ZY2, level=.95)

	## One sample method
	# RT_X = power_transform(X.reshape(-1,1)).flatten()
	# # RT_X = quantile_transform(X.reshape(-1,1), n_quantiles=n/10, output_distribution='normal')
	# RT_cor_ZX = np.dot(Z.T, RT_X)
	# RT_LS = _2SMethod._2SLS()
	# RT_LS.fit(LD_Z, RT_cor_ZX, cor_ZY)

	print('est beta based on RT-OLS: %.3f' %RT_LS.beta)

	## solve by SIR+LS
	echo = _2SCausal._2SIR(sparse_reg=None)
	## Stage-1 fit theta
	echo.fit_sir(Z1, X1)
	## Stage-2 fit beta
	echo.fit_reg(Z2, cor_ZY2)
	## generate CI for beta
	SIR_CI_tmp = echo.CI_beta(n1, n2, Z1, X1, LD_Z2, cor_ZY2, level=.95)
	## One sample method
	# echo.fit(Z, X, cor_ZY)

	print('est beta based on 2SIR: %.3f' %echo.beta)

	beta_LS.append(LS.beta*y_scale)
	beta_RT_LS.append(RT_LS.beta*y_scale)
	beta_LS_SIR.append(echo.beta*y_scale)

	CI_LS.append(LS_CI_tmp)
	CI_RT_LS.append(RT_LS_CI_tmp)
	CI_LS_SIR.append(SIR_CI_tmp)

d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
	 'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="method", y="abs_beta", data=d)
# ax = sns.swarmplot(x="method", y="abs_beta", data=d, color=".3", size=3.)
plt.show()

print('#'*60)
print('simulation setting: n: %d, p: %d, beta0: %.3f' %(n,p, beta0))
## estimation acc
print('est beta: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
		%( np.mean(beta_LS), np.std(beta_LS), np.mean(beta_RT_LS), np.std(beta_RT_LS), 
		np.mean(beta_LS_SIR), np.std(beta_LS_SIR)))
## confidence Interval
CI_LS, CI_RT_LS, CI_LS_SIR = np.array(CI_LS)*y_scale, np.array(CI_RT_LS)*y_scale, np.array(CI_LS_SIR)*y_scale
cover_LS = np.sum([(tmp[0] <= beta0) and (tmp[1] >= beta0) for tmp in CI_LS]) / n_sim
cover_RT_LS = np.sum([(tmp[0] <= beta0) and (tmp[1] >= beta0) for tmp in CI_RT_LS]) / n_sim
cover_air = np.sum([(tmp[0] <= beta0) and (tmp[1] >= beta0) for tmp in CI_LS_SIR]) / n_sim
print('coverage: 2sls: %.3f; RT_2sls: %.3f; SIR: %.3f' %(cover_LS, cover_RT_LS, cover_air)  )
print('CI length: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
	%( np.mean(CI_LS[:,1] - CI_LS[:,0]), np.std(CI_LS[:,1] - CI_LS[:,0]),
	   np.mean(CI_RT_LS[:,1] - CI_RT_LS[:,0]), np.std(CI_RT_LS[:,1] - CI_RT_LS[:,0]),
	   np.mean(CI_LS_SIR[:,1] - CI_LS_SIR[:,0]), np.std(CI_LS_SIR[:,1] - CI_LS_SIR[:,0]) ))
