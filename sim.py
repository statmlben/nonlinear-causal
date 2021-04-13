## simulate two-stage dataset

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
MSE_LS, MSE_RT_LS, MSE_LS_SIR = [], [], []
cover_LS, cover_RT_LS, cover_LS_SIR = [], [], []
n_sim = 100
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	Z, X, y = sim(n, p, theta0, beta0, case='cubic', feat='normal', range=.01)
	## normalize Z, X, y
	center = StandardScaler(with_std=False)
	Z, X, y = center.fit_transform(Z), X - X.mean(), y - y.mean()
	Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
	LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
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
	LS.CI_beta(Z2, alpha=.95)

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
	RT_LS.CI_beta(Z2, alpha=.95)

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
	echo.CI_beta(Z2, alpha=.95)

	## One sample method
	# echo.fit(Z, X, cor_ZY)

	print('est beta based on 2SIR: %.3f' %echo.beta)

	beta_LS.append(abs(LS.beta))
	beta_RT_LS.append(abs(RT_LS.beta))
	beta_LS_SIR.append(abs(echo.beta[0]))

d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
	 'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="method", y="abs_beta", data=d)
# ax = sns.swarmplot(x="method", y="abs_beta", data=d, color=".3", size=3.)
plt.show()