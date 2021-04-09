## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
n, p = 1000, 10

beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
n_sim = 1
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

	# np.cov( np.dot(Z, theta0), X )
	print('True beta: %.3f' %beta0)

	## solve by SIR+LS
	from nonlinear_causal import _2SCausal
	echo = _2SCausal._2SIR(sparse_reg=_2SCausal.elasticSUM(lam=1.))
	echo.fit(Z, X, cor_ZY)
	print('est beta based on 2SIR: %.3f' %echo.beta)

	## solve by 2sls
	from nonlinear_causal import _2SCausal
	LS = _2SCausal._2SLS(sparse_reg=_2SCausal.elasticSUM(lam=1.))
	LS.fit(LD_Z, cor_ZX, cor_ZY)
	print('est beta based on OLS: %.3f' %LS.beta)

	## solve by RT-2SLS
	from sklearn.preprocessing import power_transform, quantile_transform
	RT_X = power_transform(X.reshape(-1,1)).flatten()
	# RT_X = quantile_transform(X.reshape(-1,1), n_quantiles=n/10, output_distribution='normal')
	RT_cor_ZX = np.dot(Z.T, RT_X)
	RT_LS = _2SMethod._2SLS()
	RT_LS.fit(LD_Z, RT_cor_ZX, cor_ZY)
	print('est beta based on RT-OLS: %.3f' %RT_LS.beta)

	beta_LS.append(abs(LS.beta))
	beta_RT_LS.append(abs(RT_LS.beta))
	beta_LS_SIR.append(abs(echo.beta[0]))

# d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
# 	 'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (12,6)

# sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x="method", y="abs_beta", data=d)
# # ax = sns.swarmplot(x="method", y="abs_beta", data=d, color=".3", size=3.)
# plt.show()
