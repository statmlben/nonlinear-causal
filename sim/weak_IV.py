## simulation for weak IVs
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
n, p = 2000, 10

beta_LS, beta_RT_LS, beta_LS_SIR, range_lst = [], [], [], []
n_sim = 500
for i in range(n_sim):
	for range_tmp in [.05, .07, .09, 0.11]:
		# theta0 = np.random.randn(p)
		theta0 = np.ones(p)
		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
		beta0 = 1.
		Z, X, y = sim(n, p, theta0, beta0, case='cubic', feat='uniform', range=range_tmp)
		## normalize Z, X, y
		center = StandardScaler(with_std=False)
		Z, X, y = center.fit_transform(Z), X - X.mean(), y - y.mean()
		LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)

		# np.cov( np.dot(Z, theta0), X )
		print('True beta: %.3f' %beta0)

		## solve by 2sls
		from nonlinear_causal import _2SMethod
		LS = _2SMethod._2SLS()
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

		## solve by SIR+LS
		from nonlinear_causal import _2SMethod
		echo = _2SMethod._2SIR()
		echo.fit(Z, X, cor_ZY)
		print('est beta based on 2SIR: %.3f' %echo.beta)

		beta_LS.append(abs(LS.beta))
		beta_RT_LS.append(abs(RT_LS.beta))
		beta_LS_SIR.append(abs(echo.beta[0]))
		range_lst.append(range_tmp)

d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR,
	 'range': range_lst*3,
	 'method': ['2SLS']*(4*n_sim)+['RT-2SLS']*(4*n_sim)+['2SIR']*(4*n_sim)}

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="range", y="abs_beta", hue='method', data=d)
# ax.axhline(1, ls='--', color='r', alpha=.3)
plt.show()