## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
n, p = 1000, 100

beta_LS, beta_LS_SIR = [], []
n_sim = 50
for i in range(n_sim):
	theta0 = np.random.randn(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	Z, X, y = sim(n, p, theta0, beta0, case='linear', feat='normal')
	## normalize Z, X, y
	Z, X, y = Z - Z.mean(), X - X.mean(), y - y.mean()
	LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)

	print('True beta: %.3f' %beta0)

	## solve by 2sls
	from nonlinear_causal import _2sls
	LS = _2sls._2SLS()
	LS.fit(LD_Z, cor_ZX, cor_ZY)
	print('est beta based on 2SLS: %.3f' %LS.beta)

	## solve by SIR+LS
	from nonlinear_causal import _2sls
	echo = _2sls.SIR_LS()
	echo.fit(Z, X, cor_ZY)
	print('est beta based on SIR+LS: %.3f' %echo.beta)

	beta_LS.append(abs(LS.beta))
	beta_LS_SIR.append(abs(echo.beta[0]))

d = {'abs_beta': beta_LS+beta_LS_SIR, 
	 'method': ['LS']*n_sim+['SIR+LS']*n_sim}

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="method", y="abs_beta", data=d)
ax = sns.swarmplot(x="method", y="abs_beta", data=d, color=".3", size=3.)
plt.show()
