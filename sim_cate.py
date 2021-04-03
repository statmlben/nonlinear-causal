## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
n, p = 2000, 10

beta_LS, beta_RT_LS, beta_LS_SIR, case_lst = [], [], [], []
n_sim = 500
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	for case in ['linear', 'exp', 'cubic', 'inverse']:
		Z, X, y = sim(n, p, theta0, beta0, case=case, feat='cate', range=.06)
		## normalize Z, X, y
		center = StandardScaler(with_std=False)
		Z, X, y = center.fit_transform(Z), X - X.mean(), y - y.mean()
		LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)

		# np.cov( np.dot(Z, theta0), X )
		print('True beta: %.3f' %beta0)

		## solve by 2sls
		from nonlinear_causal import _2sls
		LS = _2sls._2SLS()
		LS.fit(LD_Z, cor_ZX, cor_ZY)
		print('est beta based on OLS: %.3f' %LS.beta)

		## solve by RT-2SLS
		from sklearn.preprocessing import power_transform, quantile_transform
		RT_X = power_transform(X.reshape(-1,1)).flatten()
		# RT_X = quantile_transform(X.reshape(-1,1), n_quantiles=n/10, output_distribution='normal')
		RT_cor_ZX = np.dot(Z.T, RT_X)
		RT_LS = _2sls._2SLS()
		RT_LS.fit(LD_Z, RT_cor_ZX, cor_ZY)
		print('est beta based on RT-OLS: %.3f' %RT_LS.beta)

		## solve by SIR+LS
		from nonlinear_causal import _2sls
		echo = _2sls.SIR_LS()
		echo.fit(Z, X, cor_ZY)
		print('est beta based on 2SIR: %.3f' %echo.beta)

		beta_LS.append(abs(LS.beta))
		beta_RT_LS.append(abs(RT_LS.beta))
		beta_LS_SIR.append(abs(echo.beta[0]))
		case_lst.append(case)

d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
	 'nonlinear-link': case_lst*3,
	 'method': ['2SLS']*(4*n_sim)+['RT-2SLS']*(4*n_sim)+['2SIR']*(4*n_sim)}

d_2SIR = {'abs_beta': beta_LS_SIR,  'nonlinear-link': case_lst}

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="nonlinear-link", y="abs_beta", hue="method", data=d)
# ax = sns.boxplot(x="nonlinear-link", y="abs_beta", data=d_2SIR)
# sns.stripplot(x="nonlinear-link", y="abs_beta",
#			   size=3, alpha=0.7, data=d, dodge=True)
plt.show()

## simulation for cate
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
n, p = 2000, 10

beta_LS, beta_RT_LS, beta_LS_SIR, case_lst, feat_lst = [], [], [], [], []
n_sim = 500
for i in range(n_sim):
	# theta0 = np.random.randn(p)
	theta0 = np.ones(p)
	theta0 = theta0 / np.sqrt(np.sum(theta0**2))
	beta0 = 1.
	for case in ['linear', 'exp', 'cubic', 'inverse']:
		for feat in ['normal', 'cate']:
			Z, X, y = sim(n, p, theta0, beta0, case=case, feat='cate', range=.06)
			## normalize Z, X, y
			center = StandardScaler(with_std=False)
			Z, X, y = center.fit_transform(Z), X - X.mean(), y - y.mean()
			LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)

			# np.cov( np.dot(Z, theta0), X )
			print('True beta: %.3f' %beta0)
			
			## solve by SIR+LS
			from nonlinear_causal import _2sls
			echo = _2sls.SIR_LS()
			echo.fit(Z, X, cor_ZY)
			print('est beta based on 2SIR: %.3f' %echo.beta)

			beta_LS_SIR.append(abs(echo.beta[0]))
			case_lst.append(case)
			feat_lst.append(feat)

d = {'abs_beta': beta_LS_SIR, 
	 'nonlinear-link': case_lst,
	 'feats': feat_lst}

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="nonlinear-link", y="abs_beta", hue="feats", data=d)
# ax = sns.boxplot(x="nonlinear-link", y="abs_beta", data=d_2SIR)
# sns.stripplot(x="nonlinear-link", y="abs_beta",
#			   size=3, alpha=0.7, data=d, dodge=True)
plt.show()