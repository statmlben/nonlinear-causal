## simulate two-stage dataset

# sim data to compare the difference btw OLS and SIR
# import numpy as np
# from sklearn.preprocessing import normalize
# from sim_data import sim
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# from sklearn.model_selection import train_test_split
# from nonlinear_causal import _2SCausal
# from sklearn.preprocessing import power_transform, quantile_transform

# n, p = 2000, 10
# for beta0 in [.05, .10, .15]:
# # for beta0 in [.00]:
# 	beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
# 	p_value = []
# 	n_sim = 100
# 	for i in range(n_sim):
# 		theta0 = np.random.randn(p)
# 		# theta0 = np.ones(p)
# 		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
# 		Z, X, y, phi = sim(n, p, theta0, beta0, case='cube-root', feat='normal', range=.01)
# 		if abs(X).max() > 1e+8:
# 			continue
# 		## normalize Z, X, y
# 		center = StandardScaler(with_std=False)
# 		mean_X, mean_y = X.mean(), y.mean()
# 		Z, X, y = center.fit_transform(Z), X - mean_X, y - mean_y
# 		y_scale = y.std()
# 		y = y / y_scale
# 		Z1, Z2, X1, X2, y1, y2 = train_test_split(Z, X, y, test_size=0.5, random_state=42)
# 		n1, n2 = len(Z1), len(Z2)
# 		# LD_Z, cor_ZX, cor_ZY = np.dot(Z.T, Z), np.dot(Z.T, X), np.dot(Z.T, y)
# 		LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
# 		LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
# 		# np.cov( np.dot(Z, theta0), X )
# 		# print('True beta: %.3f' %beta0)

# 		## solve by 2sls
# 		LS = _2SCausal._2SLS(reg=None)
# 		## Stage-1 fit theta 
# 		LS.fit_theta(LD_Z1, cor_ZX1)
# 		## Stage-2 fit beta
# 		LS.fit_beta(LD_Z2, cor_ZY2)
# 		## generate CI for beta
# 		LS.test_effect(n2, LD_Z2, cor_ZY2)
		
# 		# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

# 		## solve by RT-2SLS
# 		# RT_LS = LS
# 		RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
# 		# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
# 		RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
# 		RT_LS = _2SCausal._2SLS(reg=None)
# 		## Stage-1 fit theta
# 		RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
# 		## Stage-2 fit beta
# 		RT_LS.fit_beta(LD_Z2, cor_ZY2)
# 		## generate CI for beta
# 		RT_LS.test_effect(n2, LD_Z2, cor_ZY2)

# 		# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

# 		## solve by SIR+LS
# 		echo = _2SCausal._2SIR(reg=None)
# 		## Stage-1 fit theta
# 		echo.fit_sir(Z1, X1)
# 		## Stage-2 fit beta
# 		echo.fit_reg(LD_Z2, cor_ZY2)
# 		## generate CI for beta
# 		echo.test_effect(n2, LD_Z2, cor_ZY2)

# 		# print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))

# 		beta_LS.append(LS.beta*y_scale)
# 		beta_RT_LS.append(RT_LS.beta*y_scale)
# 		beta_LS_SIR.append(echo.beta*y_scale)
# 		p_value.append([LS.p_value, RT_LS.p_value, echo.p_value])

# 	d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
# 		'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim}
# 	p_value = np.array(p_value)

# 	print('#'*60)
# 	print('simulation setting: n: %d, p: %d, beta0: %.3f' %(n,p, beta0))
# 	## estimation acc
# 	print('est beta: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
# 			%( np.mean(beta_LS), np.std(beta_LS), np.mean(beta_RT_LS), np.std(beta_RT_LS), 
# 			np.mean(beta_LS_SIR), np.std(beta_LS_SIR)))

# 	print('Rejection: 2sls: %.3f; RT_2sls: %.3f; SIR: %.3f'
# 			%( len(p_value[p_value[:,0]<.05])/n_sim,
# 			len(p_value[p_value[:,1]<.05])/n_sim,
# 			len(p_value[p_value[:,2]<.05])/n_sim))

## plot estimation accuracy
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nonlinear_causal import _2SCausal
from sklearn.preprocessing import power_transform, quantile_transform

n, p = 2000, 10
d = {'beta': [], 'method': [], 'case': []}
# for beta0 in [.05, .10, .15]:5
for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear']:
# for case in ['normal', 'cate']:
	beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
	p_value = []
	n_sim = 100
	for i in range(n_sim):
		beta0 = .15
		theta0 = np.ones(p)
		# theta0 = np.random.randn(p)
		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
		Z, X, y, phi = sim(n, p, theta0, beta0=beta0, case=case, feat='normal', range=.01)
		if abs(X).max() > 1e+8:
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
		LD_Z1, cor_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1)
		LD_Z2, cor_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)
		# np.cov( np.dot(Z, theta0), X )
		# print('True beta: %.3f' %beta0)

		## solve by 2sls
		LS = _2SCausal._2SLS(reg=None)
		## Stage-1 fit theta 
		LS.fit_theta(LD_Z1, cor_ZX1)
		## Stage-2 fit beta
		LS.fit_beta(LD_Z2, cor_ZY2)
		## generate CI for beta
		LS.test_effect(n2, LD_Z2, cor_ZY2)
		d['beta'].append(LS.beta*y_scale)
		d['case'].append(case)
		d['method'].append('2SLS')
		# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

		## solve by RT-2SLS
		# RT_LS = LS
		RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
		# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
		RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
		RT_LS = _2SCausal._2SLS(reg=None)
		## Stage-1 fit theta
		RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
		## Stage-2 fit beta
		RT_LS.fit_beta(LD_Z2, cor_ZY2)
		## generate CI for beta
		RT_LS.test_effect(n2, LD_Z2, cor_ZY2)
		d['beta'].append(RT_LS.beta*y_scale)
		d['case'].append(case)
		d['method'].append('PT-2SLS')
		# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

		## solve by SIR+LS
		echo = _2SCausal._2SIR(reg=None)
		## Stage-1 fit theta
		echo.fit_sir(Z1, X1)
		## Stage-2 fit beta
		echo.fit_reg(LD_Z2, cor_ZY2)
		## generate CI for beta
		echo.test_effect(n2, LD_Z2, cor_ZY2)
		d['beta'].append(echo.beta*y_scale)
		d['case'].append(case)
		d['method'].append('2SIR')
		# print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))

		beta_LS.append(LS.beta*y_scale)
		beta_RT_LS.append(RT_LS.beta*y_scale)
		beta_LS_SIR.append(echo.beta*y_scale)
		p_value.append([LS.p_value, RT_LS.p_value, echo.p_value])

import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)	

sns.set_theme(style="whitegrid")
# ax = sns.boxplot(x="method", y="beta", hue='case', data=d)
ax = sns.boxplot(x="case", y="beta", hue='method', data=d)
ax.axhline(beta0, ls='--', color='r', alpha=.5)
plt.show()