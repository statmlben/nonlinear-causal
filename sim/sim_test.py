## simulate two-stage dataset
# sim data to compare the difference btw OLS and SIR
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from nl_causal.ts_models import _2SLS, _2SIR
from nl_causal.linear_reg import L0_IC
from sklearn.preprocessing import power_transform, quantile_transform
import pandas as pd

n, p = 2000, 10
# for beta0 in [.05, .10, .15]:
df = {'true_beta': [], 'case': [], 'method': [], 'pct. of signif': []}
for beta0 in [.00, .05, .10, .15]:
	for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear', 'quad']:
	# for case in ['quad']:
		beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
		p_value = []
		n_sim = 1000
		if beta0 > 0:
			n_sim = 100
		for i in range(n_sim):
			theta0 = np.random.randn(p)
			# theta0[:int(.1*p)] = 0.
			# theta0 = np.ones(p)
			theta0 = theta0 / np.sqrt(np.sum(theta0**2))
			Z, X, y, phi = sim(n, p, theta0, beta0, case=case, feat='normal')			
			if abs(X).max() > 1e+7:
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
			LS = _2SLS(sparse_reg=None)
			## Stage-1 fit theta 
			LS.fit_theta(LD_Z1, cov_ZX1)
			## Stage-2 fit beta
			LS.fit_beta(LD_Z2, cov_ZY2, n2)
			## generate CI for beta
			LS.test_effect(n2, LD_Z2, cov_ZY2)
			
			# print('est beta based on OLS: %.3f; p-value: %.5f' %(LS.beta*y_scale, LS.p_value))

			## solve by RT-2SLS
			# RT_LS = LS
			RT_X1 = power_transform(X1.reshape(-1,1)).flatten()
			# RT_X1 = quantile_transform(X1.reshape(-1,1), output_distribution='normal')
			RT_cor_ZX1 = np.dot(Z1.T, RT_X1)
			RT_LS = _2SLS(sparse_reg=None)
			## Stage-1 fit theta
			RT_LS.fit_theta(LD_Z1, RT_cor_ZX1)
			## Stage-2 fit beta
			RT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
			## generate CI for beta
			RT_LS.test_effect(n2, LD_Z2, cov_ZY2)

			# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

			## solve by SIR+LS
			echo = _2SIR(sparse_reg=None)
			## Stage-1 fit theta
			echo.fit_theta(Z1, X1)
			## Stage-2 fit beta
			echo.fit_beta(LD_Z2, cov_ZY2, n2)
			## generate CI for beta
			echo.test_effect(n2, LD_Z2, cov_ZY2)
			# print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))

			data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .4*n1, .5*n1]
			comb_pvalue, comb_beta = [], []
			for data_in_slice_tmp in data_in_slice_lst:
				num_slice = int(int(n1) / data_in_slice_tmp)
				SIR = _2SIR(sparse_reg=None, data_in_slice=data_in_slice_tmp)
				## Stage-1 fit theta
				SIR.fit_theta(Z1=Z1, X1=X1)
				## Stage-2 fit beta
				SIR.fit_beta(LD_Z2, cov_ZY2, n2)
				## generate CI for beta
				SIR.test_effect(n2, LD_Z2, cov_ZY2)
				comb_beta.append(SIR.beta)
				comb_pvalue.append(SIR.p_value)

			# correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
			comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
			correct_pvalue = min(.5 - np.arctan(comb_T)/np.pi, 1.0)
			# print('Comb-2SIR beta: %.3f' %np.mean(comb_beta))
			# print('p-value based on Comb-2SIR: %.5f' %correct_pvalue)

			beta_LS.append(LS.beta*y_scale)
			beta_RT_LS.append(RT_LS.beta*y_scale)
			beta_LS_SIR.append(echo.beta*y_scale)
			p_value.append([LS.p_value, RT_LS.p_value, echo.p_value, correct_pvalue])

		# d = {'abs_beta': beta_LS+beta_RT_LS+beta_LS_SIR, 
		# 	'method': ['2SLS']*n_sim+['RT-2SLS']*n_sim+['2SIR']*n_sim+['Comb_2SIR']*n_sim}
		p_value = np.array(p_value)

		print('#'*60)
		print('simulation setting: case: %s, n: %d, p: %d, beta0: %.3f' 
				%(case,n,p, beta0))
		## estimation acc
		print('est beta: 2sls: %.3f(%.3f); RT_2sls: %.3f(%.3f); SIR: %.3f(%.3f)'
				%( np.mean(beta_LS), np.std(beta_LS), np.mean(beta_RT_LS), np.std(beta_RT_LS), 
				np.mean(beta_LS_SIR), np.std(beta_LS_SIR)))
		
		rej_2SLS, rej_RT2SLS = len(p_value[p_value[:,0]<.05])/len(p_value), len(p_value[p_value[:,1]<.05])/len(p_value)
		rej_SIR, rej_CombSIR = len(p_value[p_value[:,2]<.05])/len(p_value), len(p_value[p_value[:,3]<.05])/len(p_value)
		
		df['true_beta'].extend([beta0]*5)
		df['case'].extend([case]*5)
		df['method'].extend(['2SLS', 'PT_2SLS', '2SIR', 'Comb-2SIR', 'signif level: .05'])
		df['pct. of signif'].extend([rej_2SLS, rej_RT2SLS, rej_SIR, rej_CombSIR, .05])
		
		print('Rejection: 2sls: %.3f; RT_2sls: %.3f; SIR: %.3f; Comb_SIR: %.3f'
				%( len(p_value[p_value[:,0]<.05])/len(p_value),
				len(p_value[p_value[:,1]<.05])/len(p_value),
				len(p_value[p_value[:,2]<.05])/len(p_value),
				len(p_value[p_value[:,3]<.05])/len(p_value)))

df = pd.DataFrame(df)

import seaborn as sns
import matplotlib.pyplot as plt
sns.relplot(data=df, x="true_beta", y="pct. of signif", hue='method', style="method", col='case', kind='line', markers=True)
plt.show()


# ## plot estimation accuracy
# import numpy as np
# from sklearn.preprocessing import normalize
# from sim_data import sim
# from sklearn.preprocessing import StandardScaler
# from scipy import stats
# from sklearn.model_selection import train_test_split
# from nonlinear_causal import _2SCausal
# from sklearn.preprocessing import power_transform, quantile_transform

# n, p = 2000, 10
# d = {'beta': [], 'method': [], 'case': []}
# # for beta0 in [.05, .10, .15]:5
# for case in ['linear', 'log', 'cube-root', 'inverse', 'piecewise_linear']:
# # for case in ['normal', 'cate']:
# 	beta_LS, beta_RT_LS, beta_LS_SIR = [], [], []
# 	p_value = []
# 	n_sim = 100
# 	for i in range(n_sim):
# 		beta0 = .15
# 		theta0 = np.ones(p)
# 		# theta0 = np.random.randn(p)
# 		theta0 = theta0 / np.sqrt(np.sum(theta0**2))
# 		Z, X, y, phi = sim(n, p, theta0, beta0=beta0, case=case, feat='normal', range=.01)
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
# 		d['beta'].append(LS.beta*y_scale)
# 		d['case'].append(case)
# 		d['method'].append('2SLS')
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
# 		d['beta'].append(RT_LS.beta*y_scale)
# 		d['case'].append(case)
# 		d['method'].append('PT-2SLS')
# 		# print('est beta based on RT-OLS: %.3f; p-value: %.5f' %(RT_LS.beta*y_scale, RT_LS.p_value))

# 		## solve by SIR+LS
# 		echo = _2SCausal._2SIR(reg=None)
# 		## Stage-1 fit theta
# 		echo.fit_sir(Z1, X1)
# 		## Stage-2 fit beta
# 		echo.fit_reg(LD_Z2, cor_ZY2)
# 		## generate CI for beta
# 		echo.test_effect(n2, LD_Z2, cor_ZY2)
# 		d['beta'].append(echo.beta*y_scale)
# 		d['case'].append(case)
# 		d['method'].append('2SIR')
# 		# print('est beta based on 2SIR: %.3f; p-value: %.5f' %(echo.beta*y_scale, echo.p_value))

# 		beta_LS.append(LS.beta*y_scale)
# 		beta_RT_LS.append(RT_LS.beta*y_scale)
# 		beta_LS_SIR.append(echo.beta*y_scale)
# 		p_value.append([LS.p_value, RT_LS.p_value, echo.p_value])

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (12,6)	

# sns.set_theme(style="whitegrid")
# # ax = sns.boxplot(x="method", y="beta", hue='case', data=d)
# ax = sns.boxplot(x="case", y="beta", hue='method', data=d)
# ax.axhline(beta0, ls='--', color='r', alpha=.5)
# plt.show()