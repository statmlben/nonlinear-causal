import pandas as pd
from nl_causal.ts_models import _2SLS, _2SIR
from nl_causal.linear_reg import L0_IC
import numpy as np
from sklearn.preprocessing import normalize
from sim_data import sim
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import power_transform, quantile_transform
np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
df = {'gene': [], 'p-value': [], 'beta': [], 'method': [], 'CI': []}
gene_files = ['A1BGJuly20_2SIR', 'ABCA7July20_2SIR', 'ABHD8July20_2SIR', 'ACER1July20_2SIR', 'ACP5July20_2SIR', 
			'ACPTJuly20_2SIR', 'ACSBG2July20_2SIR', 'ACTL9July20_2SIR', 'ACTN4July20_2SIR', 'ADAMTS10July20_2SIR']
for file_tmp in gene_files:
	# file_tmp = 'ACTN4July20_2SIR'
	print('\n##### Causal inference of %s #####' %file_tmp)
	# diff gene: ACP5July20_2SIR, 
	## load data
	dir_name = '~/GenesToAnalyze/'+file_tmp
	sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
	gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
	snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)
	## n1 and n2 is pre-given
	n1, n2, p = len(gene_exp), 54162, snp.shape[1]
	LD_Z1, cov_ZX1 = np.dot(snp.values.T, snp.values), np.dot(snp.values.T, gene_exp.values.flatten())
	LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

	Ks = range(int(p/2)-1)
	## 2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	LS = _2SLS(sparse_reg=reg_model)
	## Stage-1 fit theta
	LS.fit_theta(LD_Z1, cov_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p_value and CI for beta
	LS.test_effect(n2, LD_Z2, cov_ZY2)
	if LS.beta > 0:
		LS.CI_beta(n1=n1, n2=n2, Z1=snp.values, X1=gene_exp.values.flatten(), LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	else:
		LS.theta = -LS.theta
		LS.beta = -LS.beta
		LS.CI_beta(n1=n1, n2=n2, Z1=-snp.values, X1=gene_exp.values.flatten(), LD_Z2=LD_Z2, cov_ZY2=-cov_ZY2)
	print('LS beta: %.3f' %LS.beta)
	print('p-value based on 2SLS: %.5f' %LS.p_value)
	print('CI based on 2SLS: %s' %(LS.CI))
	## save the record
	df['gene'].append(file_tmp)
	df['method'].append('2SLS')
	df['p-value'].append(LS.p_value)
	df['beta'].append(LS.beta)
	df['CI'].append(LS.CI)


	## PT-2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	PT_X1 = power_transform(gene_exp.values.reshape(-1,1), method='yeo-johnson').flatten()
	PT_cor_ZX1 = np.dot(snp.values.T, PT_X1)
	PT_LS = _2SLS(sparse_reg=reg_model)
	## Stage-1 fit theta
	PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
	## Stage-2 fit beta
	PT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p-value and CI for beta
	PT_LS.test_effect(n2, LD_Z2, cov_ZY2)
	if PT_LS.beta > 0:
		PT_LS.CI_beta(n1=n1, n2=n2, Z1=snp.values, X1=PT_X1, LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	else:
		PT_LS.theta = -PT_LS.theta
		PT_LS.beta = -PT_LS.beta
		PT_LS.CI_beta(n1=n1, n2=n2, Z1=-snp.values, X1=PT_X1, LD_Z2=LD_Z2, cov_ZY2=-cov_ZY2)
	# gene_exp.values.flatten()
	PT_LS.CI_beta(n1, n2, Z1=snp.values, X1=PT_X1, LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	print('PT-LS beta: %.3f' %PT_LS.beta)
	print('p-value based on PT-2SLS: %.5f' %PT_LS.p_value)
	print('CI based on 2SLS: %s' %(PT_LS.CI))
	## save the record
	df['gene'].append(file_tmp)
	df['method'].append('PT-2SLS')
	df['p-value'].append(PT_LS.p_value)
	df['beta'].append(PT_LS.beta)
	df['CI'].append(PT_LS.CI)


	## 2SIR
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	SIR = _2SIR(sparse_reg=reg_model)
	## Stage-1 fit theta
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	## Stage-2 fit beta
	SIR.fit_beta(LD_Z2, cov_ZY2, n2)
	## generate CI for beta
	SIR.test_effect(n2, LD_Z2, cov_ZY2)
	SIR.CI_beta(n1, n2, Z1=snp.values, X1=gene_exp.values.flatten(), LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	print('2SIR beta: %.3f' %SIR.beta)
	print('p-value based on 2SIR: %.5f' %SIR.p_value)
	print('CI based on 2SIR: %s' %(SIR.CI))
	## save the record
	df['gene'].append(file_tmp)
	df['method'].append('2SIR')
	df['p-value'].append(SIR.p_value)
	df['beta'].append(SIR.beta)
	df['CI'].append(SIR.CI)


df = pd.DataFrame.from_dict(df)

# ## stage_two = False: 1min 4s
# ## stage_two = True: 8min 36s

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")

# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="gene", y="p-value", hue="method", palette="dark", alpha=.6, height=6
)
g.despine(left=True)
g.set_axis_labels("gene", "p-value")
g.legend.set_title("")
plt.show()