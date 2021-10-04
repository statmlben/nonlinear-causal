## Motivating examples to show why we need the combine-test

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
import random
from os import listdir
from os.path import isfile, join, isdir
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

mypath = '/home/ben/dataset/GenesToAnalyze'
gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name)) ]
# gene_folders = random.sample(gene_folders, 10)

def calculate_vif_(X, thresh=5.0, verbose=0):
	cols = X.columns
	variables = list(range(X.shape[1]))
	dropped = True
	while dropped:
		dropped = False
		vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
				for ix in range(X.iloc[:, variables].shape[1])]

		maxloc = vif.index(max(vif))
		if max(vif) > thresh:
			if verbose:
				print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
						'\' at index: ' + str(maxloc))
			del variables[maxloc]
			dropped = True
	if verbose:
		print('Remaining variables:')
		print(X.columns[variables])
	cols_new = cols[variables]
	return X.iloc[:, variables], cols_new

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
df = {'gene': [], 'p-value': [], 'beta': [], 'num_slice': []}
# NMNAT3July20_2SIR

interest_genes = ['APOC1',
				'APOC1P1',
				'APOE',
				'BCAM',
				'BCL3',
				'BIN1',
				'CBLC',
				'CEACAM19',
				'CHRNA2',
				'CLPTM1',
				'CYP27C1',
				'HLA-DRB5',
				'MS4A4A',
				'MS4A6A',
				'MTCH2',
				'NKPD1',
				'TOMM40',
				'ZNF296']

for folder_tmp in gene_folders:
# for folder_tmp in ['APOEJuly20_2SIR', 'TOMM40July20_2SIR']:
	if 'July20_2SIR' not in folder_tmp:
		continue
	gene_code = folder_tmp.replace('July20_2SIR', '')
	if gene_code not in interest_genes:
		continue
	print('\n##### Causal inference of %s #####' %gene_code)
	## load data
	dir_name = mypath+'/'+folder_tmp
	sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
	gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
	snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)
	## exclude the gene with nan in the dataset
	if sum_stat.isnull().sum().sum() + snp.isnull().sum().sum() + gene_exp.isnull().sum().sum() > 0:
		continue
	if not all(sum_stat.index == snp.columns):
		print('The cols in sum_stat is not corresponding to snp, we rename the sum_stat!')
		sum_stat.index = snp.columns
	## remove the collinear features
	snp, valid_cols = calculate_vif_(snp)
	sum_stat = sum_stat.loc[valid_cols]
	## n1 and n2 is pre-given
	n1, n2, p = len(gene_exp), 54162, snp.shape[1]
	LD_Z1, cov_ZX1 = np.dot(snp.values.T, snp.values), np.dot(snp.values.T, gene_exp.values.flatten())
	LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

	## Comb-2SIR
	data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .4*n1, .5*n1]
	comb_pvalue, comb_beta = [], []
	for data_in_slice_tmp in data_in_slice_lst:
		num_slice = int(int(n1) / data_in_slice_tmp)
		SIR = _2SIR(sparse_reg=None, data_in_slice=data_in_slice_tmp)
		## Stage-1 fit theta
		SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
		## Stage-2 fit beta
		SIR.fit_beta(LD_Z2, cov_ZY2, n2)
		## generate CI for beta
		SIR.test_effect(n2, LD_Z2, cov_ZY2)
		comb_beta.append(SIR.beta)
		comb_pvalue.append(SIR.p_value)
		df['gene'].append(gene_code)
		df['num_slice'].append(num_slice)
		df['p-value'].append(SIR.p_value)
		df['beta'].append(SIR.beta)
	correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
	print('Comb-2SIR beta: %.3f' %np.mean(comb_beta))
	print('p-value based on Comb-2SIR: %.5f' %correct_pvalue)
			
	## save the record
	df['gene'].append(gene_code)
	df['num_slice'].append('Comb')
	df['p-value'].append(correct_pvalue)
	df['beta'].append(np.mean(comb_beta))

df = pd.DataFrame.from_dict(df)

import seaborn as sns
import matplotlib.pyplot as plt

df['neg-log-p-value'] = - np.log10( df['p-value'] )

plt.rcParams["figure.figsize"] = (16,6)
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="gene", y="neg-log-p-value", hue="num_slice",
                 data=df, alpha=.7)
ax.set(xlabel="gene", ylabel="-log(p-value)")
plt.savefig('./figs/'+'sep30_ben_app_comb.png', dpi=500)
plt.show()
