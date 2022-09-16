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
df = {'gene': [], 'gene_exp': [], 'fit_gene_exp': []}
# NMNAT3July20_2SIR

interest_genes = [
				# 'APOC1',
				'APOC1P1',
				# 'APOE',
				# 'BCAM',
				# 'BCL3',
				# 'BIN1',
				# 'CBLC',
				# 'CEACAM19',
				# 'CHRNA2',
				# 'CLPTM1',
				# 'CYP27C1',
				'HLA-DRB5',
				# 'MS4A4A',
				# 'MS4A6A',
				'MTCH2',
				# 'NKPD1',
				# 'TOMM40',
				# 'ZNF296'
				]

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
	# LD_Z1 = LD_Z1 + 0. * np.finfo(np.float32).eps * np.eye(p)
	# LD_Z2 = LD_Z2 + 0. * np.finfo(np.float32).eps * np.eye(p)
	
	SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1, n_directions=1)
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	fit_gene_exp = np.dot(snp.values, SIR.theta)
	df['gene'].extend([gene_code]*n1)
	df['fit_gene_exp'].extend(list(fit_gene_exp))
	df['gene_exp'].extend(list(gene_exp.values.flatten()))

import matplotlib.pyplot as plt
import seaborn as sns
df = pd.DataFrame(df)
sns.relplot(data=df, x="gene_exp", y="fit_gene_exp", hue="gene", col="gene", kind="scatter")
plt.show()