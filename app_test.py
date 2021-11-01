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
df = {'gene': [], 'p-value': [], 'beta': [], 'method': []}
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
	# LD_Z1 = LD_Z1 + 0. * np.finfo(np.float32).eps * np.eye(p)
	# LD_Z2 = LD_Z2 + 0. * np.finfo(np.float32).eps * np.eye(p)

	Ks = range(int(p/2))
	## 2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	LS = _2SLS(sparse_reg=None)
	## Stage-1 fit theta
	LS.fit_theta(LD_Z1, cov_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p_value and CI for beta
	LS.test_effect(n2, LD_Z2, cov_ZY2)
	print('LS beta: %.3f' %LS.beta)
	print('p-value based on 2SLS: %.5f' %LS.p_value)
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('2SLS')
	df['p-value'].append(LS.p_value)
	df['beta'].append(LS.beta)


	## PT-2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	PT_X1 = power_transform(gene_exp.values.reshape(-1,1), method='yeo-johnson').flatten()
	PT_cor_ZX1 = np.dot(snp.values.T, PT_X1)
	PT_LS = _2SLS(sparse_reg=None)
	## Stage-1 fit theta
	PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
	## Stage-2 fit beta
	PT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p-value and CI for beta
	PT_LS.test_effect(n2, LD_Z2, cov_ZY2)

	# gene_exp.values.flatten()
	PT_LS.CI_beta(n1, n2, Z1=snp.values, X1=PT_X1, LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	print('PT-LS beta: %.3f' %PT_LS.beta)
	print('p-value based on PT-2SLS: %.5f' %PT_LS.p_value)
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('PT-2SLS')
	df['p-value'].append(PT_LS.p_value)
	df['beta'].append(PT_LS.beta)

	## 2SIR
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
					Ks=Ks, max_iter=50000, refit=False, find_best=False)
	SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1)
	## Stage-1 fit theta
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	## Stage-2 fit beta
	SIR.fit_beta(LD_Z2, cov_ZY2, n2)
	## generate CI for beta
	SIR.test_effect(n2, LD_Z2, cov_ZY2)
	print('2SIR beta: %.3f' %SIR.beta)
	print('2SIR eigenvalues: %.3f' %SIR.sir.eigenvalues_)
	print('p-value based on 2SIR: %.5f' %SIR.p_value)
			
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('2SIR')
	df['p-value'].append(SIR.p_value)
	df['beta'].append(SIR.beta)

	## Comb-2SIR
	data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .4*n1, .5*n1]
	comb_pvalue, comb_beta = [], []
	for data_in_slice_tmp in data_in_slice_lst:
		reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-5,3,.3),
						Ks=Ks, max_iter=50000, refit=False, find_best=False)
		SIR = _2SIR(sparse_reg=None, data_in_slice=data_in_slice_tmp)
		## Stage-1 fit theta
		SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
		## Stage-2 fit beta
		SIR.fit_beta(LD_Z2, cov_ZY2, n2)
		## generate CI for beta
		SIR.test_effect(n2, LD_Z2, cov_ZY2)
		comb_beta.append(SIR.beta)
		comb_pvalue.append(SIR.p_value)
	comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
	correct_pvalue = min( max(.5 - np.arctan(comb_T)/np.pi, np.finfo(np.float64).eps), 1.0)
	# correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
	print('Comb-2SIR beta: %.3f' %np.mean(comb_beta))
	print('p-value based on Comb-2SIR: %.5f' %correct_pvalue)
			
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('Comb-2SIR')
	df['p-value'].append(correct_pvalue)
	df['beta'].append(np.mean(comb_beta))

df = pd.DataFrame.from_dict(df)
# df.to_csv('oct04_ben_test.csv', index=False)

# ## stage_two = False: 1min 4s
# ## stage_two = True: 8min 36s