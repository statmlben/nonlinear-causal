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
from nl_causal.base.preprocessing import calculate_vif_, variance_threshold_selector
from sklearn.feature_selection import VarianceThreshold

valid_iv_th = 0.005
vif_thresh = 2.5

mypath = '/home/ben/dataset/GenesToAnalyze'
gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name))]
# gene_folders = random.sample(gene_folders, 10)

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
df = {'gene': [], 'p-value': [], 'beta': [], 'method': [], 'R2': []}
# NMNAT3July20_2SIR

interest_genes = [
				# 'APOC1',
				# 'APOC1P1',
				'APOE',
				# 'BCAM',
				# 'BCL3',
				# 'BIN1',
				# 'CBLC',
				# 'CEACAM19',
				# 'CHRNA2',
				# 'CLPTM1',
				# 'CYP27C1',
				# 'HLA-DRB5',
				# 'MS4A4A',
				# 'MS4A6A',
				# 'MTCH2',
				# 'NKPD1',
				'TOMM40',
				# 'ZNF296'
				]
# B3GALT1
for folder_tmp in gene_folders:
	if 'July20_2SIR' not in folder_tmp:
		continue
	gene_code = folder_tmp.replace('July20_2SIR', '')
	if gene_code not in interest_genes:
		continue
	print('\n##### Causal inference of %s #####' %gene_code)
	## load data
	dir_name = mypath+'/'+folder_tmp
	sum_stat = pd.read_csv(dir_name+"/sum_stat_vif.csv", sep=' ', index_col=0)
	gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
	snp = pd.read_csv(dir_name+"/snp_vif.csv", sep=' ', index_col=0)
	## exclude the gene with nan in the dataset
	if sum_stat.isnull().sum().sum() + snp.isnull().sum().sum() + gene_exp.isnull().sum().sum() > 0:
		continue
	if not all(sum_stat.index == snp.columns):
		print('The cols in sum_stat is not corresponding to snp, we rename the sum_stat!')
		sum_stat.index = snp.columns
	
	### remove low variance features
	# snp = variance_threshold_selector(snp, threshold=0.1)
	# sum_stat = sum_stat.loc[snp.columns]

	# remove the collinear features
	# doi:10.1007/s11135-017-0584-6
	# snp, valid_cols = calculate_vif_(snp, thresh=vif_thresh)
	# sum_stat = sum_stat.loc[valid_cols]

	## remove the weak IVs

	## n1 and n2 is pre-given
	n1, n2, p = len(gene_exp), 54162, snp.shape[1]
	LD_Z1, cov_ZX1 = np.dot(snp.values.T, snp.values), np.dot(snp.values.T, gene_exp.values.flatten())
	LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

	Ks = range(int(p/2) - 1)
	## 2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3),
					Ks=Ks, max_iter=10000, refit=False, find_best=False)
	LS = _2SLS(sparse_reg=reg_model)
	## Stage-1 fit theta
	LS.fit_theta(LD_Z1, cov_ZX1)
	sigma1 = np.std(gene_exp.values.flatten() - LS.theta_norm*snp.dot(LS.theta).values)
	## refine Ks after theta
	valid_iv_th = 1.96*sigma1 / np.sqrt(n1)
	p_valid = sum(abs(LS.theta) > valid_iv_th)
	LS.sparse_reg.Ks = range(int(p_valid/2)-1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p_value and CI for beta
	LS.test_effect(n2, LD_Z2, cov_ZY2)
	LS_R2 = np.var( snp.dot(LS.theta)*LS.theta_norm ) / np.var(np.array(gene_exp))
	LS_F = (n1 - p - 1) / p * ( LS_R2 / (1 - LS_R2) )
	print('-'*20)
	print('LS stage 1 R2: %.3f' %LS_R2)
	print('LS beta: %.3f' %LS.beta)
	print('p-value based on 2SLS: %.5f' %LS.p_value)
	
	## compute R2 for the 2SLS
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('2SLS')
	df['p-value'].append(LS.p_value)
	df['beta'].append(LS.beta)
	df['R2'].append(LS_R2)

	## PT-2SLS
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3),
					Ks=Ks, max_iter=10000, refit=False, find_best=False)
	PT_X1 = power_transform(gene_exp.values.reshape(-1,1), method='yeo-johnson').flatten()
	PT_X1 = PT_X1 - PT_X1.mean()
	PT_cor_ZX1 = np.dot(snp.values.T, PT_X1)
	PT_LS = _2SLS(sparse_reg=reg_model)
	## Stage-1 fit theta
	PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
	sigma1 = np.std(PT_X1 - LS.theta_norm*snp.dot(PT_LS.theta).values)
	## refine Ks after theta
	valid_iv_th = 1.96*sigma1 / np.sqrt(n1)
	p_valid = sum(abs(PT_LS.theta) > valid_iv_th)
	PT_LS.sparse_reg.Ks = range(int(p_valid/2)-1)
	## Stage-2 fit beta
	PT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
	## produce p-value and CI for beta
	PT_LS.test_effect(n2, LD_Z2, cov_ZY2)
	PT_LS_R2 = np.var( snp.dot(PT_LS.theta)*PT_LS.theta_norm ) / np.var(PT_X1)

	# gene_exp.values.flatten()
	# PT_LS.CI_beta(n1, n2, Z1=snp.values, X1=PT_X1, LD_Z2=LD_Z2, cov_ZY2=cov_ZY2)
	print('-'*20)
	print('PT-LS stage 1 R2: %.3f' %PT_LS_R2)
	print('PT-LS beta: %.3f' %PT_LS.beta)
	print('p-value based on PT-2SLS: %.5f' %PT_LS.p_value)

	## save the record
	df['gene'].append(gene_code)
	df['method'].append('PT-2SLS')
	df['p-value'].append(PT_LS.p_value)
	df['beta'].append(PT_LS.beta)
	df['R2'].append(PT_LS_R2)

	## 2SIR
	reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3),
					Ks=Ks, max_iter=10000, refit=False, find_best=False)
	SIR = _2SIR(sparse_reg=reg_model, data_in_slice=0.2*n1)
	## Stage-1 fit theta
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	# refine Ks based on estimated theta
	p_valid = sum(abs(SIR.theta) > valid_iv_th)
	SIR.sparse_reg.Ks = range(int(p_valid/2)-1)
	## Stage-2 fit beta
	SIR.fit_beta(LD_Z2, cov_ZY2, n2)
	## generate CI for beta
	SIR.test_effect(n2, LD_Z2, cov_ZY2)
	print('-'*20)
	print('2SIR eigenvalues: %.3f' %SIR.sir.eigenvalues_)
	print('2SIR beta: %.3f' %SIR.beta)
	print('p-value based on 2SIR: %.5f' %SIR.p_value)
			
	## save the record
	df['gene'].append(gene_code)
	df['method'].append('2SIR')
	df['p-value'].append(SIR.p_value)
	df['beta'].append(SIR.beta)
	df['R2'].append(SIR.sir.eigenvalues_[0])

	## Comb-2SIR
	data_in_slice_lst = [.05*n1, .1*n1, .2*n1, .3*n1, .5*n1]
	comb_pvalue, comb_beta, comb_eigenvalue = [], [], []
	for data_in_slice_tmp in data_in_slice_lst:
		reg_model = L0_IC(fit_intercept=None, alphas=10**np.arange(-3,3,.3),
						Ks=Ks, max_iter=10000, refit=False, find_best=False)
		SIR = _2SIR(sparse_reg=reg_model, data_in_slice=data_in_slice_tmp)
		## Stage-1 fit theta
		SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
		# refine Ks based on estimated theta
		p_valid = sum(abs(SIR.theta) > valid_iv_th)
		SIR.sparse_reg.Ks = range(int(p_valid/2)-1)
		## Stage-2 fit beta
		SIR.fit_beta(LD_Z2, cov_ZY2, n2)
		## generate CI for beta
		SIR.test_effect(n2, LD_Z2, cov_ZY2)
		comb_beta.append(SIR.beta)
		comb_pvalue.append(SIR.p_value)
		comb_eigenvalue.append(SIR.sir.eigenvalues_[0])
	comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
	correct_pvalue = min( max(.5 - np.arctan(comb_T)/np.pi, np.finfo(np.float64).eps), 1.0)
	# correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
	print('Comb-2SIR eigenvalues: %.3f' %np.mean(comb_eigenvalue))
	print('Comb-2SIR beta: %.3f' %np.mean(comb_beta))
	print('p-value based on Comb-2SIR: %.5f' %correct_pvalue)

	## save the record
	df['gene'].append(gene_code)
	df['method'].append('Comb-2SIR')
	df['p-value'].append(correct_pvalue)
	df['beta'].append(np.mean(comb_beta))
	df['R2'].append(np.mean(comb_eigenvalue))

df = pd.DataFrame.from_dict(df)
# df.to_csv('Apr10_22_app_test.csv', index=False)
