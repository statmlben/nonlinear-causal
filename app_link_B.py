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
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("oct04_ben_test_refined_genes.csv")
df['log-p-value'] = - np.log10( df['p-value'] )

mse_air, mse_mean, ue_air, ue_mean = [], [], [], []
mse_RT_LS, mse_LS, ue_RT_LS, ue_LS = [], [], [], []

link_plot = {'gene-code':[], 'gene-exp': [], 'phi': [], 'method': [] }

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

interest_genes = [
				# 'APOC1',
				# 'APOC1P1',
				'APOE',
				# 'BCAM',
				'BCL3',
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
				# 'TOMM40',
				# 'ZNF296'
				]

mypath = '/home/statmlben/dataset/GenesToAnalyze'
# gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name)) ]
np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

for gene_code in interest_genes:
	folder_tmp = gene_code + 'July20_2SIR'
	print('\n##### Link Estimation of %s #####' %gene_code)
	## load data
	dir_name = mypath+'/'+folder_tmp
	sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
	gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
	snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)
	# define interval of interests
	a = np.quantile(gene_exp.values, 0.05)
	b = np.quantile(gene_exp.values, 0.95)
	IoR = np.arange(a, b, (b-a)/100)
	# IoR = gene_exp.values.flatten()
	
	## exclude the gene with nan in the dataset
	if sum_stat.isnull().sum().sum() + snp.isnull().sum().sum() + gene_exp.isnull().sum().sum() > 0:
		continue
	if not all(sum_stat.index == snp.columns):
		print('The cols in sum_stat is not corresponding to snp, we rename the sum_stat!')
		sum_stat.index = snp.columns
	## remove the collinear features
	snp, valid_cols = calculate_vif_(snp)
	sum_stat = sum_stat.loc[valid_cols]
	B_num = 100
	n1, n2, p = len(gene_exp), 54162, snp.shape[1]
	for b in range(B_num):
		ind_tmp = np.random.choice(n1,n1)
		## n1 and n2 is pre-given
		X1_B, Z1_B = gene_exp.values.flatten()[ind_tmp], snp.values[ind_tmp]
		LD_Z1 = np.dot(Z1_B.T, Z1_B)
		cov_ZX1 = np.dot(Z1_B.T, X1_B)
		LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

		## 2SLS
		LS = _2SLS(sparse_reg=None)
		LS.fit_theta(LD_Z1, cov_ZX1)
		## Stage-2 fit beta
		LS.fit_beta(LD_Z2, cov_ZY2, n2)

		## PT-2SIR
		pt = PowerTransformer()
		# pt = QuantileTransformer()
		PT_X1 = pt.fit_transform(X1_B[:,np.newaxis]).flatten()
		PT_cor_ZX1 = np.dot(Z1_B.T, PT_X1)
		PT_LS = _2SLS(sparse_reg=None)
		## Stage-1 fit theta
		PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
		## Stage-2 fit beta
		PT_LS.fit_beta(LD_Z2, cov_ZY2, n2)

		pred_ior_RT_2SLS = pt.transform(IoR.reshape(-1,1)).flatten()
		pred_ior_RT_2SLS = np.sign(PT_LS.beta)*pred_ior_RT_2SLS
		pred_RT_2SLS = pt.transform(X1_B[:,np.newaxis]).flatten()
		pred_RT_2SLS = np.sign(PT_LS.beta)*pred_RT_2SLS
		pred_ior_RT_2SLS = pred_ior_RT_2SLS - np.mean(pred_RT_2SLS)
		pred_RT_2SLS = pred_RT_2SLS - np.mean(pred_RT_2SLS)

		## 2SIR
		SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1)
		# SIR.cond_mean = KNeighborsRegressor(n_neighbors=10)
		# gamma = .5/np.median(pairwise_distances(X1_B[:,np.newaxis]).flatten()**2)
		# SIR.cond_mean = KernelRidge(kernel='rbf', alpha=1e-8, gamma=gamma)

		## Stage-1 fit theta
		SIR.fit_theta(Z1=Z1_B, X1=X1_B)
		## Stage-2 fit beta
		SIR.fit_beta(LD_Z2, cov_ZY2, n2)
		
		## AIR to fit link
		# params = {'n_neighbors':[10, 30, 50, 70, 90, 110]}
		# gs_knn = GridSearchCV(KNeighborsRegressor(), params, 
		# 						scoring='neg_mean_squared_error')
		# gs_knn.fit(X1_B[:,np.newaxis], np.dot(Z1_B, SIR.theta))
		
		# # find the best param
		params = {'alpha':10.**np.arange(-3, 3., .4), 'gamma': 10.**np.arange(-3, 3., .4)}
		gs_rbf = GridSearchCV(KernelRidge(kernel='rbf'), params, cv=3,
								scoring='neg_mean_squared_error')
		gs_rbf.fit(X1_B[:,np.newaxis], np.dot(Z1_B, SIR.theta))
		# find the best param
		SIR.cond_mean = KernelRidge(kernel='rbf', alpha=gs_rbf.best_params_['alpha'], 
									gamma=gs_rbf.best_params_['gamma'])

		# SIR.cond_mean = KernelRidge(kernel='rbf', alpha=8., 
		# 							gamma=126.)
		SIR.fit_link(Z1=Z1_B, X1=X1_B)

		# print('est beta based on 2SIR: %.3f' %(echo.beta*y_scale))
		pred_phi = SIR.link(X=X1_B[:,np.newaxis]).flatten()
		pred_mean = SIR.cond_mean.predict(X1_B[:,np.newaxis]).flatten()

		pred_ior = SIR.link(X=IoR[:,None]).flatten()
		pred_mean_ior = SIR.cond_mean.predict(IoR[:,None]).flatten()

		link_plot['gene-code'].extend([gene_code]*len(IoR))
		link_plot['gene-exp'].extend(IoR)
		link_plot['phi'].extend(list(pred_ior_RT_2SLS))
		link_plot['method'].extend(['PT-2SLS']*len(IoR))
		
		link_plot['gene-code'].extend([gene_code]*len(IoR))
		link_plot['gene-exp'].extend(IoR)
		link_plot['phi'].extend(list(pred_ior))
		link_plot['method'].extend(['2SIR+AIR']*len(IoR))

		link_plot['gene-code'].extend([gene_code]*len(IoR))
		link_plot['gene-exp'].extend(IoR)
		link_plot['phi'].extend(list(IoR*np.sign(LS.beta)))
		link_plot['method'].extend(['2SLS']*len(IoR))

	# link_plot['gene-code'].extend([gene_code]*len(IoR))
	# link_plot['gene-exp'].extend(IoR)
	# link_plot['phi'].extend(list(pred_mean_ior))
	# link_plot['method'].extend(['Cond-mean']*len(IoR))

link_plot = pd.DataFrame(link_plot)

import seaborn as sns
import matplotlib.pyplot as plt

for gene_code in interest_genes:
	test_tmp = df[df['gene']==gene_code]
	title_tmp = gene_code+" ( Method:-log_10(p-value) )"+ '\n' + \
				'2SLS: '+str(test_tmp[test_tmp['method']=='2SLS']['log-p-value'].values)+'; '+\
				'PT-2SLS: '+str(test_tmp[test_tmp['method']=='PT-2SLS']['log-p-value'].values)+'; '+\
				'2SIR: '+str(test_tmp[test_tmp['method']=='2SIR']['log-p-value'].values)+'; '+\
				'Comb-2SIR: '+str(test_tmp[test_tmp['method']=='Comb-2SIR']['log-p-value'].values)

	plt.rcParams["figure.figsize"] = (20,6)
	sns.set_theme(style="whitegrid")
	sns.lineplot(data=link_plot[link_plot['gene-code'] == gene_code], 
				x="gene-exp", y="phi", hue="method", legend = True,
	style="method", alpha=.7).set_title(title_tmp)
	plt.savefig('./figs/'+gene_code+"-link.png", dpi=500)
	plt.show()