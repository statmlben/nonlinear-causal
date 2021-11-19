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

link_plot = {'gene-code':[], 'gene-exp': [], 'phi': [], 'snp_idx': [], 'method': [] }
link_r = {'gene-code':[], 'r2': [], 'method': []}

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
				# 'APOE',
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

mypath = '/home/ben/dataset/GenesToAnalyze'
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
	LD_Z1 = np.dot(snp.values.T, snp.values)
	cov_ZX1 = np.dot(snp.values.T, gene_exp.values.flatten())
	LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

	## 2SLS
	n_demo = n1
	LS = _2SLS(sparse_reg=None)
	LS.fit_theta(LD_Z1, cov_ZX1)
	## Stage-2 fit beta
	LS.fit_beta(LD_Z2, cov_ZY2, n2)
	left_LS = gene_exp.values.flatten()
	norm_phi_LS = np.max(np.abs(gene_exp.values.flatten()))
	right_LS = LS.theta_norm * np.dot(snp.values, LS.theta) / norm_phi_LS
	left_LS = left_LS / norm_phi_LS
	_, _, r_value_LS, _, _ = stats.linregress(left_LS, right_LS)

	## PT-2SIR
	pt = PowerTransformer()
	# pt = QuantileTransformer()
	PT_X1 = pt.fit_transform(gene_exp.values).flatten()
	PT_cor_ZX1 = np.dot(snp.values.T, PT_X1)
	PT_LS = _2SLS(sparse_reg=None)
	## Stage-1 fit theta
	PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
	## Stage-2 fit beta
	PT_LS.fit_beta(LD_Z2, cov_ZY2, n2)
	left_PTLS = PT_X1
	norm_phi_PTLS = np.max( np.abs(left_PTLS) )
	right_PTLS = PT_LS.theta_norm * np.dot(snp.values, PT_LS.theta) / norm_phi_PTLS
	left_PTLS = left_PTLS / norm_phi_PTLS
	_, _, r_value_PTLS, _, _ = stats.linregress(left_PTLS, right_PTLS)

	## 2SIR
	X1, Z1 = gene_exp.values.flatten(), snp.values
	X_dis = pairwise_distances(X1[:,np.newaxis]).flatten()
	gamma = 1/np.quantile(X_dis**2, .5)
	# params = {'alpha':10.**np.arange(-4, 4., .5), 'gamma': gamma*2.**np.arange(-4, 4, .5)}
	params = {'alpha':10.**np.arange(-4, 4., .5), 'gamma': [gamma]}
	Z1_train, Z1_valid, X1_train, X1_valid = train_test_split(Z1, X1, test_size=0.33, random_state=42)	
	LD_Z1 = np.dot(Z1_train.T, Z1_train)
	cov_ZX1 = np.dot(Z1_train.T, X1_train)
	LD_Z2, cov_ZY2 = LD_Z1/len(X1_train)*n2, sum_stat.values.flatten()*n2
	cv_results = []
	for alpha_tmp in params['alpha']:
		for gamma_tmp in params['gamma']:
			SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1)
			SIR.cond_mean = KernelRidge(kernel='rbf', alpha=alpha_tmp, 
							gamma=gamma_tmp)

			## Stage-1 fit theta
			SIR.fit_theta(Z1=Z1_train, X1=X1_train)
			## Stage-2 fit beta
			SIR.fit_beta(LD_Z2, cov_ZY2, n2)
			SIR.fit_link(Z1=Z1_train, X1=X1_train)
			err_tmp = np.mean((SIR.link(X1_valid[:,np.newaxis]) - np.dot(Z1_valid, SIR.theta))**2)
			cv_results.append([alpha_tmp, gamma_tmp, err_tmp])
	cv_results = np.array(cv_results)
	opt_idx = np.argmin(cv_results[:,-1])
	opt_alpha, opt_gamma = cv_results[opt_idx, 0], cv_results[opt_idx,1]


	SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1)
	# SIR.cond_mean = KNeighborsRegressor(n_neighbors=10)
	# SIR.cond_mean = KernelRidge(kernel='rbf')

	# SIR.cond_mean = IsotonicRegression(increasing='auto',
	# 								out_of_bounds='clip')

	## Stage-1 fit theta
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	## Stage-2 fit beta
	SIR.fit_beta(LD_Z2, cov_ZY2, n2)
	## AIR to fit link
	## kernel version	

	## fix best alpha + gamma
	# SIR.cond_mean = KernelRidge(kernel='rbf', alpha=15., 
	# 							gamma=251.)
	# SIR.cond_mean = KernelRidge(kernel='polynomial', alpha=0.000001, degree=25,
	# 							gamma=1.)

	## tune alpha + gamma
	# find the best param
	SIR.cond_mean = KernelRidge(kernel='rbf', alpha=opt_alpha, 
								gamma=opt_gamma)

	## tune alpha
	# X_dis = pairwise_distances(gene_exp.values).flatten()
	# # gamma = 1 / np.std(X_dis)**2
	# gamma = 2/np.quantile(X_dis**2, .5)
	# params = {'alpha':10.**np.arange(-4, 3, .1)}
	# gs_rbf = GridSearchCV(KernelRidge(kernel='rbf'), params, cv=3,
	# 						scoring='neg_mean_squared_error')
	# gs_rbf.fit(gene_exp.values, np.dot(snp.values, SIR.theta))
	# # find the best param
	# SIR.cond_mean = KernelRidge(kernel='rbf', alpha=gs_rbf.best_params_['alpha'], 
	# 							gamma=gamma)

	## knn version
	# params = {'n_neighbors':[10, 30, 50, 70, 90, 110]}
	# gs_knn = GridSearchCV(KNeighborsRegressor(), params, 
	# 						scoring='neg_mean_squared_error')
	# gs_knn.fit(gene_exp.values, np.dot(snp.values, SIR.theta))
	# # find the best param
	# SIR.cond_mean = KNeighborsRegressor(n_neighbors=gs_knn.best_params_['n_neighbors'])

	SIR.fit_link(Z1=snp.values, X1=gene_exp.values.flatten())

	# print('est beta based on 2SIR: %.3f' %(echo.beta*y_scale))
	left_SIR = SIR.link(X=gene_exp.values).flatten()
	norm_phi_SIR = np.max( np.abs(left_SIR) )
	right_SIR = np.dot(snp.values, SIR.theta) / norm_phi_SIR
	left_SIR = left_SIR / norm_phi_SIR
	_, _, r_value_SIR, _, _ = stats.linregress(left_SIR, right_SIR)

	X1_tmp_lst = list(gene_exp.values.flatten())
	link_plot['gene-code'].extend([gene_code]*n_demo)
	link_plot['gene-exp'].extend(X1_tmp_lst)
	link_plot['phi'].extend(list(left_PTLS))
	link_plot['snp_idx'].extend(list(right_PTLS))
	link_plot['method'].extend(['PT-2SLS']*n_demo)

	link_r['gene-code'].append(gene_code)
	link_r['method'].append('PT-2SLS')
	link_r['r2'].append(r_value_PTLS)

	
	link_plot['gene-code'].extend([gene_code]*n_demo)
	link_plot['gene-exp'].extend(X1_tmp_lst)
	link_plot['phi'].extend(list(left_SIR))
	link_plot['snp_idx'].extend(list(right_SIR))
	link_plot['method'].extend(['2SIR+AIR']*n_demo)

	link_r['gene-code'].append(gene_code)
	link_r['method'].append('2SIR+AIR')
	link_r['r2'].append(r_value_SIR)

	link_plot['gene-code'].extend([gene_code]*n_demo)
	link_plot['gene-exp'].extend(X1_tmp_lst)
	link_plot['phi'].extend(list(left_LS))
	link_plot['snp_idx'].extend(list(right_LS))
	link_plot['method'].extend(['2SLS']*n_demo)

	link_r['gene-code'].append(gene_code)
	link_r['method'].append('2SLS')
	link_r['r2'].append(r_value_LS)

	# link_plot['gene-code'].extend([gene_code]*n_demo)
	# link_plot['gene-exp'].extend(IoR)
	# link_plot['phi'].extend(list(pred_phi))
	# link_plot['method'].extend(['Cond-mean']*n_demo)

link_plot = pd.DataFrame(link_plot)
link_r = pd.DataFrame(link_r)

import seaborn as sns
import matplotlib.pyplot as plt

# link curve plot
# for gene_code in interest_genes:
# 	test_tmp = df[df['gene']==gene_code]
# 	title_tmp = gene_code+" ( Method:-log_10(p-value) )"+ '\n' + \
# 				'2SLS: '+str(test_tmp[test_tmp['method']=='2SLS']['log-p-value'].values)+'; '+\
# 				'PT-2SLS: '+str(test_tmp[test_tmp['method']=='PT-2SLS']['log-p-value'].values)+'; '+\
# 				'2SIR: '+str(test_tmp[test_tmp['method']=='2SIR']['log-p-value'].values)+'; '+\
# 				'Comb-2SIR: '+str(test_tmp[test_tmp['method']=='Comb-2SIR']['log-p-value'].values)

# 	plt.rcParams["figure.figsize"] = (10,6)
# 	sns.set_theme(style="whitegrid")
# 	sns.lineplot(data=link_plot[link_plot['gene-code'] == gene_code], 
# 				x="gene-exp", y="phi", hue="method", legend = True,
# 	style="method", alpha=.7).set_title(title_tmp)
# 	plt.savefig('./figs/'+gene_code+"-link.png", dpi=500)
# 	plt.show()

# scatter plot
for gene_code in interest_genes:
	test_tmp = df[df['gene']==gene_code]
	r_tmp = link_r[link_r['gene-code']==gene_code]

	# title_tmp = gene_code+" ( Method:-log_10(p-value) )"+ '\n' + \
	# 			'2SLS: '+str(test_tmp[test_tmp['method']=='2SLS']['log-p-value'].values)+'; '+\
	# 			'PT-2SLS: '+str(test_tmp[test_tmp['method']=='PT-2SLS']['log-p-value'].values)+'; '+\
	# 			'2SIR: '+str(test_tmp[test_tmp['method']=='2SIR']['log-p-value'].values)+'; '+\
	# 			'Comb-2SIR: '+str(test_tmp[test_tmp['method']=='Comb-2SIR']['log-p-value'].values)
	title_tmp = gene_code+" ( Method: R2-value for the Estimated Stage 1 Eqaution )"+ '\n' + \
				'2SLS: '+str(r_tmp[r_tmp['method']=='2SLS']['r2'].values)+'; '+\
				'PT-2SLS: '+str(r_tmp[r_tmp['method']=='PT-2SLS']['r2'].values)+'; '+\
				'2SIR: '+str(r_tmp[r_tmp['method']=='2SIR+AIR']['r2'].values)

	plt.rcParams["figure.figsize"] = (10,6)
	sns.set_theme(style="whitegrid")
	## plot R2 for the first equation
	lm = sns.lmplot(data=link_plot[link_plot['gene-code'] == gene_code], legend = True, 
				scatter_kws={"s": 5},
				x="snp_idx", y="phi", hue='method', col="method", sharex=False,sharey=False)

	## plot for True pattern 
	# lm = sns.lmplot(data=link_plot[link_plot['gene-code'] == gene_code], legend=True, scatter_kws={"s": 5},
	# 		x="snp_idx", y="gene-exp", hue='method', col="method", fit_reg=False, sharex=False,sharey=False)
			# .fig.suptitle(title_tmp)
	axes = lm.axes
	axes[0,0].set_ylim(-.75, .75)
	axes[0,1].set_ylim(-.75, .75)
	axes[0,2].set_ylim(-.75, .75)
	plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.8)
	plt.suptitle(title_tmp)
	# plt.savefig('./figs/'+gene_code+"-S1_r2.png", dpi=500)
	plt.show()