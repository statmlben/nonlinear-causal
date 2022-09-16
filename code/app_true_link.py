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

mypath = '/home/ben/dataset/GenesToAnalyze'
gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name)) ]
# gene_folders = random.sample(gene_folders, 10)
df = {'gene': [], 'gene-exp': [], 'partial-gene-exp': [], 'AD-outcome': [], 'knn-outcome': []}

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

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
outcome = pd.read_csv(mypath+"/AD_outcome.csv", sep=' ', index_col=0)
# df = outcome.copy()
outcome_tmp = list(outcome['AD.outcome'].to_numpy())
n = len(outcome)

for gene_code in interest_genes:
	folder_tmp = gene_code + 'July20_2SIR'
	## load data
	dir_name = mypath+'/'+folder_tmp
	sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
	gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
	snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)

	n1, n2, p = len(gene_exp), 54162, snp.shape[1]
	LD_Z1 = np.dot(snp.values.T, snp.values)
	cov_ZX1 = np.dot(snp.values.T, gene_exp.values.flatten())
	LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

	SIR = _2SIR(sparse_reg=None, data_in_slice=0.2*n1)
	# SIR.cond_mean = KNeighborsRegressor(n_neighbors=20)
	SIR.cond_mean = IsotonicRegression(increasing='auto',
									out_of_bounds='clip')

	## Stage-1 fit theta
	SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
	## Stage-2 fit beta
	# SIR.fit_beta(LD_Z2, cov_ZY2, n2)
	## AIR to fit link
	# SIR.fit_link(Z1=snp.values, X1=gene_exp.values.flatten())

	## exclude the gene with nan in the dataset
	if sum_stat.isnull().sum().sum() + snp.isnull().sum().sum() + gene_exp.isnull().sum().sum() > 0:
		continue
	if not all(sum_stat.index == snp.columns):
		print('The cols in sum_stat is not corresponding to snp, we rename the sum_stat!')
		sum_stat.index = snp.columns
	df['gene'].extend([gene_code]*n)
	gene_tmp = list(gene_exp.to_numpy().flatten())
	df['gene-exp'].extend(gene_tmp)
	df['AD-outcome'].extend(outcome_tmp)
	partial_exp = np.dot(snp.values, SIR.theta)
	neigh = KNeighborsRegressor(n_neighbors=50)
	neigh.fit(partial_exp[:,np.newaxis], outcome_tmp)
	fit_y_tmp = neigh.predict(partial_exp[:,np.newaxis])
	df['partial-gene-exp'].extend(list(partial_exp))
	df['knn-outcome'].extend(list(fit_y_tmp.flatten()))
df = pd.DataFrame(df)

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
sns.lmplot(x="partial-gene-exp", y="knn-outcome", col="gene", hue="gene", data=df,
           col_wrap=3, ci=None, palette="muted", height=3, robust=True,
           scatter_kws={"s": 7, "alpha": .5})
plt.show()

sns.lmplot(x="partial-gene-exp", y="AD-outcome", col="gene", hue="gene", data=df,
           col_wrap=3, y_jitter=.02, logistic=True,
           scatter_kws={"s": 7, "alpha": .5})
plt.show()

# df['AD-outcome'] = df['AD-outcome'].map({0:'No', 1: 'Yes'})
# sns.violinplot(data=df, x="gene", y="partial-gene-exp", hue="AD-outcome",
#                split=True, inner="quart", linewidth=1,
#                palette={"Yes": "b", "No": ".85"})
# sns.despine(left=True)
# plt.show()