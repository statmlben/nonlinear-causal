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

vif_thresh = 2.5

mypath = '/home/ben/dataset/GenesToAnalyze'
gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name))]
# gene_folders = random.sample(gene_folders, 10)

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
df = {'gene': [], 'p-value': [], 'beta': [], 'method': [], 'R2': []}

for folder_tmp in gene_folders:
    if 'July20_2SIR' not in folder_tmp:
        continue
    gene_code = folder_tmp.replace('July20_2SIR', '')
    print('\n##### Pre-P for %s #####' %gene_code)
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
    
    ### remove low variance features
    # snp = variance_threshold_selector(snp, threshold=0.1)
    # sum_stat = sum_stat.loc[snp.columns]

    # remove the collinear features
    # doi:10.1007/s11135-017-0584-6
    snp, valid_cols = calculate_vif_(snp, thresh=vif_thresh)
    sum_stat = sum_stat.loc[valid_cols]
    snp.to_csv(dir_name+"/snp_vif.csv", sep=' ')
    sum_stat.to_csv(dir_name+"/sum_stat_vif.csv", sep=' ')


