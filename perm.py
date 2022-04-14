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
from nl_causal.base.preprocessing import calculate_vif_
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_continuous
from scipy.stats import beta

mypath = '/home/ben/dataset/GenesToAnalyze'
gene_folders = [name for name in listdir(mypath) if isdir(join(mypath, name))]
# gene_folders = random.sample(gene_folders, 100)

np.random.seed(0)
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
df = {'gene': [], 'p-value': [], 'beta': [], 'method': [], 'R2': []}
# NMNAT3July20_2SIR

## generate a negative control dataset
y = pd.read_csv(mypath+'/AD_outcome.csv', sep=' ', index_col=0)
y = np.random.permutation(y['AD.outcome'].values)
n = len(y)
for folder_tmp in gene_folders:
# for folder_tmp in ['APOEJuly20_2SIR', 'TOMM40July20_2SIR']:
# for folder_tmp in ['TOMM40July20_2SIR', 'RBFOX1July20_2SIR']:
    if 'July20_2SIR' not in folder_tmp:
        continue
    gene_code = folder_tmp.replace('July20_2SIR', '')

    ## load data
    dir_name = mypath+'/'+folder_tmp
    # sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
    gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
    snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)
    ## remove the collinear features
    if snp.isnull().sum().sum():
        continue

    if (snp.shape[1] > 10):
        continue

    snp, valid_cols = calculate_vif_(snp, thresh=2.5)

    print('\n##### Causal inference of %s (dim: %d) #####' %(gene_code, len(valid_cols)))
    # sum_stat = sum_stat.loc[valid_cols]
    Z1, Z2, X1, X2, y1, y2 = train_test_split(snp, gene_exp, y, test_size=.2, random_state=42)
    ## n1 and n2 is pre-given 
    n1, n2, p = len(Z1), len(Z2), snp.shape[1]
    LD_Z = np.dot(snp.T, snp)
    LD_Z1, cov_ZX1 = np.dot(Z1.T, Z1), np.dot(Z1.T, X1.values.flatten())
    LD_Z2, cov_ZY2 = np.dot(Z2.T, Z2), np.dot(Z2.T, y2)

    Ks = range(int(p/2)-1)
    ## 2SLS
    reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3),
                    Ks=Ks, max_iter=50000, refit=False, find_best=False)
    LS = _2SLS(sparse_reg=None)
    ## Stage-1 fit theta
    LS.fit_theta(LD_Z1, cov_ZX1)
    ## Stage-2 fit beta
    LS.fit_beta(LD_Z2, cov_ZY2, n2)
    ## produce p_value and CI for beta
    LS.test_effect(n2, LD_Z2, cov_ZY2)
    LS_R2 = np.var( snp.dot(LS.theta)*LS.theta_norm ) / np.var(np.array(gene_exp))

    # if LS_R2 < 0.05:
    #     continue

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
                    Ks=Ks, max_iter=50000, refit=False, find_best=False)
    PT_X1 = power_transform(gene_exp.values.reshape(-1,1), method='yeo-johnson').flatten()
    PT_X1 = PT_X1 - PT_X1.mean()
    PT_cor_ZX1 = np.dot(snp.values.T, PT_X1)
    PT_LS = _2SLS(sparse_reg=None)
    ## Stage-1 fit theta
    PT_LS.fit_theta(LD_Z1, PT_cor_ZX1)
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
                    Ks=Ks, max_iter=50000, refit=False, find_best=False)
    SIR = _2SIR(sparse_reg=None, data_in_slice=0.1*n1)
    ## Stage-1 fit theta
    SIR.fit_theta(Z1=Z1.values, X1=X1.values.flatten())
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

    # ## Comb-2SIR
    # data_in_slice_lst = [.1*n1, .2*n1, .3*n1, .4*n1, .5*n1]
    # comb_pvalue, comb_beta, comb_eigenvalue = [], [], []
    # for data_in_slice_tmp in data_in_slice_lst:
    #     reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3),
    #                     Ks=Ks, max_iter=50000, refit=False, find_best=False)
    #     SIR = _2SIR(sparse_reg=None, data_in_slice=data_in_slice_tmp)
    #     ## Stage-1 fit theta
    #     SIR.fit_theta(Z1=Z1.values, X1=X1.values.flatten())
    #     ## Stage-2 fit beta
    #     SIR.fit_beta(LD_Z2, cov_ZY2, n2)
    #     ## generate CI for beta
    #     SIR.test_effect(n2, LD_Z2, cov_ZY2)
    #     comb_beta.append(SIR.beta)
    #     comb_pvalue.append(SIR.p_value)
    #     comb_eigenvalue.append(SIR.sir.eigenvalues_[0])
    # comb_T = np.tan((0.5 - np.array(comb_pvalue))*np.pi).mean()
    # correct_pvalue = min( max(.5 - np.arctan(comb_T)/np.pi, np.finfo(np.float64).eps), 1.0)
    # # correct_pvalue = min(len(data_in_slice_lst)*np.min(comb_pvalue), 1.0)
    # print('Comb-2SIR eigenvalues: %.3f' %np.mean(comb_eigenvalue))
    # print('Comb-2SIR beta: %.3f' %np.mean(comb_beta))
    # print('p-value based on Comb-2SIR: %.5f' %correct_pvalue)

    # ## save the record
    # df['gene'].append(gene_code)
    # df['method'].append('Comb-2SIR')
    # df['p-value'].append(correct_pvalue)
    # df['beta'].append(np.mean(comb_beta))
    # df['R2'].append(np.mean(comb_eigenvalue))

df = pd.DataFrame.from_dict(df)
# df.to_csv('Apr10_22_app_test.csv', index=False)


ci = 0.95

methods = ['2SLS', 'PT-2SLS', '2SIR']
QQ_plot_dis = "neg_log_uniform"

interest_genes = list(set(df['gene']))

num_gene = len(interest_genes)
low_bound = [beta.ppf((1-ci)/2, a=i, b=num_gene-i+1) for i in range(1,num_gene+1)]
up_bound = [beta.ppf((1+ci)/2, a=i, b=num_gene-i+1) for i in range(1,num_gene+1)]
ep_points = (np.arange(1,num_gene+1) - 1/2) / num_gene

df = df[df['gene'].isin(interest_genes)]

plt.style.use('seaborn')
fig, axs = plt.subplots(1,3)

if QQ_plot_dis == "uniform":
    for i in range(3):
        stats.probplot(df[df['method'] == methods[i]]['p-value'].values, dist="uniform", plot=axs[i], fit=False)

elif QQ_plot_dis == "neg_log_uniform":
    ## Define log-uniform distribution
    class neg_log_uniform(rv_continuous):
        "negative log uniform distribution"
        def _cdf(self, x):
            return 1. - 10**(-x)
    NLU_rv = neg_log_uniform()

    for i in range(3):
        sm.qqplot(-np.log10(df[df['method'] == methods[i]]['p-value'].values), dist=NLU_rv, line="45", ax=axs[i])

axs[0].get_lines()[0].set_marker('o')
axs[1].get_lines()[0].set_marker('d')
axs[2].get_lines()[0].set_marker('*')

axs[0].get_lines()[0].set_markersize(4.0)
axs[1].get_lines()[0].set_markersize(4.0)
axs[2].get_lines()[0].set_markersize(4.0)

axs[0].get_lines()[0].set_markerfacecolor('darkgoldenrod')
axs[1].get_lines()[0].set_markerfacecolor('royalblue')
axs[2].get_lines()[0].set_markerfacecolor('purple')

axs[0].get_lines()[0].set(label='2SLS')
axs[1].get_lines()[0].set(label='PT-2SLS')
axs[2].get_lines()[0].set(label='2SIR')

# Add CI
for i in range(3):
    axs[i].plot(-np.log10(ep_points), -np.log10(low_bound), 'k--', alpha=0.3)
    axs[i].plot(-np.log10(ep_points), -np.log10(up_bound), 'k--', alpha=0.3)
    axs[i].legend()

plt.title('QQ-plot for permutation dataset with num_gen: %d' %num_gene)
plt.show()
