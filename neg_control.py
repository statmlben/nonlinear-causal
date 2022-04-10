import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import rv_continuous
import statsmodels.api as sm
import random
from scipy.stats import beta
ci = 0.95

methods = ['2SLS', 'PT-2SLS', '2SIR']
QQ_plot_dis = "neg_log_uniform"

df = pd.read_csv('./Apr9_22_app_test.csv')
# df = df.drop(df[(df.R2 < 0.05) & (df.method == '2SIR')].index)
# df = df.drop(df[(df.R2 < 0.05) & (df.method == 'Comb-2SIR')].index)
# df = df.drop(df[(df.R2 < 0.1) & (df.method == '2SLS')].index)
# df = df.drop(df[(df.R2 < 0.1) & (df.method == 'PT-2SLS')].index)

all_genes = list(set(df['gene']))

postive_genes = [
                'APOE', 'BCL3', 'CEACAM19', 'CHRNA2', 'HLA-DRB5', 'CLU', 'ABCA7', 'SORL1', 'CR1', 'CD33', 'MS4A', 'TREM2', 'CD2AP',
                'PICALM', 'EPHA1', 'HLA-DRB1', 'INPP5D', 'MEF2C', 'CASS4', 'PTK2B', 'NME8', 'ZCWPW1', 'CELF1', 'FERMT2', 'SLC24A4',
                'RIN3', 'DSG2', 'PLD3', 'UNC5C', 'AKAP9', 'ADAM10', 'PSEN1', 'HFE', 'NOS3', 'PLAU', 'MPO', 'APP', 'GBA', 'SNCA', 
                'SNCB', 'TOMM40',
                'RELB', 'RBFOX1', 'CCDC83', 'ADH6',
                'APOC1', 'APOC1P1', 'BCAM', 'BIN1', 'CBLC', 'CLPTM1', 'CYP27C1', 'MS4A4A', 'MS4A6A', 'MTCH2', 'NKPD1', 'ZNF296'
]

# interest_genes = random.sample(set(all_genes) - set(postive_genes), 5000)
interest_genes = list(set(all_genes) - set(postive_genes))

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

plt.show()
