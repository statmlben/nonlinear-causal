import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import rv_continuous

methods = ['2SLS', 'PT-2SLS', '2SIR']
QQ_plot_dis = "uniform"

df = pd.read_csv('./aug24_ben_test.csv')

interest_genes = [
				'APOE',
				'BCL3',
				'CEACAM19',
				'CHRNA2',
				'HLA-DRB5',
				'TOMM40',
				'APOC1',
				'APOC1P1',
				'BCAM',
				'BIN1',
				'CBLC',
				'CLPTM1',
				'CYP27C1',
				'MS4A4A',
				'MS4A6A',
				'MTCH2',
				'NKPD1',
				'ZNF296'
				]

df = df[~df['gene'].isin(interest_genes)]

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
        stats.probplot(-np.log10(df[df['method'] == '2SLS']['p-value'].values), dist=NLU_rv, plot=axs[i], fit=False)

axs[0].get_lines()[0].set_marker('o')
axs[1].get_lines()[0].set_marker('d')
axs[2].get_lines()[0].set_marker('*')

axs[0].get_lines()[0].set_markersize(3.0)
axs[1].get_lines()[0].set_markersize(3.0)
axs[2].get_lines()[0].set_markersize(3.0)

axs[0].get_lines()[0].set_markerfacecolor('darkgoldenrod')
axs[1].get_lines()[0].set_markerfacecolor('royalblue')
axs[2].get_lines()[0].set_markerfacecolor('purple')

axs[0].get_lines()[0].set(label='2SLS')
axs[1].get_lines()[0].set(label='PT-2SLS')
axs[2].get_lines()[0].set(label='2SIR')

# Add on y=x line
for i in range(3):
    if QQ_plot_dis == 'uniform':
        y_max = 1
    else:
        y_max = (-np.log10(df['p-value'].values)).max()
    axs[i].plot([0, y_max], [0, y_max], c='r', label='45 degree line')
    axs[i].legend()

plt.show()
