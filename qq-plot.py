import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.stats import rv_continuous
import statsmodels.api as sm
import random
from scipy.stats import beta, chi2
from nl_causal.base.rv import neg_log_uniform

ci = 0.95

methods = ['2SLS', 'PT-2SLS', '2SIR']
colors = ['darkgoldenrod', 'royalblue', 'purple']
QQ_plot_dis = "neg_log_uniform"
lam_correct = False
if_ci = False

# df = pd.read_csv('./aug24_ben_test.csv')
df = pd.read_csv('./results/Apr12_22_app_test-select.csv')

all_genes = list(set(df['gene']))
num_gene = len(all_genes)

low_bound = [beta.ppf((1-ci)/2, a=i, b=num_gene-i+1) for i in range(1,num_gene+1)]
up_bound = [beta.ppf((1+ci)/2, a=i, b=num_gene-i+1) for i in range(1,num_gene+1)]
ep_points = (np.arange(1,num_gene+1) - 1/2) / num_gene

# plt.style.use('seaborn')
fig, axs = plt.subplots(1,3,figsize=(15, 5))

if QQ_plot_dis == "uniform":
    for i in range(3):
        stats.probplot(df[df['method'] == methods[i]]['p-value'].values, dist="uniform", plot=axs[i], fit=False)

elif QQ_plot_dis == "neg_log_uniform":
    for i in range(3):
        if lam_correct:
            chi2_stat = chi2.ppf(1 - df[df['method'] == methods[i]]['p-value'], 1)
            lam_ = np.median(chi2_stat)/ chi2.ppf(.5,1)
            correct_pvalue = 1 - chi2.cdf( chi2_stat/lam_, 1)
            sm.qqplot(-np.log10(correct_pvalue), dist=neg_log_uniform(), line="45", ax=axs[i])
            axs[i].set_title('QQ-plot %s (lam: %.2f)' %(methods[i], lam_))
            axs[i].get_lines()[0].set_markersize(2.0)
            axs[i].get_lines()[0].set_markerfacecolor(colors[i])
        else:
            sm.qqplot(-np.log10(df[df['method'] == methods[i]]['p-value'].values), dist=neg_log_uniform(), line="45", ax=axs[i])
            axs[i].set_xlim([0,4.5])
            axs[i].set_ylim([0,15])
            axs[i].get_lines()[0].set_markersize(3.0)
            axs[i].get_lines()[0].set_markeredgecolor(colors[i])
            axs[i].get_lines()[0].set_markerfacecolor(colors[i])
            axs[i].get_lines()[0].set_markerfacecoloralt(colors[i])

axs[0].get_lines()[0].set_marker('^')
axs[1].get_lines()[0].set_marker('d')
axs[2].get_lines()[0].set_marker('o')

# axs[0].get_lines()[0].set_markerfacecolor('darkgoldenrod')
# axs[1].get_lines()[0].set_markerfacecolor('royalblue')
# axs[2].get_lines()[0].set_markerfacecolor('purple')

axs[0].get_lines()[0].set(label='2SLS')
axs[1].get_lines()[0].set(label='PT-2SLS')
axs[2].get_lines()[0].set(label='2SIR')

# Add CI
for i in range(3):
    if if_ci:
        axs[i].plot(-np.log10(ep_points), -np.log10(low_bound), 'k--', alpha=0.3)
        axs[i].plot(-np.log10(ep_points), -np.log10(up_bound), 'k--', alpha=0.3)
    axs[i].legend()
plt.show()
