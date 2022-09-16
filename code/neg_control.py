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
QQ_plot_dis = "neg_log_uniform"
lam_correct = True

# df = pd.read_csv('./aug24_ben_test.csv')
df = pd.read_csv('./Apr12_22_app_test-select.csv')

g1 = pd.read_csv('./dataset/chromosome1.csv', sep='\t', index_col=0)['symbol']
g14 = pd.read_csv('./dataset/chromosome14.csv', sep='\t', index_col=0)['symbol']
g19 = pd.read_csv('./dataset/chromosome19.csv', sep='\t', index_col=0)['symbol']
g21 = pd.read_csv('./dataset/chromosome21.csv', sep='\t', index_col=0)['symbol']

all_genes = list(set(df['gene']))

pt_genes = list(list(g1) + list(g14) + list(g19) + list(g21))

postive_genes = [
                'APOE', 'BCL3', 'CEACAM19', 'CHRNA2', 'HLA-DRB5', 'CLU', 'ABCA7', 'SORL1', 'CR1', 'CD33', 'MS4A', 'TREM2', 'CD2AP',
                'PICALM', 'EPHA1', 'HLA-DRB1', 'INPP5D', 'MEF2C', 'CASS4', 'PTK2B', 'NME8', 'ZCWPW1', 'CELF1', 'FERMT2', 'SLC24A4',
                'RIN3', 'DSG2', 'PLD3', 'UNC5C', 'AKAP9', 'ADAM10', 'PSEN1', 'HFE', 'NOS3', 'PLAU', 'MPO', 'APP', 'GBA', 'SNCA', 
                'SNCB', 'TOMM40', 'ACP2', 'MADD', 'MYBPC3', 'NR1H3', 'NUP160', 'PSMC3', 'SPI1', 
                'RELB', 'RBFOX1', 'CCDC83', 'ADH6', 'AP4M1', 'MCM7', 'PILRA', 'PILRB', 'PVRIG', 'STAG3', 'RABEP1', 'TP53INP1', 'APBB3',
                'ZYX', 'MS4A6A', 'MS4A6E', 'GATS', 'ZKSCAN1', 'CCNE2', 'INTS8', 'TP53INP1', 'ABCA7', 'CNN2', 'CIRBP', 'NUP88',
                'RABEP1', 'AURKA', 'NFYA', 'CELF1', 'CLCN1', 'EPHA1', 'FAM131B', 'TAS2R41', 'TAS2R60', 'AP4E1', 'TRPM7', 'GPX4',
                'CHRNE', 'CD55', 'TREML2', 'CCDC83',
                'APOC1', 'APOC1P1', 'BCAM', 'BIN1', 'CBLC', 'CLPTM1', 'CYP27C1', 'MS4A4A', 'MS4A6A', 'MTCH2', 'NKPD1', 'ZNF296', 'DMWD', 
                'FBXO46', 'HLA-DQA1']

weak_genes=[]
weak_genes = list(df[(df['method']=='Comb-2SIR') & (df['R2'] <= 0.05)]['gene'])

# interest_genes = random.sample(set(all_genes) - set(postive_genes) - set(weak_genes), 3000)
interest_genes = list(set(set(all_genes) - set(postive_genes) - set(pt_genes) - set(weak_genes)))

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
    for i in range(3):
        if lam_correct:
            chi2_stat = chi2.ppf(1 - df[df['method'] == methods[i]]['p-value'], 1)
            lam_ = np.median(chi2_stat)/ chi2.ppf(.5,1)
            correct_pvalue = 1 - chi2.cdf( chi2_stat/lam_, 1)
            sm.qqplot(-np.log10(correct_pvalue), dist=neg_log_uniform(), line="45", ax=axs[i])
            axs[i].set_title('QQ-plot %s (lam: %.2f)' %(methods[i], lam_))
            axs[i].get_lines()[0].set_markersize(4.0)
        else:
            sm.qqplot(-np.log10(df[df['method'] == methods[i]]['p-value'].values), dist=neg_log_uniform(), line="45", ax=axs[i])

axs[0].get_lines()[0].set_marker('^')
axs[1].get_lines()[0].set_marker('d')
axs[2].get_lines()[0].set_marker('o')

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
