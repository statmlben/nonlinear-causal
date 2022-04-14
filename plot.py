import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df = pd.read_csv("aug24_ben_test.csv")
# df = pd.read_csv("oct04_ben_test_refined_genes.csv")
# df = pd.read_csv("Apr12_22_app_test+select.csv")
df = pd.read_csv("Apr12_22_app_test-select.csv")

gene_set = list(set(df['gene']))
num_gen = len(gene_set)
level = 0.05 / num_gen
## refine the genes
# find all gene names

# take the gene with at least one siginificant p-value
min_p = df.groupby('gene')['p-value'].min()
gene_set = list(min_p[min_p < level].index)

df = df[df['gene'].isin(gene_set)]
df['log-p-value'] = - np.log10( df['p-value'] )

gene_set = set(df['gene'])
gene_set = list(gene_set)
gene_set.sort()
## plot for the final results
sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
plt.rcParams["figure.figsize"] = (16,6)

g = sns.catplot(
    data=df, kind="bar", order=gene_set,
    x="gene", y="log-p-value", hue="method", palette="dark",
	alpha=.5, height=8, legend=False, aspect=3,
)
plt.axhline(-np.log10(level), ls='--', color='r', alpha=.8)
g.despine(left=True)
g.set_axis_labels("gene", "-log(p-value)")
plt.legend(loc='upper right')
# plt.savefig('result.png', bbox_inches='tight')
# plt.savefig('./figs/'+'oct04_ben_app_test.png', dpi=500)
plt.show()
