import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# df = pd.read_csv("aug24_ben_test.csv")
# df = pd.read_csv("oct04_ben_test_refined_genes.csv")
df = pd.read_csv("Apr9_22_app_test.csv")

num_gen = 17198
level = 0.05 / num_gen
## refine the genes
# find all gene names
gene_set = set(df['gene'])
gene_set = list(gene_set)
# take the gene with at least one siginificant p-value
for gene_tmp in gene_set:
	index_tmp = df[df['gene'] == gene_tmp].index
	if df.loc[index_tmp]['p-value'].min() > level:
		df.drop(index_tmp, inplace=True)

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
