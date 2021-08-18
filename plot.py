import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")
# Draw a nested barplot by species and sex
g = sns.catplot(
    data=df, kind="bar",
    x="gene", y="p-value", hue="method", palette="dark", 
	alpha=.5, height=8, legend=False, aspect=3,
)
plt.axhline(.05, ls='--', color='r', alpha=.8)
g.despine(left=True)
g.set_axis_labels("gene", "p-value")
g.legend.set_title("")
plt.legend(loc='upper right')
plt.savefig('result.png', bbox_inches='tight')
# plt.show()
