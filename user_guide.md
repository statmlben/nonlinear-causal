# 🧬 nonlinear-causal: **User Guide**

The proposed model is:
<p align="center">
<img src="https://latex.codecogs.com/svg.image?{\centering&space;\color{RoyalBlue}&space;\phi(x)&space;=&space;\mathbf{z}^\prime&space;\boldsymbol{\theta}&space;&plus;&space;w,&space;\quad&space;y&space;=&space;\beta&space;\phi(x)&space;&plus;&space;\mathbf{z}^\prime&space;\boldsymbol{\alpha}&space;&plus;&space;\epsilon}"" width="350">
</p>

- <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\beta" title="\color{Gray} \beta" />: marginal causal effect from X -> Y;
- <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\phi(\cdot)" tilte="\phi"/>: nonlinear causal link;

<!-- ![logo](./logo/model_black.gif) -->


## What We Can Do:
- Estimate <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\theta" title="\color{Gray} \theta" /> and <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\beta" title="\color{Gray} \beta" />.
- Hypothesis testing (HT) and confidence interval (CI) for marginal causal effect $\beta$.
- Estimate nonlinear causal link <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\phi(\cdot)" tilte="\phi"/>.

---
## Real data analysis in **ADNI** and **IGAP** dataset

### Data required

* Please check Section 4 in the manuscript for our data pre-processing.

  - `sum_stat.csv`: summary statistics between *snp* and *outcome*.
  -  `gene_exp.csv`: individual data for *gene expression*, usually obtained by reference panel.
  -  `snp.csv`: individual data for *snp*s, usually obtained by reference panel.

```python
sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)
```
* Remove *collinear* snps
<!-- doi:10.1007/s11135-017-0584-6 -->
```python
snp, valid_cols = calculate_vif_(snp, thresh=2.5)
sum_stat = sum_stat.loc[valid_cols]
```
Note that `thresh=2.5` is suggested in [this ref](doi:10.1007/s11135-017-0584-6).

* Convert data to fit our package

```python
n1, n2, p = len(gene_exp), 54162, snp.shape[1]
LD_Z1, cov_ZX1 = np.dot(snp.values.T, snp.values), np.dot(snp.values.T, gene_exp.values.flatten())
LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2
```
Note that we compute LD matrix by the reference panel for **BOTH** stages 1 and 2 models. 

---

### Train `nl_causal`

* Define the method

    - `reg_model`: sparse regression method (stage 2: invalid IVs) you want to use. Here we use `LO_IC`, i.e., using `SCAD` to generate candidate models, the using `OLS` for each candidate model and using `BIC` to select the best one.
    - `data_in_slice`: number of data in each slice. It is a tuning parameter for sliced inverse regression (SIR), here we use `data_in_slice=0.2*n1`, alternatively, we specify `slice = 5` for SIR.
    - `Ks`: range of candidate models. We bound it by `int(p/2)-1` to satisfy the identifibility condition for invalid IVs regression. 

```python
Ks = range(int(p/2)-1)
reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3), Ks=Ks, max_iter=10000, refit=False, find_best=False)
SIR = _2SIR(sparse_reg=reg_model, data_in_slice=0.2*n1)
```

* Fit the sliced inverse regression for the stage 1 model
  * Using the reference panel to fit the stage 1 model.
```python
## Stage-1 fit theta
SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())
```

* Fit the sparse regression for the stage 2 model

```python
## Stage-2 fit beta
SIR.fit_beta(LD_Z2, cov_ZY2, n2)
```

- Once `theta` and `beta` are estimated, we are ready to conduct a hypothesis testing for `beta == 0`.
```python
SIR.test_effect(n2, LD_Z2, cov_ZY2)
print('2SIR beta: %.3f' %SIR.beta)
print('p-value based on 2SIR: %.5f' %SIR.p_value)
```

- We can construct CI as well, but it is computational expensive.
```python
SIR.CI_beta(n1, n2, Z1=snp.values, X1=gene_exp.values.flatten(),
                    B_sample=1000,
                    LD_Z2=LD_Z2, cov_ZY2=cov_ZY2,
                    boot_over='theta',
                    level=CI_level)
```