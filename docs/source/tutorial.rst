🧬 Tutorial
-----------

The proposed model is:

.. math::

   \text{(Stage 1)} \quad \phi(x) = \mathbf{z}^\prime \boldsymbol{\theta} + w, \qquad \text{(Stage 2)} \quad y = \beta \phi(x) + \mathbf{z}^\prime \boldsymbol{\alpha} + \epsilon

In this **tutorial**, we illustrate with an example:

.. math::

    \mathbf{z} \to \text{snp}; \quad x \to \text{gene}; \quad y \to \text{outcome}.

Our goal is:

- Estimate `marginal causal effect` :math:`\beta`
- *Hypothesis testing (HT)* and *confidence interval (CI)* for marginal causal effect :math:`\beta`.
- Estimate `nonlinear causal link` :math:`\phi(\cdot)`.


Real data analysis in **ADNI** and **IGAP** dataset
===================================================

Data required
*************

* Please check Section 4 in the manuscript for our data pre-processing.

  - `sum_stat.csv`: summary statistics between *snp* and *outcome*.
  -  `gene_exp.csv`: individual data for *gene expression*, usually obtained by reference panel.
  -  `snp.csv`: individual data for *snp*, usually obtained by reference panel.

.. code:: python

    sum_stat = pd.read_csv(dir_name+"/sum_stat.csv", sep=' ', index_col=0)
    gene_exp = -pd.read_csv(dir_name+"/gene_exp.csv", sep=' ', index_col=0)
    snp = pd.read_csv(dir_name+"/snp.csv", sep=' ', index_col=0)

* Remove *collinear* snps

.. code:: python

    snp, valid_cols = calculate_vif_(snp, thresh=2.5)
    sum_stat = sum_stat.loc[valid_cols]

Note that `thresh=2.5` is suggested in `this ref <doi:10.1007/s11135-017-0584-6>`_ .

* Convert data to fit our package

.. code:: python

    n1, n2, p = len(gene_exp), 54162, snp.shape[1]
    LD_Z1, cov_ZX1 = np.dot(snp.values.T, snp.values), np.dot(snp.values.T, gene_exp.values.flatten())
    LD_Z2, cov_ZY2 = LD_Z1/n1*n2, sum_stat.values.flatten()*n2

Note that we compute LD matrix by the reference panel for **BOTH** stages 1 and 2 models. 


Train `nl_causal` for inference
*******************************

* Define the method

    - `reg_model`: sparse regression method (stage 2: invalid IVs) you want to use. Here we use `LO_IC`, i.e., using `SCAD` to generate candidate models, the using `OLS` for each candidate model and using `BIC` to select the best one.
    - `data_in_slice`: number of data in each slice. It is a tuning parameter for sliced inverse regression (SIR), here we use `data_in_slice=0.2*n1`, alternatively, we specify `slice = 5` for SIR.
    - `Ks`: range of candidate models. We bound it by `int(p/2)-1` to satisfy the identifibility condition for invalid IVs regression. 

.. code:: python

    Ks = range(int(p/2)-1)
    reg_model = L0_IC(fit_intercept=False, alphas=10**np.arange(-3,3,.3), Ks=Ks, max_iter=10000, refit=False, find_best=False)
    SIR = _2SIR(sparse_reg=reg_model, data_in_slice=0.2*n1)


* Fit the sliced inverse regression for the stage 1 model
  * Using the reference panel to fit the stage 1 model.

.. code:: python

  ## Stage-1 fit theta
  SIR.fit_theta(Z1=snp.values, X1=gene_exp.values.flatten())

.. code:: python

    print('\n##### Causal Model of %s #####' %gene_code)
    print('-'*20)
    print('Estimated 2SIR Stage 1 model: \n theta: \n %s; \n link: \n %s' %(SIR.theta, 'SIR.link'))

* Following is the outcome for the gene:

.. code:: 

    ##### Causal Model of ACTN4 #####
    --------------------
    Estimated 2SIR Stage 1 model: 
    theta: 
    [0.6257 -0.0177 0.1491 -0.0347 0.0795 -0.0730 -0.2680 0.0991 0.1083 0.3236
    -0.0579 -0.1197 -0.0384 -0.0614 -0.0104 -0.0327 0.2072 0.2560 -0.1272
    0.2486 -0.2207 -0.0973 0.0885 0.1211 -0.1582 -0.1811 0.1612]; 
    link: 
    SIR.link
    
* Save the stage 1 model for **Pre-train** usage.

.. code:: python

  ## save stage 1 theta:
  np.save(gene_code+'stage1_theta', SIR.theta)
  # This is how to load
  # np.load(gene_code+'stage1_theta.npy')
  ## save stage 1 link function:
  import pickle
  pickle.dump(SIR.link, gene_code+'stage1_link')
  # This is how to load
  # SIR.link = pickle.load(gene_code+'stage1_link')


* Fit the sparse regression for the stage 2 model

.. code:: python

  ## Stage-2 fit beta
  SIR.fit_beta(LD_Z2, cov_ZY2, n2)


- Once `theta` and `beta` are estimated, we are ready to conduct a hypothesis testing for `beta == 0`.

.. code:: python

    SIR.test_effect(n2, LD_Z2, cov_ZY2)
    print('2SIR beta: %.3f' %SIR.beta)
    print('p-value based on 2SIR: %.5f' %SIR.p_value)


- We can construct CI as well, but it is computational expensive.

.. code:: python

    SIR.CI_beta(n1, n2, Z1=snp.values, X1=gene_exp.values.flatten(),
                        B_sample=1000,
                        LD_Z2=LD_Z2, cov_ZY2=cov_ZY2,
                        boot_over='theta',
                        level=CI_level)


Train `nl_causal` for model estimation
**************************************

* Note that there is no need to estimate the nonlinear transformation when conducting hypothesis testing using `nl_causal`, yet if we are still interested in the model, we can further fit the nonlinear link function.

.. code:: python

    SIR.fit_link(Z1=snp.values, X1=gene_exp.values.flatten())


* After estimated the link function, we can use it to provide an estimation of any `gene_exp`

.. code:: python

    IoR = np.arange(0, 1, 1./100)
    link_IoR = SIR.link(X = IoR[:,None])


Then we can summarize the whole estimated models:

.. code:: python

    print('\n##### Causal Model of %s #####' %gene_code)
    print('-'*20)
    print('Estimated 2SIR Stage 1 model: \n theta: \n %s; \n link: \n %s' %(SIR.theta, 'SIR.link'))
    print('-'*20)
    print('Estimated 2SIR Stage 2 model: \n beta: %.3f; \n alpha: \n %s' %(SIR.beta, SIR.alpha))
    print('-'*20)
    print('p-value for causal inference: %.4f' %(SIR.p_value))


* Following is the demo outcome for gene `ACTN4`:

.. code:: 
    
    ##### Causal Model of ACTN4 #####
    --------------------
    Estimated 2SIR Stage 1 model: 
    theta: 
    [0.6257 -0.0177 0.1491 -0.0347 0.0795 -0.0730 -0.2680 0.0991 0.1083 0.3236
    -0.0579 -0.1197 -0.0384 -0.0614 -0.0104 -0.0327 0.2072 0.2560 -0.1272
    0.2486 -0.2207 -0.0973 0.0885 0.1211 -0.1582 -0.1811 0.1612]; 
    link: 
    SIR.link
    --------------------
    Estimated 2SIR Stage 2 model: 
    beta: 0.009; 
    alpha: 
    [0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
    0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
    0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000]
    --------------------
    p-value for causal inference: 0.5701
