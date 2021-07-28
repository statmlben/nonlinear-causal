<!-- ![Pypi](https://badge.fury.io/py/dnn-locate.svg) -->
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/)
<!-- ![License](https://img.shields.io/pypi/l/keras-bert.svg) -->
<!-- ![Downloads](https://static.pepy.tech/badge/dnn-locate)
![MonthDownloads](https://pepy.tech/badge/dnn-locate/month) -->


# 🧬 nonlinear-causal

<img style="float: left; max-width: 10%" src="./logo/logo_transparent.png">

<!-- ![logo](./logo/logo_transparent.png) -->

**nonlinear-causal** is a Python module for nonlinear causal inference, including **hypothesis testing** and **confidence interval** for causal effect, built on top of two-stage methods. 

- GitHub repo: [**nonlinear-causal**](https://github.com/statmlben/nonlinear-causal)
- Documentation: [**docs**]()
- PyPi: [**nl-causal**]()
- Open Source: [**MIT license**]()
- Paper: [**pdf** in ???]()
- About: [**contributor**]()


<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=default'></script>

The proposed model is:

$$
\phi(x) = \mathbf{z}^\prime \mathbf{\theta} + w, \quad y = \beta \phi(x) + \mathbf{z}^\prime \mathbf{\alpha} + \epsilon
$$

- $\beta$: marginal causal effect from $x$ to $y$;
- $\phi(\cdot)$: nonlinear causal link;

<!-- ![logo](./logo/model_black.gif) -->

## What We Can Do:
- Estimate $\theta$, $\beta$.
- Hypothesis testing (HT) and confidence interval (CI) for marginal causal effect $\beta$.
- Estimate nonlinear causal link $\phi(\cdot)$.


## Installation

### Dependencies

`nonlinear-causal` requires:

| | | | | | |
|-|-|-|-|-|-|
| Python>=3.8 | numpy | pandas | sklearn | scipy | sliced |

### User installation

Install `nonlinear-causal` using ``pip``

```bash
pip install nl_causal
pip install git+https://github.com/statmlben/nonlinear-causal.git
```
### Source code

You can check the latest sources with the command::

```bash
git clone https://github.com/statmlben/nonlinear-causal.git
```

## Examples and notebooks

- Notebook 1: [Simulation for HT and CI with standard setup](sim_main.ipynb)
- Notebook 2: [Simulation for HT and CI with invalid IVs](sim_invalid_IVS.ipynb)
- Notebook 3: [Simulation for HT and CI with categorical IVs](sim_main.ipynb)
- Notebook 4: [Real application]()














<!-- This project was created by [Ben Dai](www.bendai.org), [Chunlin Li](https://github.com/chunlinli) and [Haoran Xue](https://xue-hr.github.io/).  If there is any problem and suggestion please contact me via <bdai@umn.edu>. -->

