![Pypi](https://badge.fury.io/py/nl-causal.svg)
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<a href="https://bendai.org"><img src="https://img.shields.io/badge/Powered%20by-cuhk%40dAI-purple.svg"/></a>


<!-- [![Youtube](https://img.shields.io/badge/YouTube-Channel-red)]()
![Downloads](https://static.pepy.tech/badge/nl-causal)
![MonthDownloads](https://pepy.tech/badge/nl-causal/month)
[![Conda](https://img.shields.io/conda/vn/conda-forge/???.svg)]() -->
<!-- [![image](https://pepy.tech/badge/leafmap)](https://pepy.tech/project/leafmap) -->
<!-- [![image](https://github.com/giswqs/leafmap/workflows/build/badge.svg)](https://github.com/giswqs/leafmap/actions?query=workflow%3Abuild) -->

# 🧬 nonlinear-causal

<!-- <img style="float: left; max-width: 10%" src="./logo/logo_transparent.png"> -->

![logo](./logo/logo_cover_transparent.png)

**nonlinear-causal** is a Python module for nonlinear causal inference, including **hypothesis testing** and **confidence interval** for causal effect, built on top of two-stage methods. 

- GitHub repo: [https://github.com/statmlben/nonlinear-causal](https://github.com/statmlben/nonlinear-causal)
- Documentation: [https://dnn-inference.readthedocs.io](https://nonlinear-causal.readthedocs.io/en/latest/)
- PyPi: [https://pypi.org/project/nl-causal](https://pypi.org/project/dnn-inference/0.10/)
- Open Source: [MIT license]()
- Paper: [pdf]()


<!-- <script type="text/javascript" charset="utf-8" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML,
https://vincenttam.github.io/javascripts/MathJaxLocal.js"></script> -->

The proposed model is:
<p align="center">
<img src="https://latex.codecogs.com/svg.image?{\centering&space;\color{RoyalBlue}&space;\phi(x)&space;=&space;\mathbf{z}^\prime&space;\boldsymbol{\theta}&space;&plus;&space;w,&space;\quad&space;y&space;=&space;\beta&space;\phi(x)&space;&plus;&space;\mathbf{z}^\prime&space;\boldsymbol{\alpha}&space;&plus;&space;\epsilon}"" width="300">
</p>

<!-- $$
\phi(x) = \mathbf{z}^\prime \mathbf{\theta} + w, \quad y = \beta \phi(x) + \mathbf{z}^\prime \mathbf{\alpha} + \epsilon
$$ -->

- <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\beta" title="\color{Gray} \beta" />: marginal causal effect from X -> Y;
- <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\phi(\cdot)" tilte="\phi"/>: nonlinear causal link;

<!-- ![logo](./logo/model_black.gif) -->


## What We Can Do:
- Estimate <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\theta" title="\color{Gray} \theta" /> and <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\beta" title="\color{Gray} \beta" />.
- Hypothesis testing (HT) and confidence interval (CI) for marginal causal effect $\beta$.
- Estimate nonlinear causal link <img src="https://latex.codecogs.com/svg.image?\color{RoyalBlue}&space;\phi(\cdot)" tilte="\phi"/>.


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
- Notebook 3: [Simulation for HT and CI with categorical IVs](sim_cate.ipynb)
- Notebook 4: [Real application]()


## Contributor
This project was created by [Ben Dai](www.bendai.org), [Chunlin Li](https://github.com/chunlinli) and [Haoran Xue](https://xue-hr.github.io/). 

