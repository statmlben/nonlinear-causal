![Pypi](https://badge.fury.io/py/nonlinear-causal.svg)
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/)
[![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

- GitHub repo: [https://github.com/nl-causal/nonlinear-causal](https://github.com/nl-causal/nonlinear-causal)
- PyPi: [https://pypi.org/project/nonlinear-causal/](https://pypi.org/project/nonlinear-causal/)
- Open Source: [MIT license](https://opensource.org/licenses/MIT)
- Paper: [arXiv:2209.08889](https://arxiv.org/pdf/2209.08889.pdf)
- **Documentation**: [https://nonlinear-causal.readthedocs.io](https://nonlinear-causal.readthedocs.io/en/latest/index.html)


<!-- <script type="text/javascript" charset="utf-8" 
src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML,
https://vincenttam.github.io/javascripts/MathJaxLocal.js"></script> -->

The proposed model is:
![model](.figs/../logo/nl_causal.png)

<!-- <p align="center">
<img src="https://latex.codecogs.com/svg.image?{\centering&space;\color{RedOrange}&space;\phi(x)&space;=&space;\mathbf{z}^\prime&space;\boldsymbol{\theta}&space;&plus;&space;w,&space;\quad&space;y&space;=&space;\beta&space;\phi(x)&space;&plus;&space;\mathbf{z}^\prime&space;\boldsymbol{\alpha}&space;&plus;&space;\epsilon}"" width="350">
</p> -->

$$
\phi(x) = \mathbf{z}^\prime \mathbf{\theta} + w, \quad y = \beta \phi(x) + \mathbf{z}^\prime \mathbf{\alpha} + \epsilon
$$

- $\beta$: marginal causal effect from X -> Y;
- $\phi(\cdot)$: nonlinear causal link;

<!-- ![logo](./logo/model_black.gif) -->


## What We Can Do:
- Estimate $\theta$ and $\beta$.
- Hypothesis testing (HT) and confidence interval (CI) for marginal causal effect $\beta$.
- Estimate nonlinear causal link $\phi(\cdot)$.


## Installation

Install `nonlinear-causal` using ``pip``

```bash
pip install nonlinear-causal
```

Install the latest version in Github:
```bash
pip install git+https://github.com/nl-causal/nonlinear-causal
```

## Examples and notebooks

- [User Guide](user_guide.md)

- [Simulation for HT and CI with standard setup](sim_main.ipynb)
- [Simulation for HT and CI with invalid IVs](sim_invalid_IVS.ipynb)
- [Simulation for HT and CI with categorical IVs](sim_cate.ipynb)
- [Real application](app_test.ipynb)
<!-- - [Pipeline for plink data](user_guide.md) -->

## Reference

If you use this code please star 🌟 the repository and cite the following paper:

- Dai, B., Li, C., Xue, H., Pan, W., & Shen, X. (2022). Inference of nonlinear causal effects with GWAS summary data. *arXiv preprint* arXiv:2209.08889.

```latex
@article{dai2022inference,
  title={Inference of nonlinear causal effects with GWAS summary data},
  author={Dai, Ben and Li, Chunlin and Xue, Haoran and Pan, Wei and Shen, Xiaotong},
  journal={arXiv preprint arXiv:2209.08889},
  year={2022}
}
```