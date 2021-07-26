<!-- ![Pypi](https://badge.fury.io/py/dnn-locate.svg) -->
[![Python](https://img.shields.io/badge/python-3-blue.svg)](https://www.python.org/)
<!-- ![License](https://img.shields.io/pypi/l/keras-bert.svg) -->
<!-- ![Downloads](https://static.pepy.tech/badge/dnn-locate)
![MonthDownloads](https://pepy.tech/badge/dnn-locate/month) -->


# nonlinear-causal

**nonlinear-causal** is a Python module for nonlinear causal inference built on top of Two-stage methods. The proposed model is:

![logo](./logo/model_black.gif)

## What we can do:
- Estimate `\theta`, `\beta`.
- Hypothesis testing (HT) and confidence interval (CI) for marginal causal effect `\beta`.
- Estimate nonlinear causal link `\phi`.


## Installation

### Dependencies

`nonlinear-causal` requires:

| | | | | | |
|-|-|-|-|-|-|
| Python>=3.8 | numpy | Pandas | sklearn | SciPy | sliced |

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
- Notebook 2: [Simulation for HT and CI with invalid IVs](sim_main.ipynb)
- Notebook 3: [Real application]()














<!-- This project was created by [Ben Dai](www.bendai.org), [Chunlin Li](https://github.com/chunlinli) and [Haoran Xue](https://xue-hr.github.io/).  If there is any problem and suggestion please contact me via <bdai@umn.edu>. -->

