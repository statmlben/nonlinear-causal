.. nonlinear-causal documentation master file, created by
   sphinx-quickstart on Sat Aug  7 21:39:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

🧬 nonlinear-causal
===================

.. -*- mode: rst -*-

|dAI|_ |PyPi|_ |MIT|_ |Python3|_ |downloads|_ |downloads_month|_

.. |dAI| image:: https://img.shields.io/badge/Powered%20by-cuhk%40dAI-purple.svg
.. _dAI: https://www.bendai.org

.. |PyPi| image:: https://badge.fury.io/py/nonlinear-causal.svg
.. _PyPi: https://badge.fury.io/py/nonlinear-causal

.. |MIT| image:: https://img.shields.io/pypi/l/nonlinear-causal.svg
.. _MIT: https://opensource.org/licenses/MIT

.. |Python3| image:: https://img.shields.io/badge/python-3-green.svg
.. _Python3: www.python.org

.. |downloads| image:: https://pepy.tech/badge/nonlinear-causal
.. _downloads: https://pepy.tech/project/nonlinear-causal

.. |downloads_month| image:: https://pepy.tech/badge/nonlinear-causal/month
.. _downloads_month: https://pepy.tech/project/nonlinear-causal

.. image:: ./logo/logo_cover.png
  :width: 1030

**nonlinear-causal** is a Python module for nonlinear causal inference, including **hypothesis testing** and **confidence interval** for causal effect, built on top of two-stage methods. 

- GitHub repo: `https://github.com/statmlben/nonlinear-causal <https://github.com/statmlben/nonlinear-causal>`_
- Documentation: `https://nonlinear-causal.readthedocs.io <https://nonlinear-causal.readthedocs.io/en/latest/>`_
- PyPi: `https://pypi.org/project/nl-causal <https://pypi.org/project/nonlinear-causal>`_
- Open Source: `MIT license <https://opensource.org/licenses/MIT>`_
- Paper: `pdf <www.bendai.org>`_

The proposed model is:

.. math::

   \text{(Stage 1)} \quad \phi(x) = \mathbf{z}^\prime \boldsymbol{\theta} + w, \qquad \text{(Stage 2)} \quad y = \beta \phi(x) + \mathbf{z}^\prime \boldsymbol{\alpha} + \epsilon


🎯 What We Can Do
-----------------

- Estimate `marginal causal effect` :math:`\beta`
- *Hypothesis testing (HT)* and *confidence interval (CI)* for marginal causal effect :math:`\beta`.
- Estimate `nonlinear causal link` :math:`\phi(\cdot)`.

.. The proposed model is:
.. <p align="center">
.. <img src="https://latex.codecogs.com/svg.image?{\centering \color{RoyalBlue} \phi(x) = \mathbf{z}^\prime \boldsymbol{\theta} &plus; w, \quad y = \beta \phi(x) &plus; \mathbf{z}^\prime \boldsymbol{\alpha} &plus; \epsilon}"" width="350">
.. </p>

📒 Contents
-----------

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   api

Indices and tables
==================

* :ref:`genindex`
.. * :ref:`modindex`
* :ref:`search`
