.. factor_analyzer documentation master file, created by
   sphinx-quickstart on Wed Dec  6 19:08:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FactorAnalyzer documentation!
==============================================

This is Python module to perform exploratory factor analysis, with optional varimax and promax rotations. Estimation can be performed using a minimum residual (minres) solution, or maximum likelihood estimation (MLE).

Portions of this code are ported from the excellent R library psych.

Description
==================

Exploratory factor analysis (EFA) is a statistical technique used to identify latent relationships among sets of observed variables in a dataset. In particular, EFA seeks to model a large set of observed variables as linear combinations of some smaller set of unobserved, latent factors.

The matrix of weights, or factor loadings, generated from an EFA model describes the underlying relationships between each variable and the latent factors. Typically, a number of factors (K) is selected such that is substantially smaller than the number of variables. The factor analysis model can be estimated using a variety of standard estimation methods, including but not limited to OLS, minres, or MLE.

Factor loadings are similar to standardized regression coefficients, and variables with higher loadings on a particular factor can be interpreted as explaining a larger proportion of the variation in that factor. In many cases, factor loading matrices are rotated after the factor analysis model is estimated in order to produce a simpler, more interpretable structure to identify which variables are loading on a particular factor.

Two common types of rotations are:

* The **varimax** rotation, which rotates the factor loading matrix so as to maximize the sum of the variance of squared loadings, while preserving the orthogonality of the loading matrix.

* The **promax** rotation, a method for oblique rotation, which builds upon the varimax rotation, but ultimately allows factors to become correlated.

This package includes a stand-alone Python module with a single ``FactorAnalyzer()`` class. The class includes an ``analyze()`` method that allows users to perform factor analysis using either minres or MLE, with optional promax or varimax rotations on the factor loading matrices.

Requirements
==================
- Python 3.4 or higher
- ``numpy``
- ``pandas``
- ``scikit-learn``
- ``scipy``

Installation
==================
You can install this package via pip:
``$ pip install factor_analyzer``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
