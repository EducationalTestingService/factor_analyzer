FactorAnalyzer
--------------

.. image:: https://circleci.com/gh/EducationalTestingService/factor_analyzer/tree/master.svg?style=shield
    :alt: Build status
    :target: https://circleci.com/gh/EducationalTestingService/factor_analyzer

.. image:: https://coveralls.io/repos/github/EducationalTestingService/factor_analyzer/badge.svg?branch=master
    :target: https://coveralls.io/github/EducationalTestingService/factor_analyzer?branch=master

.. image:: https://anaconda.org/desilinguist/factor_analyzer/badges/installer/conda.svg
    :target: https://anaconda.org/desilinguist/factor_analyzer/


This is Python module to perform exploratory factor analysis, with
optional varimax and promax rotations. Estimation can be performed using
a minimum residual (minres) solution, or maximum likelihood estimation
(MLE).

Portions of this code are ported from the excellent R library ``psych``.

Please see the `official documentation <http://factor-analyzer.readthedocs.io/en/latest/index.html>`__ for additional details.


Description
-----------

Exploratory factor analysis (EFA) is a statistical technique used to
identify latent relationships among sets of observed variables in a
dataset. In particular, EFA seeks to model a large set of observed
variables as linear combinations of some smaller set of unobserved,
latent factors.

The matrix of weights, or factor loadings, generated from an EFA model
describes the underlying relationships between each variable and the
latent factors. Typically, a number of factors (K) is selected such that
is substantially smaller than the number of variables. The factor
analysis model can be estimated using a variety of standard estimation
methods, including but not limited to OLS, minres, or MLE.

Factor loadings are similar to standardized regression coefficients, and
variables with higher loadings on a particular factor can be interpreted
as explaining a larger proportion of the variation in that factor. In
many cases, factor loading matrices are rotated after the factor
analysis model is estimated in order to produce a simpler, more
interpretable structure to identify which variables are loading on a
particular factor.

Two common types of rotations are:

1. The **varimax** rotation, which rotates the factor loading matrix so
   as to maximize the sum of the variance of squared loadings, while
   preserving the orthogonality of the loading matrix.

2. The **promax** rotation, a method for oblique rotation, which builds
   upon the varimax rotation, but ultimately allows factors to become
   correlated.

This package includes a stand-alone Python module with a single
``FactorAnalyzer()`` class. The class includes an ``analyze()`` method
that allows users to perform factor analysis using either minres or MLE,
with optional promax or varimax rotations on the factor loading
matrices.

Example
-------

.. code:: python

    In [1]: import pandas as pd

    In [2]: from factor_analyzer import FactorAnalyzer

    In [3]: df_features = pd.read_csv('test02.csv')

    In [4]: fa = FactorAnalyzer()

    In [5]: fa.analyze(df_features, 3, rotation=None)

    In [6]: fa.loadings
    Out[6]: 
               Factor1   Factor2   Factor3
    sex      -0.129912 -0.163982  0.738235
    zygosity  0.038996 -0.046584  0.011503
    moed      0.348741 -0.614523 -0.072557
    faed      0.453180 -0.719267 -0.075465
    faminc    0.366888 -0.443773 -0.017371
    english   0.741414  0.150082  0.299775
    math      0.741675  0.161230 -0.207445
    socsci    0.829102  0.205194  0.049308
    natsci    0.760418  0.237687 -0.120686
    vocab     0.815334  0.124947  0.176397

    In [7]: fa.get_uniqueness()
    Out[7]: 
              Uniqueness
    sex         0.411242
    zygosity    0.996177
    moed        0.495476
    faed        0.271588
    faminc      0.668157
    english     0.337916
    math        0.380890
    socsci      0.268054
    natsci      0.350704
    vocab       0.288503

    In [8]: fa.get_factor_variance()
    Out[8]: 
                     Factor1   Factor2   Factor3
    SS Loadings     3.510189  1.283710  0.737395
    Proportion Var  0.351019  0.128371  0.073739
    Cumulative Var  0.351019  0.479390  0.553129

Requirements
------------

-  Python 3.4 or higher
-  ``numpy``
-  ``pandas``
-  ``scikit-learn``
-  ``scipy``

Contributing
------------

Contributions to FactorAnalyzer are very welcome. Please file an issue
on GitHub, or contact jbiggs@ets.org if you would like to contribute.

Installation
------------

You can install this package via ``pip`` with:

``$ pip install factor_analyzer``

Alternatively, you can install via ``conda`` with:

``$ conda install -c desilinguist factor_analyzer``

License
-------

GNU General Public License (>= 2)
