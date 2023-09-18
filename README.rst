FactorAnalyzer
--------------

.. image:: https://gitlab.com/EducationalTestingService/factor_analyzer/badges/main/pipeline.svg
   :target: https://gitlab.com/EducationalTestingService/factor_analyzer/-/pipelines
   :alt: Build status

.. image:: https://codecov.io/gh/EducationalTestingService/factor_analyzer/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/EducationalTestingService/factor_analyzer
   :alt: Code coverage

.. image:: https://anaconda.org/ets/factor_analyzer/badges/version.svg
   :target: https://anaconda.org/ets/factor_analyzer/
   :alt: Conda version

.. image:: https://img.shields.io/pypi/v/factor_analyzer
   :target: https://pypi.org/project/factor-analyzer/
   :alt: PyPI version

.. image:: https://img.shields.io/readthedocs/factor_analyzer/latest.svg
   :target: https://factor-analyzer.readthedocs.io/
   :alt: Docs

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: Pre-commit checks


This is a Python module to perform exploratory and factor analysis (EFA), with several
optional rotations. It also includes a class to perform confirmatory factor
analysis (CFA), with certain pre-defined constraints. In exploratory factor analysis,
factor extraction can be performed using a variety of estimation techniques. The
``factor_analyzer`` package allows users to perform EFA using either (1) a minimum
residual (MINRES) solution, (2) a maximum likelihood (ML) solution, or (3) a principal
factor solution. However, CFA can only be performed using an ML solution.

Both the EFA and CFA classes within this package are fully compatible with `scikit-learn`.
Portions of this code are ported from the excellent R library `psych`, and the `sem`
package provided inspiration for the CFA class.

Please see the `official documentation <https://factor-analyzer.readthedocs.io/en/latest/index.html>`__ for additional details.


Description
-----------

Exploratory factor analysis (EFA) is a statistical technique used to
identify latent relationships among sets of observed variables in a
dataset. In particular, EFA seeks to model a large set of observed
variables as linear combinations of some smaller set of unobserved,
latent factors. The matrix of weights, or factor loadings, generated
from an EFA model describes the underlying relationships between each
variable and the latent factors.

Confirmatory factor analysis (CFA), a closely associated technique, is
used to test an a priori hypothesis about latent relationships among sets
of observed variables. In CFA, the researcher specifies the expected pattern
of factor loadings (and possibly other constraints), and fits a model according
to this specification.

Typically, a number of factors (K) in an EFA or CFA model is selected
such that it is substantially smaller than the number of variables. The
factor analysis model can be estimated using a variety of standard
estimation methods, including but not limited MINRES or ML.

Factor loadings are similar to standardized regression coefficients, and
variables with higher loadings on a particular factor can be interpreted
as explaining a larger proportion of the variation in that factor. In the
case of EFA, factor loading matrices are usually rotated after the factor
analysis model is estimated in order to produce a simpler, more interpretable
structure to identify which variables are loading on a particular factor.

Two common types of rotations are:

1. The **varimax** rotation, which rotates the factor loading matrix so
   as to maximize the sum of the variance of squared loadings, while
   preserving the orthogonality of the loading matrix.

2. The **promax** rotation, a method for oblique rotation, which builds
   upon the varimax rotation, but ultimately allows factors to become
   correlated.

This package includes a ``factor_analyzer`` module with a stand-alone
``FactorAnalyzer`` class. The class includes ``fit()`` and ``transform()``
methods that enable users to perform factor analysis and score new data
using the fitted factor model. Users can also perform optional rotations
on a factor loading matrix using the ``Rotator`` class.

The following rotation options are available in both ``FactorAnalyzer``
and ``Rotator``:

    (a) varimax (orthogonal rotation)
    (b) promax (oblique rotation)
    (c) oblimin (oblique rotation)
    (d) oblimax (orthogonal rotation)
    (e) quartimin (oblique rotation)
    (f) quartimax (orthogonal rotation)
    (g) equamax (orthogonal rotation)
    (h) geomin_obl (oblique rotation)
    (i) geomin_ort (orthogonal rotation)

In addition, the package includes a ``confirmatory_factor_analyzer``
module with a stand-alone ``ConfirmatoryFactorAnalyzer`` class. The
class includes ``fit()`` and ``transform()``  that enable users to perform
confirmatory factor analysis and score new data using the fitted model.
Performing CFA requires users to specify in advance a model specification
with the expected factor loading relationships. This can be done using
the ``ModelSpecificationParser`` class.

Note that the ``ConfirmatoryFactorAnalyzer`` class is very experimental at this point,
so use it with caution, especially if your data are highly non-normal.

Examples
--------

Exploratory factor analysis example.

.. code:: python

  In [1]: import pandas as pd
     ...: from factor_analyzer import FactorAnalyzer

  In [2]: df_features = pd.read_csv('tests/data/test02.csv')

  In [3]: fa = FactorAnalyzer(rotation=None)

  In [4]: fa.fit(df_features)
  Out[4]:
  FactorAnalyzer(bounds=(0.005, 1), impute='median', is_corr_matrix=False,
                 method='minres', n_factors=3, rotation=None, rotation_kwargs={},
                 use_smc=True)

  In [5]: fa.loadings_
  Out[5]:
  array([[-0.12991218,  0.16398151,  0.73823491],
         [ 0.03899558,  0.04658425,  0.01150343],
         [ 0.34874135,  0.61452341, -0.07255666],
         [ 0.45318006,  0.7192668 , -0.0754647 ],
         [ 0.36688794,  0.44377343, -0.01737066],
         [ 0.74141382, -0.15008235,  0.29977513],
         [ 0.741675  , -0.16123009, -0.20744497],
         [ 0.82910167, -0.20519428,  0.04930817],
         [ 0.76041819, -0.23768727, -0.12068582],
         [ 0.81533404, -0.12494695,  0.17639684]])

  In [6]: fa.get_communalities()
  Out[6]:
  array([0.5887579 , 0.00382308, 0.50452402, 0.72841182, 0.33184336,
         0.66208429, 0.61911037, 0.73194557, 0.64929612, 0.71149718])

Confirmatory factor analysis example.

.. code:: python

  In [1]: import pandas as pd

  In [2]: from factor_analyzer import (ConfirmatoryFactorAnalyzer,
     ...:                              ModelSpecificationParser)

  In [3]: df_features = pd.read_csv('tests/data/test11.csv')

  In [4]: model_dict = {"F1": ["V1", "V2", "V3", "V4"],
     ...:               "F2": ["V5", "V6", "V7", "V8"]}
  In [5]: model_spec = ModelSpecificationParser.parse_model_specification_from_dict(df_features,
     ...:                                                                           model_dict)

  In [6]: cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)

  In [7]: cfa.fit(df_features.values)

  In [8]: cfa.loadings_
  Out[8]:
  array([[0.99131285, 0.        ],
         [0.46074919, 0.        ],
         [0.3502267 , 0.        ],
         [0.58331488, 0.        ],
         [0.        , 0.98621042],
         [0.        , 0.73389239],
         [0.        , 0.37602988],
         [0.        , 0.50049507]])

  In [9]: cfa.factor_varcovs_
  Out[9]:
  array([[1.        , 0.17385704],
         [0.17385704, 1.        ]])

  In [10]: cfa.transform(df_features.values)
  Out[10]:
  array([[-0.46852166, -1.08708035],
         [ 2.59025301,  1.20227783],
         [-0.47215977,  2.65697245],
         ...,
         [-1.5930886 , -0.91804114],
         [ 0.19430887,  0.88174818],
         [-0.27863554, -0.7695101 ]])

Requirements
------------

-  Python 3.8 or higher
-  ``numpy``
-  ``pandas``
-  ``scipy``
-  ``scikit-learn``

Contributing
------------

Contributions to ``factor_analyzer`` are very welcome. Please file an issue
in the repository if you would like to contribute.

You can install the development requirements in a virtual environment with:

.. code-block:: bash

   python -m pip install -e .[dev]
   pre-commit install

Installation
------------

You can install this package via ``pip`` with:

``$ pip install factor_analyzer``

Alternatively, you can install via ``conda`` with:

``$ conda install -c ets factor_analyzer``

License
-------

GNU General Public License (>= 2)
