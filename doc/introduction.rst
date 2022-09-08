Introduction
============

Exploratory factor analysis (EFA) is a statistical technique used to
identify latent relationships among sets of observed variables in a
dataset. In particular, EFA seeks to model a large set of observed
variables as linear combinations of some smaller set of unobserved,
latent factors. The matrix of weights, or factor loadings, generated
from an EFA model describes the underlying relationships between each
variable and the latent factors.

Confirmatory factor analysis (CFA), a closely associated technique, is
used to test a priori hypothesis about latent relationships among sets
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

This package includes a :mod:`~factor_analyzer.factor_analyzer` module with a
stand-alone :class:`~factor_analyzer.factor_analyzer.FactorAnalyzer` class. The
class includes :meth:`~factor_analyzer.factor_analyzer.FactorAnalyzer.fit()`
and :meth:`~factor_analyzer.factor_analyzer.FactorAnalyzer.transform()`
methods that enable users to perform factor analysis and score new data using
the fitted factor model. Users can also perform optional rotations on a factor
loading matrix using the :class:`~factor_analyzer.rotator.Rotator` class.

The following rotation options are available in both
:class:`~factor_analyzer.factor_analyzer.FactorAnalyzer` and
:class:`~factor_analyzer.rotator.Rotator`:

    (a) varimax (orthogonal rotation)
    (b) promax (oblique rotation)
    (c) oblimin (oblique rotation)
    (d) oblimax (orthogonal rotation)
    (e) quartimin (oblique rotation)
    (f) quartimax (orthogonal rotation)
    (g) equamax (orthogonal rotation)
    (h) geomin_obl (oblique rotation)
    (i) geomin_ort (orthogonal rotation)

In addition, the package includes a
:mod:`~factor_analyzer.confirmatory_factor_analyzer` module with a stand-alone
:class:`~factor_analyzer.confirmatory_factor_analyzer.ConfirmatoryFactorAnalyzer`
class. The class includes
:meth:`~factor_analyzer.confirmatory_factor_analyzer.ConfirmatoryFactorAnalyzer.fit()`
and
:meth:`~factor_analyzer.confirmatory_factor_analyzer.ConfirmatoryFactorAnalyzer.transform()`
that enable users to perform confirmatory factor analysis and score new data
using the fitted model. Performing CFA requires users to specify in advance a
model specification with the expected factor loading relationships. This can be
done using the
:class:`~factor_analyzer.confirmatory_factor_analyzer.ModelSpecificationParser`
class.

Requirements
------------
-  Python 3.7 or higher
-  ``numpy``
-  ``pandas``
-  ``scipy``
-  ``scikit-learn``
-  ``pre-commit``

Installation
------------

You can install this package via ``pip`` with:

``$ pip install factor_analyzer``

Alternatively, you can install via ``conda`` with:

``$ conda install -c ets factor_analyzer``
