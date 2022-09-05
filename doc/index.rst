.. factor_analyzer documentation master file, created by
   sphinx-quickstart on Wed Dec  6 19:08:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FactorAnalyzer documentation!
==============================================

This is a Python module to perform exploratory and factor analysis (EFA), with several
optional rotations. It also includes a class to perform confirmatory factor
analysis (CFA), with certain pre-defined constraints. In expoloratory factor analysis,
factor extraction can be performed using a variety of estimation techniques. The
``factor_analyzer`` package allows users to perfrom EFA using either (1) a minimum
residual (MINRES) solution, (2) a maximum likelihood (ML) solution, or (3) a principal
factor solution. However, CFA can only be performe using an ML solution.

Both the EFA and CFA classes within this package are fully compatible with `scikit-learn`.
Portions of this code are ported from the excellent R library `psych`, and the `sem`
package provided inspiration for the CFA class.

Description
============

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

In adddition, the package includes a ``confirmatory_factor_analyzer``
module with a stand-alone ``ConfirmatoryFactorAnalyzer`` class. The
class includes ``fit()`` and ``transform()``  that enable users to perform
confirmatory factor analysis and score new data using the fitted model.
Performing CFA requires users to specify in advance a model specification
with the expected factor loading relationships. This can be done using
the ``ModelSpecificationParser`` class.

Requirements
==================
-  Python 3.4 or higher
-  ``numpy``
-  ``pandas``
-  ``scipy``
-  ``scikit-learn``

Installation
==================

You can install this package via ``pip`` with:

``$ pip install factor_analyzer``

Alternatively, you can install via ``conda`` with:

``$ conda install -c ets factor_analyzer``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   factor_analyzer

   confirmatory_factor_analyzer

   rotator


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
