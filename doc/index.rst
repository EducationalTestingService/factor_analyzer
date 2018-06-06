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

This package includes a stand-alone Python module with a ``FactorAnalyzer()`` class. The class includes an ``analyze()`` method that allows users to perform
factor analysis using either minres or MLE, with optional rotations on the factor
loading matrices. The package also offers a stand-alone ``Rotator()`` class to
perform common rotations on an unrotated loading matrix.

The ``factor_analyzer`` package offers the following rotation methods:

* The **varimax** (orthogonal)

* The **promax** (oblique)

* The **quartimax** (orthogonal)

* The **quartimin** (oblique)

* The **obliimax** (orthogonal)

* The **oblimin** (oblique)

* The **obliimax** (orthogonal)

* The **equamax** (orthogonal)

This package includes a stand-alone Python module with a ``FactorAnalyzer()`` class. The class includes an ``analyze()`` method that allows users to perform factor analysis using either minres or MLE, with optional promax or varimax rotations on the factor loading matrices. The package also offers a stand-alone ``Rotator()`` class to perform common rotations on an unrotated loading matrix.

Requirements
==================
- Python 3.4 or higher
- ``numpy``
- ``pandas``
- ``scipy``

Installation
==================

You can install this package via ``pip`` with:

``$ pip install factor_analyzer``

Alternatively, you can install via ``conda`` with:

``$ conda install -c desilinguist factor_analyzer``

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   factor_analyzer

   rotator


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`