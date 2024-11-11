.. factor_analyzer documentation master file, created by
   sphinx-quickstart on Wed Dec  6 19:08:01 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the FactorAnalyzer documentation!
==============================================

This is a Python module to perform exploratory and factor analysis (EFA), with
several optional rotations. It also includes a class to perform confirmatory
factor analysis (CFA), with certain pre-defined constraints. In exploratory
factor analysis, factor extraction can be performed using a variety of
estimation techniques. The :ref:`factor_analyzer <factor_analyzer_api>` package
allows users to perform EFA using either (1) a minimum residual (MINRES)
solution, (2) a maximum likelihood (ML) solution, or (3) a principal factor
solution. However, CFA can only be performed using an ML solution.

Both the EFA and CFA classes within this package are fully compatible with
``scikit-learn``. Portions of this code are ported from the excellent R library
``psych``, and the ``sem`` package provided inspiration for the CFA class.

.. important::

   Please make sure to read the :ref:`important notes <important_notes>`
   section if you encounter any unexpected results.


Documentation
=============

.. toctree::
   :maxdepth: 3

   introduction
   notes
   factor_analyzer


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
