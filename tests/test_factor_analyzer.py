"""
Tests for FactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 10/25/2017
:organization: ETS
"""

import numpy as np
import pandas as pd

from nose.tools import raises
from numpy.testing import assert_array_almost_equal
from pandas.util.testing import assert_almost_equal

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from factor_analyzer.utils import smc
from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (calculate_kmo,
                                             calculate_bartlett_sphericity)


def test_calculate_bartlett_sphericity():

    path = 'tests/data/test01.csv'
    data = pd.read_csv(path)
    s, p = calculate_bartlett_sphericity(data.values)

    assert_almost_equal(s, 14185)
    assert_almost_equal(p, 0)


def test_calculate_kmo():

    path = 'tests/data/test02.csv'
    data = pd.read_csv(path)

    expected_overall = 0.81498469767761361

    values = [0.405516, 0.560049, 0.700033,
              0.705446, 0.829063, 0.848425,
              0.863502, 0.841143, 0.877076,
              0.839272]

    expected_by_item = np.array(values)

    (kmo_by_item,
     kmo_overall) = calculate_kmo(data.values)

    assert_almost_equal(kmo_by_item, expected_by_item)
    assert_almost_equal(kmo_overall, expected_overall)


def test_gridsearch():
    # make sure this doesn't fail

    X = pd.DataFrame(np.random.randn(1000).reshape(100, 10))
    y = pd.Series(np.random.choice([1, 0], size=100))

    grid = {'factoranalyzer__n_factors': [5, 7],
            'factoranalyzer__rotation': [None, 'varimax'],
            'decisiontreeclassifier__max_depth': [2, 5]}

    fa = FactorAnalyzer()
    decisiontree = DecisionTreeClassifier(random_state=123)
    pipe = make_pipeline(fa, decisiontree)

    gridsearch = GridSearchCV(pipe,
                              grid,
                              scoring='f1',
                              cv=3,
                              verbose=0)
    gridsearch.fit(X, y)


class TestFactorAnalyzer:

    def test_analyze_weights(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, 9, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        fa = FactorAnalyzer(rotation=None)
        fa.fit(data)
        _ = fa.transform(data)
        expected_weights = np.array(([[0.33536334, -2.72509646, 0],
                                      [0.33916605, -0.29388849, 0],
                                      [0.33444588, 3.03060826, 0]]))
        assert_array_almost_equal(expected_weights, fa.weights_)

    def test_analyze_impute_mean(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected.iloc[2, 1] = np.mean([4, 8, 10, 16, 18])
        expected_corr = expected.corr()
        expected_corr = expected_corr.values

        fa = FactorAnalyzer(rotation=None, impute='mean', n_factors=1)
        fa.fit(data)
        assert_array_almost_equal(fa.corr_, expected_corr)

    def test_analyze_impute_median(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected.iloc[2, 1] = np.median([4, 8, 10, 16, 18])
        expected_corr = expected.corr()
        expected_corr = expected_corr.values

        fa = FactorAnalyzer(rotation=None, impute='median', n_factors=1)
        fa.fit(data)
        assert_array_almost_equal(fa.corr_, expected_corr)

    def test_analyze_impute_drop(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected = expected.dropna()
        expected_corr = expected.corr()
        expected_corr = expected_corr.values

        fa = FactorAnalyzer(rotation=None, impute='drop', n_factors=1)
        fa.fit(data)
        assert_array_almost_equal(fa.corr_, expected_corr)

    @raises(ValueError)
    def test_analyze_bad_svd_method(self):
        fa = FactorAnalyzer(svd_method='foo')
        fa.fit(np.random.randn(500).reshape(100, 5))

    @raises(ValueError)
    def test_analyze_impute_value_error(self):
        fa = FactorAnalyzer(rotation=None, impute='blah', n_factors=1)
        fa.fit(np.random.randn(500).reshape(100, 5))

    @raises(ValueError)
    def test_analyze_rotation_value_error(self):
        fa = FactorAnalyzer(rotation='blah', n_factors=1)
        fa.fit(np.random.randn(500).reshape(100, 5))

    @raises(ValueError)
    def test_analyze_infinite(self):

        data = pd.DataFrame({'A': [1.0, 0.4, 0.5],
                             'B': [0.4, 1.0, float('inf')],
                             'C': [0.5, float('inf'), 1.0]},
                            index=['A', 'B', 'C'])

        fa = FactorAnalyzer(impute='drop', n_factors=1, is_corr_matrix=True)
        fa.fit(data)

    def test_smc_is_r_squared(self):
        # test that SMC is roughly equivalent to R-squared values.

        data = pd.DataFrame({'A': [10.5, 20.1, 30.2, 40.1, 50.3],
                             'B': [62, 71, 83, 91, 15],
                             'C': [0.45, 0.90, 0.22, 0.34, .045]})

        expected_r2 = [0.478330, 0.196223, 0.484519]
        expected_r2 = np.array(expected_r2)

        smc_result = smc(data.corr().values)

        assert_array_almost_equal(smc_result, expected_r2)

    def test_factor_variance(self):

        path = 'tests/data/test01.csv'
        data = pd.read_csv(path)

        fa = FactorAnalyzer(n_factors=3, rotation=None)
        fa.fit(data)
        loadings = fa.loadings_

        n_rows = loadings.shape[0]

        # calculate variance
        loadings = loadings ** 2
        variance = np.sum(loadings, axis=0)

        # calculate proportional variance
        proportional_variance_expected = variance / n_rows
        proportional_variance = fa.get_factor_variance()[1]

        assert_almost_equal(proportional_variance_expected, proportional_variance)
