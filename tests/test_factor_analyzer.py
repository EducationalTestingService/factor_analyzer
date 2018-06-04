"""
Tests for FactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import numpy as np
import pandas as pd

from nose.tools import raises
from pandas.util.testing import assert_frame_equal, assert_almost_equal

from factor_analyzer.factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import (calculate_kmo,
                                             calculate_bartlett_sphericity,
                                             covariance_to_correlation,
                                             partial_correlations)


def test_calculate_bartlett_sphericity():

    path = 'tests/data/test01.csv'
    data = pd.read_csv(path)
    s, p = calculate_bartlett_sphericity(data)

    assert_almost_equal(s, 14185)
    assert_almost_equal(p, 0)


def test_calculate_kmo():

    path = 'tests/data/test02.csv'
    data = pd.read_csv(path)

    expected_overall = 0.81498469767761361

    index = ['sex', 'zygosity', 'moed', 'faed',
             'faminc', 'english', 'math', 'socsci',
             'natsci', 'vocab']

    values = [0.405516, 0.560049, 0.700033,
              0.705446, 0.829063, 0.848425,
              0.863502, 0.841143, 0.877076,
              0.839272]

    expected_by_item = pd.DataFrame(values,
                                    columns=['KMO'],
                                    index=index)

    (kmo_by_item,
     kmo_overall) = calculate_kmo(data)

    assert_almost_equal(kmo_by_item, expected_by_item)
    assert_almost_equal(kmo_overall, expected_overall)


def test_covariance_to_correlation():

    path = 'tests/data/test02.csv'
    data = pd.read_csv(path)

    expected_corr = data.corr().values

    corr = covariance_to_correlation(data.cov().values)

    assert_almost_equal(corr, expected_corr)


@raises(ValueError)
def test_covariance_to_correlation_value_error():

    covariance_to_correlation(np.array([[23, 12, 23],
                                        [42, 25, 21]]))


def test_partial_correlations():

    data = pd.DataFrame([[12, 14, 15],
                         [24, 12, 52],
                         [35, 12, 41],
                         [23, 12, 42]])

    expected = [[1.0, -0.730955, -0.50616],
                [-0.730955, 1.0, -0.928701],
                [-0.50616, -0.928701, 1.0]]

    expected = pd.DataFrame(expected,
                            columns=[0, 1, 2],
                            index=[0, 1, 2])

    result = partial_correlations(data)
    assert_almost_equal(result, expected)


def test_partial_correlations_num_columns_greater():

    # columns greater than rows
    data = pd.DataFrame([[23, 12, 23],
                         [42, 25, 21]])

    empty_array = np.empty((3, 3))
    empty_array[:] = np.nan
    np.fill_diagonal(empty_array, 1.0)

    expected = pd.DataFrame(empty_array,
                            columns=[0, 1, 2],
                            index=[0, 1, 2])

    result = partial_correlations(data)
    assert_almost_equal(result, expected)


def test_partial_correlations_catch_linalgerror():

    # Covariance matrix that will be singular
    data = pd.DataFrame([[10, 10, 10, 10],
                         [12, 12, 12, 12],
                         [15, 15, 15, 15],
                         [20, 20, 20, 20],
                         [11, 11, 11, 11]])

    empty_array = np.empty((4, 4))
    empty_array[:] = np.nan
    np.fill_diagonal(empty_array, 1.0)

    expected = pd.DataFrame(empty_array,
                            columns=[0, 1, 2, 3],
                            index=[0, 1, 2, 3])

    result = partial_correlations(data)
    assert_almost_equal(result, expected)


class TestFactorAnalyzer:

    def test_analyze_impute_mean(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected.iloc[2, 1] = np.mean([4, 8, 10, 16, 18])
        expected_corr = expected.corr()

        fa = FactorAnalyzer()
        fa.analyze(data, 1, rotation=None, impute='mean')
        assert_frame_equal(fa.corr, expected_corr)

    def test_analyze_impute_median(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected.iloc[2, 1] = np.median([4, 8, 10, 16, 18])
        expected_corr = expected.corr()

        fa = FactorAnalyzer()
        fa.analyze(data, 1, rotation=None, impute='median')
        assert_frame_equal(fa.corr, expected_corr)

    def test_analyze_impute_drop(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        expected = data.copy()
        expected = expected.dropna()
        expected_corr = expected.corr()

        fa = FactorAnalyzer()
        fa.analyze(data, 1, rotation=None, impute='drop')
        assert_frame_equal(fa.corr, expected_corr)

    @raises(ValueError)
    def test_analyze_impute_value_error(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        fa = FactorAnalyzer()
        fa.analyze(data, 1, rotation=None, impute='blah')

    @raises(ValueError)
    def test_analyze_rotation_value_error(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, np.nan, 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        fa = FactorAnalyzer()
        fa.analyze(data, 1, rotation='blah')

    @raises(ValueError)
    def test_analyze_infinite(self):

        data = pd.DataFrame({'A': [2, 4, 5, 6, 8, 9],
                             'B': [4, 8, float('inf'), 10, 16, 18],
                             'C': [6, 12, 15, 12, 26, 27]})

        fa = FactorAnalyzer()
        fa.analyze(data, 1, impute='drop')

    def test_remove_all_columns(self):
        # test that columns with string values are removed.

        data = pd.DataFrame({'A': ['1', 2, 3, 4, 5],
                             'B': [6, 7, 8, 9, '10']})

        result = FactorAnalyzer().remove_non_numeric(data)

        assert result.empty

    def test_remove_no_columns(self):
        # test that no numeric columns are removed.

        data = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                             'B': [6.1, 7.2, 8.4, 9.2, 10.1]})

        result = FactorAnalyzer().remove_non_numeric(data)

        assert_frame_equal(data, result)

    def test_remove_one_column(self):
        # test that only column with string is removed.

        data = pd.DataFrame({'A': ['1', 2, 3, 4, 5],
                             'B': [6, 7, 8, 9, 10]})

        expected = pd.DataFrame({'B': [6, 7, 8, 9, 10]})

        result = FactorAnalyzer().remove_non_numeric(data)
        assert_frame_equal(expected, result)

    def test_smc_is_r_squared(self):
        # test that SMC is roughly equivalent to R-squared values.

        data = pd.DataFrame({'A': [10.5, 20.1, 30.2, 40.1, 50.3],
                             'B': [62, 71, 83, 91, 15],
                             'C': [0.45, 0.90, 0.22, 0.34, .045]})

        expected_r2 = [0.478330, 0.196223, 0.484519]
        expected_r2 = pd.DataFrame(expected_r2,
                                   index=['A', 'B', 'C'],
                                   columns=['SMC'])

        smc_result = FactorAnalyzer.smc(data)

        assert_frame_equal(smc_result, expected_r2, check_less_precise=2)

    def test_factor_variance(self):

        path = 'tests/data/test01.csv'
        data = pd.read_csv(path)

        fa = FactorAnalyzer()
        fa.analyze(data, 3, rotation=None)
        loadings = fa.loadings

        n_rows = loadings.shape[0]

        # calculate variance
        loadings = loadings ** 2
        variance = loadings.sum(axis=0)

        # calculate proportional variance
        proportional_variance_expected = variance / n_rows
        proportional_variance = fa.get_factor_variance().loc['Proportion Var']

        proportional_variance_expected.name = ''
        proportional_variance.name = ''

        assert_almost_equal(proportional_variance_expected, proportional_variance)
