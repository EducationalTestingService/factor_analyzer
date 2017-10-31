"""
Tests for FactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import pandas as pd

from pandas.util.testing import assert_frame_equal
from factor_analyzer.factor_analyzer import FactorAnalyzer


class TestFactorAnalyzer:

    def test_remove_all_columns(self):
        # test that columns with string values are removed.

        data = pd.DataFrame({'A': ['1', 2, 3, 4, 5],
                             'B': [6, 7, 8, 9, '10']})

        result = FactorAnalyzer.remove_non_numeric(data)

        assert result.empty

    def test_remove_no_columns(self):
        # test that no numeric columns are removed.

        data = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                             'B': [6.1, 7.2, 8.4, 9.2, 10.1]})

        result = FactorAnalyzer.remove_non_numeric(data)

        assert_frame_equal(data, result)

    def test_remove_one_column(self):
        # test that only column with string is removed.

        data = pd.DataFrame({'A': ['1', 2, 3, 4, 5],
                             'B': [6, 7, 8, 9, 10]})

        expected = pd.DataFrame({'B': [6, 7, 8, 9, 10]})

        result = FactorAnalyzer.remove_non_numeric(data)
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
