"""
Tests for ConfirmatoryFactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 02/12/2019
:organization: ETS
"""
import numpy as np
import pandas as pd
from factor_analyzer.test_utils import check_cfa
from factor_analyzer.confirmatory_factor_analyzer import ModelParser

from nose.tools import eq_, raises
from numpy.testing import assert_array_equal


THRESHOLD = 1.0


def test_11_cfa():

    json_name_input = 'test11'
    data_name_input = 'test11'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           fix_first=False,
                           rel_tol=0.1):
        assert check >= THRESHOLD


def test_11_fix_first_cfa():

    json_name_input = 'test11-fix-first'
    data_name_input = 'test11'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           fix_first=True,
                           rel_tol=0.1):
        assert check >= THRESHOLD


def test_12_cfa():

    json_name_input = 'test12'
    data_name_input = 'test12'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           fix_first=False,
                           rel_tol=0.1):
        assert check >= THRESHOLD


def test_12_cfa_fix_first_cfa():

    json_name_input = 'test12-fix-first'
    data_name_input = 'test12'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           fix_first=True,
                           rel_tol=0.1):
        assert check >= THRESHOLD


def test_13_cfa():

    json_name_input = 'test13'
    data_name_input = 'test13'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           index_col=0,
                           fix_first=False,
                           is_cov=True,
                           n_obs=64,
                           maxiter=250,
                           rel_tol=0.1):
        assert check >= THRESHOLD


def test_13_cfa_first_cfa():

    json_name_input = 'test13-fix-first'
    data_name_input = 'test13'

    for check in check_cfa(json_name_input,
                           data_name_input,
                           index_col=0,
                           fix_first=True,
                           is_cov=True,
                           n_obs=64,
                           maxiter=250,
                           rel_tol=0.1):
        assert check >= THRESHOLD


class TestModelParser:

    def setUp(self):

        self.elements = ['loadings',
                         'error_vars',
                         'factor_vars',
                         'factor_covs',
                         'variable_names',
                         'factor_names',
                         'n_factors',
                         'n_variables',
                         'n_lower_diag']

    def check_result(self, result, expected):
        for idx in range(len(result)):
            print(self.elements[idx])
            x, y = result[idx], expected[idx]
            if isinstance(x, np.ndarray):
                assert_array_equal(x, y)
            else:
                eq_(x, y)

    def test_model_parser_with_only_loadings(self):

        loadings = pd.DataFrame({'F1': [1, 1, 1, 1, 0, 0],
                                 'F2': [0, 0, 0, 1, 1, 1]}).values

        expected = (loadings,                  # loadings
                    np.full((6, 1), np.nan),   # error_vars
                    np.full((2, 1), np.nan),   # factor_vars
                    np.full((1, 1), np.nan),   # factor_covs
                    ['var1', 'var2', 'var3', 'var4', 'var5', 'var6'],
                    ['F1', 'F2'],
                    2,                         # n_factors
                    6,                         # n_variables
                    1)                         # n_lower_diag

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']}}

        mp = ModelParser()
        result = mp.parse(model)
        self.check_result(result, expected)

    def test_model_parser_with_loadings_and_factor_var_and_cov(self):

        loadings = pd.DataFrame({'F1': [1, 1, 1, 1, 0, 0],
                                 'F2': [0, 0, 0, 1, 1, 1]}).values

        expected = (loadings,                  # loadings
                    np.full((6, 1), np.nan),   # error_vars
                    np.ones((2, 1)),           # factor_vars
                    np.zeros((1, 1)),          # factor_covs
                    ['var1', 'var2', 'var3', 'var4', 'var5', 'var6'],
                    ['F1', 'F2'],
                    2,                         # n_factors
                    6,                         # n_variables
                    1)                         # n_lower_diag

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'factor_vars': [1, 1],
                 'factor_covs': [[0, 0], [0, 0]]}

        mp = ModelParser()
        result = mp.parse(model)
        self.check_result(result, expected)

    @raises(AssertionError)
    def test_model_parser_with_factor_cov_mismatch_shape(self):

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'factor_covs': [[0, 0], [0, 0], [0, 0]]}

        mp = ModelParser()
        mp.parse(model)

    @raises(AssertionError)
    def test_model_parser_with_factor_cov_1d_too_long(self):

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'factor_covs': [0, 0, 0, 0, 0, 0]}

        mp = ModelParser()
        mp.parse(model)

    @raises(TypeError)
    def test_model_parser_with_loadings_not_dict(self):

        model = {'loadings': 'f1 = var1 + var2 + var3'}

        mp = ModelParser()
        mp.parse(model)

    @raises(AssertionError)
    def test_model_parser_with_error_vars_too_long(self):

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'errors_vars': [0, 0, 0, 0, 0, 0, 0]}

        mp = ModelParser()
        mp.parse(model)

    @raises(AssertionError)
    def test_model_parser_with_factor_vars_too_long(self):

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'factor_vars': [0, 0, 0]}

        mp = ModelParser()
        mp.parse(model)

    @raises(KeyError)
    def test_model_parser_with_wrong_key(self):

        model = {'loadings': {'F1': ['var1', 'var2', 'var3', 'var4'],
                              'F2': ['var4', 'var5', 'var6']},
                 'factor_varsings': [0, 0, 0]}

        mp = ModelParser()
        mp.parse(model)
