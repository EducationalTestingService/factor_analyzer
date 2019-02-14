"""
Tests for ConfirmatoryFactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 02/12/2019
:organization: ETS
"""

from factor_analyzer.test_utils import check_cfa


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
