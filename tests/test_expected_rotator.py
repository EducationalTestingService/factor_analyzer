"""
Tests for Rotator class

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 10/25/2017
:organization: ETS
"""

from factor_analyzer.test_utils import check_rotation

# set a threshold of roughly 95 percent matching
THRESHOLD = 0.95


def test_01_varimax_minres_2_factors():

    test_name = 'test01'
    factors = 2
    method = 'uls'
    rotation = 'varimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_02_varimax_minres_3_factors():

    test_name = 'test02'
    factors = 3
    method = 'uls'
    rotation = 'varimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_02_oblimax_minres_3_factors():

    test_name = 'test02'
    factors = 3
    method = 'uls'
    rotation = 'oblimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_02_oblimin_minres_3_factors():

    test_name = 'test02'
    factors = 3
    method = 'uls'
    rotation = 'oblimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_02_quartimax_minres_3_factors():

    test_name = 'test02'
    factors = 3
    method = 'uls'
    rotation = 'quartimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_02_quartimin_minres_3_factors():

    test_name = 'test02'
    factors = 3
    method = 'uls'
    rotation = 'quartimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_04_oblimax_minres_3_factors():

    test_name = 'test04'
    factors = 3
    method = 'uls'
    rotation = 'oblimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_04_oblimin_minres_3_factors():

    test_name = 'test04'
    factors = 3
    method = 'uls'
    rotation = 'oblimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_04_quartimax_minres_3_factors():

    test_name = 'test04'
    factors = 3
    method = 'uls'
    rotation = 'quartimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_04_quartimin_minres_3_factors():

    test_name = 'test04'
    factors = 3
    method = 'uls'
    rotation = 'quartimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_oblimax_minres_2_factors():

    test_name = 'test07'
    factors = 2
    method = 'uls'
    rotation = 'oblimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_oblimin_minres_3_factors():

    test_name = 'test07'
    factors = 2
    method = 'uls'
    rotation = 'oblimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_quartimax_minres_2_factors():

    test_name = 'test07'
    factors = 2
    method = 'uls'
    rotation = 'quartimax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_quartimin_minres_2_factors():

    test_name = 'test07'
    factors = 2
    method = 'uls'
    rotation = 'quartimin'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_equamax_minres_2_factors():

    test_name = 'test07'
    factors = 2
    method = 'uls'
    rotation = 'equamax'

    check = check_rotation(test_name, factors, method, rotation)
    assert check > THRESHOLD


def test_07_oblimin_minres_2_factors_gamma():

    test_name = 'test07_gamma'
    factors = 2
    method = 'uls'
    rotation = 'oblimin'
    gamma = 0.5

    check = check_rotation(test_name, factors, method, rotation, gamma=gamma)
    assert check > THRESHOLD
