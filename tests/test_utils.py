"""
"""
import numpy as np
from factor_analyzer.utils import (fill_lower_diag,
                                   merge_variance_covariance,
                                   unique_elements)

from nose.tools import eq_
from numpy.testing import assert_array_equal


def test_unique_elements():

    expected = [1, 2, 3, 4, 5]

    x = [1, 2, 3, 2, 1, 3, 4, 4, 3, 5, 5]
    output = unique_elements(x)
    eq_(output, expected)


def test_fill_lower_diag():

    expected = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [2, 3, 0]])
    output = fill_lower_diag([1, 2, 3])
    assert_array_equal(output, expected)


def test_merge_variance_covariance_no_covariance():

    expected = np.eye(4)

    x = [1, 1, 1, 1]

    output = merge_variance_covariance(x)
    assert_array_equal(output, expected)


def test_merge_variance_covariance():

    expected = [[1, .25, .45],
                [.25, 1, .35],
                [.45, .35, 1]]
    expected = np.array(expected)

    x = [1, 1, 1]
    y = [.25, .45, .35]

    output = merge_variance_covariance(x, y)
    assert_array_equal(output, expected)
