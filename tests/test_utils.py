"""
Tests utilities.

:author: Jeremy Biggs (jbiggs@ets.org)
:date: 02/15/2019
:organization: ETS
"""

import numpy as np
import pandas as pd
from factor_analyzer.utils import (duplication_matrix,
                                   duplication_matrix_pre_post,
                                   commutation_matrix,
                                   fill_lower_diag,
                                   get_first_idxs_from_values,
                                   get_free_parameter_idxs,
                                   get_symmetric_lower_idxs,
                                   get_symmetric_upper_idxs,
                                   merge_variance_covariance,
                                   unique_elements)

from nose.tools import eq_
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


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


def test_get_first_idxs_from_values():
    expected = [2, 0, 1], [0, 1, 2]
    x = np.array([[np.nan, 1, np.nan],
                  [np.nan, 1, 1],
                  [1, 2, np.nan],
                  [1, 1, 1]])
    output = get_first_idxs_from_values(x)

    eq_(output, expected)


def test_get_free_parameter_idxs():

    x = np.array([[np.nan, np.nan, 1],
                  [1, np.nan, 1],
                  [1, 1, np.nan]])

    expected = np.array([0, 3, 4, 8])

    output = get_free_parameter_idxs(x)
    assert_array_equal(output, expected)


def test_duplication_matrix():

    output = duplication_matrix(3)

    expected = np.array([[1., 0., 0., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0.],
                         [0., 0., 1., 0., 0., 0.],
                         [0., 1., 0., 0., 0., 0.],
                         [0., 0., 0., 1., 0., 0.],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 1., 0., 0., 0.],
                         [0., 0., 0., 0., 1., 0.],
                         [0., 0., 0., 0., 0., 1.]])

    assert_array_equal(output, expected)


def test_duplication_matrix_pre_post():

    x = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])

    expected = np.array([[1, 5, 4],
                         [14, 34, 20],
                         [13, 29, 16]])
    expected = pd.DataFrame(expected)

    output = duplication_matrix_pre_post(x)

    assert_frame_equal(output, expected)


def test_commutation_matrix():

    expected = np.array([[1., 0., 0., 0.],
                         [0., 0., 1., 0.],
                         [0., 1., 0., 0.],
                         [0., 0., 0., 1.]])

    output = commutation_matrix(2, 2)

    assert_array_equal(output, expected)


def test_get_symmetric_lower_idxs():

    expected = np.array([0, 1, 2, 3, 4, 6, 7, 8,
                         9, 12, 13, 14, 18, 19, 24])

    output = get_symmetric_lower_idxs(5)

    assert_array_equal(output, expected)


def test_get_symmetric_lower_idxs_no_diag():

    expected = np.array([5, 10, 11, 15, 16,
                         17, 20, 21, 22, 23])

    output = get_symmetric_lower_idxs(5, diag=False)

    assert_array_equal(output, expected)


def test_get_symmetric_upper_idxs():

    expected = np.array([0, 5, 10, 15, 20, 6, 11, 16,
                         21, 12, 17, 22, 18, 23, 24])

    output = get_symmetric_upper_idxs(5)

    assert_array_equal(output, expected)


def test_get_symmetric_upper_idxs_no_diag():

    expected = np.array([1, 2, 7, 3, 8, 13,
                         4, 9, 14, 19])

    output = get_symmetric_upper_idxs(5, diag=False)

    assert_array_equal(output, expected)
