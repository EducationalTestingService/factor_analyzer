"""
Testing utilities

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 05/21/2018
:organization: ETS
"""

import os
import math
import numpy as np
import pandas as pd

from factor_analyzer import FactorAnalyzer
from factor_analyzer import Rotator


DATA_DIR = os.path.join('tests', 'data')
EXPT_DIR = os.path.join('tests', 'expected')

OUTPUT_TYPES = ['value',
                'evalues',
                'loading',
                'uniquenesses',
                'communalities']


def do_analysis(test_name, factors, method, rotation, top_dir=None):
    """
    Use the `FactorAnalyzer()` class to perform the factor analysis
    and return a dictionary with relevant results for given scenario.

    Parameters
    ----------
    test_name : str
        The name of the test
    factors : int
        The number of factors
    method : str
        The rotation method
    rotation : str
        The type of rotation
    top_dir : str, optional
        The top directory for test data
        Defaults to `DATA_DIR``

    Returns
    -------
    dict
    """
    if top_dir is None:
        top_dir = DATA_DIR

    filename = os.path.join(top_dir, test_name + '.csv')
    data = pd.read_csv(filename)

    rotation = None if rotation == 'none' else rotation
    method = {'uls': 'minres'}.get(method, method)

    fa = FactorAnalyzer()
    fa.analyze(data, factors, method=method, rotation=rotation)

    evalues, values = fa.get_eigenvalues()

    return {'value': values,
            'evalues': evalues,
            'loading': fa.loadings,
            'uniquenesses': fa.get_uniqueness(),
            'communalities': fa.get_communalities()}


def get_r_output(test_name, factors, method, rotation, top_dir=None, output_types=None):
    """
    Get the R output for the given scenario.

    Parameters
    ----------
    test_name : str
        The name of the test
    factors : int
        The number of factors
    method : str
        The rotation method
    rotation : str
        The type of rotation
    top_dir : str, optional
        The top directory for test data
        Defaults to `EXPT_DIR``

    Returns
    -------
    dict
    """
    if top_dir is None:
        top_dir = EXPT_DIR

    if output_types is None:
        output_types = OUTPUT_TYPES

    output = {}
    for output_type in output_types:

        filename = '{}_{}_{}_{}_{}.csv'.format(output_type,
                                               method,
                                               rotation,
                                               factors,
                                               test_name)

        filename = os.path.join(top_dir, test_name, filename)

        data = pd.read_csv(filename)
        output[output_type] = data

    return output


def normalize(data, absolute=True):
    """
    Normalize the data.
    """
    # check for possible index column
    possible_index = [col for col in data.columns if 'Unnamed' in col]

    # get numeric columns
    numeric_cols = [col for col in data.dtypes[data.dtypes != 'object'].index.values
                    if col not in possible_index]

    # take absolute value
    if absolute:
        data[numeric_cols] = data[numeric_cols].abs()

    # set index
    if len(possible_index) == 1:
        data.set_index(possible_index[0], inplace=True)

    # sort the values
    data = data[data.abs().sum().sort_values(ascending=False).index.values]

    # update index name and column names
    data.index.name = ''
    data.columns = ['col{}'.format(i) for i in range(1, data.shape[1] + 1)]

    return data.reset_index(drop=True)


def check_close(data1, data2, rel_tol=0, abs_tol=0.1):
    """
    Check to make sure all values are close.
    """
    data1 = normalize(data1)
    data2 = normalize(data2)

    assert data1.shape == data2.shape

    arr = np.empty(shape=data1.shape, dtype=bool)
    for i in range(data1.shape[0]):
        for j in range(data2.shape[1]):
            check = math.isclose(data1.iloc[i, j],
                                 data2.iloc[i, j],
                                 rel_tol=rel_tol,
                                 abs_tol=abs_tol)
            arr[i, j] = check

    check = arr.sum(None) / arr.size
    return check


def check_scenario(test_name,
                   factors,
                   method,
                   rotation,
                   ignore_value=False,
                   ignore_communalities=False,
                   data_dir=None,
                   expt_dir=None,
                   rel_tol=0,
                   abs_tol=0.1):
    """
    Check all results for given scenario.
    """

    output_types = ['loading', 'evalues']

    if not ignore_value:
        output_types.extend(['value'])

    if not ignore_communalities:
        output_types.extend(['uniquenesses',
                             'communalities'])

    r_output = get_r_output(test_name, factors, method, rotation, expt_dir, output_types)
    py_output = do_analysis(test_name, factors, method, rotation, data_dir)

    for output_type in output_types:

        data1 = r_output[output_type]
        data2 = py_output[output_type]

        yield check_close(data1, data2, rel_tol, abs_tol)


def check_rotation(test_name,
                   factors,
                   method,
                   rotation,
                   data_dir=None,
                   expt_dir=None,
                   rel_tol=0,
                   abs_tol=0.1):

    r_input = get_r_output(test_name, factors, method,
                           'none', output_types=['loading'])
    r_loading = r_input['loading']
    r_loading = normalize(r_loading, absolute=False)

    rotator = Rotator()
    rotated_loading, _ = rotator.rotate(r_loading, rotation)

    r_output = get_r_output(test_name, factors, method, rotation, output_types=['loading'])
    expected_loading = r_output['loading']

    data1 = normalize(rotated_loading)
    data2 = normalize(expected_loading)

    print(data1, '\n', data2)
    print(data1.shape, data2.shape)

    return check_close(data1, data2, rel_tol, abs_tol)
