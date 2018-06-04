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

from os.path import join


from factor_analyzer import FactorAnalyzer
from factor_analyzer import Rotator


DATA_DIR = os.path.join('tests', 'data')
EXPECTED_DIR = os.path.join('tests', 'expected')

OUTPUT_TYPES = ['value',
                'evalues',
                'loading',
                'uniquenesses',
                'communalities',
                'structure',
                'scores']


def calculate_py_output(test_name,
                        factors,
                        method,
                        rotation,
                        top_dir=None):
    """
    Use the `FactorAnalyzer()` class to perform the factor analysis
    and return a dictionary with relevant output for given scenario.

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
    output : dict
        A dictionary containing the outputs
        for all `OUTPUT_TYPES`.
    """
    if top_dir is None:
        top_dir = DATA_DIR

    filename = join(top_dir, test_name + '.csv')
    data = pd.read_csv(filename)

    rotation = None if rotation == 'none' else rotation
    method = {'uls': 'minres'}.get(method, method)

    fa = FactorAnalyzer()
    fa.analyze(data, factors, method=method, rotation=rotation)

    evalues, values = fa.get_eigenvalues()

    return {'value': values,
            'evalues': evalues,
            'structure': fa.structure,
            'loading': fa.loadings,
            'uniquenesses': fa.get_uniqueness(),
            'communalities': fa.get_communalities(),
            'scores': fa.get_scores(data)}


def collect_r_output(test_name,
                     factors,
                     method,
                     rotation,
                     output_types=None,
                     top_dir=None):
    """
    Get the R output for the given scenario.

    Parameters
    ----------
    test_name : str
        The name of the test (e.g. 'test01')
    factors : int
        The number of factors.
    method : str
        The rotation method (e.g. 'uls')
    rotation : str
        The type of rotation (e.g. 'varimax')
    output_types : list or None, optional
        The types of outputs:
            - 'loading'
            - 'value'
            - 'evalues'
            - 'uniqueness'
            = 'communalities'
    top_dir : str, optional
        The top directory for test data
        Defaults to `EXPECTED_DIR``

    Returns
    -------
    output : dict
        A dictionary containing the outputs
        for all `output_types`.
    """
    if top_dir is None:
        top_dir = EXPECTED_DIR

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


def normalize(data, absolute=False):
    """
    Normalize the data to ensure that Python
    and R output match. This involves ensuring
    the headers are named consistently and
    columns are sorted properly.

    Parameters
    ----------
    data : pd.DataFrame
        The data frame to normalize.
    absolute : bool, optional
        Whether to take the absolute value of
        all elements in the data frame.
        Defaults to True

    Returns
    -------
    data : pd.DataFrame
        The normalized data frame.
    """
    # check for possible index column

    # if there is an unnamed column, we want to make it the index
    possible_index = [col for col in data.columns if 'Unnamed' in col]

    # get numeric columns, in case we are taking absolute value
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


def check_close(data1, data2, rel_tol=0.0, abs_tol=0.1, absolute=False):
    """
    Check to make sure all values in two data frames
    are close. Returns the proportion that match.

    Parameters
    ----------
    data1 : pd.DataFrame
        The first data frame.
    data2 : pd.DataFrame
        The second data frame.
    rel_tol : float, optional
        The relative tolerance.
        Defaults to 0.0.
    abs_tol : float, optional
        The absolute tolerance.
        Defaults to 0.1.
    absolute : bool, optional
        Whether to take the absolute value of
        all elements in the data frame.
        Defaults to False

    Returns
    -------
    check : float
        The proportion that match.
    """
    data1 = normalize(data1, absolute)
    data2 = normalize(data2, absolute)

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
                   check_scores=False,
                   check_structure=False,
                   data_dir=None,
                   expected_dir=None,
                   rel_tol=0,
                   abs_tol=0.1):
    """
    Check all results for given scenario.

    Parameters
    ----------
    test_name : str
        The name of the test (e.g. 'test01')
    factors : int
        The number of factors.
    method : str
        The rotation method (e.g. 'uls')
    rotation : str
        The type of rotation (e.g. 'varimax')
    ignore_values : bool, optional
        Whether to ignore the `value` output type.
        Defaults to False.
    ignore_communalities : bool, optional
        Whether to ignore the `communalities` output type.
        Defaults to False.
    check_scores : bool, optional
        Check the factor scores
        Defaults to False.
    check_structure : bool, optional
        Check the structure matrix.
        This should only be used with
        oblique rotations.
        Defaults to False.
    data_dir : str, optional
        The directory with input data files.
        Defaults to `DATA_DIR`.
    expected_dir : str, optional
        The directory with output files.
        Defaults to `EXPECTED_DIR`.
    rel_tol : float, optional
        The relative tolerance.
        Defaults to 0.0.
    abs_tol : float, optional
        The absolute tolerance.
        Defaults to 0.1.

    Yields
    ------
    check : float
        The proportion that match between
        the calculated and expected.
    """

    output_types = ['loading', 'evalues']

    if not ignore_value:
        output_types.append('value')

    if not ignore_communalities:
        output_types.extend(['uniquenesses', 'communalities'])

    if check_scores:
        output_types.append('scores')

    if check_structure:
        output_types.append('structure')

    r_output = collect_r_output(test_name, factors, method, rotation, output_types, expected_dir)
    py_output = calculate_py_output(test_name, factors, method, rotation, data_dir)

    for output_type in output_types:

        data1 = r_output[output_type]
        data2 = py_output[output_type]

        yield check_close(data1, data2, rel_tol, abs_tol)


def check_rotation(test_name,
                   factors,
                   method,
                   rotation,
                   rel_tol=0,
                   abs_tol=0.1):

    """
    Check the rotation results.

    Parameters
    ----------
    test_name : str
        The name of the test (e.g. 'test01')
    factors : int
        The number of factors.
    method : str
        The rotation method (e.g. 'uls')
    rotation : str
        The type of rotation (e.g. 'varimax')
    rel_tol : float, optional
        The relative tolerance.
        Defaults to 0.0.
    abs_tol : float, optional
        The absolute tolerance.
        Defaults to 0.1.

    Returns
    ------
    check : float
        The proportion that match between
        the calculated and expected.
    """

    r_input = collect_r_output(test_name, factors, method,
                               'none', output_types=['loading'])
    r_loading = r_input['loading']
    r_loading = normalize(r_loading, absolute=False)

    rotator = Rotator()
    rotated_loading, _, _ = rotator.rotate(r_loading, rotation)

    r_output = collect_r_output(test_name, factors, method, rotation,
                                output_types=['loading'])
    expected_loading = r_output['loading']

    data1 = normalize(rotated_loading)
    data2 = normalize(expected_loading)

    return check_close(data1, data2, rel_tol, abs_tol)
