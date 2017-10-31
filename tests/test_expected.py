"""
Tests for FactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import os
import pandas as pd

from pandas.util.testing import assert_frame_equal
from factor_analyzer.factor_analyzer import FactorAnalyzer


class TestFactorAnalyzerVsR:

    def __init__(self):

        self.data_dir = 'tests/data'
        self.expected_dir = 'tests/expected'

    def do_analysis(self, filename, factors, method, rotation):
        """
        Use the `FactorAnalyzer()` class to perform the factor analysis
        and return a dictionary with relevant results.
        """

        filename = os.path.join(self.data_dir, filename)
        data = pd.read_csv(filename)

        rotation = None if rotation == 'none' else rotation

        fa = FactorAnalyzer(factors)
        fa.analyze(data, method=method, rotation=rotation)

        return {'loadings': fa.loadings,
                'eigenvals': fa.get_eigenvalues(),
                'uniqueness': fa.get_uniqueness(),
                'communalities': fa.get_communalities()}

    def get_data_by_type(self, filename, factors, method, rotation, filetype='loadings'):
        """
        Get the R output data by file type (e.g. 'loadings' or 'commonalities').
        """

        subdir, _ = os.path.splitext(filename)

        new_file_name = '_'.join([filetype, method, rotation, str(factors), subdir + '.csv'])
        new_file_name = os.path.join(self.expected_dir, subdir, new_file_name)

        data = pd.read_csv(new_file_name)
        return data

    def get_expected(self, filename, factors, method, rotation):
        """
        Get the expected output from the R `fa()` function
        and return dictionary with relevant results.
        """

        loadings = self.get_data_by_type(filename, factors, method, rotation, 'loadings')
        eigenvals = self.get_data_by_type(filename, factors, method, rotation, 'eigenvals')
        communal = self.get_data_by_type(filename, factors, method, rotation, 'communalities')
        unique = self.get_data_by_type(filename, factors, method, rotation, 'uniquenesses')

        return {'loadings': loadings,
                'eigenvals': eigenvals,
                'communalities': communal,
                'uniqueness': unique}

    def normalize(self, data, filetype='loadings'):
        """
        Normalize the DataFrame to make sure R output and Python
        output match. In particular, ignore signs, round data, sort columns,
        and make sure that column and index names are correct.
        """
        if filetype == 'loadings':

            if ('MR1' in data.columns.values or
                'ML1' in data.columns.values):

                data = data.set_index(data.columns.values[0])

            data.index.rename('new_index', inplace=True)

            data = data[data.abs().sum().sort_values(ascending=False).index.values]

            data = data.abs().round(5)

            data.columns = ['col{}'.format(num)for num in range(1, data.shape[1] + 1)]
            return data

        else:

            if ('x' in data.columns.values):

                data = data.set_index(data.columns.values[0])

            data.index.rename('new_index', inplace=True)

            if filetype == 'communalities':
                data.columns = ['Communalities']
            elif filetype == 'uniqueness':
                data.columns = ['Uniqueness']
            elif filetype == 'eigenvals':
                data.reset_index(inplace=True, drop=True)
                data.columns = ['Eigenvalues']

            return data

    def check_results(self,
                      results,
                      expected,
                      check_loadings=True,
                      check_communalities=False,
                      check_uniqueness=False,
                      check_eigenvalues=True):
        """
        Check results of R vs Python.

        By default, do not check communalities and uniqueness, since
        we will have slightly different results with Promax and Varimax rotations.
        """
        loadings_p = self.normalize(results['loadings'], 'loadings')
        loadings_r = self.normalize(expected['loadings'], 'loadings')

        communalities_p = self.normalize(results['communalities'], 'communalities')
        communalities_r = self.normalize(expected['communalities'], 'communalities')

        uniqueness_p = self.normalize(results['uniqueness'], 'uniqueness')
        uniqueness_r = self.normalize(expected['uniqueness'], 'uniqueness')

        eigenvals_p = self.normalize(results['eigenvals'], 'eigenvals')
        eigenvals_r = self.normalize(expected['eigenvals'], 'eigenvals')

        if check_loadings:
            try:
                assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)
            except AssertionError as error:
                print('Problem with loadings.')
                raise error

        if check_communalities:
            try:
                assert_frame_equal(communalities_p, communalities_r, check_less_precise=1)
            except AssertionError as error:
                print('Problem with communalities.')
                raise error

        if check_uniqueness:
            try:
                assert_frame_equal(uniqueness_p, uniqueness_r, check_less_precise=1)
            except AssertionError as error:
                print('Problem with uniquness.')
                raise error

        if check_eigenvalues:
            try:
                assert_frame_equal(eigenvals_p, eigenvals_r, check_less_precise=1)
            except AssertionError as error:
                print('Problem with eigenvalues.')
                raise error

    def test_01_none_minres_2_factors_unique(self):

        filename = 'test01.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_01_none_minres_2_factors(self):

        filename = 'test01.csv'
        factors = 2
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_01_none_minres_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_01_none_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_01_varimax_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_01_promax_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_02_none_minres_3_factors(self):

        filename = 'test02.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_03_none_minres_3_factors(self):

        filename = 'test03.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_03_promax_minres_3_factors(self):

        filename = 'test03.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_04_varimax_ml_2_factors(self):

        filename = 'test04.csv'
        factors = 2
        method = 'ml'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_05_varimax_minres_3_factors(self):

        filename = 'test05.csv'
        factors = 3
        method = 'minres'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_06_promax_ml_3_factors(self):

        filename = 'test06.csv'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)

    def test_06_none_minres_2_factors(self):

        filename = 'test06.csv'
        factors = 2
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected, check_communalities=True, check_uniqueness=True)

    def test_07_varimax_ml_3_factors(self):

        filename = 'test07.csv'
        factors = 3
        method = 'minres'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        self.check_results(results, expected)
