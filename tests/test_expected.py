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

    def get_data_by_type(self, filename, factors, method, rotation, filetype='loadings'):

        subdir, _ = os.path.splitext(filename)

        # get file path
        new_file_name = '_'.join([filetype, method, rotation, str(factors), subdir + '.csv'])
        new_file_name = os.path.join(self.expected_dir, subdir, new_file_name)

        data = pd.read_csv(new_file_name)
        return data

    def do_analysis(self, filename, factors, method, rotation):
        """
        Use the FactorAnalyzer() class to perform the factor analysis
        and return a dictionary with relevant results.
        """

        filename = os.path.join(self.data_dir, filename)
        data = pd.read_csv(filename)

        rotation = None if rotation == 'none' else rotation

        fa = FactorAnalyzer(factors)
        fa.analyze(data, method=method, rotation=rotation)

        return {'loadings': fa.loadings}

    def get_expected(self, filename, factors, method, rotation):
        """
        Get the expected output from the R `fa()` function
        and return dictionary with relevant results.
        """

        loadings = self.get_data_by_type(filename, factors, method, rotation, 'loadings')
        return {'loadings': loadings}

    def normalize(self, data):
        """
        Normalize the DataFrame to make sure R output and Python
        output match. In particular, ignore signs, round data, sort columns,
        and make sure that column and index names are correct.
        """

        if ('MR1' in data.columns.values or
            'ML1' in data.columns.values):

            data = data.set_index(data.columns.values[0])

        data.index.rename('new_index', inplace=True)

        data = data[data.abs().sum().sort_values(ascending=False).index.values]

        data = data.abs().round(5)

        # make columns names consistent
        data.columns = ['MR{}'.format(num)for num in range(1, data.shape[1] + 1)]
        return data

    def test_01_none_minres_2_factors(self):

        filename = 'test01.csv'
        factors = 2
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_01_none_minres_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_01_none_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_01_varimax_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_01_promax_ml_3_factors(self):

        filename = 'test01.csv'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_02_none_minres_3_factors(self):

        filename = 'test02.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_03_none_minres_3_factors(self):

        filename = 'test03.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_03_promax_minres_3_factors(self):

        filename = 'test03.csv'
        factors = 3
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_04_varimax_ml_2_factors(self):

        filename = 'test04.csv'
        factors = 2
        method = 'ml'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_05_varimax_minres_3_factors(self):

        filename = 'test05.csv'
        factors = 3
        method = 'minres'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_06_promax_ml_3_factors(self):

        filename = 'test06.csv'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_06_none_minres_2_factors(self):

        filename = 'test06.csv'
        factors = 2
        method = 'minres'
        rotation = 'none'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)

    def test_07_varimax_ml_3_factors(self):

        filename = 'test07.csv'
        factors = 3
        method = 'minres'
        rotation = 'varimax'

        results = self.do_analysis(filename, factors, method, rotation)
        expected = self.get_expected(filename, factors, method, rotation)

        loadings_p = self.normalize(results['loadings'])
        loadings_r = self.normalize(expected['loadings'])

        assert_frame_equal(loadings_p, loadings_r, check_less_precise=2)
