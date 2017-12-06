"""
Tests for FactorAnalyzer class

:author: Jeremy Biggs (jbiggs@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import os
import math
import numpy as np
import pandas as pd

from factor_analyzer import FactorAnalyzer


class TestFactorAnalyzerVsR:

    def __init__(self):

        self.data_dir = 'tests/data'
        self.exp_dir = 'tests/expected'
        self.threshold = 0.9

    @staticmethod
    def do_analysis(top_dir, test_name, factors, method, rotation):
        """
        Use the `FactorAnalyzer()` class to perform the factor analysis
        and return a dictionary with relevant results for given scenario.
        """
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

    @staticmethod
    def get_r_output(top_dir, test_name, factors, method, rotation):
        """
        Get the R output for the given scenario.
        """
        output = {}
        for output_type in ['value',
                            'evalues',
                            'loading',
                            'uniquenesses',
                            'communalities']:

            filename = '{}_{}_{}_{}_{}.csv'.format(output_type,
                                                   method,
                                                   rotation,
                                                   factors,
                                                   test_name)

            filename = os.path.join(top_dir, test_name, filename)

            data = pd.read_csv(filename)
            output[output_type] = data

        return output

    @staticmethod
    def normalize(data):
        """
        Normalize the data.
        """
        # check for possible index column
        possible_index = [col for col in data.columns if 'Unnamed' in col]

        # get numeric columns
        numeric_cols = [col for col in data.dtypes[data.dtypes != 'object'].index.values
                        if col not in possible_index]

        # take absolute value
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

    def check_close(self, data1, data2):
        """
        Check to make sure all values are close.
        """
        data1 = self.normalize(data1)
        data2 = self.normalize(data2)

        assert data1.shape == data2.shape

        arr = np.empty(shape=data1.shape, dtype=bool)
        for i in range(data1.shape[0]):
            for j in range(data2.shape[1]):
                check = math.isclose(data1.iloc[i, j],
                                     data2.iloc[i, j],
                                     rel_tol=0,
                                     abs_tol=0.1)
                arr[i, j] = check

        check = arr.sum(None) / arr.size
        print(check)
        return check

    def check_all(self,
                  test_name,
                  factors,
                  method,
                  rotation,
                  ignore_value=False,
                  ignore_communalities=False):
        """
        Check all results for given scenario.
        """

        output_types = ['loading', 'evalues']

        if not ignore_value:
            output_types.extend(['value'])

        if not ignore_communalities:
            output_types.extend(['uniquenesses',
                                 'communalities'])

        results1 = self.get_r_output(self.exp_dir, test_name, factors, method, rotation)
        results2 = self.do_analysis(self.data_dir, test_name, factors, method, rotation)

        for output_type in output_types:

            data1 = results1[output_type]
            data2 = results2[output_type]

            print(output_type)
            yield self.check_close(data1, data2)

    def test_01_none_minres_3_factors(self):

        test_name = 'test01'
        factors = 3
        method = 'uls'
        rotation = 'none'

        for check in self.check_all(test_name, factors, method, rotation):
            assert check > self.threshold

    def test_01_none_minres_2_factors(self):

        test_name = 'test01'
        factors = 2
        method = 'uls'
        rotation = 'none'

        for check in self.check_all(test_name, factors, method, rotation):
            assert check > self.threshold

    def test_01_varimax_ml_3_factors(self):

        test_name = 'test01'
        factors = 3
        method = 'ml'
        rotation = 'varimax'

        for check in self.check_all(test_name, factors, method, rotation):
            assert check > self.threshold

    def test_01_promax_ml_3_factors(self):

        test_name = 'test01'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_01_promax_uls_3_factors(self):

        test_name = 'test01'
        factors = 3
        method = 'uls'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_02_none_minres_3_factors(self):

        test_name = 'test02'
        factors = 3
        method = 'uls'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_02_varimax_minres_2_factors(self):

        test_name = 'test02'
        factors = 2
        method = 'uls'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_03_none_minres_3_factors(self):

        test_name = 'test03'
        factors = 3
        method = 'uls'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_03_promax_minres_3_factors(self):

        test_name = 'test03'
        factors = 3
        method = 'uls'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_03_none_ml_2_factors(self):

        test_name = 'test03'
        factors = 2
        method = 'ml'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_04_varimax_ml_2_factors(self):

        test_name = 'test04'
        factors = 2
        method = 'ml'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_04_varimax_minres_3_factors(self):

        test_name = 'test04'
        factors = 3
        method = 'uls'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_05_varimax_minres_3_factors(self):

        test_name = 'test05'
        factors = 3
        method = 'uls'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_05_none_ml_2_factors(self):

        test_name = 'test05'
        factors = 2
        method = 'ml'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_06_promax_ml_3_factors(self):

        test_name = 'test06'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_06_none_minres_2_factors(self):

        test_name = 'test06'
        factors = 2
        method = 'uls'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_07_varimax_minres_3_factors(self):

        test_name = 'test07'
        factors = 3
        method = 'uls'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_07_varimax_ml_3_factors(self):

        test_name = 'test07'
        factors = 3
        method = 'ml'
        rotation = 'varimax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_08_promax_ml_3_factors(self):

        test_name = 'test08'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_08_none_ml_2_factors(self):

        test_name = 'test08'
        factors = 2
        method = 'ml'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation):
            assert check > self.threshold

    def test_09_promax_ml_3_factors(self):

        test_name = 'test09'
        factors = 3
        method = 'ml'
        rotation = 'promax'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_09_promax_minres_2_factors(self):

        test_name = 'test09'
        factors = 2
        method = 'uls'
        rotation = 'promax'
        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_10_none_ml_3_factors(self):

        test_name = 'test10'
        factors = 3
        method = 'ml'
        rotation = 'none'

        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_10_varimax_minres_3_factors(self):

        test_name = 'test10'
        factors = 3
        method = 'uls'
        rotation = 'varimax'
        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold

    def test_10_promax_minres_3_factors(self):

        test_name = 'test10'
        factors = 3
        method = 'uls'
        rotation = 'promax'
        for check in self.check_all(test_name,
                                    factors,
                                    method,
                                    rotation,
                                    ignore_value=True,
                                    ignore_communalities=True):
            assert check > self.threshold
