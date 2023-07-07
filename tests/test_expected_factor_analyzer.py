"""
Tests for expected factor analysis.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""

import unittest

from factor_analyzer.test_utils import check_scenario

# set a slightly lower threshold than just rotation
# since we have to make sure both the fitting
# and the rotations are matching
THRESHOLD = 0.9


class TestExpectedFactorAnalysis(unittest.TestCase):
    def test_01_none_minres_3_factors_use_corr(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "uls"
        rotation = "none"

        for check in check_scenario(
            test_name, factors, method, rotation, use_corr_matrix=True
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_none_minres_3_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "uls"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_01_none_minres_2_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 2
        method = "uls"
        rotation = "none"

        for check in check_scenario(
            test_name, factors, method, rotation, check_scores=True
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_varimax_minres_2_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 2
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(
            test_name, factors, method, rotation, check_scores=True
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_promax_minres_2_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 2
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
            check_structure=True,
            check_scores=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_varimax_ml_3_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "ml"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_01_promax_ml_3_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "ml"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_promax_ml_3_factors_use_corr(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "ml"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            use_corr_matrix=True,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_01_promax_uls_3_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 3
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_02_none_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_02_varimax_minres_2_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 2
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(
            test_name, factors, method, rotation, check_scores=True
        ):
            self.assertGreater(check, THRESHOLD)

    def test_02_promax_minres_2_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 2
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
            check_structure=True,
            check_scores=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_03_none_minres_3_factors(self):  # noqa: D103
        test_name = "test03"
        factors = 3
        method = "uls"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_03_promax_minres_3_factors(self):  # noqa: D103
        test_name = "test03"
        factors = 3
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_03_none_ml_2_factors(self):  # noqa: D103
        test_name = "test03"
        factors = 2
        method = "ml"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_04_varimax_ml_2_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 2
        method = "ml"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_04_varimax_minres_3_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 3
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_05_varimax_minres_3_factors(self):  # noqa: D103
        test_name = "test05"
        factors = 3
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_05_none_ml_2_factors(self):  # noqa: D103
        test_name = "test05"
        factors = 2
        method = "ml"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_06_promax_ml_3_factors(self):  # noqa: D103
        test_name = "test06"
        factors = 3
        method = "ml"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_06_none_minres_2_factors(self):  # noqa: D103
        test_name = "test06"
        factors = 2
        method = "uls"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_07_varimax_minres_3_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 3
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_07_varimax_ml_3_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 3
        method = "ml"
        rotation = "varimax"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_08_promax_ml_3_factors(self):  # noqa: D103
        test_name = "test08"
        factors = 3
        method = "ml"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_08_none_ml_2_factors(self):  # noqa: D103
        test_name = "test08"
        factors = 2
        method = "ml"
        rotation = "none"

        for check in check_scenario(test_name, factors, method, rotation):
            self.assertGreater(check, THRESHOLD)

    def test_09_promax_ml_3_factors(self):  # noqa: D103
        test_name = "test09"
        factors = 3
        method = "ml"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_09_promax_minres_2_factors(self):  # noqa: D103
        test_name = "test09"
        factors = 2
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_10_none_ml_3_factors(self):  # noqa: D103
        test_name = "test10"
        factors = 3
        method = "ml"
        rotation = "none"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_10_varimax_minres_3_factors(self):  # noqa: D103
        test_name = "test10"
        factors = 3
        method = "uls"
        rotation = "varimax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_10_promax_minres_3_factors(self):  # noqa: D103
        test_name = "test10"
        factors = 3
        method = "uls"
        rotation = "promax"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_02_none_principal(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "principal"
        rotation = "none"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)

    def test_15_none_principal(self):  # noqa: D103
        test_name = "test15"
        factors = 20
        method = "principal"
        rotation = "none"
        svd_method = "lapack"

        for check in check_scenario(
            test_name,
            factors,
            method,
            rotation,
            svd_method=svd_method,
            ignore_value=True,
            ignore_communalities=True,
        ):
            self.assertGreater(check, THRESHOLD)
