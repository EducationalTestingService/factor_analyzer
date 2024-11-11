"""Tests for ``Rotator`` class.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""

import unittest

from factor_analyzer.test_utils import check_rotation

# set a threshold of roughly 95 percent matching
THRESHOLD = 0.95


class TestRotator(unittest.TestCase):
    def test_01_varimax_minres_2_factors(self):  # noqa: D103
        test_name = "test01"
        factors = 2
        method = "uls"
        rotation = "varimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_02_varimax_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "varimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_02_oblimax_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "oblimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_02_oblimin_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "oblimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_02_quartimax_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "quartimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_02_quartimin_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "quartimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_04_oblimax_minres_3_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 3
        method = "uls"
        rotation = "oblimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_04_oblimin_minres_3_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 3
        method = "uls"
        rotation = "oblimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_04_quartimax_minres_3_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 3
        method = "uls"
        rotation = "quartimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_04_quartimin_minres_3_factors(self):  # noqa: D103
        test_name = "test04"
        factors = 3
        method = "uls"
        rotation = "quartimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_oblimax_minres_2_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 2
        method = "uls"
        rotation = "oblimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_oblimin_minres_3_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 2
        method = "uls"
        rotation = "oblimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_quartimax_minres_2_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 2
        method = "uls"
        rotation = "quartimax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_quartimin_minres_2_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 2
        method = "uls"
        rotation = "quartimin"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_equamax_minres_2_factors(self):  # noqa: D103
        test_name = "test07"
        factors = 2
        method = "uls"
        rotation = "equamax"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertGreater(check, THRESHOLD)

    def test_07_oblimin_minres_2_factors_gamma(self):  # noqa: D103
        test_name = "test07_gamma"
        factors = 2
        method = "uls"
        rotation = "oblimin"
        gamma = 0.5

        check = check_rotation(test_name, factors, method, rotation, gamma=gamma)
        self.assertGreater(check, THRESHOLD)

    def test_02_geomin_obl_minres_2_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 2
        method = "uls"
        rotation = "geomin_obl"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)

    def test_02_geomin_obl_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "geomin_obl"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)

    def test_02_geomin_obl_ml_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "ml"
        rotation = "geomin_obl"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)

    def test_02_geomin_ort_minres_2_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 2
        method = "uls"
        rotation = "geomin_ort"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)

    def test_02_geomin_ort_minres_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "uls"
        rotation = "geomin_ort"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)

    def test_02_geomin_ort_ml_3_factors(self):  # noqa: D103
        test_name = "test02"
        factors = 3
        method = "ml"
        rotation = "geomin_ort"

        check = check_rotation(test_name, factors, method, rotation)
        self.assertEqual(check, 1)
