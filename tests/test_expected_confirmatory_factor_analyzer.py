"""
Tests for ``ConfirmatoryFactorAnalyzer`` class.

:author: Jeremy Biggs (jeremy.m.biggs@gmail.com)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: Educational Testing Service
:date: 2022-09-05
"""
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from factor_analyzer.confirmatory_factor_analyzer import (
    ConfirmatoryFactorAnalyzer,
    ModelSpecification,
    ModelSpecificationParser,
)
from factor_analyzer.test_utils import check_cfa

THRESHOLD = 1.0


class TestConfirmatoryFactorAnalysis(unittest.TestCase):
    def test_11_cfa(self):  # noqa: D103
        json_name_input = "test11"
        data_name_input = "test11"

        for check in check_cfa(json_name_input, data_name_input, rel_tol=0.1):
            self.assertGreaterEqual(check, THRESHOLD)

    def test_12_cfa(self):  # noqa: D103
        json_name_input = "test12"
        data_name_input = "test12"

        for check in check_cfa(json_name_input, data_name_input, abs_tol=0.05):
            self.assertGreaterEqual(check, THRESHOLD)

    def test_13_cfa(self):  # noqa: D103
        json_name_input = "test13"
        data_name_input = "test13"

        for check in check_cfa(
            json_name_input,
            data_name_input,
            index_col=0,
            is_cov=True,
            n_obs=64,
            rel_tol=0.1,
        ):
            self.assertGreaterEqual(check, THRESHOLD)

    def test_14_cfa(self):  # noqa: D103
        json_name_input = "test14"
        data_name_input = "test14"

        for check in check_cfa(
            json_name_input, data_name_input, index_col=0, rel_tol=0.1
        ):
            self.assertGreaterEqual(check, THRESHOLD)

    def test_14_cfa_no_model(self):  # noqa: D103
        X = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(ValueError):
            cfa = ConfirmatoryFactorAnalyzer("string_not_model")
            cfa.fit(X)

    def test_14_cfa_bad_bounds(self):  # noqa: D103
        X = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])

        with self.assertRaises(AssertionError):
            cfa = ConfirmatoryFactorAnalyzer(bounds=[(0, 1)])
            cfa.fit(X)

    def test_14_cfa_cov_with_no_obs(self):  # noqa: D103
        with self.assertRaises(ValueError):
            ConfirmatoryFactorAnalyzer(is_cov_matrix=True)


class TestModelSpecificationParser(unittest.TestCase):
    def test_model_spec_str(self):  # noqa: D102
        ms = ModelSpecification(np.array([[0, 0, 0]]), 3, 1)
        self.assertTrue(str(ms).startswith("<ModelSpecification object at "))

    def test_model_spec_as_dict(self):  # noqa: D102
        loadings = np.array([[0, 0, 0]])
        n_factors = 3
        n_variables = 1
        ms = ModelSpecification(loadings, n_factors, n_variables)

        expected = {
            "loadings": loadings,
            "n_variables": n_variables,
            "n_factors": n_factors,
        }
        new_dict = ms.get_model_specification_as_dict()
        for key, value in expected.items():
            assert key in new_dict
            if isinstance(value, np.ndarray):
                assert_array_equal(new_dict[key], value)
            else:
                self.assertEqual(new_dict[key], value)

    def test_model_spec_parser_from_dict_none(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        ms = ModelSpecificationParser.parse_model_specification_from_dict(X, None)
        self.assertTrue(isinstance(ms, ModelSpecification))
        self.assertEqual(ms.n_factors, 3)
        self.assertEqual(ms.n_variables, 3)
        assert_array_equal(ms.loadings, np.ones((3, 3), dtype=int))

    def test_model_spec_parser_from_dict_error(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        with self.assertRaises(ValueError):
            ModelSpecificationParser.parse_model_specification_from_dict(
                X, "not_a_model"
            )

    def test_model_spec_parser_from_array_none(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        ms = ModelSpecificationParser.parse_model_specification_from_array(X, None)
        self.assertTrue(isinstance(ms, ModelSpecification))
        self.assertEqual(ms.n_factors, 3)
        self.assertEqual(ms.n_variables, 3)
        assert_array_equal(ms.loadings, np.ones((3, 3), dtype=int))

    def test_model_spec_parser_from_array(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        spec = np.ones((3, 3), dtype=int)
        ms = ModelSpecificationParser.parse_model_specification_from_array(X, spec)
        self.assertTrue(isinstance(ms, ModelSpecification))
        self.assertEqual(ms.n_factors, 3)
        self.assertEqual(ms.n_variables, 3)
        assert_array_equal(ms.loadings, np.ones((3, 3), dtype=int))

    def test_model_spec_parser_from_frame(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        spec = pd.DataFrame(np.ones((3, 3), dtype=int))
        ms = ModelSpecificationParser.parse_model_specification_from_array(X, spec)
        self.assertTrue(isinstance(ms, ModelSpecification))
        self.assertEqual(ms.n_factors, 3)
        self.assertEqual(ms.n_variables, 3)
        assert_array_equal(ms.loadings, np.ones((3, 3), dtype=int))

    def test_model_spec_parser_from_array_error(self):  # noqa: D102
        X = np.array([[0, 0, 0]])
        with self.assertRaises(ValueError):
            ModelSpecificationParser.parse_model_specification_from_array(
                X, "not_a_model"
            )
