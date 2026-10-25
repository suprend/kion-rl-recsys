"""Stdlib-only contract/semantic tests; never import numerical backends."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


rm = module(ROOT / "kion_replay/rm.py", "_synthetic_portable_rm")
exporter = module(ROOT / "export_rm_fixture.py", "_synthetic_rm_exporter")


class FrozenRMContract(unittest.TestCase):
    def test_clip_precedes_affine_for_negative_tree_prediction(self):
        self.assertEqual(rm.calibrate_scalar(-1., 1, slope=.5, intercept=.1), .1)

    def test_clip_precedes_affine_for_large_tree_prediction(self):
        self.assertEqual(rm.calibrate_scalar(2., 1, slope=.5, intercept=-.1), .4)

    def test_cold_fallback_is_immune_to_calibration(self):
        self.assertEqual(rm.calibrate_scalar(2., 0, slope=2., intercept=1., global_mean=.4), .4)

    def test_nonfinite_tree_prediction_fails(self):
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            rm.calibrate_scalar(float("nan"), 1)

    def test_negative_or_boolean_count_fails(self):
        for count in (-1, True):
            with self.subTest(count=count), self.assertRaises(ValueError):
                rm.calibrate_scalar(.5, count)

    def test_export_cannot_run_without_external_guard(self):
        with self.assertRaisesRegex(ValueError, "resource guard"):
            exporter.run({})

    def test_export_rejects_changed_state_selection(self):
        with self.assertRaisesRegex(ValueError, "first four"):
            exporter.validate_request({"contract": exporter.CONTRACT, "row_positions": [1, 2, 3, 4]})

    def test_new_sources_parse_and_import_without_numerical_backends(self):
        for path in (ROOT / "kion_replay/rm.py", ROOT / "export_rm_fixture.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertFalse(any(name == "numpy" or name.startswith("numpy.") or name == "catboost"
                             or name.startswith("catboost.") or name == "torch" or name.startswith("torch.")
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
