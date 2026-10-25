"""Stdlib-only contract/semantic tests; never import numerical backends."""
from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from kion_replay import verify as verifier

ROOT = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


rm = module(ROOT / "kion_replay/rm.py", "_synthetic_portable_rm")
exporter = module(ROOT / "export_rm_fixture.py", "_synthetic_rm_exporter")


class FrozenRMContract(unittest.TestCase):
    def assert_preflight_metadata(self, manifest, error=None):
        """Mock payload I/O, not metadata predicates; no NPY/CBM integrity claim."""
        with patch.object(verifier, "read_json", return_value=manifest), \
                patch.object(verifier, "verify_array") as arrays, \
                patch.object(verifier, "verify_descriptor") as model, \
                patch.object(verifier, "sha256", return_value="mocked-manifest-hash"):
            if error is not None:
                with self.assertRaisesRegex(ValueError, error):
                    verifier.verify_rm_fixture(ROOT / "inputs-rm-r1")
                arrays.assert_not_called()
                model.assert_not_called()
            else:
                actual, receipt = verifier.verify_rm_fixture(ROOT / "inputs-rm-r1")
                self.assertIs(actual, manifest)
                self.assertEqual(arrays.call_count, 8)
                model.assert_called_once()
                self.assertFalse(receipt["numerical_execution"])

    def test_preflight_accepts_frozen_metadata_with_mocked_payload(self):
        manifest = json.loads((ROOT / "inputs-rm-r1/manifest.json").read_text(encoding="utf-8"))
        rm.validate_metadata(manifest["rm"])
        self.assert_preflight_metadata(manifest)

    def test_preflight_rejects_metadata_mismatches_before_payload(self):
        baseline = json.loads((ROOT / "inputs-rm-r1/manifest.json").read_text(encoding="utf-8"))
        mutations = (
            (("calibration", "slope"), 1.0248506983671792, "Frozen calibration/fallback differs"),
            (("calibration", "intercept"), .004431619250975063, "Frozen calibration/fallback differs"),
            (("global_mean",), .48802770002556967, "Frozen calibration/fallback differs"),
            (("row_positions",), [1, 2, 3, 4], "Fixed four-state selection required"),
            (("scorer_signature",), "0" * 64, "Frozen RM provenance differs"),
            (("schema_signature",), "0" * 64, "Frozen RM provenance differs"),
            (("mapping_sha256",), "0" * 64, "Frozen RM provenance differs"),
            (("prior_archive_sha256",), "0" * 64, "Frozen RM provenance differs"),
            (("context_width",), 2078, "Exact RM2079\\+2008 schema required"),
        )
        for path, replacement, error in mutations:
            with self.subTest(field="rm." + ".".join(path)):
                manifest = deepcopy(baseline)
                target = manifest["rm"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement
                with self.assertRaisesRegex(ValueError, error):
                    rm.validate_metadata(manifest["rm"])
                self.assert_preflight_metadata(manifest, error)

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
        for path in (ROOT / "kion_replay/rm.py", ROOT / "kion_replay/verify.py", ROOT / "export_rm_fixture.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertFalse(any(name == "numpy" or name.startswith("numpy.") or name == "catboost"
                             or name.startswith("catboost.") or name == "torch" or name.startswith("torch.")
                             for name in sys.modules))


if __name__ == "__main__":
    unittest.main()
