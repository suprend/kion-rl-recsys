"""Only synthetic metadata/payloads and mocked subprocess dispatch; no ML imports."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import train_reward_model as adapter
from test_prepare_reward_data import reward_manifest, dump


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "исходники с пробелами"
        for name in adapter.prep.PINS:
            relative = Path(adapter.prep.SOURCE) / "scripts" / (name + ".py")
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(adapter.ROOT / relative, target)
        for relative in (Path(adapter.prep.SERIES_INPUT), Path(adapter.prep.SOURCE).parent / "bundle-rm-r2/scripts/kion_reward_boosting.py"):
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(adapter.ROOT / relative, target)
        self.bundle = self.base / "новый bundle"
        self.bundle.mkdir()
        self.manifest = reward_manifest()
        self.manifest["builder_source_sha256"] = adapter.prep.PINS["kion_reward_data"]
        pins = {"builder": adapter.prep.PINS["kion_reward_data"], "boosting": adapter.prep.PINS["kion_reward_boosting"]}
        pins.update({"component/" + name: sha for name, sha in adapter.prep.PINS.items()
                     if name not in ("kion_reward_data", "kion_reward_boosting")})
        pins.update({name: adapter.prep.INPUT_HASHES[name] for name in ("metadata", "users", "items", "interactions")})
        self.manifest["catalog"]["provenance"]["source_pins"] = pins
        for field in ("files", "sidecars"):
            for name in self.manifest[field]:
                (self.bundle / name).write_bytes(b"synthetic, not arrays")
                self.manifest[field][name] = adapter.prep.digest(self.bundle / name)
        self.bundle_sha = dump(self.bundle / "manifest.json", self.manifest)
        self.kwargs = dict(stage="fit-r1", bundle=self.bundle, output=self.base / "fresh scorer",
                           python=sys.executable, root=self.root)

    def scorer(self, output, r2=False, previous=None):
        output.mkdir()
        for name in ("scorer.npz", "model.cbm"):
            (output / name).write_bytes(b"mock " + name.encode())
        manifest = {"contract": "kion-reward-boosting-v1", "backend": "catboost_cpu",
                    "score_id_space": "original_item_id", "catalog": {"mapping_sha256": adapter.prep.MAPPING_SHA},
                    "source_sha256": adapter.R2_SHA if r2 else adapter.prep.PINS["kion_reward_boosting"],
                    "config": adapter.CONFIGS[1], "report": {"bundle_manifest_sha256": self.bundle_sha},
                    "files": {name: adapter.prep.digest(output / name) for name in ("scorer.npz", "model.cbm")}}
        if r2:
            manifest["report"]["calibration_repair"] = {"legacy_signature": previous,
                "model_weights_copied_byte_for_byte": True, "catboost_fitted": False, "calibration_support_only": True}
        manifest["signature"] = adapter.canonical_hash(manifest)
        dump(output / "manifest.json", manifest)
        return manifest

    def test_plan_no_subprocess_payload_hash_or_scientific_import(self):
        before = set(sys.modules)
        with patch.object(adapter.prep, "run_child", side_effect=AssertionError("spawn")), patch.object(adapter.prep, "digest", side_effect=AssertionError("payload read")):
            result = adapter.plan(**self.kwargs)
        self.assertFalse({"numpy", "torch", "catboost", "scipy"} & (set(sys.modules) - before))
        self.assertFalse(result["input_content_verified"])
        self.assertFalse(self.kwargs["output"].exists())
        self.assertEqual(result["argv"][-4:], ["--backend", "catboost", "--seed", "2026"])

    def test_dynamic_repair_signature_and_exact_bundle_binding(self):
        old = self.base / "newly trained"
        manifest = self.scorer(old)
        args = dict(self.kwargs, stage="repair-r2", old_artifact=old)
        result = adapter.plan(**args)
        self.assertEqual(result["expected_old_signature"], manifest["signature"])
        self.assertNotEqual(result["expected_old_signature"], adapter.HISTORICAL_R1)
        self.assertIn(manifest["signature"], result["argv"])
        with self.assertRaises(ValueError):
            adapter.plan(**dict(args, lineage="historical-artifact"))
        manifest["report"]["bundle_manifest_sha256"] = "a" * 64
        manifest.pop("signature")
        manifest["signature"] = adapter.canonical_hash(manifest)
        dump(old / "manifest.json", manifest)
        with self.assertRaises(ValueError):
            adapter.plan(**args)

    def test_fit_dispatch_mock_and_repair_mock(self):
        def fit_child(argv, timeout, **kwargs):
            if "-c" in argv:
                return Mock(stdout=json.dumps(adapter.prep.ENVIRONMENT))
            self.scorer(self.kwargs["output"])
            return Mock(returncode=0)
        with patch.object(adapter.prep, "run_child", side_effect=fit_child) as child:
            result = adapter.execute(**self.kwargs, execute_training=True)
        self.assertEqual(child.call_count, 2)
        self.assertEqual(result["status"], "source-cli-completed-output-checksums-verified")
        self.assertFalse(result["historical_result_equivalence_verified"])
        old = self.kwargs["output"]
        repaired = self.base / "repaired"
        def repair_child(argv, timeout, **kwargs):
            if "-c" in argv:
                return Mock(stdout=json.dumps(adapter.prep.ENVIRONMENT))
            self.scorer(repaired, True, result["output_signature"])
            return Mock(returncode=0)
        with patch.object(adapter.prep, "run_child", side_effect=repair_child):
            done = adapter.execute(**dict(self.kwargs, stage="repair-r2", old_artifact=old, output=repaired), execute_repair=True)
        self.assertEqual(done["status"], result["status"])

    def test_failed_timeout_cancel_never_success(self):
        for index, error in enumerate((subprocess.CalledProcessError(7, "mock"), subprocess.TimeoutExpired("mock", 1), KeyboardInterrupt())):
            output = self.base / f"failed{index}"
            with patch.object(adapter.prep, "run_child", side_effect=[Mock(stdout=json.dumps(adapter.prep.ENVIRONMENT)), error]):
                with self.assertRaises(type(error)):
                    adapter.execute(**dict(self.kwargs, output=output), execute_training=True)
            receipt, _ = adapter.prep.read_json(Path(str(output) + ".rm-launch.json"))
            self.assertEqual(receipt["status"], "failed-or-cancelled")

    def test_gate_and_paths(self):
        with patch.object(adapter.prep, "run_child", side_effect=AssertionError("spawn")):
            for changes in ({}, {"execute_repair": True}, {"execute_training": True, "execute_repair": True}):
                with self.assertRaises(ValueError):
                    adapter.execute(**self.kwargs, **changes)
            for output in (self.bundle, self.bundle / "nested", self.root / "nested"):
                with self.assertRaises(ValueError):
                    adapter.plan(**dict(self.kwargs, output=output))

    def test_missing_payload_and_bad_manifest(self):
        name = next(iter(self.manifest["files"]))
        (self.bundle / name).unlink()
        with self.assertRaises(ValueError):
            adapter.plan(**self.kwargs)
        (self.bundle / name).write_bytes(b"synthetic, not arrays")
        self.manifest["exclusions"]["row_caps"]["fit"] = 10
        dump(self.bundle / "manifest.json", self.manifest)
        with self.assertRaises(ValueError):
            adapter.plan(**self.kwargs)

    def test_source_and_input_drift(self):
        request = adapter.plan(**self.kwargs)
        (self.bundle / "catalog.npz").write_bytes(b"drift")
        with self.assertRaises(ValueError):
            adapter.check_inputs(request)
        script = self.root / adapter.prep.SOURCE / "scripts/kion_reward_boosting.py"
        script.write_text("# changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            adapter.plan(**self.kwargs)

    def test_exit_zero_without_artifacts_fails(self):
        with patch.object(adapter.prep, "run_child", return_value=Mock(stdout=json.dumps(adapter.prep.ENVIRONMENT))):
            with self.assertRaises(ValueError):
                adapter.execute(**self.kwargs, execute_training=True)

    def test_large_rm_manifest_plan_and_execution_checks(self):
        # Valid production-shaped manifest plus large feature provenance. Only
        # descriptor JSON grows; synthetic array bytes are never read by plan.
        self.manifest["catalog"]["provenance"]["feature_descriptors"] = "x" * (adapter.prep.LIMIT + 1)
        self.bundle_sha = dump(self.bundle / "manifest.json", self.manifest)
        with patch.object(adapter.prep, "run_child", side_effect=AssertionError("spawn")), patch.object(adapter.prep, "digest", side_effect=AssertionError("payload read")):
            request = adapter.plan(**self.kwargs)
        adapter.check_inputs(request)
        old = self.base / "large scorer"
        manifest = self.scorer(old)
        manifest.pop("signature")
        manifest["catalog"]["provenance"] = self.manifest["catalog"]["provenance"]
        manifest["signature"] = adapter.canonical_hash(manifest)
        dump(old / "manifest.json", manifest)
        request = adapter.plan(**dict(self.kwargs, stage="repair-r2", old_artifact=old))
        adapter.check_inputs(request)
        self.assertEqual(request["expected_old_signature"], manifest["signature"])

    def test_rm_bound_does_not_relax_sources_and_rejects_bad_json(self):
        path = self.base / "manifest.json"
        for contents in ('{"a":1,"a":2}', '[]', '{broken'):
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                adapter.read_rm_manifest(path)
        path.write_bytes(b" " * 101)
        with patch.object(adapter, "RM_MANIFEST_LIMIT", 100):
            with self.assertRaises(ValueError):
                adapter.read_rm_manifest(path)
        path.write_bytes(b" " * (adapter.prep.LIMIT + 1))
        with self.assertRaises(ValueError):
            adapter.prep.bounded_bytes(path)


if __name__ == "__main__":
    unittest.main()
