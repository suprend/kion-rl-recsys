"""Standard-library contracts; no scientific imports, real preparation or training."""
import argparse
import ast
from contextlib import ExitStack, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

import prepare_reward_data as adapter


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reward_manifest():
    return {"contract": "kion-reward-boosting-v1", "layout": "npy-directory-v1", "ready": True,
            "partition": {"salt": "kion-rm-select-calibration-v1-20261020"},
            "split": {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08", "heldout_history": "frozen<2021-08-02"},
            "catalog": {"mapping_sha256": adapter.MAPPING_SHA, "provenance": {"fit_cutoff_exclusive": "2021-08-02", "candidate_features_dynamic": True, "static_candidate_fallback": False}},
            "exclusions": {"row_caps": dict(adapter.CAPS), "role_rows": dict(adapter.CAPS), "missing_imputed_as_zero": False, "unobserved_pairs_generated": 0},
            "files": {name: "a" * 64 for name in ({f"{r}.{f}.npy" for r in adapter.CAPS for f in adapter.FIELDS} | {"catalog.npz"})},
            "sidecars": {name: "b" * 64 for name in ({f"{r}.{f}.npy" for r in adapter.CAPS for f in ("x", "lengths")} | {"full_target_support.npz"})},
            "row_bindings": {r: "c" * 64 for r in adapter.CAPS}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory())).resolve()
        self.root = self.temp / "код с пробелами"
        self.source = self.root / adapter.SOURCE
        for relative in [f"scripts/{n}.py" for n in adapter.PINS] + [adapter.SERIES]:
            target = self.root / adapter.SERIES_INPUT if relative == adapter.SERIES else self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            origin = adapter.ROOT / adapter.SERIES_INPUT if relative == adapter.SERIES else adapter.ROOT / adapter.SOURCE / relative
            shutil.copyfile(origin, target)
        self.raw = self.temp / "raw"
        self.prepared = self.temp / "prepared"
        self.basic = self.temp / "basic"
        for folder in (self.raw, self.prepared, self.basic):
            folder.mkdir()
        hashes = {}
        for name in ("users", "items", "interactions"):
            path = self.raw / f"{name}.csv"
            path.write_bytes(b"synthetic fixture, not production\n")
            hashes[name] = adapter.digest(path)
        for name in ("train", "val"):
            path = self.prepared / f"{name}.npz"
            path.write_bytes(b"mock container")
            hashes[name] = adapter.digest(path)
        mapping = {"123": 3}
        mapping_hash = hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        hashes["metadata"] = dump(self.prepared / "metadata.json", {"item_id_to_token": mapping})
        self.stack.enter_context(patch.object(adapter, "INPUT_HASHES", hashes))
        self.stack.enter_context(patch.object(adapter, "MAPPING_SHA", mapping_hash))
        (self.basic / "features.npz").write_bytes(b"mock basic")
        self.basic_manifest = {"builder_contract": "kion-basic-bundle-v1", "vocabulary": {"sha256": mapping_hash},
            "inputs": {k: {"sha256": v} for k, v in hashes.items()},
            "features": {"file": "features.npz", "sha256": adapter.digest(self.basic / "features.npz"),
                         "arrays": {"item_features": {"sha256": adapter.ITEM_ARRAY_SHA}}}}
        dump(self.basic / "manifest.json", self.basic_manifest)
        self.kwargs = dict(stage="reward-data", raw=self.raw, prepared=self.prepared, basic=self.basic,
                           output=self.temp / "new output", python=sys.executable, root=self.root)

    def test_plan_is_stat_only_no_subprocess_or_scientific_import(self):
        before = set(sys.modules)
        with patch.object(adapter.subprocess, "run", side_effect=AssertionError("spawn")), patch.object(adapter, "digest", side_effect=AssertionError("payload read")):
            plan = adapter.plan(**self.kwargs)
        self.assertFalse(plan["input_content_verified"])
        self.assertFalse(plan["environment_verified"])
        self.assertTrue(all(not x["content_verified"] for x in plan["inputs"]))
        self.assertFalse({"numpy", "torch", "catboost", "scipy"} & (set(sys.modules) - before))
        self.assertEqual(plan["argv"][-8:], ["--max-fit", "100000", "--max-select", "20000", "--max-calibration", "20000", "--batch-size", "512"])
        self.assertIn((self.prepared / "metadata.json").as_posix(), plan["argv"])
        self.assertFalse(self.kwargs["output"].exists())

    def test_basic_exact_argv(self):
        args = dict(self.kwargs, stage="basic-features", basic=None)
        plan = adapter.plan(**args)
        self.assertEqual(plan["argv"][3:], ["--raw", plan["raw_runtime_root"], "--prepared", self.prepared.as_posix(), "--output", args["output"].as_posix()])
        self.assertEqual({Path(x["path"]).name for x in plan["inputs"]}, {"users.csv", "items.csv", "train.npz", "val.npz"})

    def test_every_pin_and_series_rechecked(self):
        for relative in [f"scripts/{n}.py" for n in adapter.PINS] + [adapter.SERIES]:
            with self.subTest(relative=relative):
                path = self.root / adapter.SERIES_INPUT if relative == adapter.SERIES else self.source / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(ValueError, "source SHA256"):
                    adapter.plan(**self.kwargs)
                path.write_bytes(original)

    def test_missing_and_existing_and_overlapping_output(self):
        for output in (self.raw, self.raw / "nested", self.root / "nested", self.prepared.parent, self.basic / "new"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                adapter.plan(**dict(self.kwargs, output=output))
        (self.raw / "users.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing/empty"):
            adapter.plan(**self.kwargs)

    def test_malformed_oversized_and_mismatched_metadata(self):
        path = self.basic / "manifest.json"
        for content in ("{bad", "[]", '{"x":1,"x":2}', " " * (adapter.LIMIT + 1)):
            path.write_text(content, encoding="utf-8")
            with self.assertRaises(ValueError):
                adapter.plan(**self.kwargs)
        for mutation in (lambda m: m["vocabulary"].update(sha256="a" * 64), lambda m: m["inputs"]["items"].update(sha256="a" * 64)):
            manifest = copy.deepcopy(self.basic_manifest)
            mutation(manifest)
            dump(path, manifest)
            with self.assertRaises(ValueError):
                adapter.plan(**self.kwargs)
        dump(path, self.basic_manifest)
        dump(self.raw / "manifest.json", {"files": []})
        with self.assertRaisesRegex(ValueError, "Raw manifest"):
            adapter.plan(**self.kwargs)

    def test_symlink_or_junction_component_rejected(self):
        # Portable mocked reparse-point check, no elevated symlink privilege needed.
        original = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda p: p == self.raw or original(p)):
            with self.assertRaisesRegex(ValueError, "Symlink/junction"):
                adapter.plan(**self.kwargs)

    def test_execute_rejects_payload_or_environment_before_dispatch(self):
        (self.raw / "users.csv").write_bytes(b"changed")
        with patch.object(adapter, "run_child") as child, self.assertRaisesRegex(ValueError, "Input SHA256"):
            adapter.execute(**self.kwargs)
        child.assert_not_called()
        (self.raw / "users.csv").write_bytes(b"synthetic fixture, not production\n")
        with patch.object(adapter, "run_child", return_value=Mock(stdout="{}")) as child, self.assertRaisesRegex(ValueError, "environment"):
            adapter.execute(**self.kwargs)
        self.assertEqual(child.call_count, 1)

    def test_child_failure_timeout_cancellation_all_leave_failed_receipt(self):
        probe = Mock(stdout=json.dumps(adapter.ENVIRONMENT))
        for index, error in enumerate((subprocess.CalledProcessError(3, []), subprocess.TimeoutExpired([], 1), KeyboardInterrupt())):
            args = dict(self.kwargs, output=self.temp / f"failure{index}")
            with self.subTest(error=type(error)), patch.object(adapter, "run_child", side_effect=[probe, error]), self.assertRaises(type(error)):
                adapter.execute(**args)
            receipt = json.loads(Path(str(args["output"]) + ".prepare-launch.json").read_text())
            self.assertEqual(receipt["status"], "failed-or-cancelled")

    def test_mock_success_requires_output_and_detects_postrun_drift(self):
        probe = Mock(stdout=json.dumps(adapter.ENVIRONMENT))
        with patch.object(adapter, "run_child", side_effect=[probe, Mock()]), self.assertRaises(ValueError):
            adapter.execute(**self.kwargs)
        args = dict(self.kwargs, output=self.temp / "drift")
        def child(argv, *a, **kw):
            if "-c" in argv:
                return probe
            (self.raw / "users.csv").write_bytes(b"drift")
            return Mock()
        with patch.object(adapter, "run_child", side_effect=child), self.assertRaisesRegex(ValueError, "Input SHA256"):
            adapter.execute(**args)

    def test_successful_mock_dispatch_is_not_numerical_parity(self):
        args = dict(self.kwargs, stage="basic-features", basic=None)
        def child(argv, *a, **kw):
            if "-c" in argv:
                return Mock(stdout=json.dumps(adapter.ENVIRONMENT))
            args["output"].mkdir()
            (args["output"] / "features.npz").write_bytes(b"mock basic")
            dump(args["output"] / "manifest.json", self.basic_manifest)
            return Mock()
        with patch.object(adapter, "run_child", side_effect=child) as mocked:
            result = adapter.execute(**args)
        self.assertEqual(mocked.call_count, 2)
        self.assertFalse(result["semantic_output_equivalence_verified"])
        self.assertEqual(result["status"], "source-cli-completed-output-checksums-verified")
        adapter.check_runtime(result)
        self.assertTrue((Path(result["runtime_root"]) / adapter.SERIES).is_file())
        tree = ast.parse((self.source / "scripts/kion_pack_dynamic.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_series_path")
        scope = {"Path": Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "retained-series-lookup", "exec"), scope)
        self.assertEqual(scope["_series_path"](result["raw_runtime_root"], {}), Path(result["runtime_root"]) / adapter.SERIES)

    def test_invalid_timeout_and_unsupported_caps_rejected(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                adapter.execute(**self.kwargs, timeout_seconds=value)
        with redirect_stdout(io.StringIO()), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
            adapter.main(["reward-data", "--max-fit", "1"])

    def test_reward_manifest_closure_and_semantics(self):
        valid = reward_manifest()
        adapter.validate_reward_manifest(valid)
        for key in ("files", "sidecars", "row_bindings"):
            changed = copy.deepcopy(valid)
            changed[key].pop(next(iter(changed[key])))
            with self.assertRaises(ValueError):
                adapter.validate_reward_manifest(changed)

        for path, value in ((["exclusions", "row_caps", "fit"], 1), (["catalog", "mapping_sha256"], "a" * 64), (["catalog", "provenance", "candidate_features_dynamic"], False), (["catalog", "provenance", "fit_cutoff_exclusive"], "2021-08-09")):
            changed = copy.deepcopy(valid)
            target = changed
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            with self.assertRaises(ValueError):
                adapter.validate_reward_manifest(changed)

    def test_reward_output_manifest_has_separate_large_bound(self):
        manifest = reward_manifest()
        manifest["catalog"]["provenance"]["feature_descriptors"] = "x" * (adapter.LIMIT + 1)
        def child(argv, timeout, **kwargs):
            if "-c" in argv:
                return Mock(stdout=json.dumps(adapter.ENVIRONMENT))
            output = self.kwargs["output"]
            output.mkdir()
            for group in ("files", "sidecars"):
                for name in manifest[group]:
                    (output / name).write_bytes(b"synthetic")
                    manifest[group][name] = adapter.digest(output / name)
            dump(output / "manifest.json", manifest)
            return Mock(returncode=0)
        with patch.object(adapter, "run_child", side_effect=child):
            result = adapter.execute(**self.kwargs)
        self.assertEqual(result["status"], "source-cli-completed-output-checksums-verified")
        path = self.kwargs["output"] / "manifest.json"
        with self.assertRaises(ValueError):
            adapter.read_json(path)
        value, _ = adapter.read_json(path, limit=adapter.RM_MANIFEST_LIMIT)
        adapter.validate_reward_manifest(value)
    def test_retained_cli_parsing_and_signatures_without_importing_source(self):
        # Execute only the real argparse main AST; all builders replaced with mocks.
        for name, builder, arguments in (
                ("kion_basic_bundle_features", "build", ["--raw", "сырьё", "--prepared", "prepared", "--output", "out"]),
                ("kion_reward_data", "build_reward_data", ["--raw", "сырьё", "--metadata", "prepared/metadata.json", "--basic-npz", "basic/features.npz", "--output", "out", "--max-fit", "100000", "--max-select", "20000", "--max-calibration", "20000", "--batch-size", "512"])):
            tree = ast.parse((self.source / "scripts" / f"{name}.py").read_text(encoding="utf-8"))
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
            build = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == builder)
            expected_args = ["raw", "prepared", "output"] if builder == "build" else ["raw_dir", "metadata_path", "basic_npz", "output"]
            self.assertEqual([a.arg for a in build.args.args], expected_args)
            mocked = Mock(return_value={"coverage": {}, "timing": {}, "exclusions": {"role_rows": {}, "target_support": {}}})
            scope = {"argparse": argparse, "Path": Path, "json": json, "__doc__": "test", "DEFAULT_CAPS": adapter.CAPS, builder: mocked,
                     "local": Mock(file_hash=lambda p: "mock")}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(name), "exec"), scope)
            with redirect_stdout(io.StringIO()):
                scope["main"](arguments)
            self.assertEqual(mocked.call_args.args[0], Path("сырьё"))
            if builder == "build_reward_data":
                self.assertEqual(mocked.call_args.kwargs["row_caps"], adapter.CAPS)
                self.assertEqual(mocked.call_args.kwargs["batch_size"], 512)

    def test_pinned_import_closure(self):
        closure = set()
        for name in adapter.PINS:
            tree = ast.parse((self.source / "scripts" / f"{name}.py").read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    closure.update(a.name for a in node.names if a.name.startswith("kion_"))
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("kion_"):
                    closure.add(node.module)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_module" and node.args and isinstance(node.args[0], ast.Constant):
                    closure.add(node.args[0].value)
        self.assertFalse(closure - set(adapter.PINS), closure - set(adapter.PINS))

    def rebuild_receipt(self):
        receipt = {"contract": "kion-raw-split-plan-v1", "mode": "execute-prepare", "status": "source-cli-completed",
                   "input_content_verified": True, "source_sha256": adapter.PREPARE_SHA,
                   "parameters": {"context": 64, "max-train": 1000000000, "max-eval": 1000000000, "seed": 2026},
                   "output": self.prepared.as_posix(),
                   "argv": [sys.executable, "-B", "pinned-source", "--input", self.raw.as_posix(), "--output", self.prepared.as_posix()],
                   "inputs": [{"path": (self.raw / f"{name}.csv").as_posix(), "expected_sha256": adapter.INPUT_HASHES[name], "content_verified": True} for name in ("users", "items", "interactions")]}
        dump(self.prepared / "prepare-launch.json", receipt)
        return receipt

    def test_rebuild_receipt_and_pending_semantics(self):
        args = dict(self.kwargs, stage="basic-features", basic=None, input_mode="rebuild")
        with self.assertRaises(ValueError):
            adapter.plan(**args)
        receipt = self.rebuild_receipt()
        plan = adapter.plan(**args)
        self.assertFalse(plan["prepared_row_identity_verified"])
        self.assertTrue(all(x["expected_sha256"] is None for x in plan["inputs"] if Path(x["path"]).suffix == ".npz"))
        receipt["status"] = "started"
        dump(self.prepared / "prepare-launch.json", receipt)
        with self.assertRaisesRegex(ValueError, "completed pinned"):
            adapter.plan(**args)

    def test_rebuild_exec_checks_semantics_before_builder(self):
        self.rebuild_receipt()
        args = dict(self.kwargs, stage="basic-features", basic=None, input_mode="rebuild")
        probe = Mock(stdout=json.dumps(adapter.ENVIRONMENT))
        with patch.object(adapter, "run_child", side_effect=[probe, Mock(stdout="{}")]) as child, self.assertRaisesRegex(ValueError, "row identity"):
            adapter.execute(**args)
        self.assertEqual(child.call_count, 2)
        self.assertIn("read_split", child.call_args.args[0][4])
        self.assertFalse(args["output"].exists())

    def test_rebuild_accepts_new_container_hash_only_after_semantics(self):
        self.rebuild_receipt()
        args = dict(self.kwargs, stage="basic-features", basic=None, input_mode="rebuild")
        (self.prepared / "train.npz").write_bytes(b"different rebuilt container")
        manifest = copy.deepcopy(self.basic_manifest)
        for name in ("train", "val"):
            manifest["inputs"][name].update(adapter.ROW_FACTS[name])
            manifest["inputs"][name]["sha256"] = adapter.digest(self.prepared / f"{name}.npz")
        def child(argv, *a, **kw):
            if "-c" in argv:
                return Mock(stdout=json.dumps(adapter.ROW_FACTS if "read_split" in argv[4] else adapter.ENVIRONMENT))
            args["output"].mkdir()
            (args["output"] / "features.npz").write_bytes(b"mock basic")
            dump(args["output"] / "manifest.json", manifest)
            return Mock()
        with patch.object(adapter, "run_child", side_effect=child):
            result = adapter.execute(**args)
        self.assertTrue(result["prepared_row_identity_verified"])
        self.assertFalse(result["full_prepared_history_parity_verified"])
        self.assertFalse(result["semantic_output_equivalence_verified"])


if __name__ == "__main__":
    unittest.main()
