"""Adapter contract tests only: no scientific imports, arrays, or training."""
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import train_heads as adapter


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.output = self.root / "new-output"
        self.manifest = {"contract": "kion-three-head-262k-bank-v1", "state": "completed",
                         "split": "train", "train_rows": 262144,
                         "natural_covered_only": False, "same_day_positives_excluded": True,
                         "feature_schema_sha256": adapter.FEATURE_SCHEMA,
                         "source_columns": adapter.SOURCE_COLUMNS,
                         "teacher": {"path": "/old/opaque/latest.pt", "sha256": "a" * 64,
                                     "config_path": "/old/opaque/config.json", "config_sha256": "b" * 64},
                         "arrays": {}}
        self.array_sizes = {}
        for name, (dtype, shape) in adapter.ARRAY_SPEC.items():
            size = math.prod(shape) * (8 if dtype == "int64" else 4)
            self.manifest["arrays"][name] = {"path": name + ".npy", "dtype": dtype,
                "shape": shape, "payload_bytes": size, "bytes": size + 128, "sha256": "c" * 64}
            self.array_sizes[self.bank / (name + ".npy")] = size + 128
        original_stat = Path.stat
        def mocked_stat(path, *args, **kwargs):
            if path in self.array_sizes:
                return types.SimpleNamespace(st_size=self.array_sizes[path], st_mode=stat.S_IFREG)
            return original_stat(path, *args, **kwargs)
        self.stat_patch = patch.object(Path, "stat", mocked_stat)
        self.stat_patch.start()
        self.addCleanup(self.stat_patch.stop)

    def plan(self, **kwargs):
        data = json.dumps(self.manifest).encode()
        (self.bank / "bank.json").write_bytes(data)
        return adapter.build_plan(self.bank, self.output, hashlib.sha256(data).hexdigest(), **kwargs)

    def test_plan_preserves_teacher_and_fixed_request(self):
        with patch.object(adapter, "load_worker", side_effect=AssertionError("worker imported")):
            plan = self.plan()
        self.assertEqual(plan["request"]["teacher"], self.manifest["teacher"])
        self.assertEqual(plan["request"]["epochs"], 3)
        self.assertEqual(plan["request"]["device"], "cuda")
        self.assertEqual(plan["steps_per_head"], 6144)
        self.assertEqual(len(plan["requirements"]), 10)
        self.assertNotIn("bank_verification_receipt", plan["request"])
        self.assertFalse(self.output.exists())

    def test_import_is_stdlib_only(self):
        code = ("import sys; import train_heads; "
                "assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; "
                "assert '_portable_preserved_heads' not in sys.modules")
        subprocess.run([sys.executable, "-B", "-I", "-c", "import sys; sys.path.insert(0, "
                        + repr(str(Path(adapter.__file__).parent)) + "); " + code], check=True)

    def test_epoch_shortening_refused(self):
        for epochs in [0, 1, 2, 4, True, 3.0]:
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                self.plan(epochs=epochs)

    def test_invalid_schema_refused(self):
        self.manifest["feature_schema_sha256"] = "d" * 64
        with self.assertRaises(ValueError):
            self.plan()

    def test_unsafe_array_paths_refused(self):
        for path in ["../x", "/absolute", "C:/x", "dir\\x", "a//b", "CON", "a:stream", "a/./b"]:
            with self.subTest(path=path):
                self.manifest["arrays"]["encoded"]["path"] = path
                with self.assertRaises(ValueError):
                    self.plan()

    def test_shape_and_size_refused(self):
        for key, value in [("shape", [262143, 256]), ("bytes", 1), ("dtype", "float64")]:
            old = self.manifest["arrays"]["encoded"][key]
            self.manifest["arrays"]["encoded"][key] = value
            with self.assertRaises(ValueError):
                self.plan()
            self.manifest["arrays"]["encoded"][key] = old

    def test_existing_and_input_outputs_refused(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.plan()
        self.output = self.bank / "new-output"
        with self.assertRaises(ValueError):
            self.plan()

    def test_manifest_pin_refused(self):
        self.plan()
        with self.assertRaises(ValueError):
            adapter.build_plan(self.bank, self.output, "0" * 64)

    def test_execution_gate_and_budget(self):
        plan = self.plan()
        with patch.object(adapter, "load_worker") as load:
            with self.assertRaises(ValueError):
                adapter.execute(plan)
            with self.assertRaises(ValueError):
                adapter.execute(plan, execute_training=True)
            load.assert_not_called()

    def test_mocked_worker_receives_exact_request_real_callback(self):
        plan = self.plan()
        worker = Mock()
        worker.run.return_value = {"status": "completed"}
        with patch.object(adapter, "load_worker", return_value=worker):
            result = adapter.execute(plan, execute_training=True, max_seconds=60)
        request, check = worker.run.call_args.args
        self.assertEqual(request["teacher"], self.manifest["teacher"])
        self.assertEqual(request["epochs"], 3)
        self.assertNotIn("bank_verification_receipt", request)
        self.assertIn("deadline_utc", request)
        check(phase="mock-only")
        self.assertEqual(result["status"], "completed")

    def test_edited_plan_cannot_change_device_or_seed(self):
        for key, value in [("device", "cpu"), ("seed", 1), ("batch_size", 2)]:
            plan = self.plan()
            plan["request"][key] = value
            with self.assertRaises(ValueError):
                adapter.execute(plan, execute_training=True, max_seconds=60)

    def test_deadline_and_disk_guards(self):
        with patch.object(adapter.time, "monotonic", side_effect=[0, 61]):
            check = adapter.make_guard(self.output, 60, 2)
            with self.assertRaises(TimeoutError):
                check()
        check = adapter.make_guard(self.output, 60, 2)
        with patch.object(adapter.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)):
            with self.assertRaises(OSError):
                check()

    def test_invalid_budgets_refused(self):
        for value in [float("nan"), float("inf"), 0, -1, 604801]:
            with self.assertRaises(ValueError):
                adapter.make_guard(self.output, value, 2)

    def test_source_verification_imports_no_scientific_libraries(self):
        module = adapter.load_worker()
        self.assertEqual(module.FEATURE_SCHEMA_SHA256, adapter.FEATURE_SCHEMA)
        self.assertEqual(module.EPOCHS, 3)
        self.assertEqual(module.CONFIG, adapter.HEAD_CONFIG)


if __name__ == "__main__":
    unittest.main()
