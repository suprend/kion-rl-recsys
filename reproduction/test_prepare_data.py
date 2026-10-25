"""Tiny contract tests. Never preprocess the real CSVs or train models."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_data as adapter


class PrepareContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.raw = self.base / "raw"
        self.raw.mkdir()
        self.output = self.base / "output"
        for name in adapter.RAW:
            (self.raw / name).write_bytes(b"tiny")
        tiny = {name: (4, hashlib.sha256(b"tiny").hexdigest()) for name in adapter.RAW}
        self.pins = patch.dict(adapter.RAW, tiny, clear=True)
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def test_plan_no_raw_reads_or_subprocess(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            if path.suffix == ".csv":
                raise AssertionError("CSV payload read in plan")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.object(subprocess, "run", side_effect=AssertionError):
            result = adapter.plan(self.raw, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(result["input_content_verified"])
        self.assertEqual(result["argv"][-8:], ["--context", "64", "--max-train", "1000000000", "--max-eval", "1000000000", "--seed", "2026"])

    def test_missing_wrong_size_existing_output(self):
        (self.raw / "users.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing raw"):
            adapter.plan(self.raw, self.output)
        (self.raw / "users.csv").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            adapter.plan(self.raw, self.output)
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "absent"):
            adapter.plan(self.raw, self.output)

    def test_changed_source_rejected(self):
        with patch.object(adapter, "SOURCE_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                adapter.plan(self.raw, self.output)

    def test_interpreter_launcher_path_is_not_resolved(self):
        launcher = self.base / "venv" / "bin" / "python"
        original_is_file = Path.is_file
        original_resolve = Path.resolve
        def is_file(path):
            return True if path == launcher else original_is_file(path)
        def resolve(path, *args, **kwargs):
            if path == launcher:
                raise AssertionError("venv launcher must not be dereferenced")
            return original_resolve(path, *args, **kwargs)
        # Platform-independent symlink boundary test: Windows may deny creation.
        with patch.object(Path, "is_file", is_file), patch.object(Path, "resolve", resolve):
            result = adapter.plan(self.raw, self.output, python=launcher)
        self.assertEqual(result["argv"][0], launcher.as_posix())

    def test_opt_in_passes_exact_source_cli_with_timeout(self):
        expected = adapter.plan(self.raw, self.output)["argv"]
        with patch.object(subprocess, "run") as run:
            result = adapter.execute(self.raw, self.output, timeout_seconds=17)
        run.assert_called_once_with(expected, check=True, timeout=17)
        self.assertTrue(result["input_content_verified"])
        self.assertFalse(result["semantic_output_equivalence_verified"])
        self.assertEqual(json.loads((self.output / "prepare-launch.json").read_text())["status"], "source-cli-completed")

    def test_wrong_payload_hash_stops_before_launch(self):
        (self.raw / "items.csv").write_bytes(b"fake")
        with patch.object(subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                adapter.execute(self.raw, self.output)
        run.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_timeout_records_failure_and_keeps_partial_output(self):
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("fixture", 1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                adapter.execute(self.raw, self.output, timeout_seconds=1)
        self.assertEqual(json.loads((self.output / "prepare-launch.json").read_text())["status"], "failed-or-timed-out")

    def test_default_cli_does_not_import_scientific_libraries(self):
        code = "import sys; import prepare_data; assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(adapter.__file__).parent, check=True)


class RealSourceSemantics(unittest.TestCase):
    def test_tiny_temporal_ties_and_frozen_history(self):
        # Exercise original NumPy implementation on six synthetic rows only.
        import numpy as np
        spec = importlib.util.spec_from_file_location("tiny_pinned_kion_prepare", adapter.source_path())
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        with tempfile.TemporaryDirectory() as temp:
            raw = Path(temp) / "raw"
            raw.mkdir()
            (raw / "users.csv").write_text("user_id\n1\n", encoding="utf-8")
            (raw / "items.csv").write_text("item_id\n10\n20\n30\n40\n", encoding="utf-8")
            (raw / "interactions.csv").write_text(
                "user_id,item_id,last_watch_dt,watched_pct\n"
                "1,10,2021-08-01,50\n1,20,2021-08-02,80\n"
                "1,10,2021-08-09,100\n1,30,2021-08-09,\n"
                "1,40,2021-08-10,10\n1,20,2021-08-16,60\n", encoding="utf-8")
            out = Path(temp) / "prepared"
            meta = source.prepare(raw, out, context=64, max_train=1000000000, max_eval=1000000000, seed=2026)
            self.assertEqual([meta["splits"][s]["saved"] for s in ("train", "val", "test")], [1, 2, 1])
            self.assertEqual(meta["item_id_to_token"], {"10": 3, "20": 4})
            with np.load(out / "val.npz") as val:
                np.testing.assert_array_equal(val["x"][:, :3], [[1, 3, 4], [1, 3, 4]])
                np.testing.assert_array_equal(val["lengths"], [3, 3])
                np.testing.assert_array_equal(val["y"], [3, 2])
            with np.load(out / "test.npz") as test:
                np.testing.assert_array_equal(test["x"][0, :6], [1, 3, 4, 3, 2, 2])


if __name__ == "__main__":
    unittest.main()
