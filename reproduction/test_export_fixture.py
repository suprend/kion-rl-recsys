"""Synthetic stdlib-only safety/schema tests; no real inputs or numerical imports."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name("export_fixture.py")
spec = importlib.util.spec_from_file_location("fixture_export_under_test", SOURCE)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def fake_npy(path, *, shape=(2, 3), dtype="<f4", fortran=False, suffix=b""):
    header = repr({"descr": dtype, "fortran_order": fortran, "shape": shape}).encode("latin1")
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    header += b" " * padding + b"\n"
    size = 1 if dtype == "|b1" else 4
    count = 1
    for dimension in shape:
        count *= dimension
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\x00" * (count * size) + suffix)
    return path


def limits():
    return {"maximum_seconds": 30, "maximum_hashed_source_bytes": 1 << 20,
            "maximum_output_bytes": 128 << 20, "maximum_working_bytes": 2 << 30,
            "minimum_disk_free_bytes": 15 << 30}


class ExportFixtureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "reproduction").mkdir()
        (self.root / "research").mkdir()

    def request(self):
        return {"contract": exporter.REQUEST_CONTRACT, "training_updates": 0, "provider_opened": False,
                "positions": [0, 1, 2, 3], "cpu_threads": 2, "native_admission": "root-owned",
                "deadline_utc": (datetime.now(timezone.utc)+timedelta(minutes=2)).isoformat(),
                "output_dir": (self.root / "reproduction/inputs-r1").as_posix(),
                "private_provenance_output": (self.root / "research/private.json").as_posix()}

    def test_import_and_plan_are_stdlib_only(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        names = {alias.name.split(".")[0] for node in imports for alias in node.names}
        self.assertFalse(names & {"numpy", "torch", "catboost", "transformers"})
        plan = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "plan")
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(plan)))

    def test_valid_synthetic_header(self):
        path = fake_npy(self.root / "data.npy")
        header = exporter.npy_header(path)
        self.assertEqual(header["dtype"], "float32")
        self.assertEqual(header["shape"], [2, 3])
        self.assertEqual(path.stat().st_size-header["offset"], 24)

    def test_header_rejects_object_fortran_and_trailing_bytes(self):
        for index, options in enumerate(({"dtype": "|O"}, {"fortran": True}, {"suffix": b"extra"})):
            with self.subTest(options=options):
                path = fake_npy(self.root / (str(index)+".npy"), **options)
                with self.assertRaises(ValueError):
                    exporter.npy_header(path)

    def test_npy_length_is_bounded_before_header_read(self):
        path = self.root / "huge-header.npy"
        path.write_bytes(b"\x93NUMPY\x02\x00"+struct.pack("<I", 1 << 20))
        with self.assertRaisesRegex(ValueError, "header exceeds"):
            exporter.npy_header(path)

    def test_source_paths_cannot_escape_declared_root(self):
        for value in ("../private", self.root.as_posix()):
            with self.assertRaises(ValueError):
                exporter.contained(self.root, value)

    def test_fixed_selection_and_fresh_owned_outputs(self):
        with patch.object(exporter, "ROOT", self.root):
            request = self.request()
            output, _ = exporter.validate_request(request)
            self.assertFalse(output.exists())
            for positions in ([1, 2, 3, 4], [0, 1], [0, 1, 2, 2]):
                with self.subTest(positions=positions), self.assertRaises(ValueError):
                    exporter.validate_request({**request, "positions": positions})
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "Fresh reproduction"):
                exporter.validate_request(request)

    def test_deadline_and_no_training_gates(self):
        with patch.object(exporter, "ROOT", self.root):
            for changes in ({"training_updates": 1}, {"provider_opened": True},
                            {"native_admission": "none"}, {"cpu_threads": 8},
                            {"deadline_utc": "2001-01-01T00:00:00+00:00"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    exporter.validate_request({**self.request(), **changes})

    def test_accepted_stat_pin_avoids_whole_file_hash_and_rejects_mutation(self):
        path = self.root / "opaque.bin"
        path.write_bytes(b"synthetic")
        entry = {"path": path.as_posix(), "sha256": "a"*64, "bytes": path.stat().st_size}
        accepted = {path.as_posix(): {"hashes_verified": True, "sha256": entry["sha256"], "stat": exporter.identity(path)}}
        budget = exporter.Budget(limits())
        with patch.object(budget, "hash", side_effect=AssertionError("must not hash")):
            _, evidence = exporter.verified(entry, budget, accepted)
            self.assertEqual(evidence["verification"], "accepted-full-hash-with-unchanged-stat")
            path.write_bytes(b"MUTATION!")
            previous = accepted[path.as_posix()]["stat"]["mtime_ns"]
            os.utime(path, ns=(previous+1000000000, previous+1000000000))
            with self.assertRaisesRegex(ValueError, "pin differs"):
                exporter.verified(entry, budget, accepted)

    def test_hash_budget_rejects_before_streaming_file(self):
        path = self.root / "synthetic.bin"
        path.write_bytes(b"1234")
        budget = exporter.Budget({**limits(), "maximum_hashed_source_bytes": 3})
        with patch.object(exporter, "sha", side_effect=AssertionError("must reject before read")):
            with self.assertRaisesRegex(ValueError, "hash budget"):
                budget.hash(path)

    def test_accepted_receipt_requires_pinned_hash_and_explicit_acceptance(self):
        path = self.root / "receipt.json"
        path.write_text(json.dumps({"contract": "kion-root-accepted-file-stats-v1", "state": "accepted", "files": {}}), encoding="utf-8")
        pin = {"path": path.as_posix(), "sha256": exporter.sha(path)}
        self.assertEqual(exporter.accepted_pins(pin, exporter.Budget(limits())), {})
        path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            exporter.accepted_pins(pin, exporter.Budget(limits()))

    def test_only_authorised_teacher_path_sha_and_size_are_eligible(self):
        with patch.object(exporter, "ROOT", self.root):
            expected = (self.root.parent / "night-20261023-r1/sft-recovery/run-recovery-r1/worker/model/latest.pt").resolve()
            pin = {"path": expected.as_posix(), "sha256": exporter.TEACHER_SHA256, "bytes": exporter.TEACHER_BYTES}
            cache = {"path": expected.as_posix(), "sha256": exporter.TEACHER_SHA256}
            self.assertEqual(exporter.validate_trusted_teacher(pin, cache), expected)
            for change in ({"sha256": "f"*64}, {"bytes": 42}, {"path": (self.root/"untrusted.pt").as_posix()}):
                with self.subTest(change=change), self.assertRaisesRegex(ValueError, "authorised local"):
                    exporter.validate_trusted_teacher({**pin, **change}, cache)
            with self.assertRaisesRegex(ValueError, "cache teacher"):
                exporter.validate_trusted_teacher(pin, {**cache, "sha256": "f"*64})

    def test_large_teacher_config_has_pin_only16MiB_boundary(self):
        path = self.root / "config.json"
        with path.open("wb") as stream:
            stream.truncate(8947263)
        pin = {"path": path.as_posix(), "sha256": exporter.TEACHER_CONFIG_SHA256}
        cache = {"config_path": path.as_posix(), "config_sha256": exporter.TEACHER_CONFIG_SHA256}
        with self.assertRaisesRegex(ValueError, "Oversized"):
            exporter.read_json(path)
        budget = exporter.Budget(limits())
        with patch.object(budget, "hash", return_value=exporter.TEACHER_CONFIG_SHA256):
            self.assertEqual(exporter.pin_teacher_config(pin, cache, budget), path.resolve())
        with path.open("wb") as stream:
            stream.truncate((16 << 20)+1)
        with self.assertRaisesRegex(ValueError, "exceeds16MiB"):
            exporter.pin_teacher_config(pin, cache, budget)

    def test_array_descriptor_and_actual_header_must_agree(self):
        path = fake_npy(self.root / "data.npy")
        descriptor = {"path": path.name, "sha256": exporter.sha(path), "bytes": path.stat().st_size,
                      "dtype": "float32", "shape": [2, 3]}
        result = exporter.source_array(self.root, descriptor, ("float32", (2, 3)), exporter.Budget(limits()), {})
        self.assertEqual(result["payload_bytes"], 24)
        wrong = {**descriptor, "shape": [3, 2]}
        with self.assertRaisesRegex(ValueError, "NPY schema"):
            exporter.source_array(self.root, wrong, ("float32", (3, 2)), exporter.Budget(limits()), {})

    def test_run_stops_before_numerical_imports_when_plan_fails(self):
        before = {name for name in sys.modules if name in ("torch", "numpy")}
        with patch.object(exporter, "plan", side_effect=ValueError("preflight rejected")):
            with self.assertRaisesRegex(ValueError, "preflight rejected"):
                exporter.run({})
        self.assertEqual(before, {name for name in sys.modules if name in ("torch", "numpy")})


if __name__ == "__main__":
    unittest.main()
