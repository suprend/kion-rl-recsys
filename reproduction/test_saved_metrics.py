"""Synthetic stdlib tests; never open scientific inputs or invoke an export."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import export_saved_metrics as exporter
import saved_metrics as sm


def synthetic_payload():
    eligibility = [0] * 122 + [1] * 616 + [2] * 3358
    ranks = {arm: [0] * sm.ROWS for arm in sm.ARMS}
    for arm in sm.ARMS:
        ranks[arm][738:742] = [1, 2, 10, 0]
    for seed, rank in zip(sm.SEEDS, (1, 2, 10)):
        ranks[f"completion-{seed}"][738:742] = [rank, 0, 0, 0]
    return {"contract": sm.CONTRACT, "eligibility": eligibility, "ranks": ranks}


def synthetic_expected():
    common = {"map10": 1.6 / sm.ROWS,
              "ndcg10": (1 + 1 / math.log2(3) + 1 / math.log2(11)) / sm.ROWS,
              "recall10": 3 / sm.ROWS}
    per_arm = {arm: common.copy() for arm in sm.ARMS}
    for seed, rank in zip(sm.SEEDS, (1, 2, 10)):
        per_arm[f"completion-{seed}"] = {"map10": 1 / rank / sm.ROWS,
                                         "ndcg10": 1 / math.log2(rank + 1) / sm.ROWS,
                                         "recall10": 1 / sm.ROWS}
    main = {arm: common.copy() for arm in sm.MAIN_ARMS}
    main["completion"] = {metric: common[metric] / 3 for metric in sm.METRICS}
    return {"per_arm": per_arm, "main_table": main}


def synthetic_sources():
    sources = {role: {"path": path, "bytes": size, "sha256": sha}
               for role, (path, size, sha) in sm.FIXED_SOURCES.items()}
    sources["exporter"] = sm.descriptor("reproduction/export_saved_metrics.py", b"fixture")
    for family in sm.BASES:
        for name in sm.source_array_names(family):
            sources[f"array/{family}/{name}"] = sm.descriptor(sm.BASES[family] + "/" + name + ".npy", b"fixture")
    return sources


def fixture(directory, payload=None):
    directory.mkdir()
    payload = synthetic_payload() if payload is None else payload
    raw = sm.encoded(payload)
    manifest = {"contract": sm.CONTRACT, "rows": sm.ROWS, "candidate_width": sm.WIDTH,
                "payload": sm.descriptor("rankings.json", raw),
                "counts": {"known": 3974, "unknown": 122, "offpool_known": 616, "covered": 3358},
                "sources": synthetic_sources(), "expected": synthetic_expected(), "scope": sm.SCOPE}
    (directory / "rankings.json").write_bytes(raw)
    (directory / "manifest.json").write_bytes(sm.encoded(manifest))
    return manifest, payload


def npy(values, descr="<i4", shape=None, version=1, order=False, extra=b""):
    shape = (len(values),) if shape is None else shape
    header = repr({"descr": descr, "fortran_order": order, "shape": shape}).encode("ascii")
    header += b"\n"
    prefix = b"\x93NUMPY" + bytes((version, 0))
    prefix += struct.pack("<H" if version == 1 else "<I", len(header))
    unpack = {"<i4": "<i", "<i8": "<q", "|b1": "B"}.get(descr, "<i")
    return prefix + header + b"".join(struct.pack(unpack, value) for value in values) + extra


class MetricsTests(unittest.TestCase):
    def test_independent_rank_arithmetic_and_three_policy_mean(self):
        got = sm.calculate(synthetic_payload())
        self.assertEqual(got["counts"], {"known": 3974, "unknown": 122, "offpool_known": 616, "covered": 3358})
        sm.compare_expected(got, synthetic_expected())
        self.assertEqual(got["main_table"]["completion"]["recall10"], 1 / sm.ROWS)
        self.assertAlmostEqual(got["main_table"]["completion"]["map10"], (1 + 0.5 + 0.1) / (3 * sm.ROWS))
        self.assertEqual(got["per_arm"]["GPT"]["recall10"], 3 / sm.ROWS)

    def test_rank_and_eligibility_types_lengths_and_masks_fail_closed(self):
        cases = []
        for rank in (-1, 11, True, 1.0, None, float("nan")):
            payload = synthetic_payload()
            payload["ranks"]["GPT"][738] = rank
            cases.append(payload)
        for key, value in (("eligibility", []), ("eligibility", [True] * sm.ROWS)):
            payload = synthetic_payload()
            payload[key] = value
            cases.append(payload)
        payload = synthetic_payload()
        payload["ranks"]["GPT"][0] = 1
        cases.append(payload)
        payload = synthetic_payload()
        del payload["ranks"]["C"]
        cases.append(payload)
        for payload in cases:
            with self.subTest(payload=str(payload)[:40]), self.assertRaises(sm.ValidationError):
                sm.calculate(payload)

    def test_decode_duplicate_keys_invalid_json_and_nonfinite(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}', b'{"x":Infinity}', b"", b"\xff", b"[] trailing", b"9" * 5000):
            with self.subTest(raw=raw), self.assertRaises(sm.ValidationError):
                sm.decode_json(raw)

    def test_expected_cannot_substitute_for_calculation(self):
        got = sm.calculate(synthetic_payload())
        expected = synthetic_expected()
        expected["main_table"]["GPT"]["map10"] += 0.01
        with self.assertRaisesRegex(sm.ValidationError, "metric mismatch"):
            sm.compare_expected(got, expected)
        for value in (True, float("inf"), float("nan"), -1, 2):
            expected = synthetic_expected()
            expected["per_arm"]["GPT"]["map10"] = value
            with self.assertRaises(sm.ValidationError):
                sm.compare_expected(got, expected)


class ConsumerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.data = self.base / "data"
        self.manifest, self.payload = fixture(self.data)

    def update_manifest(self):
        (self.data / "manifest.json").write_bytes(sm.encoded(self.manifest))

    def test_valid_verify_and_recompute_are_read_only_by_default(self):
        before = {path.name: path.read_bytes() for path in self.data.iterdir()}
        for command in ("verify", "recompute"):
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(sm.main([command, "--data", str(self.data)]), 0)
            result = json.loads(out.getvalue())
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["scope"]["ci"], sm.SCOPE["ci"])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.data.iterdir()})

    def test_exclusive_new_output_and_input_directory_guard(self):
        output = self.base / "result.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(sm.main(["recompute", "--data", str(self.data), "--output", str(output)]), 0)
        first = output.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(sm.main(["recompute", "--data", str(self.data), "--output", str(output)]), 1)
            self.assertEqual(sm.main(["recompute", "--data", str(self.data), "--output", str(self.data / "new.json")]), 1)
        self.assertEqual(first, output.read_bytes())
        self.assertFalse((self.data / "new.json").exists())

    def test_payload_hash_and_repinning_cannot_hide_wrong_metric(self):
        self.payload["ranks"]["GPT"][738] = 0
        raw = sm.encoded(self.payload)
        (self.data / "rankings.json").write_bytes(raw)
        with self.assertRaisesRegex(sm.ValidationError, "payload pin mismatch"):
            sm.replay(self.data)
        self.manifest["payload"] = sm.descriptor("rankings.json", raw)
        self.update_manifest()
        with self.assertRaisesRegex(sm.ValidationError, "metric mismatch"):
            sm.replay(self.data)

    def test_wrong_counts_unsafe_paths_unpinned_sources_and_extra_fields(self):
        initial = copy.deepcopy(self.manifest)
        changes = (("rows", 0), ("rows", True), ("candidate_width", 1023),
                   ("contract", "other"), ("scope", {}))
        for key, value in changes:
            self.manifest = copy.deepcopy(initial)
            self.manifest[key] = value
            self.update_manifest()
            with self.subTest(key=key), self.assertRaises(sm.ValidationError):
                sm.replay(self.data)
        for path in ("../rankings.json", "/rankings.json", "C:/rankings.json", "sub\\rankings.json", "./rankings.json"):
            self.manifest = copy.deepcopy(initial)
            self.manifest["payload"]["path"] = path
            self.update_manifest()
            with self.subTest(path=path), self.assertRaises(sm.ValidationError):
                sm.replay(self.data)
        self.manifest = copy.deepcopy(initial)
        self.manifest["sources"]["evaluation/matched"]["sha256"] = "0" * 64
        self.update_manifest()
        with self.assertRaisesRegex(sm.ValidationError, "unpinned source"):
            sm.replay(self.data)
        self.manifest = copy.deepcopy(initial)
        self.manifest["user_ids"] = [123]
        self.update_manifest()
        with self.assertRaises(sm.ValidationError):
            sm.replay(self.data)

    def test_empty_oversize_duplicate_and_unexpected_file(self):
        for raw in (b"", b" " * (sm.LIMIT + 1), b'{"contract":"x","contract":"y"}'):
            (self.data / "manifest.json").write_bytes(raw)
            with self.subTest(size=len(raw)), self.assertRaises(sm.ValidationError):
                sm.replay(self.data)
        self.update_manifest()
        (self.data / "unexpected.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(sm.ValidationError, "exactly"):
            sm.replay(self.data)

    def test_payload_and_manifest_share_one_budget(self):
        raw = (self.data / "rankings.json").read_bytes()
        padded = raw + b" " * (sm.LIMIT - len(raw) - 10)
        (self.data / "rankings.json").write_bytes(padded)
        self.manifest["payload"] = sm.descriptor("rankings.json", padded)
        self.update_manifest()
        with self.assertRaisesRegex(sm.ValidationError, "oversized"):
            sm.replay(self.data)


class ExporterPrimitiveTests(unittest.TestCase):
    def test_pinned_size_and_budget_rejections_precede_any_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_bytes(b"12345")
            for size, remaining in ((4, 20), (6, 20), (5, 4)):
                with self.subTest(size=size, remaining=remaining):
                    with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
                        with self.assertRaises(sm.ValidationError):
                            exporter.read_pinned_bytes(path, size, remaining)
            self.assertEqual(exporter.read_pinned_bytes(path, 5, 5), b"12345")

    def test_oversized_source_rejected_before_read_even_with_large_allowance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            with path.open("wb") as stream:
                stream.truncate(exporter.TOTAL_LIMIT + 1)
            with mock.patch.object(Path, "open", side_effect=AssertionError("must not open")):
                with self.assertRaisesRegex(sm.ValidationError, "size pin mismatch"):
                    exporter.read_pinned_bytes(path, 5441, exporter.TOTAL_LIMIT)

    def test_cumulative_budget_prevents_opening_second_pinned_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = b'{"pins":{}}\n'
            replacements = {}
            for family in sm.BASES:
                relative = "request-" + family + ".json"
                (root / relative).write_bytes(raw)
                pin = sm.descriptor(relative, raw)
                replacements["historical-request/" + family] = (relative, pin["bytes"], pin["sha256"])
            opened = []
            original_open = Path.open

            def record_open(path, *args, **kwargs):
                opened.append(path.name)
                return original_open(path, *args, **kwargs)

            with mock.patch.dict(sm.FIXED_SOURCES, replacements):
                with mock.patch.object(exporter, "TOTAL_LIMIT", 2 * len(raw) - 1):
                    with mock.patch.object(Path, "open", record_open):
                        with self.assertRaisesRegex(sm.ValidationError, "total source read budget"):
                            exporter.build_export(root)
            self.assertEqual(opened, ["request-supervised.json"])

    def test_npy_v1_v2_int_bool_without_numpy(self):
        for version in (1, 2):
            for dtype, descr, values in (("int32", "<i4", [3, 10, 4]),
                                         ("int64", "<i8", [2 ** 40, -5, 0]),
                                         ("bool", "|b1", [1, 0, 1])):
                got = exporter.read_npy(npy(values, descr, version=version), dtype, [3])
                self.assertEqual(got, values)

    def test_npy_rejects_object_dtype_order_shape_truncation_and_trailing_bytes(self):
        for raw, dtype, shape in ((npy([1], descr="|O"), "int32", [1]),
                                  (npy([1], descr=">i4"), "int32", [1]),
                                  (npy([1], order=True), "int32", [1]),
                                  (npy([1]), "int32", [2]),
                                  (npy([1])[:-1], "int32", [1]),
                                  (npy([1], extra=b"x"), "int32", [1]),
                                  (npy([2], descr="|b1"), "bool", [1]),
                                  (npy([1], version=3), "int32", [1])):
            with self.subTest(dtype=dtype, shape=shape), self.assertRaises(sm.ValidationError):
                exporter.read_npy(raw, dtype, shape)

    def test_source_path_escape_and_missing_historical_pin(self):
        with tempfile.TemporaryDirectory() as temporary:
            for relative in ("../outside.npy", "C:/outside.npy", "a\\b.npy", "/outside.npy"):
                with self.subTest(relative=relative), self.assertRaises(sm.ValidationError):
                    exporter.safe_source(Path(temporary), relative)
        with self.assertRaises(sm.ValidationError):
            exporter.historical_pin({"pins": {}}, "some/file.npy")
        self.assertEqual(exporter.historical_pin({"pins": {"D:\\old\\some\\file.npy": "abc"}}, "some/file.npy"), "abc")


if __name__ == "__main__":
    unittest.main()
