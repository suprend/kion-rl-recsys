"""Synthetic stdlib tests. Never open model/data payloads or import ML packages."""
from __future__ import annotations

import importlib
import copy
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kion_replay.head import buckets
from kion_replay.io import checked_path, npy_header, read_json, sha256, verify_array, verify_descriptor, verify_torch_archive
from kion_replay.metrics import distribution, rank, summarize
from kion_replay.verify import GPT_CONFIG, SCHEMA, SOURCE, validate_fixture_semantics


def synthetic_npy(path, shape=(2, 3), descr="<f4", fortran=False, payload=None):
    header = repr({"descr": descr, "fortran_order": fortran, "shape": shape})
    raw = header.encode("latin1")
    padding = (-10 - len(raw) - 1) % 64
    raw += b" " * padding + b"\n"
    size = math.prod(shape) * (int(descr[-1]) if descr != "|b1" else 1)
    path.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw
                     + (b"\0" * size if payload is None else payload))
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
            "dtype": "float32", "shape": list(shape)}


class PortableIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self):
        self.temporary.cleanup()

    def test_npy_header_and_hash(self):
        path = self.root / "synthetic.npy"
        entry = synthetic_npy(path)
        self.assertEqual(npy_header(path)["shape"], [2, 3])
        self.assertEqual(verify_array(self.root, entry), path)
        path.write_bytes(path.read_bytes()[:-1] + b"\1")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            verify_array(self.root, entry)

    def test_reject_trailing_or_truncated_npy(self):
        path = self.root / "synthetic.npy"
        synthetic_npy(path)
        path.write_bytes(path.read_bytes() + b"extra")
        with self.assertRaisesRegex(ValueError, "payload length"):
            npy_header(path)
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaisesRegex(ValueError, "payload length"):
            npy_header(path)

    def test_reject_object_big_endian_and_fortran(self):
        path = self.root / "synthetic.npy"
        for descriptor in ("|O8", ">f4"):
            synthetic_npy(path, descr=descriptor)
            with self.assertRaisesRegex(ValueError, "dtype"):
                npy_header(path)
        synthetic_npy(path, fortran=True)
        with self.assertRaisesRegex(ValueError, "C-order"):
            npy_header(path)

    def test_reject_unsafe_paths(self):
        (self.root / "real.txt").write_text("synthetic", encoding="utf-8")
        self.assertEqual(checked_path(self.root, "real.txt"), self.root / "real.txt")
        for value in ("../real.txt", "/real.txt", "C:/real.txt", "a\\real.txt", "./real.txt", "a//real.txt"):
            with self.assertRaises(ValueError):
                checked_path(self.root, value)

    def test_reject_descriptor_budget_and_header_mismatch(self):
        path = self.root / "synthetic.npy"
        entry = synthetic_npy(path)
        with self.assertRaisesRegex(ValueError, "budget"):
            verify_descriptor(self.root, entry, max_bytes=1)
        with self.assertRaisesRegex(ValueError, "header mismatch"):
            verify_array(self.root, {**entry, "shape": [3, 2]})

    def test_duplicate_and_nonfinite_json_rejected(self):
        path = self.root / "synthetic.json"
        for contents in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            path.write_text(contents, encoding="utf-8")
            with self.assertRaises(ValueError):
                read_json(path)

    def test_torch_archive_is_bounded_without_decoding_pickle(self):
        path = self.root / "synthetic.pt"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/data.pkl", b"not decoded by verifier")
            archive.writestr("archive/data/0", b"\0" * 16)
            archive.writestr("archive/version", b"3\n")
        verify_torch_archive(path)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("archive/../data.pkl", b"unsafe")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            verify_torch_archive(path)

    def test_compressed_torch_archive_rejected(self):
        path = self.root / "synthetic.pt"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("archive/data.pkl", b"synthetic")
        with self.assertRaisesRegex(ValueError, "uncompressed"):
            verify_torch_archive(path)

    def test_package_imports_are_lazy(self):
        for name in ("kion_replay.cli", "kion_replay.verify", "kion_replay.head", "kion_replay.gpt"):
            importlib.import_module(name)
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_named_fixture_rejects_alternative_scientific_metadata(self):
        fixture = {"contract": "kion-portable-fixture-v1", "state": "completed", "rows": 4,
                   "candidate_width": 1024, "raw_width": 1068, "feature_schema_sha256": SCHEMA,
                   "source_columns": SOURCE, "no_target_injection": True, "no_seen_filter": True,
                   "target_labels_included": False, "tie_break": "score-desc/rawID-asc", "temperature": 1.0,
                   "precision": "cuda-fp16", "selection": "first4-frozen-screen-cache-positions-outcome-independent",
                   "head_batch_max": 16, "candidate_chunk": 64,
                   "producer_layout": [{"size": 16, "slot": slot} for slot in range(4)],
                   "arrays": {"item_table": {"dtype": "float32", "shape": [15418, 256]}},
                   "gpt": {"config": GPT_CONFIG, "producer_batch_size": 16, "row_slots": [0, 1, 2, 3]}}
        validate_fixture_semantics(fixture)
        changes = {"state": "partial", "rows": 1, "raw_width": 1067, "candidate_width": 512,
                   "no_target_injection": False, "no_seen_filter": False, "target_labels_included": True,
                   "tie_break": "token-ID", "temperature": 2, "precision": "fp32", "head_batch_max": 4,
                   "candidate_chunk": None, "producer_layout": [{"size": 4, "slot": slot} for slot in range(4)],
                   "arrays": {"item_table": {"dtype": "float32", "shape": [15419, 256]}},
                   "gpt": {"config": GPT_CONFIG, "producer_batch_size": 16, "row_slots": [1, 2, 3, 4]}}
        for key, value in changes.items():
            changed = copy.deepcopy(fixture)
            changed[key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                validate_fixture_semantics(changed)


class NumericalDefinitionTests(unittest.TestCase):
    def test_softmax_stability(self):
        logp, probs = distribution([10000, 10000])
        self.assertEqual(probs, [0.5, 0.5])
        self.assertTrue(all(abs(value + math.log(2)) < 1e-12 for value in logp))

    def test_rank_uses_raw_id_tie_break(self):
        self.assertEqual(rank([3, 3, 2], [20, 10, 0], k=3), [1, 0, 2])

    def test_relevance_metrics_unknown_and_offpool_misses(self):
        scores = [[10 - i for i in range(10)]] * 3
        tokens = [list(range(3, 13))] * 3
        raw = [list(range(100, 110))] * 3
        result = summarize(scores, raw, tokens, scores, targets=[3, 4, -1])
        self.assertEqual(result["map10"], 0.5)
        self.assertEqual(result["recall10"], 2 / 3)
        self.assertAlmostEqual(result["ndcg10"], 0.5436432511904858)
        self.assertEqual(result["mean_KL_to_recomputed_C"], 0)

    def test_no_labels_means_no_claimed_relevance_metrics(self):
        result = summarize([list(range(10))], [list(range(10))], [list(range(10))], [list(range(10))])
        self.assertFalse(result["label_metrics_available"])
        self.assertNotIn("map10", result)

    def test_producer_slot_collisions_preserve_separate_rows(self):
        result = buckets([{"size": 16, "slot": 3}, {"size": 16, "slot": 0}, {"size": 16, "slot": 3}, {"size": 2, "slot": 1}])
        self.assertEqual(result, [{"size": 16, "slots": [3, 0], "rows": [0, 1]},
                                  {"size": 16, "slots": [3], "rows": [2]},
                                  {"size": 2, "slots": [1], "rows": [3]}])


if __name__ == "__main__":
    unittest.main()
