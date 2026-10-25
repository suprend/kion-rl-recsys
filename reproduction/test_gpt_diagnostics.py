"""Tiny, relocated fixtures only; no retained payloads or runtime imports."""
import copy
from datetime import date
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import gpt_diagnostics as adapter
from gpt_provider import BoundIO, FileBinding, canonical_hash, _close_array


DTYPES = {"raw_row_index": "int64", "user_id": "int64", "raw_item_id": "int64", "date": "int32",
    "token": "int32", "outcome": "float32", "parser_status": "uint8", "baseline": "float64",
    "residual": "float64", "n_item": "int32", "n_type": "int32", "n_global": "int32",
    "fallback_level": "uint8", "flags": "uint16"}
FLAGS = {"no_item_support": 0, "no_type_support": 1, "no_global_support": 2,
    "prior_only": 3, "type_unknown": 4, "used_c0": 5, "baseline_valid": 6,
    "scarce": 7, "residual_valid": 8}
FAMILIES = ("personal.activity", "personal.typed_completion", "personal.window_completion",
            "personal.age_rating_profile")


class Store:
    def __init__(self, root, manifest, arrays):
        self.root, self.manifest, self.arrays = Path(root), manifest, arrays
        self.query_arrays, self.catalog = {}, {}
        self.keys = arrays["user_id"]

    def close(self):
        for values in [*self.arrays.values(),
                       *(v for group in self.query_arrays.values() for v in group.values())]:
            _close_array(values)


class User:
    schema = {"columns": ["typed.film.mean"], "null_policy": "finite fill only"}

    def __init__(self, store, bits, metadata, cards, **kwargs):
        self.store = store
        if cards != {10: {"content_type": "film", "age_rating": "6"}}:
            raise ValueError("test expected parsed cards")

    def features(self, *args, **kwargs):
        return self.schema

    def close(self):
        for value in self.threshold_facts.values():
            _close_array(value)
        self.store.close()


class Fixture:
    root = "Q:/retained"

    def __init__(self, local):
        self.local, self.bindings = Path(local), {}
        history = SimpleNamespace(_day=lambda v: date.fromisoformat(v).toordinal(),
            _mapping_int=lambda v, _: int(v), _csv_rows=self.csv_rows)
        self.events = SimpleNamespace(CONTRACT="kion-completion-event-store-v1", EVENT_DTYPES=DTYPES,
            FLAG_BITS=FLAGS, QUERY_JOIN="exact-immutable-user-day-raw-item-H63-v1",
            FULL_QA="kion-completion-event-store-qa-v1", history=history,
            diagnostics=SimpleNamespace(array_hash=lambda a: hashlib.sha256(a.tobytes()).hexdigest()),
            CompletionEventStore=Store)
        self.diagnostics = SimpleNamespace(CONTRACT="kion-user-diagnostics-v1", FAMILIES=FAMILIES,
            THRESHOLD_CONTRACT="kion-exact-outcome-threshold-facts-v1", history=history, UserDiagnostics=User)
        meta = self.json("metadata.json", {"item_id_to_token": {"10": 3}}, signed=False)
        items = self.bytes("items.csv", b"item_id,content_type,age_rating\n10,film,6\n")
        source = self.bytes("builder.py", b"# tiny source\n", "source")
        prepared = {k + ".npz": self.bytes(k + ".npz", b"tiny prepared " + k.encode()) for k in ("train", "val")}
        rows = {"train": "a" * 64, "val": "b" * 64}
        arrays = {k: self.array("events/" + k + ".npy", np.array([
            date(2021, 1, 1).toordinal() if k == "date" else 3 if k == "token" else 0], dtype=v))
            for k, v in DTYPES.items()}
        catalog = self.local / "events/catalog.npz"
        np.savez(catalog, rating=np.array([1], dtype="int8"))
        cat = self.bind("events/catalog.npz", "payload")
        cat["arrays_sha256"] = canonical_hash({"rating": self.events.diagnostics.array_hash(np.array([1], dtype="int8"))})
        receipt = {"contract": self.events.QUERY_JOIN, "error_count": 0,
                   "splits": {s: {"rows": 1, "row_keys_sha256": rows[s]} for s in rows}}
        query = {s: {k: self.array("events/" + s + "_" + k + ".npy", np.array([1], dtype=d))
                      for k, d in {"user_id": "int64", "origin": "int32", "length": "uint8"}.items()} for s in rows}
        self.event = {"contract": self.events.CONTRACT, "state": "completed", "scope": "full",
            "coverage": {"raw_events": "complete_before_role_cap", "query_mapping": "complete_all_prepared_rows"},
            "metadata_path": meta["path"], "metadata_sha256": meta["sha256"], "prepared_pins": prepared,
            "row_keys_sha256": rows, "source_pins": {"items": items, "builder": source}, "arrays": arrays,
            "flag_bits": FLAGS, "events_count": 1, "fit_cutoff": "2021-08-09", "catalog": cat,
            "query_coverage_receipt": receipt, "query_arrays": query,
            "final_validation": {"contract": self.events.FULL_QA, "status": "passed", "error_count": 0,
                                 "events_count": 1, "query_counts": {"train": 1, "val": 1}}}
        epin = self.json("events/manifest.json", self.event)
        self.threshold = {"contract": self.diagnostics.THRESHOLD_CONTRACT, "state": "completed",
            "events_count": 1, "event_store": epin, "fit_cutoff": "2021-08-09", "source_pins": {"builder": source},
            "arrays": {"threshold_bits": self.array("threshold/bits.npy", np.array([0], dtype="uint8")),
                       "source_csv_data_row": self.array("threshold/row.npy", np.array([1], dtype="uint32"))}}
        tpin = self.json("threshold/manifest.json", self.threshold)
        self.user = {"contract": self.diagnostics.CONTRACT, "state": "completed", "coverage": "full_rows",
            "storage": "normalized_event_facts_exact_query_mapping", "families": list(FAMILIES),
            "diagnostic_only": True, "model_input_columns_added": [], "event_store": epin, "threshold_facts": tpin,
            "source_pins": {"builder": source}, "metadata_sha256": meta["sha256"], "prepared_pins": prepared,
            "row_keys_sha256": rows, "query_coverage_receipt": receipt, "split_counts": {"train": 1, "val": 1},
            "columns": User.schema["columns"], "null_policy": User.schema["null_policy"], "dtype": "float64",
            "null_mask_dtype": "bool", "nullable_columns": User.schema["columns"],
            "clock": "strict_h63_prior", "fit_cutoff": "2021-08-09"}
        self.upin = self.json("user/manifest.json", self.user)
        self.manifest = {"data": {"files": prepared}, "row_keys_sha256": rows}
        self.metadata_hash = meta["sha256"]

    @staticmethod
    def csv_rows(path, fields):
        import csv
        with Path(path).open(encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                yield tuple(row[k] for k in fields)

    def bind(self, rel, role):
        path = self.local / rel
        logical = self.root + "/" + rel
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.bindings[logical] = FileBinding(logical, str(path), digest, role)
        return {"path": logical, "sha256": digest}

    def bytes(self, rel, data, role="payload"):
        target = self.local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return self.bind(rel, role)

    def json(self, rel, data, signed=True):
        data = {k: v for k, v in data.items() if k != "signature"}
        if signed:
            data["signature"] = canonical_hash(data)
        return self.bytes(rel, json.dumps(data).encode(), "metadata")

    def array(self, rel, values):
        (self.local / rel).parent.mkdir(parents=True, exist_ok=True)
        np.save(self.local / rel, values)
        return {**self.bind(rel, "payload"), "shape": list(values.shape), "dtype": str(values.dtype)}

    def io(self):
        return BoundIO(self.bindings.values(), lambda: None)

    def load(self, **kwargs):
        return adapter.load_user_diagnostics(self.io(), self.root + "/user", np=np,
            events=self.events, diagnostics=self.diagnostics, **kwargs)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = Fixture(self.temp.name)

    def test_relocated_nested_load_and_original_identities(self):
        f = self.f
        store = f.load(expected_metadata_sha256=f.metadata_hash,
            expected_prepared_pins=f.user["prepared_pins"], expected_row_keys_sha256=f.user["row_keys_sha256"])
        self.addCleanup(store.close)
        self.assertEqual(store.manifest["event_store"], f.user["event_store"])
        self.assertFalse(store.store.arrays["date"].flags.writeable)

    def test_provider_summary_and_explicit_old_contract_rejection(self):
        f = self.f
        calls = []
        summary, stores = adapter.load_diagnostics(f.io(), f.root, {"user": f.upin},
            metadata_hash=f.metadata_hash, manifest=f.manifest, np=np, events=f.events,
            diagnostics=f.diagnostics, require_sources=calls.append)
        self.addCleanup(stores["user"].close)
        self.assertFalse(summary["user"]["model_features"])
        self.assertEqual(calls, [[f.diagnostics, f.events]])
        pin = f.json("old.json", {"contract": "kion-completion-diagnostics-v1"})
        with self.assertRaisesRegex(ValueError, "Unsupported diagnostic contract"):
            adapter.load_diagnostics(f.io(), f.root, {"old": pin}, metadata_hash=f.metadata_hash,
                manifest=f.manifest, np=np, events=f.events, diagnostics=f.diagnostics, require_sources=calls.append)

    def test_manifest_basename_and_prepared_logical_binding(self):
        with self.assertRaisesRegex(ValueError, "manifest.json"):
            adapter._manifest_root(self.f.root, {"path": "user/other.json"})
        pins = copy.deepcopy(self.f.user["prepared_pins"])
        pins["train.npz"]["path"] = "R:/wrong/train.npz"
        with self.assertRaisesRegex(ValueError, "prepared source"):
            self.f.load(expected_prepared_pins=pins)

    def test_event_coverage_source_and_final_qa_fail_closed(self):
        for field, value in (("scope", "probe"), ("flag_bits", {}), ("final_validation", {})):
            with self.subTest(field=field):
                event = {**self.f.event, field: value}
                self.f.json("events/manifest.json", event)
                with self.assertRaises(ValueError):
                    adapter.load_event_store(self.f.io(), self.f.root + "/events", np=np, events=self.f.events)

    def test_mappings_closed_after_late_schema_failure(self):
        f, opened = self.f, []
        f.user["null_policy"] = "wrong"
        f.json("user/manifest.json", f.user)
        original = np.load
        def tracked(*args, **kwargs):
            result = original(*args, **kwargs)
            if getattr(result, "_mmap", None) is not None:
                opened.append(result)
            return result
        with patch.object(np, "load", side_effect=tracked):
            with self.assertRaisesRegex(ValueError, "NULL schema"):
                f.load()
        self.assertGreater(len(opened), 10)
        self.assertTrue(all(a._mmap.closed for a in opened))

    def test_threshold_shape_failure_closes_current_mapping(self):
        f, opened = self.f, []
        f.threshold["arrays"]["source_csv_data_row"] = f.array("threshold/row.npy", np.array([1, 2], dtype="uint32"))
        f.user["threshold_facts"] = f.json("threshold/manifest.json", f.threshold)
        f.json("user/manifest.json", f.user)
        original = np.load
        def tracked(*args, **kwargs):
            result = original(*args, **kwargs)
            if getattr(result, "_mmap", None) is not None:
                opened.append(result)
            return result
        with patch.object(np, "load", side_effect=tracked):
            with self.assertRaisesRegex(ValueError, "Threshold payload shape"):
                f.load()
        self.assertTrue(all(a._mmap.closed for a in opened))

    def test_nested_pin_and_locality_rejection(self):
        f = self.f
        (f.local / "builder.py").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            f.load()
        with self.assertRaisesRegex(ValueError, "directly inside"):
            adapter._local_payload(f.io(), f.root + "/events", {"path": f.root + "/train.npz"})


if __name__ == "__main__":
    unittest.main()
