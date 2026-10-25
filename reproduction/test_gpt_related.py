"""Tiny generated fixtures, retained source only; no historical payload reads."""
import copy
from datetime import date
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gpt_related as port
from gpt_provider import BoundIO, FileBinding, canonical_hash, numpy_backend


class RelatedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.np = numpy_backend()
        scripts = (Path(__file__).resolve().parents[1] / "source/fullstudy/code/tlab-recsys-2026/artifacts/"
                   "task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts")
        for name in ("kion_related_features", "kion_franchise_progress"):
            spec = importlib.util.spec_from_file_location(name, scripts / (name + ".py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            setattr(cls, name, mod)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        self.meta = {"item_id_to_token": {"10": 3}, "vocab_size": 4}

    def bind(self, logical, data, role="metadata"):
        path = self.root / ("bound" + str(len(self.rows)))
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.rows.append(FileBinding(logical, path.as_posix(), digest, role))
        return {"path": logical, "sha256": digest}

    def io(self):
        return BoundIO(self.rows, Mock())

    def make_related(self, mutate=None):
        np = self.np
        source = self.kion_related_features.RelatedInterestIndex(
            self.meta, {"10": {"studios": "Tiny", "release_year": "2020"}},
            np.array([1]), np.array([3]), np.array([date(2021, 7, 1).toordinal()]))
        folder = self.root / "saved"
        manifest = source.save(folder)
        for pin in manifest["arrays"].values():
            self.bind("D:/old/index/" + pin["path"], (folder / pin["path"]).read_bytes(), "payload")
        items = self.bind("D:/old/raw/items.csv", b"tiny items", "payload")
        interactions = self.bind("D:/old/raw/interactions.csv", b"tiny interactions", "payload")
        meta = self.bind("D:/old/data/metadata.json", b"{}")
        manifest["provenance"].update(pilot_only=False, source_scope="full_source",
            source_pins={"items": items}, audience_source_pins={"items": items, "interactions": interactions, "metadata": meta})
        if mutate:
            mutate(manifest)
        manifest["state_sha256"] = canonical_hash({k: v for k, v in manifest.items() if k != "state_sha256"})
        receipt = self.bind("D:/old/index/manifest.json", json.dumps(manifest).encode())
        return source, manifest, {"index_dir": "index", "manifest": receipt}

    def load(self, config):
        return port.load_scoped_related(self.io(), "D:/old", config, self.meta,
            raw_root="D:/old/raw", data_root="D:/old/data", np=self.np, related=self.kion_related_features)

    def test_real_retained_index_relocated_and_logical_provenance(self):
        source, manifest, config = self.make_related()
        loaded, provenance = self.load(config)
        self.addCleanup(loaded.close)
        self.np.testing.assert_array_equal(source.users, loaded.users)
        self.assertEqual(source.columns, loaded.columns)
        self.assertEqual(provenance["related_interest"], manifest["provenance"])
        self.assertFalse(loaded.users.flags.writeable)
        self.assertEqual(loaded.mapping, self.meta["item_id_to_token"])

    def test_legacy_item_entities_and_state_schema_validation(self):
        _, _, config = self.make_related(lambda m: m.pop("item_entities"))
        loaded, _ = self.load(config)
        self.addCleanup(loaded.close)
        self.assertEqual(loaded.item_entities["studios"][3], ["tiny"])
        config["manifest"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.load(config)

    def test_wrong_runtime_corpus_is_rejected_and_mmaps_closed(self):
        _, _, config = self.make_related(lambda m: m["provenance"]["audience_source_pins"]["metadata"].update(path="D:/old/raw/items.csv"))
        with self.assertRaises(ValueError):
            self.load(config)

    def test_bad_support_parameters(self):
        _, _, config = self.make_related(lambda m: m["parameters"].update(cache_days=True))
        with self.assertRaisesRegex(ValueError, "support/cache"):
            self.load(config)

    def test_wrong_array_pin_and_cleanup(self):
        _, _, config = self.make_related(lambda m: m["arrays"]["days"].update(sha256="0" * 64))
        loaded = []
        real_load = self.np.load
        def track(*args, **kwargs):
            a = real_load(*args, **kwargs)
            loaded.append(a)
            return a
        with patch.object(self.np, "load", side_effect=track), self.assertRaises(ValueError):
            self.load(config)
        self.assertTrue(loaded)
        self.assertTrue(all(a._mmap.closed for a in loaded))

    def test_historical_builder_alias_exact_pin(self):
        io = Mock()
        recorded = {"path": "D:/historical/kion_audience_features.py", "sha256": port.HISTORICAL_BUILDER_SHA256}
        receipt = {"path": "source/accepted.py", "sha256": port.HISTORICAL_BUILDER_SHA256}
        result = port.related_historical_builder(io, "D:/old", recorded, receipt)
        io.pin.assert_called_once_with("D:/old", receipt, "source")
        self.assertEqual(result["recorded_builder"], recorded)
        for change in ({"sha256": "0" * 64}, {"path": "other.py"}):
            with self.assertRaises(ValueError):
                port.related_historical_builder(io, "D:/old", {**recorded, **change}, receipt)

    def test_franchise_retained_validation_and_typed_namespace(self):
        graph = {"contract": "kion-franchise-progress-v1", "works": {}, "scopes": {},
                 "memberships": [], "item_work_links": [], "edges": [], "evidence": {}}
        pin = self.bind("D:/old/graph.json", json.dumps(graph).encode())
        params = {"graph": pin, "temporal_mode": "retrospective_current", "order_kinds": list(port.FRANCHISE_ORDER_KINDS)}
        prepared, mode, columns, provenance = port.load_scoped_franchise(
            self.io(), "D:/old", params, self.meta, franchise=self.kion_franchise_progress)
        self.assertEqual(tuple(prepared.order_kinds), port.FRANCHISE_ORDER_KINDS)
        self.assertEqual(len(columns), 2 * len(self.kion_franchise_progress.COLUMNS))
        self.assertFalse(provenance["typed_orders_merged"])
        self.assertEqual(mode, "retrospective_current")
        for changes in ({"order_kinds": ["mainline_release"]}, {"temporal_mode": "unknown"},
                        {"scope_policies": {"mainline_release": {"missing": "missing"}, "catalog_release_precedes": None}}):
            with self.assertRaises(ValueError):
                port.load_scoped_franchise(self.io(), "D:/old", {**params, **changes}, self.meta, franchise=self.kion_franchise_progress)


if __name__ == "__main__":
    unittest.main()
