import copy
from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from gpt_provider import BoundIO, FileBinding
import gpt_runtime as runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        self.add("items.csv", "item_id,genres,content_type,release_year,countries,age_rating,actors,directors,studios\n10,Drama,film,2000,RU,16.0,A,D,S\n")
        self.add("interactions.csv", "user_id,item_id,last_watch_dt,watched_pct\n1,10,2021-08-08,50\n2,10,2021-08-09,80\n")
        self.add("annotation.json", json.dumps({"series": [{"id": "f", "status": "confirmed", "members": [{"item_id": 10, "status": "confirmed"}]}]}), "metadata")
        self.add("history.py", "# admitted fixture source\n", "source")
        self.io = BoundIO(self.rows, lambda: None)
        self.meta = {"item_id_to_token": {"10": 3}, "vocab_size": 4,
                     "series_annotation_path": "D:/logical/annotation.json"}

    def add(self, name, value, role="payload"):
        path = self.root / name
        path.write_text(value, encoding="utf-8", newline="")
        self.rows.append(FileBinding("D:/logical/" + name, path.as_posix(),
                                    hashlib.sha256(path.read_bytes()).hexdigest(), role))

    def test_csv_and_pin_failures(self):
        self.assertEqual(list(runtime.csv_rows(self.io, "D:/logical/items.csv", ("item_id", "actors"))), [("10", "A")])
        with self.assertRaises(ValueError):
            list(runtime.csv_rows(self.io, "D:/logical/items.csv", ("absent",)))
        (self.root / "items.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(runtime.csv_rows(self.io, "D:/logical/items.csv", ("item_id",)))

    def test_history_parsed_records_and_logical_provenance(self):
        calls = []
        def constructor(*args, **kwargs):
            calls.append((args, kwargs))
            return SimpleNamespace(provenance={})
        b = SimpleNamespace(np=np, _day=lambda x: date.fromisoformat(x).toordinal(),
            parse_outcome=lambda x: (float(x)/100, 0, 4), HistoryIndex=constructor)
        source = self.rows[-1]
        result = runtime.build_history(self.io, "D:/logical", self.meta, backend=b,
            builder_source={"path": source.logical, "sha256": source.sha256})
        args, kwargs = calls[0]
        np.testing.assert_array_equal(args[0], [1, 2])
        self.assertEqual(args[3].dtype, np.float32)
        self.assertEqual(args[6][10]["release_year"], "2000")
        self.assertFalse(kwargs["include_relative"])
        self.assertEqual(kwargs["fit_cutoff"], "2021-08-09")
        self.assertEqual(result.source_pins["history_builder"]["path"], source.logical)
        self.assertEqual(result.source_pins["items"]["path"], "D:/logical/items.csv")

    def test_personal_records_and_unmodified_metadata(self):
        original = copy.deepcopy(self.meta)
        calls = []
        def constructor(index, **kwargs):
            calls.append(kwargs)
            return kwargs
        b = SimpleNamespace(np=np, ROLES=("actor", "director", "studio", "franchise"),
                            exact_keys=lambda x: (x,), PersonalCandidateIndex=constructor)
        index = SimpleNamespace(mapping={10: 3}, vocab_size=4)
        result = runtime.build_personal(self.io, "D:/logical", self.meta, backend=b,
            history_index=index, annotation_logical=self.meta["series_annotation_path"],
            representation_manifest={"title": "signature"})
        self.assertEqual(result["key_matrices"]["franchise"]["key_columns"], ["f"])
        np.testing.assert_array_equal(result["key_matrices"]["actor"]["offsets"], [0, 0, 0, 0, 1])
        self.assertEqual(result["representation_manifest"]["title"], "signature")
        self.assertEqual(self.meta, original)

    def test_dynamic_strict_cutoff_and_relations(self):
        b = SimpleNamespace(np=np, _day=lambda x: date.fromisoformat(x).toordinal(),
            _parse_outcome=lambda x: (float(x)/100, 0), DAY_BASE=3652060,
            N_STATS=18, PAIR=0, OBS=1, SUM=2, MISSING=3, INVALID=4, ACTIVE=10, WEEKDAY=11,
            ROLES=("actors", "directors", "studios", "series"), _tags=lambda x: (x,))
        keys, values, diagnostics = runtime._interaction_table(self.io,
            "D:/logical/interactions.csv", self.meta["item_id_to_token"], 4,
            date(2021, 8, 9).toordinal(), b)
        self.assertEqual(diagnostics["source_rows"], 2)
        self.assertEqual(diagnostics["indexed_rows"], 1)
        self.assertEqual(values[0, 2], .5)
        self.assertEqual(len(keys), 1)
        tables, path = runtime._relations(self.io, "D:/logical",
            self.meta["series_annotation_path"], self.meta["item_id_to_token"], 4, b)
        np.testing.assert_array_equal(tables["series"]["entities"], [0])
        self.assertEqual(path, self.meta["series_annotation_path"])
        b._metadata = lambda meta: (meta, meta["item_id_to_token"])
        b._Prefix = lambda keys, values, size: SimpleNamespace(keys=keys, nbytes=17)
        b._entity_prefix = lambda keys, values, relation: SimpleNamespace(keys=keys, nbytes=3)
        b._canonical_hash = lambda mapping: "mapping-hash"
        b.CONTRACT = "fixture"
        b.ALPHA, b.WEEKDAY_BETA, b.POP_ALPHA, b.TREND_LAMBDA, b.TREND_CLIP = 10., 7., 1., 10., 3.
        b.DynamicIndex = lambda *args: args
        result = runtime.build_dynamic(self.io, "D:/logical", self.meta, backend=b,
                                       annotation_logical=self.meta["series_annotation_path"])
        self.assertEqual(result[-1]["index_bytes"], 46)
        self.assertEqual(result[-1]["parameters"]["completion_constants"], [.5, .1, .8, .5, .2, .05])
        self.assertEqual(result[-1]["series_sha256"], self.rows[2].sha256)

    def test_scoped_rejections(self):
        for forbidden in ("audience", "people_dynamics", "entity_links"):
            with self.assertRaises(ValueError):
                runtime.prepare_scoped_inputs(self.io, "D:/logical", self.meta, {forbidden: True})
        with self.assertRaises(ValueError):
            runtime.prepare_scoped_inputs(self.io, "D:/logical", self.meta,
                {"raw_dir": "D:/logical", "personal_candidates": True})
        pins = [{"path": r.logical, "sha256": r.sha256} for r in self.rows[:-1]]
        self.assertEqual(runtime.prepare_scoped_inputs(self.io, "D:/logical", self.meta,
            {"raw_dir": "D:/logical", "local_dynamic": True, "source_pins": pins}),
            ("D:/logical", "D:/logical/annotation.json"))

    def test_aggregator_scoped_columns_and_admitted_backends(self):
        loaded = []
        context = SimpleNamespace(load=lambda name: loaded.append(name) or object(),
            logical_source=lambda module: "D:/logical/history.py")
        provider = SimpleNamespace(scoped=True, partial=False, root="D:/logical",
            metadata=self.meta, _require_sources=lambda modules: None,
            _columns={"user": [], "candidate": []}, materialized_families=set())
        pins = [{"path": r.logical, "sha256": r.sha256} for r in self.rows[:-1]]
        with patch.object(runtime, "build_history", return_value=SimpleNamespace(provenance={"history": 1})), \
             patch.object(runtime, "build_dynamic", return_value=SimpleNamespace(columns=["a"], provenance={"dynamic": 1})):
            runtime.init_scoped_runtime(provider,
                {"raw_dir": "D:/logical", "local_dynamic": True, "source_pins": pins},
                io=self.io, runtime=context)
        self.assertEqual(loaded, ["kion_pack_history", "kion_pack_dynamic"])
        self.assertEqual(provider._columns["candidate"], ["local_dynamic.a"])
        self.assertEqual(provider.materialized_families, {"local_dynamic"})
        self.assertIsNone(provider.audience)


if __name__ == "__main__":
    unittest.main()
