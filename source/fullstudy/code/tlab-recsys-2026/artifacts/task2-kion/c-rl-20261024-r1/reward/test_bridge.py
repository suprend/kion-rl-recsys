"""Small synthetic tests only: never load data/model/provider/native runtimes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

spec = importlib.util.spec_from_file_location("C_rm_bridge", Path(__file__).with_name("bridge.py"))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.np = np


class Catalog:
    context_names = ["u" + str(i) for i in range(2079)]
    item_names = ["c" + str(i) for i in range(2008)]
    schema_signature = "synthetic"
    def indices(self, ids):
        ids = np.asarray(ids)
        if np.any((ids < 100) | (ids >= 1124)):
            raise ValueError("not raw IDs")
        return ids - 100


class Raw:
    def history_batch(self, users, origins):
        x = np.zeros((len(users), 64), np.int64)
        x[:, :2] = [1, 3]
        return x, np.full(len(users), 2), origins - 1


class Pack:
    user_columns, candidate_columns = Catalog.context_names, Catalog.item_names
    def __init__(self):
        self.calls = []
    def user_features(self, users, origins, x, lengths, history_cutoffs):
        np.testing.assert_array_equal(origins, history_cutoffs)
        self.origin = int(origins[0])
        return np.zeros((len(users), 2079), np.float32)
    def candidate_features(self, tokens, origins, history_cutoffs, user_ids):
        self.calls.append(len(tokens))
        np.testing.assert_array_equal(origins, history_cutoffs)
        assert np.all(origins == self.origin)
        result = np.zeros((len(tokens), 2008), np.float32)
        result[:, 0] = tokens
        return result


class Scorer:
    catalog = Catalog()
    priors = SimpleNamespace(counts=np.arange(1024) % 3, global_mean=0.4)
    def score_rows(self, rows):
        support = self.priors.counts[self.catalog.indices(rows["item_id"])] > 0
        return np.where(support, rows["candidate_numeric"][:, 0].astype(np.float64) / 2000., self.priors.global_mean)
    def score_with_diagnostics(self, contexts, ids):
        assert all(c["candidate_item_id"] == i for c, i in zip(contexts, ids))
        rows = {"item_id": ids, "candidate_numeric": np.asarray([c["candidate_numeric"] for c in contexts])}
        support = self.priors.counts[self.catalog.indices(ids)] > 0
        return {"rewards": self.score_rows(rows), "support": support, "fallback": ~support}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.pack = Pack()
        self.bridge = b.LocalRMBridge(self.pack, Raw(), Scorer(), {100 + i: 3 + i for i in range(1024)}, chunk_size=73)
        self.x = Raw().history_batch([1], np.array([b.ORIGIN_MIN + 7]))[0][0]
        self.ids = np.arange(100, 1124, dtype=np.int64)
    def score(self, **kw):
        values = dict(user_id=7, origin=b.ORIGIN_MIN + 7, candidate_raw_ids=self.ids, x=self.x, length=2)
        values.update(kw)
        return self.bridge.score_state(**values)
    def test_natural1024_chunk_support_clock_and_parity(self):
        result = self.score()
        self.assertEqual(self.pack.origin, b.ORIGIN_MIN + 7)
        self.assertLessEqual(max(self.pack.calls), 73)
        self.assertEqual(sum(self.pack.calls), 1025)
        np.testing.assert_array_equal(result["support"], Scorer.priors.counts > 0)
        self.assertEqual(result["parity_max_abs"], 0)
        np.testing.assert_array_equal(result["rewards"][result["fallback"]], 0.4)
    def test_reject_mismatched_state_and_raw_id(self):
        changed = self.x.copy()
        changed[2] = 9
        for kwargs in ({"x": changed}, {"length": 3}, {"candidate_raw_ids": np.array([3, 4])},
                       {"origin": b.ORIGIN_MIN - 1}, {"candidate_raw_ids": np.array([100, 100])}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.score(**kwargs)
    def test_reject_native_parity_failure(self):
        original = self.bridge.scorer.score_with_diagnostics
        def faulty(contexts, ids):
            result = original(contexts, ids)
            result["rewards"][0] += .01
            return result
        self.bridge.scorer.score_with_diagnostics = faulty
        with self.assertRaisesRegex(ValueError, "parity"):
            self.score()
    def test_reject_wrong_feature_order(self):
        self.pack.user_columns = list(reversed(self.pack.user_columns))
        with self.assertRaisesRegex(ValueError, "ordered"):
            b.LocalRMBridge(self.pack, Raw(), Scorer(), {})
    def test_reject_vectorized_candidate_feature_drift(self):
        original = self.pack.candidate_features
        def faulty(tokens, origins, history_cutoffs, user_ids):
            result = original(tokens, origins, history_cutoffs, user_ids)
            if len(tokens) > 1:
                result[0, 0] += 1
            return result
        self.pack.candidate_features = faulty
        with self.assertRaisesRegex(ValueError, "feature parity"):
            self.score()
    def test_cache_pins_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            values = {"user_ids": np.array([7]), "origins": np.array([b.ORIGIN_MIN + 7]),
                "x": self.x[None, :], "lengths": np.array([2]), "candidate_raw_ids": self.ids[None, :]}
            for name, value in values.items():
                np.save(cache / (name + ".npy"), value)
            meta = {"contract": "kion-current-C1070-natural1024-H1-v1", "state": "completed",
                "candidate_width": 1024, "raw_width": 1068, "no_target_injection": True,
                "no_seen_filter": True, "arrays": {name: {"path": name + ".npy",
                    "sha256": b.digest(cache / (name + ".npy")), "shape": list(value.shape),
                    "dtype": str(value.dtype)} for name, value in values.items()}}
            (cache / "cache.json").write_text(json.dumps(meta), encoding="utf-8")
            sources = np.zeros((1, 1024, 14), np.float32)
            sources[..., 3] = 1
            sources[:, 512:, 4] = 1
            sources[:, :100, 5] = 1
            np.save(cache / "candidate_sources.npy", sources)
            meta["arrays"]["candidate_sources"] = {"sha256": b.digest(cache / "candidate_sources.npy")}
            (cache / "cache.json").write_text(json.dumps(meta), encoding="utf-8")
            report = b.score_cache(self.bridge, cache, root / "out")
            self.assertTrue(report["ready"])
            self.assertEqual(report["rows"], 1024)
            self.assertEqual(report["native_pair_parity_max_abs"], 0)
            self.assertEqual(set(report["state_arrays"]), set(b.STATE_FIELDS) | {"candidate_sources"})
            self.assertEqual(report["candidate_strata"]["source_KNN"]["pairs"], 100)
            self.assertEqual(np.load(root / "out/rewards.npy").shape, (1, 1024))
            with self.assertRaises(FileExistsError):
                b.score_cache(self.bridge, cache, root / "out")
            values["origins"] += 1
            np.save(cache / "origins.npy", values["origins"])
            with self.assertRaisesRegex(ValueError, "policy descriptor"):
                b.score_cache(self.bridge, cache, root / "bad")
    def test_native_worker_wrapper_forwards_bounded_request_and_checks(self):
        calls = []
        request = {"repo": "D:/synthetic-repo", "cache_dir": "D:/synthetic-cache",
            "reward_output": "D:/synthetic-nonexistent-reward-output", "max_states": 32,
            "scoring_seconds": 60, "chunk_size": 17, "catboost_site": "synthetic-site",
            "catboost_pins": {"synthetic": "catboost"}, "numpy_pins": {"synthetic": "numpy"}}
        with patch.object(b, "load_runtime", return_value=(self.bridge, {"synthetic": True})) as loader, \
                patch.object(b, "bootstrap_numpy", return_value={"version": "2.5.4"}) as numerical, \
                patch.object(b, "bootstrap_catboost", return_value={"numpy_version": "2.5.4"}) as backend, \
                patch.object(b, "score_cache", return_value={"ready": True, "states": 32}) as score:
            result = b.run(request, lambda **event: calls.append(event))
            self.assertEqual(result, {"ready": True, "states": 32})
            self.assertEqual(self.bridge.chunk_size, 17)
            loader.assert_called_once_with(Path(request["repo"]))
            args, kwargs = score.call_args
            self.assertEqual(args, (self.bridge, Path(request["cache_dir"]), Path(request["reward_output"])))
            self.assertEqual(kwargs["max_states"], 32)
            self.assertEqual(kwargs["seconds"], 60)
            self.assertIs(kwargs["provenance"]["synthetic"], True)
            self.assertEqual(kwargs["provenance"]["execution_environment"]["numpy_version"], "2.5.4")
            numerical.assert_called_once_with(Path(request["repo"]), request["catboost_site"], request["numpy_pins"])
            backend.assert_called_once_with(Path(request["repo"]), request["catboost_site"], request["catboost_pins"])
            self.assertEqual([c["stage"] for c in calls], ["reward-start", "reward-runtime-ready", "reward-completed"])
        for wrong in ({"chunk_size": True}, {"chunk_size": 257}, {"max_states": 31},
                      {"scoring_seconds": float("nan")}, {"scoring_seconds": True}):
            with patch.object(b, "load_runtime") as loader, self.assertRaises(ValueError):
                b.run(dict(request, **wrong))
            loader.assert_not_called()
        with patch.object(b, "load_runtime") as loader, self.assertRaisesRegex(ValueError, "Explicit isolated"):
            b.run({key: value for key, value in request.items() if key != "numpy_pins"})
        loader.assert_not_called()
    def test_bridge_import_is_numerically_inert(self):
        spec = importlib.util.spec_from_file_location("fresh_inert_bridge", Path(__file__).with_name("bridge.py"))
        fresh = importlib.util.module_from_spec(spec)
        with patch.object(b.importlib, "import_module") as importer:
            spec.loader.exec_module(fresh)
        importer.assert_not_called()
        self.assertIsNone(fresh.np)
    def test_numpy_preimport_pins_fresh_process_path_and_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            site = repo / ".venv/Lib/site-packages"
            package = site / "numpy"
            (package / "_core").mkdir(parents=True)
            native = "_core/_multiarray_umath.cp312-win_amd64.pyd"
            for filename in ("__init__.py", native):
                (package / filename).write_bytes(b"fake unexecutable NumPy")
            pins = {filename: b.digest(package / filename) for filename in ("__init__.py", native)}
            fake = SimpleNamespace(__file__=str(package / "__init__.py"), __version__="2.5.4")
            extension = SimpleNamespace(__file__=str(package / native))
            def importer(name):
                self.assertEqual(b.sys.path[0], str(site.resolve()))
                self.assertTrue(all(b.digest(package / filename) == pin for filename, pin in pins.items()))
                return fake if name == "numpy" else extension
            with patch.object(b, "np", np), patch.object(b.sys, "path", list(b.sys.path)), \
                    patch.object(b, "loaded_numerical_modules", return_value=[]), \
                    patch.object(b.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(package / "__init__.py"))), \
                    patch.object(b.importlib, "import_module", side_effect=importer) as imports:
                report = b.bootstrap_numpy(repo, site, pins)
                self.assertIs(b.np, fake)
                self.assertEqual(report["version"], "2.5.4")
                self.assertEqual(imports.call_count, 3)
                imports.reset_mock()
                with patch.object(b, "loaded_numerical_modules", return_value=["numpy"]):
                    with self.assertRaisesRegex(ValueError, "fresh process"):
                        b.bootstrap_numpy(repo, site, pins)
                imports.assert_not_called()
                fake.__version__ = "2.2.6"
                with self.assertRaisesRegex(ValueError, "version"):
                    b.bootstrap_numpy(repo, site, pins)
                fake.__version__ = "2.5.4"
                (package / native).write_bytes(b"changed")
                imports.reset_mock()
                with self.assertRaisesRegex(ValueError, "pin mismatch"):
                    b.bootstrap_numpy(repo, site, pins)
                imports.assert_not_called()
    def test_backend_all_pins_checked_before_import_and_runtime_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            site = repo / ".venv/Lib/site-packages"
            package = site / "catboost"
            package.mkdir(parents=True)
            names = {"catboost": "__init__.py", "catboost.core": "core.py", "catboost._catboost": "_catboost.pyd"}
            for filename in names.values():
                (package / filename).write_bytes(b"synthetic unexecutable backend")
            pins = {filename: b.digest(package / filename) for filename in names.values()}
            modules = {name: SimpleNamespace(__file__=str(package / filename), __version__="1.2.10") for name, filename in names.items()}
            def fake_import(name):
                for filename, pin in pins.items():
                    self.assertEqual(b.digest(package / filename), pin)
                return modules[name]
            with patch.dict(b.sys.modules, {name: None for name in names}), \
                    patch.object(b.sys, "path", list(b.sys.path)), \
                    patch.object(b.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(package / "__init__.py"))), \
                    patch.object(b.importlib, "import_module", side_effect=fake_import) as importer:
                report = b.bootstrap_catboost(repo, site, pins)
                self.assertEqual(report["version"], "1.2.10")
                self.assertEqual(report["numpy_version"], np.__version__)
                self.assertEqual(b.sys.path[-1], str(site.resolve()))
                self.assertEqual(set(report["files"]), set(pins))
                self.assertEqual(importer.call_count, 4)
                (package / "_catboost.pyd").write_bytes(b"changed")
                importer.reset_mock()
                with self.assertRaisesRegex(ValueError, "pin mismatch"):
                    b.bootstrap_catboost(repo, site, pins)
                importer.assert_not_called()
    def test_backend_rejects_path_resolution_and_wrong_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            site = repo / ".venv/Lib/site-packages"
            package = site / "catboost"
            package.mkdir(parents=True)
            for filename in ("__init__.py", "core.py", "_catboost.pyd"):
                (package / filename).write_bytes(b"synthetic")
            pins = {filename: b.digest(package / filename) for filename in ("__init__.py", "core.py", "_catboost.pyd")}
            names = ("catboost", "catboost.core", "catboost._catboost")
            with patch.dict(b.sys.modules, {name: None for name in names}), \
                    patch.object(b.sys, "path", list(b.sys.path)), \
                    patch.object(b.importlib, "import_module") as importer:
                with self.assertRaisesRegex(ValueError, "exact existing"):
                    b.bootstrap_catboost(repo, repo / "elsewhere", pins)
                importer.assert_not_called()
                with patch.object(b.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(repo / "other/__init__.py"))):
                    with self.assertRaisesRegex(ValueError, "resolution"):
                        b.bootstrap_catboost(repo, site, pins)
                importer.assert_not_called()
                importer.return_value = SimpleNamespace(__file__=str(package / "__init__.py"), __version__="1.2.9")
                with patch.object(b.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(package / "__init__.py"))):
                    with self.assertRaisesRegex(ValueError, "version"):
                        b.bootstrap_catboost(repo, site, pins)


if __name__ == "__main__":
    unittest.main()
