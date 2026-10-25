"""Tiny local component fixtures only; no historical payload or ML execution."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gpt_provider as provider


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = []
        self.check = Mock()

    def bind(self, logical, data, role="metadata"):
        path = self.root / str(len(self.bindings))
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.bindings.append(provider.FileBinding(logical, path.as_posix(), digest, role))
        return digest

    def io(self):
        return provider.BoundIO(self.bindings, self.check)

    def test_import_does_not_load_numerical_modules(self):
        import builtins
        real_import = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "transformers"):
                raise AssertionError("eager numerical import")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            spec = importlib.util.spec_from_file_location("gpt_provider", provider.__file__)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))

    def test_signed_metadata_survives_two_relocations(self):
        value = {"path": "D:/old/payload.npy", "scope": "scoped_production"}
        value["signature"] = provider.canonical_hash(value)
        data = json.dumps(value).encode()
        self.bind("D:/old/manifest.json", data)
        io = self.io()
        self.assertEqual(io.read_json("D:/old/manifest.json", signed=True), value)
        second = self.root / "relocated.json"
        second.write_bytes(data)
        other = provider.BoundIO([provider.FileBinding("D:/old/manifest.json", second.as_posix(),
                                  hashlib.sha256(data).hexdigest(), "metadata")], self.check)
        self.assertEqual(other.read_json("D:/old/manifest.json", signed=True), value)
        self.assertGreater(self.check.call_count, 0)

    def test_invalid_signature_pin_role_and_missing_binding(self):
        self.bind("D:/old/manifest.json", b'{"signature":"bad"}')
        io = self.io()
        with self.assertRaisesRegex(ValueError, "signature"):
            io.read_json("D:/old/manifest.json", signed=True)
        with self.assertRaisesRegex(ValueError, "role"):
            io.resolve("D:/old/manifest.json", "payload")
        with self.assertRaises(KeyError):
            io.resolve("D:/old/missing.npy", "payload")
        with self.assertRaisesRegex(ValueError, "pin"):
            io.verify("D:/old/manifest.json", "metadata", "0" * 64)

    def test_duplicates_escape_cap_and_cancellation(self):
        self.bind("D:/old/a.json", b"{}")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            provider.BoundIO(self.bindings * 2, self.check)
        for path in ("../escape", "C:relative", "x/../escape"):
            with self.assertRaises(ValueError):
                provider.logical_child("D:/old", path)
        with patch.object(provider, "METADATA_LIMIT", 1):
            with self.assertRaisesRegex(ValueError, "size"):
                self.io().read_json("D:/old/a.json")
        self.check.side_effect = TimeoutError("deadline")
        with self.assertRaises(TimeoutError):
            self.io().read_json("D:/old/a.json")

    def test_scoped_base_schema_has_explicit_larger_read_and_verify_cap(self):
        logical = "D:/features/manifest.json"
        self.bind(logical, b'{"base":true}')
        io = self.io()
        with patch.object(provider, "METADATA_LIMIT", 8), patch.object(provider, "SCOPED_BASE_MANIFEST_LIMIT", 16):
            for call in (lambda: io.read_json(logical), lambda: io.verify(logical, "metadata")):
                with self.assertRaisesRegex(ValueError, "size"):
                    call()
            self.assertEqual(io.read_json(logical, metadata_schema="scoped-base-manifest"), {"base": True})
            self.assertTrue(io.verify(logical, "metadata", metadata_schema="scoped-base-manifest").is_file())
            with self.assertRaisesRegex(ValueError, "schema"):
                io.read_json(logical, metadata_schema="arbitrary")
            with self.assertRaisesRegex(ValueError, "role"):
                io.verify(logical, "payload", metadata_schema="scoped-base-manifest")
        with patch.object(provider, "SCOPED_BASE_MANIFEST_LIMIT", 8):
            with self.assertRaisesRegex(ValueError, "size"):
                io.read_json(logical, metadata_schema="scoped-base-manifest")
            with self.assertRaisesRegex(ValueError, "size"):
                io.verify(logical, "metadata", metadata_schema="scoped-base-manifest")

    def array(self, logical, values):
        import io
        np = provider.numpy_backend()
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        return self.bind(logical, buffer.getvalue(), "payload")

    def test_matrix_exact_validation_and_cleanup(self):
        np = provider.numpy_backend()
        values = np.zeros((4, 2), np.float32)
        values[3] = [1, 2]
        digest = self.array("D:/old/m.npy", values)
        declaration = {"path": "m.npy", "sha256": digest, "metadata_sha256": "m", "vocabulary_sha256": "v",
                       "shape": [4, 2], "dtype": "float32", "columns": ["a", "b"]}
        original = copy.deepcopy(declaration)
        loaded = provider.read_matrix(self.io(), "D:/old", declaration, 4, "m", "v", np=np)
        self.assertEqual(loaded.tolist(), values.tolist())
        self.assertFalse(loaded.flags.writeable)
        provider._close_array(loaded)
        self.assertEqual(declaration, original)
        bad = dict(declaration, columns=["a", "a"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            provider.read_matrix(self.io(), "D:/old", bad, 4, "m", "v", np=np)
        with self.assertRaisesRegex(ValueError, "metadata"):
            provider.read_matrix(self.io(), "D:/old", declaration, 4, "wrong", "v", np=np)

    def test_matrix_rejects_special_rows_and_nonfinite(self):
        np = provider.numpy_backend()
        for index, value in ((0, 1), (3, float("nan"))):
            logical = "D:/old/m" + str(index) + ".npy"
            values = np.zeros((4, 1), np.float32)
            values[index] = value
            digest = self.array(logical, values)
            d = {"path": logical, "sha256": digest, "metadata_sha256": "m", "vocabulary_sha256": "v",
                 "shape": [4, 1], "dtype": "float32", "columns": ["a"]}
            with self.assertRaises(ValueError):
                provider.read_matrix(self.io(), "D:/old", d, 4, "m", "v", np=np)

    def test_scoped_base_data_executes_bound_component_reader(self):
        np = provider.numpy_backend()
        inputs = {"metadata.json": {"sha256": self.bind("D:/data/metadata.json", b'{"vocab_size":4,"context":64}')}}
        for name in ("train.npz", "val.npz"):
            inputs[name] = {"sha256": self.bind("D:/data/" + name, b"mock split", "payload")}
        outputs = {}
        for name, shape in (("item_features.npy", (4, 2)), ("train_user_features.npy", (2, 1)), ("val_user_features.npy", (2, 1))):
            outputs[name] = {"sha256": self.array("D:/features/" + name, np.zeros(shape, np.float32)),
                             "shape": list(shape), "dtype": "float32"}
        columns = {"item": ["a", "b"], "user": ["c"]}
        manifest = {"inputs": inputs, "outputs": outputs, "feature_columns": columns,
                    "schema_hash": provider.canonical_hash(columns), "rowkeys_algorithm": "retained-row-key",
                    "row_keys_sha256": {"train": "rows", "val": "rows"}}
        digest = self.bind("D:/features/manifest.json", json.dumps(manifest).encode())
        load = Mock(side_effect=lambda *a, **k: {"y": np.array([3, 3])})
        baseline = SimpleNamespace(FeatureData=dict, shared=SimpleNamespace(load_data=load),
                                   _columns=Mock(), ROWKEYS_ALGORITHM="retained-row-key", make_row_keys_hash=lambda _: "rows")
        # Base manifest exceeds the mocked ordinary cap; metadata.json does not.
        # The component reader must explicitly select the larger schema policy.
        with patch.object(provider, "METADATA_LIMIT", 64):
            result = provider.load_scoped_base_data(self.io(), "D:/data", "D:/features",
                                                   manifest_sha256=digest, np=np, baseline=baseline)
        self.assertEqual(result[0]["context"], 64)
        self.assertEqual(result[-1], manifest)
        self.assertEqual([c.kwargs for c in load.call_args_list],
                         [{"require_keys": True, "unique_keys": False}, {"require_keys": True, "unique_keys": True}])
        self.assertNotEqual(load.call_args_list[0].args[0].as_posix(), "D:/data/train.npz")
        for data in result[1:3]:
            self.assertFalse(data["y"].flags.writeable)
            provider._close_array(data["user_features"])
        provider._close_array(result[3])


if __name__ == "__main__":
    unittest.main()
