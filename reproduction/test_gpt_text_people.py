"""Tiny mock-array reads only; no historical payloads or ML dependencies."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gpt_provider as p
import gpt_text_people as t


class Readers(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bindings = []
        self.check = Mock()

    def bind(self, logical, data, role):
        path = self.root / str(len(self.bindings))
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        self.bindings.append(p.FileBinding(logical, path.as_posix(), digest, role))
        return {"path": logical, "sha256": digest}

    def io(self):
        return p.BoundIO(self.bindings, self.check)

    def text_fixture(self, mutate=None):
        np = self.np
        arrays = {"embeddings": np.zeros((4, 384), np.float32),
                  "missing": np.array([1, 1, 1, 0], bool),
                  "token_lengths": np.array([0, 0, 0, 513], np.int32),
                  "truncated": np.array([0, 0, 0, 1], bool)}
        arrays["embeddings"][3, 0] = 1
        manifest = {"contract": t.CONTRACT, "state": "completed", "field": "title",
                    "encoder": {"production_encoder": True, "model_id": t.MODEL_ID,
                                "revision": t.REVISION, "tokenizer_revision": t.REVISION,
                                "dimension": t.DIMENSION, "files": t.MODEL_FILES},
                    "preprocessing": t.PREPROCESSING, "vocab_size": 4,
                    "ordered_item_ids": ["42"],
                    "vocabulary_sha256": p.canonical_hash({"ordered_item_ids": ["42"], "special_tokens": t.SPECIALS}),
                    "source_pins": {"items": self.bind("D:/old/items.csv", b"tiny", "payload"),
                                    "metadata": self.bind("D:/old/metadata.json", b"{}", "metadata")},
                    "code_sha256": self.bind("D:/old/text.py", b"# retained fixture", "source")["sha256"],
                    "outputs": {}}
        for name, array in arrays.items():
            pin = self.bind("D:/old/cache/" + name + ".npy", name.encode(), "payload")
            manifest["outputs"][name + ".npy"] = {"sha256": pin["sha256"], "shape": list(array.shape), "dtype": str(array.dtype)}
        if mutate:
            mutate(manifest, arrays)
        manifest["signature"] = p.canonical_hash({k: v for k, v in manifest.items() if k not in ("state", "outputs", "coverage", "signature")})
        self.bind("D:/old/cache/manifest.json", json.dumps(manifest).encode(), "metadata")
        backend = Mock(wraps=np)
        backend.float32, backend.int32 = np.float32, np.int32
        backend.load.side_effect = [arrays[k] for k in ("embeddings", "missing", "token_lengths", "truncated")]
        return manifest, arrays, backend

    def load_text(self, backend, **kwargs):
        return t.load_item_cache(self.io(), "D:/old/cache", source_logical="D:/old/text.py", np=backend, **kwargs)

    def test_text_interface_and_explicit_load(self):
        manifest, arrays, backend = self.text_fixture()
        cache = self.load_text(backend, expected_signature=manifest["signature"], expected_vocabulary_sha256=manifest["vocabulary_sha256"])
        self.assertEqual(cache.manifest, manifest)
        self.assertEqual(backend.load.call_count, 4)
        for name, array in arrays.items():
            self.assertIs(getattr(cache, name), array)
            self.assertFalse(array.flags.writeable)
        for call in backend.load.call_args_list:
            self.assertEqual(call.kwargs, {"allow_pickle": False})
            self.assertEqual(call.args[0].parent, self.root.resolve())

    def test_text_rejects_encoder_before_array_reads(self):
        _, _, backend = self.text_fixture(lambda m, a: m["encoder"].update(revision="wrong"))
        with self.assertRaisesRegex(ValueError, "encoder pins"):
            self.load_text(backend)
        backend.load.assert_not_called()

    def test_array_predicates(self):
        mutations = [lambda a: a["embeddings"].__setitem__((3, 0), 2),
                     lambda a: a["embeddings"].__setitem__((0, 0), 1),
                     lambda a: a["embeddings"].__setitem__((3, 0), self.np.nan),
                     lambda a: a["token_lengths"].__setitem__(3, 0),
                     lambda a: a["truncated"].__setitem__(3, False),
                     lambda a: a["missing"].__setitem__(0, False)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.bindings = []
                _, _, backend = self.text_fixture(lambda m, a: mutate(a))
                with self.assertRaisesRegex(ValueError, "Invalid text cache"):
                    self.load_text(backend)

    def test_text_pin_and_signature_fail_closed(self):
        manifest, _, backend = self.text_fixture()
        with self.assertRaisesRegex(ValueError, "signature"):
            self.load_text(backend, expected_signature="bad")
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            self.load_text(backend, expected_vocabulary_sha256="bad")
        path = self.io().resolve("D:/old/text.py", "source")
        path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.load_text(backend)
        backend.load.assert_not_called()

    def people_fixture(self, mutate=None):
        sources = {name: self.bind("D:/old/" + name, name.encode(), "source") for name in ("kion_people_embeddings.py", "kion_pack_people.py")}
        identities = {"items.csv": self.bind("D:/old/items.csv", b"tiny", "payload"),
                      "metadata": self.bind("D:/old/metadata.json", b"{}", "metadata")}
        catalog = t._rebuild_people_catalog({"42": {"actors": " Alice, Bob, ALICE", "directors": ""}},
                    {"vocab_size": 4, "item_id_to_token": {"42": 3}}, source_identity=identities)
        catalog["source_pins"] = {k: v["sha256"] for k, v in sources.items()}
        if mutate:
            mutate(catalog)
        catalog["signature"] = t._signature(catalog)
        self.bind("D:/old/catalog.json", json.dumps(catalog).encode(), "metadata")
        return catalog, {k: v["path"] for k, v in sources.items()}

    def test_people_rederivation_and_logical_identity(self):
        catalog, sources = self.people_fixture()
        result = t.load_catalog(self.io(), "D:/old/catalog.json", source_bindings=sources,
                    expected_signature=catalog["signature"], expected_sources=catalog["source_identity"])
        self.assertEqual(result, catalog)
        self.assertEqual(result["roles"]["actors"]["keys"], ["alice", "bob"])
        with self.assertRaisesRegex(ValueError, "source identity"):
            t.load_catalog(self.io(), "D:/old/catalog.json", source_bindings=sources, expected_sources={})

    def test_people_tamper_with_valid_signature_rejected(self):
        _, sources = self.people_fixture(lambda c: c["roles"]["actors"]["top3_occupancy"][3].__setitem__(0, 0))
        with self.assertRaisesRegex(ValueError, "Malformed people"):
            t.load_catalog(self.io(), "D:/old/catalog.json", source_bindings=sources)

    def test_import_stdlib_only(self):
        import builtins
        real = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split(".")[0] in ("numpy", "torch", "transformers"):
                raise AssertionError(name)
            return real(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guard):
            spec = importlib.util.spec_from_file_location("text_people_import_test", t.__file__)
            spec.loader.exec_module(importlib.util.module_from_spec(spec))


if __name__ == "__main__":
    unittest.main()
