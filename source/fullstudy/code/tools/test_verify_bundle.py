"""Tiny synthetic package-integrity checks; no producer imports or real inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location("bundle_verifier", Path(__file__).with_name("verify_bundle.py"))
verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verifier)


class PackageIntegrity(unittest.TestCase):
    def make_bundle(self, root):
        root = Path(root)
        (root / "code/fixtures").mkdir(parents=True)
        fixture = {"synthetic": True, "target_ranks": [1, 2, None],
                   "expected": {"queries": 3, "map10": .5, "recall10": 2 / 3,
                                "ndcg10": .5436432511904858}}
        path = root / "code/fixtures/synthetic-ranking.json"
        path.write_text(json.dumps(fixture), encoding="utf-8")
        source = root / "code/toy.py"
        source.write_text("answer = 42\n", encoding="utf-8")
        rows = [{"bundle_path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (path, source)]
        manifest = {"contract": "kion-small-source-bundle-v1", "files": rows,
                    "totals": {"files": len(rows), "bytes": sum(r["bytes"] for r in rows)}}
        (root / "code-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return root, manifest

    def test_intact_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.make_bundle(folder)
            self.assertEqual(verifier.verify(root)["state"], "passed")
            (root / "code/toy.py").write_text("answer = 43\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verifier.verify(root)

    def test_rejects_unlisted_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _ = self.make_bundle(folder)
            (root / "code/unlisted.py").write_text("pass\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Undeclared"):
                verifier.verify(root)

    def test_rejects_parent_traversal_and_absolute_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for relative in ("../outside.json", root.as_posix(), "code\\file.py"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    verifier.inside(root, relative)

    def test_rejects_invalid_python_without_importing_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest = self.make_bundle(folder)
            source = root / "code/toy.py"
            source.write_text("def broken(:\n", encoding="utf-8")
            row = manifest["files"][1]
            row.update(bytes=source.stat().st_size, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            manifest["totals"]["bytes"] = sum(r["bytes"] for r in manifest["files"])
            (root / "code-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SyntaxError):
                verifier.verify(root)


if __name__ == "__main__":
    unittest.main()
