import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import input_inventory as inventory


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "inputs"
        self.root.mkdir()
        (self.root / "file.bin").write_bytes(b"fixture")
        self.entry = {
            "id": "checkpoint", "source_root_alias": "PROJECT", "relative_path": "file.bin",
            "role": "dependency", "required": True, "expected_bytes": 7,
            "sha256": hashlib.sha256(b"fixture").hexdigest(),
        }
        self.manifest = {"schema_version": 1, "profile": "test", "closure_complete": True,
                         "entries": [self.entry]}
        self.roots = {"PROJECT": str(self.root)}

    def run_inventory(self, verify=True):
        return inventory.inspect(self.manifest, self.roots, verify)

    def test_inspect_never_opens_payload(self):
        with patch.object(Path, "open", side_effect=AssertionError("payload read")):
            report = self.run_inventory(False)
        self.assertFalse(report["ready"])
        self.assertEqual(report["counts"], {"unverified": 1})

    def test_matching_content_is_verified(self):
        report = self.run_inventory()
        self.assertTrue(report["ready"])
        self.assertEqual(report["counts"], {"verified": 1})

    def test_wrong_hash_and_size(self):
        self.entry["sha256"] = "0" * 64
        self.assertEqual(self.run_inventory()["counts"], {"hash_mismatch": 1})
        self.entry["expected_bytes"] = 8
        self.assertEqual(self.run_inventory()["counts"], {"size_mismatch": 1})

    def test_missing_unpinned_unresolved_and_incomplete(self):
        self.entry["relative_path"] = "missing"
        self.assertEqual(self.run_inventory()["counts"], {"missing": 1})
        self.entry["relative_path"] = "file.bin"
        self.entry["sha256"] = None
        self.assertEqual(self.run_inventory()["counts"], {"unpinned": 1})
        self.assertFalse(self.run_inventory()["ready"])
        self.roots = {}
        self.assertEqual(self.run_inventory()["counts"], {"unresolved_root": 1})

    def test_missing_size_never_counts_verified(self):
        del self.entry["expected_bytes"]
        self.assertEqual(self.run_inventory()["counts"], {"unpinned": 1})

    def test_optional_missing_and_declared_closure(self):
        self.manifest["entries"].append(dict(self.entry, id="old-output", role="provenance",
                                             required=False, relative_path="missing"))
        self.assertTrue(self.run_inventory()["ready"])
        self.manifest["closure_complete"] = False
        self.assertFalse(self.run_inventory()["ready"])

    def test_no_required_dependency_cannot_claim_ready(self):
        self.entry["required"] = False
        self.roots = {}
        with self.assertRaisesRegex(ValueError, "required dependency"):
            self.run_inventory()
        self.entry["role"] = "provenance"
        with self.assertRaisesRegex(ValueError, "required dependency"):
            self.run_inventory()
        self.entry["required"] = True
        with self.assertRaisesRegex(ValueError, "required dependency"):
            self.run_inventory()

    def test_unsafe_paths_are_rejected_before_read(self):
        for name in ("../escape", "/absolute", "C:/escape", "a\\b", "a//b", "a/./b",
                     "file:stream", "*.bin", "dir/../file", "NUL", "a. "):
            with self.subTest(name=name):
                self.entry["relative_path"] = name
                with self.assertRaises(ValueError):
                    self.run_inventory()

    def test_symlink_is_rejected(self):
        link = self.root / "link"
        try:
            link.symlink_to(self.root / "file.bin")
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable")
            raise
        self.entry["relative_path"] = "link"
        self.assertEqual(self.run_inventory()["counts"], {"unsafe_or_unreadable": 1})

    @unittest.skipUnless(__import__("os").name == "nt", "Windows junction test")
    def test_windows_junction_is_rejected(self):
        target = self.base / "outside"
        target.mkdir()
        (target / "file.bin").write_bytes(b"fixture")
        junction = self.root / "junction"
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(target)],
                       check=True, capture_output=True)
        self.addCleanup(junction.rmdir)
        self.entry["relative_path"] = "junction/file.bin"
        self.assertEqual(self.run_inventory()["counts"], {"unsafe_or_unreadable": 1})
        with self.assertRaises(ValueError):
            inventory.inspect(self.manifest, {"PROJECT": str(junction)})

    def test_cli_new_output_only_outside_inputs_and_exit_status(self):
        manifest = self.base / "manifest.json"
        roots = self.base / "roots.json"
        manifest.write_text(json.dumps(self.manifest), encoding="utf-8")
        roots.write_text(json.dumps(self.roots), encoding="utf-8")
        output = self.base / "report.json"
        args = ["--manifest", str(manifest), "--roots", str(roots), "--output", str(output)]
        self.assertEqual(inventory.main(args), 1)
        saved = output.read_bytes()
        self.assertEqual(inventory.main(args + ["--verify"]), 2)
        self.assertEqual(output.read_bytes(), saved)
        args[-1] = str(self.root / "report.json")
        self.assertEqual(inventory.main(args), 2)
        self.assertFalse((self.root / "report.json").exists())
        args[-1] = str(self.base / "verified.json")
        self.assertEqual(inventory.main(args + ["--verify"]), 0)


if __name__ == "__main__":
    unittest.main()
