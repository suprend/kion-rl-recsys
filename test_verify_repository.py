"""Synthetic standard-library checks; fixtures live outside the real checkout.

Run: python -I -S -B -m unittest discover -s . -p test_verify_repository.py -v
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import verify_repository


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="projection-verifier-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.assertFalse(self.root.is_relative_to(Path(__file__).resolve().parent))
        for relative in verify_repository.AUTHORED:
            self.write(relative, b"synthetic authored file\n")
        self.copied = [self.copy("source/README.md", b"frozen source note\n"),
                       self.copy("source/frozen.py", b"# frozen source\n")]
        self.excluded = [
            {"path": "README.md", "bytes": 9, "sha256": digest(b"old note\n"),
             "reason": "Replaced by repository documentation.",
             "disposition": "replaced-by-repository-documentation"},
            {"path": "reproduction/inputs/fixture.npy", "bytes": 7,
             "sha256": digest(b"omitted"), "reason": "Numerical fixture excluded.",
             "disposition": "omitted"},
        ]
        self.manifest = {
            "contract": "kion-source-only-git-projection-v1",
            "authored_files_not_hashed": sorted(verify_repository.AUTHORED),
            "copied_files": self.copied,
            "excluded_files": self.excluded,
        }
        self.accounting()

    def write(self, relative: str, payload: bytes):
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    def copy(self, relative: str, payload: bytes) -> dict:
        self.write(relative, payload)
        return {"path": relative, "bytes": len(payload), "sha256": digest(payload),
                "upstream_source": relative}

    def accounting(self):
        copied_bytes = sum(row["bytes"] for row in self.copied)
        excluded_bytes = sum(row["bytes"] for row in self.excluded)
        self.manifest["copied_totals"] = {"files": len(self.copied), "bytes": copied_bytes}
        self.manifest["excluded_totals"] = {"files": len(self.excluded), "bytes": excluded_bytes}
        self.manifest["upstream"] = {"payload_files": len(self.copied) + len(self.excluded),
                                     "payload_bytes": copied_bytes + excluded_bytes}

    def override(self, payload: bytes = b"reviewed source note with more context\n") -> dict:
        original = self.copied[0]
        self.write(original["path"], payload)
        row = {"path": original["path"], "baseline_sha256": original["sha256"],
               "bytes": len(payload), "sha256": digest(payload),
               "reason": "Reviewed clarification of the source/report boundary."}
        self.manifest["reviewed_copy_overrides"] = [row]
        return row

    def verify(self):
        self.write("projection.json", (json.dumps(self.manifest) + "\n").encode("utf-8"))
        return verify_repository.verify(self.root)

    def test_optional_overrides_default_to_exact_baseline(self):
        result = self.verify()
        self.assertEqual(result["copied_files"], 2)
        self.assertEqual(result["unchanged_copied_files"], 2)
        self.assertEqual(result["reviewed_override_files"], 0)
        self.assertEqual(result["current_copied_bytes"], result["copied_bytes"])
        self.assertEqual(result["excluded_members"], 2)

    def test_reviewed_override_uses_current_pin_and_keeps_historical_accounting(self):
        historical = dict(self.manifest["copied_totals"])
        upstream = dict(self.manifest["upstream"])
        row = self.override()
        result = self.verify()
        self.assertEqual(self.manifest["copied_totals"], historical)
        self.assertEqual(self.manifest["upstream"], upstream)
        self.assertEqual(result["copied_bytes"], historical["bytes"])
        self.assertEqual(result["copied_bytes_basis"], "historical baseline")
        self.assertEqual(result["unchanged_copied_files"], 1)
        self.assertEqual(result["unchanged_copied_bytes"], self.copied[1]["bytes"])
        self.assertEqual(result["reviewed_override_files"], 1)
        self.assertEqual(result["reviewed_override_baseline_bytes"], self.copied[0]["bytes"])
        self.assertEqual(result["reviewed_override_current_bytes"], row["bytes"])
        self.assertEqual(result["current_copied_bytes"], self.copied[1]["bytes"] + row["bytes"])

    def test_wrong_override_baseline_is_rejected(self):
        self.override()["baseline_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "baseline SHA256 differs"):
            self.verify()

    def test_override_cannot_authorize_an_arbitrary_copied_path(self):
        self.override()["path"] = "source/frozen.py"
        with self.assertRaisesRegex(ValueError, "Unapproved reviewed-copy override path"):
            self.verify()

    def test_allowlisted_override_still_must_be_a_copied_path(self):
        self.override()["path"] = "reports/REPORT.md"
        self.write("source/README.md", b"frozen source note\n")
        with self.assertRaisesRegex(ValueError, "override is not a copied path"):
            self.verify()

    def test_missing_or_blank_override_reason_is_rejected(self):
        row = self.override()
        for reason in ("", " \t\n", None, 123):
            with self.subTest(reason=reason):
                row["reason"] = reason
                with self.assertRaisesRegex(ValueError, "Missing reviewed-copy override reason"):
                    self.verify()

    def test_duplicate_override_is_rejected(self):
        row = self.override()
        self.manifest["reviewed_copy_overrides"].append(dict(row))
        with self.assertRaisesRegex(ValueError, "Duplicate reviewed-copy override"):
            self.verify()

    def test_override_requires_the_exact_record_fields(self):
        row = self.override()
        row["allow_anything"] = True
        with self.assertRaisesRegex(ValueError, "Invalid reviewed-copy override record"):
            self.verify()
        del row["allow_anything"]
        del row["reason"]
        with self.assertRaisesRegex(ValueError, "Invalid reviewed-copy override record"):
            self.verify()

    def test_override_list_and_values_are_validated(self):
        self.manifest["reviewed_copy_overrides"] = {}
        with self.assertRaisesRegex(ValueError, "Invalid reviewed-copy override list"):
            self.verify()
        row = self.override()
        for value in (-1, True, "5"):
            with self.subTest(size=value):
                row["bytes"] = value
                with self.assertRaisesRegex(ValueError, "Invalid reviewed-copy override size"):
                    self.verify()
        row = self.override()
        for field in ("baseline_sha256", "sha256"):
            original = row[field]
            for value in ("0" * 63, "G" * 64, None):
                with self.subTest(field=field, value=value):
                    row[field] = value
                    with self.assertRaisesRegex(ValueError, "Invalid reviewed-copy override " + field):
                        self.verify()
            row[field] = original

    def test_current_override_hash_and_size_are_both_strict(self):
        row = self.override()
        self.write(row["path"], b"X" * row["bytes"])
        with self.assertRaisesRegex(ValueError, "Changed SHA256"):
            self.verify()
        self.write(row["path"], b"short\n")
        with self.assertRaisesRegex(ValueError, "Missing or changed size"):
            self.verify()

    def test_untouched_copy_cannot_drift(self):
        self.override()
        original = self.copied[1]
        self.write(original["path"], b"X" * original["bytes"])
        with self.assertRaisesRegex(ValueError, "Changed SHA256: source/frozen.py"):
            self.verify()

    def test_override_does_not_change_baseline_totals(self):
        row = self.override()
        self.manifest["copied_totals"]["bytes"] += row["bytes"] - self.copied[0]["bytes"]
        with self.assertRaisesRegex(ValueError, "Copied totals differ"):
            self.verify()

    def test_unexpected_source_file_is_rejected_even_with_override(self):
        self.override()
        self.write("source/new_unauthorized.py", b"# extra\n")
        with self.assertRaisesRegex(ValueError, "Unexpected or missing repository files"):
            self.verify()

    def test_missing_authored_file_is_rejected(self):
        (self.root / "report-latex/main.tex").unlink()
        with self.assertRaisesRegex(ValueError, "Unexpected or missing repository files"):
            self.verify()

    def test_authored_allowance_cannot_expand(self):
        self.manifest["authored_files_not_hashed"].append("arbitrary.txt")
        self.write("arbitrary.txt", b"extra\n")
        with self.assertRaisesRegex(ValueError, "Unexpected authored-file allowance"):
            self.verify()

    def test_forbidden_payload_remains_forbidden_in_copied_inventory(self):
        self.override()
        self.copied.append(self.copy("source/weights.PT", b"forbidden"))
        self.accounting()
        with self.assertRaisesRegex(ValueError, "Forbidden copied payload"):
            self.verify()

    def test_omitted_payload_cannot_be_added(self):
        self.override()
        self.write(self.excluded[1]["path"], b"omitted")
        with self.assertRaisesRegex(ValueError, "Excluded payload is present"):
            self.verify()

    def test_linked_member_is_rejected(self):
        target = self.root / "source/frozen.py"
        link = self.root / "source/link.py"
        try:
            link.symlink_to(target)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows file symlinks require a privilege absent on this host")
            raise
        with self.assertRaisesRegex(ValueError, "Linked path"):
            self.verify()

    def test_nonportable_copied_path_is_rejected(self):
        self.copied[0]["path"] = "../escape.md"
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
