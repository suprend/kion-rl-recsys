"""Small temporary fixtures; never load models or run training."""
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

import package_submission as package


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)
        self.repo = self.base / "repo"
        self.repo.mkdir()
        (self.repo / "README.md").write_bytes(b"Current uncommitted text.\n")
        projection = {"copied_files": [], "authored_files_not_hashed": ["README.md", "projection.json"]}
        (self.repo / "projection.json").write_text(json.dumps(projection), encoding="utf-8")
        self.verifier = patch.object(package.verify_repository, "verify", return_value={"state": "passed"})
        self.verify_mock = self.verifier.start()
        self.addCleanup(self.verifier.stop)
        self.archive = self.base / "source.zip"

    def create(self):
        return package.create_archive(self.repo, self.archive)

    def rewrite(self, transform):
        with zipfile.ZipFile(self.archive) as source:
            rows = [(info, source.read(info)) for info in source.infolist()]
        changed = self.base / "changed.zip"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(changed, "w") as target:
                for info, data in transform(rows):
                    target.writestr(info, data)
        return changed

    def test_roundtrip_current_files_reproducible_and_no_local_metadata(self):
        (self.repo / ".git").mkdir()
        (self.repo / ".git" / "config").write_text("private metadata", encoding="utf-8")
        result = self.create()
        self.assertEqual(result["files"], 2)
        self.assertEqual(self.verify_mock.call_count, 2)
        with zipfile.ZipFile(self.archive) as bundle:
            self.assertEqual(bundle.read(package.PREFIX + "README.md"), b"Current uncommitted text.\n")
            self.assertEqual(len(bundle.namelist()), 3)
        other = self.base / "other.zip"
        package.create_archive(self.repo, other)
        self.assertEqual(self.archive.read_bytes(), other.read_bytes())
        self.assertEqual(package.verify_archive(other)["archive_sha256"], result["archive_sha256"])
        self.assertFalse(list(self.base.glob(".submission-*")))

    def test_tampered_content_rejected(self):
        self.create()
        archive = self.rewrite(lambda rows: [(info, b"changed" if info.filename.endswith("README.md") else data)
                                             for info, data in rows])
        with self.assertRaisesRegex(ValueError, "mismatch"):
            package.verify_archive(archive)

    def test_extracted_repository_preserves_finite_closure(self):
        self.create()
        destination = self.base / "extracted"
        with zipfile.ZipFile(self.archive) as bundle:
            bundle.extractall(destination)  # Only our already-verified synthetic archive.
        root = destination / package.PREFIX.rstrip("/")
        projection = json.loads((root / "projection.json").read_text(encoding="utf-8"))
        actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
        self.assertEqual(actual, set(package.closure(projection)))
        self.assertTrue((destination / package.MANIFEST).is_file())
        self.assertFalse((root / package.MANIFEST).exists())

    def test_unsafe_member_rejected(self):
        self.create()
        archive = self.rewrite(lambda rows: rows + [(package.PREFIX + "../escape", b"bad")])
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            package.verify_archive(archive)

    def test_duplicate_member_rejected(self):
        self.create()
        archive = self.rewrite(lambda rows: rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            package.verify_archive(archive)

    def test_symlink_member_rejected(self):
        self.create()
        def transform(rows):
            info, data = rows[0]
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            return [(info, data)] + rows[1:]
        archive = self.rewrite(transform)
        with self.assertRaisesRegex(ValueError, "Nonregular"):
            package.verify_archive(archive)

    def test_extra_unlisted_member_rejected(self):
        self.create()
        archive = self.rewrite(lambda rows: rows + [(package.PREFIX + "extra.txt", b"bad")])
        with self.assertRaisesRegex(ValueError, "closure"):
            package.verify_archive(archive)

    def test_refuse_overwrite(self):
        self.archive.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.create()
        self.assertEqual(self.archive.read_bytes(), b"keep")
        self.verify_mock.assert_not_called()

    def test_repository_verification_failure_creates_nothing(self):
        self.verify_mock.side_effect = ValueError("unlisted source")
        with self.assertRaisesRegex(ValueError, "unlisted source"):
            self.create()
        self.assertFalse(self.archive.exists())
        self.assertFalse(list(self.base.glob(".submission-*")))

    def test_atomic_publication_failure_cleans_temporary_only(self):
        def concurrent_output(source, target):
            Path(target).write_bytes(b"other process")
            raise FileExistsError("concurrent publisher")
        with patch.object(package.os, "link", side_effect=concurrent_output):
            with self.assertRaises(FileExistsError):
                self.create()
        self.assertEqual(self.archive.read_bytes(), b"other process")
        self.assertFalse(list(self.base.glob(".submission-*")))

    def test_limits_enforced_before_decompression(self):
        self.create()
        with patch.object(package, "MAX_FILE", 4):
            with self.assertRaisesRegex(ValueError, "Oversized"):
                package.verify_archive(self.archive)


if __name__ == "__main__":
    unittest.main()
