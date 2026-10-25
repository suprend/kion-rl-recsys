"""A changed explicit extra must never override a previously declared source pin."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

_spec = importlib.util.spec_from_file_location("snapshot_builder", Path(__file__).with_name("build_snapshot.py"))
builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(builder)


class SourcePinIntegrity(unittest.TestCase):
    def test_changed_extra_cannot_override_request_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            project, pas, report = (base / n for n in ("project", "pas", "report"))
            project.mkdir()
            pas.mkdir()
            source = project / "source.py"
            source.write_text("answer = 42\n", encoding="utf-8")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            (project / "request.json").write_text(json.dumps({
                "stage": "synthetic", "source_hashes": {source.as_posix(): expected}}), encoding="utf-8")
            source.write_text("answer = 43\n", encoding="utf-8")
            argv = ["build_snapshot.py", "--project-root", str(project), "--pas-root", str(pas),
                    "--report-root", str(report), "--request", "request.json", "--extra", "source.py",
                    "--commit", "synthetic"]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "Conflicting explicit source pin"):
                builder.main()
            self.assertFalse(report.exists(), "Reject mismatched pins before writing any package output")


if __name__ == "__main__":
    unittest.main()
