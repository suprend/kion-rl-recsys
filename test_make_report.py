"""Standard-library synthetic checks; never import ReportLab or author a PDF.

Run: python -I -S -B -m unittest discover -s . -p test_make_report.py -v
"""
from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

import make_report


class ReaderReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="reader-report-check-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "REPORT.md"
        self.source.write_text("# Synthetic\n\nSafe source.\n", encoding="utf-8", newline="\n")
        self.builder = make_report.reader_links(make_report.load_builder(), self.root)

    def test_local_uri_quotes_unicode_spaces_hash_and_preserves_resolved_path(self):
        relative = "материалы/файл #1.txt"
        destination = quote(relative, safe="/") + "?raw=1#" + quote("раздел 1", safe="")
        uri, local, fragment = self.builder.resolve_reference(destination, self.source)
        self.assertEqual(uri, make_report.LINK_BASE + quote(relative, safe="/") +
                         "?raw=1#" + quote("раздел 1", safe=""))
        self.assertEqual(local, (self.root / relative).resolve())
        self.assertEqual(fragment, "раздел 1")

    def test_external_urls_are_retained_exactly(self):
        for destination in ("https://example.org/a%20b?q=1&x=2#part", "http://example.org/paper",
                            "mailto:reader@example.org?subject=Report"):
            with self.subTest(destination=destination):
                uri, local, _ = self.builder.resolve_reference(destination, self.source)
                self.assertEqual(uri, destination)
                self.assertIsNone(local)

    def test_source_heading_links_remain_internal_named_destinations(self):
        self.source.write_text("# Synthetic\n\n## Раздел\n\n## Раздел\n\n"
                               "[second](#%D1%80%D0%B0%D0%B7%D0%B4%D0%B5%D0%BB-1)\n",
                               encoding="utf-8", newline="\n")
        self.assertEqual(self.builder.inspect_source(self.source)["local_link_count"], 1)
        for destination in ("#раздел-1", "REPORT.md#раздел-1", self.source.as_uri() + "#раздел-1"):
            with self.subTest(destination=destination):
                markup = self.builder.inline("[second](" + destination + ")", self.source)
                self.assertIn('href="#раздел-1"', markup)
                self.assertNotIn(make_report.LINK_BASE, markup)

    def test_other_markdown_anchor_uses_repository_url_and_existing_validation(self):
        target = self.root / "Другой файл.md"
        target.write_text("# Проверенный раздел\n", encoding="utf-8", newline="\n")
        destination = quote(target.name) + "#" + quote("проверенный-раздел")
        self.source.write_text("# Synthetic\n\n[other](" + destination + ")\n",
                               encoding="utf-8", newline="\n")
        self.assertEqual(self.builder.inspect_source(self.source)["local_link_count"], 1)
        self.assertIn(make_report.LINK_BASE + destination,
                      self.builder.inline("[other](" + destination + ")", self.source))
        self.source.write_text("# Synthetic\n\n[absent](" + quote(target.name) + "#absent)\n",
                               encoding="utf-8", newline="\n")
        with self.assertRaisesRegex(ValueError, "Missing Markdown anchor"):
            self.builder.inspect_source(self.source)

    def test_embedded_image_is_inspected_at_the_original_local_path(self):
        image = self.root / "график 1.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 80, 40))
        self.source.write_text("# Synthetic\n\n![plot](" + quote(image.name) + ")\n",
                               encoding="utf-8", newline="\n")
        with patch.object(self.builder, "image_dimensions", wraps=self.builder.image_dimensions) as dimensions:
            checked = self.builder.inspect_source(self.source)
        dimensions.assert_called_once_with(image.resolve())
        self.assertEqual(checked["embedded_images"], [{"target": quote(image.name), "width": 80, "height": 40}])

    def test_escaping_local_target_is_rejected(self):
        targets = ("../outside.md", "%2e%2e/outside.md", (self.root.parent / "outside.md").as_uri())
        for destination in targets:
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "escapes"):
                self.builder.resolve_reference(destination, self.source)

    def test_unexpected_frozen_hash_is_rejected_before_execution(self):
        fake = Mock()
        fake.read_bytes.return_value = b"raise AssertionError('unverified code executed')\n"
        with patch.object(make_report, "FROZEN", fake), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            make_report.load_builder()

    def invoke(self, arguments):
        output = io.StringIO()
        with patch.object(make_report, "ROOT", self.root), \
                patch.object(make_report, "load_builder", return_value=self.builder), redirect_stdout(output):
            result = make_report.main(["--source", str(self.source), *arguments])
        return result, json.loads(output.getvalue())

    def test_check_reports_link_base_without_writing_or_importing_optional_packages(self):
        before = set(self.root.iterdir())
        with patch.object(self.builder, "build_pdf", side_effect=AssertionError("check must not render")):
            result, checked = self.invoke(["--check"])
        self.assertEqual(result, 0)
        self.assertEqual(checked["state"], "source-checked")
        self.assertEqual(checked["local_link_base"], make_report.LINK_BASE)
        self.assertEqual(checked["renderer_sha256"], make_report.FROZEN_SHA256)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertFalse(any(name == "reportlab" or name.startswith("reportlab.") for name in sys.modules))

    def test_main_delegates_output_and_font_arguments_to_frozen_builder(self):
        external_output = self.root.parent / "fresh-reader-report.pdf"
        with patch.object(self.builder, "build_pdf", return_value={"state": "synthetic-call-only"}) as build:
            self.invoke(["--output", str(external_output), "--font", "regular.ttf", "--font-bold", "bold.ttf"])
        build.assert_called_once_with(self.source, external_output.resolve(), "regular.ttf", "bold.ttf")

    def test_frozen_existing_output_guard_runs_before_fonts_or_any_write(self):
        output = Mock(suffix=".pdf")
        output.parent.is_dir.return_value = True
        output.exists.return_value = True
        with patch.object(self.builder, "find_fonts", side_effect=AssertionError("must fail before fonts")), \
                self.assertRaisesRegex(ValueError, "Output exists"):
            self.builder.build_pdf(self.source, output)
        output.open.assert_not_called()
        # Confirm the delegated writer still uses exclusive creation, without
        # entering its ReportLab path or creating even a synthetic PDF file.
        tree = ast.parse(make_report.FROZEN.read_text(encoding="utf-8"))
        build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_pdf")
        modes = [node.args[0].value for node in ast.walk(build) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "open" and node.args
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == "output"]
        self.assertEqual(modes, ["xb"])

    def test_source_outside_repository_is_rejected_before_inspection(self):
        error = io.StringIO()
        with patch.object(make_report, "ROOT", self.root / "subdirectory"), \
                patch.object(make_report, "load_builder", return_value=self.builder), \
                patch.object(self.builder, "inspect_source", side_effect=AssertionError("must reject source first")), \
                redirect_stderr(error), self.assertRaises(SystemExit) as raised:
            make_report.main(["--source", str(self.source), "--check"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("Source must be inside", error.getvalue())


if __name__ == "__main__":
    unittest.main()
