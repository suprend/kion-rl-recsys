"""Synthetic parser, link-safety and flowable checks; never write a PDF.

Run: python -B -m unittest discover -s reports -p test_render_report.py -v
ReportLab-only flowable tests skip when the optional PDF dependency is absent.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import render_report as renderer


def png_bytes(width: int = 40, height: int = 30) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    image = b"\x89PNG\r\n\x1a\n"
    image += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    image += chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\xff\xff" * width) * height))
    return image + chunk(b"IEND", b"")


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-pdf-source-")
        self.root = Path(self.temporary.name)
        self.source = self.root / "report.md"

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, text: str) -> Path:
        self.source.write_text(text, encoding="utf-8", newline="\n")
        return self.source

    def test_balanced_link_target_and_escaped_table_pipe(self):
        ref, end = renderer.reference_at("[sample](file(one).txt)", 0)
        self.assertEqual(ref.destination, "file(one).txt")
        self.assertEqual(end, len("[sample](file(one).txt)"))
        self.assertEqual(renderer.split_cells(r"| a\|b | `x|y` | c |"), [r"a\|b", "`x|y`", "c"])

    def test_code_fences_do_not_create_links_or_headings(self):
        self.write("# Test\n\n```text\n[not executed](missing.txt)\n# not a heading\n```\n")
        checked = renderer.inspect_source(self.source)
        self.assertEqual(checked["local_link_count"], 0)
        self.assertEqual(checked["blocks"], {"h1": 1, "code": 1})

    def test_missing_caption_link_is_checked(self):
        (self.root / "plot.png").write_bytes(png_bytes())
        self.write("# Test\n\n![Alternative](plot.png)\n\n*Рисунок 1. [SVG](missing.svg).*\n")
        with self.assertRaisesRegex(ValueError, "Missing local link"):
            renderer.inspect_source(self.source)

    def test_caption_is_consumed_once_and_image_dimensions_are_kept(self):
        (self.root / "plot.png").write_bytes(png_bytes(120, 60))
        self.write("# Test\n\n![Fallback](plot.png)\n\n*Рисунок 1. Exact visible caption.*\n\nFollowing body.\n")
        blocks = renderer.parse_markdown(self.source.read_text(encoding="utf-8"))
        self.assertEqual([block.kind for block in blocks], ["h1", "figure", "p"])
        self.assertEqual(blocks[1].value[1], "*Рисунок 1. Exact visible caption.*")
        checked = renderer.inspect_source(self.source)
        self.assertEqual(checked["embedded_images"], [{"target": "plot.png", "width": 120, "height": 60}])

    def test_relocation_rebinds_local_refs_without_touching_web_urls(self):
        (self.root / "evidence (1).txt").write_text("synthetic", encoding="utf-8")
        self.write("# Test\n\n[local](<evidence (1).txt>) and [web](https://example.org/paper#part).\n")
        destination = self.root / "relocated"
        destination.mkdir()
        moved_source = destination / self.source.name
        shutil.copyfile(self.source, moved_source)
        shutil.copyfile(self.root / "evidence (1).txt", destination / "evidence (1).txt")
        self.assertEqual(renderer.inspect_source(moved_source)["local_link_count"], 1)
        markup = renderer.inline("[local](<evidence (1).txt>) and [web](https://example.org/paper#part)", moved_source)
        self.assertIn(destination.resolve().as_uri(), markup)
        self.assertIn("https://example.org/paper#part", markup)

    def test_local_anchor_validation_uses_duplicate_heading_suffix(self):
        self.write("# Test\n\n## Раздел\n\nFirst.\n\n## Раздел\n\n[second](#раздел-1)\n")
        self.assertEqual(renderer.inspect_source(self.source)["local_link_count"], 1)
        self.write(self.source.read_text(encoding="utf-8").replace("#раздел-1", "#absent"))
        with self.assertRaisesRegex(ValueError, "Missing Markdown anchor"):
            renderer.inspect_source(self.source)

    def test_links_cannot_execute_code_or_fetch_remote_images(self):
        for target in ("javascript:alert(1)", "data:text/plain,x", "//server/share"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                renderer.resolve_reference(target, self.source)
        self.write("# Test\n\n![remote](https://example.org/plot.png)\n")
        with self.assertRaisesRegex(ValueError, "Remote images"):
            renderer.inspect_source(self.source)

    def test_encoded_unc_file_links_are_rejected_before_filesystem_access(self):
        from unittest.mock import patch
        targets = [r"\\server\share\x", "%5C%5Cserver%5Cshare%5Cx",
                   "%2F%2Fserver/share/x", "file:%5c%5cserver/share/x"]
        with patch.object(Path, "resolve", side_effect=AssertionError("must reject before OS access")):
            for target in targets:
                with self.subTest(target=target), self.assertRaisesRegex(ValueError, "Network file links"):
                    renderer.resolve_reference(target, self.source)

    def test_markup_is_escaped_and_code_literals_are_not_links(self):
        markup = renderer.inline('<b>raw</b> **bold** *italic* `[link](missing)`', self.source)
        self.assertIn("&lt;b&gt;raw&lt;/b&gt;", markup)
        self.assertIn("<b>bold</b>", markup)
        self.assertIn("<i>italic</i>", markup)
        self.assertNotIn("href=", markup)

    def test_malformed_table_and_unclosed_fences_are_rejected(self):
        bad = ["# T\n\n| a | b |\n|---|---|\n| only one |\n",
               "# T\n\n```text\nunclosed\n", "# T\n\n| a |\n|x|\n| b |\n"]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                renderer.parse_markdown(text)

    def test_column_widths_fill_available_space_and_reserve_numeric_columns(self):
        rows = [["Scientific interpretation", "MAP", "NDCG"],
                ["An explanation with " * 20, ".123456", ".223456"]]
        widths = renderer.column_widths(rows, 500)
        self.assertAlmostEqual(sum(widths), 500)
        self.assertGreater(widths[0], widths[1])
        self.assertGreater(widths[1], 50)

    def test_check_runs_with_no_site_packages_and_writes_nothing(self):
        self.write("# Synthetic\n\nSafe local source check.\n")
        before = set(self.root.iterdir())
        process = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(renderer.__file__)),
                                  "--source", str(self.source), "--check"],
                                 capture_output=True, text=True, encoding="utf-8", timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertIn("source-checked", process.stdout)

    def test_existing_pdf_is_never_opened_or_overwritten(self):
        self.write("# Synthetic\n\nSafe.\n")
        output = self.root / "existing.pdf"
        sentinel = b"existing artifact must survive"
        output.write_bytes(sentinel)
        with self.assertRaisesRegex(ValueError, "Output exists"):
            renderer.build_pdf(self.source, output)
        self.assertEqual(output.read_bytes(), sentinel)

    def test_png_header_and_missing_image_fail_early(self):
        (self.root / "bad.png").write_bytes(b"not an image")
        self.write("# Test\n\n![bad](bad.png)\n")
        with self.assertRaisesRegex(ValueError, "PNG/JPEG"):
            renderer.inspect_source(self.source)
        self.write("# Test\n\n![bad](absent.png)\n")
        with self.assertRaisesRegex(ValueError, "Missing embedded image"):
            renderer.inspect_source(self.source)


@unittest.skipUnless(importlib.util.find_spec("reportlab"), "Optional ReportLab unavailable")
class FlowableTests(unittest.TestCase):
    setUp = RendererTests.setUp
    tearDown = RendererTests.tearDown
    write = RendererTests.write

    def test_table_header_code_inherits_white_without_changing_body_code(self):
        from reportlab.lib import colors
        from reportlab.platypus import Paragraph, Table
        regular, heavy = renderer.find_fonts(None, None)
        self.write("# Synthetic\n\n| Centered `G_comp` | **Centered `G_res`** |\n|---|---|\n"
                   "| `G_comp` | `G_res` |\n")
        story, _ = renderer.build_story(self.source, regular, heavy)
        table = next(flow for flow in story if isinstance(flow, Table))
        for column, token in enumerate(("G_comp", "G_res")):
            header = table._cellvalues[0][column][0]
            body = table._cellvalues[1][column][0]
            # Table.wrap caches processed word fragments. Reparse the exact
            # emitted markup/style to assert effective PDF text colors.
            header = Paragraph(header.text, header.style)
            body = Paragraph(body.text, body.style)
            header_fragments = [fragment for fragment in header.frags if getattr(fragment, "text", None) == token]
            body_fragments = [fragment for fragment in body.frags if getattr(fragment, "text", None) == token]
            self.assertEqual(len(header_fragments), 1)
            self.assertEqual(len(body_fragments), 1)
            self.assertEqual(header_fragments[0].textColor, colors.white)
            self.assertEqual(body_fragments[0].textColor, colors.HexColor("#40546A"))
            self.assertEqual(body.getPlainText(), token)

    def test_internal_bookmark_uses_logical_page_before_serialization(self):
        from reportlab.pdfgen.canvas import Canvas
        buffer = io.BytesIO()
        canvas = Canvas(buffer)
        canvas._pageNumber = 3
        self.assertEqual(canvas._doc.pageCounter, 1)
        destination = renderer.logical_bookmark(canvas, "third-page", fit="XYZ", top=700)
        self.assertEqual(destination.page.name, "Page3")
        self.assertEqual(canvas._doc.pageCounter, 1)
        self.assertEqual(buffer.getvalue(), b"")
        # Deliberately never save: this test does not generate a PDF.

    def test_synthetic_flowables_keep_caption_and_table_contract(self):
        regular, heavy = renderer.find_fonts(None, None)
        (self.root / "plot.png").write_bytes(png_bytes(800, 600))
        self.write("# Синтетическая проверка\n\n## Long heading\n\n" +
                   "| Очень длинное описание | MAP |\n|---|---|\n| " + "текст " * 30 + "| .123 |\n\n" +
                   "![Fallback](plot.png)\n\n*Рисунок 1. Exact visible caption.*\n")
        story, layout = renderer.build_story(self.source, regular, heavy)
        from reportlab.platypus import KeepTogether, Table
        tables = [flow for flow in story if isinstance(flow, Table)]
        figures = [flow for flow in story if isinstance(flow, KeepTogether)]
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].repeatRows, 1)
        self.assertEqual(tables[0].splitByRow, 1)
        self.assertEqual(tables[0].splitInRow, 0)
        self.assertAlmostEqual(sum(tables[0]._colWidths), layout["available_width"])
        self.assertEqual(len(figures), 1)
        caption = figures[0]._content[-1]
        self.assertEqual(caption.getPlainText(), "Рисунок 1. Exact visible caption.")
        for flow in story:
            if not isinstance(flow, KeepTogether):
                width, _ = flow.wrap(layout["available_width"], layout["frame_height"])
                self.assertLessEqual(width, layout["available_width"] + .01)
        self.assertFalse(any(path.suffix == ".pdf" for path in self.root.iterdir()))

    def test_pathological_table_row_is_rejected_before_output(self):
        regular, heavy = renderer.find_fonts(None, None)
        self.write("# Synthetic\n\n| A | B |\n|---|---|\n| " + "long narrative " * 3000 + "| value |\n")
        with self.assertRaisesRegex(ValueError, "Table row too tall"):
            renderer.build_story(self.source, regular, heavy)


if __name__ == "__main__":
    unittest.main()
