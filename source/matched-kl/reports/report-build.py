"""Build the short KION reader or supplement; numerical libraries are never used.

--check is standard-library only. PDF rendering is explicit, lazy, and refuses
pending text unless --allow-pending is supplied. Root owns actual PDF execution.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent
PARTS = {"reader": "reader.md", "technical": "technical.md", "reproduce": "reproduce.md"}
PENDING = re.compile(r"\bPENDING\b|\{\{[^}]+\}\}")
PAGEBREAK = "<!-- pagebreak -->"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sources(part: str) -> list[Path]:
    return [ROOT / PARTS[part]] if part != "all" else [ROOT / PARTS[key] for key in PARTS]


def inspect_source(path: Path, allow_pending: bool) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    require(b"\r" not in raw, f"LF required: {path.name}")
    require(text.startswith("# "), f"Missing document title: {path.name}")
    require(text.count("```") % 2 == 0, f"Unclosed code fence: {path.name}")
    pending = len(PENDING.findall(text))
    require(allow_pending or pending == 0, f"Unresolved pending markers in {path.name}: {pending}")
    for line in text.splitlines():
        if line.startswith("|"):
            require(line.endswith("|"), f"Unclosed table row in {path.name}: {line}")
    if path.name == "reader.md":
        require(text.count(PAGEBREAK) == 2, "Reader must declare exactly three intended pages")
    return {"path": path.as_posix(), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "pending_markers": pending,
            "intended_pages": text.count(PAGEBREAK) + 1}


def link_target(destination: str, source: Path) -> str:
    """Preserve web destinations and resolve local evidence beside the source.

    This computes a URI; it does not open the referenced file or URL. A local
    fragment is kept so the PDF can retain a Markdown section destination too.
    """
    parsed = urlsplit(destination)
    if parsed.scheme:
        require(parsed.scheme.lower() in ("https", "http", "mailto", "file"),
                "Unsupported link scheme: " + parsed.scheme)
        return destination
    require(not parsed.netloc, "Network-path links need an explicit http(s) scheme")
    local = (source.parent / unquote(parsed.path)).resolve() if parsed.path else source.resolve()
    uri = urlsplit(local.as_uri())
    return urlunsplit((uri.scheme, uri.netloc, uri.path, parsed.query, parsed.fragment))


def styled(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"`([^`]+)`", r'<font name="ReportMono">\1</font>', escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)


def inline(text: str, source: Path) -> str:
    """Small escaped Markdown subset with clickable PDF evidence links."""
    output, start = [], 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        output.append(styled(text[start:match.start()]))
        destination = html.escape(link_target(match[2], source), quote=True)
        output.append(f'<link href="{destination}"><u>{styled(match[1])}</u></link>')
        start = match.end()
    output.append(styled(text[start:]))
    return "".join(output)


def find_fonts(font: str | None, bold: str | None) -> tuple[Path, Path]:
    require((font is None) == (bold is None), "Supply both --font and --font-bold")
    pairs = [(Path(font), Path(bold))] if font and bold else [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
         Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
    ]
    for regular, heavy in pairs:
        if regular.is_file() and heavy.is_file():
            return regular, heavy
    raise ValueError("Cyrillic TTF fonts unavailable; supply --font and --font-bold")


def blocks(text: str):
    lines, index = text.splitlines(), 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue
        if line == PAGEBREAK:
            yield "break", None
            index += 1
        elif line.startswith("```"):
            code = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            yield "code", "\n".join(code)
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                index += 1
            require(len(rows) > 1 and all(len(row) == len(rows[0]) for row in rows), "Ragged/empty table")
            yield "table", rows
        elif re.match(r"^#{1,3} ", line):
            level = len(line) - len(line.lstrip("#"))
            yield "h" + str(level), line[level + 1:]
            index += 1
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not lines[index].startswith(("#", "|", "```", "<!--")):
                paragraph.append(lines[index])
                index += 1
            yield "p", " ".join(paragraph)


def build_pdf(paths: list[Path], output: Path, args) -> dict:
    # These are document layout dependencies, deliberately absent from --check.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                   Spacer, Table, TableStyle)

    require(not output.exists(), "Output exists; choose a fresh PDF path")
    require(output.suffix.lower() == ".pdf" and output.parent.is_dir(), "Existing parent and .pdf output required")
    regular, bold = find_fonts(args.font, args.font_bold)
    pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(bold)))
    pdfmetrics.registerFont(TTFont("ReportMono", str(regular)))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold",
                                italic="ReportSans", boldItalic="ReportSansBold")
    width, height = A4
    margin = 39
    usable = width - 2 * margin
    styles = {
        "h1": ParagraphStyle("Title", fontName="ReportSansBold", fontSize=17.5, leading=21,
                             textColor=colors.HexColor("#172C40"), spaceAfter=10),
        "h2": ParagraphStyle("Heading", fontName="ReportSansBold", fontSize=11.4, leading=14,
                             textColor=colors.HexColor("#172C40"), spaceBefore=8, spaceAfter=5,
                             keepWithNext=True),
        "h3": ParagraphStyle("Subheading", fontName="ReportSansBold", fontSize=10, leading=12,
                             spaceBefore=6, spaceAfter=4, keepWithNext=True),
        "p": ParagraphStyle("Body", fontName="ReportSans", fontSize=9.2, leading=12.4,
                            spaceAfter=5.5, alignment=TA_LEFT, splitLongWords=True),
        "cell": ParagraphStyle("Cell", fontName="ReportSans", fontSize=8.1, leading=10.5,
                               splitLongWords=True),
        "cell_head": ParagraphStyle("CellHead", fontName="ReportSansBold", fontSize=8.0,
                                    leading=10.4, textColor=colors.white, splitLongWords=True),
        "code": ParagraphStyle("Code", fontName="ReportMono", fontSize=7, leading=9,
                               spaceBefore=3, spaceAfter=6),
    }
    story = []
    for file_index, path in enumerate(paths):
        if file_index:
            story.append(PageBreak())
        for kind, value in blocks(path.read_text(encoding="utf-8")):
            if kind == "break":
                story.append(PageBreak())
            elif kind == "code":
                # Preformatted code is intentionally limited to short portable lines.
                story.append(Preformatted(value, styles["code"], maxLineLength=105))
            elif kind == "table":
                ncol = len(value[0])
                lengths = [max(min(len(row[j]), 80) for row in value) for j in range(ncol)]
                if ncol == 2:
                    weights = [0.25, 0.75]
                elif ncol == 3:
                    weights = [0.35, 0.23, 0.42]
                else:
                    weights = [max(9, length) ** 0.6 for length in lengths]
                    total = sum(weights)
                    weights = [x / total for x in weights]
                data = [[Paragraph(inline(cell, path), styles["cell_head" if i == 0 else "cell"])
                         for cell in row] for i, row in enumerate(value)]
                table = Table(data, colWidths=[usable * x for x in weights], repeatRows=1,
                              hAlign="LEFT", splitByRow=True)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24445F")),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F1F5F8"), colors.white]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#24445F")),
                ]))
                story.extend([table, Spacer(1, 5)])
            else:
                story.append(Paragraph(inline(value, path), styles[kind]))
    page_count = 0

    def decorate(canvas, doc):
        nonlocal page_count
        page_count = max(page_count, doc.page)
        canvas.saveState()
        canvas.setFont("ReportSans", 7)
        canvas.setFillColor(colors.HexColor("#617384"))
        canvas.drawString(margin, 21, "KION | train-KL / sampler-seed follow-up | 25.10.2026")
        canvas.drawRightString(width - margin, 21, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=margin,
                                 leftMargin=margin, topMargin=34, bottomMargin=36,
                                 title="KION: completion and residual train-KL follow-up",
                                 author="KION research", pageCompression=1)
    document.build(story, onFirstPage=decorate, onLaterPages=decorate)
    if args.part == "reader":
        require(2 <= page_count <= 3, f"Reader layout has {page_count} pages; revise before delivery")
    return {"output": output.as_posix(), "pages": page_count, "bytes": output.stat().st_size,
            "visual_QA": "required separately; PDF build does not prove legibility"}


def run(request: dict, check=None) -> dict:
    """Root-controlled worker entrypoint, with no model/data work.

    Fields: part=reader|technical|reproduce|all; output_pdf=fresh local PDF;
    optional font/font_bold; allow_pending=False unless explicitly drafting.
    The caller owns artifact-operation marker and separate render/visual QA.
    """
    from types import SimpleNamespace

    require(sys.dont_write_bytecode, "Run worker with python -B")
    part = request.get("part", "reader")
    require(part in (*PARTS, "all"), "Unknown report part")
    output = Path(request["output_pdf"]).resolve()
    require(output.parent == ROOT.resolve(), "PDF output must be directly under reports")
    allow_pending = request.get("allow_pending", False)
    require(type(allow_pending) is bool, "allow_pending must be explicit boolean")
    args = SimpleNamespace(part=part, font=request.get("font"),
                           font_bold=request.get("font_bold"))
    inspected = [inspect_source(path, allow_pending) for path in sources(part)]
    if check:
        check(stage="report-source-checked", part=part)
    result = build_pdf(sources(part), output, args)
    if check:
        check(stage="report-pdf-built", pages=result["pages"])
    return {"state": "built-awaiting-visual-QA", "source_checks": inspected, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=[*PARTS, "all"], default="reader")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font")
    parser.add_argument("--font-bold")
    args = parser.parse_args()
    require(args.check or args.output is not None, "Choose --check or explicit --output")
    require(sys.dont_write_bytecode, "Run with python -B")
    paths = sources(args.part)
    for path in paths:
        print(inspect_source(path, args.allow_pending))
    if args.output is not None:
        print(build_pdf(paths, args.output.resolve(), args))


if __name__ == "__main__":
    main()
