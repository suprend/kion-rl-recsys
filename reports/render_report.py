"""Render a portable research report from a deliberately small Markdown subset.

The --check path uses only the Python standard library. Rendering imports
ReportLab lazily, never fetches a URL, never executes links or code blocks, and
refuses to overwrite an existing output. The caller owns the PDF-operation
marker and the separate all-page visual acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import json
import os
from pathlib import Path, PureWindowsPath
import re
import struct
import sys
from urllib.parse import unquote, urlsplit, urlunsplit


PAGEBREAK = "<!-- pagebreak -->"
PENDING = re.compile(r"\b(?:PENDING|TODO|TBD)\b|\{\{[^}]+\}\}")
CAPTION = re.compile(r"^\*?(?:Рисунок|Figure)\s*\d+[.:]", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LIST_ITEM = re.compile(r"^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Reference:
    label: str
    destination: str
    image: bool = False


@dataclass(frozen=True)
class Block:
    kind: str
    value: object
    line: int


def reference_at(text: str, start: int) -> tuple[Reference, int] | None:
    """Read a Markdown link, including escaped or balanced target parentheses."""
    image = text.startswith("![", start)
    left = start + 1 if image else start
    if left >= len(text) or text[left] != "[":
        return None
    cursor, depth = left + 1, 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "[":
            depth += 1
        if text[cursor] == "]":
            depth -= 1
            if depth == 0:
                break
        cursor += 1
    if cursor + 1 >= len(text) or text[cursor:cursor + 2] != "](":
        return None
    label = text[left + 1:cursor]
    target_start = cursor + 2
    cursor, depth = target_start, 1
    while cursor < len(text):
        char = text[cursor]
        if char == "\\":
            cursor += 2
            continue
        if char == "(":
            depth += 1
        if char == ")":
            depth -= 1
            if depth == 0:
                target = text[target_start:cursor].strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                else:
                    require(not re.search(r'\s+[\"\']', target),
                            "Markdown link titles are unsupported; put the title in the label")
                target = re.sub(r"\\([()\\])", r"\1", target)
                require(bool(target), "Empty Markdown link target")
                return Reference(label, target, image), cursor + 1
        cursor += 1
    raise ValueError("Unclosed Markdown link destination")


def references(text: str):
    """Yield visible references while ignoring inline code spans."""
    index = 0
    while index < len(text):
        if text[index] == "`":
            end = text.find("`", index + 1)
            require(end >= 0, "Unclosed inline code span")
            index = end + 1
            continue
        ref = reference_at(text, index)
        if ref:
            value, end = ref
            yield value
            index = end
        else:
            index += 2 if text[index] == "\\" else 1


def split_cells(line: str) -> list[str]:
    """Split table cells without treating code/escaped pipes as separators."""
    line = line.strip()
    require(line.startswith("|") and line.endswith("|"), "Table rows need outer pipes")
    cells, cell, code, escaped = [], [], False, False
    for char in line[1:-1]:
        if escaped:
            cell.extend(("\\", char))
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "`":
            code = not code
            cell.append(char)
        elif char == "|" and not code:
            cells.append("".join(cell).strip())
            cell = []
        else:
            cell.append(char)
    require(not escaped and not code, "Unclosed escape or inline code in table")
    cells.append("".join(cell).strip())
    return cells


def parse_markdown(text: str) -> list[Block]:
    lines, index, result = text.splitlines(), 0, []

    def special(line: str) -> bool:
        return bool(HEADING.match(line) or LIST_ITEM.match(line) or
                    line.startswith(("|", "```", "![", ">", "<!--")))

    while index < len(lines):
        line, number = lines[index], index + 1
        if not line.strip():
            index += 1
            continue
        if line.strip() == PAGEBREAK:
            result.append(Block("break", None, number))
            index += 1
        elif line.startswith("<!--"):
            require("-->" in line, f"Only single-line HTML comments supported, line {number}")
            require(line.strip().endswith("-->"), f"Text after HTML comment at line {number}")
            index += 1
        elif line.startswith("```"):
            require(not line.startswith("````"), f"Only triple-backtick fences supported, line {number}")
            code = []
            index += 1
            while index < len(lines) and lines[index].strip() != "```":
                code.append(lines[index])
                index += 1
            require(index < len(lines), f"Unclosed code fence at line {number}")
            result.append(Block("code", "\n".join(code), number))
            index += 1
        elif line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].startswith("|"):
                rows.append(split_cells(lines[index]))
                index += 1
            require(len(rows) >= 3, f"Table needs a header, separator and data at line {number}")
            require(all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]),
                    f"Invalid table separator at line {number + 1}")
            require(all(len(row) == len(rows[0]) for row in rows), f"Ragged table at line {number}")
            result.append(Block("table", [rows[0], *rows[2:]], number))
        elif HEADING.match(line):
            heading = HEADING.match(line)
            result.append(Block("h" + str(len(heading[1])), heading[2], number))
            index += 1
        elif line.startswith("!["):
            parsed = reference_at(line.strip(), 0)
            require(parsed is not None and parsed[0].image and parsed[1] == len(line.strip()),
                    f"Image must occupy its own line, line {number}")
            image_ref = parsed[0]
            index += 1
            next_index = index
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            caption = None
            if next_index < len(lines) and CAPTION.match(lines[next_index]):
                caption_lines = [lines[next_index]]
                next_index += 1
                while next_index < len(lines) and lines[next_index].strip() and not special(lines[next_index]):
                    caption_lines.append(lines[next_index])
                    next_index += 1
                caption = " ".join(caption_lines)
                index = next_index
            result.append(Block("figure", (image_ref, caption), number))
        elif LIST_ITEM.match(line):
            item = LIST_ITEM.match(line)
            label = (item[2] + ".") if item[2] else "•"
            content = [item[3]]
            index += 1
            while index < len(lines) and lines[index].strip() and not special(lines[index]):
                content.append(lines[index].strip())
                index += 1
            result.append(Block("list", (label, " ".join(content)), number))
        elif line.startswith(">"):
            quote = []
            while index < len(lines) and lines[index].startswith(">"):
                quote.append(lines[index].lstrip(">").strip())
                index += 1
            result.append(Block("quote", " ".join(quote), number))
        else:
            paragraph = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not special(lines[index]):
                paragraph.append(lines[index])
                index += 1
            result.append(Block("p", " ".join(paragraph), number))
    return result


def block_texts(block: Block):
    if block.kind == "table":
        yield from (cell for row in block.value for cell in row)
    elif block.kind == "figure":
        ref, caption = block.value
        yield ref.label
        if caption:
            yield caption
    elif block.kind == "list":
        yield block.value[1]
    elif isinstance(block.value, str) and block.kind != "code":
        yield block.value


def slug(text: str) -> str:
    text = re.sub(r"[`*_]", "", text).strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def heading_ids(blocks: list[Block]) -> dict[int, str]:
    counts, result = Counter(), {}
    for block in blocks:
        if block.kind.startswith("h"):
            base = slug(str(block.value))
            count = counts[base]
            result[block.line] = base + ("-" + str(count) if count else "")
            counts[base] += 1
    return result


def resolve_reference(destination: str, source: Path) -> tuple[str, Path | None, str]:
    """Return URI, local path, fragment; computing this never opens a link."""
    require(not any(ord(char) < 32 for char in destination), "Control character in link target")
    # urlsplit interprets a Windows drive as a scheme; handle it explicitly.
    if PureWindowsPath(destination).is_absolute() and not destination.startswith(("/", "\\")):
        require(os.name == "nt", "Windows absolute link cannot be resolved on this OS: " + destination)
        local = Path(destination).resolve()
        return local.as_uri(), local, ""
    parsed = urlsplit(destination)
    scheme = parsed.scheme.lower()
    if scheme in ("http", "https", "mailto"):
        require(scheme == "mailto" or bool(parsed.netloc), "Web link needs a host: " + destination)
        return destination, None, parsed.fragment
    require(scheme in ("", "file"), "Unsupported link scheme: " + scheme)
    require(not parsed.netloc or parsed.netloc.lower() == "localhost", "Network file links are unsupported")
    decoded_path = unquote(parsed.path)
    require(not decoded_path.replace("\\", "/").startswith("//"), "Network file links are unsupported")
    if scheme == "file":
        local_text = decoded_path
        if os.name == "nt" and re.match(r"^/[A-Za-z]:/", local_text):
            local_text = local_text[1:]
        local = Path(local_text).resolve()
    else:
        local = (source.parent / decoded_path).resolve() if parsed.path else source.resolve()
    uri = urlsplit(local.as_uri())
    return urlunsplit((uri.scheme, uri.netloc, uri.path, parsed.query, parsed.fragment)), local, unquote(parsed.fragment)


def image_dimensions(path: Path) -> tuple[int, int]:
    """Read dimensions of embedded PNG/JPEG assets using only the standard library."""
    with path.open("rb") as stream:
        header = stream.read(24)
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            require(header[12:16] == b"IHDR" and len(header) == 24, "Invalid PNG header: " + path.name)
            width, height = struct.unpack(">II", header[16:24])
        elif header[:2] == b"\xff\xd8":
            stream.seek(2)
            width = height = 0
            while True:
                marker = stream.read(1)
                require(bool(marker), "JPEG dimensions not found: " + path.name)
                if marker != b"\xff":
                    continue
                marker = stream.read(1)
                while marker == b"\xff":
                    marker = stream.read(1)
                require(bool(marker), "Truncated JPEG: " + path.name)
                if marker in (b"\x00", b"\xd8") or b"\xd0" <= marker <= b"\xd7":
                    continue
                require(marker not in (b"\xd9", b"\xda"), "JPEG dimensions not found: " + path.name)
                length_raw = stream.read(2)
                require(len(length_raw) == 2, "Truncated JPEG segment: " + path.name)
                length = struct.unpack(">H", length_raw)[0]
                require(length >= 2, "Invalid JPEG segment: " + path.name)
                if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    frame = stream.read(5)
                    require(len(frame) == 5, "Truncated JPEG frame: " + path.name)
                    height, width = struct.unpack(">HH", frame[1:5])
                    break
                stream.seek(length - 2, 1)
        else:
            raise ValueError("Embedded figures must be PNG/JPEG: " + path.name)
    require(width > 0 and height > 0, "Invalid image dimensions: " + path.name)
    return width, height


def inspect_source(source: Path) -> dict:
    require(source.is_file(), "Source Markdown not found: " + source.as_posix())
    raw = source.read_bytes()
    text = raw.decode("utf-8-sig")
    require(text.startswith("# "), "Report must start with one level-one title")
    require(not PENDING.search(text), "Unresolved TODO/TBD/PENDING/template marker")
    blocks = parse_markdown(text)
    require(sum(block.kind == "h1" for block in blocks) == 1, "Report requires exactly one level-one title")
    anchors = set(heading_ids(blocks).values())
    image_entries, local_refs, web_refs = [], [], []
    for block in blocks:
        refs = [ref for part in block_texts(block) for ref in references(part)]
        if block.kind == "figure":
            refs.insert(0, block.value[0])
        for ref in refs:
            uri, local, fragment = resolve_reference(ref.destination, source)
            if ref.image:
                require(local is not None, "Remote images are not fetched: " + ref.destination)
                require(local.is_file(), "Missing embedded image: " + ref.destination)
                require(not fragment, "Image fragments unsupported: " + ref.destination)
                width, height = image_dimensions(local)
                image_entries.append({"target": ref.destination, "width": width, "height": height})
            elif local is not None:
                require(local.is_file() or local.is_dir(), "Missing local link target: " + ref.destination)
                if fragment and local.suffix.lower() == ".md":
                    target_anchors = anchors if local == source.resolve() else set(heading_ids(
                        parse_markdown(local.read_text(encoding="utf-8-sig"))).values())
                    require(fragment in target_anchors, "Missing Markdown anchor: " + ref.destination)
                local_refs.append(ref.destination)
            else:
                web_refs.append(uri)
    return {"source": source.as_posix(), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "blocks": dict(Counter(block.kind for block in blocks)),
            "embedded_images": image_entries, "local_link_count": len(local_refs),
            "web_link_count": len(web_refs), "state": "source-checked",
            "scope": "Local references and PNG/JPEG headers; no links opened, numerical libraries or visual acceptance"}


def inline(text: str, source: Path, *, links: bool = True, code_color: str | None = "#40546A") -> str:
    """Escape raw HTML and translate a compact inline Markdown subset."""
    result, index = [], 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            result.append(html.escape(text[index + 1]))
            index += 2
            continue
        if text[index] == "`":
            end = text.find("`", index + 1)
            require(end >= 0, "Unclosed inline code span")
            color = (' color="' + html.escape(code_color, quote=True) + '"') if code_color else ""
            result.append('<font name="ReportSans"' + color + '>' + html.escape(text[index + 1:end]) + "</font>")
            index = end + 1
            continue
        ref = reference_at(text, index)
        if ref:
            value, end = ref
            require(not value.image, "Inline images unsupported; put an image on its own line")
            require(links, "Nested Markdown links unsupported")
            target, local, fragment = resolve_reference(value.destination, source)
            if local == source.resolve() and fragment:
                target = "#" + fragment
            result.append('<link href="' + html.escape(target, quote=True) + '" color="#245E8B"><u>' +
                          inline(value.label, source, links=False, code_color=code_color) + "</u></link>")
            index = end
            continue
        marker = "**" if text.startswith("**", index) else ("*" if text[index] == "*" else None)
        if marker:
            end = text.find(marker, index + len(marker))
            if end >= 0:
                tag = "b" if marker == "**" else "i"
                result.append("<" + tag + ">" + inline(text[index + len(marker):end], source,
                              links=links, code_color=code_color) + "</" + tag + ">")
                index = end + len(marker)
                continue
        result.append(html.escape(text[index]))
        index += 1
    return "".join(result)


def plain_text(text: str) -> str:
    text = re.sub(r"!?\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"[`*_]", "", text)


def column_widths(rows: list[list[str]], available: float) -> list[float]:
    """Reserve narrow metric columns while allowing narrative columns to wrap."""
    columns = list(zip(*rows))
    measures = []
    for column in columns:
        lengths = sorted(len(plain_text(cell)) for cell in column)
        upper = lengths[min(len(lengths) - 1, int(len(lengths) * .85))]
        measures.append(max(7, min(72, upper)) ** .70)
    floor = min(52.0, available / (len(columns) * 1.6))
    residual = available - floor * len(columns)
    return [floor + residual * value / sum(measures) for value in measures]


def find_fonts(font: str | None, bold: str | None) -> tuple[Path, Path]:
    require((font is None) == (bold is None), "Supply both --font and --font-bold")
    pairs = [(Path(font), Path(bold))] if font and bold else [
        (Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/arialbd.ttf")),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")),
        (Path("/Library/Fonts/Arial.ttf"), Path("/Library/Fonts/Arial Bold.ttf")),
        (Path("/System/Library/Fonts/Supplemental/Arial.ttf"), Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")),
    ]
    for regular, heavy in pairs:
        if regular.is_file() and heavy.is_file():
            return regular, heavy
    raise ValueError("Cyrillic TTF fonts not found; supply --font and --font-bold")


def build_story(source: Path, regular: Path, bold: Path):
    """Prepare flowables without writing a PDF (also usable for synthetic QA)."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle

    pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(bold)))
    pdfmetrics.registerFontFamily("ReportSans", normal="ReportSans", bold="ReportSansBold",
                                italic="ReportSans", boldItalic="ReportSansBold")
    for name in ("ReportSans", "ReportSansBold"):
        require(all(code in pdfmetrics.getFont(name).face.charWidths for code in (0x410, 0x44F)),
                "Selected font lacks Cyrillic glyphs: " + name)
    width, height = A4
    margin, available, frame_height = 43, width - 86, height - 88
    body = dict(fontName="ReportSans", fontSize=9.2, leading=13.1, textColor=colors.HexColor("#263747"),
                splitLongWords=True, allowWidows=0, allowOrphans=0)
    styles = {
        "p": ParagraphStyle("Body", **body, spaceAfter=6.3),
        "quote": ParagraphStyle("Quote", **body, leftIndent=12, rightIndent=10, borderColor=colors.HexColor("#BCCCD7"),
                                borderWidth=.5, borderPadding=6, spaceBefore=5, spaceAfter=9),
        "caption": ParagraphStyle("Caption", **{**body, "fontSize": 8.3, "leading": 11.2,
                                  "textColor": colors.HexColor("#546778")}, spaceBefore=5, spaceAfter=9),
        "list": ParagraphStyle("List", **body, leftIndent=13, firstLineIndent=0, bulletIndent=0,
                               bulletFontName="ReportSans", bulletFontSize=9.2, spaceAfter=4.5),
        "code": ParagraphStyle("Code", **{**body, "fontSize": 8.0, "leading": 11.2},
                               backColor=colors.HexColor("#EEF3F7"), borderPadding=7,
                               spaceBefore=4, spaceAfter=8),
    }
    for level in range(1, 7):
        size = {1: 21, 2: 13.5, 3: 10.7}.get(level, 10)
        styles["h" + str(level)] = ParagraphStyle("H" + str(level), fontName="ReportSansBold",
            fontSize=size, leading=size * 1.27, textColor=colors.HexColor("#173C56"),
            spaceBefore=0 if level == 1 else (12 if level == 2 else 8),
            spaceAfter=11 if level == 1 else 6, keepWithNext=True,
            splitLongWords=True)
    blocks = parse_markdown(source.read_text(encoding="utf-8-sig"))
    visible = "\n".join(plain_text(part) for block in blocks for part in block_texts(block))
    visible += "\n" + "\n".join(str(block.value) for block in blocks if block.kind == "code")
    glyphs = pdfmetrics.getFont("ReportSans").face.charWidths
    missing = sorted({char for char in visible if ord(char) >= 32 and ord(char) not in glyphs})
    require(not missing, "Selected regular font lacks visible glyphs: " +
            ", ".join(f"U+{ord(char):04X} {char}" for char in missing))
    anchors = heading_ids(blocks)
    story, figure_number = [], 0
    for block in blocks:
        kind, value = block.kind, block.value
        if kind == "break":
            story.append(PageBreak())
        elif kind == "figure":
            figure_number += 1
            ref, caption = value
            _, image_path, _ = resolve_reference(ref.destination, source)
            image_width, image_height = image_dimensions(image_path)
            caption = caption or f"Рисунок {figure_number}. {ref.label}"
            caption_flow = Paragraph(inline(caption, source), styles["caption"])
            _, caption_height = caption_flow.wrap(available, frame_height)
            # A figure and its caption are one indivisible unit. Large panels
            # may use almost a page; never shrink the whole figure to a thumbnail.
            max_height = min(frame_height * .68, frame_height - caption_height - 22)
            require(max_height >= 60, f"Caption too long at line {block.line}")
            scale = min(available / image_width, max_height / image_height)
            figure = Image(str(image_path), width=image_width * scale, height=image_height * scale)
            figure.hAlign = "CENTER"
            story.append(KeepTogether([Spacer(1, 5), figure, caption_flow]))
        elif kind == "table":
            rows, size = value, (7.9 if len(value[0]) >= 5 else 8.6)
            cell_style = ParagraphStyle("Cell", **{**body, "fontSize": size, "leading": size * 1.34})
            head_style = ParagraphStyle("HeadCell", parent=cell_style, fontName="ReportSansBold", textColor=colors.white)
            data = [[Paragraph(inline(cell, source, code_color=None) if i == 0 else inline(cell, source),
                               head_style if i == 0 else cell_style)
                     for cell in row] for i, row in enumerate(rows)]
            table = Table(data, colWidths=column_widths(rows, available), repeatRows=1,
                          hAlign="LEFT", splitByRow=1, splitInRow=0, spaceBefore=4, spaceAfter=9)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#234B65")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F0F5F8"), colors.white]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#234B65")),
                ("LINEBELOW", (0, -1), (-1, -1), .4, colors.HexColor("#CCDAE4")),
            ]))
            # Splitting a huge row hides malformed input. Fail before opening
            # the output instead of cutting content or quietly shrinking type.
            table.wrap(available, frame_height)
            require(max(table._rowHeights) + table._rowHeights[0] < frame_height - 6,
                    f"Table row too tall to keep intact at line {block.line}")
            story.append(table)
        elif kind == "code":
            # Paragraph line wrapping handles long commands; every original
            # newline remains visible and no source is executed.
            code = "<br/>".join(html.escape(line).replace(" ", "&#160;") for line in value.split("\n"))
            story.append(Paragraph(code, styles["code"]))
        elif kind == "list":
            label, content = value
            story.append(Paragraph(inline(content, source), styles["list"], bulletText=label))
        else:
            content = inline(str(value), source)
            if kind.startswith("h"):
                content = '<a name="' + html.escape(anchors[block.line], quote=True) + '"/>' + content
            story.append(Paragraph(content, styles[kind]))
    return story, {"page_width": width, "page_height": height, "margin": margin,
                   "available_width": available, "frame_height": frame_height}


def logical_bookmark(canvas, key: str, *args, **kwargs):
    """Bind a heading to its logical page while actual page writes are deferred."""
    from reportlab.pdfgen.canvas import Canvas

    counter = canvas._doc.pageCounter
    canvas._doc.pageCounter = canvas._pageNumber
    try:
        return Canvas.bookmarkPage(canvas, key, *args, **kwargs)
    finally:
        canvas._doc.pageCounter = counter


def build_pdf(source: Path, output: Path, font: str | None = None, font_bold: str | None = None) -> dict:
    # All parsing, reference checks, fonts and flowable validation happen before
    # an exclusive-create handle is acquired, preserving existing artifacts.
    checked = inspect_source(source)
    require(output.suffix.lower() == ".pdf", "Output must have a .pdf suffix")
    require(output.parent.is_dir(), "Output parent directory must already exist")
    require(not output.exists(), "Output exists; choose a fresh PDF path")
    regular, heavy = find_fonts(font, font_bold)
    story, layout = build_story(source, regular, heavy)
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate
    from reportlab.lib import colors

    class CountedCanvas(Canvas):
        """Replay page states at save time to write a truthful n / N footer."""
        def __init__(self, *args, **kwargs):
            Canvas.__init__(self, *args, **kwargs)
            self.states = []
            self.total_pages = 0

        def showPage(self):
            self.states.append(dict(self.__dict__))
            self._startPage()

        def bookmarkPage(self, key, *args, **kwargs):
            return logical_bookmark(self, key, *args, **kwargs)

        def save(self):
            total = len(self.states)
            for state in self.states:
                self.__dict__.update(state)
                self.saveState()
                self.setStrokeColor(colors.HexColor("#D3DFE7"))
                self.setLineWidth(.4)
                self.line(layout["margin"], 33, layout["page_width"] - layout["margin"], 33)
                self.setFont("ReportSans", 7.4)
                self.setFillColor(colors.HexColor("#617586"))
                self.drawString(layout["margin"], 21, "KION  |  Исследовательский отчёт  |  25.10.2026")
                self.drawRightString(layout["page_width"] - layout["margin"], 21,
                                     f"{self._pageNumber} / {total}")
                self.restoreState()
                Canvas.showPage(self)
            self.total_pages = total
            Canvas.save(self)

    title = str(parse_markdown(source.read_text(encoding="utf-8-sig"))[0].value)
    canvas_holder = []

    def canvas_factory(*args, **kwargs):
        canvas = CountedCanvas(*args, **kwargs)
        canvas_holder.append(canvas)
        return canvas

    # No write, replace, rename or cleanup can touch a pre-existing PDF.
    with output.open("xb") as stream:
        document = BaseDocTemplate(stream, pagesize=(layout["page_width"], layout["page_height"]),
            leftMargin=layout["margin"], rightMargin=layout["margin"],
            topMargin=42, bottomMargin=46, title=title, author="KION research",
            pageCompression=1, allowSplitting=1)
        frame = Frame(layout["margin"], 46, layout["available_width"], layout["frame_height"],
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, id="report")
        document.addPageTemplates(PageTemplate(id="report", frames=[frame]))
        document.build(story, canvasmaker=canvas_factory)
    return {"state": "built-awaiting-all-page-visual-QA", "output": output.as_posix(),
            "bytes": output.stat().st_size, "pages": canvas_holder[0].total_pages,
            "source_sha256": checked["sha256"], "font": regular.as_posix(), "font_bold": heavy.as_posix(),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="UTF-8 Markdown; assets resolve beside this file")
    parser.add_argument("--output", type=Path, help="Fresh PDF path; its parent must exist")
    parser.add_argument("--font", help="Cyrillic regular TrueType font (supply with --font-bold)")
    parser.add_argument("--font-bold", help="Matching bold TrueType font")
    parser.add_argument("--check", action="store_true", help="Read-only standard-library source/link/image check")
    args = parser.parse_args(argv)
    try:
        require(args.check != (args.output is not None), "Choose exactly one of --check or --output")
        require((args.font is None) == (args.font_bold is None), "Supply both --font and --font-bold")
        source = args.source.resolve()
        result = inspect_source(source) if args.check else build_pdf(source, args.output.resolve(), args.font, args.font_bold)
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
