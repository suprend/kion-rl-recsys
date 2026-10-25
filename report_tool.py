"""Derive Markdown and build the canonical report with an explicit Tectonic.

Only the documented finite TeX subset is accepted. No TeX engine, network client,
or optional package is imported. ``markdown --check`` never writes files. A PDF
build always stages immutable inputs in a new external directory and requires a
new external output. PDF layout still requires independent visual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import stat
import subprocess
import sys
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
SOURCE = Path("report-latex/main.tex")
MARKDOWN = Path("REPORT.md")
BODY_BEGIN = "% REPORT_BODY_BEGIN"
BODY_END = "% REPORT_BODY_END"
NOTICE = "<!-- Generated from report-latex/main.tex by report_tool.py; edit the TeX source. -->"
MAX_SOURCE_BYTES = 2_000_000
MAX_ASSET_BYTES = 50_000_000
MAX_PDF_BYTES = 100_000_000
PACKAGES = frozenset({"fontenc", "inputenc", "babel", "geometry", "amsmath",
                      "booktabs", "graphicx", "hyperref", "fontspec", "longtable", "array"})
MAIN_FONT = "cmunrm.otf"
MAIN_FONT_STYLES = {"BoldFont": "cmunbx.otf", "ItalicFont": "cmunti.otf",
                    "BoldItalicFont": "cmunbi.otf"}
MATH_COMMANDS = frozenset({"beta", "alpha", "gamma", "delta", "Delta", "theta",
    "sigma", "mu", "pi", "approx", "times", "cdot", "le", "ge", "neq",
    "pm", "sum", "log", "exp", "in", "left", "right", "mathrm", "mathbf",
    "mathit", "frac", "sqrt", "text", "operatorname", "max", "mid", "Vert"})
ESCAPES = {"%": "%", "_": "_", "&": "&", "#": "#", "$": "$",
           "{": "{", "}": "}", " ": " ", ",": " ", ";": " "}


class ReportError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def escape_markdown(text: str) -> str:
    return "".join("\\" + c if c in "\\`*_#$[]<>|" else c for c in text)


def check_table_cell(value: str) -> None:
    # GFM splits even inline math/code on an unescaped pipe. Reject those
    # constructs rather than changing the TeX formula or code contents.
    escaped = False
    for character in value:
        require(character != "|" or escaped, "Unescaped pipes in table cells (including math/code) are unsupported")
        escaped = character == "\\" and not escaped


class Reader:
    """Balanced-brace scanner; command handling belongs to the finite grammar."""

    def __init__(self, text: str):
        self.text, self.i = text, 0

    def fail(self, message: str) -> None:
        line = self.text.count("\n", 0, self.i) + 1
        raise ReportError(f"TeX line {line}: {message}")

    def skip(self) -> None:
        while self.i < len(self.text):
            if self.text[self.i].isspace():
                self.i += 1
            elif self.text[self.i] == "%":
                self.comment()
            else:
                break

    def comment(self) -> None:
        while self.i < len(self.text) and self.text[self.i] != "\n":
            self.i += 1
        if self.i < len(self.text):
            self.i += 1

    def command(self) -> str:
        require(self.i < len(self.text) and self.text[self.i] == "\\", "Expected command")
        self.i += 1
        if self.i == len(self.text):
            self.fail("dangling backslash")
        start = self.i
        if self.text[self.i].isascii() and self.text[self.i].isalpha():
            while self.i < len(self.text) and self.text[self.i].isascii() and self.text[self.i].isalpha():
                self.i += 1
        else:
            self.i += 1
        return self.text[start:self.i]

    def peek(self, name: str) -> bool:
        saved = self.i
        answer = self.i < len(self.text) and self.text[self.i] == "\\" and self.command() == name
        self.i = saved
        return answer

    def group(self, opener: str = "{", closer: str = "}") -> str:
        self.skip()
        if self.i == len(self.text) or self.text[self.i] != opener:
            self.fail(f"expected {opener}")
        self.i += 1
        start, depth = self.i, 1
        parts = []
        while self.i < len(self.text):
            c = self.text[self.i]
            if c == "\\":
                self.i += 1
                if self.i == len(self.text):
                    self.fail("dangling backslash in group")
                self.i += 1
            elif c == "%":
                parts.append(self.text[start:self.i])
                self.comment()
                start = self.i
            elif c == opener:
                depth += 1
                self.i += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    parts.append(self.text[start:self.i])
                    self.i += 1
                    return "".join(parts)
                self.i += 1
            else:
                self.i += 1
        self.fail(f"unclosed {opener}")

    def optional(self) -> str | None:
        self.skip()
        return self.group("[", "]") if self.i < len(self.text) and self.text[self.i] == "[" else None

    def environment_end(self, name: str) -> bool:
        if not self.peek("end"):
            return False
        saved = self.i
        self.command()
        found = self.group()
        self.i = saved
        return found == name


def plain_metadata(value: str) -> str:
    require(not any(c in value for c in "\\{}$&#_^%"), "Metadata must be plain text")
    return " ".join(value.split())


def check_main_font(name: str, options: str | None) -> None:
    """Only the four named CMU Serif resources, without paths or TeX macros."""
    require(name == MAIN_FONT and options is not None, "Unsupported main font; require the bounded CMU Serif configuration")
    found = {}
    for entry in options.split(","):
        pair = entry.strip().split("=")
        require(len(pair) == 2, "Unsupported setmainfont option")
        key, value = (part.strip() for part in pair)
        require(key in MAIN_FONT_STYLES and key not in found and value == MAIN_FONT_STYLES[key],
                "Unsupported/duplicate setmainfont style or resource")
        found[key] = value
    require(found == MAIN_FONT_STYLES, "Require all explicit CMU Serif font styles")


def document_parts(text: str) -> tuple[str, str]:
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == BODY_BEGIN]
    ends = [i for i, line in enumerate(lines) if line.rstrip("\r\n") == BODY_END]
    require(len(starts) == len(ends) == 1 and starts[0] < ends[0], "Require one ordered pair of exact body markers")
    before = Reader("".join(lines[:starts[0]]))
    title, document, maketitle, documentclass = None, False, False, False
    packages, main_font = set(), False
    while True:
        before.skip()
        if before.i == len(before.text):
            break
        if before.text[before.i] != "\\":
            before.fail("only declarative preamble commands are supported")
        name = before.command()
        if name in {"documentclass", "usepackage"}:
            require(not document, "Package declaration after document start")
            option = before.optional()
            value = before.group()
            require(value and all(c.isascii() and (c.isalnum() or c in ",-") for c in value), "Unsafe package/class name")
            require(option is None or not any(c in option for c in "\\{}%"), "Unsupported package options")
            if name == "documentclass":
                require(value == "article" and not documentclass, "Require one article document class")
                documentclass = True
            else:
                require(set(value.split(",")).issubset(PACKAGES), "Unsupported package declaration")
                packages.update(value.split(","))
        elif name == "setmainfont":
            require(not document and "fontspec" in packages and not main_font,
                    "Require one setmainfont after fontspec and before document start")
            options = before.optional()
            check_main_font(before.group(), options)
            main_font = True
        elif name == "hypersetup":
            require(not document, "hypersetup must precede document")
            value = before.group()
            require(not any(c in value for c in "\\{}%"), "Unsupported hypersetup")
        elif name in {"widowpenalty", "clubpenalty"}:
            require(not document, "Page-break penalties must precede document")
            setting = re.match(r"=10000(?=\s|\\|$)", before.text[before.i:])
            require(setting is not None, "Only fixed page-break penalty =10000 is supported")
            before.i += len(setting.group())
        elif name == "begin":
            require(not document and before.group() == "document", "Expected one document start")
            document = True
        elif name in {"title", "author", "date"}:
            value = plain_metadata(before.group())
            if name == "title":
                require(title is None and value, "Require one nonempty title")
                title = value
        elif name == "maketitle":
            require(document and title is not None and not maketitle, "Require one maketitle after document start and title")
            maketitle = True
        else:
            before.fail("unsupported preamble command \\" + name)
    require(documentclass and document and maketitle and title is not None, "Missing document class/document/title/maketitle")
    require(("fontspec" in packages) == main_font, "fontspec requires the bounded explicit main font configuration")
    after = Reader("".join(lines[ends[0] + 1:]))
    after.skip()
    require(after.peek("end"), "Expected document end after body")
    after.command()
    require(after.group() == "document", "Expected document end")
    after.skip()
    require(after.i == len(after.text), "Unexpected content after document end")
    return title, "".join(lines[starts[0] + 1:ends[0]])


def relative_asset(value: str) -> Path:
    require(value and "\\" not in value and ":" not in value and "\x00" not in value, "Unsafe graphics path")
    path = Path(value)
    require(not path.is_absolute() and all(p not in {"", ".", ".."} for p in value.split("/")), "Graphics path must stay inside source directory")
    require(path.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg"}, "Unsupported graphics extension")
    return path


def hyperlink_destination(value: str) -> str:
    # These are links from the root REPORT artifacts, not TeX input assets.
    # Accept only literal escaped URL punctuation; never evaluate TeX macros.
    value = re.sub(r"\\([%_&#])", r"\1", value)
    require(0 < len(value) <= 2048 and not any(
        c.isspace() or not c.isprintable() or c in "\\{}<>" for c in value),
        "Unsafe hyperlink destination")
    try:
        parsed = urlsplit(value)
    except ValueError as error:
        raise ReportError("Malformed hyperlink destination") from error
    if parsed.scheme in {"http", "https"}:
        require(parsed.hostname is not None and parsed.username is None and parsed.password is None,
                "Require an explicit HTTP(S) host without credentials")
    else:
        require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment,
                "Only HTTP(S) or repository-relative file hyperlinks are supported")
        require(re.search(r"%(?![0-9a-fA-F]{2})", value) is None,
                "Malformed percent encoding in relative hyperlink")
        try:
            decoded = unquote(value, errors="strict")
        except UnicodeError as error:
            raise ReportError("Invalid UTF-8 in relative hyperlink") from error
        # Reject nested encoding, path separators hidden by encoding, Windows
        # drives/ADS, and every dot segment (a deliberately bounded subset).
        require(decoded.count("/") == value.count("/") and not any(
            c.isspace() or not c.isprintable() or c in "\\:{}<>%?#|\"*" for c in decoded),
            "Unsafe relative hyperlink")
        parts = decoded.split("/")
        reserved = {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)),
                    *("LPT" + str(i) for i in range(1, 10))}
        require(all(part not in {"", ".", ".."} and part.rstrip(" .") == part
                    and part.split(".")[0].upper() not in reserved for part in parts),
                "Relative hyperlink must stay inside the repository")
    return quote(value, safe="/:?#[]@!$&'*+,;=%~._-")


def inline_math(value: str) -> str:
    r, depth = Reader(value), 0
    while r.i < len(value):
        c = value[r.i]
        if c == "\\":
            name = r.command()
            require(name in MATH_COMMANDS or name in {",", ";", "!", " ", "%", "{", "}", "_"}, "Unsupported math command \\" + name)
        elif c == "{":
            depth += 1
            r.i += 1
        elif c == "}":
            depth -= 1
            require(depth >= 0, "Unbalanced math braces")
            r.i += 1
        else:
            require(c not in "$%&#\n\r", "Unsupported character in inline math")
            r.i += 1
    require(depth == 0 and value.strip(), "Empty/unbalanced inline math")
    return "$" + value.strip() + "$"


class Converter(Reader):
    def __init__(self, text: str, assets: list[Path] | None = None):
        super().__init__(text)
        self.assets = assets if assets is not None else []

    def inline(self, value: str) -> str:
        child = Converter(value, self.assets)
        result = []
        while child.i < len(value):
            result.append(child.piece())
        return "".join(result)

    def piece(self) -> str:
        c = self.text[self.i]
        if c.isspace():
            while self.i < len(self.text) and self.text[self.i].isspace():
                self.i += 1
            return " "
        if c == "%":
            self.comment()
            return ""
        if c == "~":
            self.i += 1
            return " "
        if c == "{":
            return self.inline(self.group())
        if c == "$":
            self.i += 1
            start = self.i
            while self.i < len(self.text) and self.text[self.i] != "$":
                if self.text[self.i] == "\\":
                    self.command()
                else:
                    self.i += 1
            require(self.i < len(self.text), "Unclosed inline math")
            value = self.text[start:self.i]
            self.i += 1
            return inline_math(value)
        if c == "\\":
            name = self.command()
            if name in ESCAPES:
                return escape_markdown(ESCAPES[name])
            if name in {"textbf", "emph", "texttt"}:
                raw = self.group()
                if name == "texttt":
                    code, literal = Reader(raw), []
                    while code.i < len(raw):
                        if raw[code.i] == "\\":
                            escape = code.command()
                            require(escape in ESCAPES, "Only plain text and escaped punctuation are supported in texttt")
                            literal.append(ESCAPES[escape])
                        else:
                            require(raw[code.i] not in "{}$&#_^", "Unescaped special character in texttt")
                            literal.append(raw[code.i])
                            code.i += 1
                    content = " ".join("".join(literal).split())
                    require(content, "Empty texttt")
                    fence = "`"
                    while fence in content:
                        fence += "`"
                    padding = " " if content.startswith("`") or content.endswith("`") else ""
                    return fence + padding + content + padding + fence
                content = self.inline(raw)
                require(content and content == content.strip(), "Formatting content must be nonempty without boundary whitespace")
                return ("**" if name == "textbf" else "*") + content + ("**" if name == "textbf" else "*")
            if name in {"href", "url"}:
                destination = hyperlink_destination(self.group())
                label = self.inline(self.group()) if name == "href" else escape_markdown(destination)
                return "[" + label + "](" + destination + ")"
            if name == "(":
                start = self.i
                while self.i < len(self.text) and not self.peek(")"):
                    if self.text[self.i] == "\\":
                        self.command()
                    else:
                        self.i += 1
                require(self.i < len(self.text), "Unclosed inline math")
                value = self.text[start:self.i]
                self.command()
                return inline_math(value)
            self.fail("unsupported inline command \\" + name)
        if c in "}&_^#":
            self.fail("unescaped TeX special character " + c)
        self.i += 1
        return escape_markdown(c)

    def blocks(self, stop: str | None = None, stop_item: bool = False) -> str:
        blocks = []
        while True:
            self.skip()
            if self.i == len(self.text):
                require(stop is None, "Unclosed environment " + str(stop))
                break
            if stop and self.environment_end(stop):
                if not stop_item:
                    self.command()
                    self.group()
                break
            if stop_item and self.peek("item"):
                break
            if self.peek("newpage"):
                self.command()
                self.skip()
                require(self.i == len(self.text) or self.text[self.i] not in "{[",
                        "newpage does not accept arguments")
            elif self.peek("section") or self.peek("subsection"):
                name = self.command()
                if self.i < len(self.text) and self.text[self.i] == "*":
                    self.i += 1
                blocks.append(("## " if name == "section" else "### ") + self.inline(self.group()))
            elif self.peek("["):
                self.command()
                start = self.i
                while self.i < len(self.text) and not self.peek("]"):
                    if self.text[self.i] == "\\":
                        self.command()
                    else:
                        self.i += 1
                require(self.i < len(self.text), "Unclosed display math")
                value = self.text[start:self.i]
                self.command()
                formula = inline_math(" ".join(value.split()))[1:-1]
                blocks.append("$$\n" + formula + "\n$$")
            elif self.peek("begin"):
                self.command()
                env = self.group()
                if env in {"itemize", "enumerate"}:
                    blocks.append(self.list_block(env))
                elif env == "center":
                    blocks.append(self.blocks("center"))
                elif env in {"tabular", "longtable"}:
                    blocks.append(self.table(env))
                elif env == "figure":
                    blocks.append(self.figure())
                else:
                    self.fail("unsupported environment " + env)
            else:
                pieces = []
                while self.i < len(self.text):
                    if any(self.peek(name) for name in ("section", "subsection", "begin", "end", "item", "newpage", "[")):
                        break
                    if self.text[self.i].isspace():
                        saved = self.i
                        while self.i < len(self.text) and self.text[self.i].isspace():
                            self.i += 1
                        if self.text[saved:self.i].count("\n") > 1:
                            break
                        pieces.append(" ")
                    else:
                        pieces.append(self.piece())
                value = "".join(pieces).strip()
                require(value, "Unexpected block command or empty paragraph")
                blocks.append(value)
        return "\n\n".join(blocks)

    def list_block(self, env: str) -> str:
        items = []
        while True:
            self.skip()
            if self.environment_end(env):
                self.command()
                self.group()
                break
            require(self.peek("item"), "List content must start with item")
            self.command()
            content = self.blocks(env, stop_item=True)
            require(content, "Empty list item")
            marker = "- " if env == "itemize" else f"{len(items) + 1}. "
            items.append(marker + content.replace("\n", "\n" + " " * len(marker)))
        require(items, "Empty list")
        return "\n".join(items)

    def table(self, env: str = "tabular") -> str:
        # Widths affect TeX layout only; paragraph columns are left-aligned in
        # Markdown. Never accept macros, expressions, or arbitrary TeX units.
        columns = Reader(self.group())
        spec = []
        while True:
            columns.skip()
            if columns.i == len(columns.text):
                break
            if columns.text[columns.i] == ">":
                columns.i += 1
                require(columns.group() == r"\raggedright\arraybackslash",
                        "Only the fixed ragged-right column modifier is supported")
                columns.skip()
                require(columns.i < len(columns.text) and columns.text[columns.i] == "p",
                        "Ragged-right modifier requires a paragraph column")
            kind = columns.text[columns.i]
            require(kind in "lcrp", "Only l/c/r and fixed-width p columns are supported")
            columns.i += 1
            if kind == "p":
                width = columns.group().strip()
                require(re.fullmatch(r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)\s*(?:cm|mm|pt)", width) is not None
                        and any(c in "123456789" for c in width),
                        "Paragraph column width must be a positive number in cm/mm/pt")
            spec.append("l" if kind == "p" else kind)
        require(spec, "Table requires columns")
        rows, cells, content, rules = [], [], [], []
        header_state = None
        while True:
            self.skip() if not content else None
            if self.environment_end(env):
                require(not cells and not "".join(content).strip(), "Table row must end with two backslashes")
                require(header_state != "first", "Missing repeated longtable header")
                self.command()
                self.group()
                break
            if self.peek("endfirsthead") or self.peek("endhead"):
                require(env == "longtable" and not cells and not "".join(content).strip(),
                        "Header markers require a complete longtable header")
                name = self.command()
                if name == "endfirsthead":
                    require(header_state is None and len(rows) == 1
                            and rules == [("toprule", 0), ("midrule", 1)],
                            "Require one first longtable header")
                    header_state = "first"
                else:
                    require(header_state == "first" and len(rows) == 2 and rows[0] == rows[1]
                            and rules == [("toprule", 0), ("midrule", 1), ("toprule", 1), ("midrule", 2)],
                            "Require an identical repeated longtable header")
                    rows.pop()
                    rules = rules[:2]
                    header_state = "complete"
                content = []
            elif any(self.peek(name) for name in ("toprule", "midrule", "bottomrule")):
                require(not cells and not "".join(content).strip(), "Rule inside table row")
                name = self.command()
                rules.append((name, len(rows)))
                content = []
            elif self.i == len(self.text):
                self.fail("unclosed " + env)
            elif self.text[self.i] == "&" or self.peek("\\"):
                cell = "".join(content).strip()
                check_table_cell(cell)
                cells.append(cell)
                content = []
                if self.text[self.i] == "&":
                    self.i += 1
                else:
                    self.command()
                    if self.i < len(self.text) and self.text[self.i] == "*":
                        # A row-break star only prevents a PDF page break.
                        self.i += 1
                        require(self.i == len(self.text) or self.text[self.i] != "*",
                                "Only one table row-break star is supported")
                    spacing = self.optional()
                    require(spacing is None or spacing == "4pt",
                            "Only a fixed 4pt table row gap is supported")
                    require(len(cells) == len(spec), "Table column count mismatch")
                    require(all(cells), "Empty table cell is unsupported")
                    rows.append(cells)
                    cells = []
            else:
                content.append(self.piece())
        require(len(rows) >= 2 and rules == [("toprule", 0), ("midrule", 1), ("bottomrule", len(rows))], "Require simple top/header/mid/body/bottom booktabs table")
        def row(values: list[str]) -> str:
            return "| " + " | ".join(values) + " |"
        align = [{"l": ":---", "c": ":---:", "r": "---:"}[c] for c in spec]
        return "\n".join([row(rows[0]), row(align)] + [row(r) for r in rows[1:]])

    def figure(self) -> str:
        option = self.optional()
        require(option is None or (option and all(c in "htbp!" for c in option)), "Unsupported figure placement")
        image, caption, centered = None, None, False
        while True:
            self.skip()
            if self.environment_end("figure"):
                self.command()
                self.group()
                break
            require(self.i < len(self.text) and self.text[self.i] == "\\", "Unsupported figure content")
            name = self.command()
            if name == "centering":
                require(not centered, "Repeated centering")
                centered = True
            elif name == "includegraphics":
                require(image is None, "Only one image per figure is supported")
                width = self.optional()
                require(width in {None, "width=\\textwidth", "width=\\linewidth"}, "Unsupported graphics options")
                image = relative_asset(self.group())
                self.assets.append(image)
            elif name == "caption":
                require(caption is None, "Only one caption per figure is supported")
                caption = self.inline(self.group())
            else:
                self.fail("unsupported figure command \\" + name)
        require(image is not None and caption, "Figure requires image and caption")
        destination = quote((SOURCE.parent / image).as_posix(), safe="/._-")
        if image.suffix.lower() == ".pdf":
            return "[Рисунок: " + caption + "](" + destination + ")"
        return "![" + caption + "](" + destination + ")"


def convert(text: str) -> tuple[str, list[Path]]:
    # TeX expands ^^ character encodings before tokenization, including inside
    # comments and options. Our source grammar deliberately has no such layer.
    require("^^" not in text, "TeX ^^ character encodings are unsupported anywhere in the source")
    title, body = document_parts(text)
    converter = Converter(body)
    content = converter.blocks()
    require(content, "Empty report body")
    return NOTICE + "\n\n# " + escape_markdown(title) + "\n\n" + content + "\n", sorted(set(converter.assets))


def plain_path(path: Path, *, existing: bool = True) -> Path:
    require(path.is_absolute() and ".." not in path.parts, "Require absolute path without parent traversal: " + str(path))
    require(not str(path).startswith(("\\\\", "//")), "Network paths are unsupported")
    require(all(":" not in part for part in path.parts[1:]), "Alternate stream paths are unsupported")
    if os.name == "nt":
        reserved = {"CON", "PRN", "AUX", "NUL", *("COM" + str(i) for i in range(1, 10)), *("LPT" + str(i) for i in range(1, 10))}
        require(all(part.rstrip(" .") == part and part.split(".")[0].upper() not in reserved for part in path.parts[1:]), "Ambiguous/reserved Windows path")
    for part in (path, *path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)), "Link/reparse path is unsupported: " + str(part))
    require(not existing or path.exists(), "Missing path: " + str(path))
    return path.resolve(strict=existing)


def read_bounded(path: Path, maximum: int) -> bytes:
    plain_path(path)
    require(path.is_file() and path.stat().st_size <= maximum, "Not a bounded regular file: " + str(path))
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    require(len(data) <= maximum, "File exceeds size limit: " + str(path))
    return data


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source(root: Path) -> tuple[str, dict[Path, bytes]]:
    root = plain_path(root)
    source = root / SOURCE
    raw = read_bounded(source, MAX_SOURCE_BYTES)
    markdown, assets = convert(raw.decode("utf-8"))
    files = {SOURCE: raw}
    for asset in assets:
        relative = SOURCE.parent / asset
        files[relative] = read_bounded(root / relative, MAX_ASSET_BYTES)
    return markdown, files


def markdown(root: Path, check: bool) -> dict:
    content, files = inspect_source(root)
    target = plain_path(root / MARKDOWN, existing=False)
    desired = content.encode("utf-8")
    if check:
        require(target.is_file() and target.read_bytes() == desired, "REPORT.md differs from canonical TeX; run markdown to regenerate")
    else:
        # Deliberately replace only this generated derivative after complete parse.
        temporary = target.with_name(target.name + ".report-tool.tmp")
        with temporary.open("xb") as stream:
            stream.write(desired)
        os.replace(temporary, target)
    return {"operation": "markdown-check" if check else "markdown", "markdown_sha256": sha256(desired),
            "inputs": {p.as_posix(): sha256(b) for p, b in files.items()}, "warnings": []}


def build(root: Path, tectonic: Path, output: Path, work_dir: Path, *, cache_dir: Path,
          allow_resource_downloads: bool = False, timeout: int = 120) -> dict:
    require(1 <= timeout <= 240, "Timeout must be between 1 and 240 seconds")
    root = plain_path(root)
    tectonic = plain_path(tectonic)
    require(tectonic.is_file(), "Tectonic must be an explicit executable file")
    output = plain_path(output, existing=False)
    work_dir = plain_path(work_dir, existing=False)
    cache_dir = plain_path(cache_dir)
    require(cache_dir.is_dir() and not cache_dir.is_relative_to(root), "Cache directory must exist outside repository")
    require(not output.is_relative_to(root) and not work_dir.is_relative_to(root), "PDF output and work directory must be outside repository")
    require(output.suffix.lower() == ".pdf" and not output.exists() and output.parent.is_dir(), "PDF output must be a new .pdf path in an existing directory")
    require(not work_dir.exists() and work_dir.parent.is_dir(), "Work directory must be new with an existing parent")
    require(not output.is_relative_to(work_dir) and not cache_dir.is_relative_to(work_dir) and not work_dir.is_relative_to(cache_dir), "Work, output and cache paths must be disjoint")
    _, files = inspect_source(root)
    compiler_hash = sha256(read_bounded(tectonic, MAX_PDF_BYTES))
    work_dir.mkdir()
    source_dir, out_dir = work_dir / "source", work_dir / "output"
    source_dir.mkdir()
    out_dir.mkdir()
    temp_dir = work_dir / "tmp"
    temp_dir.mkdir()
    for relative, data in files.items():
        target = source_dir / relative.relative_to(SOURCE.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    command = [str(tectonic), "--untrusted", "--keep-logs", "--keep-intermediates", "--outdir", str(out_dir)]
    if not allow_resource_downloads:
        command.append("--only-cached")
    command.append(str(source_dir / "main.tex"))
    environment = os.environ.copy()
    for name in ("TEXINPUTS", "BIBINPUTS", "BSTINPUTS", "TEXMFOUTPUT"):
        environment.pop(name, None)
    environment.update(TECTONIC_CACHE_DIR=str(cache_dir), TMP=str(temp_dir), TEMP=str(temp_dir))
    receipt = {"operation": "build", "status": "failed", "command": command, "timeout_seconds": timeout,
               "tectonic_sha256": compiler_hash, "cache_dir": cache_dir.as_posix(),
               "tool_sha256": file_hash(Path(__file__)),
               "allow_resource_downloads": allow_resource_downloads,
               "inputs": {p.as_posix(): sha256(b) for p, b in files.items()},
               "warnings": ["Compilation does not establish visual layout or scientific correctness.",
                            "Review stdout.log, stderr.log and output/main.log for compiler warnings."]}
    failure = None
    try:
        with (work_dir / "stdout.log").open("xb") as stdout, (work_dir / "stderr.log").open("xb") as stderr:
            process = subprocess.run(command, cwd=source_dir, env=environment, stdout=stdout, stderr=stderr,
                                     timeout=timeout, check=False, shell=False)
        receipt["returncode"] = process.returncode
        require(process.returncode == 0, "Tectonic returned " + str(process.returncode))
        pdf = read_bounded(out_dir / "main.pdf", MAX_PDF_BYTES)
        require(pdf.startswith(b"%PDF-"), "Compiler output does not have a PDF header")
        plain_path(output, existing=False)
        with output.open("xb") as stream:
            stream.write(pdf)
        receipt.update(status="compiled-unreviewed", output=output.as_posix(), output_sha256=sha256(pdf), output_bytes=len(pdf))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        failure = error
        receipt["error"] = f"{type(error).__name__}: {error}"
    for name in ("stdout.log", "stderr.log"):
        if (work_dir / name).is_file():
            receipt[name + "_sha256"] = file_hash(work_dir / name)
    receipt_path = work_dir / "build-receipt.json"
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if failure:
        raise ReportError(f"Build failed; inspect {receipt_path}: {failure}") from failure
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    md = commands.add_parser("markdown", help="Regenerate the Markdown derivative of main.tex")
    md.add_argument("--check", action="store_true", help="Check exact synchronization without writing")
    pdf = commands.add_parser("build", help="Build a new external PDF with an explicit Tectonic")
    pdf.add_argument("--tectonic", type=Path, required=True)
    pdf.add_argument("--output", type=Path, required=True)
    pdf.add_argument("--work-dir", type=Path, required=True)
    pdf.add_argument("--cache-dir", type=Path, required=True)
    pdf.add_argument("--allow-resource-downloads", action="store_true", help="Explicitly allow Tectonic to populate its TeX resource cache")
    pdf.add_argument("--timeout", type=int, default=120, help="Compiler timeout in seconds, maximum 240")
    args = parser.parse_args(argv)
    try:
        result = markdown(ROOT, args.check) if args.command == "markdown" else build(
            ROOT, args.tectonic, args.output, args.work_dir, cache_dir=args.cache_dir,
            allow_resource_downloads=args.allow_resource_downloads, timeout=args.timeout)
    except (OSError, ValueError, UnicodeError) as error:
        parser.error(str(error))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
