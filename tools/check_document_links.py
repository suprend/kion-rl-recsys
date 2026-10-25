"""Read-only, offline link check for reader-facing Markdown and report LaTeX.

Default scope excludes immutable evidence/source snapshots. Pass explicit files or
directories to inspect those too. External URLs are never fetched; repository
URLs are checked against local files, not against GitHub access permissions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPO = "/suprend/kion-rl-recsys/"


def prose(text: str) -> str:
    """Blank code blocks/spans while preserving positions and line numbers."""
    blank = lambda m: re.sub(r"[^\n]", " ", m.group())
    text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", blank, text)
    return re.sub(r"(`+)[^`\n]*?\1", blank, text)


def anchors(text: str) -> set[str]:
    found, counts = set(), Counter()
    without_fences = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", text)
    for heading in re.findall(r"(?m)^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", without_fences):
        heading = re.sub(r"!?\[([^\]]+)\]\([^)]*\)", r"\1", heading)
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        suffix = f"-{counts[slug]}" if counts[slug] else ""
        counts[slug] += 1
        found.add(slug + suffix)
    found.update(re.findall(r'(?:id|name)=["\']([^"\']+)["\']', text))
    return found


def links(path: Path, text: str):
    if path.suffix == ".tex":
        patterns = [r"\\(?:href|url)\{([^}]+)\}",
                    r"\\(?:includegraphics|input|include)(?:\[[^]]*\])?\{([^}]+)\}"]
        source = re.sub(r"(?m)(?<!\\)%[^\n]*", "", text)
    else:
        source = prose(text)
        patterns = [r"!?\[[^\]\n]*\]\(\s*<?([^\s>)]+)>?(?:\s+[\"'][^\n]*?[\"'])?\s*\)",
                    r"(?m)^ {0,3}\[[^\]]+\]:\s*<?([^\s>]+)",
                    r'<(?:a|img)\b[^>]*?(?:href|src)=["\']([^"\']+)["\']']
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            kind = "href" if path.suffix == ".tex" and pattern == patterns[0] else "file"
            yield source.count("\n", 0, match.start()) + 1, match.group(1), kind


def check(root: Path, paths: list[Path], branch: str) -> tuple[list[str], Counter]:
    problems, stats = [], Counter()
    for path in paths:
        relative = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.as_posix()
        text = path.read_text(encoding="utf-8")
        stats["documents"] += 1
        for line, raw, kind in links(path, text):
            stats["links"] += 1
            url = raw.replace(r"\_", "_").replace(r"\%", "%").replace(r"\#", "#")
            parsed = urlsplit(url)
            prefix = f"{relative}:{line}: "
            if parsed.scheme in {"http", "https"}:
                if parsed.netloc.lower() != "github.com" or not parsed.path.startswith(REPO):
                    stats["external_unchecked"] += 1
                    continue
                tail = parsed.path[len(REPO):]
                if not tail.startswith(("blob/", "tree/")):
                    stats["external_unchecked"] += 1
                    continue
                tail = tail.split("/", 1)[1]
                if not tail.startswith(branch + "/"):
                    problems.append(prefix + f"stale/noncanonical branch URL (expected {branch}): {raw}")
                    continue
                target = root / unquote(tail[len(branch) + 1:])
                stats["github_local_only"] += 1
            elif parsed.scheme or url.startswith(("/", "\\")):
                problems.append(prefix + "nonportable local or unsupported URL: " + raw)
                continue
            else:
                # Report hyperlinks are consumed from root REPORT.pdf/REPORT.md;
                # TeX inputs and figures are resolved next to their source file.
                base = root if path.suffix == ".tex" and kind == "href" else path.parent
                target = (base / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.exists() and path.suffix == ".tex" and not target.suffix:
                target = next((target.with_suffix(ext) for ext in (".tex", ".pdf", ".png", ".jpg")
                               if target.with_suffix(ext).exists()), target)
            if not target.exists():
                problems.append(prefix + "missing target: " + raw)
            elif parsed.fragment and target.suffix.lower() == ".md":
                if unquote(parsed.fragment) not in anchors(target.read_text(encoding="utf-8")):
                    problems.append(prefix + "missing Markdown anchor: " + raw)
        if path.suffix == ".tex":
            labels = set(re.findall(r"\\label\{([^}]+)\}", text))
            for match in re.finditer(r"\\(?:ref|pageref|autoref|eqref)\{([^}]+)\}", text):
                if match.group(1) not in labels:
                    problems.append(f"{relative}:{text.count(chr(10), 0, match.start()) + 1}: missing LaTeX label: {match.group(1)}")
    return problems, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="Files/directories relative to --root; default reader-facing docs")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--branch", default="codex/research-report", help="Expected GitHub link branch (default codex/research-report)")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.paths:
        paths = []
        for name in args.paths:
            path = root / name
            if not path.exists():
                parser.error(f"missing requested path: {name}")
            paths.extend(p for p in path.rglob("*") if p.suffix in {".md", ".tex"}) if path.is_dir() else paths.append(path)
    else:
        paths = list(root.glob("*.md"))
        for name in ("report-latex", "reports", "reproduction", "research", "delivery"):
            paths.extend(p for p in (root / name).rglob("*") if p.suffix in {".md", ".tex"})
        paths.extend(root / name for name in ("source/README.md", "evidence/README.md") if (root / name).exists())
    problems, stats = check(root, sorted(set(paths)), args.branch)
    for problem in problems:
        print(problem)
    print(f"{'FAIL' if problems else 'PASS'}: {dict(stats)}; problems={len(problems)}")
    print("Offline only: external availability, GitHub/public permissions and PDF viewer behavior were not checked.")
    print("Default scope: reader-facing docs; frozen source/evidence Markdown snapshots excluded.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
