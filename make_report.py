"""Build the reader report with repository URLs using the frozen renderer.

--check uses only the standard library. Rendering retains the frozen builder's
fresh-output guard and requires ReportLab; no URL is opened by this wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "reports" / "render_report.py"
FROZEN_SHA256 = "dfbffc0e48cb11951038e714b7a577f165730255299fe3baf7f64d6016ef2c3c"
LINK_BASE = "https://github.com/suprend/kion-rl-recsys/blob/codex/research-report/"


def load_builder():
    raw = FROZEN.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_SHA256:
        raise ValueError("Frozen renderer SHA256 mismatch; refusing to execute it")
    spec = importlib.util.spec_from_file_location("_kion_frozen_report", FROZEN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations here.
    exec(compile(raw, str(FROZEN), "exec"), module.__dict__)
    return module


def reader_links(builder, root: Path):
    original = builder.resolve_reference
    root = root.resolve()

    def resolve(destination: str, source: Path):
        uri, local, fragment = original(destination, source)
        if local is None:
            return uri, local, fragment
        try:
            relative = local.relative_to(root)
        except ValueError:
            raise ValueError("Local link escapes report repository: " + destination) from None
        query = urlsplit(uri).query
        target = LINK_BASE + quote(relative.as_posix(), safe="/")
        if query:
            target += "?" + query
        if fragment:
            target += "#" + quote(fragment, safe="")
        return target, local, fragment

    builder.resolve_reference = resolve
    return builder


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh PDF path; its parent must exist")
    parser.add_argument("--font")
    parser.add_argument("--font-bold")
    parser.add_argument("--check", action="store_true", help="Read-only standard-library source check")
    args = parser.parse_args(argv)
    try:
        builder = reader_links(load_builder(), ROOT)
        builder.require(args.check != (args.output is not None), "Choose exactly one of --check or --output")
        builder.require((args.font is None) == (args.font_bold is None), "Supply both --font and --font-bold")
        source = args.source.resolve()
        try:
            source.relative_to(ROOT)
        except ValueError:
            raise ValueError("Source must be inside the report repository") from None
        result = (builder.inspect_source(source) if args.check else
                  builder.build_pdf(source, args.output.resolve(), args.font, args.font_bold))
        result.update(local_link_base=LINK_BASE, renderer_sha256=FROZEN_SHA256)
    except (ValueError, OSError, UnicodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
