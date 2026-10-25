"""Check the preserved source tree without importing any scientific producer.

Run from any directory with Python 3.12+: python -B path/to/verify_sources.py.
This validates source bytes and Python syntax; it does not run ML or training.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.stat().st_size > 2 << 20:
        raise ValueError("Oversized source manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract") != "kion-unified-historical-source-v1":
        raise ValueError("Unknown source manifest contract")
    seen: set[str] = set()
    total = python_count = 0
    for row in manifest["files"]:
        relative = row["path"]
        if not isinstance(relative, str) or not relative:
            raise ValueError("Source path must be a nonempty string")
        portable = PurePosixPath(relative)
        if (
            "\\" in relative
            or portable.is_absolute()
            or ".." in portable.parts
            or not portable.parts
            or portable.parts[0] not in {"fullstudy", "matched-kl"}
            or ":" in relative
            or relative in seen
        ):
            raise ValueError("Unsafe or duplicate source path")
        unresolved = root / relative
        path = unresolved.resolve()
        if not path.is_relative_to(root) or unresolved.is_symlink():
            raise ValueError("Source path escapes the preserved tree")
        seen.add(relative)
        if path.stat().st_size != row["bytes"] or path.stat().st_size > 2 << 20:
            raise ValueError("Source size differs: " + relative)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError("Source SHA256 differs: " + relative)
        if path.suffix == ".py":
            tree = ast.parse(payload.decode("utf-8-sig"), filename=relative)
            compile(tree, relative, "exec")  # Compile only; never execute the code.
            python_count += 1
        total += len(payload)
    actual = {
        path.relative_to(root).as_posix()
        for subtree in (root / "fullstudy", root / "matched-kl")
        for path in subtree.rglob("*")
        if path.is_file()
    }
    if seen != actual:
        raise ValueError("Missing or undeclared preserved files: " + str(sorted(seen ^ actual)))
    if manifest["totals"] != {"files": len(seen), "bytes": total}:
        raise ValueError("Source totals differ")
    return {
        "state": "passed",
        "files": len(seen),
        "bytes": total,
        "python_files_parsed_and_compiled_without_execution": python_count,
        "portable_execution_claimed": False,
        "scope": "source bytes and syntax only; no producer, numerical import, model, data, subprocess or network",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.source_root), ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
