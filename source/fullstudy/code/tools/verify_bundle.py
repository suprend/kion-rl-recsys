"""Verify the small source bundle using only the Python standard library.

No producer module is imported and no model, array, subprocess or network is opened.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def read_json(path: Path, maximum: int = 2 << 20):
    if path.stat().st_size > maximum:
        raise ValueError(f"Oversized metadata: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def inside(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or "\\" in relative:
        raise ValueError(f"Expected a relative POSIX bundle path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path leaves bundle: {relative}")
    return path


def verify(report_root: Path):
    report_root = report_root.resolve()
    manifest = read_json(report_root / "code-manifest.json")
    if manifest.get("contract") != "kion-small-source-bundle-v1":
        raise ValueError("Unknown source bundle contract")
    seen, total, python_files = set(), 0, 0
    for row in manifest["files"]:
        relative = row["bundle_path"]
        if relative in seen:
            raise ValueError(f"Duplicate bundle path: {relative}")
        seen.add(relative)
        path = inside(report_root, relative)
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"Size mismatch: {relative}")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA256 mismatch: {relative}")
        if path.suffix == ".py":
            ast.parse(payload.decode("utf-8-sig"), filename=relative)
            python_files += 1
        total += len(payload)
    actual = {
        path.relative_to(report_root).as_posix()
        for path in (report_root / "code").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if actual != seen:
        raise ValueError(f"Undeclared or missing code files: {sorted(actual ^ seen)}")
    if manifest["totals"] != {"files": len(seen), "bytes": total}:
        raise ValueError("Manifest totals mismatch")
    fixture = read_json(report_root / "code/fixtures/synthetic-ranking.json")
    if fixture.get("synthetic") is not True:
        raise ValueError("Fixture must be explicitly invented")
    import math
    ranks = fixture["target_ranks"]
    result = {
        "queries": len(ranks),
        "map10": sum(1 / r if r and r <= 10 else 0 for r in ranks) / len(ranks),
        "recall10": sum(bool(r and r <= 10) for r in ranks) / len(ranks),
        "ndcg10": sum(1 / math.log2(r + 1) if r and r <= 10 else 0 for r in ranks) / len(ranks),
    }
    for key, expected in fixture["expected"].items():
        if not math.isclose(result[key], expected, rel_tol=0, abs_tol=1e-14):
            raise ValueError(f"Synthetic aggregate mismatch: {key}")
    return {"state": "passed", "files": len(seen), "bytes": total,
            "python_files_parsed": python_files, "synthetic": result,
            "scope": "source bytes, Python syntax and invented ranking arithmetic; no ML replay"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(verify(args.report_root), ensure_ascii=False, allow_nan=False, indent=2))
