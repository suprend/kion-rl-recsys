"""Read-only standard-library check of the source-only Git projection.

This hashes preserved payloads without importing producers, opening omitted
fixtures, running numerical inference, or making network requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


AUTHORED = {
    ".gitattributes", ".gitignore", "README.md", "REPRODUCING.md",
    "REPORT.md", "REPORT.pdf", "PROJECTION.md", "projection.json", "verify_repository.py",
    "make_report.py", "test_make_report.py",
}
FORBIDDEN_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".cbm", ".bin", ".ckpt"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def member(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "Invalid relative path")
    portable = PurePosixPath(relative)
    require(not portable.is_absolute() and portable.as_posix() == relative
            and not ({".", ".."} & set(portable.parts))
            and "\\" not in relative and ":" not in relative
            and all(ord(char) >= 32 for char in relative), "Unsafe path: " + relative)
    path = root
    for part in portable.parts:
        path /= part
        require(not path.is_symlink() and not path.is_junction(), "Linked path: " + relative)
    require(path.resolve().is_relative_to(root), "Escaping path: " + relative)
    return path


def verify(root: Path) -> dict:
    root = root.resolve(strict=True)
    require(root.is_dir(), "Repository root is not a directory")
    manifest_path = member(root, "projection.json")
    require(manifest_path.stat().st_size <= 2 << 20, "Oversized projection manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("contract") == "kion-source-only-git-projection-v1", "Unknown projection contract")
    require(set(manifest["authored_files_not_hashed"]) == AUTHORED, "Unexpected authored-file allowance")
    rows = manifest["copied_files"]
    require(type(rows) is list and 0 < len(rows) <= 1000, "Invalid copied-file list")
    seen = set()
    total = 0
    for row in rows:
        relative = row["path"]
        require(relative not in seen and relative not in AUTHORED, "Duplicate or authored copied path")
        path = member(root, relative)
        require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, "Forbidden copied payload: " + relative)
        require(path.is_file() and path.stat().st_size == row["bytes"], "Missing or changed size: " + relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], "Changed SHA256: " + relative)
        seen.add(relative)
        total += row["bytes"]
    require(manifest["copied_totals"] == {"files": len(seen), "bytes": total}, "Copied totals differ")
    excluded = set()
    omitted_bytes = 0
    for row in manifest["excluded_files"]:
        relative = row["path"]
        path = member(root, relative)
        require(relative not in excluded and relative not in seen, "Duplicate or overlapping exclusion")
        require(bool(row["reason"]), "Missing exclusion reason")
        if row["disposition"] == "replaced-by-repository-documentation":
            require(relative == "README.md", "Unexpected replacement")
        else:
            require(row["disposition"] == "omitted" and not path.exists(), "Excluded payload is present: " + relative)
        excluded.add(relative)
        omitted_bytes += row["bytes"]
    require(manifest["excluded_totals"] == {"files": len(excluded), "bytes": omitted_bytes}, "Excluded totals differ")
    require(manifest["upstream"]["payload_files"] == len(seen) + len(excluded)
            and manifest["upstream"]["payload_bytes"] == total + omitted_bytes, "Upstream accounting differs")
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        if base == root:
            dirs[:] = [name for name in dirs if name != ".git"]
            files = [name for name in files if name != ".git"]
        for name in dirs + files:
            relative = (base / name).relative_to(root).as_posix()
            member(root, relative)
        actual.update((base / name).relative_to(root).as_posix() for name in files)
    require(actual == seen | AUTHORED, "Unexpected or missing repository files: " + str(sorted(actual ^ (seen | AUTHORED))))
    return {"state": "passed", "copied_files": len(seen), "copied_bytes": total,
            "excluded_members": len(excluded), "stdlib_only": True,
            "numerical_execution": False, "training_updates": 0,
            "scope": "source-only checkout hashes and finite file closure; authored files are not checksummed"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root), ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, "repository verification failed: " + str(error) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
