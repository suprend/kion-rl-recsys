"""Inspect an explicit portable input manifest without executing experiment code.

Manifest: schema_version=1, profile, closure_complete, entries. Each entry has
id, source_root_alias, relative_path (POSIX), role (dependency/provenance),
required (bool), optional expected_bytes and sha256. Roots JSON maps aliases to
absolute local directories. Missing pins never count as verified. The supplied
closure_complete flag is an assertion by the manifest author, not discovered here.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import sys


def checked_path(path):
    """Reject symlinks and Windows reparse points in every existing component."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("root/output paths must be absolute")
    for part in reversed((path, *path.parents)):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("symlink/junction/reparse point is not allowed")
    return path.resolve()


def relative_parts(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("relative_path must be a nonempty POSIX path")
    parts = value.split("/")
    if any(p in ("", ".", "..") or any(c in p for c in ':*?[]<>|"')
           or p.endswith((" ", ".")) or PureWindowsPath(p).is_reserved()
           or any(ord(c) < 32 for c in p) for p in parts):
        raise ValueError("unsafe relative_path")
    return parts


def validate(manifest, roots):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported manifest schema_version")
    if not isinstance(manifest.get("profile"), str) or not manifest["profile"]:
        raise ValueError("profile is required")
    if type(manifest.get("closure_complete")) is not bool:
        raise ValueError("closure_complete must be explicit boolean")
    if not isinstance(roots, dict):
        raise ValueError("roots must be an alias-to-absolute-directory object")
    resolved = {}
    for alias, value in roots.items():
        if not isinstance(value, str):
            raise ValueError("root must be a path string")
        resolved[alias] = checked_path(value)
        if resolved[alias].exists() and not resolved[alias].is_dir():
            raise ValueError("root must name a directory")
    if not isinstance(manifest.get("entries"), list) or not manifest["entries"]:
        raise ValueError("entries must be a nonempty list")
    ids = set()
    for entry in manifest["entries"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError("entry requires an id")
        if entry["id"] in ids:
            raise ValueError("duplicate entry id")
        ids.add(entry["id"])
        relative_parts(entry.get("relative_path"))
        if not isinstance(entry.get("source_root_alias"), str) or not entry["source_root_alias"]:
            raise ValueError("entry requires source_root_alias")
        if entry.get("role") not in ("dependency", "provenance"):
            raise ValueError("role must be dependency or provenance")
        if type(entry.get("required")) is not bool:
            raise ValueError("required must be explicit boolean")
        size = entry.get("expected_bytes")
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("expected_bytes must be a nonnegative integer or null")
        digest = entry.get("sha256")
        if digest is not None and (not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest)):
            raise ValueError("sha256 must be 64 hex characters or null")
    if not any(entry["required"] and entry["role"] == "dependency" for entry in manifest["entries"]):
        raise ValueError("at least one required dependency is necessary")
    return resolved


def inspect(manifest, roots, verify=False):
    resolved = validate(manifest, roots)
    results = []
    for entry in manifest["entries"]:
        result = {key: entry.get(key) for key in (
            "id", "source_root_alias", "relative_path", "role", "required", "expected_bytes", "sha256")}
        result.update(status="unverified", exists=False, actual_bytes=None, actual_sha256=None)
        results.append(result)
        root = resolved.get(entry["source_root_alias"])
        if root is None:
            result.update(status="unresolved_root")
            continue
        try:
            path = checked_path(root.joinpath(*relative_parts(entry["relative_path"])))
            if not path.is_relative_to(root):
                raise ValueError("input escapes source root")
            info = path.stat()
            result.update(exists=True, actual_bytes=info.st_size)
            if not stat.S_ISREG(info.st_mode):
                result.update(status="not_file")
            elif entry.get("expected_bytes") is not None and info.st_size != entry["expected_bytes"]:
                result.update(status="size_mismatch")
            elif entry.get("expected_bytes") is None or entry.get("sha256") is None:
                result.update(status="unpinned")
            elif verify:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                after = path.stat()
                result["actual_sha256"] = digest.hexdigest()
                if (info.st_size, info.st_mtime_ns, info.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    result.update(status="changed_during_verify")
                else:
                    result["status"] = "verified" if digest.hexdigest() == entry["sha256"].lower() else "hash_mismatch"
        except FileNotFoundError:
            result.update(status="missing")
        except (OSError, ValueError) as exc:
            result.update(status="unsafe_or_unreadable", error=str(exc))
    required_ok = all(row["status"] == "verified" for row in results if row["required"])
    return {
        "schema_version": 1, "profile": manifest["profile"],
        "mode": "verify" if verify else "inspect", "hash_threads": 1,
        "closure_complete_declared": manifest["closure_complete"],
        "required_inputs_verified": required_ok,
        "ready": bool(verify and required_ok and manifest["closure_complete"]),
        "scope": "Declared files only; no transitive closure discovery or runtime validation.",
        "counts": dict(Counter(row["status"] for row in results)), "entries": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true", help="stream declared pinned files sequentially")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        roots = json.loads(args.roots.read_text(encoding="utf-8"))
        resolved = validate(manifest, roots)
        output = checked_path(args.output.absolute())
        if any(output.is_relative_to(root) for root in resolved.values()):
            raise ValueError("output must be outside all input roots")
        if output.exists():
            raise ValueError("output must be a new file")
        report = inspect(manifest, roots, args.verify)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps({"output": output.as_posix(), "ready": report["ready"], "counts": report["counts"]}))
        return 0 if report["ready"] else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"input inventory: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
