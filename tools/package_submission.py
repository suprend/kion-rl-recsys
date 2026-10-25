"""Package verified current source files; verify ZIP bytes without extraction.

The embedded manifest provides integrity, not a signature or publisher identity.
No training, model loading, numerical replay, or external fixture inclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_repository

PREFIX = "kion-rl-recsys/"
MANIFEST = "SUBMISSION-MANIFEST.json"
FORMAT = "kion-current-source-submission-v1"
MAX_FILE = 64 << 20
MAX_TOTAL = 256 << 20
MAX_FILES = 2000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_path(value: str) -> str:
    require(isinstance(value, str) and bool(value), "Invalid path")
    path = PurePosixPath(value)
    require(not path.is_absolute() and path.as_posix() == value
            and not {".", ".."}.intersection(path.parts)
            and not any(c in value for c in "\\:\x00")
            and all(ord(c) >= 32 for c in value), "Unsafe path: " + value)
    require(not any(p.lower() in {".git", "__pycache__", ".venv", ".pytest_cache", ".env"}
                    or p.lower().startswith(".env.") for p in path.parts),
            "Forbidden path: " + value)
    require(all(not p.endswith((" ", ".")) for p in path.parts), "Nonportable path: " + value)
    return value


def closure(projection: dict) -> list[str]:
    paths = [row["path"] for row in projection["copied_files"]]
    paths += projection["authored_files_not_hashed"]
    require(0 < len(paths) <= MAX_FILES, "Invalid file count")
    paths = [safe_path(p) for p in paths]
    require(len({p.casefold() for p in paths}) == len(paths), "Duplicate projection path")
    require("projection.json" in paths and MANIFEST not in paths, "Invalid projection closure")
    return sorted(paths)


def record(path: str, data: bytes) -> dict:
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def verify_archive(archive: Path) -> dict:
    require(archive.stat().st_size <= MAX_TOTAL, "Oversized archive")
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        require(0 < len(members) <= MAX_FILES + 1, "Invalid member count")
        names = [m.filename for m in members]
        require(len(set(n.casefold() for n in names)) == len(names), "Duplicate ZIP member")
        require(sum(m.file_size for m in members) <= MAX_TOTAL, "Oversized expanded archive")
        for member in members:
            require(member.filename == MANIFEST or member.filename.startswith(PREFIX),
                    "Unexpected ZIP root")
            safe_path(member.filename if member.filename == MANIFEST else member.filename[len(PREFIX):])
            mode = member.external_attr >> 16
            require(not member.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG),
                    "Nonregular ZIP member")
            require(0 <= member.file_size <= MAX_FILE and not member.flag_bits & 1,
                    "Oversized or encrypted ZIP member")
        require(MANIFEST in names and PREFIX + "projection.json" in names,
                "Missing manifest or projection")
        require(bundle.getinfo(MANIFEST).file_size <= 2 << 20
                and bundle.getinfo(PREFIX + "projection.json").file_size <= 2 << 20,
                "Oversized manifest")
        manifest = json.loads(bundle.read(MANIFEST))
        require(manifest["format"] == FORMAT, "Unknown submission format")
        projection = json.loads(bundle.read(PREFIX + "projection.json"))
        paths = closure(projection)
        require(set(names) == {PREFIX + p for p in paths} | {MANIFEST},
                "ZIP differs from projection closure")
        rows = manifest["files"]
        require(isinstance(rows, list) and len(rows) == len(paths), "Invalid manifest rows")
        require([row["path"] for row in rows] == paths, "Manifest paths differ")
        total = 0
        for row in rows:
            info = bundle.getinfo(PREFIX + row["path"])
            require(type(row["bytes"]) is int and row["bytes"] == info.file_size, "Size mismatch")
            digest = hashlib.sha256()
            count = 0
            with bundle.open(info) as stream:
                while chunk := stream.read(1 << 20):
                    count += len(chunk)
                    require(count <= MAX_FILE, "Oversized streamed member")
                    digest.update(chunk)
            require(count == row["bytes"] and digest.hexdigest() == row["sha256"],
                    "SHA256 mismatch: " + row["path"])
            total += count
    return {"state": "passed", "format": FORMAT, "files": len(paths), "bytes": total,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "training_updates": 0}


def create_archive(repo: Path, output: Path) -> dict:
    repo = repo.resolve(strict=True)
    output = output.absolute()
    require(output.parent.is_dir(), "Output parent must exist")
    require(not output.parent.resolve().is_relative_to(repo), "Output must be outside repository")
    require(output.suffix.lower() == ".zip", "Output must be a .zip")
    require(not os.path.lexists(output), "Output already exists")
    verify_repository.verify(repo)
    projection_bytes = (repo / "projection.json").read_bytes()
    paths = closure(json.loads(projection_bytes))
    payload = {}
    total = 0
    for relative in paths:
        path = verify_repository.member(repo, relative)
        require(path.is_file() and path.stat().st_size <= MAX_FILE, "Invalid source size")
        data = path.read_bytes()
        total += len(data)
        require(len(data) <= MAX_FILE and total <= MAX_TOTAL - (2 << 20), "Oversized source")
        payload[relative] = data
    require(payload["projection.json"] == projection_bytes, "Projection changed during packaging")
    verify_repository.verify(repo)
    require(all(verify_repository.member(repo, p).read_bytes() == data for p, data in payload.items()),
            "Source changed during packaging")
    manifest = {"format": FORMAT, "files": [record(p, payload[p]) for p in paths]}
    payload[MANIFEST] = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".submission-", suffix=".tmp", dir=output.parent,
                                         delete=False) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
            for relative in sorted(payload):
                name = relative if relative == MANIFEST else PREFIX + relative
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(info, payload[relative], compresslevel=9)
        receipt = verify_archive(temporary)
        # Atomic no-clobber publication on the same filesystem (including NTFS).
        os.link(temporary, output)
        return {**receipt, "archive": output.as_posix()}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (create_archive(args.repo, args.output) if args.command == "create"
                  else verify_archive(args.archive))
    except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        parser.exit(1, "submission failed: " + str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
