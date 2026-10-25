"""Download and verify the complete public KION archive pinned by RecTools."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

REVISION = "f69775be31fa5779907cf0a92ddedb70037fb5ae"
URL = f"https://raw.githubusercontent.com/irsafilo/KION_DATASET/{REVISION}/data_original.zip"
ARCHIVE_SHA256 = "50fc5b7db78cbda4c3d0ee5bb5bea4c416cff41e8af58aab0b661d0d3e9dfa9b"
FILE_SHA256 = {
    "interactions.csv": "201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf",
    "items.csv": "027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e",
    "users.csv": "37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/kion"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive_path = args.output / "data_original.zip"
    if not archive_path.exists():
        partial = archive_path.with_suffix(".zip.part")
        # A partial file is evidence of interruption; never silently replace it.
        with urllib.request.urlopen(URL, timeout=120) as src, partial.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        if digest(partial) != ARCHIVE_SHA256:
            raise ValueError("Downloaded archive SHA-256 mismatch; partial file preserved")
        partial.rename(archive_path)
    if digest(archive_path) != ARCHIVE_SHA256:
        raise ValueError("Existing archive SHA-256 mismatch; file preserved")
    output = args.output / "full"
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC failure")
        for name, expected in FILE_SHA256.items():
            target = output / name
            if not target.exists():
                with archive.open("data_original/" + name) as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(target) != expected:
                raise ValueError(f"{name} SHA-256 mismatch; file preserved")
    manifest = {
        "source": URL, "revision": REVISION, "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": ARCHIVE_SHA256, "files_sha256": FILE_SHA256,
        "warning": "Use full/*.csv. The standalone interactions.csv.zip is truncated.",
    }
    receipt = output / "verified-download.json"
    payload = json.dumps(manifest, indent=2) + "\n"
    if receipt.exists() and receipt.read_text(encoding="utf-8") != payload:
        raise ValueError("Existing receipt differs; preserved")
    if not receipt.exists():
        receipt.write_text(payload, encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
