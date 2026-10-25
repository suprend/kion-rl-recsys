"""Acquire the approved public E5/MovieLens assets; never execute remote code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import zipfile

MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_FILES = {
    "config.json": (655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    "model.safetensors": (470641600, "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
    "tokenizer.json": (17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    "tokenizer_config.json": (443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
    "special_tokens_map.json": (167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
}


def validate_model_request(request):
    if request.get("model_id") != MODEL_ID or request.get("revision") != MODEL_REVISION:
        raise ValueError("Unexpected model request")
    files = request.get("files")
    if not isinstance(files, dict) or set(files) != set(MODEL_FILES):
        raise ValueError("Exact five pinned model files required")
    for name, (size, checksum) in MODEL_FILES.items():
        spec = files[name]
        if (not isinstance(spec, dict) or type(spec.get("size")) is not int
                or spec["size"] != size or spec.get("sha256") != checksum
                or spec.get("url") != f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"):
            raise ValueError("Model pin/URL mismatch: " + name)
    if request.get("total_bytes") != sum(value[0] for value in MODEL_FILES.values()):
        raise ValueError("Model total size mismatch")


def digest(path, name="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, name).hexdigest()


def publish(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def fetch(url, target, *, powershell, expected_size, sha256=None, md5=None, timeout=600):
    if not url.startswith("https://"):
        raise ValueError("HTTPS required")
    if type(expected_size) is not int or expected_size <= 0:
        raise ValueError("Positive integer expected size required")
    if sha256 is None and md5 is None:
        raise ValueError("A checksum is mandatory")
    if ((sha256 is not None and re.fullmatch(r"[0-9a-f]{64}", sha256) is None)
            or (md5 is not None and re.fullmatch(r"[0-9a-f]{32}", md5) is None)):
        raise ValueError("Invalid checksum")
    if target.exists():
        if target.stat().st_size != expected_size:
            raise ValueError(f"Existing target size mismatch: {target}")
        if sha256 and digest(target) != sha256:
            raise ValueError(f"Existing target SHA mismatch: {target}")
        if md5 and digest(target, "md5") != md5:
            raise ValueError(f"Existing target official MD5 mismatch: {target}")
        return {"reused_verified": True, "bytes": expected_size, "sha256": digest(target)}
    part = target.with_name(target.name + ".partial")
    if part.exists():
        raise FileExistsError(f"Inspect previous partial before retry: {part}")
    if shutil.disk_usage(target.parent).free < expected_size + (2 << 30):
        raise ValueError("Insufficient free disk reserve")
    # PowerShell uses the Windows trust store. Do not disable TLS verification.
    source = Path(__file__).with_name("kion_download_asset.ps1")
    subprocess.run([powershell, "-NoProfile", "-NonInteractive", "-File", str(source),
                    "-Uri", url, "-OutFile", str(part)], check=True, timeout=timeout)
    if part.stat().st_size != expected_size:
        raise ValueError(f"Downloaded size mismatch: {part}")
    measured = digest(part)
    if sha256 and measured != sha256:
        raise ValueError(f"Downloaded SHA mismatch: {part}")
    if md5 and digest(part, "md5") != md5:
        raise ValueError(f"Downloaded official MD5 mismatch: {part}")
    part.rename(target)
    return {"reused_verified": False, "bytes": expected_size, "sha256": measured,
            "official_md5_verified": md5 is not None}


def extract_movielens(archive, output):
    wanted = ("ratings.csv", "movies.csv", "links.csv", "README.txt")
    required = {"ml-25m/" + name for name in wanted}
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        names = [entry.filename for entry in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        for entry in members:
            path = PurePosixPath(entry.filename)
            raw_name = entry.orig_filename
            if (path.is_absolute() or ".." in path.parts or "\\" in raw_name
                    or ":" in raw_name or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("Unsafe ZIP entry")
        if not required <= set(names):
            raise ValueError("Missing required MovieLens members")
        if sum(source.getinfo(name).file_size for name in required) > (3 << 30):
            raise ValueError("Uncompressed requested members exceed reserve")
        # Preflight every desired target/partial before writing the first file.
        for name in wanted:
            for candidate in (output / name, output / (name + ".partial")):
                if candidate.exists():
                    raise FileExistsError(candidate)
        output.mkdir(parents=True, exist_ok=True)
        outputs = {}
        for name in wanted:
            target = output / name
            if target.exists():
                raise FileExistsError(target)
            entry = source.getinfo("ml-25m/" + name)
            part = target.with_name(target.name + ".partial")
            with source.open(entry) as incoming, part.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1 << 20)
            if part.stat().st_size != entry.file_size:
                raise ValueError("ZIP extraction size mismatch")
            part.rename(target)
            outputs[name] = {"bytes": target.stat().st_size, "sha256": digest(target)}
        return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-request", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--powershell", required=True)
    args = parser.parse_args(argv)
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    request_bytes = args.model_request.read_bytes()
    request = json.loads(request_bytes)
    validate_model_request(request)
    model_dir = args.raw_dir / ("multilingual-e5-small-" + request["revision"])
    ml_dir = args.raw_dir / "movielens-25m-2019"
    for directory in (model_dir, ml_dir, args.receipt.parent):
        directory.mkdir(parents=True, exist_ok=True)
    receipt = {"state": "prepared", "owner": "/root", "started_utc": datetime.now(timezone.utc).isoformat(),
               "request_sha256": hashlib.sha256(request_bytes).hexdigest(), "model_dir": model_dir.as_posix(),
               "movielens_dir": ml_dir.as_posix(), "operations": [],
               "external_writes": False, "paid_requests": False, "remote_code_executed": False}
    publish(args.receipt, receipt)
    started = time.perf_counter()
    try:
        for name, spec in request["files"].items():
            if Path(name).name != name or name not in {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}:
                raise ValueError("Unexpected model file name")
            if not spec["url"].startswith("https://huggingface.co/intfloat/multilingual-e5-small/resolve/" + request["revision"] + "/"):
                raise ValueError("Unexpected model source")
            op = {"id": "e5-" + name, "url": spec["url"], "expected_bytes": spec["size"], "state": "submitted"}
            receipt["operations"].append(op)
            publish(args.receipt, receipt)
            op.update(fetch(spec["url"], model_dir / name, powershell=args.powershell,
                            expected_size=spec["size"], sha256=spec["sha256"]))
            op["state"] = "succeeded"
            publish(args.receipt, receipt)
            print(json.dumps({"event": "asset_verified", "file": name, "bytes": op["bytes"]}), flush=True)
        publish(model_dir / "acquisition.json", {"state": "verified", "model": request,
                "retrieved_utc": datetime.now(timezone.utc).isoformat()})
        archive = ml_dir / "ml-25m.zip"
        op = {"id": "movielens25m", "url": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
              "expected_bytes": 261978986, "official_md5": "6b51fb2759a8657d3bfcbfc42b592ada", "state": "submitted"}
        receipt["operations"].append(op)
        publish(args.receipt, receipt)
        op.update(fetch(op["url"], archive, powershell=args.powershell, expected_size=op["expected_bytes"], md5=op["official_md5"]))
        op["state"] = "succeeded"
        publish(args.receipt, receipt)
        op["extracted"] = extract_movielens(archive, ml_dir / "extracted")
        receipt["state"] = "succeeded"
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        publish(args.receipt, receipt)
        print(json.dumps({"state": "succeeded", "receipt": args.receipt.as_posix()}), flush=True)
    except BaseException as error:
        receipt["state"] = "failed"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        receipt["elapsed_seconds"] = time.perf_counter() - started
        publish(args.receipt, receipt)
        raise


if __name__ == "__main__":
    main()
