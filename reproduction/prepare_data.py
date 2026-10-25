"""Pinned raw split launcher. Default: stat-only input plan, no NumPy import."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_prepare.py"
SOURCE_SHA256 = "81e8ac99a6d50398418fd95e0874bc07b55b3d33487e89731ec26397960c3e77"
RAW = {
    "interactions.csv": (180032258, "201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf"),
    "users.csv": (29945389, "37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c"),
    "items.csv": (31836561, "027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e"),
}
PARAMETERS = {"context": 64, "max-train": 1000000000, "max-eval": 1000000000, "seed": 2026}


def source_path(root=ROOT):
    path = Path(root) / SOURCE
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Pinned source missing or exceeds 1 MiB")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Pinned kion_prepare.py SHA256 mismatch")
    return path.resolve()


def plan(input_dir, output_dir, python=sys.executable, root=ROOT):
    source = source_path(root)
    raw = Path(input_dir).resolve(strict=True)
    output = Path(output_dir).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("Output must be a new, absent directory")
    output = output.resolve()
    if raw == output or raw in output.parents or output in raw.parents:
        raise ValueError("Output and raw input trees must be disjoint")
    # Preserve venv launchers: resolving bin/python symlinks loses pyvenv.cfg.
    interpreter = Path(python).absolute()
    if not interpreter.is_file():
        raise ValueError("Python must be an explicit executable file")
    inputs = []
    for name, (size, digest) in RAW.items():
        path = raw / name
        if not path.is_file():
            raise ValueError(f"Missing raw CSV: {name}")
        actual = path.stat().st_size
        if actual != size:
            raise ValueError(f"Raw size mismatch for {name}: {actual} != {size}")
        inputs.append({"path": path.as_posix(), "bytes": actual,
                       "expected_sha256": digest, "content_verified": False})
    argv = [interpreter.as_posix(), "-B", source.as_posix(), "--input", raw.as_posix(), "--output", output.as_posix()]
    for key, value in PARAMETERS.items():
        argv.extend([f"--{key}", str(value)])
    return {"contract": "kion-raw-split-plan-v1", "mode": "plan",
            "source": source.as_posix(), "source_sha256": SOURCE_SHA256,
            "inputs": inputs, "output": output.as_posix(), "argv": argv,
            "parameters": dict(PARAMETERS), "input_content_verified": False,
            "environment_verified": False,
            "scope": "raw temporal split only; no features, models or training"}


def execute(input_dir, output_dir, python=sys.executable, root=ROOT, timeout_seconds=3600):
    if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    request = plan(input_dir, output_dir, python, root)
    # This full payload scan is deliberately confined to the explicit opt-in.
    for item in request["inputs"]:
        with Path(item["path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["expected_sha256"]:
            raise ValueError(f"Raw SHA256 mismatch: {item['path']}")
        item["content_verified"] = True
    source_path(root)  # Recheck after the potentially long payload verification.
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=False)
    request.update(mode="execute-prepare", input_content_verified=True, status="started")
    receipt = output / "prepare-launch.json"
    receipt.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n")
    try:
        subprocess.run(request["argv"], check=True, timeout=timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        request["status"] = "failed-or-timed-out"
        receipt.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n")
        raise
    request.update(status="source-cli-completed", semantic_output_equivalence_verified=False)
    receipt.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8", newline="\n")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--execute-prepare", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    try:
        if args.execute_prepare:
            result = execute(args.input, args.output, args.python, timeout_seconds=args.timeout_seconds)
        else:
            result = plan(args.input, args.output, args.python)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"prepare_data: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
