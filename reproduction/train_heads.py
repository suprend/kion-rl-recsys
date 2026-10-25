"""Portable exact A/B/C training entrypoint; default is a metadata-only plan.

Only --execute-training loads the preserved worker or scientific libraries.
The ten large arrays are stat'ed in plans, and fully hashed by heads.run at
execution (no verification receipt is passed). Teacher paths remain provenance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import stat
import sys
import time

from input_inventory import checked_path, relative_parts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/three-head-262k-20261024-r1/heads.py"
SOURCE_SHA256 = "795cdfe87f08079cadf067dc9c58811f7bcdacf9d1ebf2dbcb6c019dd68d6799"
FEATURE_SCHEMA = "71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d"
SOURCE_COLUMNS = ["current_detached_gpt_logit", "log1p_strict7d_count"]
HEAD_CONFIG = {"raw_width": 1068, "source_width": 2, "projection_width": 32,
               "hidden_width": 256, "mlp_input_width": 800, "dropout": 0.0,
               "projection_bias": False, "final_output_initialization": "zeros",
               "other_initialization": "fresh-common-torch-default",
               "lr": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0,
               "batch_size": 128, "positive_column": 0, "loss": "cross_entropy32",
               "amp": "cuda-fp16-with-finite-retry-then-fp32", "seed": 20261024}
ARRAY_SPEC = {
    "encoded": ("float32", [262144, 256]),
    "candidate_ids": ("int32", [262144, 32]),
    "base_logits": ("float32", [262144, 32]),
    "raw_features": ("float32", [262144, 32, 1068]),
    "source_values": ("float32", [262144, 32, 2]),
    "row_indices": ("int64", [262144]), "user_ids": ("int64", [262144]),
    "target_ids": ("int32", [262144]), "dates": ("int32", [262144]),
    "item_table": ("float32", [15418, 256]),
}


def read_metadata(path):
    path = checked_path(Path(path).absolute())
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Metadata exceeds 8 MiB")
    return path.read_bytes()


def digest_pin(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Explicit lowercase SHA-256 pin required")
    return value


def fresh_output(value, bank):
    output = checked_path(Path(value).absolute())
    source = checked_path(ROOT / "source")
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (bank, source)):
        raise ValueError("Output must be disjoint from bank and preserved source")
    if output.exists():
        raise FileExistsError("Output must be a fresh path; existing outputs are never overwritten")
    return output


def build_plan(bank_dir, output_dir, bank_manifest_sha256, epochs=3):
    if type(epochs) is not int or epochs != 3:
        raise ValueError("Exactly 3 epochs required; no shortened training")
    bank = checked_path(Path(bank_dir).absolute())
    output = fresh_output(output_dir, bank)
    data = read_metadata(bank / "bank.json")
    pin = digest_pin(bank_manifest_sha256)
    if hashlib.sha256(data).hexdigest() != pin:
        raise ValueError("Bank manifest pin mismatch")
    manifest = json.loads(data)
    required = {"contract": "kion-three-head-262k-bank-v1", "state": "completed",
                "split": "train", "train_rows": 262144,
                "natural_covered_only": False, "same_day_positives_excluded": True,
                "feature_schema_sha256": FEATURE_SCHEMA, "source_columns": SOURCE_COLUMNS}
    for key, value in required.items():
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value:
            raise ValueError(f"Exact bank requirement differs: {key}")
    teacher = manifest.get("teacher")
    if not isinstance(teacher, dict) or not teacher:
        raise ValueError("Frozen teacher provenance required")
    for key in ("sha256", "config_sha256"):
        digest_pin(teacher.get(key))
    for key in ("path", "config_path"):
        if not isinstance(teacher.get(key), str) or not teacher[key]:
            raise ValueError("Teacher provenance paths required")
    arrays = manifest.get("arrays")
    if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_SPEC):
        raise ValueError("Exactly ten bank arrays required")
    inputs, paths = [], set()
    for name, (dtype, shape) in ARRAY_SPEC.items():
        entry = arrays[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid array entry: {name}")
        path = checked_path(bank.joinpath(*relative_parts(entry.get("path"))))
        if not path.is_relative_to(bank) or path in paths:
            raise ValueError("Array paths must be unique and contained in bank")
        paths.add(path)
        # Exact original NPY layout: 128-byte header plus dense payload.
        payload = math.prod(shape) * (8 if dtype == "int64" else 4)
        size = payload + 128
        if (entry.get("dtype") != dtype or entry.get("shape") != shape
                or any(type(n) is not int for n in entry.get("shape", []))
                or type(entry.get("bytes")) is not int or entry["bytes"] != size
                or entry.get("payload_bytes") != payload):
            raise ValueError(f"Exact array shape/dtype/size differs: {name}")
        digest_pin(entry.get("sha256"))
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError(f"Array file size differs: {name}")
        inputs.append({"name": name, **entry, "status": "metadata-and-size-only-not-hash-verified"})
    request = {"bank_dir": bank.as_posix(), "output_dir": output.as_posix(),
               "bank_manifest_sha256": pin, "feature_schema_sha256": FEATURE_SCHEMA,
               "source_columns": list(SOURCE_COLUMNS), "teacher": teacher,
               "epochs": 3, "seed": 20261024, "batch_size": 128, "device": "cuda"}
    return {"mode": "plan", "training_executed": False, "request": request,
            "source": SOURCE, "source_sha256": SOURCE_SHA256,
            "head_config": HEAD_CONFIG,
            "requirements": inputs, "steps_per_head": 6144,
            "execution": "Explicit --execute-training and --max-seconds required; all ten arrays are hashed before updates.",
            "provenance": "Teacher paths are opaque unchanged lineage; no teacher files are opened for this frozen-bank stage."}


def load_worker():
    source_root = checked_path(ROOT / "source")
    path = checked_path(source_root / SOURCE)
    manifest = json.loads(read_metadata(source_root / "manifest.json"))
    entries = [e for e in manifest["files"] if e.get("path") == SOURCE]
    data = read_metadata(path)
    if (len(entries) != 1 or entries[0].get("sha256") != SOURCE_SHA256
            or entries[0].get("bytes") != len(data)
            or hashlib.sha256(data).hexdigest() != SOURCE_SHA256):
        raise ValueError("Preserved heads.py source hash/manifest mismatch")
    spec = importlib.util.spec_from_file_location("_portable_preserved_heads", path)
    module = importlib.util.module_from_spec(spec)
    # Execute verified bytes, avoiding a second read or stale bytecode cache.
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def make_guard(output, max_seconds, min_free_disk_gib):
    if (not math.isfinite(max_seconds) or not 1 <= max_seconds <= 604800
            or not math.isfinite(min_free_disk_gib) or min_free_disk_gib < 1):
        raise ValueError("Budget must be 1..604800 seconds; disk reserve at least 1 GiB")
    deadline = time.monotonic() + max_seconds
    def check(**fields):
        if time.monotonic() >= deadline:
            raise TimeoutError("Training wall-clock budget exhausted")
        probe = Path(output)
        while not probe.exists():
            probe = probe.parent
        if shutil.disk_usage(probe).free < min_free_disk_gib * 1024**3:
            raise OSError("Training stopped: free disk below requested reserve")
    return check


def execute(plan, *, execute_training=False, max_seconds=None, min_free_disk_gib=2):
    if execute_training is not True:
        raise ValueError("Explicit --execute-training required")
    if max_seconds is None:
        raise ValueError("Execution requires explicit --max-seconds")
    request = plan["request"]
    # Re-read metadata and stats immediately before execution; never trust edited plan JSON.
    current = build_plan(request["bank_dir"], request["output_dir"], request["bank_manifest_sha256"], request["epochs"])
    if current["request"] != request:
        raise ValueError("Training request differs from fixed bank contract")
    check = make_guard(request["output_dir"], max_seconds, min_free_disk_gib)
    check(phase="before-source-import")
    worker = load_worker()
    check(phase="before-output-create")
    Path(request["output_dir"]).mkdir(parents=True, exist_ok=False)
    run_request = dict(request, deadline_utc=(datetime.now(timezone.utc) + timedelta(seconds=max_seconds)).isoformat())
    return worker.run(run_request, check)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bank-manifest-sha256", required=True)
    parser.add_argument("--epochs", type=int, default=3, choices=[3])
    parser.add_argument("--plan-json", type=Path, help="Optional new JSON outside bank/source/training output")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--min-free-disk-gib", type=float, default=2)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args.bank_dir, args.output_dir, args.bank_manifest_sha256, args.epochs)
        if args.execute_training and args.plan_json:
            raise ValueError("--plan-json is for plan mode only")
        if args.execute_training:
            receipt = execute(plan, execute_training=True, max_seconds=args.max_seconds,
                              min_free_disk_gib=args.min_free_disk_gib)
            print(json.dumps({"status": receipt.get("status"), "output_dir": plan["request"]["output_dir"]}))
        else:
            if args.plan_json:
                target = fresh_output(args.plan_json, Path(plan["request"]["bank_dir"]))
                output = Path(plan["request"]["output_dir"])
                if target.is_relative_to(output) or output.is_relative_to(target):
                    raise ValueError("Plan JSON must be outside training output")
                with target.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(plan, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"train heads: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
