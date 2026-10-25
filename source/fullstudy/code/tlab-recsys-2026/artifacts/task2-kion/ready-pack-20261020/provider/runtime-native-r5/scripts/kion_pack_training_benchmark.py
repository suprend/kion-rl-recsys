"""Disposable 64x2/128x1 train benchmark; only a native supervisor may use CUDA.

Imports no Torch until the fresh GPU lease and pinned, completed external SFT
ledger have passed admission. Updates use the frozen pack runner unchanged.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
RUNNER_SOURCES = ("kion_pack_train.py", "kion_full_train.py", "kion_conditioned_train.py", "kion_train.py")
CELLS = {"micro64_acc2": (64, 2), "micro128_acc1": (128, 1)}
TERMINAL = {"completed", "failed", "stopped", "skipped_dependency"}


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    warmup_updates: int = 30
    measure_updates: int = 256
    repetitions: int = 3
    max_seconds: float = 900.
    max_training_seconds: float = 300.
    cpu_threads: int = 2
    fixture_seed: int = 2026
    nvidia_samples: bool = False

    def validate(self):
        integers = (self.warmup_updates, self.measure_updates, self.repetitions, self.cpu_threads, self.fixture_seed)
        if any(type(value) is not int for value in integers):
            raise Refused("Integer limits must be genuine integers")
        if (not 0 <= self.warmup_updates <= 30 or not 1 <= self.measure_updates <= 256
                or not 1 <= self.repetitions <= 3 or not 1 <= self.cpu_threads <= 16
                or not 0 <= self.fixture_seed < 2**32):
            raise Refused("Bounds: warmup<=30, measured<=256, repetitions<=3 per cell")
        if (not all(math.isfinite(x) for x in (self.max_seconds, self.max_training_seconds))
                or not 0 < self.max_training_seconds <= 300 or not 0 < self.max_seconds <= 900):
            raise Refused("Bounds: training<=300 seconds, total<=900 seconds")


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refused("Ambiguous duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(Refused("Nonfinite JSON: " + value)))


def read_json(path):
    return strict_json(Path(path).read_bytes())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def bound_json(path, expected):
    """Hash exactly the single byte string parsed, never two independent reads."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise Refused("Pinned JSON changed: " + str(path))
    return strict_json(raw)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


def native_helper():
    return load_module(HERE.with_name("kion_training_benchmark.py"), "pack_benchmark_native_primitives")


def request_manifest(args):
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    config.validate()
    paths = {name: getattr(args, name).resolve().as_posix() for name in
             ("runner_root", "checkpoint", "completion", "data", "feature_dir", "prior_ledger", "output")}
    sources = [HERE, HERE.with_name("kion_training_benchmark.py")]
    sources.extend(args.runner_root / "scripts" / name for name in RUNNER_SOURCES)
    inputs = [args.prior_ledger, args.completion, args.checkpoint, args.checkpoint.with_suffix(".json"),
              args.checkpoint.parent / "selected.json", args.checkpoint.parent / "config.json",
              args.data / "metadata.json", args.data / "train.npz", args.feature_dir / "manifest.json",
              args.feature_dir / "item_features.npy", args.feature_dir / "train_user_features.npy"]
    description = {"settings": asdict(config), "paths": paths, "cells": CELLS,
                   "run_order": [{"cell": cell, "repetition": repetition} for cell, repetition in run_order(config.repetitions)],
                   "max_logical_updates": 2 * config.repetitions * (config.warmup_updates + config.measure_updates),
                   "effective_batch": 128, "device": "cuda:0", "purpose": "disposable-throughput-only"}
    return {"config": description, "config_sha256": canonical_hash(description),
            "source_paths": sorted({p.resolve().as_posix() for p in sources}),
            "input_paths": sorted({p.resolve().as_posix() for p in inputs})}


def _safe_name(value):
    return (isinstance(value, str) and bool(value)
            and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in value))


def _strong_identity(value):
    return (isinstance(value, dict) and set(value) == {"pid", "exe", "creation_time_100ns"}
            and type(value["pid"]) is int and value["pid"] > 0
            and type(value["creation_time_100ns"]) is int and value["creation_time_100ns"] > 0
            and isinstance(value["exe"], str) and Path(value["exe"]).is_absolute())


def _unique_attempts(ledger):
    attempts = ledger["attempts"]
    if not isinstance(attempts, list) or any(not isinstance(item, dict) or not _safe_name(item.get("id")) for item in attempts):
        raise Refused("Malformed ledger attempts")
    result = {item["id"]: item for item in attempts}
    if len(result) != len(attempts):
        raise Refused("Ambiguous duplicate attempt identity")
    return result


def _closed(attempts, native, *, except_id=None):
    for attempt in attempts.values():
        if attempt["id"] == except_id:
            continue
        if attempt.get("state") not in TERMINAL:
            raise Refused("Another attempt is nonterminal")
        if attempt["state"] == "skipped_dependency":
            if attempt.get("identity") or attempt.get("job_name"):
                raise Refused("Skipped attempt unexpectedly owns a worker")
            continue
        if (attempt.get("cleanup") != "quiescent" or not isinstance(attempt.get("job_name"), str)
                or not attempt["job_name"] or not _strong_identity(attempt.get("identity"))):
            raise Refused("Another attempt has unknown cleanup/identity")
        active = native.job(attempt["job_name"])
        if type(active) is not int or active < 0:
            raise Refused("Ambiguous native Job probe")
        if active:
            raise Refused("Prior native Job is still active")
        identity = attempt["identity"]
        observed = native.identity(identity["pid"])
        if observed is not None and not _strong_identity(observed):
            raise Refused("Ambiguous native identity probe")
        if observed == identity:
            raise Refused("Prior worker still has its original identity")


def _producer_binding(args, authorization, producer):
    """Physical selected bytes must be the completed producer's committed output."""
    pins = authorization["input_hashes"]
    output = Path(producer["output"]).resolve()
    completion_path = args.completion.resolve()
    try:
        relative = completion_path.relative_to(output).as_posix()
    except ValueError as error:
        raise Refused("SFT completion lies outside producer output") from error
    if producer["artifact_hashes"].get(relative) != pins[completion_path.as_posix()]:
        raise Refused("Producer did not commit this completion")
    completion = bound_json(completion_path, pins[completion_path.as_posix()])
    if producer.get("receipt") != completion or completion.get("state") != "completed":
        raise Refused("Producer receipt/completion mismatch")
    epoch = completion.get("selected_epoch")
    if type(epoch) is not int or epoch < 1 or args.checkpoint.name != f"selected-epoch{epoch:04d}.pt":
        raise Refused("Checkpoint is not the completed validation selection")
    if args.checkpoint.parent.resolve() != completion_path.parent:
        raise Refused("Checkpoint and completed selection must share the model output")
    for path in (args.checkpoint, args.checkpoint.with_suffix(".json"),
                 args.checkpoint.parent / "selected.json", args.checkpoint.parent / "config.json"):
        actual = pins[path.resolve().as_posix()]
        if completion["sha256"].get(path.name) != actual or path.name not in completion["artifacts"]:
            raise Refused("Selected input is not committed by SFT completion: " + path.name)
        declared = producer["artifact_hashes"].get(path.resolve().relative_to(output).as_posix())
        if declared is not None and declared != actual:
            raise Refused("Producer artifact hash mismatch: " + path.name)
    selected = bound_json(args.checkpoint.parent / "selected.json", pins[(args.checkpoint.parent / "selected.json").resolve().as_posix()])
    receipt = bound_json(args.checkpoint.with_suffix(".json"), pins[args.checkpoint.with_suffix(".json").resolve().as_posix()])
    checkpoint_hash = pins[args.checkpoint.resolve().as_posix()]
    if (selected.get("file") != args.checkpoint.name or selected.get("epoch") != epoch
            or selected.get("sha256") != checkpoint_hash or receipt.get("sha256") != checkpoint_hash):
        raise Refused("Selected checkpoint receipt mismatch")
    for path in (args.data / "metadata.json", args.data / "train.npz", args.feature_dir / "manifest.json",
                 args.feature_dir / "item_features.npy", args.feature_dir / "train_user_features.npy"):
        if producer["input_hashes"].get(path.resolve().as_posix()) != pins[path.resolve().as_posix()]:
            raise Refused("Benchmark training inputs differ from completed producer")
    producer_cwd = Path(producer["cwd"]).resolve()
    if producer_cwd != args.runner_root.resolve():
        raise Refused("Runner must be the producer's physical frozen working directory")
    for name in RUNNER_SOURCES:
        path = (producer_cwd / "scripts" / name).resolve().as_posix()
        source_pins = [producer["source_hashes"].get("scripts/" + name),
                       producer["source_hashes"].get(path), producer["input_hashes"].get(path)]
        source_pins = [value for value in source_pins if value is not None]
        if not source_pins or any(value != authorization["source_hashes"][path] for value in source_pins):
            raise Refused("Benchmark runner differs from producer: " + name)


def authorize(args, request, *, native=None, lock_root=None, now=None):
    """Read-only admission across two namespaces; never merges or writes ledgers."""
    if args.authorization is None:
        raise Refused("Explicit native benchmark authorization required")
    try:
        authorization = read_json(args.authorization)
        now = now or datetime.now(timezone.utc)
        expiry = datetime.fromisoformat(authorization["expires_utc"].replace("Z", "+00:00"))
        if (authorization.get("schema") != 1 or authorization.get("purpose") != "kion-pack-training-benchmark"
                or authorization.get("config_sha256") != request["config_sha256"]
                or expiry.tzinfo is None or expiry <= now
                or not all(_safe_name(authorization.get(key)) for key in ("night_id", "job_id"))):
            raise Refused("Stale, malformed or mismatched authorization")
        for key, names in (("source_hashes", request["source_paths"]), ("input_hashes", request["input_paths"])):
            values = authorization.get(key, {})
            if set(values) != set(names) or any(not isinstance(v, str) or len(v) != 64
                    or any(c not in "0123456789abcdef" for c in v) for v in values.values()):
                raise Refused("Authorization needs exact source/input hash closure")
        helper = native_helper()
        root = Path(lock_root) if lock_root is not None else helper.fixed_lock_root()
        native = native or helper.NativeOwnership()
        gpu = authorization["gpu_uuid"]
        if not isinstance(gpu, str) or not gpu.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise Refused("Exactly the authorized GPU UUID must be exposed")
        prior = authorization["external_prior"]
        if (not isinstance(prior, dict) or not all(_safe_name(prior.get(key)) for key in ("night_id", "job_id"))
                or prior["night_id"] == authorization["night_id"]
                or any(not isinstance(prior.get(key), str) or len(prior[key]) != 64
                       or any(c not in "0123456789abcdef" for c in prior[key]) for key in ("ledger_sha256", "job_sha256"))):
            raise Refused("One distinct external prior namespace is required")
        expected_prior = (root / ("night-" + prior["night_id"] + ".ledger.json")).resolve()
        if Path(prior["ledger_path"]).resolve() != expected_prior or args.prior_ledger.resolve() != expected_prior:
            raise Refused("External ledger must use its fixed native OS path")
        if authorization["input_hashes"][expected_prior.as_posix()] != prior["ledger_sha256"]:
            raise Refused("External ledger SHA binding differs")
        old = bound_json(expected_prior, prior["ledger_sha256"])
        if old.get("night_id") != prior["night_id"] or old.get("budget", {}).get("gpu_uuid") != gpu:
            raise Refused("External ledger namespace/GPU mismatch")
        old_attempts = _unique_attempts(old)
        producer = old_attempts.get(prior["job_id"], {})
        if (producer.get("state") != "completed" or producer.get("cleanup") != "quiescent"
                or producer.get("job_sha256") != prior["job_sha256"]):
            raise Refused("Required producer is not completed/quiescent and pinned")
        key = hashlib.sha256(gpu.encode("utf-8")).hexdigest()
        marker = read_json(root / ("gpu-" + key + ".owner.json"))
        own_ledger = read_json(root / ("night-" + authorization["night_id"] + ".ledger.json"))
        own_attempts = _unique_attempts(own_ledger)
        own = own_attempts.get(authorization["job_id"], {})
        if (marker.get("night_id") != authorization["night_id"] or marker.get("job_id") != authorization["job_id"]
                or own_ledger.get("night_id") != authorization["night_id"] or own_ledger.get("budget", {}).get("gpu_uuid") != gpu
                or own.get("state") not in ("starting", "running") or not _strong_identity(own.get("identity"))
                or type(own.get("supervisor_pid")) is not int or own["supervisor_pid"] < 1
                or not own.get("job_name") or own["job_name"] != marker.get("job_name")):
            raise Refused("Fresh native ledger/owner does not admit this worker")
        ancestors = native.ancestors({own["identity"]["pid"], own["supervisor_pid"]})
        if (not isinstance(ancestors, dict) or ancestors.get(own["identity"]["pid"]) != own["identity"]
                or not _strong_identity(ancestors.get(own["supervisor_pid"]))
                or ancestors[own["supervisor_pid"]]["creation_time_100ns"] > own["identity"]["creation_time_100ns"]
                or own["supervisor_pid"] == own["identity"]["pid"]):
            raise Refused("Reused or foreign worker/supervisor identity")
        own_active = native.job(own["job_name"], own=True)
        if type(own_active) is not int or own_active < 1:
            raise Refused("Ambiguous or inactive owned Job")
        if native.lease_is_occupied(root / ("gpu-" + key + ".lock")) is not True:
            raise Refused("Fresh supervisor does not hold the GPU lease")
        _closed(own_attempts, native, except_id=authorization["job_id"])
        _closed(old_attempts, native)
        _producer_binding(args, authorization, producer)
        # JSON was parsed from its hashed bytes; check it stayed pinned during probes.
        if sha256(expected_prior) != prior["ledger_sha256"]:
            raise Refused("External ledger changed during native admission")
        return authorization
    except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
        raise Refused("Cannot establish benchmark admission: " + str(error)) from error


def verify_hashes(authorization):
    for key in ("source_hashes", "input_hashes"):
        for path, expected in authorization[key].items():
            if sha256(path) != expected:
                raise Refused(key + " changed: " + path)


def validate_output(args):
    output = args.output.resolve()
    protected = [args.data, args.feature_dir, args.runner_root, args.checkpoint.parent, args.prior_ledger.parent]
    if output.name.casefold() == "completion.json" or any(output.is_relative_to(p.resolve()) for p in protected):
        raise Refused("Disposable output must be outside source/data/model/ledger trees")
    if output.exists() or output.with_name("completion.json").exists():
        raise Refused("Benchmark report and completion must be fresh")


def load_runner(directory):
    saved = {key: value for key, value in sys.modules.items() if key.startswith("kion_")}
    old_path = sys.path[:]
    for key in saved:
        del sys.modules[key]
    try:
        sys.path.insert(0, str(directory))
        return load_module(directory / "kion_pack_train.py", "pack_benchmark_frozen_runner")
    finally:
        for key in list(sys.modules):
            if key.startswith("kion_"):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path


def execution_origins(module, directory, source_pins):
    paths = {"kion_pack_train.py": module.__file__, "kion_full_train.py": module.legacy.__file__,
             "kion_conditioned_train.py": module.shared.__file__,
             "kion_train.py": module.IDGPT.forward.__code__.co_filename}
    code_paths = {"pack_update": module.update.__code__.co_filename,
                  "packed_transfer": module.PackedBatcher.batch.__code__.co_filename,
                  "base_model_init": module.IDGPT.__init__.__code__.co_filename,
                  "train_loader": module.shared.load_data.__code__.co_filename,
                  "epoch_order": module.legacy.epoch_order.__code__.co_filename}
    expected_functions = {"pack_update": "kion_pack_train.py", "packed_transfer": "kion_pack_train.py",
                          "base_model_init": "kion_train.py", "train_loader": "kion_conditioned_train.py",
                          "epoch_order": "kion_full_train.py"}
    result = {}
    for name, path in paths.items():
        expected = (directory / name).resolve()
        if Path(path).resolve() != expected or sha256(expected) != source_pins[expected.as_posix()]:
            raise Refused("Executed module origin/hash mismatch: " + name)
        result[name] = {"path": expected.as_posix(), "sha256": source_pins[expected.as_posix()]}
    for function, path in code_paths.items():
        if Path(path).resolve() != (directory / expected_functions[function]).resolve():
            raise Refused("Executed function origin mismatch: " + function)
    result["function_code_filenames"] = {name: Path(path).resolve().as_posix() for name, path in code_paths.items()}
    return result


def load_train(module, args, pins):
    """Use frozen validation primitives on train only; never open val/test arrays."""
    manifest = bound_json(args.feature_dir / "manifest.json", pins[(args.feature_dir / "manifest.json").resolve().as_posix()])
    metadata = bound_json(args.data / "metadata.json", pins[(args.data / "metadata.json").resolve().as_posix()])
    for name in ("metadata.json", "train.npz"):
        if manifest["inputs"][name]["sha256"] != pins[(args.data / name).resolve().as_posix()]:
            raise Refused("Feature source manifest mismatch")
    data = module.FeatureData(module.shared.load_data(args.data / "train.npz", int(metadata["vocab_size"]),
                              int(metadata["context"]), require_keys=True, unique_keys=False))
    arrays = {}
    try:
        for name in ("item_features.npy", "train_user_features.npy"):
            declaration = manifest["outputs"][name]
            if declaration["sha256"] != pins[(args.feature_dir / name).resolve().as_posix()]:
                raise Refused("Feature cache SHA mismatch")
            values = np.load(args.feature_dir / name, mmap_mode="r", allow_pickle=False)
            arrays[name] = values
            if (values.ndim != 2 or values.dtype.kind != "f" or list(values.shape) != declaration["shape"]
                    or str(values.dtype) != declaration["dtype"]):
                raise Refused("Invalid feature matrix dtype/shape")
            module._validate_finite(values, name)
        items, users = arrays["item_features.npy"], arrays["train_user_features.npy"]
        if (len(items) != metadata["vocab_size"] or np.any(items[:module.FIRST_ITEM] != 0)
                or len(users) != len(data["y"])):
            raise Refused("Feature rows do not align")
        module._columns(manifest["feature_columns"], items.shape[1], users.shape[1])
        if (manifest.get("rowkeys_algorithm") != module.ROWKEYS_ALGORITHM
                or module.canonical_hash(manifest["feature_columns"]) != manifest["schema_hash"]
                or manifest["row_keys_sha256"]["train"] != module.make_row_keys_hash(data)):
            raise Refused("Feature row/schema binding mismatch")
        data["user_features"] = users
        for value in data.values():
            value.setflags(write=False)
        return metadata, manifest, data, items
    except BaseException:
        module.close_feature_data(*arrays.values())
        raise


def fixture_plan(module, data, vocab, training, config):
    """Select complete logical groups across four workload strata, without outcomes."""
    eligible = np.flatnonzero((data["y"] >= module.FIRST_ITEM) & (data["y"] < vocab))
    # Use the candidate's natural length grouping before batches are shuffled.
    # Pairing shuffled64 batches would create artificial short/long128 mixtures.
    order = module.legacy.epoch_order(eligible, data["lengths"], training["seed"], 0, 128, training["bucket_size"])
    rows = order[:len(order) // 128 * 128].reshape(-1, 128)
    lengths = data["lengths"][rows]
    scores = (lengths[:, :64].max(1) + lengths[:, 64:].max(1)) / 2
    strata = np.array_split(np.argsort(scores, kind="stable"), 4)
    count = config.warmup_updates + config.measure_updates
    rng = np.random.default_rng(config.fixture_seed)
    picks = []
    for stratum, candidates in enumerate(strata):
        required = len(range(stratum, count, 4))
        if len(candidates) < required:
            raise Refused("Train split too small for nonrepeating representative fixture")
        picks.append(rng.choice(candidates, size=required, replace=False))
    chosen = np.array([picks[i % 4][i // 4] for i in range(count)], dtype=np.int64)
    selected = rows[chosen].copy()
    declaration = {"method": "epoch0 natural bucketed128 groups; baseline splits64; four equal workload strata cycled",
                   "outcomes_used_for_selection": False, "epoch": 0, "eligible_rows": len(eligible),
                   "selected_logical_positions": chosen.tolist(), "strata": [i % 4 for i in range(count)],
                   "row_indices": selected.tolist(), "shape": list(selected.shape),
                   "sha256": hashlib.sha256(selected.astype("<i8", copy=False).tobytes()).hexdigest()}
    return selected, declaration


def microbatches(rows, cell):
    width, accumulation = CELLS[cell]
    if len(rows) != width * accumulation:
        raise Refused("Incomplete logical fixture group")
    return [rows[start:start + width] for start in range(0, len(rows), width)]


def batch_statistics(rows, lengths, cell):
    widths, actual, padded = [], 0, 0
    for logical in rows:
        for indices in microbatches(logical, cell):
            values = lengths[indices]
            width = int(values.max())
            widths.append(width)
            actual += int(values.sum())
            padded += len(indices) * width
    values, counts = np.unique(widths, return_counts=True)
    return {"examples": int(rows.size), "real_tokens": actual, "padded_tokens": padded,
            "padding_fraction": 1 - actual / padded, "mean_microbatch_width": float(np.mean(widths)),
            "width_histogram": {str(int(k)): int(v) for k, v in zip(values, counts)}}


def state_hash(value):
    """Exact fingerprint outside timed windows; supports tensors, optimizer and RNG."""
    digest = hashlib.sha256()
    def visit(item):
        if hasattr(item, "detach"):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(str((item.dtype.str, item.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode("utf-8")); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode("ascii"))
            for entry in item:
                visit(entry)
        else:
            digest.update(repr(item).encode("utf-8"))
        digest.update(b"\0")
    visit(value)
    return digest.hexdigest()


def reset_cell(torch, module, saved, items, device):
    model = module.NextItemGPT.from_config(saved["config"], items).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    training = saved["config"]["contract"]["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    scaler.load_state_dict(copy.deepcopy(saved["scaler"]))
    module.legacy.restore_rng(copy.deepcopy(saved["rng"]))
    model.train()
    reset_hash = state_hash({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "scaler": scaler.state_dict(), "rng": capture_rng(torch, device)})
    expected = state_hash({key: saved[key] for key in ("state_dict", "optimizer", "scaler", "rng")})
    if reset_hash != expected:
        raise Refused("Cell did not restore exact selected weights/optimizer/scaler/RNG")
    return model, optimizer, scaler, reset_hash


def capture_rng(torch, device):
    # CPU contracts never query CUDA availability or create a CUDA context.
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state(device)] if device.type == "cuda" else None}


def run_order(repetitions):
    return [(cell, repetition) for repetition in range(repetitions)
            for cell in (("micro64_acc2", "micro128_acc1") if repetition % 2 == 0
                         else ("micro128_acc1", "micro64_acc2"))]


def finite_state(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in "fc" or bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_state(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def aggregate_runs(runs, config, initial_hash, fixture_hash):
    expected = set(run_order(config.repetitions))
    if len(runs) != len(expected) or {(r["cell"], r["repetition"]) for r in runs} != expected:
        raise Refused("Missing, duplicate or unplanned benchmark windows")
    for run in runs:
        if (run["reset_state_sha256"] != initial_hash or run["fixture_sha256"] != fixture_hash
                or run.get("finite_final_state") is not True or run["successful_updates"] != config.warmup_updates + config.measure_updates
                or not all(math.isfinite(run[k]) and run[k] > 0 for k in ("measurement_seconds", "examples_per_second"))):
            raise Refused("Invalid, nonfinite or unreset benchmark window")
    summary = {cell: {"median_examples_per_second": statistics.median(r["examples_per_second"] for r in runs if r["cell"] == cell),
                      "raw_measurement_seconds": [r["measurement_seconds"] for r in runs if r["cell"] == cell],
                      "repeat_final_state_hashes_match": len({r["final_disposable_state_sha256"] for r in runs if r["cell"] == cell}) == 1}
               for cell in CELLS}
    by_run = {(r["cell"], r["repetition"]): r for r in runs}
    paired = [by_run["micro128_acc1", n]["examples_per_second"] / by_run["micro64_acc2", n]["examples_per_second"]
              for n in range(config.repetitions)]
    primary = statistics.median(paired)
    sufficient = config.repetitions == 3 and primary >= 1.10
    return {"cells": summary, "paired_throughput_ratios_128_over_64": paired,
            "primary_median_paired_throughput_ratio": primary, "minimum_ratio_for_recommendation": 1.10,
            "ratio_of_cell_median_throughputs": summary["micro128_acc1"]["median_examples_per_second"] / summary["micro64_acc2"]["median_examples_per_second"],
            "all_windows_valid_finite": True, "oom_observed": False,
            "six_window_protocol_completed": config.repetitions == 3,
            "performance_sufficient_for_future_micro128": sufficient,
            "interpretation": "future throughput candidate" if sufficient else "bounded negative or insufficient performance evidence; no extra windows/configs",
            "quality_equivalence_claimed": False}


def execute(args, request, authorization, started):
    import torch
    import transformers
    from contextlib import ExitStack
    config = Config(**request["config"]["settings"])
    training_seconds, segment_started = 0., None
    stop_file = Path(os.environ["KION_STOP_FILE"]) if os.environ.get("KION_STOP_FILE") else None
    def bounded():
        active = time.perf_counter() - segment_started if segment_started is not None else 0.
        if (time.perf_counter() - started >= config.max_seconds
                or training_seconds + active >= config.max_training_seconds
                or (stop_file is not None and stop_file.exists())):
            raise Refused("Benchmark time/stop bound reached; no fallback or partial success")
    bounded()
    module = load_runner(args.runner_root / "scripts")
    origins = execution_origins(module, args.runner_root / "scripts", authorization["source_hashes"])
    metadata, manifest, data, items = load_train(module, args, authorization["input_hashes"])
    with ExitStack() as resources:
        resources.callback(module.close_feature_data, data, items)
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        physical_config = bound_json(args.checkpoint.parent / "config.json",
                       authorization["input_hashes"][(args.checkpoint.parent / "config.json").resolve().as_posix()])
        if saved["config"] != physical_config:
            raise Refused("Physical checkpoint config mismatch")
        contract = saved["config"]["contract"]
        architecture, training = contract["architecture"], contract["training"]
        if (tuple(architecture[k] for k in ("hidden_size", "layers", "heads", "context_length")) != (256, 4, 4, 64)
                or architecture["feature_mode"] != "local-pack" or training["batch_size"] != 64
                or training["accumulation"] != 2 or training["batch_transfer"] != "pinned-packed"
                or architecture["schema_hash"] != manifest["schema_hash"]):
            raise Refused("Benchmark only admits the completed4x256 local-pack64x2 model")
        for name, expected in contract["sources_sha256"].items():
            if authorization["source_hashes"].get((args.runner_root / "scripts" / name).resolve().as_posix()) != expected:
                raise Refused("Selected checkpoint runner source mismatch")
        for name in ("metadata.json", "train.npz"):
            if contract["inputs_sha256"][name] != authorization["input_hashes"][(args.data / name).resolve().as_posix()]:
                raise Refused("Selected checkpoint prepared inputs mismatch")
        if (contract["feature_manifest_sha256"] != authorization["input_hashes"][(args.feature_dir / "manifest.json").resolve().as_posix()]
                or architecture["vocab_size"] != metadata["vocab_size"]):
            raise Refused("Selected checkpoint feature manifest mismatch")
        rows, fixture = fixture_plan(module, data, metadata["vocab_size"], training, config)
        repeated, repeated_fixture = fixture_plan(module, data, metadata["vocab_size"], training, config)
        if fixture != repeated_fixture or not np.array_equal(rows, repeated):
            raise Refused("Representative row fixture is not reproducible")
        del repeated
        bounded()
        # Slow CPU validation precedes the final admission, then the first CUDA call.
        authorize(args, request)
        torch.set_num_threads(config.cpu_threads)
        device = torch.device("cuda:0")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise Refused("Exactly one supervisor-selected CUDA device is required")
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        # Pack runner does not opt into matmul TF32. Pin/record the fresh default.
        torch.backends.cuda.matmul.allow_tf32 = False
        initial_hash = state_hash({key: saved[key] for key in ("state_dict", "optimizer", "scaler", "rng")})
        setup_seconds = time.perf_counter() - started
        runs = []
        parameter_count = None
        total_schedule = math.ceil(fixture["eligible_rows"] / 128) * training["schedule_total_epochs"]
        learning_rates = [module.shared.learning_rate(saved["state"]["updates"] + index,
                          total_schedule, training["warmup_ratio"], training["lr"]) for index in range(len(rows))]
        if any(not math.isfinite(value) or value < 0 for value in learning_rates):
            raise Refused("Disposable learning-rate sequence must be finite/nonnegative")
        for cell, repetition in run_order(config.repetitions):
            bounded()
            reset_started = time.perf_counter()
            model, optimizer, scaler, reset_hash = reset_cell(torch, module, saved, items, device)
            parameter_count = sum(p.numel() for p in model.parameters())
            if any(p.dtype != torch.float32 for p in model.parameters()):
                raise Refused("Selected model parameters must remain FP32")
            micro, _ = CELLS[cell]
            with module.PackedBatcher(data, device, micro, True) as batcher:
                data.batcher = batcher
                try:
                    torch.cuda.synchronize(device)
                    reset_seconds = time.perf_counter() - reset_started
                    retries = {"warmup": 0, "measured": 0}
                    logged_losses = {}
                    def update(index, phase):
                        bounded()
                        for group in optimizer.param_groups:
                            group["lr"] = learning_rates[index]
                        collect = index == 0 or (index + 1) % 100 == 0
                        loss, overflow = module.update(model, data, microbatches(rows[index], cell),
                                                      optimizer, scaler, device, collect_loss=collect)
                        retries[phase] += overflow
                        if collect:
                            if loss is None or not math.isfinite(loss):
                                raise Refused("Frozen update returned nonfinite logged loss")
                            logged_losses[str(index)] = loss
                    segment_started = time.perf_counter()
                    for index in range(config.warmup_updates):
                        update(index, "warmup")
                    torch.cuda.synchronize(device)
                    warmup_seconds = time.perf_counter() - segment_started
                    training_seconds += warmup_seconds
                    segment_started = None
                    torch.cuda.reset_peak_memory_stats(device)
                    before = native_helper().gpu_activity() if config.nvidia_samples else None
                    torch.cuda.synchronize(device)
                    segment_started = time.perf_counter()
                    for index in range(config.warmup_updates, len(rows)):
                        update(index, "measured")
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - segment_started
                    training_seconds += elapsed
                    segment_started = None
                    stats = batch_statistics(rows[config.warmup_updates:], data["lengths"], cell)
                    run = {"cell": cell, "repetition": repetition, "microbatch": micro,
                           "accumulation": CELLS[cell][1], "effective_batch": 128,
                           "reset_seconds": reset_seconds, "warmup_seconds": warmup_seconds,
                           "measurement_seconds": elapsed, "measured_updates": config.measure_updates,
                           "examples_per_second": stats["examples"] / elapsed,
                           "real_tokens_per_second": stats["real_tokens"] / elapsed,
                           "padded_tokens_per_second": stats["padded_tokens"] / elapsed,
                           "batch_statistics": stats, "fixture_sha256": fixture["sha256"],
                           "reset_state_sha256": reset_hash, "overflow_retries": retries,
                           "logged_losses": logged_losses,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                           "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                           "pinned_staging_bytes": batcher.capacity * len(batcher.slots),
                           "gpu_before": before, "gpu_after": native_helper().gpu_activity() if config.nvidia_samples else None}
                    final_state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scaler": scaler.state_dict(), "rng": capture_rng(torch, device)}
                    run["finite_final_state"] = finite_state(final_state)
                    run["successful_updates"] = len(rows)
                    run["final_disposable_state_sha256"] = state_hash(final_state)
                    del final_state
                    runs.append(run)
                    print(json.dumps({"event": "benchmark_cell", "cell": cell, "repetition": repetition,
                                      "examples_per_second": run["examples_per_second"]}), flush=True)
                finally:
                    data.batcher = None
            del model, optimizer, scaler
            torch.cuda.empty_cache()
        bounded()
        verify_hashes(authorization)
        authorize(args, request)
        bounded()
        summary = aggregate_runs(runs, config, initial_hash, fixture["sha256"])
        return {"schema": 1, "state": "completed", "purpose": "disposable-throughput-only", "config": request["config"],
                "config_sha256": request["config_sha256"], "source_hashes": authorization["source_hashes"],
                "input_hashes": authorization["input_hashes"], "authorization_sha256": sha256(args.authorization),
                "external_prior": authorization["external_prior"], "initial_state_sha256": initial_hash,
                "executed_source_origins": origins,
                "learning_rate": {"saved_update_cursor": saved["state"]["updates"], "nominal_schedule_updates": total_schedule,
                                  "saved_optimizer_group_rates": [g["lr"] for g in saved["optimizer"]["param_groups"]],
                                  "exact_per_update_sequence": learning_rates,
                                  "interpretation": "frozen cosine evaluated at saved cursor plus disposable index, including after nominal final cursor; no quality claim"},
                "fixture": fixture, "fixture_recomputed_identically": True, "runs": runs, "summary": summary,
                "fractional_throughput_gain_128_over_64": summary["primary_median_paired_throughput_ratio"] - 1,
                "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "gpu_uuid": authorization["gpu_uuid"],
                            "cpu_threads": torch.get_num_threads(), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_deterministic": torch.backends.cudnn.deterministic,
                            "parameters": parameter_count,
                            "weights": "float32", "autocast": "float16", "transfer": "frozen pinned-packed; no fallback"},
                "timing": {"initialization_seconds": setup_seconds, "training_seconds_including_warmup": training_seconds,
                           "measured_seconds": sum(r["measurement_seconds"] for r in runs),
                           "total_seconds": time.perf_counter() - started},
                "finite_guards": "unmodified frozen update finite loss/gradient and bounded AMP retry before optimizer commit",
                "train_only": True, "validation_accessed": False, "test_accessed": False,
                "learned_checkpoint_written": False, "quality_selection_performed": False,
                "interpretation": "same rows/effective batch and exact resets; dropout draws, padding and floating arithmetic differ across cells; no quality/parity claim"}


def save_result(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    receipt = {"schema": 1, "state": "completed", "train_only": True, "validation_accessed": False,
               "test_accessed": False, "learned_checkpoint_written": False,
               "all_windows_valid_finite": result["summary"]["all_windows_valid_finite"],
               "performance_sufficient_for_future_micro128": result["summary"]["performance_sufficient_for_future_micro128"],
               "logical_updates": sum(r["measured_updates"] + result["config"]["settings"]["warmup_updates"] for r in result["runs"]),
               "artifacts": [output.name], "sha256": {output.name: sha256(output)}}
    with output.with_name("completion.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "checkpoint", "completion", "data", "feature-dir", "prior-ledger", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--authorization", type=Path)
    result.add_argument("--describe-request", action="store_true")
    for name, field in Config.__dataclass_fields__.items():
        result.add_argument("--" + name.replace("_", "-"), default=field.default,
                            **({"action": "store_true"} if type(field.default) is bool else {"type": type(field.default)}))
    return result


def main(argv=None):
    started = time.perf_counter()
    args = parser().parse_args(argv)
    request = request_manifest(args)
    if args.describe_request:
        print(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    validate_output(args)
    authorization = authorize(args, request)
    verify_hashes(authorization)
    result = execute(args, request, authorization, started)
    save_result(result, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "reason": str(error),
                          "fallback_attempted": False, "learned_checkpoint_written": False}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
