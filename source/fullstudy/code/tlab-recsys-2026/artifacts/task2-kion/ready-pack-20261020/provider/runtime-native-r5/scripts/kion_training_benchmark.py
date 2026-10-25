"""Bounded train-only A/B; CUDA requires the existing supervisor's live lease.

This file deliberately imports neither Torch nor the trainers at module import.
An authorization is not a launch mechanism: the authoritative supervisor ledger,
native process/Job ownership and the occupied fixed GPU lease must also agree.
The supervisor supplies the hard deadline and kill-on-close descendant cleanup.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
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
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("kion_conditioned_train.py", "kion_full_train.py", "kion_train.py",
           "kion_prepare.py", "kion_basic_bundle_features.py", "kion_local_features.py",
           "kion_sex_features.py")


class Refused(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    warmup_updates: int = 20
    measure_updates: int = 200
    repetitions: int = 4
    max_updates: int = 1200
    max_seconds: float = 300
    batch_size: int = 64
    accumulation: int = 2
    hidden_size: int = 256
    layers: int = 4
    heads: int = 4
    seed: int = 2026
    bucket_size: int = 8192
    cpu_threads: int = 4
    lr: float = 1e-4
    weight_decay: float = .01
    schedule_total_epochs: int = 5
    warmup_ratio: float = .05
    log_updates: int = 100
    batch_transfer: str = "direct"
    minimum_improvement: float = .10
    require_improvement: bool = False
    rtol: float = 1e-4
    atol: float = 1e-6

    def validate(self):
        integers = (self.measure_updates, self.max_updates, self.batch_size,
                    self.accumulation, self.hidden_size, self.layers, self.heads,
                    self.cpu_threads, self.schedule_total_epochs, self.log_updates)
        if min(integers) < 1 or self.warmup_updates < 0 or self.bucket_size < 0:
            raise Refused("Invalid positive limits/model/batch settings")
        if self.repetitions not in (4, 8) or self.hidden_size % self.heads:
            raise Refused("Use four/eight ABBA runs and divisible model dimensions")
        if not 0 <= self.seed < 2**32:
            raise Refused("Seed must fit NumPy's uint32 seed domain")
        if not 0 < self.max_seconds <= 600 or not 1 <= self.max_updates <= 2000:
            raise Refused("Benchmark bounds are 600 seconds / 2000 optimizer updates")
        # One extra current/legacy run proves frozen -> current legacy parity.
        if (self.warmup_updates + self.measure_updates) * (self.repetitions + 1) > self.max_updates:
            raise Refused("Requested warmup + measurement exceeds total max_updates")
        numbers = (self.lr, self.weight_decay, self.warmup_ratio, self.minimum_improvement,
                   self.rtol, self.atol, self.max_seconds)
        if not all(math.isfinite(value) for value in numbers):
            raise Refused("All numerical settings must be finite")
        if (self.lr <= 0 or self.weight_decay < 0 or not 0 <= self.warmup_ratio < 1
                or not 0 <= self.minimum_improvement < 1 or min(self.rtol, self.atol) < 0):
            raise Refused("Invalid optimization or comparison settings")
        if self.batch_transfer not in ("direct", "packed", "pinned-packed"):
            raise Refused("Unknown batch transfer mode")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.asarray(value)
    digest = hashlib.sha256(json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                                      sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_order(repetitions):
    if repetitions not in (4, 8):
        raise Refused("ABBA requires four or eight runs")
    return ["legacy", "optimized", "optimized", "legacy"] * (repetitions // 4)


def request_manifest(args):
    """Cheap request description for the coordinator; does not hash/read data."""
    config = Config(**{name: getattr(args, name) for name in Config.__dataclass_fields__})
    config.validate()
    paths = {name: str(getattr(args, name).resolve().as_posix()) for name in
             ("data", "bundle_features", "feature_manifest", "baseline_root", "output")}
    paths["current_root"] = ROOT.as_posix()
    paths["profile_trace"] = args.profile_trace.resolve().as_posix() if args.profile_trace else None
    description = {"settings": asdict(config), "paths": paths, "device": "cuda:0",
                   "mode": "profiler_only" if args.profile_trace else "speed"}
    sources = [Path(__file__).resolve()]
    for directory in (args.baseline_root / "scripts", ROOT / "scripts"):
        sources.extend(directory / name for name in SOURCES)
    sources.extend((ROOT / "scripts/kion_cuda_batches.py", ROOT / "scripts/kion_night_supervisor.py"))
    inputs = [args.data / "metadata.json", args.data / "train.npz",
              args.bundle_features, args.feature_manifest]
    return {"config": description, "config_sha256": canonical_hash(description),
            "source_paths": sorted({path.resolve().as_posix() for path in sources}),
            "input_paths": sorted({path.resolve().as_posix() for path in inputs})}


def fixed_lock_root():
    if os.name != "nt" or not os.environ.get("LOCALAPPDATA"):
        raise Refused("CUDA admission requires the native Windows supervisor")
    return Path(os.environ["LOCALAPPDATA"]) / "KionNightSupervisor/locks"


class NativeOwnership:
    """Read-only Windows probes; unknown/inaccessible state always refuses."""
    def __init__(self):
        import ctypes
        from ctypes import wintypes as W
        self.ctypes, self.W = ctypes, W
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    def api(self, name, result, *arguments):
        function = getattr(self.kernel, name)
        function.restype, function.argtypes = result, arguments
        return function

    def identity(self, pid):
        c, w = self.ctypes, self.W
        handle = self.api("OpenProcess", w.HANDLE, w.DWORD, w.BOOL, w.DWORD)(0x1000, False, pid)
        if not handle:
            if c.get_last_error() == 87:  # ERROR_INVALID_PARAMETER: no such PID.
                return None
            raise Refused(f"Cannot establish native identity for PID {pid}")
        try:
            values = [c.c_uint64() for _ in range(4)]
            size, buffer = w.DWORD(32768), c.create_unicode_buffer(32768)
            times = self.api("GetProcessTimes", w.BOOL, w.HANDLE, *([w.LPVOID] * 4))
            image = self.api("QueryFullProcessImageNameW", w.BOOL, w.HANDLE, w.DWORD,
                             w.LPWSTR, c.POINTER(w.DWORD))
            if not times(handle, *[c.byref(value) for value in values]) or not image(handle, 0, buffer, c.byref(size)):
                raise Refused("Cannot read native process identity")
            return {"pid": pid, "exe": Path(buffer.value).resolve().as_posix(),
                    "creation_time_100ns": values[0].value}
        finally:
            self.api("CloseHandle", w.BOOL, w.HANDLE)(handle)

    def ancestors(self, targets):
        c, w = self.ctypes, self.W
        class Entry(c.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("pid", w.DWORD),
                        ("heap", c.c_size_t), ("module", w.DWORD), ("threads", w.DWORD),
                        ("parent", w.DWORD), ("priority", w.LONG), ("flags", w.DWORD),
                        ("exe", w.WCHAR * 260)]
        handle = self.api("CreateToolhelp32Snapshot", w.HANDLE, w.DWORD, w.DWORD)(2, 0)
        if handle == c.c_void_p(-1).value:
            raise Refused("Cannot inspect process ancestry")
        try:
            entry, parents = Entry(), {}
            entry.size = c.sizeof(entry)
            present = self.api("Process32FirstW", w.BOOL, w.HANDLE, c.POINTER(Entry))(handle, c.byref(entry))
            while present:
                parents[entry.pid] = entry.parent
                present = self.api("Process32NextW", w.BOOL, w.HANDLE, c.POINTER(Entry))(handle, c.byref(entry))
            chain, pid = {}, os.getpid()
            while pid and pid not in chain:
                identity = self.identity(pid)
                if identity is None:
                    raise Refused("Process ancestry disappeared during admission")
                if chain and identity["creation_time_100ns"] > min(item["creation_time_100ns"] for item in chain.values()):
                    raise Refused("Reused parent PID in process ancestry")
                chain[pid] = identity
                if set(targets) <= set(chain):
                    break
                pid = parents.get(pid, 0)
            return chain
        finally:
            self.api("CloseHandle", w.BOOL, w.HANDLE)(handle)

    def job(self, name, *, own=False):
        c, w = self.ctypes, self.W
        handle = self.api("OpenJobObjectW", w.HANDLE, w.DWORD, w.BOOL, w.LPCWSTR)(4, False, name)
        if not handle:
            if not own and c.get_last_error() == 2:
                return 0
            raise Refused("Cannot establish named Job ownership/quiescence")
        try:
            class Accounting(c.Structure):
                _fields_ = [(name, c.c_int64) for name in ("user", "kernel", "period_user", "period_kernel")] + [
                    (name, w.DWORD) for name in ("faults", "total", "active", "terminated")]
            value = Accounting()
            query = self.api("QueryInformationJobObject", w.BOOL, w.HANDLE, c.c_int, w.LPVOID, w.DWORD, w.LPVOID)
            if not query(handle, 1, c.byref(value), c.sizeof(value), None):
                raise Refused("Cannot query native Job accounting")
            if own:
                belongs = w.BOOL()
                current = self.api("GetCurrentProcess", w.HANDLE)()
                member = self.api("IsProcessInJob", w.BOOL, w.HANDLE, w.HANDLE, c.POINTER(w.BOOL))
                if not member(current, handle, c.byref(belongs)) or not belongs.value or not value.active:
                    raise Refused("Benchmark is outside its supervisor's active named Job")
            return value.active
        finally:
            self.api("CloseHandle", w.BOOL, w.HANDLE)(handle)

    def lease_is_occupied(self, path):
        c, w = self.ctypes, self.W
        create = self.api("CreateFileW", w.HANDLE, w.LPCWSTR, w.DWORD, w.DWORD,
                          w.LPVOID, w.DWORD, w.DWORD, w.HANDLE)
        handle = create(str(path), 0xC0000000, 0, None, 3, 0x80, None)
        if handle == c.c_void_p(-1).value:
            if c.get_last_error() == 32:  # Sharing violation, not permissions/missing file.
                return True
            raise Refused("GPU lease state is unknown")
        self.api("CloseHandle", w.BOOL, w.HANDLE)(handle)
        return False


def authorize(args, request, *, native=None, lock_root=None, now=None):
    """Run before trainer/Torch import, input hashing, NPZ reads or CUDA queries."""
    if args.authorization is None:
        raise Refused("Explicit benchmark start authorization is required")
    authorization = read_json(args.authorization)
    now = now or datetime.now(timezone.utc)
    try:
        expiry = datetime.fromisoformat(authorization["expires_utc"].replace("Z", "+00:00"))
        safe_names = (authorization["night_id"], authorization["job_id"])
        if any(not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name)
               for name in safe_names):
            raise Refused("Unsafe supervisor identity")
        if (authorization.get("schema") != 1 or authorization.get("purpose") != "kion-training-benchmark"
                or authorization.get("config_sha256") != request["config_sha256"]
                or expiry.tzinfo is None or expiry <= now
                or not authorization.get("prior_jobs")):
            raise Refused("Missing, stale, mismatched or incomplete authorization")
        root = Path(lock_root) if lock_root is not None else fixed_lock_root()
        gpu = authorization["gpu_uuid"]
        if not gpu.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
            raise Refused("Supervisor must expose exactly the authorized GPU UUID")
        key = hashlib.sha256(gpu.encode("utf-8")).hexdigest()
        marker = read_json(root / ("gpu-" + key + ".owner.json"))
        ledger = read_json(root / ("night-" + authorization["night_id"] + ".ledger.json"))
        if (marker.get("job_id") != authorization["job_id"] or marker.get("night_id") != authorization["night_id"]
                or ledger.get("night_id") != authorization["night_id"]
                or ledger.get("budget", {}).get("gpu_uuid") != gpu):
            raise Refused("GPU owner marker/ledger does not admit this benchmark")
        attempts = {attempt["id"]: attempt for attempt in ledger["attempts"]}
        own = attempts.get(authorization["job_id"], {})
        if (own.get("state") not in ("starting", "running") or own.get("job_name") != marker.get("job_name")
                or not own.get("identity") or not own.get("supervisor_pid")):
            raise Refused("Benchmark has no active strong supervisor identity")
        native = native or NativeOwnership()
        ancestors = native.ancestors({own["identity"]["pid"], own["supervisor_pid"]})
        if (ancestors.get(own["identity"]["pid"]) != own["identity"]
                or own["supervisor_pid"] not in ancestors):
            raise Refused("Authoritative worker identity/supervisor is outside current ancestry")
        native.job(marker["job_name"], own=True)
        if not native.lease_is_occupied(root / ("gpu-" + key + ".lock")):
            raise Refused("The supervisor's fixed GPU lease is not held")
        priors = {item["id"]: item for item in authorization["prior_jobs"]}
        if authorization["job_id"] in priors or len(priors) != len(authorization["prior_jobs"]):
            raise Refused("Invalid prior-job closure")
        for prior_id, pin in priors.items():
            previous = attempts.get(prior_id, {})
            if (previous.get("state") != "completed" or previous.get("cleanup") != "quiescent"
                    or not pin.get("job_sha256") or previous.get("job_sha256") != pin["job_sha256"]
                    or not previous.get("job_name") or not previous.get("identity")):
                raise Refused("Prior supervised fit is unfinished or cleanup is unknown")
            if native.job(previous["job_name"]):
                raise Refused("Prior supervised Job is still alive")
            identity = previous["identity"]
            if native.identity(identity["pid"]) == identity:
                raise Refused("Prior supervised worker still has the original native identity")
        for attempt in attempts.values():
            if attempt["id"] != authorization["job_id"] and attempt.get("state") in ("starting", "running", "stopping"):
                raise Refused("Another supervised attempt is still active")
            if (attempt["id"] != authorization["job_id"] and attempt.get("state") != "skipped_dependency"
                    and attempt.get("cleanup") != "quiescent"):
                raise Refused("Another supervised attempt has unknown cleanup")
        for key, names in (("source_hashes", request["source_paths"]), ("input_hashes", request["input_paths"])):
            if set(authorization.get(key, {})) != set(names):
                raise Refused("Authorization must bind exactly all source/input paths")
        return authorization
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise Refused(f"Cannot establish benchmark authorization: {error}") from error


def verify_hashes(authorization):
    for key in ("source_hashes", "input_hashes"):
        for path, expected in authorization[key].items():
            if sha256(path) != expected:
                raise Refused(f"{key} mismatch: {path}")


def batch_plan(module, data, vocab, config):
    eligible = np.flatnonzero((data["y"] >= 3) & (data["y"] < vocab))
    order = module.epoch_order(eligible, data["lengths"], config.seed, 0,
                               config.batch_size, config.bucket_size)
    count = config.warmup_updates + config.measure_updates
    result = []
    for _, microbatches in module.groups(order, config.batch_size, config.accumulation):
        result.append([indices.copy() for indices in microbatches])
        if len(result) == count:
            break
    if len(result) != count:
        raise Refused("Training split is too short for the bounded first-epoch prefix")
    declaration = [{"indices": [indices.tolist() for indices in group],
                    "widths": [int(data["lengths"][indices].max()) for indices in group]}
                   for group in result]
    return result, {"sha256": canonical_hash(declaration), "groups": declaration,
                    "eligible_training_rows": len(eligible), "epoch": 0,
                    "array_dtypes": {name: value.dtype.str for name, value in data.items()},
                    "array_shapes": {name: list(value.shape) for name, value in data.items()}}


def compare_values(left, right, *, rtol, atol):
    """Nested state comparison; integer/RNG values remain exact, floats tolerate roundoff."""
    errors, max_absolute = [], 0.0
    def visit(a, b, path):
        nonlocal max_absolute
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        if hasattr(b, "detach"):
            b = b.detach().cpu().numpy()
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(path + ": keys")
            else:
                for name in a:
                    visit(a[name], b[name], path + "." + str(name))
        elif isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                errors.append(path + ": length")
            else:
                for index, (one, two) in enumerate(zip(a, b)):
                    visit(one, two, path + f"[{index}]")
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            a, b = np.asarray(a), np.asarray(b)
            if a.shape != b.shape or a.dtype != b.dtype:
                errors.append(path + ": layout")
            elif a.dtype.kind in "fc":
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    errors.append(path + ": nonfinite")
                else:
                    if a.size:
                        max_absolute = max(max_absolute, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
                    if not np.allclose(a, b, rtol=rtol, atol=atol):
                        errors.append(path + ": tolerance")
            elif not np.array_equal(a, b):
                errors.append(path + ": exact")
        elif isinstance(a, (float, np.floating)) or isinstance(b, (float, np.floating)):
            if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
                errors.append(path + ": scalar")
        elif a != b:
            errors.append(path + ": exact")
    visit(left, right, "state")
    return {"passed": not errors, "rtol": rtol, "atol": atol,
            "max_absolute_tensor_error": max_absolute, "mismatch_paths": errors[:30]}


def aggregate(runs, correctness, minimum_improvement, *, require_improvement=False):
    summaries = {}
    for mode in ("legacy", "optimized"):
        selected = [run for run in runs if run["mode"] == mode]
        values = [run["measurement_seconds"] / run["measured_updates"] for run in selected]
        if not values or any(not math.isfinite(value) or value <= 0 for value in values):
            raise Refused("Both A/B arms require positive finite measured durations")
        summaries[mode] = {"median_seconds_per_update": statistics.median(values),
                           "raw_seconds_per_update": values,
                           "median_examples_per_second": statistics.median([
                               run["measured_examples"] / run["measurement_seconds"] for run in selected
                           ]) if all("measured_examples" in run for run in selected) else None}
    improvement = 1 - summaries["optimized"]["median_seconds_per_update"] / summaries["legacy"]["median_seconds_per_update"]
    passed = correctness and improvement >= minimum_improvement
    return {"arms": summaries, "fractional_time_reduction": improvement,
            "fractional_throughput_gain": 1 / (1 - improvement) - 1,
            "minimum_fractional_time_reduction": minimum_improvement,
            "correctness_passed": bool(correctness), "performance_passed": improvement >= minimum_improvement,
            "passed": bool(passed), "state": "completed" if correctness else "invalid",
            "require_improvement": require_improvement,
            "exit_code": 0 if correctness and (not require_improvement or passed) else 2,
            "interpretation": "operational throughput gate; no recommendation-quality claim"}


def load_trainer(directory, name):
    """Separate module globals/dependencies for frozen baseline and current code."""
    saved = {key: value for key, value in sys.modules.items() if key.startswith("kion_")}
    old_path = sys.path[:]
    for key in saved:
        del sys.modules[key]
    try:
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location(name, directory / "kion_conditioned_train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in list(sys.modules):
            if key.startswith("kion_"):
                del sys.modules[key]
        sys.modules.update(saved)
        sys.path[:] = old_path


def load_train(module, args, authorization):
    """Only metadata, train and the train/item feature blocks; never val/test."""
    metadata = read_json(args.data / "metadata.json")
    manifest = read_json(args.feature_manifest)
    data = module.load_data(args.data / "train.npz", metadata["vocab_size"], metadata["context"],
                            require_keys=True, unique_keys=False)
    with np.load(args.bundle_features, allow_pickle=False) as source:
        if set(source.files) != {"train_user_features", "val_user_features", "item_features"}:
            raise Refused("Expected the established basic-bundle sidecar")
        data["bundle_user"], items = source["train_user_features"], source["item_features"]
    import kion_basic_bundle_features as builder
    builder.validate_bundle_layout(data["bundle_user"], items, manifest["groups"])
    binding = manifest["inputs"]["train"]
    declared = manifest["features"]["arrays"]
    if (manifest.get("builder_contract") != "kion-basic-bundle-v1"
            or manifest["inputs"]["metadata"]["sha256"] != authorization["input_hashes"][(args.data / "metadata.json").resolve().as_posix()]
            or binding["sha256"] != authorization["input_hashes"][(args.data / "train.npz").resolve().as_posix()]
            or manifest["features"]["sha256"] != authorization["input_hashes"][args.bundle_features.resolve().as_posix()]
            or binding["rows"] != len(data["y"]) or binding["user_id_sha256"] != array_hash(data["user_id"])
            or binding["row_keys_sha256"] != canonical_hash({name: array_hash(data[name]) for name in module.ROW_KEYS})
            or manifest["vocabulary"]["sha256"] != canonical_hash(metadata["item_id_to_token"])):
        raise Refused("Train feature/input/vocabulary binding mismatch")
    for name, value in (("train_user_features", data["bundle_user"]), ("item_features", items)):
        declaration = declared[name]
        if (declaration["shape"] != list(value.shape) or declaration["dtype"] != "float32"
                or value.dtype != np.float32 or declaration["sha256"] != array_hash(value)):
            raise Refused("Feature array layout/hash mismatch")
    return metadata, manifest, data, items


def gpu_activity():
    """A bounded external telemetry sample; absence is explicit and non-fatal."""
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        return {"available": result.returncode == 0, "raw_csv": result.stdout.strip(),
                "columns": ["uuid", "utilization_percent", "memory_used_mib", "power_w", "sm_mhz"],
                "sampling": "bounded endpoint sample; not a sustained utilization average"}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}


def _cpu_copy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {name: _cpu_copy(item) for name, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def measure_run(torch, module, current, make_model, data, plan, config, initial,
                device, bounded, total_schedule, mode, index, profile_trace):
    """One repetition owns/always drains its staging provider, including errors."""
    with ExitStack() as resources:
        reset_started = time.perf_counter()
        model = make_model(module).to(device)
        model.load_state_dict(initial["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer.load_state_dict(initial["optimizer"])
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        scaler.load_state_dict(initial["scaler"])
        execution_data = current._validated_execution_data(data, model) if mode == "optimized" else data
        provider = {"requested": config.batch_transfer if mode == "optimized" else "direct",
                    "mode": "direct", "fallback_reason": None, "staging_reserved_bytes": 0}
        batcher = None
        if mode == "optimized" and config.batch_transfer != "direct":
            from kion_cuda_batches import PackedBatcher
            batcher = resources.enter_context(PackedBatcher(execution_data, device, max_batch_size=config.batch_size,
                                          pin_memory=config.batch_transfer == "pinned-packed"))
            execution_data.batcher = batcher
        rng = initial["rng"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state(rng["cuda"], device)
        model.train()
        torch.cuda.synchronize(device)
        reset_seconds = time.perf_counter() - reset_started
        losses, overflows = {}, 0
        def update(update_index):
            nonlocal overflows
            bounded()
            for group in optimizer.param_groups:
                group["lr"] = module.learning_rate(update_index, total_schedule, config.warmup_ratio, config.lr)
            collect = update_index == 0 or (update_index + 1) % config.log_updates == 0
            kwargs = {"execution_mode": "optimized", "collect_loss": collect} if mode == "optimized" else {}
            loss, overflow = module.update(model, execution_data, plan[update_index], optimizer, scaler, device, **kwargs)
            overflows += overflow
            if collect:
                if loss is None or not math.isfinite(loss):
                    raise Refused("Logged update loss must be finite")
                losses[str(update_index + 1)] = loss
        warmup_started = time.perf_counter()
        for update_index in range(config.warmup_updates):
            update(update_index)
        torch.cuda.synchronize(device)
        warmup_seconds = time.perf_counter() - warmup_started
        torch.cuda.reset_peak_memory_stats(device)
        activities_before = gpu_activity()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(config.measure_updates + 1)]
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True) if profile_trace else None
        if profiler:
            profiler.__enter__()
        measurement_started = time.perf_counter()
        events[0].record()
        try:
            for relative in range(config.measure_updates):
                update(config.warmup_updates + relative)
                events[relative + 1].record()
                if profiler:
                    profiler.step()
            torch.cuda.synchronize(device)
        finally:
            if profiler:
                profiler.__exit__(None, None, None)
        measurement_seconds = time.perf_counter() - measurement_started
        if profiler:
            profile_trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(profile_trace))
        durations = [events[n].elapsed_time(events[n + 1]) / 1000 for n in range(config.measure_updates)]
        peak = torch.cuda.max_memory_allocated(device)
        state = {"model": _cpu_copy(model.state_dict()), "optimizer": _cpu_copy(optimizer.state_dict()),
                 "scaler": scaler.state_dict(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(device), "python_rng": random.getstate(),
                 "numpy_rng": np.random.get_state(), "losses": losses, "overflows": overflows,
                 "updates": len(plan), "examples": sum(sum(map(len, group)) for group in plan)}
        examples = sum(sum(map(len, group)) for group in plan[config.warmup_updates:])
        if batcher is not None:
            provider.update(mode=batcher.mode, fallback_reason=batcher.fallback_reason,
                            staging_reserved_bytes=batcher.staging_reserved_bytes)
        run = {"run": index, "mode": mode, "reset_seconds": reset_seconds,
            "warmup_seconds": warmup_seconds, "warmup_updates": config.warmup_updates,
            "measurement_seconds": measurement_seconds, "measured_updates": config.measure_updates,
            "measured_examples": examples, "examples_per_second": examples / measurement_seconds,
            "updates_per_second": config.measure_updates / measurement_seconds,
            "per_update_cuda_stream_seconds": durations, "peak_allocated_bytes": peak,
            "logged_losses": losses, "overflow_retries": overflows, "batch_provider": provider,
            "gpu_activity_before": activities_before, "gpu_activity_after": gpu_activity()}
        return run, state


def execute(args, request, authorization, started):
    import torch
    import transformers
    config = Config(**request["config"]["settings"])
    deadline = started + config.max_seconds
    stop_file = Path(os.environ["KION_STOP_FILE"]) if os.environ.get("KION_STOP_FILE") else None
    def bounded():
        if time.perf_counter() >= deadline or (stop_file is not None and stop_file.exists()):
            raise Refused("Benchmark time/stop bound reached; no throughput success recorded")
    bounded()
    sys.path.insert(0, str(ROOT / "scripts"))
    baseline = load_trainer(args.baseline_root / "scripts", "benchmark_frozen")
    current = load_trainer(ROOT / "scripts", "benchmark_current")
    metadata, manifest, data, items = load_train(current, args, authorization)
    plan, plan_info = batch_plan(baseline, data, metadata["vocab_size"], config)
    other_plan, other_info = batch_plan(current, data, metadata["vocab_size"], config)
    if plan_info != other_info:
        raise Refused("Legacy/current actual training microbatch plan differs")
    del other_plan
    bounded()
    # Repeat native ownership after slow input hashing/loading, before first CUDA call.
    authorize(args, request)
    torch.set_num_threads(config.cpu_threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Refused("Exactly one supervisor-selected CUDA device is required")
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    make_model = lambda module: module.ConditionedGPT(metadata["vocab_size"], metadata["context"],
        config.hidden_size, config.layers, config.heads, "basic-bundle", items,
        bundle_groups=manifest["groups"])
    template = make_model(baseline)
    initial_model = _cpu_copy(template.state_dict())
    initial_optimizer = _cpu_copy(torch.optim.AdamW(template.parameters(), lr=config.lr,
                                                    weight_decay=config.weight_decay).state_dict())
    initial_scaler = torch.amp.GradScaler("cuda", init_scale=1024).state_dict()
    rng = {"python": random.getstate(), "numpy": np.random.get_state(),
           "torch": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state(device).clone()}
    parameters = sum(parameter.numel() for parameter in template.parameters())
    del template
    setup_seconds = time.perf_counter() - started
    total_schedule = math.ceil(plan_info["eligible_training_rows"] / (config.batch_size * config.accumulation)) * config.schedule_total_epochs
    initial = {"model": initial_model, "optimizer": initial_optimizer, "scaler": initial_scaler, "rng": rng}
    initial_fingerprint = {"model": canonical_hash({name: array_hash(value.numpy()) for name, value in initial_model.items()}),
        "optimizer": canonical_hash(initial_optimizer), "scaler": canonical_hash(initial_scaler),
        "torch_rng": array_hash(rng["torch"].numpy()), "cuda_rng": array_hash(rng["cuda"].numpy()),
        "numpy_rng": array_hash(rng["numpy"][1]), "python_rng": canonical_hash(rng["python"])}
    runs, comparisons, reference = [], [], None
    profile = args.profile_trace is not None
    modes = ["optimized"] if profile else run_order(config.repetitions) + ["current_legacy_parity"]
    for index, mode in enumerate(modes):
        bounded()
        module = baseline if mode == "legacy" else current
        run, state = measure_run(torch, module, current, make_model, data, plan, config, initial,
                                 device, bounded, total_schedule, mode, index, args.profile_trace)
        if reference is None:
            reference = state
        else:
            comparisons.append({"run": index, "mode": mode, **compare_values(reference, state, rtol=config.rtol, atol=config.atol)})
        runs.append(run)
        del state
        torch.cuda.empty_cache()
    finite = all(torch.isfinite(value).all().item() for value in reference["model"].values() if value.is_floating_point())
    correctness = finite and bool(comparisons) and all(item["passed"] for item in comparisons)
    verdict = None if profile else aggregate(runs, correctness, config.minimum_improvement,
                                           require_improvement=config.require_improvement)
    elapsed = time.perf_counter() - started
    return {"schema": 1, "mode": "profiler_only" if profile else "speed", "config": request["config"],
        "config_sha256": request["config_sha256"], "source_hashes": authorization["source_hashes"],
        "input_hashes": authorization["input_hashes"], "authorization_sha256": sha256(args.authorization),
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(device), "gpu_uuid": authorization["gpu_uuid"],
                    "cpu_threads": torch.get_num_threads(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "autocast": "float16", "parameters": parameters},
        "initial_state_sha256": initial_fingerprint,
        "batch_plan": plan_info, "runs": runs, "correctness": {"finite_model": finite, "comparisons": comparisons,
            "frozen_to_current_legacy_parity_required": not profile,
            "compared": ["model", "optimizer", "scaler", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "losses", "overflows", "updates", "examples"]},
        "verdict": verdict, "profile_trace": args.profile_trace.resolve().as_posix() if profile else None,
        "timing": {"setup_seconds": setup_seconds, "total_including_setup_seconds": elapsed,
                   "total_excluding_initial_setup_seconds": elapsed - setup_seconds,
                   "model_reset_seconds": sum(run["reset_seconds"] for run in runs),
                   "warmup_seconds": sum(run["warmup_seconds"] for run in runs),
                   "measured_only_seconds": sum(run["measurement_seconds"] for run in runs),
                   "per_update_cuda_stream_seconds_are_not_wall_throughput": True,
                   "telemetry_state_copy_reset_and_warmup_excluded_from_throughput": True},
        "train_only": True, "validation_accessed": False, "test_accessed": False,
        "quality_selection_performed": False, "hard_cleanup_owner": "existing kion_night_supervisor named Job"}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "bundle-features", "feature-manifest", "output"):
        result.add_argument("--" + name, required=True, type=Path)
    result.add_argument("--authorization", type=Path)
    result.add_argument("--baseline-root", type=Path,
                        default=ROOT / "artifacts/task2-kion/basic-bundle-20261020/bundle-r1")
    result.add_argument("--profile-trace", type=Path, help="Separate profiler-only run; produces no speed verdict")
    for name, field in Config.__dataclass_fields__.items():
        if isinstance(field.default, bool):
            result.add_argument("--" + name.replace("_", "-"), action="store_true", default=field.default)
        else:
            result.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    return result


def completion(result, output):
    profile = result["mode"] == "profiler_only"
    correctness = None if profile else result["verdict"]["correctness_passed"]
    return {"schema": 1, "state": "completed" if profile or correctness else "invalid",
            "mode": result["mode"], "correctness_passed": correctness,
            "performance_passed": None if profile else result["verdict"]["performance_passed"],
            "target_met": None if profile else result["verdict"]["passed"],
            "train_only": True, "test_accessed": False, "validation_accessed": False,
            "artifacts": [output.name], "sha256": {output.name: sha256(output)}}


def save_result(result, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    receipt = completion(result, output)
    with output.with_name("completion.json").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    started = time.perf_counter()
    args = parser().parse_args(argv)
    if (args.output.name == "completion.json" or args.output.exists() or args.output.with_name("completion.json").exists()
            or (args.profile_trace is not None and args.profile_trace.exists())):
        raise Refused("Benchmark outputs must be new; existing artifacts are preserved")
    request = request_manifest(args)
    authorization = authorize(args, request)
    verify_hashes(authorization)
    result = execute(args, request, authorization, started)
    save_result(result, args.output)
    return 0 if result["mode"] == "profiler_only" else result["verdict"]["exit_code"]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Refused as error:
        print(json.dumps({"status": "refused", "reason": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(3)
