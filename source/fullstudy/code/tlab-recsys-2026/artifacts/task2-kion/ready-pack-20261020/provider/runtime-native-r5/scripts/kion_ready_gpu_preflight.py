"""Disposable user-approved ready-GPT GPU preflight. Root/native launch only.

Copy these exact bytes to scripts/kion_ready_gpu_preflight.py before the
final runtime/source freeze and NEW provider manifest. No Torch at import.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

MODEL_SHA = "8d1be2352cd268481f4a1f7b764943f745e910695a1a270fd3d0b1c4e7ec24e7"
RUNNER_SHA = "16180fc30db79f6a5b5c0fc82080eb89658d4b466c65f3a96f66d313c85746d9"
SCOPE_ID = "ready-pack-20261020-r1"
SCOPE_SHA = "a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962"
PAIR_CAP = 64 << 20
CELLS = (("b64a2", 64, 2), ("b128a1", 128, 1))
WORKER_WALL_SECONDS = 600
GPU_NOMINAL_BYTES = 8 << 30
GPU_CAPACITY_TOLERANCE_BYTES = 1 << 20


class Refused(RuntimeError):
    pass


class BudgetExpired(RuntimeError):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def bound_json(pin):
    path = Path(pin["path"])
    if not path.is_absolute() or sha(path) != pin["sha256"]:
        raise Refused("Missing/changed pinned JSON: " + str(path))
    return read(path)


def gpu_admission(properties, report):
    # Root's exact RTX 3070 reports 8191.5 MiB via CUDA Driver API.
    # Admit at most 1 MiB below nominal 8 GiB; UUID/lease/quiet checks remain separate.
    minimum = GPU_NOMINAL_BYTES - GPU_CAPACITY_TOLERANCE_BYTES
    report["gpu"] = {"name": properties.name, "total_bytes": properties.total_memory,
                     "nominal_total_bytes": GPU_NOMINAL_BYTES,
                     "capacity_tolerance_bytes": GPU_CAPACITY_TOLERANCE_BYTES,
                     "minimum_total_bytes": minimum}
    if "RTX 3070" not in properties.name or properties.total_memory < minimum:
        raise Refused("Expected nominal8GiB RTX3070 physical device: "
                      f"name={properties.name!r}, total_bytes={properties.total_memory}, "
                      f"minimum_total_bytes={minimum}, tolerance_bytes={GPU_CAPACITY_TOLERANCE_BYTES}")


def chunk_size(batch, dim):
    if type(batch) is not int or type(dim) is not int or min(batch, dim) < 1:
        raise Refused("Positive actual B and candidate D required")
    value = min(16, PAIR_CAP // (batch * dim * 4))
    if value < 1:
        raise Refused("One candidate already exceeds the 64MiB pair output cap")
    return value


def expected_training_pair_calls(batch, candidates, dim, chunk, microbatches, retries):
    single_cache = candidates <= chunk and batch * candidates * dim * 4 <= PAIR_CAP
    return (1 if single_cache else 2) * math.ceil(candidates / chunk) * microbatches * (retries + 1)


def validate_request(request):
    if request.get("contract") != "kion-ready-gpu-preflight-request-v1" or request.get("state") != "root-bound-ready":
        raise Refused("A concrete root-bound request is required; proposal is not executable")
    if request.get("launch_owner") != "root" or request.get("cells") != [list(cell) for cell in CELLS]:
        raise Refused("Only the two predeclared root benchmark cells are admitted")
    if request.get("worker_wall_seconds") != WORKER_WALL_SECONDS or request.get("pair_output_byte_cap") != PAIR_CAP:
        raise Refused("Changed finite time/memory proposal requires another review")
    freeze = bound_json(request["source_freeze"])
    cpu = bound_json(request["cpu_preflight"])
    decision = bound_json(request["scope_decision"])
    if (request["scope_decision"]["sha256"] != SCOPE_SHA or decision.get("scope_id") != SCOPE_ID
            or request.get("scope_id") != SCOPE_ID or request.get("active_families") != decision["active_families"]):
        raise Refused("Exact human-approved ready scope and ordered active families required")
    quiet = bound_json(request["native_quiet"])
    if quiet.get("state") != "quiet" or not quiet.get("gpu_uuid", "").startswith("GPU-"):
        raise Refused("Pinned native quiet policy is required")
    if not isinstance(quiet.get("benign_gpu_processes"), list) or "protected_dwm_pid" not in quiet:
        raise Refused("Explicit quiet process identity policy is required")
    if freeze.get("state") != "frozen" or not Path(freeze["bundle_dir"]).is_absolute():
        raise Refused("Final frozen runtime required")
    sources = freeze["source_hashes"]
    if not {"torch", "numpy", "transformers", "scipy"}.issubset(freeze.get("runtime_versions", {})):
        raise Refused("Frozen production package version closure is required")
    for key in ("data", "feature_dir"):
        if not Path(request[key]).is_absolute():
            raise Refused("Absolute production paths required")
    for path in (Path(request["data"]) / name for name in ("metadata.json", "train.npz", "val.npz")):
        if freeze["input_hashes"].get(path.as_posix()) != sha(path):
            raise Refused("Prepared data absent from frozen input closure")
    manifest = Path(request["feature_dir"]) / "manifest.json"
    if freeze["input_hashes"].get(manifest.as_posix()) != sha(manifest):
        raise Refused("Full manifest absent from frozen input closure")
    if sources.get("scripts/kion_complete_model.py") != MODEL_SHA or sources.get("scripts/kion_complete_train.py") != RUNNER_SHA:
        raise Refused("Accepted complete model/runner changed")
    required = {"scripts/kion_ready_gpu_preflight.py", "scripts/kion_complete_pack.py",
                "scripts/kion_training_benchmark.py", "scripts/kion_night_supervisor.py", "scripts/hm_job.py"}
    if not required.issubset(sources):
        raise Refused("Incomplete frozen entrypoint/native/provider closure")
    if (cpu.get("contract") != "kion-ready-production-cpu-preflight-v1" or cpu.get("state") != "passed"
            or cpu.get("source_freeze_sha256") != request["source_freeze"]["sha256"]
            or cpu.get("provider_manifest_sha256") != sha(Path(request["feature_dir"]) / "manifest.json")
            or cpu.get("provider_status") != "admitted_scoped" or cpu.get("scope_id") != SCOPE_ID
            or cpu.get("scope_decision") != request["scope_decision"]
            or cpu.get("active_families") != decision["active_families"]
            or cpu.get("full_package_admitted") is not False or cpu.get("full_row_fit_admitted") is not True
            or cpu.get("fitsteps") != 0 or cpu.get("numeric_audit", {}).get("fp16_cast_finite") is not True
            or cpu.get("numeric_audit", {}).get("late_aug09_popular_stress") is not True):
        raise Refused("Exact scoped ready-provider CPU/magnitude prerequisite is not met")
    if len(cpu.get("popular_candidate_tokens", [])) < 16:
        raise Refused("CPU preflight must bind at least16 actual popular known candidates")
    return freeze, cpu


def verify_closure(freeze):
    bundle = Path(freeze["bundle_dir"])
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*.py")}
    if actual != set(freeze["source_hashes"]):
        raise Refused("Frozen Python inventory differs")
    for relative, expected in freeze["source_hashes"].items():
        path = (bundle / relative).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != expected:
            raise Refused("Frozen source changed: " + relative)
    for path, expected in freeze["input_hashes"].items():
        if not Path(path).is_absolute() or sha(path) != expected:
            raise Refused("Source/input closure changed: " + path)
    if sha(freeze["interpreter"]) != freeze["interpreter_sha256"]:
        raise Refused("Interpreter changed")
    return bundle


def authorize(request, request_path, authorization_path, freeze):
    authorization = read(authorization_path)
    expiry = datetime.fromisoformat(authorization["expires_utc"].replace("Z", "+00:00"))
    if (authorization.get("purpose") != "kion-ready-gpu-preflight-v1"
            or authorization.get("request_sha256") != sha(request_path)
            or expiry.tzinfo is None or expiry <= datetime.now(timezone.utc)
            or authorization.get("gpu_uuid") != os.environ.get("CUDA_VISIBLE_DEVICES")
            or authorization.get("native_quiet") != request["native_quiet"]):
        raise Refused("Stale/mismatched root authorization or GPU UUID")
    helper_path = Path(freeze["bundle_dir"]) / "scripts/kion_training_benchmark.py"
    if sha(helper_path) != freeze["source_hashes"]["scripts/kion_training_benchmark.py"]:
        raise Refused("Native ownership helper changed")
    spec = importlib.util.spec_from_file_location("complete_preflight_native", helper_path)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    native = helper.NativeOwnership()
    root = helper.fixed_lock_root()
    key = hashlib.sha256(authorization["gpu_uuid"].encode()).hexdigest()
    owner = read(root / ("gpu-" + key + ".owner.json"))
    ledger = read(root / ("night-" + authorization["night_id"] + ".ledger.json"))
    attempts = [value for value in ledger["attempts"] if value["id"] == authorization["job_id"]]
    if len(attempts) != 1:
        raise Refused("Ambiguous native attempt identity")
    own = attempts[0]
    if (owner.get("night_id") != authorization["night_id"] or owner.get("job_id") != authorization["job_id"]
            or own.get("state") not in ("starting", "running") or own.get("job_name") != owner.get("job_name")
            or own.get("identity", {}).get("pid") != os.getpid()
            or Path(own.get("identity", {}).get("exe", "")).resolve() != Path(freeze["interpreter"]).resolve()
            or Path(sys.executable).resolve() != Path(freeze["interpreter"]).resolve()
            or own.get("source_hashes") != freeze["source_hashes"]
            or ledger.get("budget", {}).get("gpu_uuid") != authorization["gpu_uuid"]):
        raise Refused("Native owner/ledger does not admit this exact worker")
    ancestors = native.ancestors({os.getpid(), own["supervisor_pid"]})
    if ancestors.get(os.getpid()) != own["identity"] or own["supervisor_pid"] not in ancestors:
        raise Refused("Native process creation-time identity/ancestry differs")
    if native.job(own["job_name"], own=True) < 1 or not native.lease_is_occupied(root / ("gpu-" + key + ".lock")):
        raise Refused("Named Job membership or exclusive GPU lease missing")
    if any(value["id"] != own["id"] and value.get("state") in ("starting", "running", "stopping") for value in ledger["attempts"]):
        raise Refused("Another attempt remains active")


def quiet_before_cuda(request, freeze, *, probe=None):
    """Reinspect actual consumers after slow CPU work using the same native policy."""
    quiet = bound_json(request["native_quiet"])
    if quiet.get("state") != "quiet" or quiet.get("gpu_uuid") != os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise Refused("Pinned quiet policy differs from the authorized physical GPU")
    policy = {name: quiet[name] for name in ("gpu_uuid", "protected_dwm_pid", "benign_gpu_processes")}
    if probe is None:
        path = Path(freeze["bundle_dir"]) / "scripts/kion_night_supervisor.py"
        if sha(path) != freeze["source_hashes"]["scripts/kion_night_supervisor.py"]:
            raise Refused("Frozen native quiet probe changed")
        spec = importlib.util.spec_from_file_location("complete_preflight_quiet", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        probe = module.gpu_quiet
    started = time.perf_counter()
    exceptions = probe(policy, False)
    return {"policy_receipt_sha256": request["native_quiet"]["sha256"],
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started, "exceptions": exceptions}


def provider_admission(contract, dim):
    """Admit both fixed shapes before CUDA; never silently shrink a cell."""
    limits = dict(contract.get("limits", {}))
    for key in ("max_batch_rows", "max_pair_cells", "max_pair_bytes", "profile_cache_rows", "bank_cache_days"):
        if type(limits.get(key)) is not int or limits[key] < 1:
            raise Refused("Missing positive actual provider limit: " + key)
    shapes = []
    for name, batch, accumulation in CELLS:
        chunk = chunk_size(batch, dim)
        cells, size = batch * chunk, batch * chunk * dim * 4
        if batch > limits["max_batch_rows"] or cells > limits["max_pair_cells"] or size > limits["max_pair_bytes"]:
            raise Refused("Actual provider limits refuse fixed cell " + name + "; no hidden shape fallback")
        shapes.append({"name": name, "batch_size": batch, "accumulation": accumulation,
                       "candidate_chunk": chunk, "pair_cells": cells, "pair_output_bytes": size})
    return {"limits": limits, "admitted_shapes": shapes}


def summarize_calls(calls):
    """Actual invocations, including rematerialization/retries and failed calls."""
    result = {"batch_calls": 0, "candidate_features_calls": 0, "failed_calls": 0,
              "provider_cpu_wall_seconds": 0., "candidate_pair_cells": 0,
              "max_rows": 0, "max_pair_cells": 0, "max_pair_output_bytes": 0, "by_phase": {}}
    for call in calls:
        result[call["kind"] + "_calls"] += 1
        result["failed_calls"] += not call["completed"]
        result["provider_cpu_wall_seconds"] += call["wall_seconds"]
        phase = result["by_phase"].setdefault(call["phase"], {"batch_calls": 0, "candidate_features_calls": 0,
            "provider_cpu_wall_seconds": 0., "failed_calls": 0})
        phase[call["kind"] + "_calls"] += 1
        phase["provider_cpu_wall_seconds"] += call["wall_seconds"]
        phase["failed_calls"] += not call["completed"]
        rows = call.get("rows", call.get("requested_shape", [0])[0])
        result["max_rows"] = max(result["max_rows"], rows)
        if call["kind"] == "candidate_features":
            b, c = call["requested_shape"]
            result["candidate_pair_cells"] += b * c
            result["max_pair_cells"] = max(result["max_pair_cells"], b * c)
            result["max_pair_output_bytes"] = max(result["max_pair_output_bytes"], call.get("output_bytes", 0))
    return result


def module_closure(freeze):
    bundle = Path(freeze["bundle_dir"]).resolve()
    observed = {}
    for name, module in list(sys.modules.items()):
        if name.startswith("kion_") and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(bundle):
                raise Refused("Runtime KION module escaped frozen bundle: " + name)
            relative = path.relative_to(bundle).as_posix()
            if freeze["source_hashes"].get(relative) != sha(path):
                raise Refused("Unpinned runtime module: " + name)
            observed[name] = relative
    return observed


def peak_rss():
    import ctypes as c
    from ctypes import wintypes as w
    class Counters(c.Structure):
        _fields_ = [("cb", w.DWORD), ("faults", w.DWORD)] + [(key, c.c_size_t) for key in
            ("peak", "working", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]
    kernel = c.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = w.HANDLE
    value = Counters()
    value.cb = c.sizeof(value)
    function = c.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    function.argtypes, function.restype = [w.HANDLE, c.POINTER(Counters), w.DWORD], w.BOOL
    if not function(kernel.GetCurrentProcess(), c.byref(value), c.sizeof(value)):
        raise Refused("Cannot measure process peak RSS")
    return int(value.peak)


class Budget:
    def __init__(self, stop_file):
        self.started = time.perf_counter()
        self.stop_file = Path(stop_file)

    def check(self, reserve=0):
        if self.stop_file.exists() or time.perf_counter() - self.started + reserve >= WORKER_WALL_SECONDS:
            raise BudgetExpired("Finite fixture stopped before completion; no epoch ETA")
        if peak_rss() > 24 << 30:
            raise BudgetExpired("24GiB host peak-RSS budget exceeded")


def fixture(provider, np):
    data = provider.splits["train"]
    eligible = np.flatnonzero((data["y"] >= 3) & (data["y"] < provider.metadata["vocab_size"]))
    origins = provider.origin_keys("train")
    keys, counts = np.unique(origins[eligible], return_counts=True)
    keys = keys[counts >= 384]
    if len(keys) < 3:
        raise Refused("Need3 actual origins with384 distinct eligible rows each")
    chosen = keys[[0, len(keys) // 2, -1]]
    groups = []
    for origin in chosen:
        rows = eligible[origins[eligible] == origin]
        rows = rows[np.argsort(data["lengths"][rows], kind="stable")]
        starts = (0, len(rows) // 2 - 64, len(rows) - 128)
        for stratum, start in enumerate(starts):
            selected = rows[start:start + 128].copy()
            rng = np.random.default_rng(np.random.SeedSequence([20261020, int(origin), stratum]))
            rng.shuffle(selected)
            groups.append({"origin": int(origin), "warmup": stratum == 0, "stratum": stratum,
                           "rows": selected.tolist(), "max_length": int(data["lengths"][selected].max())})
    flat = [row for group in groups for row in group["rows"]]
    if len(set(flat)) != len(flat):
        raise Refused("Benchmark queries must not repeat within a cell")
    val = provider.splits["val"]
    indices = np.argsort(val["lengths"], kind="stable")[-128:]
    if len(indices) != 128:
        raise Refused("Need128 real validation rows")
    return groups, indices


class FamilyTimings:
    """Aggregate inclusive BENCH-only timings; nested entries are nonadditive."""
    def __init__(self, audited, franchise=None):
        self.audited, self.franchise = audited, franchise
        self.stats, self.restores, self.installed, self.skipped = {}, [], [], []
        self.classification_errors = 0

    def _query_label(self, args, kwargs):
        provider = self.audited.inner
        split = args[0] if args else kwargs["split"]
        row = args[1] if len(args) > 1 else kwargs["row"]
        data = provider.splits[split]
        key = (int(data["user_id"][row]), provider._origin(split, row), int(data["lengths"][row]), data["x"][row].tobytes())
        return "query.cache_hit" if key in provider._profiles else "query.cache_miss"

    def _bank_label(self, args, kwargs):
        declaration = args[0] if args else kwargs["declaration"]
        origin = args[1] if len(args) > 1 else kwargs["origin"]
        key = (declaration["family"], tuple(declaration["columns"]), origin)
        return "external_bank.cache_hit" if key in self.audited.inner._banks else "external_bank.cache_miss"

    @staticmethod
    def _scope_label(prefix):
        def label(args, kwargs):
            scope = args[1] if len(args) > 1 else kwargs.get("scope")
            return prefix + (scope if scope in ("user", "candidate") else "other")
        return label

    @staticmethod
    def _franchise_label(prefix, position):
        def label(args, kwargs):
            kind = kwargs.get("order_kind", args[position] if len(args) > position else None)
            return prefix + (kind if kind in ("mainline_release", "story_chronology", "catalog_release_precedes") else "other")
        return label

    def _wrap(self, owner, name, label):
        if owner is None or not callable(getattr(owner, name, None)):
            self.skipped.append(name + ":unavailable")
            return
        if any(item[0] is owner and item[1] == name for item in self.restores):
            self.skipped.append(name + ":alias")
            return
        original = getattr(owner, name)
        try:
            local = vars(owner)
        except TypeError:
            self.skipped.append(name + ":no_attribute_dict")
            return
        had_local, saved = name in local, local.get(name)

        def wrapped(*args, **kwargs):
            started, failed = time.perf_counter(), False
            try:
                key = label(args, kwargs) if callable(label) else label
            except Exception:
                self.classification_errors += 1
                key = "observer.classification_unknown"
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                phase = self.stats.setdefault(self.audited.phase, {})
                entry = phase.setdefault(key, {"calls": 0, "failed_calls": 0, "wall_seconds": 0.})
                entry["calls"] += 1
                entry["failed_calls"] += int(failed)
                entry["wall_seconds"] += time.perf_counter() - started

        try:
            setattr(owner, name, wrapped)
        except (AttributeError, TypeError):
            self.skipped.append(name + ":not_writable")
            return
        self.restores.append((owner, name, had_local, saved))
        self.installed.append(name if callable(label) else label)

    def install(self):
        provider = self.audited.inner
        try:
            self._wrap(provider, "_query", self._query_label)
            personal = getattr(provider, "personal", None)
            self._wrap(personal, "prepare_queries", "personal.prepare_queries")
            self._wrap(personal, "candidate_features", "personal.candidate_features")
            for family in ("people_dynamics", "dynamic", "related"):
                self._wrap(getattr(provider, family, None), "features", family + ".features")
            self._wrap(provider, "_external_bank", self._bank_label)
            self._wrap(provider, "_matrix", "external_bank.matrix_open")
            self._wrap(provider, "_transform_numeric", self._scope_label("numeric_transform."))
            self._wrap(provider, "_numeric_audit", self._scope_label("numeric_audit."))
            self._wrap(provider, "_validate_model_range", self._scope_label("numeric_range."))
            franchise = self.franchise or sys.modules.get("kion_franchise_progress")
            self._wrap(franchise, "candidate_progress", self._franchise_label("franchise.unprepared.", 8))
            self._wrap(getattr(franchise, "PreparedGraph", None), "candidate_progress", self._franchise_label("franchise.prepared.", 7))
            self._wrap(franchise, "_candidate_progress", self._franchise_label("franchise.score.", 7))
        except BaseException:
            self.restore()
            raise

    def reset(self):
        self.stats = {}
        self.classification_errors = 0

    def snapshot(self):
        return {"semantics": "inclusive nested timings; nonadditive with each other and provider/update wall",
                "by_phase": {phase: {key: dict(value) for key, value in entries.items()} for phase, entries in self.stats.items()},
                "installed": list(self.installed), "skipped": list(self.skipped),
                "classification_errors": self.classification_errors}

    def restore(self):
        errors = []
        for owner, name, had_local, saved in reversed(self.restores):
            try:
                setattr(owner, name, saved) if had_local else delattr(owner, name)
            except Exception as error:
                errors.append(name + ":" + type(error).__name__)
        self.restores = []
        return errors


class AuditedProvider:
    def __init__(self, provider, np, budget):
        self.inner, self.np, self.budget = provider, np, budget
        self.calls, self.phase = [], "startup"

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def audit(self, values, scope):
        values = self.np.asarray(values)
        maximum = float(self.np.max(self.np.abs(values))) if values.size else 0.
        if not self.np.isfinite(values).all() or maximum > 60000:
            raise Refused("Unsafe pre-transfer finite/FP16 magnitude in " + scope)
        return maximum

    def batch(self, rows, split):
        self.budget.check()
        started = time.perf_counter()
        call = {"kind": "batch", "phase": self.phase, "split": split, "rows": len(rows), "completed": False}
        self.calls.append(call)
        try:
            result = self.inner.batch(rows, split)
            call["max_abs"] = self.audit(result[3], "user")
            call["completed"] = True
            return result
        finally:
            call["wall_seconds"] = time.perf_counter() - started

    def candidate_features(self, rows, ids, split):
        self.budget.check()
        started = time.perf_counter()
        call = {"kind": "candidate_features", "phase": self.phase, "split": split,
                "requested_shape": list(ids.shape), "completed": False}
        self.calls.append(call)
        try:
            values = self.inner.candidate_features(rows, ids, split)
            call.update(shape=list(values.shape), output_bytes=values.nbytes)
            expected = (len(rows), ids.shape[1], len(self.feature_columns["candidate"]))
            if values.shape != expected or values.nbytes > PAIR_CAP or str(values.dtype) != "float32":
                raise Refused("Pair output shape/dtype/64MiB cap differs")
            call["max_abs"] = self.audit(values, "candidate")
            call["completed"] = True
            return values
        finally:
            call["wall_seconds"] = time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("request", "authorization", "output", "stop-file"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args(argv)
    # Supervisor creates only the output/log root; never overwrite a report.
    if (args.output / "report.json").exists():
        raise Refused("Choose a fresh disposable output")
    budget = Budget(args.stop_file)
    request = read(args.request)
    freeze, cpu = validate_request(request)
    authorize(request, args.request, args.authorization, freeze)
    verification_started = time.perf_counter()
    bundle = verify_closure(freeze)
    if Path(__file__).resolve() != bundle / "scripts/kion_ready_gpu_preflight.py":
        raise Refused("Execute only the frozen benchmark entrypoint")
    sys.path.insert(0, str(bundle / "scripts"))
    import numpy as np
    import torch
    import kion_complete_train as train
    from kion_complete_model import sample_training_candidates
    import kion_complete_pack as pack
    import importlib.metadata
    for package, expected in freeze["runtime_versions"].items():
        if importlib.metadata.version(package) != expected:
            raise Refused("Runtime package differs: " + package)
    if torch.cuda.is_initialized():
        raise Refused("Unexpected CUDA initialization before provider/closure admission")
    report = {"state": "running", "purpose": "disposable_execution_not_quality", "full_fit": False,
        "scope_id": SCOPE_ID, "scope_decision": request["scope_decision"],
        "full_package_admitted": False, "full_row_fit_admitted": True,
        "request_sha256": sha(args.request), "source_verification_seconds": time.perf_counter() - verification_started,
        "cells": [], "epoch_eta": None, "instrumentation": "CPU audits and synchronization are included in measured wall"}
    provider, model, optimizer, scaler, families = None, None, None, None, None
    try:
        provider_started = time.perf_counter()
        provider = pack.load_training_provider(argparse.Namespace(data=Path(request["data"]),
            feature_dir=Path(request["feature_dir"]), allow_partial_integration=False,
            integration_mode="scoped_production", scope_decision=Path(request["scope_decision"]["path"])))
        if canonical(provider.contract()) != cpu["provider_contract_sha256"]:
            raise Refused("Scoped ready provider contract differs from CPU admission")
        report["provider_load_seconds"] = time.perf_counter() - provider_started
        report["provider_admission"] = provider_admission(provider.contract(), len(provider.feature_columns["candidate"]))
        audited = AuditedProvider(provider, np, budget)
        families = FamilyTimings(audited)
        families.install()
        popular = np.asarray(cpu["popular_candidate_tokens"][:16], np.int64)
        if len(np.unique(popular)) != 16 or np.any(popular < 3) or np.any(popular >= provider.metadata["vocab_size"]):
            raise Refused("Popular stress must use16 distinct known tokens")
        audited.audit(provider.item_features, "all_static_items")
        for field, values in provider.text_features.items():
            audited.audit(values, "text_" + field)
        groups, val_indices = fixture(provider, np)
        write(args.output / "fixture.json", {"groups": groups, "val_indices": val_indices.tolist(),
            "selection_uses_outcomes": False, "same_rows_both_cells": True,
            "cache_state": "new active origin; OS file cache is uncontrolled"})
        report["module_closure_before_cuda"] = module_closure(freeze)
        # Repeat authoritative live ownership after slow CPU loading/hashing.
        authorize(request, args.request, args.authorization, freeze)
        report["quiet_recheck_before_cuda"] = quiet_before_cuda(request, freeze)
        budget.check(reserve=60)
        torch.set_num_threads(2)
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(0)
        gpu_admission(properties, report)

        def phase(label, callback):
            budget.check()
            audited.phase = label
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = callback()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            budget.check()
            if torch.cuda.max_memory_allocated() > 7 << 30 or torch.cuda.max_memory_reserved() > 15 << 29:
                raise BudgetExpired("GPU headroom budget exceeded")
            return result, elapsed

        for name, batch_size, accumulation in CELLS:
            budget.check(reserve=20)
            families.reset()
            torch.manual_seed(2026)
            torch.cuda.manual_seed_all(2026)
            parsed = train.parser().parse_args(["--data", request["data"], "--feature-dir", request["feature_dir"],
                "--integration-mode", "scoped_production", "--scope-decision", request["scope_decision"]["path"],
                "--output", str(args.output / "unused-model-output"), "--device", "cuda", "--training-order", "origin-grouped-v1",
                "--batch-size", str(batch_size), "--accumulation", str(accumulation), "--hard-negatives", "7", "--uniform-negatives", "8"])
            train.integration_contract(provider, parsed)
            dim = len(provider.feature_columns["candidate"])
            chunk = chunk_size(batch_size, dim)
            torch.cuda.reset_peak_memory_stats()
            model = train.make_model(provider, parsed).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
            scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024)
            cell = {"name": name, "batch_size": batch_size, "accumulation": accumulation, "candidate_dim": dim,
                    "candidate_chunk": chunk, "pair_output_bound_bytes": batch_size * chunk * dim * 4,
                    "pair_training_cache_policy": "single chunk <=64MiB immutable CPU snapshot; GPU head rematerialized; multi-chunk provider replay",
                    "updates": [], "calls_start": len(audited.calls)}
            initial = hashlib.sha256()
            for parameter_name, parameter in model.named_parameters():
                initial.update(parameter_name.encode("utf-8"))
                initial.update(parameter.detach().cpu().numpy().tobytes())
            cell["initial_parameters_sha256"] = initial.hexdigest()
            if report["cells"] and cell["initial_parameters_sha256"] != report["cells"][0]["initial_parameters_sha256"]:
                raise Refused("Cells did not start from identical seeded parameters")

            def catalog():
                with torch.no_grad(), train.legacy.autocast(device):
                    table = model.candidate_table()
                    if not torch.isfinite(table).all():
                        raise Refused("Nonfinite fullV candidate table")
                return list(table.shape)

            cell["catalog_shape"], cell["fullV_catalog_forward_seconds"] = phase("fullV_catalog", catalog)
            for position, group in enumerate(groups):
                rows = np.array(group["rows"], np.int64)
                if group["warmup"]:
                    probe_rows = rows[:batch_size]
                    probe = np.tile(np.asarray(cpu["popular_candidate_tokens"][:16], np.int64), (len(probe_rows), 1))
                    def cold_probe():
                        for start in range(0, probe.shape[1], chunk):
                            audited.candidate_features(probe_rows, probe[:, start:start + chunk], "train")
                    _, seconds = phase("cold_prefix_plus_probe_plus_first_bank_verification", cold_probe)
                    cell.setdefault("cold_origin_probes", []).append({"origin": group["origin"], "seconds": seconds})
                micros = [rows[start:start + batch_size] for start in range(0, 128, batch_size)]
                calls_start = len(audited.calls)
                (losses, retries), seconds = phase("warmup" if group["warmup"] else "measured_new_queries",
                    lambda: train.update(model.train(), audited, micros, optimizer, scaler, device,
                        hard_negatives=7, uniform_negatives=8, candidate_chunk_size=chunk))
                cell["updates"].append({"origin": group["origin"], "warmup": group["warmup"],
                    "wall_seconds": seconds, "examples": 128, "losses": losses, "amp_retries": retries,
                    "provider_call_summary": summarize_calls(audited.calls[calls_start:]),
                    "expected_pair_calls_with_forward_replay_and_retries": expected_training_pair_calls(batch_size, 16, dim, chunk, len(micros), retries),
                    "call_stage_split": "single-chunk backward reuses CPU snapshot; multi-chunk replays provider; actual count above"})
                cell["family_timings"] = families.snapshot()
                write(args.output / "progress.json", {**report, "active_cell": cell})

            diagnostic_rows = np.array(groups[-1]["rows"][:batch_size], np.int64)

            def backward(pair):
                optimizer.zero_grad(set_to_none=True)
                x, lengths, y, user = train.batch(audited, diagnostic_rows, "train", device)
                with train.legacy.autocast(device):
                    table = model.candidate_table()
                    encoded = model.encode_context(x, lengths, user, candidate_table=table)
                    logits = model.retrieval_logits(encoded, table)
                    if pair:
                        ids = sample_training_candidates(logits, y, hard_negatives=7, uniform_negatives=8)
                        scores = train.pair_scores(model, audited, diagnostic_rows, "train", encoded, table, logits, ids, chunk)
                        loss = torch.nn.functional.cross_entropy(scores.float(), torch.zeros(len(y), dtype=torch.long, device=device))
                    else:
                        loss = torch.nn.functional.cross_entropy(logits[:, 3:].float(), y - 3)
                loss.backward()
                finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
                optimizer.zero_grad(set_to_none=True)
                if not finite:
                    raise Refused("Nonfinite unscaled diagnostic backward")
                return float(loss.detach())

            _, cell["diagnostic_retrieval_forward_backward_seconds"] = phase("diagnostic_retrieval_FB", lambda: backward(False))
            _, cell["diagnostic_pair_forward_backward_seconds"] = phase("diagnostic_pair_FB_includes_replay", lambda: backward(True))

            def popular_stress():
                rows = val_indices[:batch_size]
                x, lengths, _, user = train.batch(audited, rows, "val", device)
                ids = torch.tensor(np.tile(cpu["popular_candidate_tokens"][:16], (len(rows), 1)), device=device)
                with torch.no_grad(), train.legacy.autocast(device):
                    table = model.candidate_table()
                    encoded = model.encode_context(x, lengths, user, candidate_table=table)
                    logits = model.retrieval_logits(encoded, table)
                    scores = train.pair_scores(model, audited, rows, "val", encoded, table, logits, ids, chunk)
                if not torch.isfinite(scores).all():
                    raise Refused("Nonfinite lateAug9 popular-candidate AMP stress")
                return {"pairs": ids.numel(), "ranking_use": False}

            cell["popular_stress"], cell["popular_stress_seconds"] = phase("late_aug09_popular_numeric_probe", popular_stress)
            cell["validation_subset_metrics"], cell["validation_fullV_unforced_rerank128_seconds"] = phase("validation_unforced_fullV_rerank128",
                lambda: train.evaluate(model, audited, batch_size, device, retrieval_topk=128,
                    candidate_chunk_size=chunk, indices=val_indices, should_stop=lambda: "runtime_budget" if args.stop_file.exists() else None))
            measured = [value for value in cell["updates"] if not value["warmup"]]
            cell["measured_new_query_examples_per_second"] = sum(value["examples"] for value in measured) / sum(value["wall_seconds"] for value in measured)
            cell["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            cell["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
            cell["peak_rss_bytes"] = peak_rss()
            cell["provider_calls"] = audited.calls[cell.pop("calls_start"):]
            cell["provider_call_summary"] = summarize_calls(cell["provider_calls"])
            cell["family_timings"] = families.snapshot()
            cell["diagnostic_timing_semantics"] = "nonadditive; retrieval/pair diagnostics repeat shared work already included in update wall"
            report["cells"].append(cell)
            cell = None
            model, optimizer, scaler = None, None, None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        report["module_closure_after_fixture"] = module_closure(freeze)
        report["state"] = "completed"
    except BaseException as error:
        report.update(state="incomplete" if isinstance(error, BudgetExpired) else "failed",
                      error_type=type(error).__name__, error=str(error))
        if "cell" in locals() and cell is not None:
            cell["provider_calls"] = audited.calls[cell.pop("calls_start"):]
            cell["provider_call_summary"] = summarize_calls(cell["provider_calls"])
            cell["family_timings"] = families.snapshot() if families is not None else None
            report["failed_or_partial_cell"] = cell
        raise
    finally:
        model, optimizer, scaler = None, None, None
        if families is not None:
            errors = families.restore()
            report["family_timing_wrappers_restored"] = not errors
            if errors:
                report.update(state="failed", family_timing_restore_errors=errors)
        gc.collect()
        if provider is not None:
            try:
                provider.close()
            except BaseException as cleanup_error:
                report.update(state="failed", provider_cleanup_error=str(cleanup_error))
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except BaseException as cleanup_error:
                report["cuda_cleanup_error"] = str(cleanup_error)
        report["worker_wall_seconds"] = time.perf_counter() - budget.started
        report["peak_rss_bytes"] = peak_rss()
        write(args.output / "report.json", report)
        write(args.output / "completion.json", {"state": report["state"], "purpose": "disposable_execution_not_quality",
            "scope_id": SCOPE_ID, "full_package_admitted": False, "full_row_fit_admitted": True,
            "full_fit": False, "test_accessed": False, "cells_completed": len(report["cells"]),
            "artifacts": ["report.json", "fixture.json"], "sha256": {name: sha(args.output / name) for name in
                ("report.json", "fixture.json") if (args.output / name).exists()}})


if __name__ == "__main__":
    main()
