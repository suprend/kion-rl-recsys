"""One bounded local-only E5 inference job for three independent catalog fields.

The CLI runs only inside the reviewed native supervisor. It loads one frozen
float32 model on CPU, checks actual CPU/CUDA vectors, then reuses that same model
for title, title_orig and description. It never fits or writes model weights.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

import kion_text_embeddings as text

CONTRACT = "kion-e5-three-field-inference-v1"
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "numpy")
SOURCES = ("scripts/kion_encode_text_bundle.py", "scripts/kion_text_embeddings.py",
           "scripts/kion_night_supervisor.py", "scripts/hm_job.py")
PARITY_ATOL = 1e-4
NIGHT_ID = "kion-all-agreed-e5-20261020-r3"
JOB_ID = "e5-three-fields"
PUBLICATION_RETRY_SECONDS = 1.0
PUBLICATION_RETRY_INTERVAL = .02
PUBLICATION_MAX_ATTEMPTS = 51
PUBLICATION_POLICY = {"retry_winerrors": [5, 32, 33], "max_seconds": PUBLICATION_RETRY_SECONDS,
                      "max_attempts": PUBLICATION_MAX_ATTEMPTS, "interval_seconds": PUBLICATION_RETRY_INTERVAL,
                      "stop_and_deadline_checked_each_attempt": True, "persistent_failure": "raise"}
PARITY_STRINGS = (
    "query: Космический корабль отправляется к далёкой планете.",
    "query: A young woman learns to play the piano.",
    "query: " + "История космоса и путешествий. " * 220,
)


def stamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path, value, *, check=None, retry_seconds=PUBLICATION_RETRY_SECONDS,
                monotonic=None, sleep=None):
    """Retry only transient Windows replace conflicts within a finite budget.

    A reader without FILE_SHARE_DELETE can deny atomic replacement temporarily.
    The existing target remains intact until replacement succeeds. Stop/deadline
    checks run before every attempt and again after each bounded wait.
    """
    if (isinstance(retry_seconds, bool) or not isinstance(retry_seconds, (int, float))
            or not math.isfinite(retry_seconds) or not 0 < retry_seconds <= PUBLICATION_RETRY_SECONDS):
        raise ValueError("Publication retry budget must be finite, positive, and at most one second")
    monotonic, sleep = monotonic or time.monotonic, sleep or time.sleep
    check = check or (lambda: None)
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    deadline = monotonic() + retry_seconds
    check()
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(PUBLICATION_MAX_ATTEMPTS):
        check()
        try:
            os.replace(temporary, path)
            return
        except OSError as error:
            remaining = deadline - monotonic()
            if (getattr(error, "winerror", None) not in (5, 32, 33) or remaining <= 0
                    or attempt + 1 >= PUBLICATION_MAX_ATTEMPTS):
                raise
            check()
            sleep(min(PUBLICATION_RETRY_INTERVAL, remaining))


def versions():
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    return {name: text.file_hash(root / name) for name in SOURCES}


def production_spec(model_dir):
    return {**text.validate_local_model(model_dir), "compute_dtype": "float32",
            "device": "cuda", "pooling_dtype": "float32", "threads": 2,
            "library_versions": versions()}


def make_request(items, metadata, model_dir, output_dir, *, encoder_spec=None):
    """CPU-only planning: read/hash files and metadata; never import torch."""
    items, metadata, model_dir, output_dir = map(lambda p: Path(p).resolve(),
                                                (items, metadata, model_dir, output_dir))
    spec = production_spec(model_dir) if encoder_spec is None else encoder_spec
    fields = []
    for field in text.FIELDS:
        plan = text.plan_catalog(items, metadata, field)
        signature = text.canonical_hash(text._descriptor(plan, spec))
        fields.append({"field": field, "signature": signature,
                       "vocab_size": plan["vocab_size"],
                       "vocabulary_sha256": plan["vocabulary_sha256"],
                       "normalized_text_sha256": plan["normalized_text_sha256"],
                       "present": sum(bool(value) for value in plan["texts"]),
                       "missing": plan["empty_counts"]["normalized_empty"]})
    inputs = {path.as_posix(): text.file_hash(path) for path in (items, metadata)}
    for name in text.MODEL_FILES:
        path = model_dir / name
        inputs[path.as_posix()] = text.file_hash(path)
    runtime = {"interpreter": Path(sys.executable).resolve().as_posix(),
               "base_interpreter": Path(sys._base_executable).resolve().as_posix()}
    runtime.update({key + "_sha256": text.file_hash(path) for key, path in list(runtime.items())})
    request = {"contract": CONTRACT, "items": items.as_posix(), "metadata": metadata.as_posix(),
               "model_dir": model_dir.as_posix(), "output_dir": output_dir.as_posix(),
               "fields": fields, "encoder": spec, "batch_size": 16, "max_seconds": 1800,
               "source_hashes": source_hashes(), "input_hashes": inputs,
               "library_versions_sha256": text.canonical_hash(spec["library_versions"]),
               "runtime": runtime,
               "publication_policy": PUBLICATION_POLICY,
               "python_version": platform.python_version(), "parity_atol": PARITY_ATOL,
               "parity_fixture_sha256": text.canonical_hash(PARITY_STRINGS),
               "training": False, "optimizer": False, "trainable_weights": False}
    request["request_sha256"] = text.canonical_hash(request)
    return request


def verify_request(request, output):
    if request.get("request_sha256") != text.canonical_hash(
            {key: value for key, value in request.items() if key != "request_sha256"}):
        raise ValueError("Request signature mismatch")
    if (request.get("contract") != CONTRACT or request.get("batch_size") != 16
            or request.get("max_seconds") != 1800
            or [value["field"] for value in request["fields"]] != list(text.FIELDS)
            or request.get("output_dir") != Path(output).resolve().as_posix()
            or request.get("parity_atol") != PARITY_ATOL
            or request.get("publication_policy") != PUBLICATION_POLICY
            or request.get("parity_fixture_sha256") != text.canonical_hash(PARITY_STRINGS)
            or any(request.get(key) is not False for key in ("training", "optimizer", "trainable_weights"))):
        raise ValueError("Unsupported inference request")
    if request["source_hashes"] != source_hashes():
        raise ValueError("Inference source drift")
    for path, expected in request["input_hashes"].items():
        if text.file_hash(path) != expected:
            raise ValueError("Inference input drift: " + path)
    if (request["encoder"].get("library_versions") != versions()
            or request["library_versions_sha256"] != text.canonical_hash(versions())
            or request["python_version"] != platform.python_version()):
        raise ValueError("Inference runtime version drift")
    for name in ("interpreter", "base_interpreter"):
        actual = sys.executable if name == "interpreter" else sys._base_executable
        if (request["runtime"][name] != Path(actual).resolve().as_posix()
                or request["runtime"][name + "_sha256"] != text.file_hash(actual)):
            raise ValueError("Inference interpreter drift")


def parameter_snapshot(encoder):
    if encoder.model.training:
        raise ValueError("Encoder entered training mode")
    parameters = list(encoder.model.parameters())
    if not parameters or any(value.requires_grad for value in parameters):
        raise ValueError("Encoder weights must be frozen")
    return [(id(value), value._version) for value in parameters]


def move_to_cuda(encoder):
    torch = encoder.torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("Reviewed CUDA device unavailable")
    encoder.model.to("cuda").eval()
    encoder.device = "cuda"
    encoder.spec["device"] = "cuda"
    torch.cuda.synchronize()


def checked_vectors(vectors, lengths):
    vectors, lengths = np.asarray(vectors), np.asarray(lengths)
    if (vectors.shape != (3, text.DIMENSION) or vectors.dtype != np.float32
            or lengths.shape != (3,) or lengths.dtype.kind not in "iu"
            or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=text.NORM_ATOL)
            or np.any(lengths[:2] >= 512) or lengths[2] <= 512):
        raise ValueError("Invalid actual parity fixture vectors/lengths")
    return vectors, lengths


def native_parent_pid(supervisor, target):
    """Snapshot one process parent without invoking another process or WMI."""
    w = supervisor.W

    class Entry(ctypes.Structure):
        _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("pid", w.DWORD),
                    ("heap", ctypes.c_size_t), ("module", w.DWORD), ("threads", w.DWORD),
                    ("parent", w.DWORD), ("priority", w.LONG), ("flags", w.DWORD),
                    ("exe", w.WCHAR * 260)]
    handle = supervisor.winapi("CreateToolhelp32Snapshot", w.HANDLE, w.DWORD, w.DWORD)(2, 0)
    if handle == ctypes.c_void_p(-1).value:
        raise ValueError("Cannot inspect native process parent")
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        present = supervisor.winapi("Process32FirstW", w.BOOL, w.HANDLE, ctypes.POINTER(Entry))(
            handle, ctypes.byref(entry))
        while present:
            if entry.pid == target:
                return entry.parent
            present = supervisor.winapi("Process32NextW", w.BOOL, w.HANDLE, ctypes.POINTER(Entry))(
                handle, ctypes.byref(entry))
        raise ValueError("Native process disappeared during parent inspection")
    finally:
        supervisor.CLOSE_HANDLE(handle)


def verify_worker_lineage(attempt, current, launcher, owner, current_parent, launcher_parent, runtime):
    """Accept the exact launcher or its direct Python venv child; reject PID reuse."""
    if (launcher != attempt.get("identity") or launcher["pid"] != attempt.get("pid")
            or launcher["exe"] != runtime["interpreter"]
            or owner["pid"] != attempt.get("supervisor_pid") or launcher_parent != owner["pid"]
            or owner["pid"] == launcher["pid"]
            or owner["creation_time_100ns"] > launcher["creation_time_100ns"]):
        raise ValueError("Native launcher/supervisor identity or ancestry mismatch")
    if current["pid"] == launcher["pid"]:
        if current != launcher:
            raise ValueError("Native launcher PID was reused")
    elif (current_parent != launcher["pid"] or current["exe"] != runtime["base_interpreter"]
          or current["creation_time_100ns"] < launcher["creation_time_100ns"]):
        raise ValueError("Worker is not the exact direct Python child of the native launcher")


def verify_native_owner(request_path, request, output, stop_file):
    """Require this exact live supervisor attempt, named Job, and occupied lease."""
    if (os.name != "nt" or os.environ.get("KION_OUTPUT_DIR") != output.resolve().as_posix()
            or os.environ.get("KION_STOP_FILE") != stop_file.resolve().as_posix()
            or stop_file.resolve() != output.resolve() / "stop-request.json"):
        raise ValueError("Native supervisor-owned output/stop paths required")
    path = Path(__file__).with_name("kion_night_supervisor.py")
    spec = importlib.util.spec_from_file_location("kion_text_frozen_supervisor", path)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    lock_root = supervisor.default_lock_root()
    ledger_path = lock_root / ("night-" + NIGHT_ID + ".ledger.json")
    ledger = supervisor.read_json(ledger_path)
    if (ledger.get("night_id") != NIGHT_ID or ledger.get("source_hashes") != request["source_hashes"]
            or ledger.get("input_hashes", {}).get(request_path.resolve().as_posix()) != text.file_hash(request_path)
            or ledger.get("budget", {}).get("gpu_uuid") != os.environ.get("CUDA_VISIBLE_DEVICES")):
        raise ValueError("Current native ledger does not bind this inference request")
    attempts = [value for value in ledger.get("attempts", []) if value.get("id") == JOB_ID]
    if len(attempts) != 1:
        raise ValueError("Exact native inference attempt required")
    attempt = attempts[0]
    identity = supervisor.process_identity(os.getpid())
    if (attempt.get("state") not in ("starting", "running") or attempt.get("kind") != "eval"
            or attempt.get("output") != output.resolve().as_posix()
            or attempt.get("cwd") != Path(__file__).resolve().parents[1].as_posix()
            or attempt.get("source_hashes") != request["source_hashes"]):
        raise ValueError("Process is not the exact active reviewed inference worker")
    launcher = supervisor.process_identity(attempt["pid"])
    owner = supervisor.process_identity(attempt["supervisor_pid"])
    verify_worker_lineage(attempt, identity, launcher, owner, os.getppid(),
                          native_parent_pid(supervisor, launcher["pid"]), request["runtime"])
    for name in ("deadline_utc", "drain_utc"):
        if supervisor.utc_now() >= supervisor.instant(ledger["budget"][name]):
            raise ValueError("Native inference deadline expired")
    handle = supervisor.OPEN_JOB(4, False, attempt["job_name"])
    if not handle:
        raise ValueError("Native inference Job unavailable")
    try:
        belongs = supervisor.W.BOOL()
        is_member = supervisor.winapi("IsProcessInJob", supervisor.W.BOOL, supervisor.W.HANDLE,
                                     supervisor.W.HANDLE, ctypes.POINTER(supervisor.W.BOOL))
        current = supervisor.winapi("GetCurrentProcess", supervisor.W.HANDLE)()
        if not is_member(current, handle, ctypes.byref(belongs)) or not belongs.value:
            raise ValueError("Inference worker is outside the supervisor's named Job")
    finally:
        supervisor.CLOSE_HANDLE(handle)
    gpu = ledger["budget"]["gpu_uuid"]
    key = hashlib.sha256(gpu.encode()).hexdigest()
    marker = supervisor.read_json(lock_root / ("gpu-" + key + ".owner.json"))
    if (marker.get("night_id") != NIGHT_ID or marker.get("job_id") != JOB_ID
            or marker.get("job_name") != attempt["job_name"]):
        raise ValueError("Native GPU ownership marker does not bind this exact Job")
    lease = lock_root / ("gpu-" + key + ".lock")
    opened = supervisor.CREATE_FILE(str(lease), 0xC0000000, 0, None, 3, 0x80, None)
    if opened != ctypes.c_void_p(-1).value:
        supervisor.CLOSE_HANDLE(opened)
        raise ValueError("Native GPU lease is unoccupied")
    if ctypes.get_last_error() != 32:
        raise ValueError("Native GPU lease state is unknown")
    return {"night_id": NIGHT_ID, "job_id": JOB_ID, "gpu_uuid": gpu, "pid": os.getpid(),
            "launcher_identity": launcher, "worker_identity": identity, "supervisor_identity": owner,
            "job_name": attempt["job_name"], "named_job_member": True, "GPU_lease_occupied": True,
            "ledger_path": ledger_path.as_posix(), "verified_utc": stamp()}


class ProgressEncoder:
    def __init__(self, encoder, check, update, total):
        self.encoder, self.spec = encoder, encoder.spec
        self.check, self.update, self.total, self.done = check, update, total, 0

    def encode(self, strings):
        self.check()
        result = self.encoder.encode(strings)
        self.check()
        self.done += len(strings)
        self.update(self.done, self.total)
        return result


def build_text_bundle(request, output_dir, *, stop_file=None, max_seconds=1800,
                      encoder_factory=None, cuda_mover=None, monotonic=time.monotonic,
                      native_receipt=None):
    """Execute one request. Explicit encoder/mover injection exists only for CPU tests."""
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1800):
        raise ValueError("max_seconds must be finite and at most 1800")
    output = Path(output_dir).resolve()
    started = monotonic()
    state = {"contract": CONTRACT, "state": "starting", "started_utc": stamp(),
             "request_sha256": request["request_sha256"], "fields": {}}

    def check():
        if monotonic() - started >= max_seconds:
            raise TimeoutError("Inference wall-time cap reached")
        if stop_file is not None and Path(stop_file).exists():
            raise InterruptedError("Native supervisor requested stop")

    def publish():
        state.update(updated_utc=stamp(), elapsed_seconds=monotonic() - started)
        atomic_json(output / "progress.json", state, check=check, monotonic=monotonic)

    check()
    verify_request(request, output)
    if output.exists() and any((output / name).exists() for name in
                               (*text.FIELDS, "progress.json", "report.json", "completion.json")):
        raise FileExistsError("Inference output already contains a prior attempt")
    output.mkdir(parents=True, exist_ok=True)  # Supervisor owns this newly created directory.
    publish()
    try:
        if encoder_factory is None and request["encoder"] != production_spec(request["model_dir"]):
            raise ValueError("Production encoder request mismatch")
        if encoder_factory is None and native_receipt is None:
            raise ValueError("Verified native inference ownership required")
        check()
        encoder = (encoder_factory or text.E5Encoder)(request["model_dir"], device="cpu",
                                                      compute_dtype="float32", threads=2)
        check()
        cpu, cpu_lengths = checked_vectors(*encoder.encode(list(PARITY_STRINGS)))
        check()
        (cuda_mover or move_to_cuda)(encoder)
        if encoder.spec != request["encoder"] or encoder.device != "cuda":
            raise ValueError("Actual CUDA encoder spec differs from request")
        frozen = parameter_snapshot(encoder)
        check()
        cuda, cuda_lengths = checked_vectors(*encoder.encode(list(PARITY_STRINGS)))
        check()
        error = float(np.max(np.abs(cpu - cuda)))
        if not np.array_equal(cpu_lengths, cuda_lengths) or error > PARITY_ATOL:
            raise ValueError("Actual CPU/CUDA parity failed")
        parity = {"state": "passed", "rows": 3, "max_abs_error": error, "atol": PARITY_ATOL,
                  "CPU_vectors_sha256": text.array_hash(cpu), "CUDA_vectors_sha256": text.array_hash(cuda),
                  "token_lengths": cpu_lengths.tolist(), "fixture_sha256": request["parity_fixture_sha256"]}
        state.update(state="encoding", parity=parity)
        publish()
        fields, artifacts = {}, ["report.json"]
        for expected in request["fields"]:
            field = expected["field"]
            check()
            state["fields"][field] = {"state": "encoding", "done": 0, "total": expected["present"]}
            publish()

            def update(done, total):
                state["fields"][field].update(done=done, total=total)
                publish()

            wrapped = ProgressEncoder(encoder, check, update, expected["present"])
            cache = text.build_item_cache(request["items"], request["metadata"], field,
                                         output / field, wrapped, batch_size=16)
            check()
            if (cache.manifest["signature"] != expected["signature"]
                    or cache.manifest["coverage"]["present"] != expected["present"]
                    or cache.manifest["coverage"]["missing"] != expected["missing"]):
                raise ValueError("Field cache request mismatch: " + field)
            directory = output / field / expected["signature"]
            # Reload verifies disk hashes and finite/unit/missing invariants.
            text.load_item_cache(directory, expected_signature=expected["signature"],
                                 allow_test_encoder=encoder_factory is not None)
            prefix = directory.relative_to(output).as_posix()
            artifacts.extend(prefix + "/" + name for name in
                             ("embeddings.npy", "missing.npy", "token_lengths.npy", "truncated.npy", "manifest.json"))
            fields[field] = {"relative_dir": prefix, "signature": expected["signature"],
                             "coverage": cache.manifest["coverage"],
                             "all_arrays_finite": True, "unit_norms_and_missing_valid": True}
            state["fields"][field].update(state="completed")
            publish()
        if parameter_snapshot(encoder) != frozen:
            raise ValueError("Frozen parameters changed during inference")
        verify_request(request, output)
        check()
        report = {"contract": CONTRACT, "state": "completed", "request_sha256": request["request_sha256"],
                  "fields": fields, "field_count": 3, "cache_file_count": 15,
                  "encoder": encoder.spec, "library_versions_sha256": request["library_versions_sha256"],
                  "source_hashes": request["source_hashes"], "input_hashes": request["input_hashes"],
                  "CPU_CUDA_parity": parity, "all_arrays_finite": True,
                  "native_ownership": native_receipt,
                  "publication_policy": PUBLICATION_POLICY,
                  "unit_norms_and_missing_valid": True, "parameters_frozen_and_unchanged": True,
                  "training": False, "optimizer": False, "trainable_weights": False,
                  "learned_checkpoint_written": False, "histories_or_outcomes_accessed": False,
                  "model_load_count": 1, "device": "cuda", "compute_dtype": "float32", "batch_size": 16,
                  "started_utc": state["started_utc"], "finished_utc": stamp(),
                  "elapsed_seconds": monotonic() - started}
        atomic_json(output / "report.json", report, check=check, monotonic=monotonic)
        receipt = {key: report[key] for key in ("contract", "state", "field_count", "cache_file_count",
                   "all_arrays_finite", "unit_norms_and_missing_valid", "parameters_frozen_and_unchanged",
                   "training", "optimizer", "trainable_weights", "learned_checkpoint_written",
                   "histories_or_outcomes_accessed", "model_load_count", "device", "compute_dtype", "batch_size")}
        receipt.update(request_sha256=request["request_sha256"], artifacts=artifacts,
                       sha256={name: text.file_hash(output / name) for name in artifacts})
        check()
        state.update(state="completed")
        publish()
        atomic_json(output / "completion.json", receipt, check=check, monotonic=monotonic)
        return report
    except Exception as error:
        state.update(state="failed", error_type=type(error).__name__, error=str(error))
        try:
            publish()
        except Exception as publication_error:
            error.add_note("Failure-state publication also failed: " + type(publication_error).__name__
                           + ": " + str(publication_error))
            if publication_error is not error:
                raise error from publication_error
            raise error
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    if (os.environ.get("KION_OUTPUT_DIR") != args.output_dir.resolve().as_posix()
            or os.environ.get("KION_STOP_FILE") != args.stop_file.resolve().as_posix()):
        parser.error("Production worker requires native supervisor-owned output/stop paths")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    native_receipt = verify_native_owner(args.request, request, args.output_dir, args.stop_file)
    report = build_text_bundle(request, args.output_dir, stop_file=args.stop_file,
                               max_seconds=args.max_seconds, native_receipt=native_receipt)
    print(json.dumps({"state": report["state"], "field_count": report["field_count"],
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
