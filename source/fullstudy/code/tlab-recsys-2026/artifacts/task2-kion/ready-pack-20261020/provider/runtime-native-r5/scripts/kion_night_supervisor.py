"""Finite, reviewed KION queue with native Windows process ownership.

No inference, recovery policy, downloads, or model code lives here. A retry or
resume is another fully specified job in the approved plan. Run --help for the
small CLI; approval binds the canonical JSON bytes returned by plan-hash.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

SCHEMA = 1
TERMINAL = {"completed", "failed", "stopped", "skipped_dependency"}


class Refused(RuntimeError):
    """Fail closed before admitting another writer."""


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise Refused(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(Refused(f"Nonfinite JSON: {value}")))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now():
    return datetime.now(timezone.utc)


def stamp():
    return utc_now().isoformat().replace("+00:00", "Z")


def instant(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset().total_seconds() != 0:
        raise Refused("Deadlines must explicitly use UTC")
    return result


def number(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise Refused(f"Invalid {name}")
    return value


def local_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise Refused("Expected a relative path inside its owner directory")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise Refused("Path escapes its owner directory")
    return result


def absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise Refused(f"Expected absolute path: {value}")
    return path.resolve()


def validate_plan(plan, approval, *, cpu_fixtures=False):
    if plan.get("schema") != SCHEMA or approval.get("schema") != SCHEMA:
        raise Refused("Unsupported schema")
    if approval.get("verdict") != "approved" or not approval.get("reviewer") or not approval.get("evidence"):
        raise Refused("An independent approved review receipt with evidence is required")
    if approval.get("plan_sha256") != canonical_hash(plan):
        raise Refused("Review receipt does not bind these plan bytes")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", plan.get("night_id", "")):
        raise Refused("Invalid night ID")
    if instant(plan["drain_utc"]) >= instant(plan["deadline_utc"]):
        raise Refused("Drain must precede deadline")
    gpu = plan["gpu_uuid"]
    if not (re.fullmatch(r"GPU-[A-Fa-f0-9-]+", gpu) or (cpu_fixtures and gpu == "CPU-fixture")):
        raise Refused("Physical NVIDIA UUID required; CPU-fixture needs explicit fixture mode")
    # The approved plan keeps its exact immutable budget. A later, separately
    # authorized package + seven ablations may need eight scientific starts,
    # one plumbing smoke and one technical retry; old frozen plans stay at six.
    for name, maximum in (("train_cap", 10), ("eval_cap", 10), ("retry_cap", 2)):
        value = plan[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
            raise Refused(f"{name} exceeds the supported execution contract")
    bundle = absolute(plan["bundle_dir"])
    interpreter = absolute(plan["interpreter"])
    if not bundle.is_dir() or not interpreter.is_file():
        raise Refused("Missing source bundle or interpreter")
    hashes = plan["source_hashes"]
    if not hashes or "scripts/hm_job.py" not in hashes or "scripts/kion_night_supervisor.py" not in hashes:
        raise Refused("Bundle must include supervisor and owned Job helper")
    for relative in hashes:
        local_path(bundle, relative)
    if not cpu_fixtures and Path(__file__).resolve() != bundle / "scripts/kion_night_supervisor.py":
        raise Refused("Execute the supervisor from the reviewed frozen bundle")
    for path in plan["input_hashes"]:
        absolute(path)
    seen, outputs = set(plan.get("prior_jobs", {})), []
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or len(jobs) > 16:
        raise Refused("The queue must be a finite list of at most 16 jobs")
    counts = {"train": 0, "eval": 0, "retry": 0}
    for job in jobs:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job["id"]) or job["id"] in seen:
            raise Refused("Invalid or duplicate job ID")
        if job["kind"] not in ("train", "eval"):
            raise Refused("Every start, including smoke/resume, must be train or eval")
        counts[job["kind"]] += 1
        dependencies = job.get("depends_on", [])
        if not isinstance(dependencies, list) or not set(dependencies) <= seen:
            raise Refused("Dependencies must refer to earlier reviewed jobs")
        retry = job.get("retry_of")
        if retry is not None:
            if retry not in seen:
                raise Refused("Retry must name an earlier reviewed job")
            counts["retry"] += 1
        seen.add(job["id"])
        output = absolute(job["output"])
        for other in outputs + [bundle, *map(Path, plan["input_hashes"])]:
            if output == other or output.is_relative_to(other) or other.is_relative_to(output):
                raise Refused("Outputs must be unique and separate from sources and inputs")
        outputs.append(output)
        args = job["argv"]
        if not isinstance(args, list) or not args or any(not isinstance(x, str) for x in args):
            raise Refused("argv must be a nonempty string array")
        if args[0] not in hashes or not args[0].endswith(".py"):
            raise Refused("The executable script must be in the frozen source hash list")
        for arg in args:
            if "{" in arg.replace("{output}", "").replace("{stop_file}", ""):
                raise Refused("Only output and stop_file placeholders are supported")
        for name in ("estimate_seconds", "max_seconds", "stop_grace_seconds", "cleanup_reserve_seconds"):
            number(job[name], name, .01)
        if not cpu_fixtures and job["cleanup_reserve_seconds"] < 12:
            raise Refused("Reserve at least 12 seconds for owned Job cleanup")
        for name in ("disk_reserve_bytes", "disk_required_bytes", "log_cap_bytes"):
            if not isinstance(job[name], int) or isinstance(job[name], bool) or job[name] < 1:
                raise Refused(f"Positive measured {name} required")
        completion = job["completion"]
        local_path(output, completion["receipt"])
        checks = completion.get("equals", {})
        smoke = job.get("purpose") == "plumbing-smoke"
        if smoke:
            cap = job.get("max_updates")
            if (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0
                    or checks.get("state") != "stopped" or checks.get("reason") != "runtime_max_updates"
                    or checks.get("validation_rows") != 0 or checks.get("test_accessed") is not False
                    or type(checks.get("validation_rows")) is not int
                    or type(checks.get("updates")) is not int or checks.get("updates") != cap):
                raise Refused("Plumbing smoke requires an exact positive update cap and explicit stopped evidence")
        if not completion.get("artifacts") or (not smoke and checks.get("state") != "completed"):
            raise Refused("Completion needs artifacts and explicit receipt checks")
        for path in completion["artifacts"]:
            local_path(output, path)
    if any(counts[key] > plan[key + "_cap"] for key in counts):
        raise Refused("Finite job list exceeds a reviewed cap")
    return plan


def verify_frozen(plan):
    bundle = absolute(plan["bundle_dir"])
    for relative, expected in plan["source_hashes"].items():
        path = local_path(bundle, relative)
        if not path.is_file() or digest(path) != expected:
            raise Refused(f"Frozen source mismatch: {relative}")
    extra = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*.py")} - set(plan["source_hashes"])
    if extra:
        raise Refused(f"Unreviewed Python modules in bundle: {sorted(extra)}")
    if digest(plan["interpreter"]) != plan["interpreter_sha256"]:
        raise Refused("Interpreter bytes changed")
    for path, expected in plan["input_hashes"].items():
        if digest(path) != expected:
            raise Refused(f"Frozen input mismatch: {path}")


if os.name == "nt":
    from ctypes import wintypes as W
    KERNEL = ctypes.WinDLL("kernel32", use_last_error=True)

    def winapi(name, result, *arguments):
        function = getattr(KERNEL, name)
        function.restype, function.argtypes = result, arguments
        return function

    CREATE_FILE = winapi("CreateFileW", W.HANDLE, W.LPCWSTR, W.DWORD, W.DWORD, W.LPVOID, W.DWORD, W.DWORD, W.HANDLE)
    CLOSE_HANDLE = winapi("CloseHandle", W.BOOL, W.HANDLE)
    OPEN_JOB = winapi("OpenJobObjectW", W.HANDLE, W.DWORD, W.BOOL, W.LPCWSTR)
    OPEN_PROCESS = winapi("OpenProcess", W.HANDLE, W.DWORD, W.BOOL, W.DWORD)
    PROCESS_TIMES = winapi("GetProcessTimes", W.BOOL, W.HANDLE, W.LPVOID, W.LPVOID, W.LPVOID, W.LPVOID)
    PROCESS_IMAGE = winapi("QueryFullProcessImageNameW", W.BOOL, W.HANDLE, W.DWORD, W.LPWSTR, ctypes.POINTER(W.DWORD))


class FileLock:
    """Exclusive OS-held handle; metadata is deliberately separate from it."""
    def __init__(self, path):
        self.path, self.handle = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            handle = CREATE_FILE(str(self.path), 0xC0000000, 0, None, 4, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise Refused(f"Busy or inaccessible OS lock: {self.path}")
            self.handle = handle
            os.set_handle_inheritable(handle, True)
        else:
            import fcntl
            self.handle = self.path.open("a+b")
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                self.handle.close()
                raise Refused(f"Busy OS lock: {self.path}") from error
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            CLOSE_HANDLE(self.handle) if os.name == "nt" else self.handle.close()
            self.handle = None


@contextmanager
def publication_lock(path, deadline, *, wait_seconds=2):
    """Allow brief writer contention without extending the cleanup deadline.

    Admission intentionally remains nonblocking. A worker already admitted to
    the queue gets a bounded opportunity to publish while another plan finishes
    its short atomic ledger update. An expired deadline permits no waiting.
    """
    stop = min(deadline, time.monotonic() + wait_seconds)
    while True:
        lock = FileLock(path)
        try:
            lock.__enter__()
            break
        except Refused as error:
            remaining = stop - time.monotonic()
            if remaining <= 0:
                raise Refused("Queue publication lock wait exhausted before cleanup deadline") from error
            time.sleep(min(.02, remaining))
    try:
        yield lock
    finally:
        lock.__exit__(None, None, None)


def process_identity(pid):
    if os.name != "nt":
        raise Refused("Process identity receipts require native Windows")
    handle = OPEN_PROCESS(0x1000, False, pid)
    if not handle:
        raise Refused(f"Cannot inspect process {pid}")
    try:
        values = [ctypes.c_uint64() for _ in range(4)]
        if not PROCESS_TIMES(handle, *(ctypes.byref(value) for value in values)):
            raise ctypes.WinError(ctypes.get_last_error())
        buffer, size = ctypes.create_unicode_buffer(32768), W.DWORD(32768)
        if not PROCESS_IMAGE(handle, 0, buffer, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"pid": pid, "exe": Path(buffer.value).resolve().as_posix(), "creation_time_100ns": values[0].value}
    finally:
        CLOSE_HANDLE(handle)


def protected_dwm_identity(pid):
    """Explicit reviewed exception for an inaccessible desktop compositor PID.

    This intentionally is weaker than a creation-time identity. The exception
    is recorded, applies only to this PID, and grants no process control.
    """
    if os.name != "nt":
        return False
    handle = OPEN_PROCESS(0x1000, False, pid)
    if handle:
        CLOSE_HANDLE(handle)
        return False
    if ctypes.get_last_error() != 5:
        return False
    result = subprocess.run(["tasklist.exe", "/FI", f"PID eq {pid}", "/V", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    rows = list(csv.reader(io.StringIO(result.stdout)))
    return (result.returncode == 0 and len(rows) == 1 and len(rows[0]) >= 4
            and rows[0][0].lower() == "dwm.exe" and rows[0][1] == str(pid)
            and rows[0][2] == "Console")


def gpu_quiet(plan, cpu_fixtures):
    if cpu_fixtures and plan["gpu_uuid"] == "CPU-fixture":
        return []
    def query(fields):
        result = subprocess.run(["nvidia-smi", *fields, "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, encoding="utf-8", timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            # NVIDIA output is diagnostic data; retain both streams without an
            # environment dump or an unbounded driver error in the ledger.
            stdout = json.dumps(result.stdout or "", ensure_ascii=True)[:400]
            stderr = json.dumps(result.stderr or "", ensure_ascii=True)[:400]
            raise Refused(f"NVIDIA identity/consumer inspection failed: returncode={result.returncode}; "
                          f"stdout={stdout}; stderr={stderr}")
        return result.stdout.splitlines()
    if plan["gpu_uuid"] not in [line.strip() for line in query(["--query-gpu=uuid"])]:
        raise Refused("Reviewed physical GPU is absent")
    allowed = {item["pid"]: item for item in plan.get("benign_gpu_processes", [])}
    exceptions = []
    for line in query(["--query-compute-apps=gpu_uuid,pid"]):
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise Refused("Ambiguous NVIDIA consumer record")
        if fields[0] == plan["gpu_uuid"]:
            try:
                pid = int(fields[1])
                if pid == plan.get("protected_dwm_pid") and protected_dwm_identity(pid):
                    exceptions.append({"pid": pid, "image": "dwm.exe", "identity_verified": False,
                                       "warning": "reviewed protected desktop exception; creation time unavailable"})
                    continue
                identity = process_identity(pid)
            except (ValueError, OSError) as error:
                raise Refused("Ambiguous GPU process identity") from error
            if allowed.get(pid) != identity:
                raise Refused(f"Foreign GPU consumer blocks admission: {pid}")
    return exceptions


def job_helper(plan):
    path = absolute(plan["bundle_dir"]) / "scripts/hm_job.py"
    spec = importlib.util.spec_from_file_location("kion_frozen_hm_job", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_quiescent(marker, helper):
    if not marker.exists() or os.name != "nt":
        return
    previous = read_json(marker)
    handle = OPEN_JOB(4, False, previous["job_name"])
    if not handle:
        if ctypes.get_last_error() == 2:
            return  # Object no longer exists; no associated process remains.
        raise Refused("Cannot establish previous Job quiescence")
    try:
        account = helper._Accounting()
        if not helper._query_job(handle, 1, ctypes.byref(account), ctypes.sizeof(account), None):
            raise Refused("Cannot query previous Job ownership")
        if account.ActiveProcesses:
            raise Refused("Previous owned Job has not become quiescent")
    finally:
        CLOSE_HANDLE(handle)


def new_named_job(helper, name):
    if os.name != "nt":
        return None
    handle = helper._create_job(None, name)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = helper._ExtendedLimits()
    limits.Basic.Flags = 0x2000
    if not helper._set_job(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        helper._close(handle)
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def admission_reason(plan, job, now, counts):
    if now >= instant(plan["drain_utc"]):
        return "drain_started"
    if now.timestamp() + job["estimate_seconds"] + job["cleanup_reserve_seconds"] > instant(plan["drain_utc"]).timestamp():
        return "insufficient_time_before_drain"
    if counts[job["kind"]] >= plan[job["kind"] + "_cap"]:
        return "start_cap"
    if job.get("retry_of") and counts["retry"] >= plan["retry_cap"]:
        return "retry_cap"
    return None


def nested(value, key):
    for part in key.split("."):
        value = value[part]
    return value


def completion_evidence(job):
    output = absolute(job["output"])
    contract = job["completion"]
    receipt_path = local_path(output, contract["receipt"])
    value = read_json(receipt_path)
    for key, expected in contract["equals"].items():
        actual = nested(value, key)
        if type(actual) is not type(expected) or actual != expected:
            raise Refused(f"Completion mismatch: {key}")
    for key in contract.get("finite", []):
        number(nested(value, key), f"completion {key}", -float("inf"))
    declarations, declared_hashes = value.get("artifacts"), value.get("sha256")
    if (not isinstance(declarations, list) or not declarations or any(not isinstance(x, str) for x in declarations)
            or len(set(declarations)) != len(declarations) or not isinstance(declared_hashes, dict)
            or set(declared_hashes) != set(declarations)):
        raise Refused("Completion needs exact artifact and SHA-256 declarations")
    hashes, declared_paths = {}, set()
    for relative in declarations:
        path = local_path(receipt_path.parent, relative)
        expected = declared_hashes[relative]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise Refused(f"Invalid declared artifact hash: {relative}")
        if not path.is_file() or path.stat().st_size == 0:
            raise Refused(f"Missing committed artifact: {relative}")
        actual = digest(path)
        if actual != expected:
            raise Refused(f"Committed artifact SHA-256 mismatch: {relative}")
        declared_paths.add(path)
        hashes[path.relative_to(output).as_posix()] = actual
    for relative in contract["artifacts"]:
        if local_path(output, relative) not in declared_paths:
            raise Refused(f"Required artifact has no committed hash declaration: {relative}")
    hashes[contract["receipt"]] = digest(receipt_path)
    return {"receipt": value, "artifact_hashes": hashes}


class BoundedLogs:
    def __init__(self, output, cap):
        self.cap, self.total = cap, 0
        self.limit = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.output = output
        self.threads = []

    def copy(self, pipe, name):
        try:
            with (self.output / name).open("wb") as stream:
                while block := pipe.read(8192):
                    with self.lock:
                        remaining = max(0, self.cap - self.total)
                        stream.write(block[:remaining])
                        self.total += min(len(block), remaining)
                        if len(block) >= remaining:
                            self.limit.set()
                    stream.flush()
        except (OSError, ValueError) as error:
            self.error = str(error)
            self.limit.set()
        finally:
            pipe.close()

    def start(self, process):
        for pipe, name in ((process.stdout, "stdout.log"), (process.stderr, "stderr.log")):
            thread = threading.Thread(target=self.copy, args=(pipe, name), daemon=True)
            thread.start()
            self.threads.append(thread)

    def finish(self):
        for thread in self.threads:
            thread.join(timeout=2)
            if thread.is_alive():
                raise Refused("Owned log pipe did not close after cleanup")


def execute(plan, job, attempt, locks, marker, helper, publish, hard_monotonic, drain_monotonic):
    output = absolute(job["output"])
    output.mkdir(parents=True, exist_ok=False)
    stop_file = output / "stop-request.json"
    args = [arg.replace("{output}", output.as_posix()).replace("{stop_file}", stop_file.as_posix()) for arg in job["argv"]]
    argv = [plan["interpreter"], *args]
    allowed_environment = {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "LOCALAPPDATA", "USERPROFILE",
                           "APPDATA", "PROGRAMDATA", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
                           "PROGRAMFILES", "PROGRAMW6432"}
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed_environment}
    environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       KION_STOP_FILE=stop_file.as_posix(), KION_OUTPUT_DIR=output.as_posix())
    if plan["gpu_uuid"] != "CPU-fixture":
        environment.update(CUDA_VISIBLE_DEVICES=plan["gpu_uuid"], CUDA_DEVICE_ORDER="PCI_BUS_ID")
    job_name = "Local\\KionNight-" + hashlib.sha256((plan["night_id"] + job["id"] + str(time.time_ns())).encode()).hexdigest()
    owned_job = new_named_job(helper, job_name)
    process = None
    logs = BoundedLogs(output, job["log_cap_bytes"])
    attempt.update(state="starting", argv=argv, cwd=plan["bundle_dir"], interpreter=plan["interpreter"],
                   job_name=job_name, start_utc=stamp(), supervisor_pid=os.getpid())
    started = time.monotonic()
    stop_at, reason, returncode = None, None, None
    try:
        atomic_json(marker, {"job_name": job_name, "night_id": plan["night_id"], "job_id": job["id"]})
        publish()
        kwargs = dict(cwd=plan["bundle_dir"], stdin=subprocess.DEVNULL,
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, close_fds=True)
        if os.name == "nt":
            info = subprocess.STARTUPINFO()
            info.lpAttributeList = {"handle_list": [lock.handle for lock in locks]}
            kwargs.update(creationflags=4 | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW,
                          startupinfo=info)
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(argv, **kwargs)
        attempt.update(pid=process.pid, identity=process_identity(process.pid) if os.name == "nt" else {"pid": process.pid})
        publish()
        if os.name == "nt":
            if not helper._assign_job(owned_job, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            helper._resume_initial_thread(process.pid)
        logs.start(process)
        attempt["state"] = "running"
        publish()
        next_heartbeat = started + 10
        while process.poll() is None:
            monotonic = time.monotonic()
            remaining = min((instant(plan["deadline_utc"]) - utc_now()).total_seconds(), hard_monotonic - monotonic)
            if remaining <= job["cleanup_reserve_seconds"]:
                reason = reason or "hard_deadline_cleanup"
                break
            trigger = None
            if utc_now() >= instant(plan["drain_utc"]) or monotonic >= drain_monotonic:
                trigger = "drain_started"
            elif monotonic - started >= job["max_seconds"]:
                trigger = "max_seconds"
            elif logs.limit.is_set():
                trigger = "log_write_failure" if logs.error else "log_cap"
            elif shutil.disk_usage(output).free < job["disk_reserve_bytes"]:
                trigger = "disk_reserve"
            if trigger and stop_at is None:
                reason, stop_at = trigger, monotonic
                atomic_json(stop_file, {"reason": reason, "requested_utc": stamp()})
                attempt.update(state="stopping", reason=reason)
                publish()
            if stop_at is not None and monotonic - stop_at >= job["stop_grace_seconds"]:
                break
            if monotonic >= next_heartbeat:
                attempt.update(elapsed_seconds=monotonic - started, deadline_remaining_seconds=max(0, remaining))
                publish()
                next_heartbeat = monotonic + 10
            time.sleep(.05)
        returncode = process.poll()
    finally:
        try:
            if process is not None:
                helper._cleanup(process, owned_job)
            elif owned_job is not None:
                helper._close(owned_job)
        finally:
            if process is not None and not logs.threads:
                process.stdout.close()
                process.stderr.close()
            logs.finish()
        attempt.update(end_utc=stamp(), elapsed_seconds=time.monotonic() - started,
                       returncode=returncode, log_bytes=logs.total, cleanup="quiescent")
    if reason:
        attempt.update(state="stopped", reason=reason)
    elif returncode != 0:
        attempt.update(state="failed", reason=f"exit_{returncode}")
    else:
        try:
            attempt.update(completion_evidence(job), state="completed",
                           reason="verified_plumbing_smoke" if job.get("purpose") == "plumbing-smoke" else "verified_completion")
        except (OSError, ValueError, KeyError, Refused) as error:
            attempt.update(state="failed", reason=f"completion_evidence: {error}")
    atomic_json(output / "supervisor-receipt.json", attempt)
    publish()


def default_lock_root():
    if os.name == "nt":
        return Path(os.environ["LOCALAPPDATA"]) / "KionNightSupervisor" / "locks"
    return Path.home() / ".local/state/kion-night-supervisor/locks"


def run(plan_path, approval_path, state_dir, *, cpu_fixtures=False, lock_root=None):
    plan, approval = read_json(plan_path), read_json(approval_path)
    validate_plan(plan, approval, cpu_fixtures=cpu_fixtures)
    if os.name != "nt" and not cpu_fixtures:
        raise Refused("Production ownership currently requires native Windows")
    if lock_root is not None and not (cpu_fixtures and plan["gpu_uuid"] == "CPU-fixture"):
        raise Refused("Production GPU locks use one fixed per-user OS location")
    lock_root = Path(lock_root) if lock_root else default_lock_root()
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "status.json"
    queue_path = lock_root / ("night-" + plan["night_id"] + ".queue.lock")
    ledger_path = lock_root / ("night-" + plan["night_id"] + ".ledger.json")
    budget = {key: plan[key] for key in ("night_id", "deadline_utc", "drain_utc", "gpu_uuid", "train_cap", "eval_cap", "retry_cap")}
    plan_hash = canonical_hash(plan)
    initial_wall, initial_monotonic = utc_now(), time.monotonic()
    hard_monotonic = initial_monotonic + max(0, (instant(plan["deadline_utc"]) - initial_wall).total_seconds())
    drain_monotonic = initial_monotonic + max(0, (instant(plan["drain_utc"]) - initial_wall).total_seconds())
    state = None
    queue_lock = None
    attempt = None

    def read_ledger(*, require_quiescent):
        if ledger_path.exists():
            value = read_json(ledger_path)
            if value.get("budget") != budget:
                raise Refused("A revised plan cannot reset or extend the night budget")
            if require_quiescent and any(a["state"] not in TERMINAL for a in value["attempts"]):
                raise Refused("Interrupted or live attempt requires explicit reconciliation; no blind restart")
            return value
        return {"schema": SCHEMA, "night_id": plan["night_id"], "plan_sha256": plan_hash,
                "budget": budget, "reviewed_plan_hashes": [],
                "source_hashes": plan["source_hashes"], "input_hashes": plan["input_hashes"],
                "attempts": [], "counts": {"train": 0, "eval": 0, "retry": 0}}

    def write_locked():
        state.update(updated_utc=stamp(), heartbeat_utc=stamp(), sequence=state.get("sequence", 0) + 1)
        atomic_json(ledger_path, state)
        atomic_json(state_path, state)

    def publish():
        nonlocal state
        if queue_lock is not None and queue_lock.handle is not None:
            write_locked()
            return
        # A worker owns only its claimed attempt. Other plans may finalize while
        # it runs, so merge that attempt under the same queue lock as admission.
        publication_deadline = hard_monotonic - job["cleanup_reserve_seconds"]
        with publication_lock(queue_path, publication_deadline):
            current = read_ledger(require_quiescent=False)
            if attempt is None:
                raise Refused("Unclaimed publication outside queue ownership")
            for index, existing in enumerate(current["attempts"]):
                if existing["id"] == attempt["id"]:
                    if existing["job_sha256"] != attempt["job_sha256"]:
                        raise Refused("Claimed attempt identity changed")
                    current["attempts"][index] = attempt
                    break
            else:
                raise Refused("Claimed attempt disappeared from authoritative ledger")
            state = current
            write_locked()

    for job in plan["jobs"]:
        with ExitStack() as stack:
            queue_lock = stack.enter_context(FileLock(queue_path))
            state = read_ledger(require_quiescent=True)
            previous = {attempt["id"]: attempt for attempt in state["attempts"]}
            for prior_id, prior_hash in plan.get("prior_jobs", {}).items():
                if previous.get(prior_id, {}).get("job_sha256") != prior_hash:
                    raise Refused("Prior job reference does not match the authoritative ledger")
            job_hash = canonical_hash({"job": job, **{key: plan[key] for key in (
                "bundle_dir", "source_hashes", "input_hashes", "interpreter", "interpreter_sha256")}})
            if job["id"] in previous:
                if previous[job["id"]].get("job_sha256") != job_hash:
                    raise Refused("Previously claimed job bytes cannot change")
                continue
            if plan_hash not in state["reviewed_plan_hashes"]:
                state["reviewed_plan_hashes"].append(plan_hash)
            state["plan_sha256"] = plan_hash
            dependencies = job.get("depends_on", [])
            retry = job.get("retry_of")
            dependency_failed = any(previous.get(dep, {}).get("state") != "completed" for dep in dependencies)
            retry_unneeded = retry and previous.get(retry, {}).get("state") not in ("failed", "stopped")
            if dependency_failed or retry_unneeded:
                state["attempts"].append({"id": job["id"], "job_sha256": job_hash, "state": "skipped_dependency", "reason": "dependency_gate"})
                publish()
                continue
            reason = admission_reason(plan, job, utc_now(), state["counts"])
            if not reason and time.monotonic() + job["estimate_seconds"] + job["cleanup_reserve_seconds"] > drain_monotonic:
                reason = "insufficient_monotonic_time_before_drain"
            if time.monotonic() >= hard_monotonic:
                reason = "hard_deadline"
            if reason:
                state["reason"] = reason
                publish()
                return state
            verify_frozen(plan)
            helper = job_helper(plan)
            gpu_key = hashlib.sha256(plan["gpu_uuid"].encode()).hexdigest()
            output_key = hashlib.sha256(os.path.normcase(str(absolute(job["output"]))).encode()).hexdigest()
            gpu_lock = stack.enter_context(FileLock(lock_root / ("gpu-" + gpu_key + ".lock")))
            output_lock = stack.enter_context(FileLock(lock_root / ("output-" + output_key + ".lock")))
            marker = lock_root / ("gpu-" + gpu_key + ".owner.json")
            check_quiescent(marker, helper)
            gpu_exceptions = gpu_quiet(plan, cpu_fixtures)
            output = absolute(job["output"])
            if output.exists():
                raise Refused("Attempt output already exists; immutable attempts are never overwritten")
            ancestor = output.parent
            while not ancestor.exists():
                ancestor = ancestor.parent
            if shutil.disk_usage(ancestor).free < job["disk_required_bytes"] + job["disk_reserve_bytes"] + job["log_cap_bytes"]:
                raise Refused("Insufficient disk for declared artifacts, temporary copies, logs and reserve")
            # Hashing and telemetry may take time: repeat admission immediately before claim.
            reason = admission_reason(plan, job, utc_now(), state["counts"])
            if not reason and time.monotonic() + job["estimate_seconds"] + job["cleanup_reserve_seconds"] > drain_monotonic:
                reason = "insufficient_monotonic_time_before_drain"
            if reason or time.monotonic() >= hard_monotonic:
                state["reason"] = reason or "hard_deadline"
                publish()
                return state
            attempt = {"id": job["id"], "kind": job["kind"], "state": "starting", "plan_sha256": plan_hash, "job_sha256": job_hash,
                       "purpose": job.get("purpose", "experiment"),
                       "gpu_consumer_exceptions": gpu_exceptions,
                       "output": job["output"], "source_hashes": plan["source_hashes"], "input_hashes": plan["input_hashes"]}
            state["attempts"].append(attempt)
            state["counts"][job["kind"]] += 1
            if retry:
                state["counts"]["retry"] += 1
            publish()
            # Release only the queue lock; GPU/output remain held until cleanup and receipts.
            queue_lock.__exit__(None, None, None)
            try:
                execute(plan, job, attempt, [gpu_lock, output_lock], marker, helper, publish, hard_monotonic, drain_monotonic)
            except BaseException as error:
                attempt.update(state="needs_action", reason=f"supervisor_error: {type(error).__name__}: {error}", updated_utc=stamp())
                publish()
                raise
    with FileLock(queue_path) as queue_lock:
        state = read_ledger(require_quiescent=False)
        # Finalization is metadata only; it cannot roll back another revision's
        # starts or replace the current worker's mutable progress snapshot.
        state["last_finished_plan_sha256"] = plan_hash
        if not any(a["state"] not in TERMINAL for a in state["attempts"]):
            state["reason"] = "finite_queue_finished" if plan["jobs"] else "empty_queue"
        else:
            state["reason"] = "another_reviewed_plan_running"
        publish()
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fingerprint = commands.add_parser("plan-hash")
    fingerprint.add_argument("--plan", required=True, type=Path)
    identity = commands.add_parser("inspect-process")
    identity.add_argument("--pid", required=True, type=int)
    for name in ("check", "run", "launch"):
        command = commands.add_parser(name)
        command.add_argument("--plan", required=True, type=Path)
        command.add_argument("--approval", required=True, type=Path)
        command.add_argument("--state-dir", required=True, type=Path)
        command.add_argument("--allow-cpu-fixtures", action="store_true")
        command.add_argument("--lock-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan-hash":
            print(canonical_hash(read_json(args.plan)))
        elif args.command == "inspect-process":
            print(json.dumps(process_identity(args.pid)))
        elif args.command == "check":
            plan = validate_plan(read_json(args.plan), read_json(args.approval), cpu_fixtures=args.allow_cpu_fixtures)
            verify_frozen(plan)
            print(json.dumps({"state": "checked", "plan_sha256": canonical_hash(plan)}))
        elif args.command == "launch":
            if os.name != "nt":
                raise Refused("Hidden detached launch requires native Windows")
            plan = validate_plan(read_json(args.plan), read_json(args.approval), cpu_fixtures=args.allow_cpu_fixtures)
            verify_frozen(plan)
            command = [sys.executable, str(Path(__file__).resolve()), "run", "--plan", str(args.plan.resolve()),
                       "--approval", str(args.approval.resolve()), "--state-dir", str(args.state_dir.resolve())]
            if args.allow_cpu_fixtures:
                command.append("--allow-cpu-fixtures")
            if args.lock_root:
                command.extend(["--lock-root", str(args.lock_root.resolve())])
            process = subprocess.Popen(command, cwd=plan["bundle_dir"], stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True,
                                       creationflags=0x00000008 | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
            print(json.dumps({"state": "launched", "pid": process.pid, "argv": command}))
        else:
            state = run(args.plan, args.approval, args.state_dir, cpu_fixtures=args.allow_cpu_fixtures, lock_root=args.lock_root)
            print(json.dumps({"reason": state["reason"], "counts": state["counts"]}))
    except (OSError, ValueError, KeyError, Refused, subprocess.SubprocessError) as error:
        if hasattr(args, "state_dir"):
            try:
                atomic_json(args.state_dir / "supervisor-error.json", {"state": "needs_action", "updated_utc": stamp(),
                                                                      "reason": f"{type(error).__name__}: {error}"})
            except OSError:
                pass
        print(f"kion_night_supervisor: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
