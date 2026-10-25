"""One admitted pilot stage. No heavy imports on import; root owns native supervisor."""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path("C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe")
BASE_SHA = "5640e9be2fd33d05a1a9ae1228e2354609a69ab831605995ec4e2bd6da8fff5e"
DEADLINE = datetime(2026, 10, 24, 18, 48, 30, tzinfo=timezone.utc)
CACHE = Path("C:/Users/Михаил/.cache/codex-kion/three-head-262k-20261024-r1")
STAGES = {"bank": ("bank.py", "run"), "heads": ("heads.py", "run"),
          "evaluate": ("evaluate.py", "evaluate"), "analyze": ("analyze.py", "run"),
          "smoke": ("smoke.py", "run")}
sys.dont_write_bytecode = True

def require(value, message):
    if not value:
        raise ValueError(message)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    temp.replace(path)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate(request, request_path, expected):
    require(sha(request_path) == expected, "Request digest mismatch")
    require(request["contract"] == "kion-three-head-262k-native-v1", "Unexpected contract")
    require(request["stage"] in STAGES, "Unknown stage")
    require(datetime.fromisoformat(request["deadline_utc"]) == DEADLINE, "Deadline changed")
    require(datetime.now(timezone.utc) < DEADLINE, "Goal deadline reached")
    require(request["rows"] == 262144 and request["validation_rows"] == 4096, "Fixed pilot scope only")
    require(request["epochs"] == 3 and request["seed"] == 20261024, "Unexpected common budget")
    require(request["full_gpt_training"] is False and request["rl"] is False and request["final_test"] is False, "Excluded work")
    require(request["cpu_threads"] == 4 and request["reserve_gib"] == 2 and request["cap_gib"] in (4, 8, 12), "Native resource contract")
    require(Path(request["bank_dir"]).resolve() == (CACHE / "train").resolve(), "Foreign bank output")
    require(Path(request["bank_output"]).resolve() == (CACHE / "train").resolve(), "Foreign bank output")
    require(Path(request["output_dir"]).resolve().is_relative_to(ROOT), "Foreign head output")
    require(Path(request["evaluation_dir"]).resolve().is_relative_to(ROOT), "Foreign evaluation output")
    require(Path(request["analysis_dir"]).resolve().is_relative_to(ROOT), "Foreign analysis output")
    require(request["teacher"]["sha256"] == "5e7101857c2dbe100fb4e1587c0369abc95c108303ed149123ee183f62aeac26", "Only accepted E2 teacher")
    if request["stage"] == "bank":
        require(shutil.disk_usage(CACHE).free >= 37_000_000_000 + 15 * 2**30, "Insufficient space for full262k bank plus reserve")
    require(not (ROOT / "STOP").exists(), "Pilot stop requested")
    require(Path(sys.executable).resolve() == BASE.resolve() and sha(BASE) == BASE_SHA, "Pinned BASE interpreter required")
    require(os.name == "nt", "Native Windows required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "GPU-41efe8ae-15a1-c34d-27f2-7d4d504ea76b", "GPU identity missing")
    require(os.environ.get("OMP_NUM_THREADS") == "4" and os.environ.get("MKL_NUM_THREADS") == "4", "Thread contract missing")
    for field in ("source_hashes", "input_hashes"):
        for path, digest in request[field].items():
            require(Path(path).is_absolute() and sha(path) == digest, "Changed " + field + ": " + path)
    filename = STAGES[request["stage"]][0]
    dependencies = {"bank": ["bank.py"], "heads": ["heads.py"], "evaluate": ["evaluate.py", "bank.py", "heads.py"], "analyze": ["analyze.py", "evaluate.py", "bank.py", "heads.py"], "smoke": ["smoke.py", "bank.py", "heads.py", "evaluate.py", "analyze.py", "test_bank_source.py", "test_heads_source.py", "test_eval_source.py"]}
    for required_source in dependencies[request["stage"]]:
        require((ROOT / required_source).as_posix() in request["source_hashes"], "Stage dependency unpinned: " + required_source)
    require((ROOT / "worker.py").as_posix() in request["source_hashes"], "Worker unpinned")
    require((ROOT / "supervise.py").as_posix() in request["source_hashes"], "Supervisor unpinned")
    return request

def job_evidence(cap_gib):
    # Same native structure and WinJob admission used by the original reviewed guard.
    sys.path.insert(0, "D:/personal-agent-system/scripts")
    import portability
    limits = portability._ExtendedLimits()
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    require(kernel.QueryInformationJobObject(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None), "Missing native Job")
    require(limits.Basic.Flags & 0x2210 == 0x2210, "Native kill/memory/affinity flags required")
    require(limits.JobMemory == cap_gib * 2**30 and limits.Basic.Affinity == 0x55, "Native cap/affinity mismatch")
    return dict(flags=int(limits.Basic.Flags), cap_bytes=int(limits.JobMemory), affinity=int(limits.Basic.Affinity))

class Budget:
    def __init__(self, request, request_path):
        self.request = request
        self.started = time.monotonic()
        self.last_stat = self.started
        self.pins = {**request["source_hashes"], **request["input_hashes"], str(request_path): sha(request_path)}
        self.stats = {p: (Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in self.pins}
    def __call__(self, **fields):
        require(datetime.now(timezone.utc) < DEADLINE, "Goal deadline reached")
        require(time.monotonic() - self.started < self.request["maximum_seconds"], "Stage duration reached")
        require(not (ROOT / "STOP").exists(), "Pilot stop requested")
        import psutil
        require(psutil.virtual_memory().available >= 2 * 2**30, "Host reserve reached")
        require(shutil.disk_usage(ROOT).free >= 15 * 2**30 and shutil.disk_usage(CACHE).free >= 15 * 2**30, "Disk reserve reached")
        if time.monotonic() - self.last_stat >= 30:
            for path, expected in self.stats.items():
                require((Path(path).stat().st_size, Path(path).stat().st_mtime_ns) == expected, "Pinned file changed: " + path)
            self.last_stat = time.monotonic()
        if fields:
            print(json.dumps(dict(event="pilot_stage_progress", utc=utc(), seconds=time.monotonic()-self.started, **fields), ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--request-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    request = validate(read(args.request), args.request, args.request_sha256)
    if args.preflight:
        print(json.dumps(dict(state="preflight_passed", stage=request["stage"], heavy_imports=False)))
        return
    operation = ROOT / "operations" / request["operation"]
    require(operation.is_dir(), "Native operation receipt absent")
    receipt_path = operation / "stage-completion.json"
    require(not receipt_path.exists(), "Duplicate stage completion")
    receipt = dict(contract=request["contract"], stage=request["stage"], started_utc=utc(), state="running",
                   rows=request["rows"], validation_rows=request["validation_rows"], epochs=request["epochs"],
                   request_path=args.request.as_posix(), request_sha256=args.request_sha256, full_gpt_training=False)
    code = 1
    try:
        receipt["job"] = job_evidence(request["cap_gib"])
        budget = Budget(request, args.request)
        budget()
        if request["stage"] in ("heads", "analyze"):
            runtime_path = Path("D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py")
            require(request["source_hashes"].get(runtime_path.as_posix()) == sha(runtime_path), "Runtime bootstrap unpinned")
            runtime = load(runtime_path, "_expanded_ready_runtime")
            runtime.configure_paths()
        module_file, function_name = STAGES[request["stage"]]
        module = load(ROOT / module_file, "_three_head_" + request["stage"])
        result = getattr(module, function_name)(request, budget)
        budget()
        receipt.update(state="completed", result=result)
        code = 0
    except BaseException as error:
        receipt.update(state="failed", error=repr(error))
        import traceback
        traceback.print_exc()
    finally:
        receipt["finished_utc"] = utc()
        write(receipt_path, receipt)
        print(json.dumps(dict(event="pilot_stage_completion", state=receipt["state"], stage=request["stage"], error=receipt.get("error")), ensure_ascii=False), flush=True)
    raise SystemExit(code)

if __name__ == "__main__":
    main()
