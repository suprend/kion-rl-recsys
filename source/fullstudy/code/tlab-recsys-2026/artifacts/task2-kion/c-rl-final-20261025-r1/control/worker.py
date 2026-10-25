"""Validate pins and native limits before importing the root-admitted stage."""
from __future__ import annotations
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from common import BASE, BASE_SHA, CONTROL, GIB, PAS, ROOT, RUNTIME, deadline, probe_space, read, remaining, require, sha, stat_key, utc, validate, write

sys.dont_write_bytecode = True

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def job_evidence(cap_gib):
    sys.path.insert(0, str(PAS/'scripts'))
    import portability
    limits = portability._ExtendedLimits()
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = ctypes.c_int
    require(kernel.QueryInformationJobObject(None, 9, ctypes.byref(limits), ctypes.sizeof(limits), None), 'Missing native Job')
    require(limits.Basic.Flags & 0x2210 == 0x2210, 'Native kill/memory/affinity flags required')
    require(limits.JobMemory == cap_gib*GIB and limits.Basic.Affinity == 0x55, 'Native cap/affinity mismatch')
    return {'flags': int(limits.Basic.Flags), 'cap_bytes': int(limits.JobMemory), 'affinity': int(limits.Basic.Affinity)}

class Budget:
    def __init__(self, request, request_path):
        self.request = request
        self.started = time.monotonic()
        self.pins = {**request['source_hashes'], **request['input_hashes'], str(request_path): sha(request_path), str(ROOT/'campaign.json'): sha(ROOT/'campaign.json')}
        self.stats = {path: stat_key(path) for path in self.pins}
    def __call__(self, **fields):
        require(remaining(self.request) > 0, 'Request deadline reached')
        require(time.monotonic()-self.started < self.request['maximum_seconds'], 'Stage duration reached')
        require(not (ROOT/'STOP').exists(), 'Campaign stop requested')
        import psutil
        require(psutil.virtual_memory().available >= 2*GIB, 'Host reserve reached')
        probe_space(self.request, estimated=False)
        for path, expected in self.stats.items():
            require(stat_key(path) == expected, 'Pinned file changed: ' + path)
        if fields:
            print(json.dumps({'event': 'campaign_stage_progress', 'utc': utc(), 'seconds': time.monotonic()-self.started, **fields}, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    request = validate(read(args.request), args.request, args.request_sha256)
    require(os.name == 'nt' and Path(sys.executable).resolve() == BASE.resolve() and sha(BASE) == BASE_SHA, 'Pinned native BASE interpreter required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == 'GPU-41efe8ae-15a1-c34d-27f2-7d4d504ea76b', 'GPU identity missing')
    require(os.environ.get('OMP_NUM_THREADS') == '4' and os.environ.get('MKL_NUM_THREADS') == '4', 'Thread contract missing')
    operation = ROOT/'operations'/request['operation']
    require(operation.is_dir() and not (operation/'stage-completion.json').exists(), 'Missing or duplicate operation')
    receipt = {'contract': request['contract'], 'stage': request['stage'], 'started_utc': utc(), 'state': 'running', 'request_path': args.request.as_posix(), 'request_sha256': args.request_sha256}
    code = 1
    try:
        receipt['job'] = job_evidence(request['cap_gib'])
        budget = Budget(request, args.request)
        budget()
        runtime = load(RUNTIME, '_c3_ready_runtime')
        runtime.configure_paths()
        # Runtime bootstrap may alter sys.path; recheck source stats before stage import.
        budget()
        module = load(Path(request['module_path']), '_c3_' + request['operation'].replace('-', '_'))
        result = getattr(module, request['function'])(request, budget)
        budget()
        receipt.update(state='completed', result=result)
        code = 0
    except BaseException as error:
        receipt.update(state='failed', error=repr(error))
        import traceback
        traceback.print_exc()
    finally:
        receipt['finished_utc'] = utc()
        write(operation/'stage-completion.json', receipt)
        print(json.dumps({'event': 'campaign_stage_completion', 'stage': request['stage'], 'state': receipt['state'], 'error': receipt.get('error')}, ensure_ascii=False), flush=True)
    return code

if __name__ == '__main__':
    raise SystemExit(main())
