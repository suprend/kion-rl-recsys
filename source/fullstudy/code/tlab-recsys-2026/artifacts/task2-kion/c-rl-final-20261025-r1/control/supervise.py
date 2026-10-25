"""One bounded owned Windows Job; lock survives through complete tree cleanup."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import BASE, BASE_SHA, CONTROL, GIB, PAS, ROOT, probe_space, read, remaining, require, sha, stat_key, utc, validate, write

sys.dont_write_bytecode = True

def stop_reason(request, elapsed, available, rss, private):
    if (ROOT/'STOP').exists():
        return 'stop_requested'
    if elapsed >= request['maximum_seconds'] or remaining(request) <= 0:
        return 'time_limit'
    if available < request['reserve_gib']*GIB:
        return 'host_reserve'
    if max(rss, private) > request['cap_gib']*GIB:
        return 'memory_cap'
    return None

def previous_quiescent(operation):
    for directory in (ROOT/'operations').iterdir():
        if directory == operation or not directory.is_dir():
            continue
        path = directory/'receipt.json'
        require(path.is_file() and read(path).get('cleanup_quiescent') is True, 'Previous native operation lacks proven quiescence: ' + directory.name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    request = validate(read(args.request), args.request, args.request_sha256, verify_pins=False)
    require(os.name == 'nt' and Path(sys.executable).resolve() == BASE.resolve() and sha(BASE) == BASE_SHA, 'Pinned native BASE interpreter required')
    # Verify supervisor/control/portability sources before loading native helpers.
    for path, digest in request['source_hashes'].items():
        require(sha(path) == digest, 'Pinned source changed: ' + path)
    sys.path.insert(0, str(PAS/'scripts'))
    import portability
    import psutil
    from launch import native_env
    cpus = [0, 2, 4, 6]
    require(set(cpus).issubset(psutil.Process().cpu_affinity()), 'Physical logical CPU positions unavailable')
    operation = ROOT/'operations'/request['operation']
    operation.mkdir(parents=True, exist_ok=False)
    cap = request['cap_gib']*GIB
    available = psutil.virtual_memory().available
    receipt = {'name': request['operation'], 'state': 'prepared', 'started_utc': utc(), 'request_sha256': args.request_sha256, 'argv': [BASE.as_posix(), '-B', (CONTROL/'worker.py').as_posix(), '--request', args.request.resolve().as_posix(), '--request-sha256', args.request_sha256], 'maximum_seconds': request['maximum_seconds'], 'deadline_utc': request['deadline_utc'], 'cap_gib': request['cap_gib'], 'reserve_gib': 2, 'cpu_affinity': cpus, 'worker_pid': None, 'exit_code': None, 'initial_available_bytes': available, 'minimum_available_bytes': available, 'peak_rss_bytes': 0, 'peak_private_bytes': 0, 'cleanup_quiescent': False}
    write(operation/'receipt.json', receipt)
    original_new_job = portability._new_job
    def limited_job():
        job = original_new_job()
        limits = portability._ExtendedLimits()
        limits.Basic.Flags = 0x2000 | 0x200 | 0x10
        limits.Basic.Affinity = 0x55
        limits.JobMemory = cap
        if not portability._set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            portability._close(job)
            raise error
        return job
    portability._new_job = limited_job
    process = None
    stack = ExitStack()
    started = time.monotonic()
    try:
        lock = stack.enter_context((ROOT/'heavy.owner.lock').open('a+b'))
        stack.enter_context(portability.exclusive_lock(lock, timeout=0))
        previous_quiescent(operation)
        validate(request, args.request, args.request_sha256, verify_pins=False)
        require(psutil.virtual_memory().available >= cap + 2*GIB, 'Insufficient RAM for cap plus 2GiB reserve')
        probe_space(request, write_probe=True)
        watched = [*request['source_hashes'], *request['input_hashes'], str(args.request), str(ROOT/'campaign.json')]
        stats = {path: stat_key(path) for path in watched}
        power_call = portability._api('SetThreadExecutionState', ctypes.c_uint32, ctypes.c_uint32)
        previous_power = power_call(0x80000001)
        if not previous_power:
            raise ctypes.WinError(ctypes.get_last_error())
        stack.callback(power_call, previous_power)
        receipt['idle_sleep_prevention_while_owned'] = True
        with (operation/'worker.log').open('x', encoding='utf-8') as output, (operation/'resources.jsonl').open('x', encoding='utf-8') as resources:
            process = portability.popen_isolated(receipt['argv'], cwd=str(ROOT), env=native_env(), stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            receipt.update(state='submitted', worker_pid=process.pid)
            write(operation/'receipt.json', receipt)
            try:
                identity = psutil.Process(process.pid)
            except psutil.NoSuchProcess:
                if process.poll() is None:
                    raise
                identity = None
            while process.poll() is None:
                rss = private = 0
                try:
                    members = [identity, *identity.children(recursive=True)]
                except psutil.NoSuchProcess:
                    members = []
                for member in members:
                    try:
                        info = member.memory_info()
                        rss += info.rss
                        private += getattr(info, 'private', info.vms)
                    except psutil.NoSuchProcess:
                        pass
                available = psutil.virtual_memory().available
                elapsed = time.monotonic()-started
                receipt['minimum_available_bytes'] = min(receipt['minimum_available_bytes'], available)
                receipt['peak_rss_bytes'] = max(receipt['peak_rss_bytes'], rss)
                receipt['peak_private_bytes'] = max(receipt['peak_private_bytes'], private)
                reason = stop_reason(request, elapsed, available, rss, private)
                if reason is None:
                    try:
                        probe_space(request, estimated=False)
                        for path, expected in stats.items():
                            require(stat_key(path) == expected, 'Pinned file changed: ' + path)
                    except BaseException as error:
                        reason = 'disk_or_pin_guard'
                        receipt['guard_error'] = repr(error)
                resources.write(json.dumps({'utc': utc(), 'elapsed': elapsed, 'rss': rss, 'private': private, 'host_available': available, 'stop_reason': reason})+'\n')
                resources.flush()
                if reason:
                    receipt['state'] = reason
                    portability.terminate_process_tree(process)
                    break
                time.sleep(1)
            receipt['exit_code'] = process.wait(timeout=15)
            if receipt['state'] == 'submitted':
                receipt['state'] = 'succeeded' if receipt['exit_code'] == 0 else 'failed'
    except BaseException as error:
        receipt.update(state='supervisor_error', error=repr(error))
    finally:
        try:
            if process is not None:
                portability.terminate_process_tree(process)
                receipt['cleanup_quiescent'] = process.poll() is not None and process._job is None
            else:
                receipt['cleanup_quiescent'] = True
        except BaseException as error:
            receipt.update(state='cleanup_failed', cleanup_error=repr(error), cleanup_quiescent=False)
        finally:
            try:
                receipt.update(finished_utc=utc(), elapsed_seconds=time.monotonic()-started)
                write(operation/'receipt.json', receipt)
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
            finally:
                portability._new_job = original_new_job
                stack.close()
    # A failed cleanup receipt blocks every subsequent admission, even after supervisor exit.
    return 0 if receipt['state'] == 'succeeded' and receipt['cleanup_quiescent'] else 1

if __name__ == '__main__':
    raise SystemExit(main())
