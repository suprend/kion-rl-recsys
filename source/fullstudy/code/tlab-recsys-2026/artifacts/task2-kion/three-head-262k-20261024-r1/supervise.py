"""One finite owned Windows worker, hard Job memory/affinity and host-reserve monitoring."""
import argparse
from contextlib import ExitStack
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
PAS = Path('D:/personal-agent-system')
sys.path.insert(0, str(PAS / 'scripts'))
import portability
import psutil

ROOT = Path(__file__).resolve().parent
BASE = Path('C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
GIB = 2**30


def utc():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    parser.add_argument('--cap-gib', type=int, default=10)
    parser.add_argument('--reserve-gib', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output-env', type=Path)
    parser.add_argument('worker', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.name != 'nt' or not args.name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('Native Windows and plain operation name required')
    if not 1 <= args.cap_gib <= 12 or not 1 <= args.reserve_gib <= 4 or not 1 <= args.seconds <= 172800:
        raise ValueError('Out of bounded resource range')
    if args.threads not in (2,4):
        raise ValueError('Only bounded two/four CPU threads')
    command = args.worker[1:] if args.worker[:1] == ['--'] else args.worker
    if not command:
        raise ValueError('Worker script and arguments required')
    script = Path(command[0]).resolve(strict=True)
    if not script.is_relative_to(ROOT):
        raise ValueError('Only campaign-owned worker scripts')
    operation = ROOT / 'operations' / args.name
    operation.mkdir(parents=True, exist_ok=False)
    cpus = [0,2,4,6][:args.threads]
    if not set(cpus).issubset(psutil.Process().cpu_affinity()):
        raise ValueError('Requested physical logical CPU positions unavailable')
    cap = args.cap_gib * GIB
    reserve = args.reserve_gib * GIB
    available = psutil.virtual_memory().available
    receipt = {'name':args.name,'state':'prepared','started_utc':utc(), 'argv':[BASE.as_posix(),'-B',script.as_posix(),*command[1:]], 'seconds':args.seconds,'cap_gib':args.cap_gib,'reserve_gib':args.reserve_gib,'cpu_affinity':cpus,'initial_available_bytes':available,'worker_pid':None,'exit_code':None,'peak_rss_bytes':0,'peak_private_bytes':0,'minimum_available_bytes':available,'cleanup_quiescent':False}
    dump(operation / 'receipt.json',receipt)
    if available < cap + reserve:
        receipt.update(state='not_admitted_memory',finished_utc=utc(),cleanup_quiescent=True)
        dump(operation / 'receipt.json',receipt)
        raise MemoryError('Not enough host RAM for chosen cap plus reserve')
    original_new_job = portability._new_job
    def limited_job():
        job = original_new_job()
        limits = portability._ExtendedLimits()
        limits.Basic.Flags = 0x2000 | 0x200 | 0x10  # kill-on-close, aggregate job memory, affinity
        limits.Basic.Affinity = sum(1 << cpu for cpu in cpus)
        limits.JobMemory = cap
        if not portability._set_job(job,9,ctypes.byref(limits),ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            portability._close(job)
            raise error
        return job
    portability._new_job = limited_job
    env = os.environ.copy()
    env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES='GPU-41efe8ae-15a1-c34d-27f2-7d4d504ea76b')
    if args.output_env is not None:
        selected_output = args.output_env.resolve()
        if not selected_output.is_relative_to(ROOT):
            raise ValueError('Diagnostic environment output must belong to campaign')
        env['KION_OUTPUT_DIR'] = selected_output.as_posix()
        receipt['KION_OUTPUT_DIR'] = selected_output.as_posix()
    process = None
    started = time.monotonic()
    stack = ExitStack()
    try:
        lock = stack.enter_context((ROOT / 'heavy.owner.lock').open('a+b'))
        stack.enter_context(portability.exclusive_lock(lock, timeout=0))
        if (ROOT / 'STOP').exists():
            raise RuntimeError('Stop requested before launch')
        if psutil.virtual_memory().available < cap + reserve:
            raise MemoryError('Host admission changed while waiting for ownership')
        # Hold an execution request for this supervisor thread only; screen and power plan unchanged.
        # https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
        power_call = portability._api('SetThreadExecutionState', ctypes.c_uint32, ctypes.c_uint32)
        previous_power = power_call(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        if not previous_power:
            raise ctypes.WinError(ctypes.get_last_error())
        stack.callback(power_call, previous_power)
        receipt['idle_sleep_prevention_while_owned'] = True
        with (operation / 'worker.log').open('x',encoding='utf-8') as output, (operation / 'resources.jsonl').open('x',encoding='utf-8') as stats:
                process = portability.popen_isolated(receipt['argv'],cwd=str(ROOT),env=env,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                receipt.update(state='submitted',worker_pid=process.pid)
                dump(operation / 'receipt.json',receipt)
                try:
                    identity = psutil.Process(process.pid)
                except psutil.NoSuchProcess:
                    if process.poll() is None:
                        raise
                    identity = None
                while process.poll() is None:
                    rss = private = 0
                    try:
                        members = [identity,*identity.children(recursive=True)]
                    except psutil.NoSuchProcess:
                        if process.poll() is None:
                            raise
                        members = []
                    for member in members:
                        try:
                            info = member.memory_info()
                            rss += info.rss
                            private += getattr(info,'private',info.vms)
                        except psutil.NoSuchProcess:
                            pass
                    free = psutil.virtual_memory().available
                    elapsed = time.monotonic()-started
                    observation = {'utc':utc(),'elapsed':round(elapsed,3),'rss':rss,'private':private,'host_available':free}
                    stats.write(json.dumps(observation)+'\n')
                    stats.flush()
                    receipt['peak_rss_bytes'] = max(receipt['peak_rss_bytes'],rss)
                    receipt['peak_private_bytes'] = max(receipt['peak_private_bytes'],private)
                    receipt['minimum_available_bytes'] = min(receipt['minimum_available_bytes'],free)
                    reason = 'stop_requested' if (ROOT/'STOP').exists() else 'time_limit' if elapsed>args.seconds else 'host_reserve' if free<reserve else 'memory_cap' if max(rss,private)>cap else None
                    if reason:
                        receipt['state'] = reason
                        portability.terminate_process_tree(process)
                        break
                    time.sleep(1)
                receipt['exit_code'] = process.wait(timeout=15)
                if receipt['state'] == 'submitted':
                    receipt['state'] = 'succeeded' if receipt['exit_code']==0 else 'failed'
    except BaseException as error:
        receipt.update(state='supervisor_error',error=repr(error))
        raise
    finally:
        try:
            if process is not None:
                portability.terminate_process_tree(process)
                receipt['cleanup_quiescent'] = process.poll() is not None and process._job is None
            else:
                receipt['cleanup_quiescent'] = True
        except BaseException as error:
            receipt.update(state='cleanup_failed',cleanup_error=repr(error),cleanup_quiescent=False)
        finally:
            try:
                receipt.update(finished_utc=utc(),elapsed_seconds=time.monotonic()-started)
                dump(operation / 'receipt.json',receipt)
                print(json.dumps(receipt,ensure_ascii=False),flush=True)
            finally:
                # Ownership is held until the complete Job has been reaped and receipt recorded.
                stack.close()
    return 0 if receipt['state']=='succeeded' and receipt['cleanup_quiescent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
