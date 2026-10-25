"""Source-only by default. Exact reviewed one-worker native Windows phase guard."""
import argparse
from contextlib import ExitStack
import csv
import ctypes
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
READY = ROOT.parent / 'ready-pack-20261020'
BASE = Path('C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
BASE_SHA = 'b141a0115de4be263a5cb0bdeb580ab67d074aaf1aadf10622fdf88ac2a1d8cb'
GPU = 'GPU-41efe8ae-15a1-c34d-27f2-7d4d504ea76b'
GIB = 1 << 30
COMMIT = RSS = 12 * GIB
INITIAL = 14 * GIB
RESERVE = 2 * GIB
VRAM = 7680 << 20
LOG_LIMIT = 8 << 20
CLEANUP = 45
AUDIT_COMMIT = 128 << 20
PHASES = {'cpu-pilot': 900, 'cpu-build': 24000, 'cpu-benchmark': 1200,
          'gpu-shortbench': 1200, 'gpu-e1': 24000}
HELPERS = {
    'portability': (Path('D:/personal-agent-system/scripts/portability.py'),
        '5a63306d80767c7676d066cb3d148856f047a09013590ba9c2b21acab9d54684'),
    'native_job': (READY / 'model/rl-native-resource-r1/native_job.py',
        '0da87bf737038e08602ddaf18929a9441543269adf0edd191f1ac4151e916296'),
    'working_set': (READY / 'model/owned-sft-ws-r1/working_set.py',
        '9dcbf1a43606511f936fadfbe323bde7c597b2a91bf98c9dd99e0204b2a6e1be'),
    'psutil_loader': (READY / 'reward-data/load_psutil_runtime_r1.py',
        '25f20e6e766e5cc53ecfaac5a676c3b8435d713993f5331ebddb937fce729bda'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def worker_request(plan):
    payload = Path(plan['worker_request']['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan['worker_request']['sha256']:
        raise ValueError('Worker request changed before exact-byte read')
    request = json.loads(payload.decode('utf-8'))
    output = Path(plan['run_dir']).resolve()
    if request.get('phase') != plan['phase'] or Path(request['output']).resolve() != output / 'worker':
        raise ValueError('Worker request phase/output must equal exact admitted phase/output')
    if request.get('stop_file') and not Path(request['stop_file']).resolve().is_relative_to(output):
        raise ValueError('Declared stop file must belong to fresh output subtree')
    return request


def phase_limits(phase):
    cap = (10 if phase.startswith('gpu-') else 8) * GIB
    return {'aggregate_commit_bytes': cap, 'aggregate_rss_bytes': cap,
        'initial_host_available_bytes': cap + RESERVE, 'host_reserve_bytes': RESERVE}


def pin(value):
    path = Path(value['path'])
    if not path.is_absolute() or not re.fullmatch('[0-9a-f]{64}', value['sha256']) or sha(path) != value['sha256']:
        raise ValueError('Exact absolute pinned file required: ' + str(path))
    return path.resolve()


def checked(plan_path, approval_path, *, now=None):
    plan, approval = read(plan_path), read(approval_path)
    if (plan.get('contract') != 'kion-precompute-e1-phase-v1'
            or approval.get('verdict') != 'approved' or not approval.get('reviewer')
            or approval.get('plan_sha256') != canonical(plan)):
        raise ValueError('Exact independently approved phase plan required')
    pin(approval['evidence'])
    phase = plan['phase']
    maximum = plan['max_seconds']
    if phase not in PHASES or type(maximum) is not int or not 1 <= maximum <= PHASES[phase]:
        raise ValueError('Only admitted finite phase budgets supported')
    if plan.get('resource_limits') != phase_limits(phase):
        raise ValueError('Only exact CPU8GiB/initial10GiB or GPU10GiB/initial12GiB limits admitted')
    deadline = datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00'))
    now = datetime.now(timezone.utc) if now is None else now
    if deadline.utcoffset() != timedelta(0) or deadline < now + timedelta(seconds=maximum + CLEANUP):
        raise ValueError('Fresh UTC deadline must cover full phase plus45s cleanup')
    if pin(plan['interpreter']) != BASE.resolve() or plan['interpreter']['sha256'] != BASE_SHA:
        raise ValueError('Only exact root-bound BASE_EXE admitted')
    worker, request_path = pin(plan['worker']), pin(plan['worker_request'])
    allowed_workers = [(ROOT / name / 'worker.py').resolve() for name in ('preparation', 'training')]
    if phase == 'cpu-build':
        allowed_workers.append((ROOT / 'full-preparation/worker.py').resolve())
    if worker not in allowed_workers:
        raise ValueError('Only experiment preparation/training or CPU-build full-preparation worker admitted')
    if not request_path.is_relative_to(ROOT) or request_path == ROOT:
        raise ValueError('Worker request must belong to this experiment')
    run_dir = Path(plan['run_dir']).resolve()
    if run_dir.parent != ROOT or run_dir.name in ('execution', 'preparation', 'training') or run_dir.exists():
        raise ValueError('Fresh experiment output sibling required')
    worker_request(plan)
    expected_argv = [BASE.as_posix(), '-B', worker.as_posix(), '--request', request_path.as_posix()]
    if plan.get('worker_argv') != expected_argv:
        raise ValueError('Only literal BASE -B worker --request ABS_JSON argv admitted')
    if Path(plan['cwd']).resolve() != ROOT:
        raise ValueError('Exact experiment cwd required')
    cpus = plan['cpu_affinity']
    gpu = phase.startswith('gpu-')
    if (not isinstance(cpus, list) or len(cpus) != (2 if gpu else 1) or len(set(cpus)) != len(cpus)
            or any(type(cpu) is not int or not 0 <= cpu < 64 for cpu in cpus)):
        raise ValueError('Explicit CPU1 prep / CPU2 GPU affinity required')
    sources = plan['source_hashes']
    required = {Path(__file__).resolve().as_posix(): sha(__file__), worker.as_posix(): plan['worker']['sha256']}
    required.update({path.resolve().as_posix(): digest for path, digest in HELPERS.values()})
    if any(sources.get(path) != digest for path, digest in required.items()):
        raise ValueError('Plan must pin guard, worker and every exact immutable native helper')
    for path, digest in {**sources, **plan['metadata_pins']}.items():
        pin({'path': path, 'sha256': digest})
    if not plan['metadata_pins']:
        raise ValueError('Frozen metadata pins required')
    console = pin(plan['console_host'])
    if console != (Path(os.environ['SYSTEMROOT']) / 'System32/conhost.exe').resolve():
        raise ValueError('Only exact OS console host admitted')
    protected = [Path(plan_path).resolve(), Path(approval_path).resolve(), request_path, worker,
                 *[Path(p).resolve() for p in sources], *[Path(p).resolve() for p in plan['metadata_pins']]]
    if any(path == run_dir or path.is_relative_to(run_dir) for path in protected):
        raise ValueError('Output overlaps protected input/source')
    if gpu:
        if plan.get('selected_gpu_uuid') != GPU:
            raise ValueError('Only exact root-bound RTX3070 UUID admitted')
        pin(plan['nvidia_smi'])
    elif 'nvidia_smi' in plan or 'selected_gpu_uuid' in plan:
        raise ValueError('CPU phases use CUDA hidden and no GPU audit')
    return plan, deadline


def load(name):
    path, expected = HELPERS[name]
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Immutable helper changed before direct compilation')
    spec = importlib.util.spec_from_file_location('_precompute_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8', newline='\n')
    os.replace(temporary, path)


class BoundedLog:
    def __init__(self, pipe, path, limit=LOG_LIMIT, exclusive=True):
        self.pipe, self.path, self.limit, self.exclusive = pipe, path, limit, exclusive
        self.overflow = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.drain, daemon=True)

    def drain(self):
        try:
            with self.path.open('xb' if self.exclusive else 'wb') as log:
                written = 0
                while chunk := self.pipe.read1(65536):
                    keep = min(len(chunk), self.limit - written)
                    log.write(chunk[:keep])
                    log.flush()  # Small progress lines are visible before worker EOF.
                    written += keep
                    if keep < len(chunk):
                        self.overflow.set()
        except BaseException as error:
            self.error = repr(error)
            self.overflow.set()
        finally:
            self.pipe.close()


def role(identity, argv, plan, leader_pid):
    image = Path(identity['exe']).resolve()
    if identity['pid'] == leader_pid and image == BASE.resolve() and argv == plan['worker_argv']:
        return 'worker'
    if (image == Path(plan['console_host']['path']).resolve() and len(argv) == 2
            and re.fullmatch(r'0x[0-9a-fA-F]+', argv[1])):
        spelling = argv[0][4:] if argv[0].startswith('\\??\\') else argv[0]
        if Path(spelling).resolve() == image:
            return 'console'
    raise ValueError('Unexpected child/image/argv inside owned worker Job')


def sample(api, psutil, plan, leader_pid):
    rss, private, count, identities = 0, 0, 0, []
    for pid in api.members():
        handle = None
        try:
            try:
                handle = api.open(pid)
            except OSError:
                try:
                    psutil.Process(pid)
                except psutil.NoSuchProcess:
                    continue
                raise
            identity = api.identity(handle)
            member = psutil.Process(pid)
            argv = member.cmdline()
            tag = role(identity, argv, plan, leader_pid)
            # Recheck original native handle after command-line lookup (PID reuse).
            if api.identity(handle) != identity:
                raise ValueError('Owned member identity changed during sample')
            info = api.memory(handle)
            rss += info['rss']
            private += info['private']
            count += tag == 'console'
            if count > 1:
                raise ValueError('More than one owned console process')
            identities.append({**identity, 'argv': argv, 'role': tag})
        except (ProcessLookupError, psutil.NoSuchProcess):
            pass
        except OSError:
            if handle is None or not api.exited(handle):
                raise
        finally:
            if handle is not None:
                api.close(handle)
    return rss, private, identities


def parse_gpu(payload):
    rows = list(csv.reader(io.StringIO(payload)))
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError('Exactly one selected GPU audit row required')
    uuid, name, memory = (value.strip() for value in rows[0])
    if uuid != GPU or name != 'NVIDIA GeForce RTX 3070' or not re.fullmatch(r'\d+', memory):
        raise ValueError('Exact RTX3070 UUID and numeric selected-device memory required')
    return int(memory) << 20


def gpu_sample(plan, owned, psutil, receipt, worker_rss=lambda: 0):
    # Separate short owned audit Job; no command is sent to worker or another app.
    command = [plan['nvidia_smi']['path'], '--id=' + GPU,
        '--query-gpu=uuid,name,memory.used', '--format=csv,noheader,nounits']
    factory = owned._new_job
    owned._new_job = limited_factory(owned, plan['cpu_affinity'], AUDIT_COMMIT)
    try:
        process = owned.popen_isolated(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        owned._new_job = factory
    receipt['audit_cleanup_quiescent'] = False
    log = None
    audit_peak = 0
    started = time.monotonic()
    try:
        log = BoundedLog(process.stdout, Path(plan['run_dir']) / 'gpu-audit.log', 4096, exclusive=False)
        log.thread.start()
        while process.poll() is None:
            if time.monotonic() - started >= 3 or log.overflow.is_set():
                raise RuntimeError('GPU audit timeout/output bound')
            try:
                audit_rss = psutil.Process(process.pid).memory_info().rss
            except psutil.NoSuchProcess:
                audit_rss = 0
            audit_peak = max(audit_peak, audit_rss)
            total_rss = worker_rss() + audit_rss
            receipt['peak_tree_RSS_bytes'] = max(receipt['peak_tree_RSS_bytes'], total_rss)
            if total_rss > phase_limits(plan['phase'])['aggregate_rss_bytes']:
                raise MemoryError('Owned worker plus GPU audit RSS above admitted aggregate cap')
            time.sleep(.05)
        if process.returncode:
            raise RuntimeError('Selected-device native GPU audit failed')
    finally:
        try:
            owned.terminate_process_tree(process)
            receipt['audit_cleanup_quiescent'] = bool(process._tree_closed)
        finally:
            if log is not None and log.thread.ident is not None:
                log.thread.join(timeout=5)
    if log.thread.is_alive() or log.error or log.overflow.is_set():
        raise RuntimeError('Bounded GPU audit log did not finish cleanly')
    return parse_gpu(log.path.read_text(encoding='utf-8')), audit_peak


def stop_reason(rss, available, elapsed, now, deadline, maximum, *, overflow=False, gpu_bytes=0, rss_cap=RSS):
    if available < RESERVE:
        return 'host_available_below2GiB'
    if rss > rss_cap:
        return 'owned_aggregate_RSS_above_admitted_cap'
    if elapsed >= maximum or now >= deadline - timedelta(seconds=CLEANUP):
        return 'finite_wall_or_deadline_cleanup_boundary'
    if overflow:
        return 'bounded_log_overflow'
    if gpu_bytes > VRAM:
        return 'selected_GPU_above7680MiB'
    return None


def limited_factory(owned, cpus, cap=COMMIT):
    def create():
        job = owned._create_job(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = owned._ExtendedLimits()
        limits.Basic.Flags = 0x2000 | 0x200 | 0x10  # kill-close / aggregate commit / affinity
        limits.JobMemory = cap
        limits.Basic.Affinity = sum(1 << cpu for cpu in cpus)
        if not owned._set_job(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            owned._close(job)
            raise error
        return job
    return create


def lock_marker(lock, value):
    lock.seek(0)
    lock.write((json.dumps(value, sort_keys=True) + '\n').encode('utf-8'))
    lock.truncate()
    lock.flush()
    os.fsync(lock.fileno())


def run(plan_path, approval_path):
    if os.name != 'nt':
        raise OSError('Native Windows Job guard required')
    plan, deadline = checked(plan_path, approval_path)
    owned, native, ws = load('portability'), load('native_job'), load('working_set')
    psutil = load('psutil_loader').load()
    run_dir = Path(plan['run_dir'])
    gpu = plan['phase'].startswith('gpu-')
    policy = phase_limits(plan['phase'])
    cap = policy['aggregate_commit_bytes']
    worker_commit = cap - AUDIT_COMMIT if gpu else cap
    receipt = {'contract': 'kion-precompute-e1-phase-receipt-v1', 'state': 'failed',
        'plan_sha256': canonical(plan), 'phase': plan['phase'], 'worker_argv': plan['worker_argv'],
        'native_JobMemory_bytes': worker_commit, 'audit_JobMemory_bytes': AUDIT_COMMIT if gpu else 0,
        'all_owned_JobMemory_sum_bytes': cap, 'sampled_RSS_cap_bytes': policy['aggregate_rss_bytes'],
        'initial_host_available_bytes': policy['initial_host_available_bytes'], 'host_reserve_bytes': RESERVE,
        'CPU_affinity': plan['cpu_affinity'], 'max_seconds': plan['max_seconds'],
        'deadline_utc': plan['deadline_utc'], 'cleanup_reserve_seconds': CLEANUP,
        'peak_tree_RSS_bytes': 0, 'peak_tree_private_bytes': 0, 'minimum_host_available_bytes': None,
        'peak_selected_GPU_bytes': 0, 'selected_GPU_limit_bytes': VRAM,
        'GPU_bound_scope': 'conservative whole selected device; all owned CUDA allocation on exact UUID',
        'parent_scope': 'lightweight metadata owner outside owned worker/audit Job caps',
        'native_peak_process_commit_bytes': None, 'cleanup_quiescent': False,
        'audit_cleanup_quiescent': True,
        'worker_pid': None, 'exit_code': None, 'reason': None}
    process = None
    log = None
    admission = ExitStack()
    created = False
    lock = None
    admitted = False
    stop = None
    started = time.monotonic()
    try:
        # One fixed experiment lock, no stale-lock replacement or override.
        lock = admission.enter_context(os.fdopen(owned.open_regular_no_follow(
            ROOT / 'heavy.lock', os.O_RDWR | os.O_CREAT), 'r+b'))
        admission.enter_context(owned.exclusive_lock(lock, timeout=0))
        with ExitStack():
            previous = lock.read(4097)
            if previous and (len(previous) > 4096 or read_marker(previous).get('cleanup_quiescent') is not True):
                raise RuntimeError('Previous owned cleanup unproven; no automatic override')
            receipt['last_positive_quiescent_marker'] = read_marker(previous) if previous else None
            rechecked, _ = checked(plan_path, approval_path)  # recheck under exclusive admission lock
            if canonical(rechecked) != canonical(plan):
                raise ValueError('Plan changed between initial check and exclusive admission')
            request = worker_request(plan)
            stop = Path(request['stop_file']).resolve() if request.get('stop_file') else None
            run_dir.mkdir()
            created = True
            lock_marker(lock, {'state': 'active', 'plan_sha256': canonical(plan), 'cleanup_quiescent': False})
            admitted = True
            available = psutil.virtual_memory().available
            receipt['minimum_host_available_bytes'] = available
            if available < policy['initial_host_available_bytes']:
                raise MemoryError('Initial host available below phase cap plus2GiB reserve')
            if not set(plan['cpu_affinity']).issubset(psutil.Process().cpu_affinity()):
                raise ValueError('Requested affinity unavailable')
            if gpu:
                used, audit_rss = gpu_sample(plan, owned, psutil, receipt)
                receipt['peak_selected_GPU_bytes'] = used
                receipt['peak_tree_RSS_bytes'] = audit_rss
                if used > VRAM:
                    raise MemoryError('Selected GPU memory above7680MiB before admission')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU if gpu else '-1',
                CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1')
            if gpu:
                env['KION_OUTPUT_DIR'] = Path(request['output']).resolve().as_posix()
            for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env[name] = str(len(plan['cpu_affinity']))
            factory = owned._new_job
            owned._new_job = limited_factory(owned, plan['cpu_affinity'], worker_commit)
            try:
                process = owned.popen_isolated(plan['worker_argv'], cwd=plan['cwd'], env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            finally:
                owned._new_job = factory
            receipt['worker_pid'] = process.pid
            log = BoundedLog(process.stdout, run_dir / 'worker.log')
            log.thread.start()
            api = native.JobAPI(process._job, ws)
            last_gpu = 0
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    available = psutil.virtual_memory().available
                    receipt['minimum_host_available_bytes'] = min(receipt['minimum_host_available_bytes'], available)
                    rss, private, identities = sample(api, psutil, plan, process.pid)
                    receipt['peak_tree_RSS_bytes'] = max(receipt['peak_tree_RSS_bytes'], rss)
                    receipt['peak_tree_private_bytes'] = max(receipt['peak_tree_private_bytes'], private)
                    receipt['last_owned_members'] = identities
                    reason = stop_reason(rss, available, now - started, datetime.now(timezone.utc),
                        deadline, plan['max_seconds'], overflow=log.overflow.is_set(),
                        rss_cap=policy['aggregate_rss_bytes'])
                    if reason:
                        raise RuntimeError(reason)
                    if gpu and now - last_gpu >= 1:
                        used, audit_rss = gpu_sample(plan, owned, psutil, receipt,
                            lambda: sample(api, psutil, plan, process.pid)[0])
                        receipt['peak_tree_RSS_bytes'] = max(receipt['peak_tree_RSS_bytes'], rss + audit_rss)
                        receipt['peak_selected_GPU_bytes'] = max(receipt['peak_selected_GPU_bytes'], used)
                        if used > VRAM:
                            raise MemoryError('Selected GPU memory above7680MiB')
                        last_gpu = time.monotonic()
                    time.sleep(.1)
                receipt['exit_code'] = process.returncode
                if process.returncode != 0:
                    raise RuntimeError('Worker exited unsuccessfully')
            finally:
                try:
                    receipt['native_peak_process_commit_bytes'] = owned.process_peak_memory_bytes(process)
                finally:
                    owned.terminate_process_tree(process)
                    receipt['cleanup_quiescent'] = bool(process._tree_closed)
            log.thread.join(timeout=5)
            if log.thread.is_alive() or log.error or log.overflow.is_set():
                raise RuntimeError('Bounded worker log did not finish cleanly')
            receipt['state'] = 'completed'
    except BaseException as error:
        receipt['reason'] = repr(error)
        raise
    finally:
        # Covers errors between launch and entry into the sampling finally block.
        try:
            try:
                if process is not None and process.poll() is None and stop is not None:
                    if not stop.exists():
                        write(stop, {'stop_requested': True, 'reason': receipt['reason'] or 'owned_cleanup'})
            except BaseException as error:
                receipt['stop_request_error'] = repr(error)  # Never skip owned kill for a stop-file I/O error.
            if process is not None and not process._tree_closed:
                owned.terminate_process_tree(process)
                receipt['cleanup_quiescent'] = bool(process._tree_closed)
            if log is not None:
                log.thread.join(timeout=5)
        finally:
            receipt['elapsed_seconds'] = time.monotonic() - started
            receipt['exit_code'] = process.poll() if process is not None else None
            receipt['log_error'] = log.error if log is not None else None
            receipt['cleanup_quiescent'] = ((process is None or process._tree_closed)
                and receipt['audit_cleanup_quiescent'])
            try:
                try:
                    if created:
                        write(run_dir / 'guard.json', receipt)
                finally:
                    if admitted:
                        quiescent = receipt['cleanup_quiescent']
                        lock_marker(lock, {'state': 'quiescent' if quiescent else 'cleanup_unproven',
                            'plan_sha256': canonical(plan), 'cleanup_quiescent': quiescent,
                            'receipt': (run_dir / 'guard.json').as_posix()})
            finally:
                admission.close()  # lock remains held through all owned cleanup, even launch errors
    return receipt


def read_marker(payload):
    value = json.loads(payload.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Invalid owned lock marker')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(run(args.plan, args.approval), indent=2))
    else:
        plan, _ = checked(args.plan, args.approval)
        print(json.dumps({'state': 'source_only_ready', 'root_GO_required': True,
            'phase': plan['phase'], 'worker_argv': plan['worker_argv'], 'plan_sha256': canonical(plan)}, indent=2))


if __name__ == '__main__':
    main()
