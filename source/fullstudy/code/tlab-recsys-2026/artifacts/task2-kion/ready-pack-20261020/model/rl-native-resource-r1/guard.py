"""RL C512/B1/H1-only sampled resource ceiling around reviewed native adapter run."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
READY = HERE.parents[1]
RUNS = READY.parent / 'ready-runs-20261020'
COMMON = READY / 'provider/runtime-native-r5'
ADAPTER = READY / 'rl-bridge/launch-prep-r1/receipt-adapter-r1/supervisor_adapter.py'
HELDOUT_ADAPTER = READY / 'rl-bridge/launch-prep-r1/heldout-native-adapter-r1/supervisor_adapter.py'
AUDIT_TAILS = [['--query-gpu=uuid', '--format=csv,noheader,nounits'],
               ['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits']]
WS = HERE.parent / 'owned-sft-ws-r1/working_set.py'
PSUTIL = READY / 'reward-data/load_psutil_runtime_r1.py'
PORTABILITY = Path('D:/personal-agent-system/scripts/portability.py')
HELPER_HASHES = {
    WS.as_posix(): '9dcbf1a43606511f936fadfbe323bde7c597b2a91bf98c9dd99e0204b2a6e1be',
    PSUTIL.as_posix(): '25f20e6e766e5cc53ecfaac5a676c3b8435d713993f5331ebddb937fce729bda',
    PORTABILITY.as_posix(): '5a63306d80767c7676d066cb3d148856f047a09013590ba9c2b21acab9d54684',
}
GIB = 1 << 30
GRACE_SECONDS = 15
CLEANUP_SECONDS = 30
STOP_RESERVE_SECONDS = GRACE_SECONDS + CLEANUP_SECONDS
BASE_SHA = 'abc1aa0b5a63f745804e88789b2dfe6857582ec19ed36b3b5f9dc6b340212344'
NATIVE_SHA = 'b88ec1765cbe4756a7fc3ec95e6b8a0d5d8142f4a0d6c62e5802b6be13bfe4d4'
DRAIN = datetime(2026, 10, 21, 6, 15, tzinfo=timezone.utc)
ASSESSMENT_DRAIN = datetime(2026, 10, 21, 6, 45, tzinfo=timezone.utc)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound(pin):
    path = Path(pin['path'])
    if not path.is_absolute() or sha(path) != pin['sha256']:
        raise ValueError('Pinned input/source mismatch: ' + str(path))
    return read(path) if path.suffix == '.json' else path


def load(name, path, expected_sha256):
    path = Path(path).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError('Helper source changed before direct compilation: ' + path.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(payload, str(path), 'exec'), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def option(argv, flag):
    if argv.count(flag) != 1:
        raise ValueError('Exactly one ' + flag + ' required')
    return argv[argv.index(flag) + 1]


def limits(aggregate_gib):
    if type(aggregate_gib) is not int or aggregate_gib not in (8, 10, 12):
        raise ValueError('Only explicit aggregate8/10/12GiB regimes admitted')
    return {'aggregate_rss_bytes': aggregate_gib * GIB, 'host_reserve_bytes': 2 * GIB,
        'initial_available_bytes': (aggregate_gib + 2) * GIB,
        'main_worker': (aggregate_gib - 2) * GIB, 'rpc_server': 11 * GIB // 8,
        'native_supervisor': GIB // 8, 'rpc_launcher': GIB // 8, 'native_audit': GIB // 8,
        'console_host': 32 << 20}


def stop_reason(rss, available, elapsed, now, policy, deadline):
    if available < policy['host_reserve_bytes']:
        return 'host_available_below2GiB'
    if rss > policy['aggregate_rss_bytes']:
        return 'owned_Job_aggregate_RSS_exceeded'
    if elapsed >= policy['max_seconds'] - STOP_RESERVE_SECONDS or now >= deadline - timedelta(seconds=STOP_RESERVE_SECONDS):
        return 'whole_clock_or_drain_cleanup_boundary'
    return None


def deadline_fits(phase, deadline, native_drain, maximum):
    if phase == 'rl2':
        cutoff = DRAIN
    elif phase in ('heldout-admission', 'heldout-score'):
        cutoff = ASSESSMENT_DRAIN
    else:
        raise ValueError('Only the three declared reviewed RL/heldout phases admitted')
    return (deadline.utcoffset() is not None and deadline <= cutoff and deadline <= native_drain
        and (deadline - datetime.now(timezone.utc)).total_seconds() >= maximum + STOP_RESERVE_SECONDS)


def checked(request_path, policy_approval_path):
    request = read(request_path)
    approved = read(policy_approval_path)
    if (request.get('contract') != 'kion-ready-rl-resource-request-v1'
            or approved.get('verdict') != 'approved' or not approved.get('reviewer')
            or not approved.get('evidence') or approved.get('request_sha256') != canonical(request)):
        raise ValueError('Exact independently approved RL resource request required')
    bound(approved['evidence'])
    plan, approval = bound(request['native_plan']), bound(request['native_approval'])
    adapter = bound(request['native_adapter'])
    bound(request['native_adapter_review'])
    descriptor = bound(request['descriptor'])
    phase = request['phase']
    if phase not in ('rl2', 'heldout-admission', 'heldout-score'):
        raise ValueError('Only the three declared reviewed RL/heldout phases admitted')
    expected_adapter = ADAPTER if phase == 'rl2' else HELDOUT_ADAPTER
    if (adapter.resolve() != expected_adapter.resolve() or approval.get('verdict') != 'approved'
            or approval.get('plan_sha256') != canonical(plan) or not approval.get('reviewer')
            or Path(plan['bundle_dir']).resolve() != COMMON.resolve()
            or plan['source_hashes'].get('scripts/kion_night_supervisor.py') != NATIVE_SHA
            or plan.get('interpreter_sha256') != BASE_SHA or sha(plan['interpreter']) != BASE_SHA
            or len(plan['jobs']) != 1):
        raise ValueError('Exact COMMON/base/adapter/one-job native approval required')
    sources = {Path(__file__).as_posix(): sha(__file__), (HERE / 'native_job.py').as_posix(): sha(HERE / 'native_job.py')}
    for path, expected in {**HELPER_HASHES, **sources, adapter.as_posix(): request['native_adapter']['sha256']}.items():
        if sha(path) != expected or request['source_hashes'].get(path) != expected or plan['input_hashes'].get(path) != expected:
            raise ValueError('Native plan and resource request must pin every exact external helper: ' + path)
    job = plan['jobs'][0]
    argv = job['argv']
    exact = {'--factory': 'kion_complete_rl:build_factory', '--device': 'cuda', '--cpu-threads': '2',
        '--mode': 'rl', '--steps': '100', '--max-updates': '2', '--max-seconds': '120',
        '--batch-size': '1', '--rollout-steps': '1', '--checkpoint-updates': '1'}
    if phase == 'rl2' and (argv[:2] != ['scripts/kion_ready_runtime.py', 'rl'] or any(option(argv, k) != v for k, v in exact.items())
            or '--resume' in argv or job.get('kind') != 'train' or job.get('purpose') != 'plumbing-smoke'
            or job.get('max_updates') != 2 or job.get('receipt_adapter') != 'frozen3ff82-rl-c512-b1h1-max2-v1'
            or option(argv, '--output') != '{output}/model'
            or option(argv, '--stop-file') != '{stop_file}'
            or Path(option(argv, '--bundle')).resolve() != Path(request['descriptor']['path']).resolve()
            or descriptor.get('budget', {}).get('batch_rows') != 1
            or descriptor.get('budget', {}).get('candidate_chunk_size') != 512):
        raise ValueError('Only exact new reviewed RL C512/B1/H1/stopped2 plan supported')
    if phase != 'rl2':
        selector = 'prepare-evaluation' if phase == 'heldout-admission' else 'evaluate'
        if (argv[:3] != ['scripts/kion_ready_runtime.py', 'rl-bridge', selector]
                or job.get('kind') != 'eval' or job.get('purpose') == 'plumbing-smoke'
                or job.get('heldout_phase') != phase or job.get('receipt_adapter') != 'frozen-e533-heldout-operation-v1'
                or job['completion']['equals'].get('state') != 'completed'):
            raise ValueError('Exact independently adapted heldout operation required')
        if phase == 'heldout-admission':
            if (option(argv, '--bundle') != request['descriptor']['path'] or option(argv, '--max-rows') != '64'
                    or option(argv, '--seed') != '2026' or '--rl-checkpoint' not in argv):
                raise ValueError('Explicit learned RL checkpoint and canonical heldout cohort required')
        elif option(argv, '--device') != 'cuda' or option(argv, '--max-seconds') != '600':
            raise ValueError('Exact CUDA paired heldout600s selector required')
    for pin in (request['descriptor'], request['native_adapter_review']):
        if plan['input_hashes'].get(pin['path']) != pin['sha256']:
            raise ValueError('Native learned/adapter review pin missing')
    for key in ('nvidia_smi', 'tasklist', 'console_host'):
        bound(request[key])
        if plan['input_hashes'].get(request[key]['path']) != request[key]['sha256']:
            raise ValueError('Every reachable native helper executable needs an exact native input pin')
    system32 = Path(os.environ['SYSTEMROOT']) / 'System32'
    if (Path(request['tasklist']['path']).resolve() != (system32 / 'tasklist.exe').resolve()
            or Path(request['console_host']['path']).resolve() != (system32 / 'conhost.exe').resolve()):
        raise ValueError('Only exact OS tasklist/conhost paths admitted')
    policy = limits(request['aggregate_rss_gib'])
    maximum = request['max_seconds']
    allowed_clocks = (900, 1200, 1500) if phase != 'heldout-score' else (1800,)
    if type(maximum) is not int or maximum not in allowed_clocks:
        raise ValueError('Use a phase-specific finite whole-process clock')
    if job['max_seconds'] + job['stop_grace_seconds'] + job['cleanup_reserve_seconds'] > maximum - STOP_RESERVE_SECONDS:
        raise ValueError('Outer clock must cover native job, grace, cleanup and precheck margin')
    policy['max_seconds'] = maximum
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    native_drain = datetime.fromisoformat(plan['drain_utc'].replace('Z', '+00:00'))
    if not deadline_fits(phase, deadline, native_drain, maximum):
        raise ValueError('Whole finite budget plus cleanup must fit before native/night drain')
    outputs = [Path(request[key]).resolve() for key in ('run_dir', 'state_dir')] + [Path(job['output']).resolve()]
    protected = [COMMON.resolve(), HERE.resolve(), *[Path(x).resolve() for x in plan['input_hashes']],
        Path(request_path).resolve(), Path(policy_approval_path).resolve()]
    for path in outputs:
        if not path.is_relative_to(RUNS) or path == RUNS or path.exists():
            raise ValueError('Fresh separate ready-runs output directories required')
        if any(path == x or path.is_relative_to(x) or x.is_relative_to(path) for x in protected):
            raise ValueError('Output/input/source overlap')
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise ValueError('Resource/native state/job outputs must be separate siblings')
    return request, plan, job, descriptor, policy, deadline


def role(identity, argv, context):
    image = Path(identity['exe']).resolve()
    if image == Path(context['interpreter']).resolve() and argv == context['supervisor_argv']:
        return 'native_supervisor'
    if image == Path(context['interpreter']).resolve() and argv == context['rl_argv']:
        return 'main_worker'
    rpc = context['rpc_argv']
    if rpc and argv[1:] == rpc[1:] and Path(argv[0]).resolve() in (Path(rpc[0]).resolve(), Path(context['interpreter']).resolve()):
        if image == Path(rpc[0]).resolve():
            return 'rpc_launcher'
        if image == Path(context['interpreter']).resolve():
            return 'rpc_server'
    if image == Path(context['nvidia_smi']).resolve() and argv[1:] in context['audit_argv_tails']:
        return 'native_audit'
    if image == Path(context['tasklist']).resolve() and argv[1:] == context['tasklist_argv_tail']:
        return 'native_audit'
    if image == Path(context['console_host']).resolve() and len(argv) == 2 and re.fullmatch(r'0x[0-9a-fA-F]+', argv[1]):
        spelling = argv[0][4:] if argv[0].startswith('\\??\\') else argv[0]
        if Path(spelling).resolve() == image:
            return 'console_host'
    raise ValueError('Unreviewed executable/argv inside owned Job')


class Observer:
    def __init__(self, api, psutil, ws, context, policy):
        self.api, self.psutil, self.ws, self.context, self.policy = api, psutil, ws, context, policy
        self.seen = {}

    def sample(self):
        rss, private, roles = 0, 0, {}
        for pid in self.api.members():
            handle = None
            try:
                try:
                    handle = self.api.open(pid)
                except ProcessLookupError:
                    raise  # Positive vanished-member exception; outer handler skips it.
                except OSError:
                    # Open failed before we obtained a stable handle. A native
                    # error number alone cannot distinguish death from failure.
                    # Only a positive NoSuchProcess lookup permits omission.
                    try:
                        self.psutil.Process(pid)
                    except self.psutil.NoSuchProcess:
                        continue
                    raise
                identity = self.api.identity(handle)
                info = self.api.memory(handle)
                key = (pid, identity['creation_time_100ns'])
                if key not in self.seen:
                    member = self.psutil.Process(pid)
                    argv, parent = member.cmdline(), member.ppid()
                    tag = role(identity, argv, self.context)
                    expected = {**identity, 'parent_pid': parent, 'argv': argv}
                    query = lambda wanted, m=member: {'pid': wanted, 'parent_pid': m.ppid(), 'argv': m.cmdline()}
                    cap = self.ws.apply_verified(expected, self.policy[tag], api=self.api, query=query)
                    if len(self.seen) >= 64:
                        raise RuntimeError('Bounded process-observation inventory exceeded')
                    self.seen[key] = {'role': tag, 'identity': identity, 'working_set': cap}
                tag = self.seen[key]['role']
                roles[tag] = roles.get(tag, 0) + 1
                if roles[tag] > (4 if tag == 'console_host' else 1):
                    raise RuntimeError('More than one live process in an admitted role')
                rss += info['rss']
                private += info['private']
            except ProcessLookupError:
                pass
            except OSError:
                # Memory/query/setter failure for a live or unverified member
                # must abort the guard, even for87/1168. Recheck only the
                # original already-open handle to prove an actual exit race.
                if handle is None or not self.api.exited(handle):
                    raise
            except self.psutil.NoSuchProcess:
                pass
            finally:
                if handle is not None:
                    self.api.close(handle)
        return {'rss': rss, 'private': private, 'roles': roles}


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    os.replace(temporary, path)


def supervise(args):
    started = time.perf_counter()
    request, plan, job, descriptor, policy, deadline = checked(args.request, args.approval)
    source_pins = request['source_hashes']
    psutil = load('_rl_guard_pinned_psutil', PSUTIL, source_pins[PSUTIL.as_posix()]).load()
    portability = load('_rl_guard_pinned_portability', PORTABILITY, source_pins[PORTABILITY.as_posix()])
    ws = load('_rl_guard_pinned_ws', WS, source_pins[WS.as_posix()])
    native = load('_rl_guard_job', HERE / 'native_job.py', source_pins[(HERE / 'native_job.py').as_posix()])
    if os.name != 'nt':
        raise OSError('Windows native resource guard required')
    available = psutil.virtual_memory().available
    if available < policy['initial_available_bytes']:
        raise MemoryError('Need aggregate budget plus2GiB headroom before native spawn')
    command = [plan['interpreter'], request['native_adapter']['path'], 'run', '--plan', request['native_plan']['path'],
        '--approval', request['native_approval']['path'], '--state-dir', request['state_dir']]
    output = Path(job['output'])
    expanded = [x.replace('{output}', output.as_posix()).replace('{stop_file}', (output / 'stop-request.json').as_posix()) for x in job['argv']]
    context = {'interpreter': plan['interpreter'], 'supervisor_argv': command,
        'rl_argv': [plan['interpreter'], *expanded], 'rpc_argv': request.get('rpc_argv', []),
        'nvidia_smi': request['nvidia_smi']['path'], 'audit_argv_tails': request['audit_argv_tails'],
        'tasklist': request['tasklist']['path'], 'console_host': request['console_host']['path'],
        'tasklist_argv_tail': ['/FI', f"PID eq {plan['protected_dwm_pid']}", '/V', '/FO', 'CSV', '/NH']}
    bound(request['nvidia_smi'])
    rpc = context['rpc_argv']
    expected_script = (COMMON / 'scripts/kion_reward_boosting.py').as_posix()
    if request['phase'] == 'rl2' and (len(rpc) != 7 or rpc[0] != descriptor['catboost_python'] or rpc[1:4] != [expected_script, 'serve', '--artifact']
            or rpc[5] != '--expected-mapping-sha256' or len(rpc[6]) != 64
            or Path(rpc[4]).resolve() != Path(descriptor['files']['rm_model_manifest']['path']).parent.resolve()):
        raise ValueError('Only the declared exact CatBoost RPC command admitted')
    if request['phase'] != 'rl2' and rpc:
        raise ValueError('Heldout phases admit no RM/RPC process')
    if request['audit_argv_tails'] != AUDIT_TAILS:
        raise ValueError('Only frozen native read-only GPU audit tails admitted')
    run_dir = Path(request['run_dir'])
    run_dir.mkdir()
    operation = {'contract': 'kion-ready-RL-resource-operation-v1', 'state': 'running', 'policy': policy,
        'request_sha256': canonical(request), 'native_plan_sha256': canonical(plan),
        'whole_clock_includes': 'all source/learned verification, cold providers, RPC, actor loop, save and cleanup',
        'poll_seconds': .1, 'discovery_cap_window': 'sampled, plus identity/role/cap work; not an absolute commit limit',
        'peak_tree_rss_bytes': 0, 'minimum_host_available_bytes': available, 'scientific_fit_completed': False}
    process, observer, error, stop_at = None, None, None, None
    try:
        with (run_dir / 'native-cli.log').open('xb') as log:
            if psutil.virtual_memory().available < policy['initial_available_bytes']:
                raise MemoryError('Headroom fell before owned native spawn')
            process = portability.popen_isolated(command, cwd=COMMON, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            observer = Observer(native.JobAPI(process._job, ws), psutil, ws, context, policy)
            operation['owned_native_cli_pid'] = process.pid
            while True:
                snapshot = observer.sample()
                available = psutil.virtual_memory().available
                elapsed = time.perf_counter() - started
                operation['peak_tree_rss_bytes'] = max(operation['peak_tree_rss_bytes'], snapshot['rss'])
                operation['minimum_host_available_bytes'] = min(operation['minimum_host_available_bytes'], available)
                operation['last_sample'] = snapshot
                reason = stop_reason(snapshot['rss'], available, elapsed, datetime.now(timezone.utc), policy, deadline)
                if reason and stop_at is None:
                    operation['stop_reason'], stop_at = reason, time.perf_counter()
                    if output.is_dir():
                        stop_file = output / 'stop-request.json'
                        if not stop_file.exists():
                            with stop_file.open('x', encoding='utf-8') as stream:
                                json.dump({'reason': 'outer_resource_' + reason}, stream)
                if stop_at is not None and time.perf_counter() - stop_at >= GRACE_SECONDS:
                    raise RuntimeError('Resource guard stopped owned native Job: ' + operation['stop_reason'])
                if process.poll() is not None:
                    if stop_at is not None or process.returncode:
                        raise RuntimeError('Native CLI stopped/failed under resource guard')
                    break
                if (run_dir / 'native-cli.log').stat().st_size > 8 << 20:
                    raise RuntimeError('Outer native CLI log cap exceeded')
                operation['elapsed_seconds'] = elapsed
                operation['observed_processes'] = list(observer.seen.values())
                write(run_dir / 'operation.json', operation)
                time.sleep(.1)
        status = read(Path(request['state_dir']) / 'status.json')
        matches = [a for a in status['attempts'] if a.get('plan_sha256') == canonical(plan) and a.get('id') == job['id']]
        if len(matches) != 1 or matches[0].get('state') != 'completed' or matches[0].get('cleanup') != 'quiescent':
            raise RuntimeError('Frozen native completion/owned cleanup did not pass')
        operation.update(state='completed_resource_and_native_boundary', native_status_sha256=sha(Path(request['state_dir']) / 'status.json'))
    except BaseException as caught:
        error = caught
        operation.update(state='failed', error=type(caught).__name__ + ': ' + str(caught))
    finally:
        if process is not None:
            try:
                portability.terminate_process_tree(process)
                operation['owned_Job_cleanup'] = 'quiescent'
            except BaseException as caught:
                operation.update(state='failed', cleanup_error=str(caught))
                error = error or caught
        operation['elapsed_seconds'] = time.perf_counter() - started
        if observer is not None:
            operation['observed_processes'] = list(observer.seen.values())
        write(run_dir / 'operation.json', operation)
    if error is not None:
        raise error
    print(json.dumps({'state': operation['state'], 'operation': (run_dir / 'operation.json').as_posix()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        checked(args.request, args.approval)
        print(json.dumps({'state': 'metadata_checked_native_check_still_required'}))
    else:
        supervise(args)
