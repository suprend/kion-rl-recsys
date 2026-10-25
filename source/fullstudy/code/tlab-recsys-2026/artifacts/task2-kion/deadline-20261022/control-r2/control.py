"""New deadline-only metadata; unchanged owned Job and singlepass lifecycle."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD = ROOT.parent / 'quality-recovery-20261021'
OLD_CONTROL = OLD / 'uniform-control-r1/control.py'
OLD_CONTROL_SHA = 'f9f226ece0294b0c5402ae1af454c9d61833d0fa4be904a601955da9c80e4cd6'
# This import compiles only reviewed stdlib metadata and its unchanged guard.
import hashlib
import importlib.util

def direct(path, digest, name):
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError('Exact source changed: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module

old = direct(OLD_CONTROL, OLD_CONTROL_SHA, '_d22_reviewed_singlepass_metadata')
read, sha, require, g = old.read, old.sha, old.require, old.g
READY, COMMON, BASE_PLAN, BASE_SHA, BASE_CONFIG = old.READY, old.COMMON, old.BASE_PLAN, old.BASE_SHA, old.BASE_CONFIG
CONFIG_SHA = BASE_CONFIG_SHA = old.CONFIG_SHA
GUARD, GUARD_SHA, JOB_SHA = old.GUARD, old.GUARD_SHA, old.JOB_SHA
NEGATIVE = ROOT / 'negative-worker-r2'
HOOKS, WORKER = NEGATIVE / 'hooks.py', NEGATIVE / 'worker.py'
ADAPTER = HERE / 'native_adapter.py'
OUTPUT, STATE, RESOURCE = [HERE / name for name in ('job', 'state', 'resource')]
KIND = 'kion-D22-retrieval128-uniform3-E1-singlepass-v1'
BENCH_KIND = 'kion-D22-retrieval128-cacheOFF-ON-36update-v1'
SAMPLER_CONTRACT = 'retrieval128-uniform3-forced-training-positive-v1'
CONTRACT = ROOT / 'contract-r1.json'
CONTRACT_SHA = '065c5d84f286500b511eba7d5fd9373b81bd361e73653c99f1516ba6ae453b52'
STOP = datetime(2026, 10, 22, 11, tzinfo=timezone.utc)
CACHE = ROOT / 'perf-audit-r1/dynamic_feature_cache.py'
MIGRATION = HERE / 'runtime-migration.json'
MIGRATION_SHA = '4018e1107a10427caca41ddce817b5bca38a2f166cbf31cefeac62e1bedf69ac'
NEW_INTERPRETER_SHA = 'b141a0115de4be263a5cb0bdeb580ab67d074aaf1aadf10622fdf88ac2a1d8cb'
CACHE_SHA = '3a9ba56b34561a3e129a0f116159e25deb6e3b2f81a1680cef0e770071f0b611'
sampler = direct(NEGATIVE / 'sampler.py', '71d8747bfb080bc6ea82059c28b899e5c1bd266649deb9943de80de431933c44', '_d22_sampler_metadata')


def load(path, digest, name):
    module = direct(path, digest, name)
    if Path(path).resolve() == COMMON / 'scripts/kion_complete_train.py':
        request = read(HERE / 'worker-request.json')
        perf = performance(request['runtime_performance'])
        if perf['dynamic_cache']:
            module._negative_dynamic_cache_install = direct(CACHE, CACHE_SHA, '_d22_admitted_cache').install_dynamic_cache
    return module


def performance(value):
    require(type(value) is dict and set(value) == {'cpu_threads', 'dynamic_cache'}
        and type(value['cpu_threads']) is int and value['cpu_threads'] == 2
        and type(value['dynamic_cache']) is bool, 'CPU2 fixed; exact selected cache flag only')
    return value


def specification():
    result = old.specification()
    result.update(contract=KIND, scientific_delta_from_scheduler_control={'negative_proposal': sampler.specification()},
        sampler_contract=SAMPLER_CONTRACT, sampler_specification=sampler.specification(),
        runtime_migration={'path': MIGRATION.as_posix(), 'sha256': MIGRATION_SHA},
        absolute_goal_stop_utc=STOP.isoformat(), reserve_after_phase_seconds=1800,
        performance_selection={'cpu_threads': 2, 'dynamic_cache_allowed': [False, True],
            'requires_actual_cacheOFF_ON_benchmark_and_raw_feature_parity': True},
        prerequisites='new actual paired cache benchmark plus root decision; old stopped export is not required')
    return result


def sources():
    paths = [Path(__file__), HERE / 'probe.py', ADAPTER, HERE / 'benchmark_bridge.py',
        NEGATIVE / 'sampler.py', HOOKS, WORKER, NEGATIVE / 'benchmark.py', CACHE,
        OLD_CONTROL, OLD / 'uniform-control-r1/worker.py', OLD / 'uniform-control-r1/hooks.py',
        OLD / 'pair-probe-r1/probe.py', OLD / 'pair-probe-r1/worker.py']
    return {**g.HELPER_HASHES, GUARD.as_posix(): GUARD_SHA,
        GUARD.with_name('native_job.py').as_posix(): JOB_SHA,
        **{path.as_posix(): sha(path) for path in paths}}


def train_argv(deadline):
    argv = old.train_argv(deadline)
    argv[argv.index('--output') + 1] = (OUTPUT / 'model').as_posix()
    argv[argv.index('--stop-file') + 1] = (OUTPUT / 'stop-request.json').as_posix()
    return argv


def worker_argv(phase='train'):
    if phase == 'benchmark':
        return [(NEGATIVE / 'benchmark.py').as_posix(), '--worker-request', (HERE / 'benchmark-worker-request.json').as_posix(),
            '--output', '{output}/benchmark', '--stop-file', '{stop_file}']
    return [WORKER.as_posix(), '--request', (HERE / 'worker-request.json').as_posix()]


def completion(phase='train'):
    if phase == 'benchmark':
        return {'receipt': 'benchmark/completion.json', 'equals': {'state': 'completed', 'operation': BENCH_KIND,
            'quality_claim': False, 'production_checkpoint': False, 'validation_evaluated': False,
            'validation_rows_used_for_updates': 0, 'test_accessed': False, 'sampler_contract': SAMPLER_CONTRACT},
            'finite': ['accepted_updates'], 'artifacts': ['benchmark/report.json', 'benchmark/fixture.json']}
    result = old.completion()
    result['equals'].update(operation=KIND, sampler_contract=SAMPLER_CONTRACT, sampler_binding_verified=True,
        sampler_specification=sampler.specification())
    return result


def paths(phase):
    require(phase in ('train', 'benchmark'), 'only two explicit new phases')
    return (OUTPUT, STATE, RESOURCE) if phase == 'train' else tuple(HERE / ('benchmark-' + name) for name in ('job', 'state', 'resource'))


def phase_spec(phase):
    return specification() if phase == 'train' else direct(HERE / 'probe.py', sha(HERE / 'probe.py'), '_d22_probe_spec').specification()


def validate_decision(value, phase):
    require(value.get('contract') == 'kion-D22-root-decision-v1' and value.get('state') == 'approved'
        and value.get('phase') == phase and value.get('sampler_specification') == sampler.specification()
        and value.get('runtime_migration') == {'path': MIGRATION.as_posix(), 'sha256': MIGRATION_SHA}
        and value.get('sampler_contract') == SAMPLER_CONTRACT
        and value.get('goal_contract') == {'path': CONTRACT.as_posix(), 'sha256': CONTRACT_SHA}, 'explicit fresh root scientific decision bound to goal and sampler')
    if phase == 'benchmark':
        require(value.get('decision') == 'admit-disposable-cache-benchmark' and value.get('cpu_threads') == 2,
            'root benchmark admission CPU2')
        return 780
    require(value.get('decision') == 'admit-fresh-retrieval128-uniform3-singlepass-E1', 'explicit complete E1 decision')
    performance(value['runtime_performance'])
    phases = value['point_phase_seconds']
    require(set(phases) == {'train', 'full_validation', 'checkpoint', 'cold_init', 'sidecar_and_provenance'}
        and all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in phases.values()), 'all fullE1 phases positive measured estimates')
    margin = value['timing_margin_ratio']
    require(type(margin) in (int, float) and math.isfinite(margin) and margin >= .15, 'minimum15% measured margin')
    estimate = math.ceil(sum(phases.values()) * (1 + margin))
    require(estimate <= 23910, 'complete E1 with all phases and margin fits native cap')
    return estimate


def lease(deadline_text, native_deadline_text, phase, *, now=None):
    deadline = datetime.fromisoformat(deadline_text.replace('Z', '+00:00'))
    native = datetime.fromisoformat(native_deadline_text.replace('Z', '+00:00'))
    seconds = 24000 if phase == 'train' else 900
    require(deadline.utcoffset() is not None and native == deadline + timedelta(seconds=60)
        and native <= STOP - timedelta(seconds=1800)
        and (deadline - (now or datetime.now(timezone.utc))).total_seconds() >= seconds + 45,
        'entire fixed phase/cleanup and1800s reserve fit new11UTC absolute cutoff')
    return deadline


def migrated_inputs(baseline):
    result = dict(baseline['input_hashes'])
    exe = baseline['interpreter']
    require(result.get(exe) == baseline['interpreter_sha256'], 'original input interpreter pin exactly matches original baseline')
    result[exe] = NEW_INTERPRETER_SHA
    return result

def validate_runtime(plan, baseline):
    require(all(plan[k] == baseline[k] for k in ('bundle_dir', 'source_hashes', 'interpreter', 'gpu_uuid')), 'unchanged original bundle/source/interpreter path/GPU closure')
    require(plan['interpreter_sha256'] == NEW_INTERPRETER_SHA and sha(plan['interpreter']) == NEW_INTERPRETER_SHA
        and sha(MIGRATION) == MIGRATION_SHA, 'explicit reviewed current interpreter bytes; real native verifier remains unchanged')
    require(all(plan['input_hashes'].get(path) == digest for path, digest in migrated_inputs(baseline).items()), 'whole original input closure except sole documented interpreter digest migration')


def validate_plan(plan, request):
    phase = request['phase']
    output, state, resource = paths(phase)
    kind = KIND if phase == 'train' else BENCH_KIND
    require(request.get('state') == 'admitted' and request['contract'] == kind
        and request['specification'] == phase_spec(phase), 'non-template exact admitted specification')
    baseline = read(BASE_PLAN)
    require(sha(BASE_PLAN) == BASE_SHA and sha(BASE_CONFIG) == CONFIG_SHA and sha(CONTRACT) == CONTRACT_SHA,
        'exact baseline/goal metadata pins')
    validate_runtime(plan, baseline)
    for p, digest in ((BASE_PLAN, BASE_SHA), (BASE_CONFIG, CONFIG_SHA), (CONTRACT, CONTRACT_SHA), (MIGRATION, MIGRATION_SHA)):
        require(plan['input_hashes'].get(p.as_posix()) == digest, 'direct baseline/goal metadata pin')
    require(plan['train_cap'] == 1 and plan['eval_cap'] == 0 and plan['retry_cap'] == 0
        and plan['prior_jobs'] == {} and len(plan['jobs']) == 1, 'one owned operation no retry')
    decision = g.bound(request['root_decision'])
    estimate = validate_decision(decision, phase)
    job = plan['jobs'][0]
    require(job['id'] == 'D22-' + phase + '-r2' and job['kind'] == 'train'
        and job['purpose'] == 'D22-' + phase and job.get('receipt_adapter') == kind
        and job['argv'] == worker_argv(phase) and job['completion'] == completion(phase)
        and job['depends_on'] == [] and job.get('retry_of') is None, 'exact fixed new worker/receipt')
    native_cap, outer_cap = (23910, 24000) if phase == 'train' else (810, 900)
    require(job['estimate_seconds'] == estimate and job['max_seconds'] == native_cap and job['stop_grace_seconds'] == 15
        and job['cleanup_reserve_seconds'] == 30 and job['disk_required_bytes'] == 6 << 30
        and job['disk_reserve_bytes'] == 10 << 30 and job['log_cap_bytes'] == 8 << 20, 'unchanged exact native caps')
    require(Path(job['output']).resolve() == output and Path(request['run_dir']).resolve() == resource
        and Path(request['state_dir']).resolve() == state and request['max_seconds'] == outer_cap
        and request['aggregate_rss_gib'] == 12 and request.get('rpc_argv') == [], 'fixed fresh resources12GiB')
    require(plan['drain_utc'] == request['deadline_utc'], 'same worker/resource/native drain lease')
    return job, lease(request['deadline_utc'], plan['deadline_utc'], phase)


def validate_benchmark(value, *, require_match=True):
    require(value.get('state') == 'completed' and value.get('operation') == BENCH_KIND
        and value.get('sampler_contract') == SAMPLER_CONTRACT and value.get('sampler_specification') == sampler.specification()
        and value.get('validation_evaluated') is False and value.get('validation_rows_used_for_updates') == 0
        and value.get('quality_claim') is False and value.get('production_checkpoint') is False, 'honest new benchmark semantics')
    if require_match:
        audit, perf = value['sampler_audit'], value['runtime_performance']
        require(value.get('benchmark_match') is True and value.get('accepted_updates') == 36
            and audit['baseline_calls'] == 0 and audit['treatment_calls'] >= 36
            and audit['treatment_rows'] == audit['treatment_calls'] * 128 and audit['support_checks_passed'] is True,
            'complete actual new-sampler36 updates and support checks')
        require(perf['cpu_threads'] == 2 and perf['cache_cells'] == ['OFF', 'ON', 'ON', 'OFF', 'OFF', 'ON']
            and perf['raw_dynamic_parity_passed'] is True and perf['full_candidate_parity_passed'] is True
            and perf['parity_cells'] == 3 and perf['same_seed_candidate_hashes_match'] is True, 'actual counterbalanced cache parity before fullrun selection')
    return value


def checked(request_path, approval_path, *, running=False):
    request, approval = read(request_path), read(approval_path)
    require(approval['verdict'] == 'approved' and approval.get('reviewer')
        and approval['request_sha256'] == g.canonical(request), 'independent exact resource approval')
    g.bound(approval['evidence'])
    plan, native_approval = g.bound(request['native_plan']), g.bound(request['native_approval'])
    require(native_approval['verdict'] == 'approved' and native_approval.get('reviewer')
        and native_approval['plan_sha256'] == g.canonical(plan), 'independent exact native approval')
    job, deadline = validate_plan(plan, request)
    phase = request['phase']
    output, state, resource = paths(phase)
    require(job['uniform_control'] == {'request': Path(request_path).resolve().as_posix(), 'approval': Path(approval_path).resolve().as_posix()}, 'exact approved resource references')
    require(request['source_hashes'] == sources(), 'exact full execution source closure')
    for path, digest in sources().items():
        require(sha(path) == digest and plan['input_hashes'].get(path) == digest, 'every execution source directly pinned')
    require(request['native_adapter'] == {'path': ADAPTER.as_posix(), 'sha256': sha(ADAPTER)}, 'fixed adapter')
    for key in ('root_decision', 'native_adapter_review', 'worker_request', 'native_quiet'):
        g.bound(request[key])
        require(plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'direct bound decision/review/worker/quiet pin')
    reviewed = g.bound(request['native_adapter_review'])
    require(reviewed.get('verdict') == 'approved' and reviewed.get('reviewer')
        and all(reviewed.get('source_hashes', {}).get(path) == digest for path, digest in sources().items()), 'independent exact execution source review')
    worker = g.bound(request['worker_request'])
    filename = 'worker-request.json' if phase == 'train' else 'benchmark-worker-request.json'
    require(Path(request['worker_request']['path']).resolve() == HERE / filename and worker.get('state') == 'admitted'
        and worker['specification'] == request['specification'] and worker['source_hashes'] == request['source_hashes']
        and worker['deadline_utc'] == request['deadline_utc'], 'same admitted exact worker/deadline/source metadata')
    decision = g.bound(request['root_decision'])
    if phase == 'train':
        require(worker['train_argv'] == train_argv(request['deadline_utc'])
            and performance(worker['runtime_performance']) == performance(decision['runtime_performance']), 'exact full scientific argv and measured cache selection')
        require(set(request['prerequisite_evidence']) == {'benchmark', 'benchmark_resource'}, 'new actual benchmark only; no old export requirement')
        benchmark = g.bound(request['prerequisite_evidence']['benchmark'])
        validate_benchmark(benchmark)
        require(Path(request['prerequisite_evidence']['benchmark']['path']).resolve() == paths('benchmark')[0] / 'benchmark/completion.json', 'fixed actual new benchmark')
        require(benchmark.get('execution_source_hashes') == request['source_hashes'], 'new benchmark exact execution source closure')
        for name in ('report.json', 'fixture.json'):
            require(benchmark['sha256'][name] == sha(paths('benchmark')[0] / 'benchmark' / name), 'actual benchmark committed artifact hash')
        boundary = g.bound(request['prerequisite_evidence']['benchmark_resource'])
        require(Path(request['prerequisite_evidence']['benchmark_resource']['path']).resolve() == paths('benchmark')[2] / 'operation.json'
            and boundary['state'] == 'completed_resource_and_native_boundary' and boundary['owned_Job_cleanup'] == 'quiescent'
            and boundary['native_status_sha256'] == sha(paths('benchmark')[1] / 'status.json'), 'new benchmark owned native cleanup passed')
        native_status = read(paths('benchmark')[1] / 'status.json')
        attempts = native_status['attempts']
        require(len(attempts) == 1 and attempts[0]['state'] == 'completed' and attempts[0]['cleanup'] == 'quiescent'
            and attempts[0]['receipt'] == benchmark, 'actual native completed receipt equals bound benchmark and cleanup')
        for pin in request['prerequisite_evidence'].values():
            require(plan['input_hashes'].get(pin['path']) == pin['sha256'], 'direct new actual benchmark evidence pin')
    else:
        require(request['prerequisite_evidence'] == {} and worker['baseline_plan'] == {'path': BASE_PLAN.as_posix(), 'sha256': BASE_SHA}
            and worker['baseline_config'] == {'path': BASE_CONFIG.as_posix(), 'sha256': CONFIG_SHA}, 'new standalone bounded benchmark exact baseline pins')
    quiet = g.bound(request['native_quiet'])
    stamp = datetime.fromisoformat(quiet['updated_utc'].replace('Z', '+00:00'))
    require(quiet['state'] == 'quiet' and 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 120
        and all(plan[k] == quiet[k] for k in ('gpu_uuid', 'protected_dwm_pid', 'benign_gpu_processes')), 'fresh quiet physical identities')
    helpers = read(READY.parent / 'ready-runs-20261020/sft-finalize-plan-r1/resource-request.json')
    for key in ('nvidia_smi', 'tasklist', 'console_host'):
        g.bound(request[key])
        require(request[key] == helpers[key] and plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'same accepted audit executables')
    require(request['audit_argv_tails'] == g.AUDIT_TAILS, 'same audit commands')
    for target in (output, state, resource):
        require((running and target == resource and target.is_dir()) or not target.exists(), 'fresh owned output')
        require(all(target != Path(p) and not target.is_relative_to(Path(p)) and not Path(p).is_relative_to(target)
            for p in plan['input_hashes']), 'outputs separate from inputs')
    cap = 24000 if phase == 'train' else 900
    return request, plan, job, {}, {**g.limits(12), 'main_worker': 11274289152, 'max_seconds': cap}, deadline


def validate_result(job):
    if job.get('purpose') == 'D22-benchmark':
        value = read(Path(job['output']) / 'benchmark/completion.json')
        validate_benchmark(value, require_match=False)
        require(value.get('execution_source_hashes') == sources(), 'benchmark exact executed source closure')
        return value
    hook = old.validate_result(job)
    audit = hook.get('sampler_audit', {})
    require(hook.get('sampler_contract') == SAMPLER_CONTRACT and hook.get('sampler_specification') == sampler.specification()
        and hook.get('sampler_binding_verified') is True and type(audit.get('calls_including_amp_retries')) is int
        and audit['calls_including_amp_retries'] >= 27235 and audit.get('rows_including_amp_retries', 0) >= 3477437
        and audit.get('candidate_width') == 4 and audit.get('random_width') == 128, 'actual new sampler binding/counts including AMP retries')
    request = read(HERE / 'worker-request.json')
    perf = hook.get('runtime_performance', {})
    require(perf.get('cpu_threads') == 2 and perf.get('dynamic_cache') == request['runtime_performance']['dynamic_cache'], 'actual selected performance mode')
    receipt_path = Path(job['output']) / 'singlepass-completion.json'
    if receipt_path.exists():
        value = read(receipt_path)
        # First old-worker validation precedes receipt creation; all existing final receipts must be enriched.
        for key in ('sampler_contract', 'sampler_specification', 'sampler_audit', 'sampler_binding_verified', 'runtime_performance'):
            require(value.get(key) == hook.get(key), 'operation receipt exactly matches actual hook: ' + key)
        require(value.get('execution_source_hashes') == request['source_hashes'], 'operation execution source closure')
    return hook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'run'))
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--approval', type=Path, required=True)
    args = parser.parse_args()
    checked(args.request, args.approval)
    if args.command == 'check':
        print(json.dumps({'state': 'metadata_checked_no_launch'}))
        return
    g.checked = checked
    return g.supervise(args)

if __name__ == '__main__':
    main()
