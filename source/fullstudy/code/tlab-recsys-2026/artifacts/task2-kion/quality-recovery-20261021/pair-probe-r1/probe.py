"""Metadata-only contract and existing owned-resource admission for sampler timing."""
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
READY = Path('D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020')
COMMON = READY / 'provider/runtime-native-r5'
BASE_PLAN = READY.parent / 'ready-runs-20261020/sft-plan-r1/plan.json'
BASE_SHA = '4c631fbca534eff09de9861596a6bc78a5a09f1cbaaf4fd55282875f2deb3c38'
BASE_CONFIG = READY.parent / 'ready-runs-20261020/sft-job-r1/model/config.json'
BASE_CONFIG_SHA = '71cc784f10269f196d4ad2c0a6f34f461d5c18b6bbd0b0c89d8d4b04f26141b3'
METHOD = ROOT / 'methods-r1/next-pair-plan-r1.json'
METHOD_SHA = 'd5ba7b2379aac07897fff9108724f3b75ad9f7101da36f00b2cbde314cd9ed4f'
GUARD = READY / 'model/rl-native-resource-r1/guard.py'
GUARD_SHA = 'b1bee55f2ec947c188bc8f6065d66b08bfce85b16a2bc7b631d72a8e3b785e87'
JOB_HELPER_SHA = '0da87bf737038e08602ddaf18929a9441543269adf0edd191f1ac4151e916296'
WORKER = HERE / 'worker.py'
ADAPTER = HERE / 'native_adapter.py'
KIND = 'kion-C4-negative-sampler-disposable-benchmark-v1'
CELLS = [{'id': 'reference-H2-U1', 'hard_negatives': 2, 'uniform_negatives': 1, 'auxiliary_weight': 1.0},
    {'id': 'uniform-H0-U3', 'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.0}]
ORIGINS = [737863, 737937, 738010]
ORDERS = [[CELLS[0]['id'], CELLS[1]['id']], [CELLS[1]['id'], CELLS[0]['id']], [CELLS[0]['id'], CELLS[1]['id']]]
GROUPS = ['warm-mixed', 'warm-mixed', 'short', 'medium', 'long', 'measured-mixed']
OUTPUT, STATE, RESOURCE = [HERE / name for name in ('job', 'state', 'resource')]


def require(condition, message):
    if not condition:
        raise ValueError('Sampler probe: ' + message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, digest, name):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, 'exact direct source bytes: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


g = load(GUARD, GUARD_SHA, '_pair_probe_unchanged_resource_guard')


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def specification():
    return {'contract': KIND, 'state': 'source-only-not-admitted', 'seed': 2026, 'cells': CELLS,
        'origins': ORIGINS, 'origin_cell_orders': ORDERS, 'group_labels': GROUPS,
        'fixture': {'batch_size': 128, 'groups_per_origin': 6, 'warm_groups': 2, 'measured_groups': 4,
            'eligible': 'known-target training rows only; original frozen TrainingSchedule origin_rows',
            'length_strata': 'stable length,row-index sort then equal-count rank thirds; no quality/label selection beyond original known-target eligibility',
            'mixed_counts': [43, 43, 42], 'seed': 'SHA256(2026:origin:stratum:row) ascending; independent priority order in each rank-third',
            'row_groups': 'disjoint128 groups: two mixed warm; pure short/medium/long; one mixed measured',
            'same_fixture_both_cells': True},
        'updates_per_cell': 18, 'total_updates': 36, 'measured_updates_per_cell': 12,
        'fresh_model_optimizer_scaler_per_cell_origin': True, 'provider_loads': 1,
        'provider_reuse': 'same immutable raw provider; no model object/weights/optimizer passed into raw-feature callbacks; deterministic raw caches shared and order counterbalanced AB/BA/AB',
        'batch_size': 128, 'accumulation': 1, 'candidate_chunk_size': 64, 'sampled_candidates': 4,
        'cpu_threads': 2, 'amp': 'unchanged frozen train.update and legacy.autocast; CUDA GradScaler init_scale1024',
        'lr': .0001, 'warmup_ratio': .05, 'weight_decay': .01,
        'lr_schedule': 'frozen shared.learning_rate(local-update0..5,total136175,warmup.05,lr1e-4); reset with each fresh model',
        'schedule_total_updates': 136175, 'worker_wall_seconds': 900, 'outer_wall_seconds': 900,
        'aggregate_rss_gib': 12, 'main_worker_ws_bytes': 11274289152, 'host_reserve_gib': 2,
        'initial_available_gib': 14, 'production_checkpoint': False, 'provider_loads_validation': True,
        'validation_evaluation': False, 'validation_rows_used_for_updates': 0,
        'quality_claim': False, 'no_fallback_or_whole_update_retry': True,
        'method_pin': {'path': METHOD.as_posix(), 'sha256': METHOD_SHA}}


def validate_spec(value):
    require(value == specification(), 'exact fixed two-cell36-update fixture/resources/semantics')
    require(sha(METHOD) == METHOD_SHA and sha(BASE_PLAN) == BASE_SHA, 'original methods/baseline metadata')


def fixture_rows(rows, lengths, origin):
    """Pure metadata algorithm, testable on synthetic lists without NumPy."""
    require(len(rows) == len(lengths) and len(set(map(int, rows))) == len(rows), 'aligned unique origin fixture rows')
    ordered = [row for _, row in sorted((int(length), int(row)) for row, length in zip(rows, lengths))]
    n, remainder = divmod(len(ordered), 3)
    widths = [n + (index < remainder) for index in range(3)]
    pools, begin = [], 0
    for index, width in enumerate(widths):
        pool = ordered[begin:begin + width]
        pools.append(sorted(pool, key=lambda row: (hashlib.sha256(f'2026:{origin}:{index}:{row}'.encode()).hexdigest(), row)))
        begin += width
    cursor = [0, 0, 0]

    def take(counts):
        chosen = []
        for index, count in enumerate(counts):
            require(cursor[index] + count <= len(pools[index]), 'enough disjoint rows in each length stratum; no fallback')
            chosen.extend(pools[index][cursor[index]:cursor[index] + count])
            cursor[index] += count
        require(len(chosen) == 128, 'exactB128')
        return chosen

    result = [take([43, 43, 42]), take([43, 43, 42]), take([128, 0, 0]),
        take([0, 128, 0]), take([0, 0, 128]), take([43, 43, 42])]
    require(len(set(sum(result, []))) == 768, 'six disjoint groups')
    return result


def sources():
    return {**g.HELPER_HASHES, GUARD.as_posix(): GUARD_SHA,
        GUARD.with_name('native_job.py').as_posix(): JOB_HELPER_SHA,
        Path(__file__).as_posix(): sha(__file__), WORKER.as_posix(): sha(WORKER), ADAPTER.as_posix(): sha(ADAPTER)}


def expected_argv():
    return [WORKER.as_posix(), '--worker-request', (HERE / 'worker-request.json').as_posix(),
        '--output', '{output}/benchmark', '--stop-file', '{stop_file}']


def completion():
    return {'receipt': 'benchmark/completion.json', 'equals': {'state': 'completed', 'operation': KIND,
        'quality_claim': False, 'production_checkpoint': False, 'validation_evaluated': False,
        'validation_rows_used_for_updates': 0, 'test_accessed': False},
        'finite': ['accepted_updates'], 'artifacts': ['benchmark/report.json', 'benchmark/fixture.json']}


def validate_plan(plan, request):
    validate_spec(request['specification'])
    old = read(BASE_PLAN)
    require(all(plan[k] == old[k] for k in ('bundle_dir', 'source_hashes', 'interpreter', 'interpreter_sha256', 'gpu_uuid')),
        'same exact frozen source/runtime/GPU')
    require(all(plan['input_hashes'].get(p) == digest for p, digest in old['input_hashes'].items()), 'entire original immutable input closure')
    require(plan['input_hashes'].get(METHOD.as_posix()) == METHOD_SHA
        and plan['input_hashes'].get(BASE_PLAN.as_posix()) == BASE_SHA
        and plan['input_hashes'].get(BASE_CONFIG.as_posix()) == BASE_CONFIG_SHA
        and sha(BASE_CONFIG) == BASE_CONFIG_SHA, 'direct exact method/baseline metadata pins')
    require(plan['train_cap'] == 1 and plan['eval_cap'] == 0 and plan['retry_cap'] == 0 and plan['prior_jobs'] == {}
        and len(plan['jobs']) == 1, 'one disposable train benchmark, no retry')
    job = plan['jobs'][0]
    require(job['id'] == 'C4-sampler-probe-r1' and job['kind'] == 'train' and job['purpose'] == 'disposable-sampler-performance'
        and job.get('receipt_adapter') == KIND and job['argv'] == expected_argv() and job['completion'] == completion()
        and job['depends_on'] == [] and job.get('retry_of') is None, 'exact fixed benchmark entry/receipt')
    require(job['max_seconds'] == 810 and job['estimate_seconds'] == 780 and job['stop_grace_seconds'] == 15
        and job['cleanup_reserve_seconds'] == 30 and job['disk_required_bytes'] == 6 << 30
        and job['disk_reserve_bytes'] == 10 << 30 and job['log_cap_bytes'] == 8 << 20,
        'original resources and finite900s outer cap including90s admission/cleanup reserve')
    require(Path(job['output']).resolve() == OUTPUT and Path(request['run_dir']).resolve() == RESOURCE
        and Path(request['state_dir']).resolve() == STATE and request['max_seconds'] == 900
        and request['aggregate_rss_gib'] == 12 and request.get('rpc_argv') == [], 'fixed separate resource/state/job and noRPC')
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None and plan['drain_utc'] == request['deadline_utc']
        and datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')) == deadline + timedelta(seconds=60)
        and (deadline - datetime.now(timezone.utc)).total_seconds() >= 945, 'entire finite benchmark and cleanup fit fresh deadline')
    return job, deadline


def checked(request_path, approval_path, *, running=False):
    request, approval = read(request_path), read(approval_path)
    require(request['contract'] == KIND and approval['verdict'] == 'approved' and approval.get('reviewer')
        and approval['request_sha256'] == g.canonical(request), 'independent exact resource approval')
    g.bound(approval['evidence'])
    plan, native_approval = g.bound(request['native_plan']), g.bound(request['native_approval'])
    require(native_approval['verdict'] == 'approved' and native_approval.get('reviewer')
        and native_approval['plan_sha256'] == g.canonical(plan), 'independent exact native plan approval')
    job, deadline = validate_plan(plan, request)
    require(job['pair_probe'] == {'request': Path(request_path).resolve().as_posix(),
        'approval': Path(approval_path).resolve().as_posix()}, 'exact approved resource paths')
    for path, digest in sources().items():
        require(sha(path) == digest and request['source_hashes'].get(path) == digest
            and plan['input_hashes'].get(path) == digest, 'exact reviewed existing/new helper source: ' + path)
    require(request['native_adapter'] == {'path': ADAPTER.as_posix(), 'sha256': sha(ADAPTER)}, 'fixed adapter')
    for key in ('native_adapter_review', 'worker_request', 'native_quiet'):
        g.bound(request[key])
        require(plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'direct pinned review/worker/quiet metadata')
    require(set(request['prerequisite_evidence']) == {'control', 'selected_export'}, 'both completed predecessor operations required')
    for name, pin in request['prerequisite_evidence'].items():
        status = g.bound(pin)
        expected_path = ROOT / 'scheduler-control-r1' / ('train-state' if name == 'control' else 'export-state') / 'status.json'
        require(Path(pin['path']).resolve() == expected_path and plan['input_hashes'].get(pin['path']) == pin['sha256'], 'fixed pinned predecessor native status')
        attempts = status['attempts']
        require(len(attempts) == 1 and attempts[0]['state'] == 'completed' and attempts[0]['cleanup'] == 'quiescent', 'predecessor operation completed and quiescent')
        receipt = attempts[0]['receipt']
        require(receipt['updates'] == 27235 and receipt['selected_epoch'] == 1 and receipt['validation']['examples'] == 158760,
            'complete controlled E1 and fullvalidation/selectedexport evidence')
    worker = g.bound(request['worker_request'])
    require(Path(request['worker_request']['path']).resolve() == HERE / 'worker-request.json'
        and worker['specification'] == request['specification'] and worker['source_hashes'] == request['source_hashes']
        and worker['deadline_utc'] == request['deadline_utc'] and worker['baseline_plan'] == {'path': BASE_PLAN.as_posix(), 'sha256': BASE_SHA}, 'fixed independently pinned worker request')
    require(worker['baseline_config'] == {'path': BASE_CONFIG.as_posix(), 'sha256': BASE_CONFIG_SHA}, 'fixed original allfeatures baseline config')
    quiet = g.bound(request['native_quiet'])
    stamp = datetime.fromisoformat(quiet['updated_utc'].replace('Z', '+00:00'))
    require(quiet['state'] == 'quiet' and 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 120
        and all(plan[k] == quiet[k] for k in ('gpu_uuid', 'protected_dwm_pid', 'benign_gpu_processes')), 'fresh quiet and physical identity')
    helpers = read(READY.parent / 'ready-runs-20261020/sft-finalize-plan-r1/resource-request.json')
    for key in ('nvidia_smi', 'tasklist', 'console_host'):
        g.bound(request[key])
        require(request[key] == helpers[key] and plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'same accepted OS helper pins')
    require(request['audit_argv_tails'] == g.AUDIT_TAILS, 'same GPU audit commands')
    for output in (OUTPUT, STATE, RESOURCE):
        require((running and output == RESOURCE and output.is_dir()) or not output.exists(), 'fresh outputs')
        require(all(output != Path(p) and not output.is_relative_to(Path(p)) and not Path(p).is_relative_to(output)
            for p in plan['input_hashes']), 'outputs separate from immutable inputs')
    return request, plan, job, {}, {**g.limits(12), 'main_worker': 11274289152, 'max_seconds': 900}, deadline


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
