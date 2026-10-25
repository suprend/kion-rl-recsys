"""Fixed fresh H0/U3 fullE1 metadata and existing owned-resource admission."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
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
CONFIG_SHA = '71cc784f10269f196d4ad2c0a6f34f461d5c18b6bbd0b0c89d8d4b04f26141b3'
METHOD = ROOT / 'methods-r1/singlepass-export-feasibility-r1.json'
METHOD_SHA = 'c9ee85a978a931058404e4c85330e219c4b03c57f285b286bfbff2c2b8fecfe8'
PAIR_METHOD = ROOT / 'methods-r1/next-pair-plan-r1.json'
PAIR_METHOD_SHA = 'd5ba7b2379aac07897fff9108724f3b75ad9f7101da36f00b2cbde314cd9ed4f'
GUARD = READY / 'model/rl-native-resource-r1/guard.py'
GUARD_SHA = 'b1bee55f2ec947c188bc8f6065d66b08bfce85b16a2bc7b631d72a8e3b785e87'
JOB_SHA = '0da87bf737038e08602ddaf18929a9441543269adf0edd191f1ac4151e916296'
WORKER, HOOKS, ADAPTER = [HERE / name for name in ('worker.py', 'hooks.py', 'native_adapter.py')]
OUTPUT, STATE, RESOURCE = [HERE / name for name in ('job', 'state', 'resource')]
KIND = 'kion-fresh-H0U3-E1-singlepass-sidecar-v1'


def require(condition, message):
    if not condition:
        raise ValueError('Uniform fullE1: ' + message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path, digest, name):
    path = Path(path).resolve()
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, 'exact direct source bytes: ' + path.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


g = load(GUARD, GUARD_SHA, '_uniform_unchanged_owned_resource_guard')


def specification():
    return {'contract': KIND, 'state': 'source-only-not-admitted', 'fresh_seed': 2026,
        'scientific_delta_from_scheduler_control': {'hard_negatives': 0, 'uniform_negatives': 3},
        'all73ready_retained': True, 'external8_deferred': True, 'auxiliary_weight': 1.,
        'batch_size': 128, 'accumulation': 1, 'candidate_chunk_size': 64, 'cpu_threads': 2,
        'schedule_total_epochs': 5, 'stop_after_validation': 1, 'train_rows': 3477437,
        'successful_updates': 27235, 'validation_rows': 158760, 'known_targets': 154564,
        'unknown_target_misses': 4196, 'resume': False, 'validation_evaluate_calls': 1,
        'sidecar': 'original soleE1 evaluate with only predictions_path supplied; no second scoring pass',
        'raw_runner_status': 'stopped/runtime_stop_after_validation/deferred-runtime-stop',
        'evaluated_selected_weights': 'bounded state_dict hash around evaluate/original selected publication and CPU checkpoint reload',
        'evaluation_preserves': ['model weights', 'optimizer state', 'scaler state', 'Python/NumPy/Torch/CUDA RNG'],
        'phase_max_seconds': 24000, 'aggregate_rss_gib': 12, 'main_worker_ws_bytes': 11274289152,
        'host_reserve_gib': 2, 'initial_available_gib': 14, 'disk_required_bytes': 6 << 30,
        'disk_reserve_bytes': 10 << 30, 'automatic_launch': False, 'automatic_retry': False,
        'fiveepoch_completed_claim': False, 'timing_label': 'validation-plus-sidecar-export-plus-readonly-state-digests'}


def sources():
    return {**g.HELPER_HASHES, GUARD.as_posix(): GUARD_SHA, GUARD.with_name('native_job.py').as_posix(): JOB_SHA,
        **{path.as_posix(): sha(path) for path in (Path(__file__), WORKER, HOOKS, ADAPTER)}}


def train_argv(deadline):
    require(sha(BASE_PLAN) == BASE_SHA, 'original scientific baseline metadata')
    argv = list(read(BASE_PLAN)['jobs'][0]['argv'][2:])
    for flag, value in (('--schedule-total-epochs', '5'), ('--hard-negatives', '0'), ('--uniform-negatives', '3'),
        ('--output', (OUTPUT / 'model').as_posix()), ('--stop-file', (OUTPUT / 'stop-request.json').as_posix()), ('--deadline-utc', deadline)):
        argv[argv.index(flag) + 1] = value
    return argv + ['--stop-after-validation', '1']


def worker_argv():
    return [WORKER.as_posix(), '--request', (HERE / 'worker-request.json').as_posix()]


def completion():
    return {'receipt': 'singlepass-completion.json', 'equals': {'state': 'completed', 'operation': KIND,
        'scientific_training_state': 'stopped', 'scientific_reason': 'runtime_stop_after_validation',
        'schedule_completed': False, 'selected_epoch': 1, 'updates': 27235, 'train_examples': 3477437,
        'validation.examples': 158760, 'validation.known_targets': 154564, 'validation.unknown_target_misses': 4196,
        'evaluate_calls': 1, 'optimizer_updates_after_evaluation': 0, 'selected_weights_equal_evaluated': True,
        'sidecar_status': 'externally-bound-selectedE1-singlepass', 'test_accessed': False},
        'finite': ['validation.ndcg@10', 'validation.recall@10'],
        'artifacts': ['model/selected-epoch0001.pt', 'model/selected.json', 'model/config.json',
            'model/metrics.json', 'model/completion.json', 'adjunct/predictions.npz', 'adjunct/hook-evidence.json', 'adjunct/replay.json']}


def validate_decision(decision):
    require(decision.get('contract') == 'kion-uniform-fullE1-root-decision-v1'
        and decision.get('decision') == 'admit-fresh-H0U3-singlepass-E1', 'explicit new root scientific decision')
    phases = decision['point_phase_seconds']
    require(set(phases) == {'train', 'full_validation', 'checkpoint', 'cold_init', 'sidecar_and_provenance'}, 'all measured fullE1 phases budgeted')
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in phases.values()), 'finite positive phase estimates')
    margin = decision['timing_margin_ratio']
    require(type(margin) in (int, float) and math.isfinite(margin) and margin >= .15, 'at least15% measured timing margin')
    estimate = math.ceil(sum(phases.values()) * (1 + margin))
    require(estimate <= 23910, 'entire measured uniform E1/sidecar/provenance plus margin fits bounded native phase; no partial trial')
    return estimate


def validate_plan(plan, request):
    require(request['contract'] == KIND and request['specification'] == specification(), 'exact source-only specification bound for future admission')
    old = read(BASE_PLAN)
    require(sha(BASE_PLAN) == BASE_SHA and sha(BASE_CONFIG) == CONFIG_SHA and sha(METHOD) == METHOD_SHA
        and sha(PAIR_METHOD) == PAIR_METHOD_SHA, 'original baseline/method source pins')
    require(all(plan[key] == old[key] for key in ('bundle_dir', 'source_hashes', 'interpreter', 'interpreter_sha256', 'gpu_uuid')),
        'unchanged COMMON/interpreter/source/GPU')
    require(all(plan['input_hashes'].get(path) == digest for path, digest in old['input_hashes'].items()), 'whole original immutable input closure')
    for path, digest in ((BASE_PLAN, BASE_SHA), (BASE_CONFIG, CONFIG_SHA), (METHOD, METHOD_SHA), (PAIR_METHOD, PAIR_METHOD_SHA)):
        require(plan['input_hashes'].get(path.as_posix()) == digest, 'direct baseline/method metadata pin')
    require(plan['train_cap'] == 1 and plan['eval_cap'] == 0 and plan['retry_cap'] == 0
        and plan['prior_jobs'] == {} and len(plan['jobs']) == 1, 'one fresh scientific train with no retry')
    decision = g.bound(request['root_decision'])
    estimate = validate_decision(decision)
    job = plan['jobs'][0]
    require(job['id'] == 'uniform-H0U3-singlepass-E1-r1' and job['kind'] == 'train'
        and job['purpose'] == 'fresh-uniform-singlepass-fullE1' and job.get('receipt_adapter') == KIND
        and job['argv'] == worker_argv() and job['completion'] == completion()
        and job['depends_on'] == [] and job.get('retry_of') is None, 'exact fixed external worker/raw operation contract')
    require(job['estimate_seconds'] == estimate and job['max_seconds'] == 23910 and job['stop_grace_seconds'] == 15
        and job['cleanup_reserve_seconds'] == 30 and job['disk_required_bytes'] == 6 << 30
        and job['disk_reserve_bytes'] == 10 << 30 and job['log_cap_bytes'] == 8 << 20, 'unchanged native resource/disk/log limits')
    require(Path(job['output']).resolve() == OUTPUT and Path(request['run_dir']).resolve() == RESOURCE
        and Path(request['state_dir']).resolve() == STATE and request['max_seconds'] == 24000
        and request['aggregate_rss_gib'] == 12 and request.get('rpc_argv') == [], 'fixed fresh24000s one-owner resources')
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None and plan['drain_utc'] == request['deadline_utc']
        and datetime.fromisoformat(plan['deadline_utc'].replace('Z', '+00:00')) == deadline + timedelta(seconds=60)
        and (deadline - datetime.now(timezone.utc)).total_seconds() >= 24045, 'full bounded phase and cleanup before new absolute lease')
    return job, deadline


def checked(request_path, approval_path, *, running=False):
    request, approval = read(request_path), read(approval_path)
    require(approval['verdict'] == 'approved' and approval.get('reviewer')
        and approval['request_sha256'] == g.canonical(request), 'independent exact resource approval')
    g.bound(approval['evidence'])
    plan, native_approval = g.bound(request['native_plan']), g.bound(request['native_approval'])
    require(native_approval['verdict'] == 'approved' and native_approval.get('reviewer')
        and native_approval['plan_sha256'] == g.canonical(plan), 'independent exact native approval')
    job, deadline = validate_plan(plan, request)
    require(job['uniform_control'] == {'request': Path(request_path).resolve().as_posix(),
        'approval': Path(approval_path).resolve().as_posix()}, 'exact approved resource references')
    for path, digest in sources().items():
        require(sha(path) == digest and request['source_hashes'].get(path) == digest
            and plan['input_hashes'].get(path) == digest, 'every exact existing/new execution helper: ' + path)
    require(request['native_adapter'] == {'path': ADAPTER.as_posix(), 'sha256': sha(ADAPTER)}, 'fixed native adapter')
    for key in ('root_decision', 'native_adapter_review', 'worker_request', 'native_quiet'):
        g.bound(request[key])
        require(plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'direct bound decision/review/worker/quiet metadata')
    worker = g.bound(request['worker_request'])
    require(Path(request['worker_request']['path']).resolve() == HERE / 'worker-request.json'
        and worker['specification'] == specification() and worker['source_hashes'] == request['source_hashes']
        and worker['deadline_utc'] == request['deadline_utc'] and worker['train_argv'] == train_argv(request['deadline_utc']), 'fixed immutable worker metadata and exact fresh argv')
    require(set(request['prerequisite_evidence']) == {'control', 'selected_export', 'benchmark', 'benchmark_resource'}, 'all actual predecessor integrity/resource evidence required')
    for name, pin in request['prerequisite_evidence'].items():
        value = g.bound(pin)
        require(plan['input_hashes'].get(pin['path']) == pin['sha256'], 'direct actual predecessor evidence pin')
        if name == 'benchmark_resource':
            require(Path(pin['path']).resolve() == ROOT / 'pair-probe-r1/resource/operation.json'
                and value['state'] == 'completed_resource_and_native_boundary'
                and value['owned_Job_cleanup'] == 'quiescent'
                and value['native_status_sha256'] == sha(ROOT / 'pair-probe-r1/state/status.json'), 'benchmark original native/resource ownership and cleanup passed')
        elif name == 'benchmark':
            require(Path(pin['path']).resolve() == ROOT / 'pair-probe-r1/job/benchmark/completion.json'
                and value['state'] == 'completed' and value['benchmark_match'] is True and value['accepted_updates'] == 36
                and value['quality_claim'] is False and value['validation_evaluated'] is False
                and value['validation_rows_used_for_updates'] == 0, 'complete passed actual36-update sampler benchmark')
        else:
            expected = ROOT / 'scheduler-control-r1' / ('train-state' if name == 'control' else 'export-state') / 'status.json'
            require(Path(pin['path']).resolve() == expected and len(value['attempts']) == 1
                and value['attempts'][0]['state'] == 'completed' and value['attempts'][0]['cleanup'] == 'quiescent'
                and value['attempts'][0]['receipt']['updates'] == 27235
                and value['attempts'][0]['receipt']['validation']['examples'] == 158760, 'actual control/export complete and quiescent')
    quiet = g.bound(request['native_quiet'])
    stamp = datetime.fromisoformat(quiet['updated_utc'].replace('Z', '+00:00'))
    require(quiet['state'] == 'quiet' and 0 <= (datetime.now(timezone.utc) - stamp).total_seconds() <= 120
        and all(plan[key] == quiet[key] for key in ('gpu_uuid', 'protected_dwm_pid', 'benign_gpu_processes')), 'fresh quiet GPU identities')
    helpers = read(READY.parent / 'ready-runs-20261020/sft-finalize-plan-r1/resource-request.json')
    for key in ('nvidia_smi', 'tasklist', 'console_host'):
        g.bound(request[key])
        require(request[key] == helpers[key] and plan['input_hashes'].get(request[key]['path']) == request[key]['sha256'], 'same accepted OS audit executable pins')
    require(request['audit_argv_tails'] == g.AUDIT_TAILS, 'same native GPU audit commands')
    for output in (OUTPUT, STATE, RESOURCE):
        require((running and output == RESOURCE and output.is_dir()) or not output.exists(), 'fresh owned output boundary')
        require(all(output != Path(path) and not output.is_relative_to(Path(path)) and not Path(path).is_relative_to(output)
            for path in plan['input_hashes']), 'new outputs separate from immutable inputs')
    return request, plan, job, {}, {**g.limits(12), 'main_worker': 11274289152, 'max_seconds': 24000}, deadline


def validate_result(job):
    root = Path(job['output'])
    metrics, completion, selected, config = [read(root / 'model' / name)
        for name in ('metrics.json', 'completion.json', 'selected.json', 'config.json')]
    expected = copy.deepcopy(read(BASE_CONFIG))
    expected['contract']['training'].update(schedule_total_epochs=5, hard_negatives=0, uniform_negatives=3)
    require(config == expected, 'whole original config except declared horizon/sampler treatment')
    state = metrics['state']
    require(metrics['status'] == completion['state'] == 'stopped' and metrics['reason'] == 'runtime_stop_after_validation'
        and metrics['predictions_status'] == completion['predictions_status'] == 'deferred-runtime-stop', 'raw original stopped/deferred receipts preserved')
    require(all(type(state.get(key)) is int and state[key] == value for key, value in
        (('epoch', 1), ('offset', 0), ('updates', 27235), ('examples', 3477437), ('selected_epoch', 1)))
        and state['history'] == [{'epoch': 1, 'validation': metrics['validation']}], 'fullE1 cursor/history')
    require(metrics['validation'] == completion['validation'] == selected['validation']
        and metrics['validation']['examples'] == 158760 and metrics['validation']['known_targets'] == 154564
        and metrics['validation']['unknown_target_misses'] == 4196, 'one identical fullvalidation selection')
    declarations = completion.get('artifacts')
    require(isinstance(declarations, list) and declarations and len(set(declarations)) == len(declarations)
        and set(completion.get('sha256', {})) == set(declarations), 'original raw artifact declarations intact')
    for name in declarations:
        require(type(name) is str and Path(name).name == name and not Path(name).is_absolute()
            and sha(root / 'model' / name) == completion['sha256'][name], 'every original committed raw model artifact hash: ' + str(name))
    hook = read(root / 'adjunct/hook-evidence.json')
    require(hook['successful_updates'] == 27235 and hook['train_examples'] == 3477437 and hook['evaluate_calls'] == 1
        and hook['selected_publications'] == 1 and hook['optimizer_updates_after_evaluation'] == 0
        and hook['actual_run_global_binding_verified'] is True and hook['evaluation_succeeded'] is True,
        'one actual run-bound successful hook and selected publication, no extra updates')
    require(hook['evaluated_snapshot'] == hook['selected_save_snapshot']
        and hook['selected_checkpoint_weights_digest'] == hook['evaluated_snapshot']['weights'], 'evaluated/original saved/CPU checkpoint weight identity')
    require(selected['file'] == 'selected-epoch0001.pt' and selected['epoch'] == 1
        and selected['sha256'] == sha(root / 'model/selected-epoch0001.pt') == hook['selected_checkpoint_sha256'], 'exact committed selected checkpoint file identity')
    side = read(root / 'model/selected-epoch0001.json')
    require(side['sha256'] == selected['sha256'] and side['epoch'] == 1 and side['offset'] == 0 and side['updates'] == 27235,
        'original selected checkpoint committed sidecar identity/cursor')
    require(not (root / 'model/predictions.npz').exists(), 'raw protocol deferred; sidecar remains external')
    replay = read(root / 'adjunct/replay.json')
    require(replay['rank_metrics_and_coverage_match'] is True and replay['semantic_rows_targets_known_match'] is True
        and replay['examples'] == 158760 and replay['unknown_target_misses'] == 4196, 'CPU replay validated complete sidecar semantics')
    return hook


def main():
    import argparse
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
