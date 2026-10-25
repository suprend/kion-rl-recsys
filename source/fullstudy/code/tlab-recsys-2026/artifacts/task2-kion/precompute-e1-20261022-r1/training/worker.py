"""Fresh corrected E1 worker. Native admission/resources belong to root.

Module import, request validation and --describe are stdlib-only. Runtime loading
occurs only after a fresh admitted, pinned request and a live budget check.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
CONTRACT = 'kion-precomputed-fresh-corrected-E1-v1'
SAMPLER = BASE / 'deadline-20261022/negative-worker-r2/sampler.py'
HOOKS = SAMPLER.with_name('hooks.py')
IDENTITY = BASE / 'quality-recovery-20261021/uniform-control-r1/worker.py'
CACHE = BASE / 'deadline-20261022/perf-audit-r1/dynamic_feature_cache.py'
ANCHORS = {
    SAMPLER: '71d8747bfb080bc6ea82059c28b899e5c1bd266649deb9943de80de431933c44',
    HOOKS: '2c3399638216a980797e07b70132bdcec05b938ce9a189f8b0803de5ec84f6cc',
    HOOKS.parents[2] / 'quality-recovery-20261021/uniform-control-r1/hooks.py':
        '2dc7ba017dac6314a30ca4e2ff33ebd5d26602c4d99129f5535f0b5637465989',
    IDENTITY: 'c39ec2b3cf50723591e0b5da2fef46012947e852970f36dc8c513ab30abe3f9f',
    CACHE: '3a9ba56b34561a3e129a0f116159e25deb6e3b2f81a1680cef0e770071f0b611',
}
FROZEN = {
    'kion_complete_train.py': '16180fc30db79f6a5b5c0fc82080eb89658d4b466c65f3a96f66d313c85746d9',
    'kion_complete_model.py': '8d1be2352cd268481f4a1f7b764943f745e910695a1a270fd3d0b1c4e7ec24e7',
}
CAPS = {'aggregate_rss_gib': 12, 'main_worker_ws_bytes': 11274289152,
        'host_reserve_gib': 2}
FIXED = {'provider': 'kion_complete_pack:load_training_provider', 'device': 'cuda',
         'schedule_total_epochs': 5, 'batch_size': 128, 'accumulation': 1,
         'hidden_size': 256, 'layers': 4, 'heads': 4, 'bottleneck': 32,
         'person_dim': 32, 'seed': 2026, 'patience': 3, 'bucket_size': 8192,
         'checkpoint_updates': 1000, 'log_updates': 100,
         'cpu_threads': 2, 'lr': .0001, 'weight_decay': .01, 'warmup_ratio': .05,
         'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.,
         'candidate_chunk_size': 64, 'retrieval_topk': 128,
         'training_order': 'origin-grouped-v1', 'scoring_mode': 'rerank',
         'integration_mode': 'scoped_production', 'text_variant': 'all',
         'people_variants': ['actor-mean', 'director-mean', 'actor-top3-concat'],
         'allow_partial_integration': False, 'resume': False, 'resume_from': 'latest'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    # Fresh adjunct receipts only; the frozen trainer owns model publication.
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(path, digest, name):
    path = Path(path)
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == digest, 'Pinned source differs: ' + path.as_posix())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(payload, str(path), 'exec'), module.__dict__)
    return module


def pin(request, path):
    key = Path(path).resolve().as_posix()
    digest = request['source_hashes'].get(key)
    require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
            'Missing/changed source pin: ' + key)
    return digest


def validate_request(request):
    require(request['contract'] == CONTRACT and request['state'] == 'admitted',
            'Fresh root admitted request required')
    require(request['mode'] in ('full-e1', 'benchmark'), 'Unknown mode')
    require(request['phase'] == ('gpu-e1' if request['mode'] == 'full-e1' else 'gpu-shortbench'),
            'Native phase/mode mismatch')
    require(request['variant'] in ('baseline', 'prepared'), 'Unknown execution variant')
    require(request['mode'] != 'full-e1' or request['variant'] == 'prepared', 'Full E1 requires prepared adapter')
    require(request['resource_caps'] == CAPS, 'Fixed resource caps must not increase')
    require(type(request['benchmark_updates']) is int and 1 <= request['benchmark_updates'] <= 12,
            'Benchmark cap must be1..12 successful updates')
    require(type(request['maximum_seconds']) in (int, float)
            and math.isfinite(request['maximum_seconds']) and 0 < request['maximum_seconds'] <= 24000,
            'Finite bounded worker duration required')
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None, 'Aware deadline required')
    require(Path(request['output_dir']).is_absolute(), 'Absolute output required')
    output = Path(request['output_dir']).resolve()
    require(request['output'] == request['output_dir'], 'Native output alias mismatch')
    require(output.is_absolute() and output.is_relative_to(HERE.parent.resolve())
            and output != HERE.parent.resolve() and not output.is_relative_to(HERE.resolve()),
            'Fresh outputs must be inside this new operation, outside training sources')
    require(not output.exists(), 'Fresh output directory required')
    require(Path(request['stop_file']).resolve().parent == output.parent, 'Stop file belongs to fresh operation')
    runtime = Path(request['runtime_root']).resolve()
    require(runtime == (BASE / 'ready-pack-20261020/provider/runtime-native-r5').resolve(), 'Exact frozen runtime required')
    require(type(request['train_argv']) is list and all(type(x) is str for x in request['train_argv']), 'Exact argv list required')
    flags = [x for x in request['train_argv'] if x.startswith('--')]
    require(len(flags) == len(set(flags)), 'Duplicate options forbidden')
    pin(request, Path(__file__))
    for path, expected in ANCHORS.items():
        require(pin(request, path) == expected, 'Reviewed anchor mismatch')
    for name, expected in FROZEN.items():
        require(pin(request, runtime / 'scripts' / name) == expected, 'Frozen model/train mismatch')
    pin(request, runtime / 'scripts/kion_ready_runtime.py')
    for path, digest in request['source_hashes'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Execution source map mismatch: ' + path)
    expected = request['expected_config']
    require(sha(expected['path']) == expected['sha256'], 'Accepted E1 config pin mismatch')
    interpreter = request['interpreter']
    require(Path(sys.executable).resolve() == Path(interpreter['path']).resolve()
            and sha(sys.executable) == interpreter['sha256'], 'Pinned interpreter mismatch')
    hook = request['provider_hook']
    require(type(hook) is dict and set(hook) == {'path', 'sha256', 'install_name', 'options'}, 'Exact provider hook descriptor required')
    require(pin(request, hook['path']) == hook['sha256'], 'Provider hook must be pinned')
    require(type(hook['install_name']) is str and hook['install_name'].isidentifier()
            and type(hook['options']) is dict, 'Provider installer/options invalid')
    require(hook['options'].get('enabled') is (request['variant'] == 'prepared'),
            'Provider prepared/baseline flag mismatch')
    if request['mode'] == 'full-e1':
        require(hook['options'].get('allow_missing_fallback', False) is False
                and hook['options'].get('purpose') != 'pilot-shortbench',
                'Full E1 requires complete prepared coverage without pilot fallback')
    return output, deadline


def validate_args(args, request):
    for name, value in FIXED.items():
        require(getattr(args, name) == value, 'Corrected E1 recipe mismatch: ' + name)
    require(args.output.resolve() == Path(request['output_dir']).resolve() / 'model', 'Exact model output required')
    require(args.stop_file is not None and args.stop_file.resolve() == Path(request['stop_file']).resolve()
            and args.deadline_utc == request['deadline_utc'], 'Bounded runner controls mismatch')
    require(args.stop_after_validation == (1 if request['mode'] == 'full-e1' else 0)
            and args.max_updates == (0 if request['mode'] == 'full-e1' else request['benchmark_updates']),
            'Full E1/partial benchmark limits mismatch')


def validate_config(actual, expected):
    require(actual == expected, 'Exact accepted E1 architecture/provider/schedule/runtime config differs')
    columns = actual['contract']['architecture']['feature_columns']
    require({k: len(columns[k]) for k in ('item', 'user', 'candidate')} ==
            {'item': 5736, 'user': 3252, 'candidate': 1068}, 'All feature widths required')
    require(actual['contract']['training_schedule']['updates_per_epoch'] == 27235
            and actual['contract']['training_schedule']['eligible_count'] == 3477437,
            'Exact full-row schedule required')


class Budget:
    def __init__(self, request, deadline):
        self.request, self.deadline, self.started = request, deadline, time.perf_counter()

    def check(self):
        require(datetime.now(timezone.utc) < self.deadline
                and time.perf_counter() - self.started < self.request['maximum_seconds']
                and not Path(self.request['stop_file']).exists(), 'Worker deadline/stop/duration exhausted')


def snapshot(identity, train, model, provider, optimizer, scaler):
    digest = lambda value: identity.state_digest(value, train.torch, train.np)
    return {'weights': digest(model.state_dict()), 'optimizer': digest(optimizer.state_dict()),
            'scaler': digest(scaler.state_dict()), 'rng': digest(train.legacy.rng_state())}


def full_e1(request, train, hooks, identity, budget, output):
    adjunct = output / 'adjunct'
    adjunct.mkdir()
    sidecar = adjunct / 'predictions.npz'
    expected = read(request['expected_config']['path'])
    checked = False

    def snap(*objects):
        budget.check()
        return snapshot(identity, train, *objects)

    hook = hooks.SinglePass(train, sidecar, snap)
    original_main, original_run, original_config = train.main, train.run, train.make_config
    # Check the actual generated config before the first optimizer update.
    def checked_config(*args, **kwargs):
        nonlocal checked
        budget.check()
        config = original_config(*args, **kwargs)
        validate_config(config, expected)
        checked = True
        return config

    train.make_config = checked_config
    try:
        hook.install()
        require(train.main is original_main and train.run is original_run
                and original_main.__globals__['run'] is original_run, 'Original main/run binding required')
        result = original_main(request['train_argv'])
    finally:
        try:
            hook.restore()
        finally:
            train.make_config = original_config
    budget.check()
    evidence = hook.finished()
    require(checked and result['status'] == 'stopped' and result['reason'] == 'runtime_stop_after_validation'
            and result['predictions_status'] == 'deferred-runtime-stop', 'Only complete E1 accepted')
    selected = read(output / 'model/selected.json')
    config = read(output / 'model/config.json')
    validate_config(config, expected)
    require(result['validation'] == selected['validation'] == hook.metrics, 'Selected/hook/raw metrics differ')
    saved = train.load_checkpoint(output / 'model/selected-epoch0001.pt', config, map_location='cpu')
    digest = lambda value: identity.state_digest(value, train.torch, train.np)
    require(saved['state'] == result['state'], 'Selected final E1 cursor/history mismatch')
    for source, target in (('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng')):
        require(digest(saved[source]) == hook.evaluated[target], 'Committed selected/evaluated state differs: ' + target)
    del saved
    # Reuse exact CPU NPZ verification without invoking its old control/main.
    from types import SimpleNamespace
    replay_context = SimpleNamespace(require=require, g=SimpleNamespace(canonical=canonical))
    replay = identity.replay_sidecar(replay_context, train, hook.objects[1], sidecar, selected['validation'])
    evidence.update(selected_checkpoint_sha256=sha(output / 'model/selected-epoch0001.pt'),
                    selected_weights_equal_evaluated=True, raw_config_sha256=sha(output / 'model/config.json'),
                    sampler_contract=hooks.sampler.CONTRACT)
    require(selected['sha256'] == evidence['selected_checkpoint_sha256'], 'Selected receipt digest mismatch')
    write_new(adjunct / 'hook-evidence.json', evidence)
    write_new(adjunct / 'replay.json', replay)
    artifacts = ['model/selected-epoch0001.pt', 'model/selected.json', 'model/config.json',
                 'model/metrics.json', 'model/completion.json', 'adjunct/predictions.npz',
                 'adjunct/hook-evidence.json', 'adjunct/replay.json']
    return {'state': 'completed', 'scientific_training_state': 'stopped', 'reason': result['reason'],
            'full_epoch_completed': True, 'schedule_completed': False, 'selected_epoch': 1,
            'successful_updates': 27235, 'train_examples': 3477437, 'evaluate_calls': 1,
            'optimizer_updates_after_evaluation': 0, 'validation': result['validation'],
            'selected_weights_equal_evaluated': True, 'selected_checkpoint_sha256': selected['sha256'],
            'sidecar_sha256': sha(sidecar), 'sampler_contract': hooks.sampler.CONTRACT,
            'artifacts': artifacts, 'sha256': {name: sha(output / name) for name in artifacts}}


def benchmark(request, train, sampler, identity, cache_module, budget, output):
    """Fresh one-cell prefix; compare separate same-seed receipts, no checkpoints."""
    args = train.parser().parse_args(request['train_argv'])
    validate_args(args, request)
    module, factory = args.provider.split(':', 1)
    provider = getattr(train.importlib.import_module(module), factory)(args)
    installation = sampler.InstalledSampler(train)
    installation.install()
    candidate_tensors = []
    def audited_candidates(*args, **kwargs):
        result = installation.replacement(*args, **kwargs)
        candidate_tensors.append(result.detach())
        return result
    train.sample_training_candidates = audited_candidates
    records = []
    dynamic_cache = None
    try:
        train.integration_contract(provider, args)
        torch, np = train.torch, train.np
        train.random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.cpu_threads)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        device = torch.device('cuda')
        schedule = train.TrainingSchedule(provider, args)
        model = train.make_model(provider, args).to(device)
        validate_config(train.make_config(provider, model, args, device, schedule=schedule),
                        read(request['expected_config']['path']))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024)
        dynamic_cache = cache_module.install_dynamic_cache(provider)
        model.train()
        for offset, groups, group_index, origin in schedule.groups(0, 0):
            if len(records) >= request['benchmark_updates']:
                break
            budget.check()
            for group in optimizer.param_groups:
                group['lr'] = train.shared.learning_rate(group_index, schedule.updates_per_epoch * 5,
                                                         args.warmup_ratio, args.lr)
            before = snapshot(identity, train, model, provider, optimizer, scaler)
            rows_digest = identity.state_digest(tuple(groups), torch, np)
            candidate_tensors.clear()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            losses, retries = train.update(model, provider, groups, optimizer, scaler, device,
                hard_negatives=0, uniform_negatives=3, auxiliary_weight=1., candidate_chunk_size=64,
                collect_loss=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            after = snapshot(identity, train, model, provider, optimizer, scaler)
            candidate_digest = identity.state_digest(candidate_tensors, torch, np)
            records.append({'group_index': group_index, 'offset': offset, 'origin': origin,
                'rows': sum(map(len, groups)), 'rows_sha256': rows_digest,
                'before': before, 'after': after, 'losses': losses, 'amp_retries': retries,
                'candidate_sha256_including_retries': candidate_digest,
                'update_wall_seconds': elapsed, 'learning_rate': optimizer.param_groups[0]['lr']})
        require(len(records) == request['benchmark_updates'], 'Partial benchmark count mismatch')
        budget.check()
        write_new(output / 'benchmark-records.json', records)
        return {'state': 'completed-disposable-prefix', 'full_epoch_completed': False, 'schedule_completed': False,
                'partial': True, 'successful_updates': len(records), 'evaluate_calls': 0,
                'validation_rows_used_for_updates': 0, 'production_checkpoint': False,
                'checkpoints_written': False, 'quality_claim': False,
                'sampler_contract': sampler.CONTRACT, 'sampler_audit': installation.replacement.audit,
                'runtime_performance': {'cpu_threads': 2, 'dynamic_cache': True,
                                        'dynamic_cache_stats': dynamic_cache.stats()},
                'records_sha256': sha(output / 'benchmark-records.json'),
                'artifacts': ['benchmark-records.json'],
                'sha256': {'benchmark-records.json': sha(output / 'benchmark-records.json')},
                'timing_scope': 'synchronized single-update including loss readback; state/candidate digests excluded',
                'timing_limit': 'prefix diagnostic; extra loss reads vs full runner, no epoch acceleration claim'}
    finally:
        try:
            train.sample_training_candidates = installation.replacement
            installation.restore()
        finally:
            try:
                if dynamic_cache is not None:
                    dynamic_cache.uninstall()
            finally:
                if hasattr(provider, 'close'):
                    provider.close()


def compare_records(baseline, prepared):
    """Exact update/RNG trajectory parity; timings are deliberately excluded."""
    require(len(baseline) == len(prepared) and len(baseline) > 0, 'Matching nonempty windows required')
    compared = ('group_index', 'offset', 'origin', 'rows', 'rows_sha256', 'before', 'after',
                'losses', 'amp_retries', 'learning_rate', 'candidate_sha256_including_retries')
    for i, (left, right) in enumerate(zip(baseline, prepared)):
        for key in compared:
            require(left[key] == right[key], 'Same-state parity failed at update ' + str(i) + ': ' + key)
    return {'same_state_parity': True, 'successful_updates_per_variant': len(baseline),
            'compared_fields': list(compared), 'whole_epoch_equivalence_claim': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--describe', action='store_true')
    cli = parser.parse_args(argv)
    if cli.describe:
        print(json.dumps({'contract': CONTRACT, 'fixed_training': FIXED, 'resource_caps': CAPS,
                          'runtime_imported': False, 'native_admission_implemented_here': False}))
        return
    require(cli.request is not None, '--request required')
    request = read(cli.request)
    output, deadline = validate_request(request)
    budget = Budget(request, deadline)
    budget.check()
    require(os.environ.get('KION_OUTPUT_DIR') == output.as_posix()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == request['gpu_uuid'], 'Root native environment mismatch')
    output.mkdir()
    provider_hook = None
    report = None
    try:
        runtime_root = Path(request['runtime_root']) / 'scripts'
        sys.path.insert(0, runtime_root.as_posix())
        runtime = load(runtime_root / 'kion_ready_runtime.py', pin(request, runtime_root / 'kion_ready_runtime.py'),
                       '_precomputed_exact_bootstrap')
        runtime.configure_paths()
        budget.check()
        train = load(runtime_root / 'kion_complete_train.py', FROZEN['kion_complete_train.py'],
                     '_precomputed_exact_train')
        identity = load(IDENTITY, ANCHORS[IDENTITY], '_precomputed_state_identity_only')
        cache_module = load(CACHE, ANCHORS[CACHE], '_precomputed_exact_dynamic_cache')
        args = train.parser().parse_args(request['train_argv'])
        validate_args(args, request)
        descriptor = request['provider_hook']
        module = load(descriptor['path'], descriptor['sha256'], '_precomputed_pinned_provider_hook')
        provider_hook = getattr(module, descriptor['install_name'])(train, descriptor['options'])
        require(callable(getattr(provider_hook, 'uninstall', None)) and callable(getattr(provider_hook, 'evidence', None)),
                'Provider hook requires uninstall/evidence methods')
        budget.check()
        if request['mode'] == 'full-e1':
            hooks = load(HOOKS, ANCHORS[HOOKS], '_precomputed_exact_E1_hooks')
            require(not hasattr(train, '_negative_dynamic_cache_install'), 'Dynamic cache installer already attached')
            train._negative_dynamic_cache_install = cache_module.install_dynamic_cache
            try:
                report = full_e1(request, train, hooks, identity, budget, output)
            finally:
                del train._negative_dynamic_cache_install
        else:
            sampler = load(SAMPLER, ANCHORS[SAMPLER], '_precomputed_exact_sampler')
            report = benchmark(request, train, sampler, identity, cache_module, budget, output)
        report.update(provider_hook_evidence=provider_hook.evidence() if provider_hook is not None else None)
        budget.check()
    except BaseException as error:
        report = {'state': 'failed', 'full_epoch_completed': False, 'schedule_completed': False,
                  'selected_export_claim': False, 'error': type(error).__name__ + ': ' + str(error)}
        raise
    finally:
        try:
            if provider_hook is not None:
                provider_hook.uninstall()
        except BaseException as error:
            report = {'state': 'failed-cleanup', 'full_epoch_completed': False, 'selected_export_claim': False,
                      'error': type(error).__name__ + ': ' + str(error)}
            raise
        finally:
            if report is not None:
                report.update(contract=CONTRACT, mode=request['mode'], variant=request['variant'],
                    elapsed_seconds=time.perf_counter() - budget.started, test_accessed=False,
                    source_hashes=request['source_hashes'], request_sha256=sha(cli.request),
                    expected_config=request['expected_config'], interpreter=request['interpreter'],
                    native_resource_admission='external-root-owned-not-established-by-this-receipt')
                # A final budget miss must never leave a successful adjunct receipt.
                try:
                    budget.check()
                except ValueError as error:
                    report.update(state='failed-budget', full_epoch_completed=False, selected_export_claim=False,
                                  error=str(error))
                write_new(output / 'worker-completion.json', report)
    require(report['state'] in ('completed', 'completed-disposable-prefix'), 'Worker did not complete: ' + report['state'])
    return report


if __name__ == '__main__':
    main()
