"""Pinned E1 -> E2+ continuation. Stdlib-only import/describe; root owns native guard."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RUNTIME = BASE / 'ready-pack-20261020/provider/runtime-native-r5/scripts'
E1 = BASE / 'deadline-20261022/control-r2/job'
HELPER = BASE / 'precompute-e1-20261022-r1/training/worker.py'
IDENTITY = BASE / 'quality-recovery-20261021/uniform-control-r1/worker.py'
SAMPLER = BASE / 'deadline-20261022/negative-worker-r2/sampler.py'
CACHE = BASE / 'deadline-20261022/perf-audit-r1/dynamic_feature_cache.py'
STORE_HOOK = BASE / 'precompute-e1-20261022-r1/preparation/store_hook.py'
CONTRACT = 'kion-night-E1-continuation-v1'
CHECKPOINT_SHA = 'a4c91aa4c3e892322c58f5dff6142ce19f7d038d661c909ebfd043d35e9dd663'
CAPS = {'allowed_aggregate_gib': [14, 16], 'host_reserve_gib': 2}
FIXED = {'provider': 'kion_complete_pack:load_training_provider', 'device': 'cuda',
    'schedule_total_epochs': 5, 'batch_size': 128, 'accumulation': 1, 'hidden_size': 256,
    'layers': 4, 'heads': 4, 'bottleneck': 32, 'person_dim': 32, 'seed': 2026,
    'patience': 3, 'bucket_size': 8192, 'checkpoint_updates': 1000, 'log_updates': 100,
    'cpu_threads': 4, 'lr': .0001, 'weight_decay': .01, 'warmup_ratio': .05,
    'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.,
    'candidate_chunk_size': 64, 'retrieval_topk': 128, 'training_order': 'origin-grouped-v1',
    'scoring_mode': 'rerank', 'integration_mode': 'scoped_production', 'text_variant': 'all',
    'people_variants': ['actor-mean', 'director-mean', 'actor-top3-concat'],
    'allow_partial_integration': False, 'resume': True, 'resume_from': 'latest', 'max_updates': 0}
ANCHORS = {
    IDENTITY: 'c39ec2b3cf50723591e0b5da2fef46012947e852970f36dc8c513ab30abe3f9f',
    SAMPLER: '71d8747bfb080bc6ea82059c28b899e5c1bd266649deb9943de80de431933c44',
    CACHE: '3a9ba56b34561a3e129a0f116159e25deb6e3b2f81a1680cef0e770071f0b611',
    RUNTIME / 'kion_complete_train.py': '16180fc30db79f6a5b5c0fc82080eb89658d4b466c65f3a96f66d313c85746d9',
    RUNTIME / 'kion_complete_model.py': '8d1be2352cd268481f4a1f7b764943f745e910695a1a270fd3d0b1c4e7ec24e7',
    STORE_HOOK: 'cb1307d3e869325174db9791f6665979bf7efbe12f38c6fbf546db56793d34bf',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def pinned(request, path):
    path = Path(path).resolve()
    digest = request['source_hashes'].get(path.as_posix())
    require(type(digest) is str and len(digest) == 64 and sha(path) == digest,
            'Missing/changed execution source: ' + path.as_posix())
    return digest


def load(request, path, name):
    path = Path(path)
    pinned(request, path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def successor_config(old):
    new = copy.deepcopy(old)
    require(new['runtime']['cpu_threads'] == 2, 'Only accepted E1 CPU2 lineage is supported')
    new['runtime']['cpu_threads'] = 4
    return new


def validate_config(actual, expected):
    require(actual == successor_config(expected), 'Only CPU2 -> CPU4 runtime change is admitted')
    columns = actual['contract']['architecture']['feature_columns']
    require({key: len(columns[key]) for key in ('item', 'user', 'candidate')} ==
            {'item': 5736, 'user': 3252, 'candidate': 1068}, 'Complete feature widths required')
    schedule = actual['contract']['training_schedule']
    require(schedule['updates_per_epoch'] == 27235 and schedule['eligible_count'] == 3477437,
            'Exact H5 full-row schedule required')


def validate_resume_pair(saved, current, old, new):
    # No broad ignore-runtime/source escape hatch: exactly these two complete configs.
    require(current == new and saved in (old, new), 'Unbound resume config migration')
    require(saved['contract'] == current['contract'], 'Scientific continuation contract changed')


def validate_start_state(state):
    for key, value in (('epoch', 1), ('offset', 0), ('updates', 27235), ('examples', 3477437),
                       ('selected_epoch', 1), ('group_index', 0), ('bad_epochs', 0)):
        require(type(state.get(key)) is int and state[key] == value, 'Exact completed E1 cursor required: ' + key)
    require(len(state['history']) == 1 and state['history'][0]['epoch'] == 1,
            'One completed original validation required')
    require(state['best_ndcg'] == state['history'][0]['validation']['ndcg@10'], 'E1 selection history mismatch')


def validate_deadline(value):
    deadline = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None and deadline <= datetime(2026, 10, 23, 7, 0, tzinfo=timezone.utc),
            'Aware SFT deadline no later than10:00MSK required')
    return deadline


def validate_request(request):
    require(request['contract'] == CONTRACT and request['state'] == 'admitted', 'Root admitted successor request required')
    require(request['phase'] == 'sft-continuation' and request['cpu_threads'] == 4
            and request['cpu_affinity'] == [0, 2, 4, 6], 'Root execution contract mismatch')
    limits = request['resource_limits']
    cap = limits['aggregate_rss_bytes']
    require(cap in (14 << 30, 16 << 30) and limits == {
        'aggregate_rss_bytes': cap, 'aggregate_commit_bytes': cap,
        'initial_host_available_bytes': cap + (2 << 30), 'host_reserve_bytes': 2 << 30},
        'Root aggregate resource contract mismatch')
    require(type(request['stop_after_validation']) is int and 2 <= request['stop_after_validation'] <= 4,
            'Choose completed epoch2..4; H5 terminal export requires another reviewed path')
    require(type(request['maximum_seconds']) in (int, float) and math.isfinite(request['maximum_seconds'])
            and 0 < request['maximum_seconds'] <= 54000, 'Finite bounded duration required')
    deadline = validate_deadline(request['deadline_utc'])
    output = Path(request['output_dir']).resolve()
    require(Path(request['output_dir']).is_absolute() and output.is_relative_to(HERE)
            and output != HERE and not output.exists(), 'Fresh output below successor sft directory required')
    require(request['output'] == request['output_dir'] and Path(request['stop_file']).resolve().parent == output.parent,
            'Root output/stop binding mismatch')
    require(Path(request['runtime_root']).resolve() == RUNTIME.parent.resolve(), 'Frozen runtime required')
    require(type(request['train_argv']) is list and all(type(x) is str for x in request['train_argv']), 'Explicit argv required')
    flags = [item for item in request['train_argv'] if item.startswith('--')]
    require(len(flags) == len(set(flags)), 'Duplicate flags forbidden')
    require(Path(request['input_job']).resolve() == E1.resolve(), 'Only verified corrected E1 is admitted')
    for name in ('model/config.json', 'model/selected.json', 'model/selected-epoch0001.json',
                 'singlepass-completion.json', 'adjunct/predictions.npz'):
        require(sha(E1 / name) == request['input_hashes'].get(name), 'Missing/changed E1 input: ' + name)
    require(request['input_hashes'].get('model/selected-epoch0001.pt') == CHECKPOINT_SHA
            and sha(E1 / 'model/selected-epoch0001.pt') == CHECKPOINT_SHA, 'Exact selected complete E1 checkpoint required')
    receipt = read(E1 / 'singlepass-completion.json')
    require(receipt['state'] == 'completed' and receipt['selected_epoch'] == 1
            and receipt['selected_weights_equal_evaluated'] is True and receipt['selected_checkpoint_sha256'] == CHECKPOINT_SHA,
            'Accepted E1 provenance required')
    for path, expected in ANCHORS.items():
        require(pinned(request, path) == expected, 'Reviewed frozen anchor mismatch')
    for path in (Path(__file__), RUNTIME / 'kion_ready_runtime.py', RUNTIME / 'kion_train.py',
                 RUNTIME / 'kion_full_train.py', RUNTIME / 'kion_conditioned_train.py',
                 RUNTIME / 'kion_people_embeddings.py', RUNTIME / 'kion_pack_people.py'):
        pinned(request, path)
    for path, digest in request['source_hashes'].items():
        require(Path(path).is_absolute() and sha(path) == digest, 'Changed source closure: ' + path)
    interpreter = request['interpreter']
    require(Path(sys.executable).resolve() == Path(interpreter['path']).resolve()
            and sha(sys.executable) == interpreter['sha256'], 'Current interpreter pin required')
    hook = request['provider_hook']
    require(set(hook) == {'path', 'sha256', 'install_name', 'options'} and hook['install_name'] == 'install'
            and Path(hook['path']).resolve() == STORE_HOOK.resolve()
            and pinned(request, hook['path']) == hook['sha256'], 'Pinned prepared store hook required')
    require(hook['options'].get('enabled') is True and hook['options'].get('allow_missing_fallback', False) is False
            and hook['options'].get('purpose') != 'pilot-shortbench', 'Complete exact prepared coverage required')
    perf = request.get('performance_hook')
    if perf is not None:
        require(set(perf) == {'path', 'sha256', 'options', 'acceptance_receipt'}
                and pinned(request, perf['path']) == perf['sha256'] and type(perf['options']) is dict,
                'Optional performance hook must be independently admitted')
        approval = perf['acceptance_receipt']
        require(sha(approval['path']) == approval['sha256'], 'Performance acceptance receipt changed')
        require(read(approval['path']).get('state') == 'accepted', 'Performance acceptance required; use null baseline otherwise')
    return output, deadline


def validate_args(args, request):
    for key, value in FIXED.items():
        require(getattr(args, key) == value, 'Scientific/continuation argument changed: ' + key)
    require(args.stop_after_validation == request['stop_after_validation'], 'Absolute target epoch mismatch')
    require(args.output.resolve() == Path(request['output_dir']).resolve() / 'model', 'Model output mismatch')
    require(args.stop_file.resolve() == Path(request['stop_file']).resolve() and args.deadline_utc == request['deadline_utc'],
            'Runtime deadline/stop mismatch')


class Budget:
    def __init__(self, request, deadline):
        self.request, self.deadline, self.started = request, deadline, time.perf_counter()

    def check(self):
        require(datetime.now(timezone.utc) < self.deadline
                and time.perf_counter() - self.started < self.request['maximum_seconds']
                and not Path(self.request['stop_file']).exists(), 'Stop/deadline/duration exhausted')


def stage_e1(output, old, budget):
    """Copy immutable input, never hardlink mutable latest/best publication names."""
    model = output / 'model'
    model.mkdir()
    write(model / 'config.json', successor_config(old))
    for source, destination in (('selected-epoch0001.pt', 'selected-epoch0001.pt'),
                                ('selected-epoch0001.pt', 'latest.pt'),
                                ('selected-epoch0001.json', 'selected-epoch0001.json'),
                                ('selected-epoch0001.json', 'latest.json'), ('selected.json', 'selected.json')):
        budget.check()
        shutil.copyfile(E1 / 'model' / source, model / destination)
    require(sha(model / 'latest.pt') == sha(model / 'selected-epoch0001.pt') == CHECKPOINT_SHA,
            'Staged checkpoint copy mismatch')


class Continuation:
    def __init__(self, train, identity, budget, output, old, target):
        self.train, self.identity, self.budget, self.output, self.old, self.target = train, identity, budget, output, old, target
        self.new = successor_config(old)
        self.original = {name: getattr(train, name) for name in ('update', 'evaluate', 'make_config', 'load_checkpoint')}
        self.original_resume = train.legacy.validate_resume
        self.objects = self.loaded_snapshot = self.loaded_state = None
        self.updates = self.examples = 0
        self.epochs = []
        self.cache = None
        self.first_update_verified = False

    def digest(self, value):
        return self.identity.state_digest(value, self.train.torch, self.train.np)

    def snapshot(self, model, provider, optimizer, scaler):
        return {key: self.digest(value) for key, value in (
            ('weights', model.state_dict()), ('optimizer', optimizer.state_dict()),
            ('scaler', scaler.state_dict()), ('rng', self.train.legacy.rng_state()))}

    def config(self, *args, **kwargs):
        self.budget.check()
        actual = self.original['make_config'](*args, **kwargs)
        validate_config(actual, self.old)
        return actual

    def load_checkpoint(self, path, config, **kwargs):
        saved = self.original['load_checkpoint'](path, config, **kwargs)
        if Path(path) == self.output / 'model/latest.pt' and self.loaded_snapshot is None:
            require(saved is not None, 'Resume must load full training state')
            validate_start_state(saved['state'])
            require(saved['config'] == self.old, 'Initial checkpoint is not original E1 config')
            self.loaded_state = copy.deepcopy(saved['state'])
            self.loaded_lrs = [group['lr'] for group in saved['optimizer']['param_groups']]
            self.loaded_snapshot = {target: self.digest(saved[source]) for source, target in (
                ('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng'))}
        return saved

    def install(self):
        require(self.train.run.__globals__ is self.train.__dict__, 'Frozen run binding changed')
        self.train.make_config, self.train.load_checkpoint = self.config, self.load_checkpoint
        self.train.update, self.train.evaluate = self.update, self.evaluate
        self.train.legacy.validate_resume = lambda saved, current: validate_resume_pair(saved, current, self.old, self.new)

    def restore(self):
        for name, value in self.original.items():
            setattr(self.train, name, value)
        self.train.legacy.validate_resume = self.original_resume
        if self.cache is not None:
            self.cache_stats = self.cache.stats()
            self.cache.uninstall()
            self.cache = None

    def update(self, *args, **kwargs):
        self.budget.check()
        require(len(args) == 6, 'Frozen update positional contract changed')
        model, provider, batches, optimizer, scaler, device = args
        require(str(device) == 'cuda' and len(batches) == 1 and 1 <= len(batches[0]) <= 128,
                'B128/accum1 CUDA update required')
        require(kwargs == {'hard_negatives': 0, 'uniform_negatives': 3, 'auxiliary_weight': 1.,
                          'candidate_chunk_size': 64, 'collect_loss': kwargs.get('collect_loss')}
                and type(kwargs['collect_loss']) is bool, 'Original U3/C4 objective required')
        objects = (model, provider, optimizer, scaler)
        if self.objects is None:
            require(self.loaded_snapshot is not None, 'Full E1 resume load required before any update')
            restored = self.snapshot(*objects)
            # Frozen run assigns the NEXT H5 scheduled LR immediately before update.
            # All moments/steps/groups must still equal the loaded checkpoint.
            current_optimizer = optimizer.state_dict()
            expected_lr = self.train.shared.learning_rate(27235, 27235 * 5, .05, .0001)
            require(len(current_optimizer['param_groups']) == len(self.loaded_lrs)
                    and all(group['lr'] == expected_lr for group in current_optimizer['param_groups']),
                    'First resumed LR must use global update27235 of H5')
            normalized = dict(current_optimizer)
            normalized['param_groups'] = [dict(group, lr=old_lr) for group, old_lr
                                         in zip(current_optimizer['param_groups'], self.loaded_lrs)]
            restored['optimizer'] = self.digest(normalized)
            require(self.loaded_snapshot is not None and restored == self.loaded_snapshot,
                    'Restored weights/AdamW/scaler/RNG differ from exact E1 checkpoint')
            self.first_update_verified = True
            self.objects = objects
            self.cache = self.cache_installer(provider)
        require(all(a is b for a, b in zip(self.objects, objects)), 'Training object ownership changed')
        require(self.updates < (self.target - 1) * 27235, 'No updates beyond target validation epoch')
        result = self.original['update'](*args, **kwargs)
        self.updates += 1
        self.examples += sum(map(len, batches))
        return result

    def evaluate(self, *args, **kwargs):
        self.budget.check()
        epoch = len(self.epochs) + 2
        require(epoch <= self.target and self.updates == (epoch - 1) * 27235
                and self.examples == (epoch - 1) * 3477437, 'Evaluation requires each new full epoch exactly once')
        require(len(args) == 4 and args[0] is self.objects[0] and args[1] is self.objects[1]
                and args[2] == 128 and str(args[3]) == 'cuda', 'Full original evaluator required')
        require(set(kwargs) == {'retrieval_topk', 'candidate_chunk_size', 'scoring_mode', 'should_stop'}
                and kwargs['retrieval_topk'] == 128 and kwargs['candidate_chunk_size'] == 64
                and kwargs['scoring_mode'] == 'rerank', 'Unchanged full rerank validation required')
        path = self.output / 'adjunct' / ('epoch%04d-predictions.npz' % epoch)
        require(not path.exists(), 'Fresh per-epoch predictions required')
        before = self.snapshot(*self.objects)
        metrics = self.original['evaluate'](*args, **kwargs, predictions_path=path)
        require(before == self.snapshot(*self.objects), 'Evaluation altered training weights/optimizer/scaler/RNG')
        require(metrics['examples'] == 158760 and metrics['known_targets'] == 154564
                and metrics['unknown_target_misses'] == 4196 and path.is_file(), 'Incomplete validation')
        context = SimpleNamespace(require=require, g=SimpleNamespace(canonical=canonical))
        replay = self.identity.replay_sidecar(context, self.train, self.objects[1], path, metrics)
        record = {'epoch': epoch, 'metrics': metrics, 'snapshot': before,
                  'sidecar': path.relative_to(self.output).as_posix(), 'sidecar_sha256': sha(path), 'replay': replay}
        self.epochs.append(record)
        write(path.with_suffix('.json'), record)
        return metrics


def finish(hook, result, sampler, output, request):
    """Prove latest cursor and selected weights independently; no model forward."""
    train = hook.train
    require(hook.first_update_verified, 'No verified continuation update occurred')
    latest = train.load_checkpoint(output / 'model/latest.pt', hook.new, map_location='cpu')
    require(latest['state'] == result['state'], 'Latest result cursor differs')
    if hook.objects is not None:
        expected = hook.snapshot(*hook.objects)
        for source, target in (('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng')):
            require(hook.digest(latest[source]) == expected[target], 'Latest complete state differs: ' + target)
    del latest
    state = result['state']
    selected = read(output / 'model/selected.json')
    selected_path = output / 'model' / selected['file']
    require(selected['epoch'] == state['selected_epoch'] and sha(selected_path) == selected['sha256'], 'Selected receipt differs')
    if selected['epoch'] == 1:
        require(selected['sha256'] == CHECKPOINT_SHA, 'E1 fallback checkpoint altered')
        selected_sidecar = (E1 / 'adjunct/predictions.npz').as_posix()
        selected_proof = {'source': 'accepted-E1-singlepass-receipt', 'checkpoint_sha256': CHECKPOINT_SHA}
    else:
        record = next(item for item in hook.epochs if item['epoch'] == selected['epoch'])
        require(selected['validation'] == record['metrics'], 'Selected/new epoch metrics differ')
        saved = train.load_checkpoint(selected_path, hook.new, map_location='cpu')
        for source, target in (('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng')):
            require(hook.digest(saved[source]) == record['snapshot'][target], 'Selected/evaluated state differs: ' + target)
        del saved
        selected_sidecar = (output / record['sidecar']).as_posix()
        selected_proof = {'source': 'successor-one-pass-epoch', 'checkpoint_sha256': selected['sha256']}
    completed_target = state['epoch'] == request['stop_after_validation'] and state['offset'] == 0
    require(result['reason'] in ('runtime_stop_after_validation', 'runtime_stop_file', 'runtime_deadline')
            or result['status'] == 'completed', 'Unexpected trainer termination')
    return {'state': 'completed' if completed_target else 'stopped-resumable',
            'target_completed': completed_target, 'schedule_completed': result['status'] == 'completed',
            'scientific_training_state': result['status'], 'reason': result['reason'],
            'continuation_from_epoch': 1, 'final_cursor': state, 'new_updates': hook.updates,
            'new_train_examples': hook.examples, 'first_update_full_state_equal_input': True,
            'evaluate_calls': len(hook.epochs), 'epochs': hook.epochs,
            'selected_epoch': selected['epoch'], 'selected_sidecar': selected_sidecar,
            'selected_checkpoint': selected_path.as_posix(), 'selected_proof': selected_proof,
            'sampler_contract': sampler.CONTRACT, 'sampler_audit': hook.sampler.replacement.audit,
            'test_accessed': False, 'runtime_performance': {'cpu_threads': 4, 'cpu_affinity': [0, 2, 4, 6],
            'dynamic_cache': hook.cache.stats() if hook.cache is not None else getattr(hook, 'cache_stats', None)},
            'execution_equivalence_claim': 'scientific-continuation; CPU2-to4 may change reduction rounding',
            'first_update_lr': 'global update27235 under original H5 denominator136175'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'contract': CONTRACT, 'fixed_training': FIXED, 'resource_caps': CAPS,
            'runtime_imported': False, 'target_epochs': [2, 3, 4], 'anchors': {p.as_posix(): h for p, h in ANCHORS.items()},
            'input_job': E1.as_posix(), 'checkpoint_sha256': CHECKPOINT_SHA,
            'native_resource_admission': 'root-owned', 'performance_hook_default': None}))
        return
    require(args.request is not None, '--request required')
    request = read(args.request)
    output, deadline = validate_request(request)
    budget = Budget(request, deadline)
    budget.check()
    require(os.environ.get('KION_OUTPUT_DIR') == output.as_posix()
            and os.environ.get('CUDA_VISIBLE_DEVICES') == request['gpu_uuid'], 'Root native environment differs')
    output.mkdir()
    (output / 'adjunct').mkdir()
    hook = provider_hook = None
    perf_handles = []
    provider_module = original_factory = None
    report = None
    try:
        old = read(E1 / 'model/config.json')
        stage_e1(output, old, budget)
        sys.path.insert(0, RUNTIME.as_posix())
        runtime = load(request, RUNTIME / 'kion_ready_runtime.py', '_night_bootstrap')
        # Explicit successor interpreter rebind; dependency paths and frozen scripts unchanged.
        runtime.BASE_EXE = Path(request['interpreter']['path'])
        runtime.configure_paths()
        train = load(request, RUNTIME / 'kion_complete_train.py', '_night_complete_train')
        parsed = train.parser().parse_args(request['train_argv'])
        validate_args(parsed, request)
        identity = load(request, IDENTITY, '_night_state_identity')
        sampler = load(request, SAMPLER, '_night_sampler')
        cache = load(request, CACHE, '_night_dynamic_cache')
        descriptor = request['provider_hook']
        store_module = load(request, descriptor['path'], '_night_store_hook')
        provider_hook = store_module.install(train, descriptor['options'])
        perf = request.get('performance_hook')
        if perf is not None:
            perf_module = load(request, perf['path'], '_night_optional_performance')
            module_name, factory_name = parsed.provider.split(':', 1)
            provider_module = train.importlib.import_module(module_name)
            original_factory = getattr(provider_module, factory_name)
            def performance_factory(factory_args):
                provider = original_factory(factory_args)
                try:
                    handle = perf_module.install(provider, **perf['options'])
                    require(callable(handle.uninstall) and callable(handle.evidence), 'Performance lifecycle API differs')
                    perf_handles.append(handle)
                    return provider
                except BaseException:
                    provider.close()
                    raise
            setattr(provider_module, factory_name, performance_factory)
        hook = Continuation(train, identity, budget, output, old, request['stop_after_validation'])
        hook.cache_installer = cache.install_dynamic_cache
        hook.sampler = sampler.InstalledSampler(train)
        hook.sampler.install()
        hook.install()
        try:
            result = train.main(request['train_argv'])
            budget.check()
            report = finish(hook, result, sampler, output, request)
        finally:
            hook.restore()
            hook.sampler.restore()
        report['provider_hook_evidence'] = provider_hook.evidence()
        report['performance_hook_evidence'] = [handle.evidence() for handle in perf_handles]
    except BaseException as error:
        report = {'state': 'failed', 'target_completed': False, 'selected_export_claim': False,
                  'error': type(error).__name__ + ': ' + str(error), 'test_accessed': False}
        raise
    finally:
        try:
            for handle in reversed(perf_handles):
                handle.uninstall()
            if provider_module is not None:
                setattr(provider_module, factory_name, original_factory)
            if provider_hook is not None:
                provider_hook.uninstall()
        except BaseException as error:
            report = {'state': 'failed-cleanup', 'target_completed': False, 'error': str(error)}
            raise
        finally:
            if report is not None:
                report.update(contract=CONTRACT, request_sha256=sha(args.request), source_hashes=request['source_hashes'],
                    interpreter=request['interpreter'], input_hashes=request['input_hashes'],
                    elapsed_seconds=time.perf_counter() - budget.started,
                    actual_runtime_change={'cpu_threads': {'from': 2, 'to': 4},
                        'bootstrap_interpreter': request['interpreter'], 'scientific_contract_changed': False},
                    native_resources='external-root-owned; this receipt is not guard admission')
                try:
                    budget.check()
                except ValueError as error:
                    report.update(state='stopped-budget', target_completed=False, budget_reason=str(error))
                write(output / 'worker-completion.json', report)
    return report


if __name__ == '__main__':
    main()
