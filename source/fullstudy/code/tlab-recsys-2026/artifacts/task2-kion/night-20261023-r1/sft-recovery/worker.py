"""Optional pinned partial-E2 recovery. Import/describe/tests use stdlib only."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE_WORKER = HERE.parent / 'sft/worker.py'
BASE_SHA = 'fde07c575c26f4cb9cdb29a9d49c93e78243702592738da5e500ab146853efb0'
FAILED_JOB = HERE.parent / 'sft/run-r1/worker'
PRODUCER_CONTRACT = 'kion-night-E1-continuation-v1'
CONTRACT = 'kion-pinned-partial-E2-recovery-v1'
MANIFEST_CONTRACT = 'kion-closed-failed-partial-E2-input-v1'
LATEST_SHA = 'f42ce2712f22bd5c883deeade811e1cac836d9337f97a9fc12b846e5929b528e'
PREFIX_UPDATES, PREFIX_EXAMPLES = 6765, 863676
START_UPDATES, START_EXAMPLES = 34000, 4341113
SOFT_HELPER = HERE / 'soft_residency.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_base():
    require(hashlib.sha256(BASE_WORKER.read_bytes()).hexdigest() == BASE_SHA, 'Accepted frozen worker changed')
    spec = importlib.util.spec_from_file_location('_partial_E2_recovery_base', BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


b = load_base()  # This fixed helper is stdlib-only until admitted main executes.


def checked_pin(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact path/SHA256 pin required')
    path = Path(value['path'])
    require(path.is_absolute() and b.sha(path) == value['sha256'], 'Input/provenance pin changed: ' + str(path))
    return path.resolve()


def validate_partial_state(state, e1_validation, saved_log):
    for key, expected in (('epoch', 1), ('offset', PREFIX_EXAMPLES), ('group_index', PREFIX_UPDATES),
                           ('updates', START_UPDATES), ('examples', START_EXAMPLES),
                           ('selected_epoch', 1), ('bad_epochs', 0)):
        require(type(state.get(key)) is int and state[key] == expected, 'Exact persisted partial E2 cursor required: ' + key)
    require(state['history'] == [{'epoch': 1, 'validation': e1_validation}]
            and state['best_ndcg'] == e1_validation['ndcg@10'], 'Original evaluated E1 selection/history must remain intact')
    require(state['amp_overflow_retries'] == saved_log['amp_overflow_retries'], 'Saved AMP retry cursor differs from same-update log')


def check_failed_producer(request, guard):
    require(request['contract'] == PRODUCER_CONTRACT and request['state'] == 'admitted'
            and request['stop_after_validation'] == 2 and request['cpu_threads'] == 4
            and request['source_hashes'].get(BASE_WORKER.as_posix()) == BASE_SHA,
            'Only accepted original E1-to-E2 producer is supported')
    require(guard['contract'] == 'kion-night-owned-phase-receipt-v1' and guard['phase'] == 'sft-continuation'
            and guard['state'] == 'failed' and guard['exit_code'] == 1 and guard['cleanup_quiescent'] is True
            and guard['reason'] == "RuntimeError('owned_aggregate_RSS_above_admitted_cap')"
            and guard['resource_limits']['aggregate_rss_bytes'] == 14 << 30
            and guard['peak_tree_RSS_bytes'] > 14 << 30,
            'Only this closed, failed14GiB RSS producer is eligible for explicit recovery')


def validate_manifest(request):
    manifest = b.read(checked_pin(request['input_manifest']))
    require(manifest['contract'] == MANIFEST_CONTRACT and manifest['state'] == 'accepted',
            'Separately reviewed root partial-recovery input manifest required')
    producer_request_path = checked_pin(manifest['producer_request'])
    guard_path = checked_pin(manifest['producer_guard'])
    producer, guard = b.read(producer_request_path), b.read(guard_path)
    job = Path(manifest['input_job']).resolve()
    require(job == FAILED_JOB and job == Path(request['input_job']).resolve()
            and job == Path(producer['output_dir']).resolve() and guard_path == job.parent / 'guard.json',
            'Only exact preserved run-r1 partial output is supported')
    check_failed_producer(producer, guard)
    argv = guard['worker_argv']
    require(len(argv) == 5 and argv[1] == '-B' and argv[3] == '--request'
            and Path(argv[0]).resolve() == Path(producer['interpreter']['path']).resolve()
            and Path(argv[2]).resolve() == BASE_WORKER and Path(argv[4]).resolve() == producer_request_path,
            'Closed guard worker/interpreter/request binding differs')
    require(request['lineage_input_hashes'] == producer['input_hashes']
            and request['provider_hook'] == producer['provider_hook']
            and request.get('performance_hook') == producer.get('performance_hook'),
            'Preserve exact E1 ancestry and accepted provider/performance settings')
    required = {'model/config.json', 'model/latest.pt', 'model/latest.json', 'model/selected.json',
                'model/selected-epoch0001.pt', 'model/selected-epoch0001.json', 'model/progress.jsonl'}
    require(set(manifest['files']) == required and request['input_hashes'] == manifest['files'], 'Exact closed input file set required')
    for name, digest in manifest['files'].items():
        path = (job / name).resolve()
        require(path.is_relative_to(job) and b.sha(path) == digest, 'Closed partial input changed: ' + name)
    latest = b.read(job / 'model/latest.json')
    require(latest == {'sha256': LATEST_SHA, 'epoch': 1, 'offset': PREFIX_EXAMPLES, 'updates': START_UPDATES}
            and manifest['files']['model/latest.pt'] == LATEST_SHA, 'Physically pinned checkpoint34000 required')
    old = b.read(b.E1 / 'model/config.json')
    require(b.read(job / 'model/config.json') == b.successor_config(old), 'Exact accepted CPU4 scientific config required')
    selected = b.read(job / 'model/selected.json')
    e1_selected = b.read(b.E1 / 'model/selected.json')
    require(selected == e1_selected and selected['epoch'] == 1
            and manifest['files']['model/selected-epoch0001.pt'] == b.CHECKPOINT_SHA
            and b.read(job / 'model/selected-epoch0001.json')['sha256'] == b.CHECKPOINT_SHA,
            'Original evaluated E1 checkpoint remains selected')
    baseline = checked_pin(manifest['baseline_sidecar'])
    require(baseline == b.E1 / 'adjunct/predictions.npz'
            and manifest['baseline_sidecar']['sha256'] == request['lineage_input_hashes']['adjunct/predictions.npz'],
            'Original E1 evaluated sidecar pin required')
    lines = [json.loads(line) for line in (job / 'model/progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(not [line for line in lines if line.get('event') == 'validation'], 'No E2 validation may already have happened')
    saved = [line for line in lines if line.get('event') == 'training' and line['updates'] == START_UPDATES]
    require(len(saved) == 1, 'Unique same-update prefix log evidence required')
    saved = saved[0]
    validate_partial_state(saved, selected['validation'], saved)
    observed = max((line for line in lines if line.get('event') == 'training'), key=lambda line: line['updates'])
    require(observed['updates'] >= START_UPDATES, 'Observed log cannot predate saved prefix')
    timing_keys = ('updated_utc', 'epoch', 'offset', 'updates', 'examples', 'group_index', 'elapsed_seconds',
        'epoch_train_wall_seconds', 'epoch_validation_wall_seconds', 'epoch_checkpoint_wall_seconds',
        'runner_initialization_wall_seconds', 'total_session_elapsed')
    return {'job': job, 'manifest': manifest, 'producer_request': producer, 'guard': guard,
            'e1_validation': selected['validation'], 'saved_log': saved,
            'prefix_timing_observed_at_saved_update': {key: saved[key] for key in timing_keys},
            'last_observed_cursor': {key: observed[key] for key in ('updated_utc', 'epoch', 'offset', 'updates', 'examples', 'group_index')}}


class RecoveryContinuation(b.Continuation):
    def __init__(self, *args, lineage, **kwargs):
        super().__init__(*args, **kwargs)
        self.lineage = lineage
        # Parent hook counts all E2 work. Offset counters acknowledge only the
        # persisted prefix; original frozen scheduler consumes this same offset.
        self.updates, self.examples = PREFIX_UPDATES, PREFIX_EXAMPLES
        self.soft_control = None
        self.soft_evidence = {'enabled': False, 'native_calls': 0}

    def load_checkpoint(self, path, config, **kwargs):
        saved = self.original['load_checkpoint'](path, config, **kwargs)
        if Path(path) == self.output / 'model/latest.pt' and self.loaded_snapshot is None:
            require(saved is not None and saved['config'] == self.new, 'Full preserved CPU4 partial checkpoint required')
            validate_partial_state(saved['state'], self.lineage['e1_validation'], self.lineage['saved_log'])
            self.loaded_state = copy.deepcopy(saved['state'])
            self.loaded_lrs = [group['lr'] for group in saved['optimizer']['param_groups']]
            self.loaded_snapshot = {target: self.digest(saved[source]) for source, target in (
                ('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng'))}
        return saved

    def update(self, *args, **kwargs):
        self.budget.check()
        if self.objects is None:
            require(self.loaded_snapshot is not None and len(args) == 6, 'Verified partial checkpoint load required before update')
            model, provider, batches, optimizer, scaler, device = args
            objects = (model, provider, optimizer, scaler)
            restored, current = self.snapshot(*objects), optimizer.state_dict()
            expected_lr = self.train.shared.learning_rate(START_UPDATES, 136175, .05, .0001)
            require(len(current['param_groups']) == len(self.loaded_lrs)
                    and all(group['lr'] == expected_lr for group in current['param_groups']),
                    'First resumed LR must use global34000/H5 denominator136175')
            normalized = dict(current, param_groups=[dict(group, lr=lr) for group, lr in zip(current['param_groups'], self.loaded_lrs)])
            restored['optimizer'] = self.digest(normalized)
            require(restored == self.loaded_snapshot, 'Restored partial weights/AdamW/scaler/RNG differ')
            self.objects, self.first_update_verified = objects, True
            self.cache = self.cache_installer(provider)
        result = super().update(*args, **kwargs)
        # This is after the original entire update returned: backward, clipping,
        # scaler step/update and any AMP retry are finished. No feature callback.
        module = self.lineage.get('soft_module')
        if module is not None:
            if self.soft_control is None:
                self.soft_evidence = {'enabled': True, 'constructor_returned': False, 'empty_working_set_calls': 0}
                self.soft_control = module.Control(self.lineage['soft_options'],
                    self.output / 'adjunct/self-residency.jsonl', initial_global_update=START_UPDATES,
                    budget_check=self.budget.check)
            self.soft_control.after_successful_update(self.updates - PREFIX_UPDATES)
        return result

    def restore(self):
        try:
            if self.soft_control is not None:
                try:
                    self.soft_control.close()
                finally:
                    self.soft_evidence = self.soft_control.evidence()
        finally:
            super().restore()


def stage(output, budget, lineage):
    (output / 'model').mkdir()
    for name, digest in lineage['manifest']['files'].items():
        if name == 'model/progress.jsonl':
            continue  # Preserve original log externally; new segment has its own log.
        budget.check()
        shutil.copyfile(lineage['job'] / name, output / name)
        require(b.sha(output / name) == digest, 'Independent staged checkpoint copy differs: ' + name)


def adapt(request):
    original_validate, original_finish, original_write = b.validate_request, b.finish, b.write
    b.HERE, b.CONTRACT = HERE, CONTRACT
    context = {}

    def validate(value):
        require(value['contract'] == CONTRACT and value['stop_after_validation'] == 2, 'Only full remainder of E2 is supported')
        require(value['resource_limits']['aggregate_rss_bytes'] == 16 << 30, 'Recovery needs separate root16GiB admission')
        b.pinned(value, Path(__file__))
        lineage = validate_manifest(value)
        descriptor = value.get('soft_residency')
        if descriptor is None:
            lineage.update(soft_module=None, soft_options=None)
        else:
            require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256', 'options'}
                    and Path(descriptor['path']).resolve() == SOFT_HELPER
                    and b.pinned(value, descriptor['path']) == descriptor['sha256'],
                    'Exact local pinned SELF helper descriptor required')
            module = b.load(value, descriptor['path'], '_owned_recovery_soft_residency')
            module.validate_options(descriptor['options'])  # Probe thresholds rejected.
            lineage.update(soft_module=module, soft_options=descriptor['options'])
        translated = dict(value, input_job=b.E1.as_posix(), input_hashes=value['lineage_input_hashes'])
        result = original_validate(translated)
        context.update(lineage)
        return result

    class BoundRecovery(RecoveryContinuation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, lineage=context, **kwargs)
            context['active_hook'] = self

    def finish(hook, result, sampler, output, value):
        report = original_finish(hook, result, sampler, output, value)
        remaining_updates, remaining_examples = hook.updates - PREFIX_UPDATES, hook.examples - PREFIX_EXAMPLES
        require(len(hook.epochs) == 1 and hook.epochs[0]['epoch'] == 2
                and remaining_updates == 20470 and remaining_examples == 2613761,
                'Exactly full remaining E2 training and one validation required')
        report.update(continuation_from_epoch=1, continuation_from_offset=PREFIX_EXAMPLES,
            continuation_from_global_update=START_UPDATES, new_updates=remaining_updates,
            new_train_examples=remaining_examples, epoch2_total_updates=hook.updates, epoch2_total_examples=hook.examples,
            first_update_lr='global34000/H5 denominator136175',
            persisted_prefix={'updates': PREFIX_UPDATES, 'examples': PREFIX_EXAMPLES,
                'timing_observed_at_saved_update': context['prefix_timing_observed_at_saved_update'],
                'last_observed_cursor': context['last_observed_cursor'],
                'observed_updates_beyond_saved_checkpoint': context['last_observed_cursor']['updates'] - START_UPDATES,
                'producer_run_guard_elapsed_seconds': context['guard']['elapsed_seconds']},
            epoch_timing_policy='new frozen trainer timings describe resumed segment; prior prefix observations kept separately, not added as an exact whole-epoch time',
            execution_equivalence_claim='scientific continuation from fixed CPU4 partial checkpoint; no CPU thread change')
        return report

    def write(path, value):
        if Path(path).name == 'worker-completion.json':
            active = context.get('active_hook')
            value = dict(value, input_manifest=request['input_manifest'],
                actual_runtime_change={'cpu_threads': {'from': 4, 'to': 4},
                    'bootstrap_interpreter': request['interpreter'], 'scientific_contract_changed': False},
                soft_residency=active.soft_evidence if active is not None else {'enabled': False},
                residency_control='optional advisory SELF-only completed-update hook; no16GiB fit/commit guarantee')
        original_write(path, value)

    b.validate_request, b.Continuation = validate, BoundRecovery
    b.stage_e1 = lambda output, old, budget: stage(output, budget, context)
    b.finish, b.write = finish, write


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'contract': CONTRACT, 'manifest_contract': MANIFEST_CONTRACT,
            'base_worker': {'path': BASE_WORKER.as_posix(), 'sha256': BASE_SHA},
            'checkpoint_sha256': LATEST_SHA, 'global_update': START_UPDATES, 'epoch': 1,
            'offset': PREFIX_EXAMPLES, 'group_index': PREFIX_UPDATES, 'target_epoch': 2,
            'remaining_updates': 20470, 'remaining_examples': 2613761, 'runtime_imported': False,
            'soft_residency_default': None, 'soft_residency_helper': SOFT_HELPER.as_posix(),
            'native_resources_and_RL_priority_and_deadline_fit_admission': 'external-root-only; source does not authorize a run'}))
        return
    require(args.request is not None, '--request required')
    request = b.read(args.request)
    adapt(request)
    return b.main(['--request', args.request.as_posix()])


if __name__ == '__main__':
    main()
