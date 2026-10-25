"""Explicit recovered-E2 -> H5 continuation; import/describe use stdlib only."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
H = BASE / 'night-20261023-r1'
BASE_WORKER = H / 'sft/worker.py'
BASE_SHA = 'fde07c575c26f4cb9cdb29a9d49c93e78243702592738da5e500ab146853efb0'
RECOVERY_WORKER = H / 'sft-recovery/worker.py'
RECOVERY_SHA = 'cf142bf5309e236570594575c14cb67549d460d10c47042d53b183f9395cc236'
SOFT_HELPER = H / 'sft-recovery/soft_residency.py'
SOFT_SHA = '0ff85698e0b25eb675751d1c78cbf1b78561e89e051b2888147ce3e046797462'
ORIGIN_JOB = H / 'sft-recovery/run-recovery-r1/worker'
CONTRACT = 'kion-recovered-E2-to-H5-v1'
MANIFEST_CONTRACT = 'kion-explicit-H5-continuation-input-v1'
RECOVERY_CONTRACT = 'kion-pinned-partial-E2-recovery-v1'
UPDATES, EXAMPLES, TOTAL_UPDATES = 27235, 3477437, 136175


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_base():
    require(hashlib.sha256(BASE_WORKER.read_bytes()).hexdigest() == BASE_SHA, 'Frozen base worker changed')
    spec = importlib.util.spec_from_file_location('_explicit_H5_base', BASE_WORKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


b = load_base()  # Frozen helper imports no ML/provider/database modules.


def checked_pin(pin):
    require(type(pin) is dict and set(pin) == {'path', 'sha256'}, 'Exact absolute path/SHA256 pin required')
    path = Path(pin['path'])
    require(path.is_absolute() and b.sha(path) == pin['sha256'], 'Changed input/provenance: ' + str(path))
    return path.resolve()


def validate_cursor(state, e1_validation=None):
    """Metadata checks; frozen TrainingSchedule additionally checks exact row/group boundary."""
    for key in ('epoch', 'offset', 'group_index', 'updates', 'examples', 'selected_epoch', 'bad_epochs', 'amp_overflow_retries'):
        require(type(state.get(key)) is int, 'Integer cursor required: ' + key)
    epoch, offset, group = state['epoch'], state['offset'], state['group_index']
    require(2 <= epoch <= 5 and 0 <= offset <= EXAMPLES and 0 <= group <= UPDATES,
            'Only completed E2 ancestry through H5 is supported')
    require((offset == 0) == (group == 0) and (offset == EXAMPLES) == (group == UPDATES),
            'Partial row/group boundary inconsistent')
    require(epoch < 5 or (offset == 0 and group == 0), 'No partial work beyond E5')
    require(state['updates'] == epoch * UPDATES + group and state['examples'] == epoch * EXAMPLES + offset,
            'Global counters must retain every completed/persisted prefix exactly once')
    history = state['history']
    require(type(history) is list and len(history) == epoch and [r['epoch'] for r in history] == list(range(1, epoch + 1)),
            'Exactly one validation per completed epoch is required')
    if e1_validation is not None:
        require(history[0]['validation'] == e1_validation, 'Original E1 validation history changed')
    best, selected, bad = -1., None, 0
    for row in history:
        score = row['validation']['ndcg@10']
        require(type(score) in (int, float) and math.isfinite(score), 'Finite validation metric required')
        if score > best:
            best, selected, bad = score, row['epoch'], 0
        else:
            bad += 1
    require(state['best_ndcg'] == best and state['selected_epoch'] == selected and state['bad_epochs'] == bad,
            'Strict-greater selection/best/patience history changed')
    require(state['amp_overflow_retries'] >= 0, 'AMP retry cursor invalid')
    require(history[1]['validation']['ndcg@10'] > history[0]['validation']['ndcg@10'],
            'Actual accepted E2 must improve E1; patience3 then permits all remaining epochs')


def check_guard(guard, producer, request_path):
    require(guard['contract'] == 'kion-night-owned-phase-receipt-v1'
            and guard['phase'] == 'sft-continuation' and guard['cleanup_quiescent'] is True,
            'Producer process tree must be closed and quiescent')
    require(guard['state'] == 'completed' and guard['exit_code'] == 0, 'Successful recovered E2 guard required')
    argv = guard['worker_argv']
    require(len(argv) == 5 and argv[1] == '-B' and argv[3] == '--request'
            and Path(argv[0]).resolve() == Path(producer['interpreter']['path']).resolve()
            and Path(argv[2]).resolve() == RECOVERY_WORKER and Path(argv[4]).resolve() == request_path,
            'Producer guard must bind exact interpreter/worker/request')


def validate_origin(pins, request):
    require(set(pins) == {'producer_request', 'producer_completion', 'producer_guard', 'root_closure', 'result_review'},
            'Exact accepted recovery origin evidence required')
    paths = {key: checked_pin(pin) for key, pin in pins.items()}
    values = {key: b.read(path) for key, path in paths.items()}
    producer, completion = values['producer_request'], values['producer_completion']
    require(paths['producer_completion'] == ORIGIN_JOB / 'worker-completion.json'
            and paths['producer_guard'] == ORIGIN_JOB.parent / 'guard.json'
            and Path(producer['output_dir']).resolve() == ORIGIN_JOB,
            'Only actual recovered E2 origin is supported')
    require(paths['root_closure'] == H / 'recovery-root-closure-r1.json'
            and paths['result_review'] == H / 'review-sft-recovery/actual-result-review-r2.json',
            'Exact root closure and final independent result review required')
    require(producer['contract'] == completion['contract'] == RECOVERY_CONTRACT
            and producer['state'] == 'admitted' and completion['state'] == 'completed'
            and producer['stop_after_validation'] == 2 and completion['target_completed'] is True
            and completion['schedule_completed'] is False and completion['evaluate_calls'] == 1
            and completion['new_updates'] == 20470 and completion['new_train_examples'] == 2613761
            and completion['persisted_prefix']['updates'] == 6765
            and completion['persisted_prefix']['examples'] == 863676
            and completion['first_update_full_state_equal_input'] is True,
            'Recovered E2 must account saved prefix plus exactly remaining work')
    require(producer['source_hashes'].get(RECOVERY_WORKER.as_posix()) == RECOVERY_SHA
            and completion['source_hashes'] == producer['source_hashes']
            and completion['request_sha256'] == pins['producer_request']['sha256'], 'Recovery source/request closure differs')
    check_guard(values['producer_guard'], producer, paths['producer_request'])
    cursor = completion['final_cursor']
    validate_cursor(cursor)
    require((cursor['epoch'], cursor['offset'], cursor['updates'], cursor['selected_epoch'], cursor['bad_epochs'])
            == (2, 0, 54470, 2, 0), 'Actual completed selected E2 boundary required')
    closure, review = values['root_closure'], values['result_review']
    require(closure['completed_sft_epochs'] == closure['selected_epoch'] == 2
            and closure['final_cursor'] == cursor and closure['cleanup_quiescent'] is True
            and closure['worker_completion']['sha256'] == pins['producer_completion']['sha256']
            and closure['guard']['sha256'] == pins['producer_guard']['sha256']
            and closure['selected_checkpoint']['sha256'] == completion['selected_proof']['checkpoint_sha256']
            and closure['baseline_predictions']['sha256'] == completion['epochs'][0]['sidecar_sha256'],
            'Root physical checkpoint/sidecar/closure differs')
    require(review['verdict'] == 'approved_actual_closed_result_with_stated_limits'
            and review['findings'] == [], 'Final actual-result source/physical evidence review required')
    require(request['lineage_input_hashes'] == producer['lineage_input_hashes']
            and request['provider_hook'] == producer['provider_hook']
            and request.get('performance_hook') == producer.get('performance_hook')
            and request['soft_residency'] == producer['soft_residency'], 'Prepared/performance/SELF/E1 ancestry changed')
    return producer, completion


def validate_manifest(request):
    manifest = b.read(checked_pin(request['input_manifest']))
    require(manifest['contract'] == MANIFEST_CONTRACT and manifest['state'] == 'accepted',
            'Explicit root-accepted closed input manifest required; no automatic retry')
    origin_request, origin_completion = validate_origin(manifest['origin_e2'], request)
    job = Path(manifest['input_job']).resolve()
    producer = origin_request
    require(job == ORIGIN_JOB == Path(request['input_job']).resolve(), 'First successor accepts actual closed recovered E2 only')
    state = manifest['saved_cursor']
    validate_cursor(state, origin_completion['final_cursor']['history'][0]['validation'])
    require(state['epoch'] < request['stop_after_validation'], 'Target must be after saved completed epoch')
    require(manifest['producer_request'] == manifest['origin_e2']['producer_request']
                and manifest['producer_guard'] == manifest['origin_e2']['producer_guard']
                and manifest['producer_completion'] == manifest['origin_e2']['producer_completion']
                and state == origin_completion['final_cursor']
                and manifest['files']['model/latest.pt'] == origin_completion['selected_proof']['checkpoint_sha256'],
                'Initial hop must restore actual physically accepted E2 exactly')
    require(producer['cpu_threads'] == 4 and producer['cpu_affinity'] == [0, 2, 4, 6], 'CPU4 lineage required')
    old, selected = b.read(b.E1 / 'model/config.json'), b.read(job / 'model/selected.json')
    require(b.read(job / 'model/config.json') == b.successor_config(old), 'Exact unchanged CPU4 scientific config required')
    require(selected['epoch'] == state['selected_epoch']
            and selected['validation'] == state['history'][selected['epoch'] - 1]['validation']
            and selected['file'] == 'selected-epoch%04d.pt' % selected['epoch'], 'Selected checkpoint/history differs')
    required = {'model/config.json', 'model/latest.pt', 'model/latest.json', 'model/selected.json',
                'model/selected-epoch0001.pt', 'model/selected-epoch0001.json',
                'model/' + selected['file'], 'model/' + selected['file'].replace('.pt', '.json')}
    require(set(manifest['files']) == required and request['input_hashes'] == manifest['files'], 'Exact staged model input file set required')
    for name, digest in manifest['files'].items():
        path = (job / name).resolve()
        require(path.is_relative_to(job) and b.sha(path) == digest, 'Persisted input changed: ' + name)
    latest = b.read(job / 'model/latest.json')
    require(latest == {'sha256': manifest['files']['model/latest.pt'], 'epoch': state['epoch'],
                       'offset': state['offset'], 'updates': state['updates']}, 'Latest receipt/persisted cursor differs')
    require(selected['sha256'] == manifest['files']['model/' + selected['file']]
            == b.read(job / 'model' / selected['file'].replace('.pt', '.json'))['sha256']
            and manifest['files']['model/selected-epoch0001.pt'] == b.CHECKPOINT_SHA, 'Selected/E1 physical pin differs')
    records = []
    require(len(manifest['epoch_records']) == state['epoch'] - 1, 'Every completed epoch since E2 needs immutable one-pass evidence')
    for expected_epoch, pins in enumerate(manifest['epoch_records'], 2):
        require(set(pins) == {'record', 'predictions'}, 'Exact record/predictions pair required')
        record_path, predictions = checked_pin(pins['record']), checked_pin(pins['predictions'])
        record = b.read(record_path)
        require(record['epoch'] == expected_epoch and record['metrics'] == state['history'][expected_epoch - 1]['validation']
                and record['sidecar_sha256'] == pins['predictions']['sha256']
                and (record_path.parent.parent / record['sidecar']).resolve() == predictions
                and record['replay']['rank_metrics_and_coverage_match'] is True
                and record['replay']['semantic_rows_targets_known_match'] is True
                and record['replay']['no_model_forward'] is True, 'Inherited one-pass record/history/replay differs')
        if expected_epoch == 2:
            require(record_path == ORIGIN_JOB / 'adjunct/epoch0002-predictions.json'
                    and record == origin_completion['epochs'][0], 'Exact original recovered E2 record required')
        record = copy.deepcopy(record)
        record['sidecar'] = predictions.as_posix()
        records.append(record)
    return {'job': job, 'manifest': manifest, 'start_state': state, 'records': records,
            'origin_request': origin_request, 'old_config': old}


class H5Continuation(b.Continuation):
    def __init__(self, *args, lineage, **kwargs):
        super().__init__(*args, **kwargs)
        self.lineage = lineage
        self.start = copy.deepcopy(lineage['start_state'])
        require((self.start['epoch'], self.start['offset'], self.start['group_index'], self.start['updates'], self.start['examples'])
                == (2, 0, 0, 54470, 6954874), 'This first successor starts only at closed full E2')
        self.updates, self.examples = self.start['updates'] - UPDATES, self.start['examples'] - EXAMPLES
        self.epochs = copy.deepcopy(lineage['records'])
        self.inherited_count = len(self.epochs)
        self.soft_control = None
        self.soft_evidence = {'enabled': True, 'constructor_returned': False, 'empty_working_set_calls': 0}
        self.first_operation = None

    def load_checkpoint(self, path, config, **kwargs):
        saved = self.original['load_checkpoint'](path, config, **kwargs)
        if Path(path) == self.output / 'model/latest.pt' and self.loaded_snapshot is None:
            require(saved is not None and saved['config'] == self.new and saved['state'] == self.start,
                    'Full pinned CPU4 checkpoint cursor/config must equal accepted manifest')
            validate_cursor(saved['state'])
            self.loaded_state = copy.deepcopy(saved['state'])
            self.loaded_lrs = [group['lr'] for group in saved['optimizer']['param_groups']]
            self.loaded_snapshot = {target: self.digest(saved[source]) for source, target in (
                ('state_dict', 'weights'), ('optimizer', 'optimizer'), ('scaler', 'scaler'), ('rng', 'rng'))}
        return saved

    def verify_objects(self, model, provider, optimizer, scaler):
        if self.objects is not None:
            return
        require(self.loaded_snapshot is not None, 'Full pinned restore required before first operation')
        objects = (model, provider, optimizer, scaler)
        restored, current = self.snapshot(*objects), optimizer.state_dict()
        expected = self.train.shared.learning_rate(self.start['updates'], TOTAL_UPDATES, .05, .0001)
        require(len(current['param_groups']) == len(self.loaded_lrs)
                    and all(group['lr'] == expected for group in current['param_groups']), 'Next LR must retain original global H5 schedule')
        normalized = dict(current, param_groups=[dict(group, lr=lr) for group, lr in zip(current['param_groups'], self.loaded_lrs)])
        restored['optimizer'] = self.digest(normalized)
        require(restored == self.loaded_snapshot, 'Restored weights/AdamW/scaler/RNG differ from full pinned checkpoint')
        self.objects, self.first_update_verified, self.first_operation = objects, True, 'update'
        self.cache = self.cache_installer(provider)

    def update(self, *args, **kwargs):
        self.budget.check()
        require(len(args) == 6, 'Frozen update positional contract changed')
        self.verify_objects(args[0], args[1], args[3], args[4])
        result = super().update(*args, **kwargs)
        # Entire original backward/scaler/AMP retry has returned successfully.
        if self.soft_control is None:
            self.soft_control = self.lineage['soft_module'].Control(self.lineage['soft_options'],
                self.output / 'adjunct/self-residency.jsonl', initial_global_update=self.start['updates'],
                budget_check=self.budget.check)
        self.soft_control.after_successful_update(self.updates - (self.start['updates'] - UPDATES))
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
        budget.check()
        shutil.copyfile(lineage['job'] / name, output / name)
        require(b.sha(output / name) == digest, 'Independent staged full-state copy differs: ' + name)


def validate_deadline(value):
    deadline = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None and deadline.utcoffset().total_seconds() == 0
            and deadline > datetime.now(timezone.utc), 'Fresh explicit absolute UTC deadline required')
    return deadline


def adapt(request):
    original_validate, original_finish, original_write = b.validate_request, b.finish, b.write
    b.HERE, b.CONTRACT, b.validate_deadline = HERE, CONTRACT, validate_deadline
    context = {}

    def validate(value):
        require(value['phase'] == 'sft-e5' and type(value['stop_after_validation']) is int
                and value['stop_after_validation'] == 5, 'This first successor must target five completed epochs')
        require(type(value['maximum_seconds']) in (int, float) and math.isfinite(value['maximum_seconds'])
                and 0 < value['maximum_seconds'] <= 72000, 'Bounded at most20h duration required')
        b.pinned(value, Path(__file__))
        require(b.pinned(value, BASE_WORKER) == BASE_SHA and b.pinned(value, RECOVERY_WORKER) == RECOVERY_SHA,
                'Exact frozen producer/helper sources required')
        lineage = validate_manifest(value)
        descriptor = value['soft_residency']
        require(type(descriptor) is dict and set(descriptor) == {'path', 'sha256', 'options'}
                and Path(descriptor['path']).resolve() == SOFT_HELPER and descriptor['sha256'] == SOFT_SHA
                and b.pinned(value, descriptor['path']) == SOFT_SHA, 'Exact accepted SELF helper required')
        module = b.load(value, descriptor['path'], '_H5_self_residency')
        module.validate_options(descriptor['options'])
        lineage.update(soft_module=module, soft_options=descriptor['options'])
        # Base validation remains intact for sources, original E1 ancestry,
        # resource bounds, runtime argv, interpreter, provider and performance.
        # Only the explicitly reviewed new phase/target/duration are projected.
        translated = dict(value, phase='sft-continuation', stop_after_validation=3,
            maximum_seconds=min(value['maximum_seconds'], 54000), input_job=b.E1.as_posix(),
            input_hashes=value['lineage_input_hashes'])
        result = original_validate(translated)
        context.update(lineage)
        return result

    class BoundContinuation(H5Continuation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, lineage=context, **kwargs)
            context['active_hook'] = self

    def finish(hook, result, sampler, output, value):
        report = original_finish(hook, result, sampler, output, value)
        state = result['state']
        validate_cursor(state)
        added = hook.epochs[hook.inherited_count:]
        require([r['epoch'] for r in added] == list(range(hook.start['epoch'] + 1, state['epoch'] + 1)),
                'Exactly one validation for every newly completed epoch required')
        require(state['updates'] == hook.updates + UPDATES and state['examples'] == hook.examples + EXAMPLES,
                'Trainer cursor and all persisted/new update counters differ')
        if report['target_completed']:
            require(state['epoch'] == value['stop_after_validation'] and state['offset'] == state['group_index'] == 0,
                    'Complete absolute target boundary required')
        if state['epoch'] == 5:
            require(state['updates'] == 136175 and state['examples'] == 17387185
                    and result['status'] == 'completed' and result['reason'] == 'runtime_stop_after_validation'
                    and result['predictions_status'] == 'deferred-runtime-stop', 'H5 must close with no terminal duplicate forward')
        report.update(continuation_from_epoch=hook.start['epoch'], continuation_from_offset=hook.start['offset'],
            continuation_from_global_update=hook.start['updates'],
            new_updates=state['updates'] - hook.start['updates'], new_train_examples=state['examples'] - hook.start['examples'],
            evaluate_calls=len(added), epochs=added, inherited_epochs=hook.epochs[:hook.inherited_count],
            first_operation_full_state_equal_input=True, first_operation=hook.first_operation,
            first_update_full_state_equal_input=True if state['updates'] > hook.start['updates'] else None,
            first_update_lr='global%d/H5 denominator136175' % hook.start['updates'],
            schedule_completed=state['epoch'] == 5 and state['offset'] == 0,
            execution_equivalence_claim='scientific full-state continuation from accepted CPU4 checkpoint; CPU4 unchanged')
        return report

    def write(path, value):
        if Path(path).name == 'worker-completion.json':
            active = context.get('active_hook')
            latest = Path(path).parent / 'model/latest.json'
            value = dict(value, input_manifest=request['input_manifest'],
                origin_e2=context.get('manifest', {}).get('origin_e2'),
                actual_runtime_change={'cpu_threads': {'from': 4, 'to': 4},
                    'bootstrap_interpreter': request['interpreter'], 'scientific_contract_changed': False},
                soft_residency=active.soft_evidence if active is not None else {'constructor_returned': False},
                persisted_latest_receipt=b.read(latest) if latest.is_file() else None,
                resume_policy='diagnostic receipt only; closed partial output needs separately investigated source and actual-manifest review; never automatically rerun E2',
                residency_control='unchanged bounded advisory SELF-only after completed update; no fit/commit/speed guarantee')
        original_write(path, value)

    b.validate_request, b.Continuation = validate, BoundContinuation
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
            'start_global_update': 54470, 'targets': [5], 'maximum_seconds': 72000,
            'schedule_total_updates': TOTAL_UPDATES, 'final_examples': EXAMPLES * 5,
            'runtime_imported': False, 'no_RL': True, 'actual_native_admission': 'external-root-only'}))
        return
    require(args.request is not None, '--request required')
    request = b.read(args.request)
    adapt(request)
    return b.main(['--request', args.request.as_posix()])


if __name__ == '__main__':
    main()
