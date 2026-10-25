"""Root metadata preparation only after successful owned SELF-probe review."""
from datetime import datetime, timedelta, timezone
import copy
import hashlib
import json
from pathlib import Path

h = Path(__file__).resolve().parent
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def pin(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.as_posix(), sha256=digest)
def write_new(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

now = datetime.now(timezone.utc)
assert now + timedelta(hours=6, minutes=30) <= datetime(2026, 10, 23, 7, tzinfo=timezone.utc)
probe_run = h / 'sft-recovery/run-probe-r1'
probe_guard = read(probe_run / 'guard.json')
probe = read(probe_run / 'worker/probe-completion.json')
assert probe_guard['state'] == 'completed' and probe_guard['exit_code'] == 0
assert probe_guard['cleanup_quiescent'] is True and probe_guard['last_owned_members'] == []
assert probe['state'] == 'completed' and probe['checksum_equal'] is True
assert probe['helper_evidence']['empty_working_set_calls'] == 1
assert probe['helper_evidence']['self_pseudo_handle_verified'] is True
for name in ('core-review-r2.json', 'memory-review-r2.json', 'probe-result-review-r1.json'):
    review = read(h / 'review-sft-recovery' / name)
    assert review.get('findings', []) == []
    for source, digest in review.get('source_hashes', {}).items():
        assert pin(source)['sha256'] == digest
primary_guard = read(h / 'rl/run-rl-r1/guard.json')
assert primary_guard['state'] == 'completed' and primary_guard['cleanup_quiescent'] is True
assert read(h / 'review-rl-actual/review-r1.json')['findings'] == []

old_request_path = h / 'execution/sft-request-r1.json'
original = read(old_request_path)
partial = h / 'sft/run-r1/worker'
failed_guard = read(partial.parent / 'guard.json')
assert failed_guard['state'] == 'failed' and failed_guard['cleanup_quiescent'] is True
assert failed_guard['reason'] == "RuntimeError('owned_aggregate_RSS_above_admitted_cap')"
names = ('model/config.json', 'model/latest.pt', 'model/latest.json', 'model/selected.json',
    'model/selected-epoch0001.pt', 'model/selected-epoch0001.json', 'model/progress.jsonl')
files = {name: pin(partial / name)['sha256'] for name in names}
assert files['model/latest.pt'] == 'f42ce2712f22bd5c883deeade811e1cac836d9337f97a9fc12b846e5929b528e'
manifest = dict(contract='kion-closed-failed-partial-E2-input-v1', state='accepted',
    input_job=partial.as_posix(), files=files,
    producer_request=pin(old_request_path), producer_guard=pin(partial.parent / 'guard.json'),
    producer_plan=pin(h / 'execution/sft-plan-r1.json'),
    baseline_sidecar=pin(h.parent / 'deadline-20261022/control-r2/job/adjunct/predictions.npz'),
    closed_SELF_probe_guard=pin(probe_run / 'guard.json'),
    closed_SELF_probe_completion=pin(probe_run / 'worker/probe-completion.json'),
    probe_actual_review=pin(h / 'review-sft-recovery/probe-result-review-r1.json'),
    original_E1_acceptance=pin(h.parent / 'deadline-20261022/root-primary-acceptance-r1.json'),
    primary_RL_closure=pin(h / 'closure/run-rl-r1/root-closure.json'),
    initial_cursor=dict(epoch=1, offset=863676, group_index=6765, updates=34000, examples=4341113),
    remaining=dict(updates=20470, examples=2613761),
    no_whole_SFT_fit_guarantee=True, root_metadata_prepared_utc=now.isoformat())
manifest_path = h / 'execution/sft-recovery-input-manifest-r1.json'
write_new(manifest_path, manifest)
request = copy.deepcopy(original)
run = h / 'sft-recovery/run-recovery-r1'
assert not run.exists()
worker = run / 'worker'
stop = run / 'stop-request.json'
request.update(contract='kion-pinned-partial-E2-recovery-v1', state='admitted',
    input_job=partial.as_posix(), input_hashes=files, lineage_input_hashes=original['input_hashes'],
    input_manifest=pin(manifest_path), output=worker.as_posix(), output_dir=worker.as_posix(),
    stop_file=stop.as_posix(), stop_after_validation=2, deadline_utc='2026-10-23T07:00:00Z',
    maximum_seconds=21600,
    resource_limits=dict(aggregate_commit_bytes=16 << 30, aggregate_rss_bytes=16 << 30,
        initial_host_available_bytes=18 << 30, host_reserve_bytes=2 << 30))
for filename, expected in (
    ('worker.py', 'cf142bf5309e236570594575c14cb67549d460d10c47042d53b183f9395cc236'),
    ('soft_residency.py', '0ff85698e0b25eb675751d1c78cbf1b78561e89e051b2888147ce3e046797462')):
    value = pin(h / 'sft-recovery' / filename)
    assert value['sha256'] == expected
    request['source_hashes'][value['path']] = value['sha256']
request['source_hashes'][Path(__file__).resolve().as_posix()] = pin(__file__)['sha256']
request['soft_residency'] = {**pin(h / 'sft-recovery/soft_residency.py'), 'options': dict(
    contract='kion-self-soft-residency-v1', enabled=True,
    boundary='after-successful-train-update-return', trigger_rss_bytes=12 << 30,
    minimum_seconds_between_checks=60, minimum_successful_updates_between_checks=100,
    maximum_empty_working_set_calls=60, maximum_observed_native_api_seconds=30)}
for flag, value in (('--output', (worker / 'model').as_posix()),
                    ('--stop-file', stop.as_posix()), ('--deadline-utc', request['deadline_utc'])):
    assert request['train_argv'].count(flag) == 1
    request['train_argv'][request['train_argv'].index(flag) + 1] = value
request['root_recovery_admission'] = dict(
    basis='Measured original remaining training3.33–3.66h +historicalvalidation1.16h;6h whole-chain cap includes startup/checkpoints/paging allowance, not a proven upper bound',
    whole_budget_seconds=21600, target_completion_utc='2026-10-23T06:30:00Z',
    absolute_SFT_stop_utc='2026-10-23T07:00:00Z',
    recheck='After first500updates and firstSELFcall plus100updates; stop new work if conservative whole completion no longer fits',
    independent_native_plan_review_required=True, primary_result_preserved=True)
request_path = h / 'execution/sft-recovery-request-r1.json'
write_new(request_path, request)
checkpoint_path = h / 'checkpoint.json'
d = read(checkpoint_path)
assert d['heavy_jobs'] == []
d['closed_jobs'].append(dict(phase='owned_SELF_correctness_probe', root_session_id=62060,
    exit_code=0, cleanup_quiescent=True, elapsed_seconds=probe_guard['elapsed_seconds'],
    receipt=(probe_run / 'guard.json').as_posix(), actual_SELF_calls=1,
    output=probe_run.as_posix(), checksum_equal=True))
d['updated_utc'] = datetime.now(timezone.utc).isoformat()
d['pending_self_probe'].update(state='closed_pass_independently_reviewed', actual_SELF_calls=1,
    receipt=pin(probe_run / 'guard.json'), result=pin(probe_run / 'worker/probe-completion.json'))
d['optional_sft_recovery'].update(state='actual_manifest_and_request_prepared_native_plan_review_pending',
    input_manifest=pin(manifest_path), request=pin(request_path), maximum_seconds=21600,
    deadline_utc=request['deadline_utc'], native_probe_qualifier='64MiB SELF checksum only; no full model fit/speed guarantee')
d['next'] = 'Fresh quiet/resource/GPU observation, native binding and independent actual recovery plan review; only then one owned SFT job. Primary RL negative result is retained.'
checkpoint_path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(dict(state='recovery_metadata_prepared_not_launched', request=pin(request_path),
    input_manifest=pin(manifest_path), maximum_seconds=21600, deadline_utc=request['deadline_utc'])))
