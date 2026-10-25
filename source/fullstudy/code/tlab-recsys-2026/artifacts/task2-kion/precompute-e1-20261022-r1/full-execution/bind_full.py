"""Describe a full-store source binding or emit one root-decided request/plan; no jobs/E writes."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STORE = 'E:/codex-kion-cache-20261022-e1-r1/queries.sqlite3'
STORAGE_CONTRACT = 'kion-external-derived-query-store-v1'
GUARD_SHA = '9d25a5942cda0be54e56297c9c71dd28a3f57307165288c7b372af0e22a2cf2a'
WORKER_SHA = 'bb220140d90c0c3e032e96f3c79c973440138c44e8f2d20bc8a2ce959f7edc1d'
COUNT_PLAN_SHA = 'bd221e170e30f464719aeacd92e8bdccfcb42b5223343cd11eeee5e929b2f16b'
ROWS = {'train': 3477437, 'val': 158760}
MAX_SMALL = 8 << 20
MAX_DISK = 110_000_000_000
E_RESERVE = 16 << 30
D_RESERVE = 8 << 30


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def bound_json(path, expected=None):
    record, value = bound_bytes(path, expected)
    return json.loads(value.decode('utf-8')), record


def bound_bytes(path, expected=None):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_SMALL:
        raise ValueError('Only <=8MiB sources/metadata are read; no raw/cache/model bytes: ' + path.as_posix())
    value = path.read_bytes()
    digest = hashlib.sha256(value).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Exact source/metadata changed: ' + path.as_posix())
    return {'path': path.as_posix(), 'sha256': digest}, value


def exact(path, expected):
    if Path(path).resolve() != expected.resolve():
        raise ValueError('Only the exact experiment prerequisite path is admitted: ' + str(expected))
    return expected


def basis(paired_prefix, count_result, count_guard, storage_extension):
    paired_path = exact(paired_prefix, ROOT / 'paired-prefix-result-r1.json')
    count_path = exact(count_result, ROOT / 'key-count-r1/worker/result.json')
    guard_path = exact(count_guard, ROOT / 'key-count-r1/guard.json')
    extension_path = exact(storage_extension, ROOT / 'storage-extension-r1.json')
    paired, paired_pin = bound_json(paired_path)
    if (paired.get('state') != 'completed-actual-paired-prefix'
            or paired.get('parity', {}).get('same_state_parity') is not True
            or paired.get('parity', {}).get('successful_updates_per_variant') != 12
            or paired.get('useful_prefix_acceleration') is not True):
        raise ValueError('Actual useful12-update same-state prefix evidence required; no epoch inference')
    count, count_pin = bound_json(count_path)
    cleanup, count_guard_pin = bound_json(guard_path)
    count_plan_path = ROOT / 'keycount-execution/key-count-plan-r1.json'
    count_plan, count_plan_pin = bound_json(count_plan_path)
    count_request_path = ROOT / 'keycount-execution/key-count-request-r1.json'
    count_request, count_request_pin = bound_json(count_request_path, count_plan['worker_request']['sha256'])
    unique = count.get('result', {}).get('global_digest_distinct')
    if (canonical(count_plan) != COUNT_PLAN_SHA or count.get('state') != 'completed_digest_count'
            or count.get('result', {}).get('scanned_rows') != ROWS
            or type(unique) is not int or not 1 <= unique <= sum(ROWS.values())
            or count.get('source_pins') != count_request['sources']
            or count.get('metadata_pins') != count_request['metadata']
            or cleanup.get('state') != 'completed' or cleanup.get('cleanup_quiescent') is not True
            or cleanup.get('exit_code') != 0 or cleanup.get('plan_sha256') != COUNT_PLAN_SHA):
        raise ValueError('Exact full train/val digest count and positively closed native phase required')
    extension, extension_pin = bound_json(extension_path)
    cap = extension.get('max_total_bytes')
    if (extension.get('contract') != STORAGE_CONTRACT or extension.get('store_path') != STORE
            or type(cap) is not int or not (1 << 20) <= cap <= MAX_DISK
            or extension.get('free_space_reserve_bytes') != E_RESERVE
            or extension.get('receipt_free_space_reserve_bytes') != D_RESERVE
            or extension.get('extends_prior_30GB_D_store_cap') is not True):
        raise ValueError('Explicit root storage extension with exact E-store/reserves/bounded cap required')
    old_plan, _ = bound_json(ROOT / 'binding/pilot-plan-r2.json')
    old_request, _ = bound_json(ROOT / 'binding/pilot-request-r2.json', old_plan['worker_request']['sha256'])
    pilot_path = ROOT / 'pilot-r2/worker/result.json'
    pilot, pilot_pin = bound_json(pilot_path)
    if (pilot.get('state') != 'completed' or pilot.get('mode') != 'pilot'
            or pilot.get('parity', {}).get('state') != 'exact_bytes_equal'):
        raise ValueError('Pinned completed exact-byte pilot required')
    sources, metadata = dict(old_request['sources']), dict(old_request['metadata'])
    for path, expected in sources.items():
        bound_bytes(path, expected)
    for record in metadata.values():
        bound_bytes(record['path'], record['sha256'])
    for path, expected in ((ROOT / 'full-preparation/worker.py', WORKER_SHA),
                           (HERE / 'native_phase_guard.py', GUARD_SHA), (Path(__file__), None)):
        record, _ = bound_bytes(path, expected)
        sources[record['path']] = record['sha256']
    full_review, _ = bound_bytes(ROOT / 'full-preparation/independent-review-r2.json')
    guard_review, _ = bound_bytes(HERE / 'guard-review-r1.json')
    metadata.update({'paired_prefix': paired_pin, 'full_key_count_result': count_pin,
        'full_key_count_guard': count_guard_pin, 'full_key_count_plan': count_plan_pin,
        'full_key_count_request': count_request_pin, 'full_pilot_result': pilot_pin,
        'storage_extension': extension_pin, 'full_preparation_review': full_review,
        'full_guard_review': guard_review})
    store_pins = {**sources, **{value['path']: value['sha256'] for value in metadata.values()}}
    refs = {'pilot_result': pilot_pin, 'paired_prefix': paired_pin, 'count_result': count_pin,
        'count_guard': count_guard_pin, 'storage_extension': extension_pin}
    return {'sources': sources, 'metadata': metadata, 'store_source_pins': store_pins,
        'store_source_pins_sha256': canonical(store_pins), 'refs': refs,
        'max_total_bytes': cap, 'old_request': old_request, 'old_plan': old_plan,
        'global_digest_distinct': unique}


def build(values, decision, deadline_utc, wall_seconds, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    deadline = datetime.fromisoformat(deadline_utc.replace('Z', '+00:00'))
    if (type(wall_seconds) is not int or not 1 <= wall_seconds <= 24000
            or deadline.utcoffset() != timedelta(0)
            or deadline < now + timedelta(seconds=wall_seconds + 45)):
        raise ValueError('Root finite1..24000s wall plus45s cleanup must fit fresh UTC deadline')
    decision_path = Path(decision).resolve()
    if not decision_path.is_relative_to(ROOT) or decision_path.suffix != '.json':
        raise ValueError('Root decision must be small experiment JSON')
    if decision_path.as_posix() in values['store_source_pins']:
        raise ValueError('GO must remain outside store pins to avoid a hash cycle')
    go, decision_pin = bound_json(decision_path)
    if (go.get('state') != 'GO' or go.get('mode') != 'full'
            or go.get('store_source_pins_sha256') != values['store_source_pins_sha256']
            or any(go.get(name) != record for name, record in values['refs'].items())):
        raise ValueError('Separate root full-build GO must bind exact source basis/pilot/paired/count/storage evidence')
    run_dir = ROOT / 'full-store-r1'
    request_path, plan_path = HERE / 'full-request-r1.json', HERE / 'full-plan-r1.json'
    if any(path.exists() for path in (run_dir, request_path, plan_path)):
        raise FileExistsError('Fresh full-store-r1 and unissued binding files required; no overwrite/retry')
    old = values['old_request']
    request = {key: old[key] for key in ('expected_rows', 'common_scripts', 'data_dir', 'feature_dir',
        'scope_decision', 'codec_source', 'query_method_source')}
    request.update({'phase': 'cpu-build', 'mode': 'full', 'output': (run_dir / 'worker').as_posix(),
        'stop_file': (run_dir / 'stop-request.json').as_posix(), 'store_path': STORE,
        'deadline_utc': deadline.isoformat(), 'wall_seconds': wall_seconds, 'resume': False,
        'max_total_bytes': values['max_total_bytes'], 'free_space_reserve_bytes': E_RESERVE,
        'receipt_free_space_reserve_bytes': D_RESERVE,
        'sources': values['sources'], 'metadata': values['metadata'],
        'store_source_pins': values['store_source_pins'], 'storage_extension': values['refs']['storage_extension'],
        'full_build_decision': decision_pin})
    encoded = (json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    old_plan = values['old_plan']
    worker = {'path': (ROOT / 'full-preparation/worker.py').as_posix(), 'sha256': WORKER_SHA}
    plan = {'contract': 'kion-precompute-e1-phase-v1', 'phase': 'cpu-build',
        'interpreter': old_plan['interpreter'], 'worker': worker,
        'worker_request': {'path': request_path.as_posix(), 'sha256': hashlib.sha256(encoded).hexdigest()},
        'worker_argv': [old_plan['interpreter']['path'], '-B', worker['path'], '--request', request_path.as_posix()],
        'max_seconds': wall_seconds, 'deadline_utc': deadline.isoformat(), 'run_dir': run_dir.as_posix(),
        'cwd': ROOT.as_posix(), 'cpu_affinity': list(old_plan['cpu_affinity']),
        'resource_limits': {'aggregate_commit_bytes': 8 << 30, 'aggregate_rss_bytes': 8 << 30,
            'initial_host_available_bytes': 10 << 30, 'host_reserve_bytes': 2 << 30},
        'console_host': old_plan['console_host'], 'source_hashes': values['sources'],
        'metadata_pins': {**{v['path']: v['sha256'] for v in values['metadata'].values()},
            decision_pin['path']: decision_pin['sha256']}}
    for record in (plan['interpreter'], plan['console_host']):
        bound_bytes(record['path'], record['sha256'])
    if any(path.as_posix() in values['store_source_pins'] or path.as_posix() in plan['metadata_pins']
            for path in (request_path, plan_path)):
        raise ValueError('Request/plan must not participate in their own source/metadata binding')
    return request_path, encoded, plan_path, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('paired-prefix', 'count-result', 'count-guard', 'storage-extension'):
        parser.add_argument('--' + name, type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--decision', type=Path)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--wall-seconds', type=int)
    args = parser.parse_args()
    values = basis(args.paired_prefix, args.count_result, args.count_guard, args.storage_extension)
    if args.describe:
        if args.deadline_utc is not None or args.wall_seconds is not None:
            parser.error('Describe is source-only; clocks belong to a separate root GO binding')
        print(json.dumps({'state': 'source_only_basis_no_GO_no_job', 'store_path': STORE,
            'store_source_pins_sha256': values['store_source_pins_sha256'], 'refs': values['refs'],
            'max_total_bytes': values['max_total_bytes'], 'global_digest_distinct': values['global_digest_distinct'],
            'source_count': len(values['sources']), 'metadata_count': len(values['metadata'])}, indent=2))
        return
    if args.deadline_utc is None or args.wall_seconds is None:
        parser.error('Emission requires explicit root --deadline-utc and --wall-seconds')
    request_path, encoded, plan_path, plan = build(values, args.decision, args.deadline_utc, args.wall_seconds)
    with request_path.open('xb') as stream:
        stream.write(encoded)
    with plan_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'state': 'source_only_bound_no_approval_no_job', 'request': request_path.as_posix(),
        'plan': plan_path.as_posix(), 'plan_sha256': canonical(plan)}, indent=2))


if __name__ == '__main__':
    main()
