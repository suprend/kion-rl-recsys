"""Metadata/source-only builder. Emits one fresh CPU pilot request/plan, no approval or job."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ARTIFACTS = ROOT.parent
PREV = ARTIFACTS / 'deadline-20261022'
CONTROL = PREV / 'control-r2'
READY = ARTIFACTS / 'ready-pack-20261020'
COMMON = READY / 'provider/runtime-native-r5'
DATA = Path('D:/tlab-recsys-2026/data/processed/kion-full-context64')
FEATURES = Path('D:/tlab-recsys-2026/data/processed/kion-ready-sft-20261020-r1/native-r6')
BASE = Path('C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
BASE_SHA = 'b141a0115de4be263a5cb0bdeb580ab67d074aaf1aadf10622fdf88ac2a1d8cb'
NEW_SOURCES = {
    'preparation/worker.py': '13f61bd71f285ca43991e66e1036bc1498b1920d3e585c8f5a5a33cff1c3398f',
    'preparation/store_hook.py': 'cb1307d3e869325174db9791f6665979bf7efbe12f38c6fbf546db56793d34bf',
    'query-store/query_store.py': '11d00afcebbcdc3b9b455a7f0fcfba3cb5d2ebc7a8128ec918c4b2ce316e34c3',
    'execution/native_phase_guard.py': '0b45946f866b59aae94f9cf2f1eaa6a550641c849d9468eb04d6d6b4c9024e1f',
}
QUERY_METHOD = PREV / 'perf-batch-query-r1/kion_complete_pack.py'
QUERY_SHA = '9897cf3f9c361d9c9c86f7929c43dd10795147119e6679fe1696621c19921014'
MAX_SMALL = 8 << 20


def read(path):
    if Path(path).stat().st_size > MAX_SMALL:
        raise ValueError('Only small source/metadata reads are admitted')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def small_pin(path, expected=None):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size > MAX_SMALL:
        raise ValueError('Only existing <=8MiB source/metadata files are hashed: ' + path.as_posix())
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Frozen source/metadata changed: ' + path.as_posix())
    return {'path': path.as_posix(), 'sha256': digest}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def build(now=None):
    now = datetime.now(timezone.utc) if now is None else now
    if now.utcoffset() != timedelta(0):
        raise ValueError('UTC clock required')
    deadline = (now + timedelta(minutes=45)).isoformat()
    state_path, prior_request_path = CONTROL / 'state/status.json', CONTROL / 'worker-request.json'
    state, prior = read(state_path), read(prior_request_path)
    if len(state['source_hashes']) != 68:
        raise ValueError('Expected exact68 COMMON-relative frozen source records')
    sources = {}
    for relative, expected in state['source_hashes'].items():
        path = (COMMON / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(COMMON.resolve()):
            raise ValueError('Invalid frozen COMMON source path')
        record = small_pin(path, expected)
        sources[record['path']] = record['sha256']
    for path, expected in prior['source_hashes'].items():
        record = small_pin(path, expected)
        if record['path'] in sources and sources[record['path']] != record['sha256']:
            raise ValueError('Conflicting prior external source pin')
        sources[record['path']] = record['sha256']
    for relative, expected in NEW_SOURCES.items():
        record = small_pin(ROOT / relative, expected)
        sources[record['path']] = record['sha256']
    for path in (Path(__file__), QUERY_METHOD):
        record = small_pin(path, QUERY_SHA if path == QUERY_METHOD else None)
        sources[record['path']] = record['sha256']
    # Preserve old expected bindings for these small provider inputs, never raw arrays.
    originals = {'scope_decision': READY / 'scope-decision-r1.json',
        'data_metadata': DATA / 'metadata.json', 'feature_manifest': FEATURES / 'manifest.json',
        'provider_source_freeze': READY / 'provider/source-freeze-native-r8.json'}
    metadata = {name: small_pin(path, state['input_hashes'][path.as_posix()])
        for name, path in originals.items()}
    metadata.update({'experiment_contract': small_pin(ROOT / 'contract.json'),
        'resource_admission': small_pin(ROOT / 'resource-admission-r1.json'),
        'guard_source_review': small_pin(HERE / 'guard-review-r2.json'),
        'preparation_source_review': small_pin(ROOT / 'preparation/independent-review-r2.json'),
        'codec_source_review': small_pin(ROOT / 'query-store/independent-review-r1.json'),
        'prior_completed_state': small_pin(state_path), 'prior_worker_request': small_pin(prior_request_path),
        'runtime_migration': small_pin(prior['specification']['runtime_migration']['path'],
            prior['specification']['runtime_migration']['sha256'])})
    store_pins = {**sources, **{value['path']: value['sha256'] for value in metadata.values()}}
    run_dir = ROOT / 'pilot-r2'
    request_path, plan_path = HERE / 'pilot-request-r2.json', HERE / 'pilot-plan-r2.json'
    if run_dir.exists() or request_path.exists() or plan_path.exists() or (HERE / 'pilot-review-r2').exists():
        raise FileExistsError('Fresh pilot-r2 and unissued binding files required; no overwrite/retry')
    request = {'phase': 'cpu-pilot', 'mode': 'pilot', 'output': (run_dir / 'worker').as_posix(),
        'stop_file': (run_dir / 'stop-request.json').as_posix(), 'deadline_utc': deadline,
        'wall_seconds': 900, 'resume': False, 'expected_rows': {'train': 3477437, 'val': 158760},
        'max_total_bytes': 1 << 30, 'free_space_reserve_bytes': 8 << 30, 'pilot_max_profiles': 4096,
        'common_scripts': (COMMON / 'scripts').as_posix(), 'data_dir': DATA.as_posix(),
        'feature_dir': FEATURES.as_posix(), 'scope_decision': (READY / 'scope-decision-r1.json').as_posix(),
        'sources': sources, 'metadata': metadata, 'store_source_pins': store_pins,
        'codec_source': small_pin(ROOT / 'query-store/query_store.py', NEW_SOURCES['query-store/query_store.py']),
        'query_method_source': small_pin(QUERY_METHOD, QUERY_SHA)}
    payload = (json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    console = Path(os.environ['SYSTEMROOT']) / 'System32/conhost.exe'
    plan = {'contract': 'kion-precompute-e1-phase-v1', 'phase': 'cpu-pilot',
        'interpreter': small_pin(BASE, BASE_SHA),
        'worker': small_pin(ROOT / 'preparation/worker.py', NEW_SOURCES['preparation/worker.py']),
        'worker_request': {'path': request_path.as_posix(), 'sha256': hashlib.sha256(payload).hexdigest()},
        'worker_argv': [BASE.as_posix(), '-B', (ROOT / 'preparation/worker.py').as_posix(),
            '--request', request_path.as_posix()],
        'max_seconds': 900, 'deadline_utc': deadline, 'run_dir': run_dir.as_posix(), 'cwd': ROOT.as_posix(),
        'resource_limits': {'aggregate_commit_bytes': 8 << 30, 'aggregate_rss_bytes': 8 << 30,
            'initial_host_available_bytes': 10 << 30, 'host_reserve_bytes': 2 << 30},
        'cpu_affinity': [4], 'console_host': small_pin(console, state['input_hashes'][console.as_posix()]),
        'source_hashes': sources, 'metadata_pins': {value['path']: value['sha256'] for value in metadata.values()}}
    if request_path.as_posix() in store_pins or plan_path.as_posix() in store_pins:
        raise ValueError('No request/plan hash cycles admitted')
    return request_path, payload, plan_path, plan


def snapshot(request_path, plan_path, plan):
    destination = HERE / 'pilot-review-r2'
    destination.mkdir()
    files = [request_path, plan_path, Path(__file__), ROOT / 'contract.json', ROOT / 'resource-admission-r1.json',
        HERE / 'guard-review-r2.json', ROOT / 'preparation/independent-review-r2.json',
        ROOT / 'query-store/independent-review-r1.json', ROOT / 'execution/test_native_phase_guard.py']
    files += [ROOT / relative for relative in NEW_SOURCES]
    files += [COMMON / 'scripts' / name for name in
        ('kion_complete_pack.py', 'kion_complete_train.py', 'kion_personal_candidates.py', 'kion_pack_history.py')]
    files += [QUERY_METHOD, ROOT / 'preparation/README.md', ROOT / 'preparation/test_preparation.py',
        ROOT / 'query-store/test_query_store.py']
    manifest = {}
    for index, original in enumerate(files):
        record = small_pin(original)
        expected = plan['source_hashes'].get(record['path'], plan['metadata_pins'].get(record['path']))
        if expected is not None and record['sha256'] != expected:
            raise ValueError('Bound source/metadata changed during review snapshot')
        payload = original.read_bytes()
        if hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('Snapshot source changed during exact-byte read')
        copied = destination / (str(index).zfill(2) + '-' + original.name)
        with copied.open('xb') as stream:
            stream.write(payload)
        manifest[record['path']] = {'snapshot_path': copied.as_posix(), 'sha256': record['sha256']}
    with (destination / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump({'contract': 'kion-pilot-review-snapshot-v1', 'plan_sha256': canonical(plan),
            'files': manifest, 'metadata_hashes_only': plan['metadata_pins'],
            'excluded': 'raw arrays, databases, learned bytes, large data/feature metadata; provider checks remain inside owned worker'},
            stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination


def main():
    request_path, payload, plan_path, plan = build()
    with request_path.open('xb') as stream:
        stream.write(payload)
    with plan_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    review = snapshot(request_path, plan_path, plan)
    print(json.dumps({'state': 'source_only_bound_no_approval_no_job', 'request': request_path.as_posix(),
        'plan': plan_path.as_posix(), 'plan_sha256': canonical(plan),
        'source_count': len(plan['source_hashes']), 'metadata_count': len(plan['metadata_pins']),
        'review_snapshot': review.as_posix(),
        'deadline_utc': plan['deadline_utc']}, indent=2))


if __name__ == '__main__':
    main()
