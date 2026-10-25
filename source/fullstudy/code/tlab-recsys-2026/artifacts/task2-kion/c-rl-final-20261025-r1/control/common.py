"""Small, stdlib-only admission boundary for the root-owned C3 campaign."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

CONTROL = Path(__file__).resolve().parent
ROOT = CONTROL.parent
BASE = Path('C:/Users/Михаил/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe')
BASE_SHA = '5640e9be2fd33d05a1a9ae1228e2354609a69ab831605995ec4e2bd6da8fff5e'
PAS = Path('D:/personal-agent-system')
RUNTIME = Path('D:/tlab-recsys-2026/artifacts/task2-kion/ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py')
GIB = 2**30
STAGES = {'prepare_cache', 'smoke', 'interpret', 'cohort_audit', 'reward_probe', 'frontier', 'train_rl', 'evaluate', 'live_prefix', 'analyze'}
FUNCTIONS = {'run', 'analyze', 'probe', 'score', 'prepare_cache', 'frontier', 'inference_frontier', 'train', 'score_checkpoint', 'evaluate_selected', 'rollout', 'smoke', 'evaluate'}
SOURCE_DIRS = {'interpretation', 'cohorts', 'reward', 'policy', 'reports'}

def require(value, message):
    if not value:
        raise ValueError(message)

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)

def stat_key(path):
    value = Path(path).stat()
    return {'size': value.st_size, 'mtime_ns': value.st_mtime_ns, 'ctime_ns': value.st_ctime_ns}

def deadline(request):
    value = datetime.fromisoformat(request['deadline_utc'])
    require(value.tzinfo is not None, 'Deadline must be timezone aware')
    return value

def remaining(request):
    return (deadline(request) - datetime.now(timezone.utc)).total_seconds()

def paths(request):
    """Only explicit campaign-owned output roots; no inferred broad drive grants."""
    roots = [Path(value).resolve() for value in request['owned_output_roots']]
    require(bool(roots), 'Owned output roots absent')
    for root in roots:
        drive = root.drive.upper()
        require(drive in {'C:', 'D:', 'E:'}, 'Only C/D/E output volumes admitted')
        require(root != Path(root.anchor), 'Drive root cannot be owned')
        # The campaign name is an ownership namespace even for external caches.
        require(root.is_relative_to(ROOT) or ROOT.name in root.parts, 'Output root missing campaign ownership namespace')
    outputs = [Path(value).resolve() for value in request['output_paths']]
    require(bool(outputs), 'Explicit output paths absent')
    require(all(any(path.is_relative_to(root) for root in roots) for path in outputs), 'Foreign output path')
    return roots, outputs

def probe_space(request, *, write_probe=False, estimated=True):
    roots, _ = paths(request)
    required = request['required_output_bytes']
    require(isinstance(required, dict), 'Per-volume output estimates required')
    drives = {root.drive.upper() for root in roots} | {ROOT.drive.upper()}
    require(set(required) == drives, 'Exact per-volume output estimates required')
    for drive in drives:
        estimate = required[drive]
        require(type(estimate) is int and estimate >= 0, 'Invalid output estimate')
        require(shutil.disk_usage(drive + '/').free >= 15 * GIB + (estimate if estimated else 0), 'Disk reserve or estimated output space unavailable: ' + drive)
    if write_probe:
        for index, root in enumerate(roots):
            root.mkdir(parents=True, exist_ok=True)
            target = root / ('.space-probe-' + request['operation'] + '-' + str(index))
            with target.open('xb') as stream:
                stream.write(b'owned C3 campaign write probe\n')
                stream.flush()
                os.fsync(stream.fileno())
            target.unlink()

def validate(request, request_path, expected, *, verify_pins=True):
    request_path = Path(request_path).resolve(strict=True)
    require(request_path.parent == ROOT and re.fullmatch(r'request-[A-Za-z0-9_-]+\.json', request_path.name), 'Campaign-root request required')
    require(sha(request_path) == expected, 'Request digest mismatch')
    require(re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,79}', request['operation']), 'Invalid operation name')
    campaign_path = ROOT / 'campaign.json'
    campaign = read(campaign_path)
    require(campaign['contract'] == 'kion-C3-reward-RL-execution-v1' and campaign['state'] == 'running', 'Campaign not admitted')
    limits = campaign['limits']
    require(all(limits[key] == value for key, value in {'supervised_updates': 0, 'new_RM_fits': 0, 'final_test_reads': 0, 'GPU_jobs': 1, 'heavy_native_jobs': 1, 'external_downloads': False, 'paid_operations': False}.items()), 'Campaign exclusions changed')
    require(0 < limits['RL_updates_max'] <= 1000 and campaign['round_limit'] <= 3, 'Campaign budget changed')
    admission = campaign['admitted_requests'][request['operation']]
    require(Path(admission['request_path']).resolve() == request_path and admission['sha256'] == expected, 'Request lacks root campaign admission')
    require(request['contract'] == 'kion-c-rl-native-v1' and request['stage'] in STAGES, 'Unknown request contract/stage')
    require(request['supervised_updates'] == 0 and request['new_rm_fits'] == 0 and request['final_test_reads'] == 0, 'Excluded work')
    require(type(request['rl_updates']) is int and 0 <= request['rl_updates'] <= limits['RL_updates_max'], 'RL update budget exceeded')
    require(request['cpu_threads'] == 4 and request['reserve_gib'] == 2 and request['cap_gib'] in (4, 8, 12), 'Native resource contract')
    upper = 7200 if request['stage'] in {'prepare_cache', 'reward_probe'} else 3600
    require(type(request['maximum_seconds']) is int and 1 <= request['maximum_seconds'] <= upper, 'Stage time budget exceeded')
    require(remaining(request) > 0, 'Request deadline reached')
    if campaign.get('deadline_utc'):
        require(deadline(request) <= datetime.fromisoformat(campaign['deadline_utc']), 'Request exceeds campaign deadline')
    require(not (ROOT / 'STOP').exists(), 'Campaign stop requested')
    module_path = Path(request['module_path']).resolve(strict=True)
    require(module_path.is_relative_to(ROOT) and module_path.relative_to(ROOT).parts[0] in SOURCE_DIRS and module_path.suffix == '.py', 'Foreign stage module')
    require(request['function'] in FUNCTIONS, 'Unsupported entry function')
    paths(request)
    for field in ('source_hashes', 'input_hashes'):
        require(isinstance(request[field], dict) and bool(request[field]), 'Pins absent: ' + field)
        for path, digest in request[field].items():
            require(Path(path).is_absolute() and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid pin')
    for path in [module_path, CONTROL/'common.py', CONTROL/'launch.py', CONTROL/'supervise.py', CONTROL/'worker.py', PAS/'scripts/portability.py', RUNTIME]:
        require(path.as_posix() in request['source_hashes'], 'Required source unpinned: ' + path.as_posix())
    if verify_pins:
        verify_request_pins(request)
    probe_space(request)
    return request

def verify_request_pins(request):
    """Sources always rehashed. Large immutable inputs may use stat-bound receipts."""
    receipts = request.get('input_digest_receipts', {})
    for field in ('source_hashes', 'input_hashes'):
        for path, digest in request[field].items():
            cached = receipts.get(path) if field == 'input_hashes' else None
            if cached is not None:
                require(cached['sha256'] == digest and cached['stat'] == stat_key(path), 'Input digest receipt stale: ' + path)
            else:
                require(sha(path) == digest, 'Pinned content changed: ' + path)
