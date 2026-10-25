"""Finite metadata/stat binding plan for the preserved nested 262k bank.

This is not an inference launcher. Execution remains closed until provider I/O,
graph closure and native resource supervision have portable reviewed bindings.
"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import stat
import sys

from input_inventory import checked_path, relative_parts

ROOT = Path(__file__).resolve().parents[1]
ART = 'D:/tlab-recsys-2026/artifacts/task2-kion'
SOURCE = 'fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/three-head-262k-20261024-r1/bank.py'
SOURCE_SHA = 'cb171615f7ed3b63f4ea6e6bf7c61b109c4fe7b4e9108b16c0d3f9a1cb227c36'
REQUEST_SHA = 'd5d1e624b670a69c910f002a4d8b6fac2cd018e6a9c5b63a24f2a414aa7b9f6f'
REQUEST = ROOT / 'reproduction/profiles/head-bank-request-r1.json'
ORIGINAL = ART + '/sft-e5-20261023-r1/admission/request-r1.json'
ORIGINAL_SHA = 'a8ffeb9cb94146c9ee29e24ef814ff5951d0a81184a5daf0c669ffb1fb9fdb4f'
ARRAYS = {'encoded', 'candidate_ids', 'base_logits', 'raw_features', 'source_values',
          'row_indices', 'user_ids', 'target_ids', 'dates', 'item_table'}
SEAMS = [
    'bank.open_context: ORIGINAL, DATA, RUNTIME, EVENT_MANIFEST; teacher logical-path comparison and interpreter identity',
    'bank.load: resolve I/O separately from original absolute source_hashes keys',
    'provider and store/performance/dynamic hooks: signed identities, imports, nested file access and make_config equality',
    'bank.open_reuse/run: REUSE_DIR, BANK_OUTPUT, item_table copy and fresh output ownership',
    'LiveCandidates: graph root and descriptor signatures, forbid _build; replace recursive disk census with bounded inventory',
    'worker/supervise: new owned deadline/STOP, Windows Job 12 GiB cap, CPU affinity 0x55, host reserve 2 GiB, disk reserve 15 GiB',
]


def pin(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('Expected lowercase SHA256')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate metadata key: ' + key)
        result[key] = value
    return result


def metadata(path, expected=None, maximum=8 << 20):
    path = checked_path(Path(path).absolute())
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError('Metadata must be a bounded regular file')
    with path.open('rb') as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError('Metadata exceeds bound')
    if expected and hashlib.sha256(data).hexdigest() != pin(expected):
        raise ValueError('Metadata pin mismatch: ' + str(path))
    return json.loads(data, object_pairs_hook=unique_object)


def logical_path(value):
    if not isinstance(value, str) or not PureWindowsPath(value).is_absolute() or '\\' in value:
        raise ValueError('Logical identities must be absolute Windows POSIX paths')
    relative_parts(value[3:])
    return value


def bindings_map(bindings, allowed):
    if not isinstance(bindings, dict) or len(bindings) > 2048:
        raise ValueError('Bindings must be a finite exact-identity object')
    result, occupied = {}, set()
    for identity, local in bindings.items():
        logical_path(identity)
        if identity not in allowed:
            raise ValueError('Undeclared identity binding: ' + identity)
        if not isinstance(local, str):
            raise ValueError('Local binding must be an absolute path string')
        path = checked_path(Path(local))
        if path in occupied:
            raise ValueError('Two logical inputs alias one local path')
        occupied.add(path)
        result[identity] = path
    return result


def requirements(request):
    """Enumerate only named pin maps and explicit hook/directory fields."""
    result = {}
    def add(identity, sha=None, kind='file'):
        logical_path(identity)
        if sha is not None:
            pin(sha)
        old = result.get(identity)
        if old and (old['kind'] != kind or (sha is not None and old['sha256'] not in (None, sha))):
            raise ValueError('Conflicting dependency identity')
        if old and sha is None:
            sha = old['sha256']
        result[identity] = {'logical_path': identity, 'sha256': sha, 'kind': kind}
    for field in ('source_hashes', 'input_hashes'):
        values = request[field]
        if not isinstance(values, dict) or len(values) > 1024:
            raise ValueError('Pin map exceeds bound')
        for identity, sha in values.items():
            add(identity, sha)
    add(ORIGINAL, ORIGINAL_SHA)
    hook = request['provider_hook']
    for obj in (hook, hook['options']['coverage_receipt'], request['performance_hook'],
                request['performance_hook']['acceptance_receipt']):
        add(obj['path'], obj['sha256'])
    pins = hook['options']['store_source_pins']
    if len(pins) > 1024:
        raise ValueError('Store source pin map exceeds bound')
    for identity, sha in pins.items():
        add(identity, sha)
    add(hook['options']['store_path'])
    for field in ('graph_cache', 'reuse_bank_dir'):
        add(request[field], kind='directory')
    argv = request['train_argv']
    for field in ('--data', '--feature-dir'):
        add(argv[argv.index(field) + 1], kind='directory')
    add(argv[argv.index('--scope-decision') + 1])
    return result


def observation(entry, local):
    row = dict(entry, local_path=local.as_posix() if local else None,
               status='blocked_input:unbound', payload_verified=False)
    if local is not None:
        try:
            info = checked_path(local).stat()
            correct = stat.S_ISDIR(info.st_mode) if entry['kind'] == 'directory' else stat.S_ISREG(info.st_mode)
            row.update(status='stat-only' if correct else 'blocked_input:wrong-kind', bytes=info.st_size)
        except FileNotFoundError:
            row['status'] = 'blocked_input:missing'
    return row


def reuse_inventory(request, bindings):
    """Only a byte-pinned old manifest may declare the ten arrays and graph names."""
    identity = request['reuse_bank_dir'] + '/bank.json'
    local = bindings.get(identity)
    if local is None or not local.exists():
        return [], ['old32768 manifest unavailable: array and selected-origin graph closure unresolved']
    old = metadata(local, request['reuse_bank_manifest_sha256'])
    if (old.get('teacher') != request['teacher'] or old.get('train_rows') != 32768
            or old.get('state') != 'completed' or set(old.get('arrays', {})) != ARRAYS):
        raise ValueError('Old bank lineage/schema mismatch')
    rows = []
    root = bindings.get(request['reuse_bank_dir'])
    if root is not None and local != root / 'bank.json':
        raise ValueError('Old manifest and array root bindings disagree')
    for name, entry in old['arrays'].items():
        if entry['path'] != name + '.npy':
            raise ValueError('Old array filename differs')
        pin(entry['sha256'])
        row = observation({'logical_path': request['reuse_bank_dir'] + '/' + entry['path'],
                           'kind': 'file', 'sha256': entry['sha256'],
                           'expected_bytes': entry['bytes']}, root / entry['path'] if root else None)
        if row.get('bytes') is not None and row['bytes'] != entry['bytes']:
            row['status'] = 'blocked_input:size-mismatch'
        rows.append(row)
    counts = old['date_counts']
    if not isinstance(counts, dict) or not 1 <= len(counts) <= 4096:
        raise ValueError('Date census exceeds bound')
    counts = {int(day): n for day, n in counts.items()}
    if any(type(n) is not int or n <= 0 for n in counts.values()) or sum(counts.values()) < 262144:
        raise ValueError('Invalid original date census')
    total = sum(counts.values())
    quotas = {day: n * 262144 // total for day, n in counts.items()}
    order = sorted(counts, key=lambda day: (-(counts[day] * 262144 % total), day))
    for day in order[:262144 - sum(quotas.values())]:
        quotas[day] += 1
    key = pin(old['candidate_contract']['cache_key'])
    graph = bindings.get(request['graph_cache'])
    for day in sorted(d for d, n in quotas.items() if n):
        for suffix in ('.json', '-neighbors.npy', '-weights.npy', '-global_counts.npy', '-recent_counts.npy'):
            name = f'{key}-{day}{suffix}'
            rows.append(observation({'logical_path': request['graph_cache'] + '/' + name,
                                     'kind': 'file', 'sha256': None}, graph / name if graph else None))
    return rows, ['graph descriptor signatures/payload pins not verified; no graph reconstruction permitted']


def build_plan(bindings, output_dir, request_path=REQUEST):
    request = metadata(request_path, REQUEST_SHA)
    required = requirements(request)
    bound = bindings_map(bindings, required)
    output = checked_path(Path(output_dir).absolute())
    protected = [checked_path(ROOT / 'source'), checked_path(ROOT / 'reproduction'),
                 checked_path(Path(request_path).absolute()), *bound.values()]
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Bank output must be fresh and disjoint from all bound inputs/source')
    # Hash small preserved source only; do not import it or any numerical backend.
    source = checked_path(ROOT / 'source' / SOURCE)
    if source.stat().st_size > 8 << 20 or hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError('Preserved bank source pin mismatch')
    entries = [observation(entry, bound.get(identity)) for identity, entry in required.items()]
    extra, blockers = reuse_inventory(request, bound)
    entries.extend(extra)
    return {'mode': 'metadata-stat-plan', 'ready': False, 'preparation_executed': False,
            'inference_executed': False, 'training_executed': False,
            'historical_request_sha256': REQUEST_SHA, 'historical_request': request,
            'original_callable': SOURCE + ':run(request, check)', 'source_sha256': SOURCE_SHA,
            'proposed_output': output.as_posix(), 'requirements': entries,
            'selection': {'rows': 262144, 'reused_rows': 32768, 'new_rows': 229376,
                          'seed': 20261024, 'slots': 32, 'positive_column': 0,
                          'selection': 'nested largest-remainder per-date residual draw; sorted(date,row_index)',
                          'negative_rng': 'SHA256(seed:original_train_row)',
                          'same_user_same_origin_positives_excluded': True,
                          'copy_old_rows_and_item_table_verbatim': True},
            'blocked': blockers + ['portable provider and resource-guard seams unresolved; execution disabled'],
            'unresolved_execution_seams': SEAMS,
            'scope': 'Declared dependencies only; no transitive provider closure, payload reads, hashes, imports or inference. Historical expired deadline is provenance, not a new authorization.'}


def execute(plan, *, execute_preparation=False):
    if execute_preparation is not True:
        raise ValueError('Explicit preparation execution required')
    raise ValueError('Preparation unavailable: portable provider/guard seams unresolved; no backend loaded')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings', type=Path, help='Exact logical identity -> absolute local path JSON; default none')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--request', type=Path, default=REQUEST)
    parser.add_argument('--execute-preparation', action='store_true', help='Currently fails closed; no numerical backend')
    args = parser.parse_args(argv)
    try:
        plan = build_plan(metadata(args.bindings) if args.bindings else {}, args.output_dir, args.request)
        if args.execute_preparation:
            execute(plan, execute_preparation=True)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0  # Plan creation success; ready stays false explicitly.
    except (OSError, ValueError, KeyError, TypeError) as error:
        print('head bank plan: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
