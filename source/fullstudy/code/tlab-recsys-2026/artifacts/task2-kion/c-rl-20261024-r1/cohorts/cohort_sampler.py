"""C3 cohort selection. Import/tests are stdlib-only; root owns real execution."""
import argparse
from collections import Counter
from datetime import date
import hashlib
import heapq
import json
from pathlib import Path
import random
import sqlite3
import zipfile

HERE = Path(__file__).resolve().parent
ART = HERE.parent.parent
DATA = ART.parent.parent / 'data/processed/kion-full-context64'
RM = ART.parent.parent / 'data/processed/kion-reward-local-20261020-r1'
EVENT = ART / 'all-agreed-20261020/completion-diagnostics/event-store-r1/full-r1'
PREVIOUS = ART / 'three-head-262k-20261024-r1'
CONTRACT = 'kion-C3-eligible-cohorts-v1'
TRAIN_START, DEV_START, DEV_STOP = (date(2021, 8, d).toordinal() for d in (2, 9, 16))
SALT = 'kion-C3-RL-context-20261024-r1'
ROLE_SALT = 'kion-rm-select-calibration-v1-20261020'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def read(path, cap=1 << 20):
    path = Path(path)
    require(path.stat().st_size <= cap, 'JSON byte cap: ' + str(path))
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def context_hash(user, origin, length, x):
    values = [int(v) for v in x]
    require(len(values) == 64 and 1 <= int(length) <= 64 and values[0] == 1
        and all(v >= 2 for v in values[1:int(length)])
        and all(v == 0 for v in values[int(length):]), 'Invalid BOS/H63 context')
    return canonical([int(user), int(origin), int(length), values])


def select_train(users, origins, excluded_users, maximum=512):
    """One state per (user,day); no target, completion, reward or support input."""
    require(type(maximum) is int and maximum > 0 and len(users) == len(origins), 'Invalid cohort input')
    seen, heap, counts = set(), [], Counter()
    for row, (uid, day) in enumerate(zip(users, origins)):
        uid, day = int(uid), int(day)
        if not TRAIN_START <= day < DEV_START:
            counts['outside_train_window'] += 1
            continue
        counts['rows_in_window'] += 1
        if uid in excluded_users:
            counts['excluded_RM_user_rows'] += 1
            continue
        key = (uid, day)
        if key in seen:
            counts['duplicate_state_rows'] += 1
            continue
        seen.add(key)
        counts['eligible_unique_states'] += 1
        rank = int.from_bytes(hashlib.sha256(f'{SALT}:train:{uid}:{day}'.encode('ascii')).digest(), 'big')
        entry = (-rank, -row, row)
        if len(heap) < maximum:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    rows = sorted(v[2] for v in heap)
    counts['selected'] = len(rows)
    require(len(rows) == maximum, f'Insufficient eligible train states: {len(rows)}/{maximum}')
    return rows, dict(counts)


def select_screen(dev_rows, maximum=256):
    require(len(dev_rows) == 4096 and len(set(dev_rows)) == 4096, 'Exact frozen4096 dev required')
    require(0 < maximum <= 4096, 'Invalid screen cap')
    positions = sorted(range(4096), key=lambda p: (hashlib.sha256(
        f'{SALT}:screen:{int(dev_rows[p])}'.encode('ascii')).digest(), p))[:maximum]
    positions.sort()
    return [int(dev_rows[p]) for p in positions], positions


def request_document():
    """Read only small metadata/source receipts; no arrays or database access."""
    old = read(PREVIOUS / 'request-evaluate-r1.json')
    event = read(EVENT / 'manifest.json')
    original = read(ART / 'sft-e5-20261023-r1/admission/request-r1.json')
    hook = original['prepared_profile_hook'] if 'prepared_profile_hook' in original else None
    if hook is None:
        # The original request calls this seam query_profile_hook.
        hook = next(v for v in original.values() if isinstance(v, dict)
            and isinstance(v.get('options'), dict) and 'store_path' in v['options'])
    options = hook['options']
    pins = {str(DATA / name).replace('\\', '/'): old['input_hashes'][(DATA / name).as_posix()]
        for name in ('metadata.json', 'train.npz', 'val.npz')}
    for category in ('arrays', 'query_arrays'):
        groups = [event[category]] if category == 'arrays' else list(event[category].values())
        for group in groups:
            for key, declaration in group.items():
                if key in ('user_id', 'date', 'token', 'origin', 'length'):
                    pins[(EVENT / declaration['path']).as_posix()] = declaration['sha256']
    for path in (EVENT / 'manifest.json', PREVIOUS / 'validation-selection.json',
                 PREVIOUS / 'evaluation/evaluation.json', RM / 'coordinator-audit.json',
                 ART / 'local-pack-20261020/bundle-rm-r1/scripts/kion_reward_data.py',
                 ART / 'local-pack-20261020/bundle-rm-r1/scripts/kion_reward_boosting.py'):
        pins[path.as_posix()] = sha(path)
    pins[(RM / 'manifest.json').as_posix()] = '42d7532a5daea8af36198e0a4ba6ba3b134cd66e93dcfd75961913a14624144c'
    pins[options['coverage_receipt']['path']] = options['coverage_receipt']['sha256']
    return dict(contract=CONTRACT, source_pins=pins, query_store=options['store_path'],
        query_store_receipt=options['coverage_receipt'], train_states=512, screen_states=256,
        dataset_rows={'train': 3477437, 'val': 158760}, sampler_sha256=sha(__file__),
        limits={'max_prepared_bytes': 3 << 30, 'max_RM_manifest_bytes': 32 << 20},
        actor_outcomes_read=False, final_test_reads=0)


def checked_npy(np, path, pins, shape=None):
    path = Path(path)
    require(pins.get(path.as_posix()) == sha(path), 'Array pin changed: ' + str(path))
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(array.dtype.kind in 'iu' and (shape is None or array.shape == shape), 'Integer shape mismatch: ' + str(path))
    return array


def selected_npz_rows(np, path, rows, population):
    """Stream compressed x only, with <=1MiB scratch; never load reward/y."""
    result, cursor = {}, 0
    with zipfile.ZipFile(path) as archive, archive.open('x.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        require(shape == (population, 64) and not fortran and dtype.kind in 'iu'
            and dtype.itemsize in (4, 8), 'Unexpected prepared x layout')
        stride = 64 * dtype.itemsize
        for row in rows:
            count = (row - cursor) * stride
            while count:
                block = stream.read(min(count, 1 << 20))
                require(block, 'Truncated prepared context')
                count -= len(block)
            data = stream.read(stride)
            require(len(data) == stride, 'Truncated selected context')
            result[row] = np.frombuffer(data, dtype=dtype).copy()
            cursor = row + 1
    return result


def probe(request, check=None):
    """Outer native-controller seam; the inner cohort request stays separate."""
    if check is not None:
        check(stage='cohort-probe-before')
    path = Path(request['cohort_request'])
    require(sha(path) == request['cohort_request_sha256'], 'Inner cohort request pin changed')
    result = run(read(path), Path(request['cohort_output']))
    if check is not None:
        check(stage='cohort-probe-after')
    return result


def run(request, output):
    """Explicit root-launched native stage; all real reads live below this seam."""
    require(request.get('contract') == CONTRACT and request['sampler_sha256'] == sha(__file__), 'Sampler request/source differs')
    require(request['train_states'] == 512 and request['screen_states'] == 256, 'Frozen cohort caps differ')
    require(request.get('final_test_reads') == 0 and request.get('actor_outcomes_read') is False, 'Forbidden data route')
    pins = request['source_pins'].copy()
    for split in ('train', 'val'):
        require((DATA / f'{split}.npz').stat().st_size <= request['limits']['max_prepared_bytes'],
            'Prepared archive exceeds admitted byte cap')
    for path, digest in pins.items():
        require(sha(path) == digest, 'Input/source changed: ' + path)
    import numpy as np
    metadata = read(DATA / 'metadata.json')
    event = read(EVENT / 'manifest.json')
    require(event['state'] == 'completed' and event['metadata_sha256'] == pins[(DATA / 'metadata.json').as_posix()], 'Event lineage mismatch')
    rm = read(RM / 'manifest.json', request['limits']['max_RM_manifest_bytes'])
    mapping = metadata['item_id_to_token']
    require(metadata['special_tokens'] == {'PAD': 0, 'BOS': 1, 'UNK': 2}
        and set(mapping.values()) == set(range(3, metadata['vocab_size']))
        and len(mapping) == metadata['vocab_size'] - 3, 'Nonbijective token mapping')
    require(rm['ready'] is True and rm['catalog']['mapping_sha256'] == canonical(mapping)
        and rm['partition']['salt'] == ROLE_SALT, 'RM role/mapping provenance mismatch')
    role_users, role_keys, role_counts = set(), set(), {}
    for role in ('select', 'calibration'):
        arrays = {}
        for name in ('user_id', 'target_date', 'history_cutoff', 'x', 'lengths'):
            path = RM / f'{role}.{name}.npy'
            digest = {**rm['files'], **rm['sidecars']}[path.name]
            pins[path.as_posix()] = digest
            arrays[name] = checked_npy(np, path, pins, (20000, 64) if name == 'x' else (20000,))
        for i in range(20000):
            uid, day = int(arrays['user_id'][i]), int(arrays['target_date'][i])
            bucket = int.from_bytes(hashlib.sha256(f'{ROLE_SALT}:{uid}'.encode()).digest()[:8], 'big') < 2**63
            require(bucket == (role == 'select') and TRAIN_START <= day < DEV_START
                and int(arrays['history_cutoff'][i]) == TRAIN_START, 'Saved RM role/time assignment mismatch')
            role_users.add(uid)
            role_keys.add(context_hash(uid, TRAIN_START, arrays['lengths'][i], arrays['x'][i]))
        role_counts[role] = {'rows': 20000, 'users': len(set(map(int, arrays['user_id'])))}
    queries = {}
    for split, population in request['dataset_rows'].items():
        queries[split] = {key: checked_npy(np, EVENT / item['path'], pins, (population,))
            for key, item in event['query_arrays'][split].items()}
    train, counts = select_train(queries['train']['user_id'], queries['train']['origin'], role_users)
    dev = read(PREVIOUS / 'validation-selection.json')
    expected_dev = sorted(random.Random(20261024).sample(range(158760), 4096))
    require(dev['row_indices'] == expected_dev, 'Original dev4096 selection differs')
    evaluation = read(PREVIOUS / 'evaluation/evaluation.json')
    require(evaluation['state'] == 'completed' and evaluation['rows'] == 4096
        and evaluation['candidate_width'] == 1024 and evaluation['no_target_injection'] is True
        and evaluation['no_seen_filter'] is True
        and evaluation['selection']['sha256'] == sha(PREVIOUS / 'validation-selection.json'),
        'Original natural4096 evaluation receipt differs')
    screen, positions = select_screen(expected_dev)
    user_events = checked_npy(np, EVENT / 'user_id.npy', pins, (event['events_count'],))
    dates = checked_npy(np, EVENT / 'date.npy', pins, (event['events_count'],))
    tokens = checked_npy(np, EVENT / 'token.npy', pins, (event['events_count'],))
    receipt = read(request['query_store_receipt']['path'])
    require(receipt['state'] == 'completed' and receipt['split_rows'] == request['dataset_rows']
        and receipt['store_path'] == request['query_store'], 'Original QueryStore coverage mismatch')
    # Do not rehash the 39GB immutable store: original provider hook performs its
    # full integrity gate at cache admission. This stage checks exact key presence.
    connection = sqlite3.connect(Path(request['query_store']).as_uri() + '?mode=ro&immutable=1', uri=True)
    states = {}
    try:
        connection.execute('PRAGMA query_only=ON')
        for role, split, rows in (('train', 'train', train), ('screen', 'val', screen)):
            prepared = selected_npz_rows(np, DATA / f'{split}.npz', rows, request['dataset_rows'][split])
            bound = []
            for row in rows:
                uid, origin, length = (int(queries[split][k][row]) for k in ('user_id', 'origin', 'length'))
                require(TRAIN_START <= origin < DEV_START if role == 'train' else origin == DEV_START, 'Actor origin outside declared window')
                x = prepared[row]
                digest = context_hash(uid, origin, length, x)
                lo, hi = (int(np.searchsorted(user_events, uid, side=side)) for side in ('left', 'right'))
                end = lo + int(np.searchsorted(dates[lo:hi], origin, side='left'))
                start = max(lo, end - 63)
                require(length == end - start + 1 and np.array_equal(x[1:length], tokens[start:end])
                    and (end == start or int(dates[end-1]) < origin), 'Original observed strict-past history mismatch')
                if role == 'train':
                    require(uid not in role_users and digest not in role_keys, 'RM heldout user/context leakage')
                key = (uid, origin, length, x.tobytes())
                require(connection.execute('SELECT 1 FROM profiles WHERE uid=? AND origin=? AND query_len=? AND x=?', key).fetchone(), 'Original QueryStore key missing')
                bound.append(dict(row_index=row, user_id=uid, origin=origin, length=length, context_sha256=digest))
            states[role] = bound
    finally:
        connection.close()
    result = dict(contract=CONTRACT, state='completed', exclusions_verified=True,
        cohorts={'train': dict(split='train', row_indices=train), 'screen': dict(split='val', row_indices=screen, dev_positions=positions)},
        states=states, source_pins=pins, role_counts=role_counts, train_counts=counts,
        excluded_unique_users=len(role_users), excluded_unique_contexts=len(role_keys),
        mapping_sha256=canonical(mapping), query_store=request['query_store'],
        query_store_check='exact selected keys present; full database hash deferred to pinned provider admission',
        full_dev_rows=4096, full_dev_selection_sha256=sha(PREVIOUS / 'validation-selection.json'),
        unknown_and_offpool_retained=True, reward_screen_selection='fixed salted row hash before RM score/support/outcome access',
        sampling='outcome-independent bottom SHA256 unique(user,origin); no observed target/reward arrays accessed',
        caveats=['Offline learned proxy: calibrator used Aug2–8 outcomes and is not contemporaneously known for train replay.',
            'E2 and catalogue are retrospective; event dates are last-watch aggregate dates, not true exposure/session logs.'],
        final_test_reads=0, actor_outcomes_read=False, request_sha256=canonical(request))
    result['selection_sha256'] = canonical(result['cohorts'])
    write_new(output, result)
    return {'state': 'completed', 'output': Path(output).resolve().as_posix(), 'sha256': sha(output), 'train': len(train), 'screen': len(screen)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('make-request', 'run'))
    parser.add_argument('--request', type=Path, default=HERE / 'request-r1.json')
    parser.add_argument('--output', type=Path, default=HERE / 'selection-r1.json')
    args = parser.parse_args()
    if args.mode == 'make-request':
        write_new(args.request, request_document())
        print(args.request.as_posix())
    else:
        print(json.dumps(run(read(args.request), args.output)))
