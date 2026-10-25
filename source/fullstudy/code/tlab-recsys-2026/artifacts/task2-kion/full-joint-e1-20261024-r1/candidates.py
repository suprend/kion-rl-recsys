"""Live GPT + strict-origin CPU candidates. Importing uses only stdlib.

NumPy, the frozen Q helpers, read-only event arrays, and the actual provider are
injected by the guarded native worker. This module never imports a model.
"""
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import uuid

CONTRACT = 'kion-live-gpt-strict-origin-mixed1024-v1'
SOURCE_FEATURE_NAMES = tuple(
    [f'owner_{s}' for s in ('gpt', 'pop7d', 'knn')]
    + [f'nominated_current2048_{s}' for s in ('gpt', 'pop7d', 'knn')]
    + [f'capped_rank_over2049_{s}' for s in ('gpt', 'pop7d', 'knn')]
    + [f'missing_rank_{s}' for s in ('gpt', 'pop7d', 'knn')]
    + ['current_detached_gpt_logit', 'log1p_strict7d_count'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compose(sources, vocab):
    """Natural pool; each quota counts *new* unique items, in source order."""
    require(len(sources) == 3, 'Three candidate sources required')
    ids, owners, seen = [], [], set()
    for owner, (source, quota) in enumerate(zip(sources, (512, 256, 256))):
        added = 0
        for value in source:
            item = int(value)
            require(3 <= item < vocab, 'Reserved/unknown source candidate')
            if item in seen:
                continue
            ids.append(item)
            owners.append(owner)
            seen.add(item)
            added += 1
            if added == quota:
                break
        require(added == quota, 'Source exhausted before new-unique quota')
    require(len(ids) == len(set(ids)) == 1024, 'Natural pool width differs')
    return ids, owners


def draw(pool, target, copositives, seed, row):
    """All rows retained; target first even when outside the natural pool."""
    require(len(pool) == len(set(pool)) == 1024, 'Natural distinct pool required')
    blocked = set(map(int, copositives)) | {int(target)}
    eligible = [int(item) for item in pool if int(item) not in blocked]
    require(len(eligible) >= 31, 'Insufficient distinct natural negatives')
    key = hashlib.sha256(f'{seed}:{int(row)}'.encode('ascii')).digest()
    rng = random.Random(int.from_bytes(key, 'little'))
    return [int(target), *rng.sample(eligible, 31)]


def metadata(ids, owners, sources, logits, recent_counts):
    require(len(ids) == len(owners) == len(logits) == len(recent_counts), 'Metadata alignment differs')
    require(len(sources) == 3 and all(len(s) == 2048 for s in sources), 'Source2048 required')
    ranks = [{int(item): rank for rank, item in enumerate(source, 1)} for source in sources]
    output = []
    for item, owner, logit, count in zip(ids, owners, logits, recent_counts):
        rank = [table.get(int(item), 2049) for table in ranks]
        present = [value != 2049 for value in rank]
        require(math.isfinite(float(logit)) and int(count) >= 0, 'Nonfinite live score/negative count')
        require(int(owner) in (-1, 0, 1, 2), 'Invalid owner')
        output.append([float(owner == s) for s in range(3)] + list(map(float, present))
                      + [value / 2049 for value in rank] + [float(not p) for p in present]
                      + [float(logit), math.log1p(int(count))])
    return output


class LiveCandidates:
    """Bounded batch candidates, with immutable per-origin weight-free tables.

    Train batches must have one origin; their ordering may be globally shuffled.
    Validation always uses the fixed validation cutoff, never target dates.
    ``raw_ids`` is the canonical V-3 array (known tokens map at token-3).
    ``context_signature`` must pin prepared metadata/events and frozen Q source.
    """
    source_feature_names = SOURCE_FEATURE_NAMES

    def __init__(self, provider, np, q, events, raw_ids, cache_root,
                 context_signature, check=None, seed=20261024,
                 validation_origin=None, max_disk_bytes=5 * 1024**3,
                 min_disk_free_bytes=15 * 1024**3, cache_origins=2):
        self.provider, self.np, self.q, self.events = provider, np, q, events
        self.raw_ids = np.asarray(raw_ids, dtype=np.int64)
        self.vocab = len(self.raw_ids) + 3
        require(self.vocab >= 2051 and len(np.unique(self.raw_ids)) == len(self.raw_ids), 'Invalid catalog')
        require(set(events) >= {'user_id', 'token', 'date'}, 'Missing prefix event arrays')
        require(len(events['user_id']) == len(events['token']) == len(events['date']), 'Event alignment differs')
        require(bool(context_signature), 'Immutable context/source pins required')
        require(1 <= cache_origins <= 2, 'At most two resident origin tables allowed')
        require(0 < max_disk_bytes <= 5 * 1024**3, 'Disk cache cap exceeds 5 GiB')
        require(min_disk_free_bytes >= 15 * 1024**3, 'Required disk reserve is 15 GiB')
        self.check = check or (lambda **fields: None)
        self.seed = int(seed)
        self.validation_origin = int(q.CUTOFF if validation_origin is None else validation_origin)
        self.max_disk_bytes, self.min_disk_free_bytes = max_disk_bytes, min_disk_free_bytes
        self.cache_origins, self.cache, self.verified = cache_origins, OrderedDict(), {}
        self.root = Path(cache_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature = dict(contract=CONTRACT, context=context_signature,
                              adapter_sha256=digest(__file__), vocab=self.vocab,
                              raw_ids_sha256=hashlib.sha256(self.raw_ids.tobytes()).hexdigest(),
                              neighbors=200, source_width=2048, history_last=20)
        self.cache_key = hashlib.sha256(json.dumps(self.signature, sort_keys=True,
                                                   separators=(',', ':')).encode('utf-8')).hexdigest()
        self.origins = np.asarray(provider.origin_keys('train'), dtype=np.int64)
        require(len(self.origins) == len(provider.splits['train']['y']), 'Train origin alignment differs')
        require(bool(np.all(self.origins < self.validation_origin)), 'Training origin crosses validation')
        # One compact sorted row index, never per-user candidate/feature banks.
        self.train_order = np.lexsort((provider.splits['train']['user_id'], self.origins))
        ordered = self.origins[self.train_order]
        self.sorted_users = provider.splits['train']['user_id'][self.train_order]
        dates, starts = np.unique(ordered, return_index=True)
        ends = list(starts[1:]) + [len(ordered)]
        self.origin_bounds = {int(day): (int(a), int(b)) for day, a, b in zip(dates, starts, ends)}
        self.stats = dict(graphs_built=0, tables_loaded=0, train_rows_sampled=0, validation_rows=0)
        self._disk_check()

    def raw_item_ids(self, ids):
        values = self.np.asarray(ids)
        require(bool(self.np.all((values >= 3) & (values < self.vocab))), 'Unknown raw-ID query')
        return self.raw_ids[values - 3]

    def contract(self):
        return dict(contract=CONTRACT, source_feature_names=list(self.source_feature_names),
                    natural_quotas=[512, 256, 256], new_unique_quotas=True,
                    current_detached_gpt=True, train_positive_column=0, train_negatives=31,
                    all_training_rows_retained=True, same_user_same_origin_positives_excluded=True,
                    validation_target_injection=False, unknown_validation_targets_are_misses=True,
                    cutoff='event_date < query_origin; fixed validation origin',
                    validation_origin=self.validation_origin, train_origins=len(self.origin_bounds),
                    seed=self.seed, cache_key=self.cache_key, cache_root=self.root.as_posix(),
                    cache_ram_origins=self.cache_origins, max_cache_disk_bytes=self.max_disk_bytes,
                    minimum_disk_free_bytes=self.min_disk_free_bytes)

    def _disk_check(self, extra=0):
        used = sum(path.stat().st_size for path in self.root.rglob('*') if path.is_file())
        require(used + extra <= self.max_disk_bytes, 'Candidate cache exceeds 5 GiB budget')
        require(shutil.disk_usage(self.root).free - extra >= self.min_disk_free_bytes,
                'Candidate cache would violate 15 GiB free-space reserve')

    def _paths(self, origin):
        prefix = self.root / f'{self.cache_key}-{int(origin)}'
        names = ('neighbors', 'weights', 'global_counts', 'recent_counts')
        return prefix.with_suffix('.json'), {name: Path(str(prefix) + '-' + name + '.npy') for name in names}

    def _build(self, origin, descriptor, paths):
        np, q = self.np, self.q
        expected = (self.vocab - 3) * (200 * 8 + 16) + 4096
        self._disk_check(expected)
        require(not descriptor.exists() and not any(p.exists() for p in paths.values()),
                'Partial/existing candidate cache requires explicit repair; refusing overwrite')
        self.check(stage='candidate-origin-build-start', origin=origin)
        global_counts = q.prefix_counts(self.events['token'], self.events['date'], self.vocab, cutoff=origin)
        recent_counts = q.prefix_counts(self.events['token'], self.events['date'], self.vocab, window=7, cutoff=origin)
        matrix, census = q.binary_fit(self.events['user_id'], self.events['token'], self.events['date'], self.vocab, cutoff=origin)
        seeds = np.arange(self.vocab - 3, dtype=np.int64)
        try:
            neighbors, weights = q.cosine_neighbors(matrix, seeds, self.raw_ids, self.check, limit=200)
        finally:
            del matrix, seeds
            gc.collect()
        arrays = dict(neighbors=neighbors, weights=weights,
                      global_counts=np.asarray(global_counts, dtype=np.int64),
                      recent_counts=np.asarray(recent_counts, dtype=np.int64))
        entries = {}
        for name, values in arrays.items():
            self.check(stage='candidate-cache-save', origin=origin, array=name)
            temporary = paths[name].with_name(paths[name].name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, paths[name])
            finally:
                if temporary.exists():
                    temporary.unlink()
            entries[name] = dict(path=paths[name].name, shape=list(values.shape),
                                 dtype=str(values.dtype), bytes=paths[name].stat().st_size,
                                 sha256=digest(paths[name]))
        document = dict(state='completed', signature=self.signature, cache_key=self.cache_key,
                        origin=int(origin), cutoff=int(origin), arrays=entries, census=census)
        temporary = descriptor.with_name(descriptor.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            temporary.write_text(json.dumps(document, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
            os.replace(temporary, descriptor)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.stats['graphs_built'] += 1
        self.check(stage='candidate-origin-build-completed', origin=origin, census=census)
        self._disk_check()

    @staticmethod
    def _close_tables(tables):
        for values in tables.values():
            mapping = getattr(values, '_mmap', None)
            if mapping is not None:
                mapping.close()

    def _tables(self, origin):
        origin = int(origin)
        require(origin in self.origin_bounds or origin == self.validation_origin, 'Unapproved candidate origin')
        if origin in self.cache:
            self.cache.move_to_end(origin)
            return self.cache[origin]
        while len(self.cache) >= self.cache_origins:
            _, evicted = self.cache.popitem(last=False)
            self._close_tables(evicted)
        descriptor, paths = self._paths(origin)
        if not descriptor.exists():
            self._build(origin, descriptor, paths)
        require(descriptor.stat().st_size <= 8 * 1024**2, 'Oversized cache descriptor')
        document = json.loads(descriptor.read_text(encoding='utf-8'))
        require(document.get('state') == 'completed' and document.get('signature') == self.signature
                and document.get('cache_key') == self.cache_key
                and document.get('origin') == document.get('cutoff') == origin,
                'Candidate cache context/origin mismatch')
        expected = dict(neighbors=((self.vocab - 3, 200), 'int32'),
                        weights=((self.vocab - 3, 200), 'float32'),
                        global_counts=((self.vocab - 3,), 'int64'),
                        recent_counts=((self.vocab - 3,), 'int64'))
        tables = {}
        try:
            for name, path in paths.items():
                entry = document['arrays'][name]
                shape, dtype = expected[name]
                require(entry['path'] == path.name and entry['shape'] == list(shape)
                        and entry['dtype'] == dtype and path.stat().st_size == entry['bytes'],
                        'Candidate cache schema/size mismatch')
                stat = path.stat()
                identity = (stat.st_size, stat.st_mtime_ns, entry['sha256'])
                if self.verified.get(path) != identity:
                    self.check(stage='candidate-cache-verify', origin=origin, array=name)
                    require(digest(path) == entry['sha256'], 'Candidate cache digest mismatch')
                    self.verified[path] = identity
                values = self.np.load(path, mmap_mode='r', allow_pickle=False)
                tables[name] = values
                require(tuple(values.shape) == shape and str(values.dtype) == dtype, 'Candidate NPY header mismatch')
            self.cache[origin] = tables
        except BaseException:
            self._close_tables(tables)
            raise
        self.stats['tables_loaded'] += 1
        return tables

    def _origin(self, rows, split):
        require(split in ('train', 'val'), 'Only train/val candidate paths authorized')
        require(len(rows) > 0, 'Empty candidate batch')
        if split == 'val':
            return self.validation_origin
        values = self.origins[rows]
        require(bool(self.np.all(values == values[0])), 'Training candidate batch must have one origin')
        return int(values[0])

    def _copositives(self, origin, uid):
        # Sorted by date then user once: full-shuffle batches never scan a date.
        start, stop = self.origin_bounds[origin]
        users = self.sorted_users[start:stop]
        left = start + int(self.np.searchsorted(users, uid, side='left'))
        right = start + int(self.np.searchsorted(users, uid, side='right'))
        return self.provider.splits['train']['y'][self.train_order[left:right]]

    def _sources(self, rows, split, live_logits):
        np, q = self.np, self.q
        rows = np.asarray(rows, dtype=np.int64)
        require(rows.ndim == 1 and bool(np.all(rows >= 0)), 'Invalid row indices')
        require(bool(np.all(rows < len(self.provider.splits[split]['y']))), 'Row out of bounds')
        origin = self._origin(rows, split)
        tables = self._tables(origin)
        values = live_logits.detach().float().cpu().numpy() if hasattr(live_logits, 'detach') else live_logits
        logits = np.asarray(values, dtype=np.float32)
        require(logits.shape == (len(rows), self.vocab), 'Live full-catalog logits required')
        require(bool(np.isfinite(logits[:, 3:]).all()), 'Nonfinite current GPT logits')
        data = self.provider.splits[split]
        histories = q.histories(data['x'][rows], data['lengths'][rows], self.vocab, last=20)
        seeds = np.arange(self.vocab - 3, dtype=np.int64)
        knn, _ = q.infer_knn(histories, seeds, tables['neighbors'], tables['weights'],
                            self.raw_ids, tables['global_counts'], self.check, width=2048)
        pop = q.rank(tables['recent_counts'], self.raw_ids, width=2048) + 3
        sources = [(q.rank(logits[i, 3:], self.raw_ids, width=2048) + 3, pop, knn[i])
                   for i in range(len(rows))]
        return rows, origin, logits, tables, sources

    def pool(self, rows, split, live_logits):
        """Natural1024 only, with no target arguments or target injection."""
        rows, origin, logits, tables, sources = self._sources(rows, split, live_logits)
        ids, features = [], []
        for index, source in enumerate(sources):
            pool, owners = compose(source, self.vocab)
            ids.append(pool)
            features.append(metadata(pool, owners, source, logits[index, pool],
                                     tables['recent_counts'][self.np.asarray(pool) - 3]))
        if split == 'val':
            self.stats['validation_rows'] += len(rows)
        return self.np.asarray(ids, dtype=self.np.int64), self.np.asarray(features, dtype=self.np.float32)

    def sample(self, rows, split, live_logits, targets):
        require(split == 'train', 'Positive injection is train-only')
        rows, origin, logits, tables, sources = self._sources(rows, split, live_logits)
        target_values = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else targets
        targets = self.np.asarray(target_values, dtype=self.np.int64)
        require(targets.shape == rows.shape and bool(self.np.array_equal(targets, self.provider.splits['train']['y'][rows])),
                'Targets must align with all original train rows')
        require(bool(self.np.all((targets >= 3) & (targets < self.vocab))), 'Unknown training target cannot be injected')
        ids, features = [], []
        for index, (row, target, source) in enumerate(zip(rows, targets, sources)):
            pool, owner = compose(source, self.vocab)
            owner_map = dict(zip(pool, owner))
            uid = int(self.provider.splits['train']['user_id'][row])
            chosen = draw(pool, target, self._copositives(origin, uid), self.seed, row)
            ids.append(chosen)
            features.append(metadata(chosen, [owner_map.get(item, -1) for item in chosen], source,
                                     logits[index, chosen], tables['recent_counts'][self.np.asarray(chosen) - 3]))
        self.stats['train_rows_sampled'] += len(rows)
        return self.np.asarray(ids, dtype=self.np.int64), self.np.asarray(features, dtype=self.np.float32)

    def close(self):
        for tables in self.cache.values():
            self._close_tables(tables)
        self.cache.clear()

