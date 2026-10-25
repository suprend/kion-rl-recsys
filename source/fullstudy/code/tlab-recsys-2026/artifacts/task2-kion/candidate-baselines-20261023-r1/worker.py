"""Frozen CPU candidate controls; standalone, with no model or training imports."""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '4'
import argparse
from contextlib import nullcontext
from datetime import date
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import scipy
from scipy import sparse
import psutil
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = lambda limits: nullcontext()

BASE = Path('D:/tlab-recsys-2026')
ART = BASE / 'artifacts/task2-kion'
DATA = BASE / 'data/processed/kion-full-context64'
STORE = ART / 'all-agreed-20261020/completion-diagnostics/event-store-r1/full-r1'
E2 = ART / 'candidate-k-sweep-20261023-r1'
E2W = E2 / 'runs/run-r1/worker'
COHORT = ART / 'order-control-20261023-r1/runs/run-r1/worker/validation-cohort.json'
FIXED = {
    COHORT: 'cf61a382139671fcdb266477701a0e07c03bbc2a38fe3a5b5817aae14d82d9de',
    DATA / 'metadata.json': 'f02acaa9a5e5532e79b060279ba8a5027f268f202132e85ebab97cd4912fbd11',
    STORE / 'manifest.json': '382b410d564bb73eba268b5e2e49d092c25d2af4d1df677020ce46f546dc0cbf',
    E2 / 'root-finalization-r1.json': '7828cd918beddfacaf30305abef4bd96e8bde75bc7ecd29fca9e6918e74dd4df',
}
CUTOFF = date(2021, 8, 9).toordinal()
KS = (128, 256, 512)
IDENTITY = ('x', 'lengths', 'y', 'user_id', 'target_item_id', 'target_date',
            'row_index', 'target_token', 'known_target')
E2_IDENTITY = IDENTITY[3:]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def required_input_paths():
    """Read only tiny manifests; never open arrays, model checkpoints or test data."""
    manifest = read_json(STORE / 'manifest.json')
    paths = list(FIXED) + [DATA / 'train.npz', DATA / 'val.npz',
        E2 / 'runs/run-r1/guard.json', E2W / 'worker-completion.json']
    paths += [STORE / manifest['arrays'][key]['path'] for key in ('user_id', 'raw_item_id', 'date', 'token')]
    paths += [E2W / f'k{k}-predictions.npz' for k in KS]
    paths += [Path(manifest['source_pins'][key]['path']) for key in
              ('history_builder', 'event_store_builder', 'completion_diagnostics')]
    return sorted(set(paths), key=str)


def validate_bindings(bindings):
    pins = {Path(path): sha for path, sha in bindings['sha256'].items()}
    require(set(required_input_paths()).issubset(pins), 'Missing required immutable inputs')
    require(all(pins.get(path) == sha for path, sha in FIXED.items()), 'Canonical pin changed')
    for path, sha in pins.items():
        require(path.is_absolute() and 'test.npz' != path.name.lower(), 'Invalid input binding')
        require(digest(path) == sha, f'Input digest differs: {path}')
    return pins


class Monitor:
    def __init__(self, run_dir, seconds=1200, rss_bytes=8 * 1024**3):
        self.run_dir, self.started = run_dir, time.perf_counter()
        self.seconds, self.rss_bytes, self.peak = seconds, rss_bytes, 0
        self.process = psutil.Process()

    def check(self, stage=None, **fields):
        elapsed, rss = time.perf_counter() - self.started, self.process.memory_info().rss
        self.peak = max(self.peak, rss)
        require(elapsed < self.seconds, '20 minute CPU baseline wall budget exceeded')
        require(rss < self.rss_bytes, '8 GiB baseline RSS budget exceeded')
        if stage:
            with (self.run_dir / 'progress.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(dict(stage=stage, elapsed_seconds=elapsed, rss_bytes=rss, **fields)) + '\n')


def prefix_counts(tokens, days, vocab, window=None, cutoff=CUTOFF):
    mask = (days < cutoff) & (tokens >= 3) & (tokens < vocab)
    if window is not None:
        mask &= days >= cutoff - window
    return np.bincount(tokens[mask], minlength=vocab)[3:vocab]


def rank(scores, raw_ids, fallback=None, width=512):
    require(np.isfinite(scores).all(), 'Nonfinite ranking scores')
    fallback = np.zeros_like(scores) if fallback is None else fallback
    return np.lexsort((raw_ids, -fallback, -scores))[:width]


def binary_fit(users, tokens, days, vocab, cutoff=CUTOFF):
    mask = (days < cutoff) & (tokens >= 3) & (tokens < vocab)
    unique_users, user_rows = np.unique(users[mask], return_inverse=True)
    matrix = sparse.csr_matrix((np.ones(mask.sum(), np.float32),
        (user_rows, tokens[mask] - 3)), shape=(len(unique_users), vocab - 3), dtype=np.float32)
    require(matrix.nnz == int(mask.sum()) and np.all(matrix.data == 1), 'Duplicate user/item prefix rows')
    matrix.sort_indices()
    return matrix, {'raw_prefix_rows': len(users), 'known_fit_rows': int(mask.sum()),
                    'unique_users': len(unique_users), 'binary_nnz': matrix.nnz,
                    'supported_items': int(np.count_nonzero(matrix.getnnz(axis=0))),
                    'duplicate_user_item_pairs': 0}


def histories(x, lengths, vocab, last=None):
    output = []
    for row, length in zip(x, lengths):
        require(1 <= length <= 64 and row[0] == 1, 'Invalid prepared BOS/history length')
        values = row[1:int(length)]
        if last is not None:
            values = values[-last:]
        output.append(np.unique(values[(values >= 3) & (values < vocab)] - 3))
    return output


def cosine_neighbors(matrix, seeds, raw_ids, check=lambda **kw: None, limit=200):
    """Only seed rows, at most 64 by catalog dense; no full item-item matrix."""
    item_user = matrix.T.tocsr()
    norms = np.sqrt(np.asarray(item_user.sum(axis=1)).ravel())
    ids = np.empty((len(seeds), min(limit, matrix.shape[1] - 1)), np.int32)
    weights = np.empty(ids.shape, np.float32)
    for start in range(0, len(seeds), 64):
        check(stage='neighbors', seed_rows_done=start, seed_rows_total=len(seeds))
        block = seeds[start:start + 64]
        counts = (item_user[block] @ item_user.T).toarray()
        require(counts.shape[0] <= 64 and counts.shape[1] == len(raw_ids), 'Unbounded cosine block')
        denom = norms[block, None] * norms[None, :]
        np.divide(counts, denom, out=counts, where=denom > 0)
        counts[denom == 0] = 0
        for offset, seed in enumerate(block):
            scores = counts[offset]
            scores[seed] = -1
            chosen = np.lexsort((raw_ids, -scores))[:ids.shape[1]]
            require(seed not in chosen, 'Self neighbor retained')
            ids[start + offset], weights[start + offset] = chosen, scores[chosen]
    return ids, weights


def infer_knn(query_histories, seeds, neighbors, weights, raw_ids, pop, check=lambda **kw: None, width=512):
    lookup = {int(seed): index for index, seed in enumerate(seeds)}
    predictions = np.empty((len(query_histories), width), np.int32)
    no_evidence = 0
    for index, values in enumerate(query_histories):
        if index % 128 == 0:
            check(stage='inference', rows_done=index, rows_total=len(query_histories))
        scores = np.zeros(len(raw_ids), np.float32)
        rows = np.asarray([lookup[int(value)] for value in values], np.int64)
        if len(rows):
            np.add.at(scores, neighbors[rows].ravel(), weights[rows].ravel())
        no_evidence += int(not np.any(scores > 0))
        predictions[index] = rank(scores, raw_ids, pop, width) + 3
    return predictions, no_evidence


def row_metrics(predictions, targets, known):
    hits = (predictions[:, :10] == targets[:, None]) & known[:, None]
    require(not np.any(hits[~known]), 'Unknown target counted as hit')
    return {'map@10': (hits / np.arange(1, hits.shape[1] + 1)).sum(axis=1),
            'recall@10': hits.any(axis=1).astype(float),
            'ndcg@10': (hits / np.log2(np.arange(hits.shape[1]) + 2)).sum(axis=1)}


def summarize(predictions, targets, known):
    result = {key: float(values.mean()) for key, values in row_metrics(predictions, targets, known).items()}
    for k in KS:
        covered = (predictions[:, :k] == targets[:, None]).any(axis=1) & known
        result[f'coverage@{k}'] = float(covered.mean())
        result[f'coverage_count@{k}'] = int(covered.sum())
        result[f'coverage_known@{k}'] = float(covered.sum() / known.sum()) if known.any() else None
    return result


def paired_bootstrap(delta, users, repetitions=1000, seed=20261023):
    """Cluster resampling by user, ratio of summed row differences to row counts."""
    unique, groups = np.unique(users, return_inverse=True)
    sums = np.bincount(groups, weights=delta)
    counts = np.bincount(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for index in range(repetitions):
        sample = rng.integers(0, len(unique), len(unique))
        estimates[index] = sums[sample].sum() / counts[sample].sum()
    return {'delta_baseline_minus_e2': float(np.mean(delta)),
            'ci95_percentile': np.quantile(estimates, [0.025, 0.975]).tolist(),
            'resampling_unit': 'user_id cluster; row-weighted ratio in each replicate',
            'users': len(unique), 'repetitions': repetitions, 'seed': seed}


def validate_sources(pins):
    manifest = read_json(STORE / 'manifest.json')
    require(manifest['contract'] == 'kion-completion-event-store-v1' and manifest['state'] == 'completed'
        and manifest['fit_cutoff'] == '2021-08-09' and manifest['events_count'] == 4649162
        and manifest['coverage']['raw_events'] == 'complete_before_role_cap'
        and manifest['final_validation']['status'] == 'passed', 'Event-store source contract changed')
    require(manifest['source_pins']['interactions']['sha256'] ==
        '201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf', 'Raw provenance differs')
    for key in ('user_id', 'raw_item_id', 'date', 'token'):
        require(pins[STORE / manifest['arrays'][key]['path']] == manifest['arrays'][key]['sha256'], 'Array SHA differs')
    for split in ('train', 'val'):
        require(pins[DATA / f'{split}.npz'] == manifest['prepared_pins'][f'{split}.npz']['sha256'], 'Prepared SHA differs')
    for key in ('history_builder', 'event_store_builder', 'completion_diagnostics'):
        source = manifest['source_pins'][key]
        require(pins[Path(source['path'])] == source['sha256'], 'Source SHA differs')
    closure = read_json(E2 / 'root-finalization-r1.json')
    require(closure['state'] == 'completed' and closure['guard_exit_code'] == 0 and closure['session_reaped'], 'E2 closure failed')
    for relative in ('runs/run-r1/guard.json', 'runs/run-r1/worker/worker-completion.json'):
        require(pins[E2 / relative] == closure['artifact_sha256'][relative], 'E2 closure digest differs')
    guard, completion = read_json(E2 / 'runs/run-r1/guard.json'), read_json(E2W / 'worker-completion.json')
    require(guard['state'] == 'completed' and guard['exit_code'] == 0 and guard['cleanup_quiescent']
            and guard['cleanup_native_tree_closed'] and guard['cleanup_tree_closed_after'], 'E2 guard incomplete')
    require(completion['state'] == 'completed' and completion['optimizer_updates'] == 0
            and not completion['train_called'] and not completion['test_accessed'], 'E2 contract differs')
    for k in KS:
        require(pins[E2W / f'k{k}-predictions.npz'] == completion['results'][str(k)]['predictions_sha256'], 'E2 prediction pin differs')
    return manifest


def replay_e2(archive, identity):
    require(all(np.array_equal(archive[key], identity[key]) for key in E2_IDENTITY), 'E2 semantic cohort differs')
    top, covered = archive['retrieval_top_items'].copy(), archive['retrieval_covered'].copy()
    require(top.shape == (len(identity['row_index']), 10) and covered.dtype == bool
            and not np.any(covered & ~identity['known_target']), 'E2 retrieval shape/unknown error')
    return top, covered


def run(run_dir, bindings_path):
    require(run_dir.resolve() == Path(__file__).resolve().parent / 'runs/run-r1', 'Unexpected output directory')
    bindings_digest = digest(bindings_path)
    pins = validate_bindings(read_json(bindings_path))
    manifest = validate_sources(pins)
    run_dir.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(run_dir)
    try:
        monitor.check(stage='validated-inputs')
        started = time.perf_counter()
        metadata, cohort = read_json(DATA / 'metadata.json'), read_json(COHORT)
        vocab = int(metadata['vocab_size'])
        require(vocab == 15418, 'Catalog vocabulary changed')
        raw_ids = np.empty(vocab - 3, np.int64)
        mapping = metadata['item_id_to_token']
        require(sorted(mapping.values()) == list(range(3, vocab)), 'Noncontiguous known catalog')
        for raw, token in mapping.items():
            raw_ids[token - 3] = int(raw)
        rows = np.asarray(cohort['row_indices'], np.int64)
        require(rows.shape == (4096,) and np.all(np.diff(rows) > 0), 'Cohort rows changed')
        with np.load(DATA / 'val.npz', allow_pickle=False) as archive:
            identity = {key: archive[key][rows].copy() for key in IDENTITY[:6]}
        targets = identity['y']
        known = (targets >= 3) & (targets < vocab)
        identity.update(row_index=rows, target_token=targets, known_target=known)
        require(int(known.sum()) == 3982 and int((~known).sum()) == 114, 'Target census changed')
        require(len(np.unique(identity['user_id'])) == 4031, 'Cohort user census changed')
        require(np.array_equal(raw_ids[targets[known] - 3], identity['target_item_id'][known]), 'Known target raw mapping differs')
        e2, e2metrics = {}, {}
        for k in KS:
            with np.load(E2W / f'k{k}-predictions.npz', allow_pickle=False) as archive:
                top, covered = replay_e2(archive, identity)
            e2[k] = covered
            e2metrics[str(k)] = {key: float(values.mean()) for key, values in row_metrics(top, targets, known).items()}
            e2metrics[str(k)].update(coverage=float(covered.mean()), coverage_known=float(covered.sum() / known.sum()),
                                     coverage_count=int(covered.sum()), inference_seconds=None)
        require(abs(e2metrics['128']['map@10'] - 0.07768438430059524) < 1e-12, 'Unexpected E2 retrieval head metric')
        arrays = {key: np.load(STORE / manifest['arrays'][key]['path'], mmap_mode='r', allow_pickle=False)
                  for key in ('user_id', 'raw_item_id', 'date', 'token')}
        require(all(array.shape == (4649162,) for array in arrays.values()), 'Raw prefix census differs')
        require(np.all(arrays['date'] < CUTOFF), 'Post-cutoff event in frozen prefix')
        require(np.all((arrays['token'] >= 2) & (arrays['token'] < vocab)), 'Invalid raw tokens')
        raw_known = arrays['token'] >= 3
        require(np.array_equal(raw_ids[arrays['token'][raw_known] - 3], arrays['raw_item_id'][raw_known]), 'Raw token mapping differs')
        with np.load(DATA / 'train.npz', allow_pickle=False) as archive:
            train_y = archive['y']
            train_counts = np.bincount(train_y[(train_y >= 3) & (train_y < vocab)], minlength=vocab)[3:]
        del train_y
        pop = prefix_counts(arrays['token'], arrays['date'], vocab)
        specs = [('train_target_count', train_counts), ('raw_prefix_all', pop)]
        specs += [(f'raw_prefix_{window}d', prefix_counts(arrays['token'], arrays['date'], vocab, window)) for window in (7, 30)]
        all_hist = histories(identity['x'], identity['lengths'], vocab)
        short_hist = histories(identity['x'], identity['lengths'], vocab, 20)
        seeds = np.unique(np.concatenate(all_hist))
        parse_seconds = time.perf_counter() - started
        monitor.check(stage='parsed-inputs', seed_items=len(seeds))
        started = time.perf_counter()
        matrix, census = binary_fit(arrays['user_id'], arrays['token'], arrays['date'], vocab)
        csr_seconds = time.perf_counter() - started
        started = time.perf_counter()
        neighbors, weights = cosine_neighbors(matrix, seeds, raw_ids, monitor.check)
        neighbor_seconds = time.perf_counter() - started
        output, results = dict(identity), {}
        for k, covered in e2.items():
            output[f'e2_coverage_{k}'] = covered
        for name, counts in specs:
            started = time.perf_counter()
            predictions = np.tile(rank(counts, raw_ids) + 3, (len(rows), 1)).astype(np.int32)
            seconds = time.perf_counter() - started
            output[name] = predictions
            results[name] = dict(summarize(predictions, targets, known), inference_seconds=seconds, no_evidence_queries=None)
        for name, hist in [('itemknn_cosine_last20', short_hist), ('itemknn_cosine_all63', all_hist)]:
            started = time.perf_counter()
            predictions, empty = infer_knn(hist, seeds, neighbors, weights, raw_ids, pop, monitor.check)
            seconds = time.perf_counter() - started
            output[name] = predictions
            results[name] = dict(summarize(predictions, targets, known), inference_seconds=seconds, no_evidence_queries=empty)
        for name in results:
            predictions = output[name]
            require(predictions.shape == (4096, 512) and np.all((predictions >= 3) & (predictions < vocab))
                and np.all(np.diff(np.sort(predictions, axis=1), axis=1) > 0), 'Invalid candidate set')
            hit_matrix = (predictions == targets[:, None]) & known[:, None]
            output[f'{name}_target_rank'] = np.where(hit_matrix.any(axis=1), hit_matrix.argmax(axis=1) + 1, 0).astype(np.int16)
            for k in KS:
                covered = (predictions[:, :k] == targets[:, None]).any(axis=1) & known
                output[f'{name}_coverage_{k}'] = covered
                results[name][f'paired_bootstrap_coverage@{k}'] = paired_bootstrap(covered.astype(float) - e2[k], identity['user_id'])
            results[name]['returned_unique_known_512_every_row'] = True
            monitor.check(stage='metrics', model=name)
        np.savez_compressed(run_dir / 'predictions.npz', **output)
        require(digest(bindings_path) == bindings_digest, 'Bindings changed during run')
        validate_bindings(read_json(bindings_path))
        monitor.check(stage='inputs-revalidated')
        result = dict(state='completed', contract='kion-cpu-candidate-baselines-v1', models=results,
            e2_retrieval=e2metrics, e2_retrieval_speed='N/A: saved sweep timer included reranking',
            fit_seconds=dict(parsing_and_counts=parse_seconds, binary_csr=csr_seconds, cosine_neighbors=neighbor_seconds),
            fit_census=dict(census, query_seed_items=len(seeds), neighbor_limit=200, dense_block_rows=64,
                raw_unknown_rows=int((~raw_known).sum()), earliest_prefix_ordinal=int(arrays['date'].min()),
                latest_prefix_ordinal=int(arrays['date'].max()), unique_cohort_users=4031),
            examples=4096, known_targets=3982, unknown_target_misses=114, no_seen_item_mask=True,
            new_training=False, gpu_used=False, test_opened=False, model_imported=False,
            cpu_threads=4, peak_rss_bytes=monitor.peak, elapsed_seconds=time.perf_counter() - monitor.started,
            runtime=dict(numpy=np.__version__, scipy=scipy.__version__, psutil=psutil.__version__,
                thread_environment={name: os.environ[name] for name in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}),
            timing_policy='Fit parsing includes train/raw counts and query seeds; inference includes complete 512 selection; popularity top-list reused across users; hashes/metrics/output excluded from inference',
            binding_inputs={path.as_posix(): sha for path, sha in pins.items()}, bindings_sha256=bindings_digest,
            predictions_sha256=digest(run_dir / 'predictions.npz'))
        (run_dir / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    except BaseException as error:
        (run_dir / 'failure.json').write_text(json.dumps({'state': 'failed', 'error': repr(error)}) + '\n', encoding='utf-8', newline='\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--bindings', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.run_dir, args.bindings)
