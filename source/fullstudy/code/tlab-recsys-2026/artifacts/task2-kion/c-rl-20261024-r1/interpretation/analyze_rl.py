"""Root-owned matched saved-output analysis; no model/provider imports or replay."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ARMS = ('C-recomputed', 'latest', 'selected')
METRICS = ('map10', 'ndcg10', 'recall10')
SEED, REPLICATES = 20261024, 2000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def pinned(entry):
    path = Path(entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'], 'Source SHA differs: '+path.as_posix())
    return path, {'path': path.as_posix(), 'sha256': entry['sha256'], 'bytes': path.stat().st_size}


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def rank_oracle(pool, scores, raw_ids):
    """Tiny independent stdlib oracle for tie/target regression tests."""
    require(len(pool) == len(scores) == len(raw_ids), 'Rank shape differs')
    require(len(set(pool)) == len(pool) and len(set(raw_ids)) == len(raw_ids), 'Duplicate IDs')
    require(all(math.isfinite(float(x)) for x in scores), 'Nonfinite score')
    return [pool[j] for j in sorted(range(len(pool)), key=lambda j: (-scores[j], raw_ids[j]))]


def row_metrics(top, target, known):
    rank = next((j for j, item in enumerate(top[:10], 1) if known and item == target), 0)
    return (1/rank if rank else 0., 1/math.log2(rank+1) if rank else 0., float(rank > 0))


def exposure(tops, weights, catalog_size):
    require(len(tops) == len(weights) and len(tops) > 0, 'Exposure alignment differs')
    counts, first = Counter(), Counter()
    for top, weight in zip(tops, weights):
        require(len(top) == 10 and len(set(map(int, top))) == 10 and weight > 0, 'Invalid weighted top10')
        counts.update({int(item): weight for item in top})
        first[int(top[0])] += weight
    total = math.fsum(weights)
    hhi = math.fsum((count/(10*total))**2 for count in counts.values())
    return {'top1_max_item_share': max(first.values())/total, 'top10_hhi': hhi,
            'effective_catalog_items': 1/hhi, 'unique_top10_items': len(counts),
            'catalog_coverage': len(counts)/catalog_size}


def cluster_oracle(users, differences, draw):
    """Exact query-weighted ratio for a supplied resampling of unique users."""
    require(len(users) == len(differences) and bool(users), 'Cluster input differs')
    groups = {}
    for user, value in zip(users, differences):
        groups.setdefault(user, []).append(float(value))
    require(len(draw) == len(groups) and all(user in groups for user in draw), 'Invalid user draw')
    return math.fsum(math.fsum(groups[user]) for user in draw)/sum(len(groups[user]) for user in draw)


def bootstrap(np, users, differences, check=None):
    """One common2000 user-multinomial schedule across both arms×three metrics."""
    unique, inverse = np.unique(users, return_inverse=True)
    require(len(unique) > 1 and differences.shape == (len(users), 6), 'Bootstrap shape differs')
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=differences[:, j], minlength=len(unique))
                     for j in range(6)], axis=1)
    rng = np.random.default_rng(SEED)
    result = np.empty((REPLICATES, 6), dtype=np.float64)
    probabilities = np.full(len(unique), 1/len(unique))
    for start in range(0, REPLICATES, 32):
        if check:
            check(stage='RL-paired-user-bootstrap', replicate=start, total=REPLICATES)
        stop = min(start+32, REPLICATES)
        multiplicity = rng.multinomial(len(unique), probabilities, size=stop-start)
        result[start:stop] = (multiplicity @ sums)/(multiplicity @ counts)[:, None]
    return result, len(unique)


def load_array(np, folder, descriptor, name, check):
    path = folder/descriptor['path']
    require(path.resolve().parent == folder.resolve() and path.suffix == '.npy', 'Nonlocal array descriptor '+name)
    if check:
        check(stage='RL-array-integrity', array=name)
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'], 'Array SHA/size differs '+name)
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(array.shape) == descriptor['shape'] and str(array.dtype) == descriptor['dtype'], 'Array schema differs '+name)
    return array


def checkpoint_steps(evaluation, training):
    require(training['state'] == 'completed' and 0 < training['updates'] <= 1000,
            'Completed bounded real RL training receipt required')
    require(training['supervised_updates'] == 0 and training['frozen_encoder'] is True, 'Training scope differs')
    require({'latest', 'selected'} <= set(evaluation['checkpoints']), 'Both latest and selected required')
    by_sha = {entry['sha256']: entry for entry in training['checkpoints']}
    steps = {}
    for arm in ('latest', 'selected'):
        entry = evaluation['checkpoints'][arm]
        require(entry['sha256'] in by_sha, 'Evaluated checkpoint absent from training receipt')
        trained = by_sha[entry['sha256']]
        require(Path(entry['path']).resolve() == Path(trained['path']).resolve(), 'Checkpoint path differs')
        steps[arm] = trained['step']
    require(steps['latest'] == training['updates'] == max(x['step'] for x in training['checkpoints']),
            'Latest is not actual final trained checkpoint')
    return steps


def run(request, check=None):
    """Request: original/evaluation/training pin objects, output; optional context[]."""
    started = time.monotonic()
    original_path, original_pin = pinned(request['original'])
    evaluation_path, evaluation_pin = pinned(request['evaluation'])
    training_path, training_pin = pinned(request['training'])
    original, evaluation, training = map(read, (original_path, evaluation_path, training_path))
    require(original.get('state') == evaluation.get('state') == 'completed', 'Incomplete evaluation')
    require(original['rows'] == original['metric_denominator'] == evaluation['metric_denominator'] == 4096
            and original['candidate_width'] == 1024, 'Full denominator/pool width differs')
    require(evaluation['original_evaluation_sha256'] == original_pin['sha256'] and
            evaluation['selection_sha256'] == original['selection']['sha256'], 'Original row/pool source pin differs')
    require(all(doc['no_seen_filter'] and doc['no_target_injection'] for doc in (original, evaluation)) and
            original['unknown_and_offpool_are_zero'] is True and evaluation['final_test_reads'] == 0 and
            evaluation['latest_and_selected_share_raw_pass'] is True, 'Evaluation protocol differs')
    require(original['tie_break'] == 'score descending, raw item ID ascending', 'Tie policy differs')
    parity = evaluation['original_C_parity']
    require(parity['score_tolerance_passed'] is True and parity['changes_with_margin_above_tolerance'] == 0,
            'C recomputation parity failed')
    steps = checkpoint_steps(evaluation, training)
    require(training['C_checkpoint_sha256'] == original['head_files']['C']['sha256'],
            'RL C ancestor differs')
    output = Path(request['output']).resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent), 'Output must remain inside interpretation')
    require(not output.exists(), 'Fresh analysis output required')
    context = []
    for entry in request.get('context', []):
        path, item = pinned(entry)
        require(item['bytes'] <= 512*1024, 'Context metadata too large')
        context.append({'label': entry['label'], **item})
    import numpy as np
    loaded = []
    try:
        arrays = {}
        for key in ('user_id', 'row_index', 'target_token', 'known_target', 'candidate_coverage', 'pool_ids', 'catalog_raw_ids'):
            value = load_array(np, original_path.parent, original['arrays'][key], key, check)
            loaded.append(value)
            arrays[key] = value
        for arm in ARMS:
            for suffix in ('_scores', '_top10'):
                key = arm+suffix
                require(key in evaluation['arrays'], 'Missing latest/selected/C array '+key)
                value = load_array(np, evaluation_path.parent, evaluation['arrays'][key], key, check)
                loaded.append(value)
                arrays[key] = value
        n, width = 4096, 1024
        require(all(arrays[key].shape == (n,) for key in ('user_id', 'row_index', 'target_token', 'known_target', 'candidate_coverage')),
                'Original row binding shapes differ')
        require(arrays['pool_ids'].shape == (n, width) and len(np.unique(arrays['row_index'])) == n,
                'Original pool/row identities differ')
        raw = arrays['catalog_raw_ids']
        require(raw.ndim == 1 and len(np.unique(raw)) == len(raw), 'Catalog mapping differs')
        known = (arrays['target_token'] >= 3) & (arrays['target_token'] < len(raw)+3)
        require(np.array_equal(known, arrays['known_target']), 'Known target mask differs')
        coverage = known & (arrays['pool_ids'] == arrays['target_token'][:, None]).any(1)
        require(np.array_equal(coverage, arrays['candidate_coverage']), 'Coverage mask differs')
        require(int((~known).sum()) == 122 and int((known & ~coverage).sum()) == 616, 'Unknown/offpool boundary differs')
        values, tops = {}, {}
        for arm in ARMS:
            scores, top = arrays[arm+'_scores'], arrays[arm+'_top10']
            require(scores.shape == (n, width) and top.shape == (n, 10), 'Evaluated score/top10 shape differs')
            values[arm] = np.empty((n, 3), dtype=np.float64)
            tops[arm] = top
            for row in range(n):
                if check and row % 128 == 0:
                    check(stage='RL-saved-rank-metrics', arm=arm, completed_rows=row, rows=n)
                pool = arrays['pool_ids'][row]
                require(len(np.unique(pool)) == width and bool(np.all((pool >= 3) & (pool < len(raw)+3))), 'Invalid original pool')
                require(bool(np.isfinite(scores[row]).all()), 'Nonfinite policy score')
                order = np.lexsort((raw[pool-3], -scores[row]))[:10]
                require(np.array_equal(pool[order], top[row]), 'Saved raw-ID top10 differs '+arm)
                values[arm][row] = row_metrics(top[row], arrays['target_token'][row], bool(known[row]))
            actual = values[arm].mean(0)
            for j, metric in enumerate(METRICS):
                require(math.isclose(float(actual[j]), evaluation['metrics'][arm][metric], abs_tol=1e-12, rel_tol=1e-12),
                        'Recomputed metric differs '+arm+metric)
        selected0_identity = None
        if steps['selected'] == 0:
            selected0_identity = {'scores_identical': bool(np.array_equal(arrays['selected_scores'], arrays['C-recomputed_scores'])),
                                  'top10_identical': bool(np.array_equal(tops['selected'], tops['C-recomputed']))}
            require(all(selected0_identity.values()), 'Selected0 differs from C-recomputed')
        differences = np.concatenate([values[arm]-values['C-recomputed'] for arm in ('latest', 'selected')], axis=1)
        samples, user_count = bootstrap(np, arrays['user_id'], differences, check)
        ci = np.quantile(samples, [.025, .975], axis=0)
        frequencies = Counter(map(int, arrays['user_id']))
        weights = {'query_weighted': [1.]*n,
                   'equal_user_weighted': [1/frequencies[int(user)] for user in arrays['user_id']]}
        summaries = {arm: {'metrics': {metric: float(values[arm][:, j].mean()) for j, metric in enumerate(METRICS)},
                          'exposure': {name: exposure(tops[arm], weight, len(raw)) for name, weight in weights.items()}}
                     for arm in ARMS}
        comparisons = {}
        for a, arm in enumerate(('latest', 'selected')):
            delta = differences[:, a*3:(a+1)*3]
            comparisons[arm+'-minus-C-recomputed'] = {
                'metrics': {metric: {'difference': float(delta[:, j].mean()), 'user_cluster_95ci': list(map(float, ci[:, a*3+j])),
                                      'gains': int((delta[:, j] > 0).sum()), 'losses': int((delta[:, j] < 0).sum()),
                                      'ties': int((delta[:, j] == 0).sum())} for j, metric in enumerate(METRICS)},
                'top10_overlap_mean': math.fsum(len(set(map(int, left)) & set(map(int, right)))/10
                    for left, right in zip(tops[arm], tops['C-recomputed']))/n}
        report = {'contract': 'C3-post-RL-matched-saved-analysis-v1', 'state': 'completed',
            'source': {'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
            'original': original_pin, 'evaluation': evaluation_pin, 'training': training_pin,
            'context': context, 'rows': n, 'unique_users': user_count, 'candidate_width': width,
            'unknown_target_misses': 122, 'offpool_known_misses': 616, 'candidate_coverage_count': int(coverage.sum()),
            'baseline': 'C-recomputed scored in same evaluation pass as latest/selected',
            'pool_binding': 'immutable original pool/row manifest; producer verifies every recomputed pool row against original',
            'original_C_parity': parity, 'training_updates': training['updates'], 'checkpoint_steps': steps,
            'checkpoint_pins': evaluation['checkpoints'], 'selected0_identity': selected0_identity,
            'summaries': summaries, 'comparisons': comparisons,
            'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'batch_size': 32,
                'unit': 'unique user sampled with replacement; carry all original queries',
                'estimand': 'query-weighted paired difference, ratio of resampled query sums/counts',
                'CI': 'percentile2.5/97.5 with NumPy linear quantiles; common draws across six columns',
                'numpy_version': np.__version__, 'multiple_comparison_adjustment': False},
            'limitations': [
                'Reused development4096 cohort; screen overlap/selection makes estimates exploratory, not untouched holdout.',
                'Conditional on one fitted C/encoder/RM, fixed candidate pools, support protocol and one RL fit.',
                'Observed-history RM is a support-limited proxy; unknown outcomes and synthetic policy prefixes are not causal counterfactual outcomes.',
                'Recommendation exposure is not actual impressions, satisfaction, watchtime or retention.',
                'Selected update0 does not erase actual latest training; report latest and selected separately.',
                'No replay, new features, SHAP, model load, policy improvement selection or final-test read in this analyzer.'],
            'seconds': time.monotonic()-started}
        output.mkdir(parents=True)
        write(output/'rl-analysis.json', report)
        with (output/'per-query.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row_index', 'user_id', 'known', 'covered'] + [arm+'_'+metric for arm in ARMS for metric in METRICS])
            for row in range(n):
                writer.writerow([int(arrays['row_index'][row]), int(arrays['user_id'][row]), bool(known[row]), bool(coverage[row])]
                                + [float(x) for arm in ARMS for x in values[arm][row]])
        lines = ['# Current C → bounded RL: matched development results', '',
                 f"Actual RL training completed **{training['updates']} updates**; latest={steps['latest']}, selected={steps['selected']}. "
                 'A selected update0 means selection retained C; the latest checkpoint was still trained.', '',
                 'All4096 original natural1024 queries are retained, including122 unknown and616 known offpool misses. '
                 'Comparisons use C recomputed in the same pass, identical original pools and raw-ID tie policy.', '',
                 '| Arm | MAP@10 | NDCG@10 | Recall@10 | top1 max share | top10 HHI | effective items | unique items |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm, summary in summaries.items():
            m, e = summary['metrics'], summary['exposure']['query_weighted']
            lines.append(f"| {arm} | {m['map10']:.6f} | {m['ndcg10']:.6f} | {m['recall10']:.6f} | "
                         f"{e['top1_max_item_share']:.6f} | {e['top10_hhi']:.6f} | {e['effective_catalog_items']:.1f} | {e['unique_top10_items']} |")
        lines += ['', '| Paired comparison | Metric | Difference | User-cluster95%CI | gain/loss/tie queries |',
                  '|---|---|---:|---|---|']
        for name, result in comparisons.items():
            for metric, value in result['metrics'].items():
                low, high = value['user_cluster_95ci']
                lines.append(f"| {name} | {metric} | {value['difference']:+.6f} | [{low:+.6f}, {high:+.6f}] | "
                             f"{value['gains']}/{value['losses']}/{value['ties']} |")
        lines += ['', 'Bootstrap:2000 paired user-cluster draws, seed20261024, common schedule across all six differences. '
                  'Full source pins, equal-user exposure, recomputation parity, checkpoint steps and per-query results '
                  'are in [rl-analysis.json](rl-analysis.json) and [per-query.csv](per-query.csv).', '',
                  '**Exploratory reused development:** screen overlap and selection preclude a fresh-holdout claim. '
                  'Results are conditional on one fit and its support protocol. RM reward is a proxy; synthetic-prefix '
                  'and unsupported recommendations do not identify causal counterfactual outcomes or long-term user benefit.', '']
        if context:
            lines.append('Pinned campaign evidence:')
            lines.extend('- ['+entry['label']+']('+entry['path']+') — SHA256 '+entry['sha256'] for entry in context)
        (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
        return report
    finally:
        for value in loaded:
            mmap = getattr(value, '_mmap', None)
            if mmap is not None:
                mmap.close()
