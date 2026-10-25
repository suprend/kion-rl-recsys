"""Analyze a completed matched-KL, three-seed evaluation; numerical imports are lazy.

Import and unit tests use only the standard library. run() is for the admitted
root worker: it reads pinned evaluation arrays, never models or raw datasets.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

METRICS = ('map10', 'ndcg10', 'recall10')
SEEDS = (20261024, 20261025, 20261026)
BOOTSTRAP_SEED = 20261024
REPLICATES = 2000
CONTRACT = 'kion-matched-kl-multiseed-evaluation-v1'
TRAIN_CACHE_SHA = 'e76c365d1245355d5314e383ba53499800eb3c099fa0af7de106cced5fad15e5'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def validate_pin(entry, label):
    require(isinstance(entry, dict) and isinstance(entry.get('path'), str) and bool(entry['path']),
            'Missing path pin: ' + label)
    digest = entry.get('sha256')
    require(isinstance(digest, str) and len(digest) == 64 and
            all(c in '0123456789abcdef' for c in digest), 'Invalid SHA256 pin: ' + label)


def pinned(entry):
    validate_pin(entry, 'input')
    path = Path(entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'], 'Pinned source differs: ' + path.as_posix())
    return path, {'path': path.as_posix(), 'sha256': entry['sha256'], 'bytes': path.stat().st_size}


def rank_oracle(pool, scores, raw_ids):
    """Entire candidate pool, with ascending raw-item ID for exact score ties."""
    require(len(pool) == len(scores) == len(raw_ids) and len(pool) > 0, 'Rank shape differs')
    require(len(set(pool)) == len(pool) and len(set(raw_ids)) == len(raw_ids), 'Duplicate candidate ID')
    require(all(math.isfinite(float(score)) for score in scores), 'Nonfinite score')
    return [pool[j] for j in sorted(range(len(pool)), key=lambda j: (-scores[j], raw_ids[j]))]


def target_rank(order, target, known):
    return next((j for j, item in enumerate(order, 1) if item == target), None) if known else None


def row_metrics(rank):
    require(rank is None or (isinstance(rank, int) and rank >= 1), 'Invalid target rank')
    return (1 / rank, 1 / math.log2(rank + 1), 1.) if rank is not None and rank <= 10 else (0., 0., 0.)


def cluster_oracle(users, differences, draw):
    """Query-weighted ratio: repeated users carry every original query with them."""
    require(len(users) == len(differences) and len(users) > 0, 'Cluster shape differs')
    groups = {}
    for user, value in zip(users, differences):
        require(math.isfinite(float(value)), 'Nonfinite cluster value')
        groups.setdefault(user, []).append(float(value))
    require(len(draw) == len(groups) and all(user in groups for user in draw), 'Invalid cluster draw')
    return math.fsum(math.fsum(groups[user]) for user in draw) / sum(len(groups[user]) for user in draw)


def descriptive(values):
    require(len(values) == 3 and all(math.isfinite(x) for x in values), 'Three finite seed values required')
    return {'mean': statistics.mean(values), 'min': min(values), 'max': max(values),
            'sample_std': statistics.stdev(values), 'seed_count': 3,
            'interpretation': 'descriptive fitted-seed spread, not a seed-population confidence interval'}


def validate_metadata(doc):
    require(doc.get('contract') == CONTRACT and doc.get('state') == 'completed', 'Completed multiseed evaluation required')
    stage = doc.get('stage')
    require(stage in ('screen', 'full4096'), 'Unknown cohort stage')
    n = 256 if stage == 'screen' else 4096
    require(doc['rows'] == doc['metric_denominator'] == n and doc['candidate_width'] == 1024,
            'Fixed cohort, natural1024 or denominator differs')
    require(doc['temperature'] == 1. and doc['tie_break'] == 'score descending, raw item ID ascending',
            'Temperature or tie rule differs')
    require(all(doc[key] is True for key in ('no_target_injection', 'no_seen_filter', 'unknown_and_offpool_are_zero')),
            'Evaluation masking protocol differs')
    require(doc['final_test_reads'] == 0, 'Final-test access forbidden')
    parity = doc['original_C_parity']
    require(parity['max_abs_error'] == 0 and parity['top10_changed_rows'] == 0 and
            parity['score_tolerance_passed'] is True and parity['changes_with_margin_above_tolerance'] == 0,
            'Exact C score/top10 parity required')
    checkpoints = doc['checkpoints']
    require(isinstance(checkpoints, dict) and len(checkpoints) == 7 and 'C' in checkpoints,
            'C and exactly six fitted checkpoints required')
    require(all(isinstance(arm, str) for arm in checkpoints) and
            len({arm.casefold() for arm in checkpoints}) == 7,
            'Arm names collide on case-insensitive filesystems')
    validate_pin(checkpoints['C'], 'C')
    by_seed = {seed: {} for seed in SEEDS}
    for arm, checkpoint in checkpoints.items():
        if arm == 'C':
            continue
        seed, utility = checkpoint['seed'], checkpoint['utility']
        require(type(seed) is int and seed in SEEDS and utility in ('completion', 'residual'), 'Unexpected seed/utility')
        require(isinstance(arm, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}', arm), 'Unsafe arm name')
        require(utility not in by_seed[seed] and checkpoint['step'] == 1000, 'Duplicate arm or non-latest checkpoint')
        validate_pin(checkpoint, arm)
        validate_pin(checkpoint['training'], arm + '/training')
        by_seed[seed][utility] = arm
    require(all(set(pair) == {'completion', 'residual'} for pair in by_seed.values()), 'Missing matched seed pair')
    arms = ('C',) + tuple(by_seed[seed][utility] for seed in SEEDS for utility in ('completion', 'residual'))
    require(isinstance(doc.get('arms'), list) and len(doc['arms']) == 7 and set(doc['arms']) == set(arms),
            'Declared arms differ from checkpoint bindings')
    require(set(doc['metrics']) == set(arms), 'Saved metrics must contain exactly seven arms')
    for arm in arms:
        require(all(math.isfinite(doc['metrics'][arm][metric]) for metric in METRICS), 'Invalid saved metric')
    return stage, n, arms, by_seed


def comparison_plan(by_seed):
    """Every column uses the same bootstrap draws, including average contrasts."""
    plan = []
    for seed in SEEDS:
        completion, residual = by_seed[seed]['completion'], by_seed[seed]['residual']
        for kind, left, right in (('completion-minus-C', 'C', completion),
                                  ('residual-minus-C', 'C', residual),
                                  ('residual-minus-completion', completion, residual)):
            plan.append({'name': str(seed) + '/' + kind, 'kind': kind, 'seed': seed,
                         'terms': [(left, right)], 'left': left, 'right': right})
    for kind in ('completion-minus-C', 'residual-minus-C', 'residual-minus-completion'):
        terms = [row['terms'][0] for row in plan if row['kind'] == kind]
        plan.append({'name': 'three-seed-average/' + kind, 'kind': kind, 'seed': None,
                     'terms': terms, 'equal_seed_weight': 1 / 3})
    return plan


def small_pinned_json(entry):
    validate_pin(entry, 'small JSON')
    path = Path(entry['path']).resolve()
    require(path.suffix == '.json' and path.is_file() and path.stat().st_size <= 1024 * 1024,
            'Small pinned JSON required')
    path, pin = pinned(entry)
    value = read(path)
    require(isinstance(value, dict), 'Pinned JSON metadata object required')
    return value, pin


def matching_metadata(request, by_seed, checkpoints=None):
    """Preserve three separate train-KL flags; absence never implies matched."""
    entry = request.get('matching_receipt')
    if entry is None:
        return {'receipt': None, 'train_KL_match_by_seed': {str(seed): None for seed in SEEDS},
                'all_three_train_KL_matches': None, 'status': 'matching receipt not supplied'}
    value, pin = small_pinned_json(entry)
    require(isinstance(checkpoints, dict), 'Evaluation checkpoints required to bind matching receipt')
    arm_names = {arm for pair in by_seed.values() for arm in pair.values()}
    selected = value.get('selected_checkpoints')
    require(isinstance(selected, dict) and set(selected) == arm_names and
            all(selected[arm] == checkpoints[arm]['sha256'] for arm in arm_names),
            'Matching receipt selected checkpoint bindings differ')
    target = value.get('target_train_KL')
    require(type(target) in (int, float) and math.isfinite(target) and target > 1e-8,
            'Finite target train KL strictly above 1e-8 required')
    require(value.get('tolerance') == [.8, 1.2], 'Train KL tolerance must be exactly [0.8, 1.2]')
    kls = value.get('train_KL_by_arm')
    require(isinstance(kls, dict) and set(kls) == arm_names and
            all(type(kl) in (int, float) and math.isfinite(kl) and kl >= 0 for kl in kls.values()),
            'Six finite nonnegative train KL values required')
    require(kls[by_seed[20261024]['completion']] == target,
            'Original completion seed20261024 train KL must equal the calibration target')
    cache_metadata, cache_pin = small_pinned_json(value['cache'])
    require(cache_pin['sha256'] == TRAIN_CACHE_SHA, 'Frozen TRAIN512 cache pin differs')
    reference_metadata, reference_pin = small_pinned_json(value['reference'])
    require(reference_metadata.get('state') == 'completed' and
            reference_metadata.get('mode') == 'evaluate_train512' and reference_metadata.get('rows') == 512 and
            reference_metadata.get('cache_sha256') == TRAIN_CACHE_SHA,
            'Completed reference TRAIN512 receipt on the frozen cache required')
    reference_results = reference_metadata.get('results')
    require(isinstance(reference_results, dict), 'Reference TRAIN512 results object required')
    completion24_sha = checkpoints[by_seed[20261024]['completion']]['sha256']
    targets = [result for result in reference_results.values() if isinstance(result, dict) and
               isinstance(result.get('checkpoint'), dict) and result['checkpoint'].get('sha256') == completion24_sha]
    require(len(targets) == 1, 'Exactly one original completion seed20261024 reference result required')
    reference_kl = targets[0].get('metrics', {}).get('completion', {}).get('exact_KL')
    require(type(reference_kl) in (int, float) and math.isfinite(reference_kl) and reference_kl > 1e-8 and target == reference_kl,
            'Target train KL differs from original completion seed20261024 reference')
    flags = value.get('train_KL_match_by_seed')
    require(isinstance(flags, dict) and set(flags) == {str(seed) for seed in SEEDS}, 'Three explicit train-KL flags required')
    normalized, ratios = {}, {}
    for seed in SEEDS:
        flag = flags[str(seed)]
        require(type(flag) is bool, 'Train-KL matching flag must be boolean for ' + str(seed))
        completion_kl = kls[by_seed[seed]['completion']]
        residual_kl = kls[by_seed[seed]['residual']]
        ratio = residual_kl / completion_kl if completion_kl > 0 else None
        expected = ratio is not None and .8 <= ratio <= 1.2
        require(flag == expected, 'Train-KL matching flag differs from within-seed residual/completion ratio for ' + str(seed))
        ratios[str(seed)] = ratio
        normalized[str(seed)] = flag
    return {'receipt': pin, 'train_KL_match_by_seed': normalized,
            'all_three_train_KL_matches': all(normalized.values()), 'receipt_metadata': value,
            'train_KL_residual_over_completion_by_seed': ratios,
            'train_KL_distance_from_calibration_target_by_arm': {arm: kls[arm] - target for arm in sorted(arm_names)},
            'matching_rule': 'within-seed KL_residual / KL_completion in [0.8,1.2]; zero completion KL fails; calibration target distances are descriptive',
            'cache': cache_pin, 'reference': reference_pin,
            'reference_provenance': 'Pinned small TRAIN512 reference receipt; arrays are never opened here.',
            'status': 'checkpoint-bound TRAIN512 matching receipt; three flags independently verified'}


def load_array(np, folder, descriptor, name, check):
    path = (folder / descriptor['path']).resolve()
    require(path.parent == folder.resolve() and path.suffix == '.npy', 'Nonlocal array ' + name)
    if check:
        check(stage='multiseed-array-integrity', array=name)
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'], 'Array integrity differs ' + name)
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(value.shape) == descriptor['shape'] and str(value.dtype) == descriptor['dtype'], 'Array schema differs ' + name)
    return value


def bootstrap(np, users, differences, check=None):
    unique, inverse = np.unique(users, return_inverse=True)
    require(len(unique) > 1 and differences.ndim == 2 and len(differences) == len(users), 'Bootstrap shape differs')
    require(bool(np.isfinite(differences).all()), 'Nonfinite paired contrast')
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=differences[:, j], minlength=len(unique))
                     for j in range(differences.shape[1])], axis=1)
    result = np.empty((REPLICATES, differences.shape[1]), dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    probabilities = np.full(len(unique), 1 / len(unique))
    for start in range(0, REPLICATES, 32):
        if check:
            check(stage='multiseed-common-user-bootstrap', replicate=start, total=REPLICATES)
        stop = min(start + 32, REPLICATES)
        multiplicity = rng.multinomial(len(unique), probabilities, size=stop - start)
        result[start:stop] = (multiplicity @ sums) / (multiplicity @ counts)[:, None]
    return result, len(unique)


def run(request, check=None):
    started = time.monotonic()
    evaluation_path, evaluation_pin = pinned(request['evaluation'])
    evaluation = read(evaluation_path)
    stage, n, arms, by_seed = validate_metadata(evaluation)
    output = Path(request.get('output_dir', request.get('output', ''))).resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent) and output != Path(__file__).resolve().parent
            and not output.exists(), 'Fresh output directory beneath interpretation required')
    matching = matching_metadata(request, by_seed, evaluation['checkpoints'])
    # Executed only inside an explicitly admitted numerical job, never at import.
    import numpy as np
    arrays, loaded = {}, []
    try:
        common = ('row_index', 'user_id', 'target_token', 'known_target', 'candidate_coverage', 'pool_ids', 'candidate_raw_ids')
        for name in common + tuple(arm + suffix for arm in arms for suffix in ('_scores', '_top10')):
            value = load_array(np, evaluation_path.parent, evaluation['arrays'][name], name, check)
            loaded.append(value)
            arrays[name] = value
        require(all(arrays[name].shape == (n,) for name in common[:5]), 'Row binding shape differs')
        require(all(np.issubdtype(arrays[name].dtype, np.integer) for name in common[:3]), 'Identity arrays must be integers')
        require(len(np.unique(arrays['row_index'])) == n, 'Duplicate row index')
        pool, raw = arrays['pool_ids'], arrays['candidate_raw_ids']
        require(pool.shape == raw.shape == (n, 1024) and all(np.issubdtype(value.dtype, np.integer) for value in (pool, raw)),
                'Candidate binding shape/dtype differs')
        require(bool((pool >= 3).all()), 'Special tokens in natural candidate pool')
        known = arrays['known_target']
        require(known.dtype == arrays['candidate_coverage'].dtype == np.bool_, 'Masks must be boolean')
        coverage = known & (pool == arrays['target_token'][:, None]).any(1)
        require(np.array_equal(coverage, arrays['candidate_coverage']), 'Candidate coverage differs')
        for row in range(n):
            require(len(np.unique(pool[row])) == len(np.unique(raw[row])) == 1024, 'Duplicate candidate ID')
        values, ranks, tops, kl = {}, {}, {}, {}
        base_logp = None
        for arm in arms:
            scores, saved_top = arrays[arm + '_scores'], arrays[arm + '_top10']
            require(scores.shape == (n, 1024) and saved_top.shape == (n, 10), 'Score/top10 shape differs ' + arm)
            require(np.issubdtype(scores.dtype, np.floating) and np.issubdtype(saved_top.dtype, np.integer)
                    and bool(np.isfinite(scores).all()), 'Score/top10 dtype or finiteness differs ' + arm)
            values[arm] = np.empty((n, 3), dtype=np.float64)
            ranks[arm], tops[arm] = [], []
            for row in range(n):
                if check and row % 128 == 0:
                    check(stage='multiseed-independent-entire1024-ranks', arm=arm, row=row, total=n)
                order = np.lexsort((raw[row], -scores[row]))
                require(np.array_equal(pool[row, order[:10]], saved_top[row]), 'Saved top10 differs ' + arm)
                rank = target_rank(pool[row, order], int(arrays['target_token'][row]), bool(known[row]))
                ranks[arm].append(rank)
                values[arm][row] = row_metrics(rank)
                tops[arm].append(tuple(map(int, raw[row, order[:10]])))
            for j, metric in enumerate(METRICS):
                require(math.isclose(float(values[arm][:, j].mean()), evaluation['metrics'][arm][metric],
                                     abs_tol=1e-12, rel_tol=1e-12), 'Saved metric differs ' + arm + '/' + metric)
            z = np.asarray(scores, dtype=np.float64)
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            if arm == 'C':
                base_logp = logp
            kl[arm] = float((np.exp(logp) * (logp - base_logp)).sum(axis=1).mean())
        plan = comparison_plan(by_seed)
        deltas = [sum((values[right] - values[left] for left, right in item['terms'])) / len(item['terms']) for item in plan]
        samples, unique_users = bootstrap(np, arrays['user_id'], np.concatenate(deltas, axis=1), check)
        intervals = np.quantile(samples, [.025, .975], axis=0, method='linear')
        comparisons = {}
        for index, (item, delta) in enumerate(zip(plan, deltas)):
            comparisons[item['name']] = {**item, 'metrics': {
                metric: {'difference': float(delta[:, j].mean()),
                         'user_cluster_95ci': list(map(float, intervals[:, index * 3 + j])),
                         'gains': int((delta[:, j] > 0).sum()), 'losses': int((delta[:, j] < 0).sum()),
                         'ties': int((delta[:, j] == 0).sum())} for j, metric in enumerate(METRICS)}}
        seed_spread = {kind: {metric: descriptive([comparisons[str(seed) + '/' + kind]['metrics'][metric]['difference']
                                                  for seed in SEEDS]) for metric in METRICS}
                       for kind in ('completion-minus-C', 'residual-minus-C', 'residual-minus-completion')}
        prospective_mean = {kind: {metric: statistics.mean(
            comparisons[str(seed) + '/' + kind]['metrics'][metric]['difference'] for seed in (20261025, 20261026))
            for metric in METRICS} for kind in ('completion-minus-C', 'residual-minus-C', 'residual-minus-completion')}
        summaries = {arm: {'metrics': {metric: float(values[arm][:, j].mean()) for j, metric in enumerate(METRICS)},
                          'exact_KL_to_C': kl[arm]} for arm in arms}
        report = {'contract': 'kion-matched-kl-multiseed-analysis-v1', 'state': 'completed', 'stage': stage,
                  'source': {'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
                  'evaluation': evaluation_pin, 'rows': n, 'metric_denominator': n, 'candidate_width': 1024,
                  'arms': list(arms), 'seeds': list(SEEDS), 'checkpoints': evaluation['checkpoints'],
                  'unique_users': unique_users, 'unknown_target_misses': int((~known).sum()),
                  'offpool_known_misses': int((known & ~coverage).sum()), 'candidate_coverage_count': int(coverage.sum()),
                  'original_C_parity': evaluation['original_C_parity'], 'training_KL_matching': matching,
                  'summaries': summaries, 'comparisons': comparisons, 'seed_spread': seed_spread,
                  'primary_comparison': 'three-seed-average/residual-minus-completion',
                  'primary_metric': 'map10', 'secondary_metrics': ['ndcg10', 'recall10'],
                  'seed_role_note': '20261024 is the calibration seed; 20261025 and 20261026 are prospective seeds. The pooled three-seed mean is descriptive.',
                  'prospective_seed_comparisons': ['20261025/residual-minus-completion', '20261026/residual-minus-completion'],
                  'prospective_two_seed_mean': prospective_mean,
                  'prospective_two_seed_mean_interpretation': 'descriptive equal-weight paired mean of seeds20261025/20261026; no seed-population interval',
                  'optional_producer_diagnostics': evaluation.get('diagnostics'),
                  'optional_diagnostics_provenance': 'Copied from pinned evaluator metadata; only exact_KL_to_C is recomputed here.',
                  'bootstrap': {'replicates': REPLICATES, 'seed': BOOTSTRAP_SEED, 'batch_size': 32,
                      'unit': 'unique user clusters sampled with replacement; all queries follow their user',
                      'estimand': 'query-weighted ratio of resampled cluster sums to resampled query counts',
                      'seed_average': 'equal weight 1/3 on the three fitted heads, before user resampling',
                      'common_draw_schedule': True, 'columns': len(plan) * len(METRICS),
                      'CI': 'unadjusted percentile 2.5/97.5, NumPy linear quantiles',
                      'numpy_version': np.__version__, 'multiple_comparison_adjustment': False},
                  'limitations': ['Reused development cohort; exploratory, not an untouched holdout.',
                      'User-cluster intervals are conditional on these six fitted heads and frozen C, not over training-seed populations.',
                      'Only three seeds: min/max/sample standard deviation describe fitted-seed spread; no population p-value is claimed.',
                      'TRAIN512 KL-match flags are reported separately from evaluation-cohort KL; absence of a receipt is not a match.',
                      'Offline natural1024 relevance metrics do not measure causal satisfaction, watchtime or retention.',
                      'No models, provider, raw datasets, final test, training or policy selection are accessed by this analyzer.'],
                  'seconds': time.monotonic() - started}
        output.mkdir(parents=True)
        write(output / 'analysis.json', report)
        write(output / 'plot-data.json', {key: value for key, value in report.items() if key not in ('source', 'seconds')})
        with (output / 'paired-metrics.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['comparison', 'metric', 'difference', 'ci_low', 'ci_high', 'gains', 'losses', 'ties', 'queries', 'users'])
            for name, comparison in comparisons.items():
                for metric, value in comparison['metrics'].items():
                    writer.writerow([name, metric, value['difference'], *value['user_cluster_95ci'],
                                     value['gains'], value['losses'], value['ties'], n, unique_users])
        with (output / 'per-query.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row_index', 'user_id', 'known', 'covered'] +
                            [arm + '_' + field for arm in arms for field in ('target_rank',) + METRICS])
            for row in range(n):
                writer.writerow([int(arrays['row_index'][row]), int(arrays['user_id'][row]), bool(known[row]), bool(coverage[row])] +
                                [value for arm in arms for value in (ranks[arm][row], *map(float, values[arm][row]))])
        return report
    finally:
        for value in loaded:
            mmap = getattr(value, '_mmap', None)
            if mmap is not None:
                mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    result = run(read(parser.parse_args().request))
    print(json.dumps({key: result[key] for key in ('state', 'stage', 'rows')}))
