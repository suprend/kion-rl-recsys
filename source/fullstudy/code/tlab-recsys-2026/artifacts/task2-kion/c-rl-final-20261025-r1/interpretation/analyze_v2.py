"""Matched scientific three-arm analysis v2. Numerical dependencies are lazy.

No model, provider, training, raw-dataset, GPU or network access belongs here.
Only root-owned completed evaluation artifacts may be analyzed by run().
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ARMS = ('C', 'RL-completion', 'RL-residual')
METRICS = ('map10', 'ndcg10', 'recall10')
PAIRS = (('C', 'RL-completion'), ('C', 'RL-residual'), ('RL-completion', 'RL-residual'))
SEED, REPLICATES = 20261024, 2000
EVALUATION_PROTOCOL_SHA256 = 'ee0e1e06f060f27e4dd149a8c26c02827f9b1c9fb67a57d688d84e166fddfd25'
RANK_BINS = ('unknown', 'offpool', '1', '2–3', '4–10', '11–50', '51–100', '101–1024')
DELTA_BINS = ('[-1,-0.5]', '(-0.5,-0.1]', '(-0.1,0)', '0', '(0,0.1]', '(0.1,0.5]', '(0.5,1]')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def pinned(entry):
    path = Path(entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'], 'Source SHA differs: ' + path.as_posix())
    return path, {'path': path.as_posix(), 'sha256': entry['sha256'], 'bytes': path.stat().st_size}


def rank_oracle(pool, scores, raw_ids):
    """Independent stdlib oracle: entire pool, deterministic raw-ID ties."""
    require(len(pool) == len(scores) == len(raw_ids), 'Rank shape differs')
    require(len(set(pool)) == len(pool) and len(set(raw_ids)) == len(raw_ids), 'Duplicate IDs')
    require(all(math.isfinite(float(x)) for x in scores), 'Nonfinite score')
    return [pool[j] for j in sorted(range(len(pool)), key=lambda j: (-scores[j], raw_ids[j]))]


def target_rank(ordered_pool, target, known):
    if not known:
        return None
    return next((j for j, item in enumerate(ordered_pool, 1) if item == target), None)


def row_metrics(rank):
    return ((1 / rank, 1 / math.log2(rank + 1), 1.) if rank is not None and rank <= 10
            else (0., 0., 0.))


def rank_bin(rank, known):
    if not known:
        return 'unknown'
    if rank is None:
        return 'offpool'
    require(1 <= rank <= 1024, 'Rank outside natural1024')
    for bound, label in ((1, '1'), (3, '2–3'), (10, '4–10'), (50, '11–50'), (100, '51–100'), (1024, '101–1024')):
        if rank <= bound:
            return label


def delta_bin(value):
    require(math.isfinite(value) and -1 <= value <= 1, 'Invalid RR delta')
    if value == 0:
        return '0'
    for bound, label in ((-.5, DELTA_BINS[0]), (-.1, DELTA_BINS[1]), (0, DELTA_BINS[2]),
                         (.1, DELTA_BINS[4]), (.5, DELTA_BINS[5]), (1, DELTA_BINS[6])):
        if value <= bound:
            return label


def exposure(tops, weights, catalog_size=None):
    require(len(tops) == len(weights) and len(tops) > 0, 'Exposure alignment differs')
    counts, first = Counter(), Counter()
    for top, weight in zip(tops, weights):
        require(len(top) == 10 and len(set(map(int, top))) == 10 and math.isfinite(weight) and weight > 0,
                'Invalid weighted top10')
        counts.update({int(item): weight for item in top})
        first[int(top[0])] += weight
    total = math.fsum(weights)
    shares = sorted((count / (10 * total) for count in counts.values()), reverse=True)
    hhi = math.fsum(share ** 2 for share in shares)
    return {'top1_max_item_share': max(first.values()) / total, 'top10_hhi': hhi,
            'effective_catalog_items': 1 / hhi, 'unique_top10_items': len(counts),
            'catalog_size': catalog_size,
            'catalog_coverage': len(counts) / catalog_size if catalog_size else None,
            'top10_exposure_concentration': {str(k): math.fsum(shares[:k]) for k in (1, 5, 10)},
            'sorted_top10_exposure': [{'raw_id': item, 'share': count / (10 * total)}
                                    for item, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]}


def cluster_oracle(users, differences, draw):
    require(len(users) == len(differences) and len(users) > 0, 'Cluster input differs')
    groups = {}
    for user, value in zip(users, differences):
        groups.setdefault(user, []).append(float(value))
    require(len(draw) == len(groups) and all(user in groups for user in draw), 'Invalid user draw')
    return math.fsum(math.fsum(groups[user]) for user in draw) / sum(len(groups[user]) for user in draw)


def neutral_oracle(reference_probs, utility, support):
    """Unsupported actions have exactly zero advantage, never a zero outcome."""
    require(len(reference_probs) == len(utility) == len(support), 'Utility shape differs')
    require(all(math.isfinite(p) and p >= 0 for p in reference_probs) and
            math.isclose(math.fsum(reference_probs), 1., abs_tol=1e-12), 'Invalid probabilities')
    mass = math.fsum(p for p, supported in zip(reference_probs, support) if supported)
    baseline = math.fsum(p * u for p, u, s in zip(reference_probs, utility, support) if s) / mass if mass else 0.
    return [u - baseline if s else 0. for u, s in zip(utility, support)]


def shift_oracle(reference_probs, left_probs, right_probs, utility, support):
    """Exact centered contribution per candidate for one matched state."""
    advantage = neutral_oracle(reference_probs, utility, support)
    require(len(left_probs) == len(right_probs) == len(advantage), 'Shift shape differs')
    return [(right - left) * a for left, right, a in zip(left_probs, right_probs, advantage)]


def bootstrap(np, users, differences, check=None):
    """Common user-multinomial schedule for all nine paired metric columns."""
    unique, inverse = np.unique(users, return_inverse=True)
    require(len(unique) > 1 and differences.shape == (len(users), 9), 'Bootstrap shape differs')
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    sums = np.stack([np.bincount(inverse, weights=differences[:, j], minlength=len(unique))
                     for j in range(9)], axis=1)
    rng = np.random.default_rng(SEED)
    result = np.empty((REPLICATES, 9), dtype=np.float64)
    probabilities = np.full(len(unique), 1 / len(unique))
    for start in range(0, REPLICATES, 32):
        if check:
            check(stage='scientific3-user-bootstrap', replicate=start, total=REPLICATES)
        stop = min(start + 32, REPLICATES)
        multiplicity = rng.multinomial(len(unique), probabilities, size=stop - start)
        result[start:stop] = (multiplicity @ sums) / (multiplicity @ counts)[:, None]
    return result, len(unique)


def load_array(np, folder, descriptor, name, check=None):
    path = (folder / descriptor['path']).resolve()
    require(path.parent == folder.resolve() and path.suffix == '.npy', 'Nonlocal array descriptor ' + name)
    if check:
        check(stage='scientific3-array-integrity', array=name)
    require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'], 'Array SHA/size differs ' + name)
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(value.shape) == descriptor['shape'] and str(value.dtype) == descriptor['dtype'], 'Array schema differs ' + name)
    return value


def validate_metadata(doc):
    require(doc.get('contract') == 'kion-scientific3-evaluation-v1' and doc.get('state') == 'completed',
            'Completed scientific3 evaluation required')
    stage = doc.get('stage')
    require(stage in ('screen', 'full4096'), 'Unknown evaluation stage')
    n = 256 if stage == 'screen' else 4096
    require(doc['rows'] == doc['metric_denominator'] == n and doc['candidate_width'] == 1024,
            'Cohort/pool/denominator differs')
    require(doc['temperature'] == 1. and doc['tie_break'] == 'score descending, raw item ID ascending',
            'Temperature/tie policy differs')
    require(all(doc[key] is True for key in ('no_target_injection', 'no_seen_filter', 'unknown_and_offpool_are_zero')),
            'Evaluation masking protocol differs')
    require(doc['final_test_reads'] == 0, 'Final-test access forbidden')
    parity = doc['original_C_parity']
    require(parity['score_tolerance_passed'] is True and
            parity['changes_with_margin_above_tolerance'] == 0, 'C parity failed')
    require(set(ARMS) <= set(doc['metrics']), 'All scientific arms required')
    for arm in ARMS[1:]:
        checkpoint = doc['checkpoints'][arm]
        require(checkpoint['step'] == 1000 and isinstance(checkpoint['sha256'], str) and len(checkpoint['sha256']) == 64,
                'Actual latest1000 checkpoint required ' + arm)
    return stage, n


def curve_data(entries):
    """Read small pinned legacy screen JSON only; never blend with full cohort."""
    output, pins = [], []
    require(not entries or sorted(int(entry['step']) for entry in entries) == [0, 250, 500, 1000],
            'Complete legacy checkpoint curve required')
    for entry in sorted(entries, key=lambda x: x['step']):
        path, pin = pinned(entry)
        require(pin['bytes'] < 64 * 1024, 'Curve metadata too large')
        value = read(path)
        require(value['rows'] == 256, 'Legacy curve is screen256 only')
        output.append({'step': entry['step'], **{key: value[key] for key in (
            'map10', 'ndcg10', 'recall10', 'expected_effective_reward', 'exact_KL',
            'unsupported_mass', 'top1_max_share', 'top10_HHI', 'effective_catalog')}})
        pins.append(pin)
    if output:
        baseline = output[0]
        for row in output:
            row['gates'] = {'map': row['map10'] - baseline['map10'] >= -.002,
                            'KL': row['exact_KL'] <= .05,
                            'unsupported': row['unsupported_mass'] - baseline['unsupported_mass'] <= .01,
                            'top1': row['top1_max_share'] - baseline['top1_max_share'] <= .05}
            row['feasible'] = all(row['gates'].values())
    return output, pins


def _distribution(np, scores):
    z = np.asarray(scores, dtype=np.float64)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return logp, np.exp(logp)


def _neutral(np, reference, utility, support):
    mass = (reference * support).sum(1, keepdims=True)
    clean = np.where(support, utility, 0.)
    baseline = (reference * clean).sum(1, keepdims=True) / np.where(mass > 0, mass, 1.)
    return np.where(support, clean - baseline, 0.)


def _screen(np, arrays, raw, orders, tops, weights, check):
    n, width = arrays['pool_ids'].shape
    support = arrays['support']
    require(support.dtype == np.bool_ and arrays['fallback'].dtype == np.bool_ and
            support.shape == arrays['fallback'].shape == (n, width) and not bool((support & arrays['fallback']).any()),
            'RM support/fallback differs')
    utilities = {name: arrays[key] for name, key in (
        ('completion', 'raw_reward'), ('residual', 'utility_residual'), ('prior', 'utility_prior'))}
    for name, value in utilities.items():
        require(value.shape == (n, width) and bool(np.isfinite(value[support]).all()), 'Invalid supported utility ' + name)
    require(bool(((utilities['completion'][support] >= 0) & (utilities['completion'][support] <= 1)).all()), 'Raw RM range differs')
    require(bool(((utilities['prior'][support] >= 0) & (utilities['prior'][support] <= 1)).all()), 'Observed prior range differs')
    require(bool(((utilities['residual'][support] >= -1) & (utilities['residual'][support] <= 1)).all()), 'Residual range differs')
    require(bool(np.allclose(utilities['residual'][support],
                            utilities['completion'][support] - utilities['prior'][support], atol=1e-12, rtol=0)),
            'Supported residual is not raw RM minus frozen observed-fit prior')
    history = arrays['history_tokens']
    require(history.shape == (n, 64) and np.issubdtype(history.dtype, np.integer), 'H63 binding differs')
    popularity = arrays['strict7d_count']
    require(popularity.shape == (n, width) and bool(np.isfinite(popularity).all()) and bool((popularity >= 0).all()),
            'Strict7d count is missing/invalid; must never coerce missing to zero')
    seen = np.zeros((n, width), dtype=np.bool_)
    for row in range(n):
        seen[row] = np.isin(arrays['pool_ids'][row], history[row][history[row] >= 3])
    logpop = np.log1p(popularity)
    logps, probabilities = {}, {}
    for arm in ARMS:
        logps[arm], probabilities[arm] = _distribution(np, arrays[arm + '_scores'])
    advantages = {name: _neutral(np, probabilities['C'], value, support) for name, value in utilities.items()}
    summaries = {}
    for arm in ARMS:
        p, logp, order = probabilities[arm], logps[arm], orders[arm]
        top_support = np.take_along_axis(support, order, axis=1)
        supported_mass = (p * support).sum(1)
        denominator = float(supported_mass.sum())
        ut = {}
        for name, utility in utilities.items():
            clean = np.where(support, utility, 0.)
            numerator = float((p * clean).sum())
            top_values = np.take_along_axis(clean, order, axis=1)
            ut[name] = {'centered_supported_neutral_proxy': float((p * advantages[name]).sum(1).mean()),
                        'supported_probability_numerator': numerator,
                        'supported_probability_denominator': denominator,
                        'supported_conditional_mean': numerator / denominator if denominator else None,
                        'supported_top10_numerator': float(top_values.sum()),
                        'supported_top10_denominator': int(top_support.sum()),
                        'supported_top10_conditional_mean': float(top_values.sum() / top_support.sum()) if top_support.sum() else None}
        top_seen = np.take_along_axis(seen, order, axis=1)
        top_pop = np.take_along_axis(logpop, order, axis=1)
        top_zero = np.take_along_axis(popularity == 0, order, axis=1)
        behavior = {'observed_H63_seen_at10': float(top_seen.mean()),
                    'observed_H63_unseen_at10': float((~top_seen).mean()),
                    'mean_log1p_strict7d_count_at10': float(top_pop.mean()),
                    'zero_strict7d_count_share_at10': float(top_zero.mean()),
                    'missing_count_share_at10': 0., 'unsupported_share_at10': float((~top_support).mean()),
                    'per_slot': [{'slot': j + 1, 'observed_H63_seen': float(top_seen[:, j].mean()),
                                  'mean_log1p_strict7d_count': float(top_pop[:, j].mean()),
                                  'zero_count_share': float(top_zero[:, j].mean()),
                                  'unsupported_share': float((~top_support[:, j]).mean())} for j in range(10)],
                    'novelty_self_information': None,
                    'novelty_unavailable_reason': 'No full-vocabulary strict7d denominator; pool counts are not a catalog distribution.'}
        if 'candidate_sources' in arrays:
            source = arrays['candidate_sources']
            require(source.shape == (n, width, 14) and bool(np.isfinite(source).all()), 'Source14 schema differs')
            behavior['candidate_source_channels_at10'] = [
                {'channel': j, 'mean_value': float(np.take_along_axis(source[:, :, j], order, axis=1).mean())}
                for j in range(12)]
        summaries[arm] = {'utilities': ut, 'unsupported_mass': float((1 - supported_mass).mean()),
                          'fallback_mass': float((p * arrays['fallback']).sum(1).mean()),
                          'states_without_support': int((~support.any(1)).sum()),
                          'exact_KL_to_C': float((p * (logp - logps['C'])).sum(1).mean()),
                          'entropy': float(-(p * logp).sum(1).mean()),
                          'behavior': behavior}
    flat_raw, inverse = np.unique(raw.ravel(), return_inverse=True)
    item_rows, decompositions = [], {}
    for left, right in PAIRS:
        name = right + '-minus-' + left
        if check:
            check(stage='scientific3-exact-item-decomposition', comparison=name)
        dp = probabilities[right] - probabilities[left]
        probability_shift = np.bincount(inverse, weights=dp.ravel(), minlength=len(flat_raw)) / n
        supported_shift = np.bincount(inverse, weights=(dp * support).ravel(), minlength=len(flat_raw)) / n
        contributions = {u: np.bincount(inverse, weights=(dp * a).ravel(), minlength=len(flat_raw)) / n
                         for u, a in advantages.items()}
        decompositions[name] = {}
        for utility, values in contributions.items():
            expected = (summaries[right]['utilities'][utility]['centered_supported_neutral_proxy'] -
                        summaries[left]['utilities'][utility]['centered_supported_neutral_proxy'])
            total = float(values.sum())
            require(math.isclose(total, expected, abs_tol=1e-12, rel_tol=0), 'Item contribution identity failed')
            order = sorted(range(len(flat_raw)), key=lambda j: (-abs(float(values[j])), int(flat_raw[j])))
            selected = order[:20]
            decompositions[name][utility] = {
                'proxy_difference': expected, 'sum_all_item_contributions': total,
                'identity_absolute_error': abs(total - expected),
                'positive_contributions': float(values[values > 0].sum()),
                'negative_contributions': float(values[values < 0].sum()),
                'top_items_by_absolute_contribution': [
                    {'raw_id': int(flat_raw[j]), 'contribution': float(values[j]),
                     'probability_shift_per_query': float(probability_shift[j]),
                     'supported_probability_shift_per_query': float(supported_shift[j])} for j in selected],
                'remaining_items_contribution': math.fsum(float(values[j]) for j in order[20:])}
        for j, raw_id in enumerate(flat_raw):
            item_rows.append([name, int(raw_id), float(probability_shift[j]), float(supported_shift[j])] +
                             [float(contributions[u][j]) for u in ('completion', 'residual', 'prior')])
    return summaries, decompositions, item_rows


def producer_top10_raw_reward(np, raw_reward, support, order, parent_dtype):
    """Replicate just the legacy raw-RM top10 reduction, not scientific utilities.

    The exporter widened original float32 values losslessly to float64. Prove
    every exported value can be restored exactly before using parent arithmetic.
    """
    require(parent_dtype == 'float32', 'Parent RM reward dtype must be float32')
    require(raw_reward.dtype == np.float64, 'Exported raw RM reward dtype must be float64')
    restored = raw_reward.astype(np.float32)
    require(np.array_equal(restored.astype(np.float64), raw_reward),
            'Every raw RM export value must roundtrip exactly through float32')
    top_support = np.take_along_axis(support, order, axis=1)
    top_reward = np.take_along_axis(np.where(support, restored, 0.), order, axis=1)
    require(top_reward.dtype == np.float32, 'Producer top10 accumulation dtype differs')
    return float(top_reward.sum() / top_support.sum()) if top_support.sum() else None


def validate_parent_reward_metadata(parent, n, cache_sha256):
    require(parent['contract'] == 'current-C-local-RM-r2-bridge-v1' and parent['ready'] is True,
            'Completed parent RM bridge metadata required')
    require(parent['states'] == n and parent['candidates_per_state'] == 1024 and parent['rows'] == n * 1024 and
            parent['cache']['sha256'] == cache_sha256, 'Parent RM cohort/cache differs')
    require(parent['files']['rewards.npy']['dtype'] == 'float32', 'Parent RM reward dtype must be float32')
    require(parent['files']['rewards.npy']['shape'] == [n, 1024], 'Parent RM reward shape differs')
    return 'float32'


def run(request, check=None):
    """Analyze one completed cohort, pinned by SHA; never silently substitutes cohorts."""
    started = time.monotonic()
    protocol_path, protocol_pin = pinned(request['protocol'])
    require(protocol_pin['sha256'] == EVALUATION_PROTOCOL_SHA256, 'Frozen evaluation protocol differs')
    protocol = read(protocol_path)
    require(protocol['contract'] == 'kion-final-three-arm-evaluation-v1' and protocol['arms'] == list(ARMS) and
            protocol['uncertainty']['draws'] == REPLICATES and protocol['uncertainty']['seed'] == SEED and
            protocol['separate_cohorts'] is True and protocol['natural_pool'] == 1024,
            'Evaluation protocol estimator differs')
    evaluation_path, evaluation_pin = pinned(request['evaluation'])
    evaluation = read(evaluation_path)
    stage, n = validate_metadata(evaluation)
    producer_precision = None
    if stage == 'screen':
        parent_path, parent_pin = pinned(evaluation['reward'])
        require(parent_path.suffix == '.json' and parent_pin['bytes'] <= 64 * 1024,
                'Small parent RM metadata JSON required')
        require(request['input_hashes'].get(parent_path.as_posix()) == parent_pin['sha256'],
                'Parent RM metadata must be monitored in input_hashes')
        parent = read(parent_path)
        parent_dtype = validate_parent_reward_metadata(parent, n, evaluation['cache']['sha256'])
        require(evaluation['arrays']['raw_reward']['dtype'] == 'float64', 'Raw RM export must declare float64')
        producer_precision = {'parent_reward': parent_pin,
            'parent_reward_array': parent['files']['rewards.npy'], 'parent_dtype': parent_dtype,
            'export_dtype': 'float64', 'scientific_utility_dtype': 'float64',
            'only_producer_check_reconstructed': 'supported_top10_raw_reward',
            'reduction': 'original np.where(support,float32 rewards,0.0), identical top order, float32 sum / int64 supported count',
            'all_raw_export_values_roundtrip_float32': False, 'values_checked': n * 1024, 'by_arm': {}}
    output = Path(request['output']).resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent) and not output.exists(), 'Fresh interpretation output required')
    curve, curve_pins = curve_data(request.get('checkpoint_curve', []))
    # Only executed by an explicitly admitted numerical analysis, never by import/tests.
    import numpy as np
    loaded = []
    try:
        arrays = {}
        common = ['user_id', 'row_index', 'target_token', 'known_target', 'candidate_coverage', 'pool_ids', 'candidate_raw_ids']
        common += ['catalog_raw_ids'] if stage == 'full4096' else ['raw_reward', 'utility_residual',
            'utility_prior', 'support', 'fallback', 'history_tokens', 'strict7d_count']
        if stage == 'screen' and 'candidate_sources' in evaluation['arrays']:
            common.append('candidate_sources')
        for key in common + [arm + suffix for arm in ARMS for suffix in ('_scores', '_top10')]:
            value = load_array(np, evaluation_path.parent, evaluation['arrays'][key], key, check)
            loaded.append(value)
            arrays[key] = value
        require(all(arrays[key].shape == (n,) for key in common[:5]), 'Row binding shape differs')
        require(all(np.issubdtype(arrays[key].dtype, np.integer) for key in ('user_id', 'row_index', 'target_token')),
                'Row/user/target identities must be integer')
        require(len(np.unique(arrays['row_index'])) == n and arrays['pool_ids'].shape == (n, 1024), 'Pool/row identity differs')
        require(np.issubdtype(arrays['pool_ids'].dtype, np.integer), 'Pool tokens must be integer')
        require(arrays['known_target'].dtype == arrays['candidate_coverage'].dtype == np.bool_, 'Masks must be boolean')
        known, pool = arrays['known_target'], arrays['pool_ids']
        coverage = known & (pool == arrays['target_token'][:, None]).any(1)
        require(np.array_equal(coverage, arrays['candidate_coverage']), 'Coverage mask differs')
        catalog_size = evaluation.get('catalog_size')
        if stage == 'full4096':
            catalog = arrays['catalog_raw_ids']
            require(catalog.ndim == 1 and np.issubdtype(catalog.dtype, np.integer) and
                    len(np.unique(catalog)) == len(catalog), 'Catalog mapping differs')
            require(bool(((pool >= 3) & (pool < len(catalog) + 3)).all()), 'Invalid pool token')
            known_recomputed = (arrays['target_token'] >= 3) & (arrays['target_token'] < len(catalog) + 3)
            require(np.array_equal(known, known_recomputed), 'Known-target binding differs')
            require(int((~known).sum()) == 122 and int((known & ~coverage).sum()) == 616, 'Full unknown/offpool boundary differs')
            raw = catalog[pool - 3]
            require(np.array_equal(raw, arrays['candidate_raw_ids']), 'Full raw-ID/catalog mapping differs')
            catalog_size = len(catalog)
        else:
            raw = arrays['candidate_raw_ids']
            require(raw.shape == pool.shape and np.issubdtype(raw.dtype, np.integer) and bool((pool >= 3).all()), 'Raw candidate binding differs')
        if catalog_size is not None:
            require(isinstance(catalog_size, int) and catalog_size >= 1024, 'Invalid full catalog size')
        for row in range(n):
            require(len(np.unique(pool[row])) == len(np.unique(raw[row])) == 1024, 'Duplicate natural-pool ID')
        values, ranks, tops, orders = {}, {}, {}, {}
        for arm in ARMS:
            scores, top = arrays[arm + '_scores'], arrays[arm + '_top10']
            require(scores.shape == (n, 1024) and top.shape == (n, 10), 'Arm score/top10 shape differs')
            require(np.issubdtype(scores.dtype, np.floating) and np.issubdtype(top.dtype, np.integer), 'Arm dtype differs')
            require(bool(np.isfinite(scores).all()), 'Nonfinite policy score')
            values[arm], ranks[arm] = np.empty((n, 3), dtype=np.float64), []
            orders[arm] = np.empty((n, 10), dtype=np.int64)
            for row in range(n):
                if check and row % 128 == 0:
                    check(stage='scientific3-full-pool-ranks', arm=arm, row=row, total=n)
                order = np.lexsort((raw[row], -scores[row]))
                require(np.array_equal(pool[row, order[:10]], top[row]), 'Saved top10 differs ' + arm)
                rank = target_rank(pool[row, order], arrays['target_token'][row], bool(known[row]))
                ranks[arm].append(rank)
                values[arm][row] = row_metrics(rank)
                orders[arm][row] = order[:10]
            tops[arm] = np.take_along_axis(raw, orders[arm], axis=1)
            for j, metric in enumerate(METRICS):
                require(math.isclose(float(values[arm][:, j].mean()), evaluation['metrics'][arm][metric],
                                     abs_tol=1e-12, rel_tol=1e-12), 'Recomputed metric differs ' + arm + '/' + metric)
        differences = np.concatenate([values[right] - values[left] for left, right in PAIRS], axis=1)
        samples, users = bootstrap(np, arrays['user_id'], differences, check)
        ci = np.quantile(samples, [.025, .975], axis=0)
        frequencies = Counter(map(int, arrays['user_id']))
        weights = {'query_weighted': [1.] * n,
                   'equal_user_weighted': [1 / frequencies[int(user)] for user in arrays['user_id']]}
        summaries = {arm: {'metrics': {metric: float(values[arm][:, j].mean()) for j, metric in enumerate(METRICS)},
                          'rank_bins': {label: sum(rank_bin(rank, bool(known[row])) == label for row, rank in enumerate(ranks[arm]))
                                        for label in RANK_BINS},
                          'exposure': {name: exposure(tops[arm], weight, catalog_size) for name, weight in weights.items()}}
                     for arm in ARMS}
        comparisons = {}
        for a, (left, right) in enumerate(PAIRS):
            delta = values[right] - values[left]
            comparisons[right + '-minus-' + left] = {
                'left': left, 'right': right,
                'metrics': {metric: {'difference': float(delta[:, j].mean()),
                                     'user_cluster_95ci': list(map(float, ci[:, a * 3 + j])),
                                     'gains': int((delta[:, j] > 0).sum()), 'losses': int((delta[:, j] < 0).sum()),
                                     'ties': int((delta[:, j] == 0).sum())} for j, metric in enumerate(METRICS)},
                'delta_RR10_bins': {label: sum(delta_bin(float(value)) == label for value in delta[:, 0]) for label in DELTA_BINS},
                'top10_overlap_mean': math.fsum(len(set(map(int, x)) & set(map(int, y))) / 10
                    for x, y in zip(tops[left], tops[right])) / n}
        screen, decomposition, item_rows = (None, None, [])
        if stage == 'screen':
            screen, decomposition, item_rows = _screen(np, arrays, raw, orders, tops, weights, check)
            for arm in ARMS:
                expected = evaluation['diagnostics'][arm]
                actual = screen[arm]
                producer_top10 = producer_top10_raw_reward(np, arrays['raw_reward'], arrays['support'], orders[arm], parent_dtype)
                scientific_top10 = actual['utilities']['completion']['supported_top10_conditional_mean']
                producer_precision['by_arm'][arm] = {
                    'producer_float32_top10_mean': producer_top10,
                    'scientific_float64_top10_mean': scientific_top10,
                    'float64_minus_producer': scientific_top10 - producer_top10 if scientific_top10 is not None else None,
                    'expected_saved_producer_mean': expected['supported_top10_raw_reward']}
                comparisons_to_producer = {
                    'expected_effective_completion_utility': actual['utilities']['completion']['centered_supported_neutral_proxy'],
                    'expected_effective_residual_utility': actual['utilities']['residual']['centered_supported_neutral_proxy'],
                    'supported_conditional_raw_reward': actual['utilities']['completion']['supported_conditional_mean'],
                    'supported_probability_denominator': actual['utilities']['completion']['supported_probability_denominator'],
                    'supported_top10_raw_reward': producer_top10,
                    'supported_top10_denominator': actual['utilities']['completion']['supported_top10_denominator'],
                    'unsupported_mass': actual['unsupported_mass'], 'exact_KL': actual['exact_KL_to_C'],
                    'entropy': actual['entropy'],
                    'top1_max_share': summaries[arm]['exposure']['query_weighted']['top1_max_item_share'],
                    'top10_HHI': summaries[arm]['exposure']['query_weighted']['top10_hhi']}
                for key, value in comparisons_to_producer.items():
                    require((value is None and expected[key] is None) or
                            (value is not None and expected[key] is not None and
                             math.isclose(value, expected[key], abs_tol=1e-12, rel_tol=1e-12)),
                            'Recomputed screen diagnostic differs ' + arm + '/' + key)
            producer_precision['all_raw_export_values_roundtrip_float32'] = True
        report = {'contract': 'kion-scientific3-analysis-v1', 'state': 'completed', 'stage': stage,
            'source': {'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
            'protocol': protocol_pin, 'evaluation': evaluation_pin, 'checkpoints': evaluation['checkpoints'], 'rows': n, 'unique_users': users,
            'candidate_width': 1024, 'metric_denominator': n, 'unknown_target_misses': int((~known).sum()),
            'offpool_known_misses': int((known & ~coverage).sum()), 'candidate_coverage_count': int(coverage.sum()),
            'original_C_parity': evaluation['original_C_parity'], 'summaries': summaries, 'comparisons': comparisons,
            'screen_diagnostics': screen, 'item_decomposition': decomposition,
            'screen_diagnostic_precision': producer_precision,
            'checkpoint_curve': curve, 'checkpoint_curve_pins': curve_pins,
            'utility_name': evaluation.get('utility_name', 'residual'),
            'utility_metadata': evaluation.get('utility'),
            'source_channel_names': evaluation.get('candidate_source_columns'),
            'actor_source_columns': evaluation.get('actor_source_columns'),
            'bootstrap': {'replicates': REPLICATES, 'seed': SEED, 'batch_size': 32,
                'unit': 'unique users sampled with replacement; every original query follows its user',
                'estimand': 'query-weighted paired mean difference: resampled query-sum / resampled query-count',
                'CI': 'unadjusted percentile 2.5/97.5, NumPy linear quantiles; common draws for all nine columns',
                'numpy_version': np.__version__, 'multiple_comparison_adjustment': False},
            'definitions': {'utility_proxy': 'mean_s sum_i p_arm(i|s)*support(s,i)*(u(s,i)−E_C[u|supported,s]); unsupported advantage=0',
                'item_contribution': 'mean_s (p_right−p_left)*support*(u−E_C[u|supported,s]); sums exactly to paired proxy shift',
                'conditional_means': 'Reported with arm-specific probability/top10 denominators; differences are not comparable expected-utility gains.',
                'exposure': 'recommended unique top10 items per query; query weighting and equal-user weighting shown separately',
                'prior': 'frozen observed-fit item prior m_fit; residual is a proxy, not a causal personalization effect',
                'H63': 'membership in original observed truncated token prefix after excluding tokens <3; no generated actions',
                'popularity': 'log1p strict7d count at the original matched origin; no lifetime-popularity claim'},
            'limitations': [
                'Reused development, with screen selection/overlap; exploratory, not untouched holdout.',
                'One fit per utility, fixed beta; this is not a matched-KL comparison or beta sweep.',
                'Conditional on frozen encoder/C/local RM, natural1024 pools and RM support protocol.',
                'RM predictions and residual utility are offline proxies, not observed policy outcomes or causal retention effects.',
                'Full4096 has relevance/exposure only; reward diagnostics belong exclusively to screen256.',
                'Exposure is a recommendation distribution, not actual impressions, satisfaction, watchtime or retention.',
                'No model/provider/raw-data replay, supervised updates, final-test access or policy selection occurs in this analyzer.'],
            'seconds': time.monotonic() - started}
        output.mkdir(parents=True)
        write(output / 'analysis.json', report)
        # Plot data has no per-user identifiers. This is the portable level-A input.
        plotdata = {key: value for key, value in report.items() if key not in ('source', 'seconds')}
        write(output / 'plot-data.json', plotdata)
        with (output / 'per-query.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row_index', 'user_id', 'known', 'covered'] +
                            [arm + '_' + field for arm in ARMS for field in ('target_rank',) + METRICS])
            for row in range(n):
                writer.writerow([int(arrays['row_index'][row]), int(arrays['user_id'][row]), bool(known[row]), bool(coverage[row])] +
                                [x for arm in ARMS for x in [ranks[arm][row], *map(float, values[arm][row])]])
        with (output / 'paired-metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['comparison', 'metric', 'difference', 'ci_low', 'ci_high', 'gains', 'losses', 'ties', 'queries', 'users'])
            for name, comparison in comparisons.items():
                for metric, value in comparison['metrics'].items():
                    writer.writerow([name, metric, value['difference'], *value['user_cluster_95ci'],
                                     value['gains'], value['losses'], value['ties'], n, users])
        if item_rows:
            with (output / 'item-shifts.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['comparison', 'raw_id', 'probability_shift_per_query', 'supported_probability_shift_per_query',
                                 'completion_centered_neutral_contribution', 'residual_centered_neutral_contribution',
                                 'prior_centered_neutral_contribution'])
                writer.writerows(item_rows)
        return report
    finally:
        for value in loaded:
            mmap = getattr(value, '_mmap', None)
            if mmap is not None:
                mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    result = run(read(args.request))
    print(json.dumps({'state': result['state'], 'stage': result['stage'], 'rows': result['rows']}, ensure_ascii=False))
