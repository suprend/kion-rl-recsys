"""Independent stdlib CSV verifier. Actual campaign calls belong to the root worker.

This module does not import the analyzer or open arrays, checkpoints or datasets.
Ranks are input evidence, not independently rescored here. Bootstrap intervals
are checked for structure/provenance only, never independently resampled.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20261024, 20261025, 20261026)
METRICS = ('map10', 'ndcg10', 'recall10')
KINDS = ('completion-minus-C', 'residual-minus-C', 'residual-minus-completion')
TRAIN_CACHE_SHA = 'e76c365d1245355d5314e383ba53499800eb3c099fa0af7de106cced5fad15e5'
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def pin_fields(pin, label):
    require(isinstance(pin, dict) and isinstance(pin.get('path'), str), 'Missing pin: ' + label)
    path = Path(pin['path'])
    require(path.is_absolute(), 'Absolute pinned path required: ' + label)
    require(isinstance(pin.get('sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', pin['sha256']),
            'Invalid SHA256 pin: ' + label)
    return path.resolve()


def duplicate_safe_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant: ' + value)


def json_pin(pin, label):
    path = pin_fields(pin, label)
    require(path.suffix == '.json' and path.is_file() and path.stat().st_size <= 2 * 1024 * 1024,
            'Small JSON metadata required: ' + label)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'JSON digest differs: ' + label)
    if 'bytes' in pin:
        require(type(pin['bytes']) is int and pin['bytes'] == len(raw), 'JSON size differs: ' + label)
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=duplicate_safe_object, parse_constant=reject_constant)
    require(isinstance(value, dict), 'JSON object required: ' + label)
    return value, {'path': path.as_posix(), 'sha256': pin['sha256'], 'bytes': len(raw)}


def finite(value, label):
    require(type(value) in (float, int) and math.isfinite(value), 'Finite number required: ' + label)
    return float(value)


def close(actual, expected, label):
    actual, expected = finite(actual, label), finite(expected, label)
    require(abs(actual - expected) <= TOLERANCE, 'Numeric difference exceeds 1e-12: ' + label)


def same_pin(left, right, label):
    require(pin_fields(left, label) == pin_fields(right, label) and left['sha256'] == right['sha256'],
            'Metadata binding differs: ' + label)


def integer(value, label, minimum=0):
    require(isinstance(value, str) and re.fullmatch(r'-?(?:0|[1-9][0-9]*)', value),
            'Integer CSV value required: ' + label)
    parsed = int(value)
    require(parsed >= minimum, 'Integer outside domain: ' + label)
    return parsed


def boolean(value, label):
    require(value in ('True', 'False'), 'Boolean CSV value required: ' + label)
    return value == 'True'


def checkpoint_bindings(analysis, evaluation):
    require(analysis.get('contract') == 'kion-matched-kl-multiseed-analysis-v1' and
            analysis.get('state') == 'completed', 'Completed multiseed analysis required')
    require(evaluation.get('contract') == 'kion-matched-kl-multiseed-evaluation-v1' and
            evaluation.get('state') == 'completed', 'Completed multiseed evaluation metadata required')
    stage = analysis.get('stage')
    require(stage in ('screen', 'full4096') and evaluation.get('stage') == stage, 'Cohort stage differs')
    n = {'screen': 256, 'full4096': 4096}[stage]
    for label, doc in (('analysis', analysis), ('evaluation', evaluation)):
        require(type(doc.get('rows')) is int and type(doc.get('metric_denominator')) is int and
                doc['rows'] == doc['metric_denominator'] == n and doc.get('candidate_width') == 1024,
                'Fixed shared denominator differs: ' + label)
    require(analysis.get('seeds') == list(SEEDS), 'Fixed three seeds differ')
    require(analysis.get('checkpoints') == evaluation.get('checkpoints'), 'Analysis/evaluation checkpoints differ')
    checkpoints = analysis['checkpoints']
    require(isinstance(checkpoints, dict) and len(checkpoints) == 7 and 'C' in checkpoints,
            'Exactly C plus six fitted checkpoints required')
    require(all(isinstance(arm, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,79}', arm)
                for arm in checkpoints) and len({arm.casefold() for arm in checkpoints}) == 7,
            'Unsafe or colliding arm name')
    pairs = {seed: {} for seed in SEEDS}
    for arm, cp in checkpoints.items():
        pin_fields(cp, 'checkpoint metadata/' + arm)
        if arm != 'C':
            seed, utility = cp.get('seed'), cp.get('utility')
            require(type(seed) is int and seed in SEEDS and utility in ('completion', 'residual') and
                    cp.get('step') == 1000 and utility not in pairs[seed], 'Seed/utility/latest binding differs')
            pin_fields(cp.get('training'), 'training metadata/' + arm)
            pairs[seed][utility] = arm
    require(all(set(pair) == {'completion', 'residual'} for pair in pairs.values()), 'Missing paired seed arm')
    arms = ['C'] + [pairs[seed][utility] for seed in SEEDS for utility in ('completion', 'residual')]
    require(analysis.get('arms') == arms and evaluation.get('arms') == arms, 'Ordered arm bindings differ')
    require(set(analysis['summaries']) == set(evaluation['metrics']) == set(arms), 'Metric arm keys differ')
    require(evaluation.get('temperature') == 1 and
            evaluation.get('tie_break') == 'score descending, raw item ID ascending' and
            all(evaluation.get(key) is True for key in
                ('no_target_injection', 'no_seen_filter', 'unknown_and_offpool_are_zero')) and
            evaluation.get('final_test_reads') == 0, 'Evaluation protocol metadata differs')
    parity = evaluation.get('original_C_parity', {})
    require(analysis.get('original_C_parity') == parity and parity.get('max_abs_error') == 0 and
            parity.get('top10_changed_rows') == 0 and parity.get('score_tolerance_passed') is True and
            parity.get('changes_with_margin_above_tolerance') == 0, 'Exact C parity metadata differs')
    return stage, n, arms, pairs


def csv_metrics(pin, arms, n, check=None):
    path = pin_fields(pin, 'per_query')
    require(path.suffix == '.csv' and path.is_file() and path.stat().st_size <= 16 * 1024 * 1024,
            'Bounded per-query CSV required')
    require(digest(path) == pin['sha256'], 'CSV digest differs')
    size = path.stat().st_size
    if 'bytes' in pin:
        require(type(pin['bytes']) is int and pin['bytes'] == size, 'CSV size differs')
    header = ['row_index', 'user_id', 'known', 'covered'] + [
        arm + '_' + field for arm in arms for field in ('target_rank',) + METRICS]
    values = {arm: [] for arm in arms}
    identities, users, bindings = set(), set(), hashlib.sha256()
    unknown = offpool = covered_count = 0
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == header, 'Exact shared per-query CSV schema differs')
        count = 0
        for count, row in enumerate(reader, 1):
            require(count <= n and None not in row and None not in row.values(), 'CSV row width/count differs')
            if check and (count - 1) % 64 == 0:
                check(stage='independent-csv-metrics', row=count - 1, total=n)
            row_id = integer(row['row_index'], 'row_index')
            user = integer(row['user_id'], 'user_id')
            require(row_id not in identities, 'Duplicate shared row_index')
            identities.add(row_id)
            users.add(user)
            known = boolean(row['known'], 'known')
            covered = boolean(row['covered'], 'covered')
            require(not covered or known, 'Unknown target cannot be covered')
            unknown += not known
            offpool += known and not covered
            covered_count += covered
            bindings.update(json.dumps([row_id, user, known, covered], separators=(',', ':')).encode('ascii') + b'\n')
            for arm in arms:
                rank_text = row[arm + '_target_rank']
                if covered:
                    rank = integer(rank_text, arm + '/target_rank', 1)
                    require(rank <= 1024, 'Rank exceeds natural1024 pool')
                else:
                    require(rank_text == '', 'Unknown/offpool target must have an empty rank')
                    rank = None
                if rank is not None and rank <= 10:
                    derived = (1.0 / rank, 1.0 / math.log2(rank + 1.0), 1.0)
                else:
                    derived = (0.0, 0.0, 0.0)
                for metric, value in zip(METRICS, derived):
                    try:
                        saved = float(row[arm + '_' + metric])
                    except (TypeError, ValueError) as error:
                        raise ValueError('Invalid CSV metric: ' + arm + '/' + metric) from error
                    close(saved, value, 'CSV row ' + str(count) + '/' + arm + '/' + metric)
                    if not covered:
                        require(saved == 0, 'Unknown/offpool metric must be exactly zero')
                values[arm].append(derived)
        require(count == n, 'CSV fixed denominator differs')
    require(digest(path) == pin['sha256'], 'CSV changed during verification')
    return values, {'rows': n, 'unique_users': len(users), 'unknown_target_misses': unknown,
                    'offpool_known_misses': offpool, 'candidate_coverage_count': covered_count,
                    'shared_row_binding_sha256': bindings.hexdigest()}, {
                        'path': path.as_posix(), 'sha256': pin['sha256'], 'bytes': size}


def train_metadata(doc, label):
    require(doc.get('state') == 'completed' and doc.get('mode') == 'evaluate_train512' and
            doc.get('rows') == 512 and doc.get('cache_sha256') == TRAIN_CACHE_SHA and
            doc.get('final_test_reads') == doc.get('learning_updates') == 0 and
            doc.get('optimizer_created') is False and doc.get('update0_actor_reference_exact') is True and
            doc.get('update0_cached_scores_exact') is True and doc.get('update0_top10_exact') is True and
            doc.get('update0_max_abs_error') == 0, 'TRAIN512 diagnostic metadata differs: ' + label)
    require(isinstance(doc.get('results'), dict), 'TRAIN512 results absent: ' + label)


def matching_numbers(pin, analysis, pairs):
    receipt, receipt_pin = json_pin(pin, 'matching_receipt')
    require(receipt.get('contract') == 'completed-train-KL-match-binding-v1' and
            receipt.get('no_dev_selection') is True, 'Completed no-dev-selection matching receipt required')
    checkpoints = analysis['checkpoints']
    arm_names = set(checkpoints) - {'C'}
    require(receipt.get('selected_checkpoints') == {arm: checkpoints[arm]['sha256'] for arm in arm_names},
            'Matching selected checkpoint hashes differ')
    require(receipt.get('tolerance') == [.8, 1.2], 'Matching tolerance differs')
    target = finite(receipt.get('target_train_KL'), 'target_train_KL')
    require(target > 1e-8, 'Positive TRAIN KL calibration target required')
    kls = receipt.get('train_KL_by_arm')
    require(isinstance(kls, dict) and set(kls) == arm_names, 'Exactly six TRAIN KL scalars required')
    for arm, value in kls.items():
        require(finite(value, arm + '/TRAIN_KL') >= 0, 'Negative TRAIN KL scalar')
    cache, cache_pin = json_pin(receipt['cache'], 'TRAIN cache metadata')
    require(cache_pin['sha256'] == TRAIN_CACHE_SHA and cache.get('state') == 'completed' and
            cache.get('cohort') == 'train' and cache.get('rows') == 512 and cache.get('candidate_width') == 1024 and
            cache.get('temperature') == 1 and cache.get('no_target_injection') is True and
            cache.get('no_seen_filter') is True and cache.get('C_checkpoint_sha256') == checkpoints['C']['sha256'],
            'Frozen TRAIN512 cache metadata differs')
    reference, reference_pin = json_pin(receipt['reference'], 'TRAIN reference metadata')
    final, final_pin = json_pin(receipt['final_train_diagnostics'], 'six-arm TRAIN metadata')
    train_metadata(reference, 'reference')
    train_metadata(final, 'six arms')
    require(set(final['results']) == arm_names | {'original-C'}, 'Final TRAIN512 result arm set differs')
    for arm in arm_names:
        result = final['results'][arm]
        cp = result.get('checkpoint', {})
        same_pin(cp, checkpoints[arm], 'TRAIN checkpoint/' + arm)
        require(cp.get('name') == arm and cp.get('step') == 1000 and
                cp.get('utility') == checkpoints[arm]['utility'], 'TRAIN checkpoint descriptor differs: ' + arm)
        for utility in ('completion', 'residual'):
            metrics = result.get('metrics', {}).get(utility, {})
            require(metrics.get('rows') == 512 and finite(metrics.get('exact_KL'), arm + '/' + utility) == kls[arm],
                    'Matching scalar differs from six-arm TRAIN diagnostics: ' + arm)
    calibration_arm = pairs[SEEDS[0]]['completion']
    matches = [result for result in reference['results'].values() if isinstance(result, dict) and
               isinstance(result.get('checkpoint'), dict) and
               result['checkpoint'].get('sha256') == checkpoints[calibration_arm]['sha256']]
    require(len(matches) == 1, 'Exactly one reference calibration checkpoint required')
    same_pin(matches[0]['checkpoint'], checkpoints[calibration_arm], 'reference calibration checkpoint')
    require(finite(matches[0].get('metrics', {}).get('completion', {}).get('exact_KL'), 'reference KL') == target ==
            kls[calibration_arm], 'Calibration target differs from reference completion24 TRAIN KL')
    flags, ratios = {}, {}
    for seed in SEEDS:
        c, r = kls[pairs[seed]['completion']], kls[pairs[seed]['residual']]
        ratio = r / c if c > 0 else None
        ratios[str(seed)] = ratio
        flags[str(seed)] = ratio is not None and .8 <= ratio <= 1.2
    declared_flags = receipt.get('train_KL_match_by_seed')
    require(isinstance(declared_flags, dict) and set(declared_flags) == set(flags) and
            all(type(value) is bool for value in declared_flags.values()) and declared_flags == flags,
            'TRAIN KL per-seed match flags differ')
    require(receipt.get('ratio_by_seed') == ratios, 'TRAIN KL per-seed ratios differ')
    saved = analysis.get('training_KL_matching', {})
    same_pin(saved.get('receipt'), receipt_pin, 'analysis matching receipt')
    same_pin(saved.get('cache'), cache_pin, 'analysis TRAIN cache metadata')
    same_pin(saved.get('reference'), reference_pin, 'analysis TRAIN reference metadata')
    require(saved.get('receipt_metadata') == receipt and saved.get('train_KL_match_by_seed') == flags and
            type(saved.get('all_three_train_KL_matches')) is bool and
            saved['all_three_train_KL_matches'] == all(flags.values()) and
            saved.get('train_KL_residual_over_completion_by_seed') == ratios,
            'Analysis matching claims differ from six TRAIN KL scalars')
    distances = saved.get('train_KL_distance_from_calibration_target_by_arm')
    require(isinstance(distances, dict) and set(distances) == arm_names, 'Calibration distance arm keys differ')
    for arm in arm_names:
        close(distances[arm], kls[arm] - target, 'calibration distance/' + arm)
    return {'train_KL_by_arm': kls, 'train_KL_residual_over_completion_by_seed': ratios,
            'train_KL_match_by_seed': flags, 'all_three_train_KL_matches': all(flags.values()),
            'target_train_KL': target,
            'scope': 'Scalar consistency and checkpoint/cache/reference metadata bindings; KL was not recomputed from policies.'}, {
                'matching_receipt': receipt_pin, 'train_cache_metadata': cache_pin,
                'train_reference_metadata': reference_pin, 'six_arm_train_diagnostics': final_pin}


def mean(values):
    return math.fsum(values) / len(values)


def compare_numbers(analysis, evaluation, arms, pairs, values, n):
    summaries = {arm: {metric: mean([row[j] for row in values[arm]])
                       for j, metric in enumerate(METRICS)} for arm in arms}
    for arm in arms:
        require(set(analysis['summaries'][arm]['metrics']) == set(METRICS) and
                set(METRICS).issubset(evaluation['metrics'][arm]), 'Metric field keys differ: ' + arm)
        for metric in METRICS:
            close(analysis['summaries'][arm]['metrics'][metric], summaries[arm][metric], 'analysis/' + arm + '/' + metric)
            close(evaluation['metrics'][arm][metric], summaries[arm][metric], 'evaluation/' + arm + '/' + metric)
    plans = []
    for seed in SEEDS:
        c, r = pairs[seed]['completion'], pairs[seed]['residual']
        for kind, left, right in zip(KINDS, ('C', 'C', c), (c, r, r)):
            plans.append((str(seed) + '/' + kind, kind, seed, [(left, right)]))
    for kind in KINDS:
        plans.append(('three-seed-average/' + kind, kind, None,
                      [terms[0] for _, item_kind, _, terms in plans if item_kind == kind]))
    require(set(analysis['comparisons']) == {name for name, _, _, _ in plans}, 'Exactly twelve comparisons required')
    comparisons = {}
    for name, kind, seed, terms in plans:
        saved = analysis['comparisons'][name]
        require(saved.get('name') == name and saved.get('kind') == kind and saved.get('seed') == seed and
                saved.get('terms') == [list(term) for term in terms] and set(saved['metrics']) == set(METRICS),
                'Comparison bindings differ: ' + name)
        if seed is None:
            close(saved.get('equal_seed_weight'), 1 / 3, name + '/seed weight')
        else:
            require(saved.get('left') == terms[0][0] and saved.get('right') == terms[0][1],
                    'Comparison arm direction differs: ' + name)
        comparisons[name] = {}
        for j, metric in enumerate(METRICS):
            delta = [sum(values[right][row][j] - values[left][row][j] for left, right in terms) / len(terms)
                     for row in range(n)]
            derived = {'difference': mean(delta), 'gains': sum(x > 0 for x in delta),
                       'losses': sum(x < 0 for x in delta), 'ties': sum(x == 0 for x in delta)}
            declared = saved['metrics'][metric]
            close(declared.get('difference'), derived['difference'], name + '/' + metric)
            for field in ('gains', 'losses', 'ties'):
                require(type(declared.get(field)) is int and declared[field] == derived[field],
                        'Paired query sign counts differ: ' + name + '/' + metric + '/' + field)
            ci = declared.get('user_cluster_95ci')
            require(isinstance(ci, list) and len(ci) == 2, 'CI endpoints required: ' + name)
            low, high = (finite(endpoint, name + '/CI') for endpoint in ci)
            require(-1 <= low <= high <= 1, 'CI endpoints out of order/domain: ' + name)
            comparisons[name][metric] = derived
    require(set(analysis['seed_spread']) == set(analysis['prospective_two_seed_mean']) == set(KINDS),
            'Seed descriptive contrast keys differ')
    spreads, prospective = {}, {}
    for kind in KINDS:
        require(set(analysis['seed_spread'][kind]) == set(analysis['prospective_two_seed_mean'][kind]) == set(METRICS),
                'Seed descriptive metric keys differ')
        spreads[kind], prospective[kind] = {}, {}
        for metric in METRICS:
            three = [comparisons[str(seed) + '/' + kind][metric]['difference'] for seed in SEEDS]
            average = mean(three)
            derived = {'mean': average, 'min': min(three), 'max': max(three),
                       'sample_std': math.sqrt(math.fsum((x - average) ** 2 for x in three) / 2)}
            declared = analysis['seed_spread'][kind][metric]
            require(type(declared.get('seed_count')) is int and declared['seed_count'] == 3, 'Seed count differs')
            for field, value in derived.items():
                close(declared.get(field), value, kind + '/' + metric + '/' + field)
            prospective[kind][metric] = mean(three[1:])
            close(analysis['prospective_two_seed_mean'][kind][metric], prospective[kind][metric],
                  'prospective mean/' + kind + '/' + metric)
            spreads[kind][metric] = derived
    return summaries, comparisons, spreads, prospective


def run(request, check=None):
    if check:
        check(stage='independent-csv-verification-start')
    analysis, analysis_pin = json_pin(request['analysis'], 'analysis')
    evaluation, evaluation_pin = json_pin(analysis['evaluation'], 'evaluation metadata')
    stage, n, arms, pairs = checkpoint_bindings(analysis, evaluation)
    folder = ROOT / 'interpretation' / {'screen': 'screen-r1', 'full4096': 'full-r1'}[stage]
    require(Path(analysis_pin['path']) == (folder / 'analysis.json').resolve() and
            pin_fields(request['per_query'], 'per_query') == (folder / 'per-query.csv').resolve(),
            'Stage analysis/CSV paths differ')
    require(pin_fields(request['matching_receipt'], 'matching_receipt') == (ROOT / 'protocols/matching.json').resolve(),
            'Campaign matching receipt path differs')
    output = Path(request.get('output', ROOT / 'review' / ('numeric-' + stage + '.json'))).resolve()
    require(output == (ROOT / 'review' / ('numeric-' + stage + '.json')).resolve() and not output.exists(),
            'Fresh exact campaign numeric review output required')
    bootstrap = analysis.get('bootstrap', {})
    require(bootstrap.get('replicates') == 2000 and bootstrap.get('seed') == 20261024 and
            bootstrap.get('batch_size') == 32 and bootstrap.get('columns') == 36 and
            bootstrap.get('common_draw_schedule') is True and bootstrap.get('multiple_comparison_adjustment') is False and
            bootstrap.get('CI') == 'unadjusted percentile 2.5/97.5, NumPy linear quantiles' and
            bootstrap.get('unit') == 'unique user clusters sampled with replacement; all queries follow their user' and
            bootstrap.get('estimand') == 'query-weighted ratio of resampled cluster sums to resampled query counts' and
            bootstrap.get('seed_average') == 'equal weight 1/3 on the three fitted heads, before user resampling',
            'Bootstrap provenance metadata differs')
    require(analysis.get('primary_comparison') == 'three-seed-average/residual-minus-completion' and
            analysis.get('primary_metric') == 'map10' and analysis.get('secondary_metrics') == ['ndcg10', 'recall10'],
            'Primary/secondary comparison metadata differs')
    matching, matching_pins = matching_numbers(request['matching_receipt'], analysis, pairs)
    values, counts, per_query_pin = csv_metrics(request['per_query'], arms, n, check)
    for key in ('unique_users', 'unknown_target_misses', 'offpool_known_misses', 'candidate_coverage_count'):
        require(type(analysis.get(key)) is int and analysis[key] == counts[key], 'CSV cohort count differs: ' + key)
    summaries, comparisons, spreads, prospective = compare_numbers(analysis, evaluation, arms, pairs, values, n)
    if check:
        check(stage='independent-csv-verification-complete', rows=n)
    pins = {'analysis': analysis_pin, 'per_query': per_query_pin, 'evaluation_metadata': evaluation_pin, **matching_pins}
    for label, pin in pins.items():
        require(digest(pin['path']) == pin['sha256'], 'Pinned input changed before proof: ' + label)
    report = {'contract': 'kion-independent-csv-numeric-verification-v1', 'state': 'completed', 'verified': True,
              'stage': stage, 'rows': n, 'metric_denominator': n, 'candidate_width': 1024,
              'verified_utc': datetime.now(timezone.utc).isoformat(),
              'source': {'path': Path(__file__).resolve().as_posix(), 'sha256': digest(__file__)},
              'inputs': pins, 'arms': arms, 'seeds': list(SEEDS), 'cohort': counts,
              'absolute_tolerance': TOLERANCE, 'independent_metric_recomputation': True,
              'summaries': summaries, 'comparisons': comparisons, 'seed_spread': spreads,
              'prospective_two_seed_mean': prospective, 'training_KL_matching': matching,
              'bootstrap_intervals_independently_resampled': False,
              'bootstrap_interval_validation': 'Finite ordered endpoints within [-1,1] and declared 2000-draw user-cluster provenance checked; point containment is not required.',
              'claim_scope': 'Independent arithmetic from the pinned per-query target ranks: seven-arm MAP/NDCG/Recall, nine seed contrasts, three equal-seed average contrasts, paired signs, fitted-seed spread, and prospective two-seed means. TRAIN KL scalar/flag consistency checked against six-arm pinned diagnostic metadata.',
              'limitations': ['Target ranks, row/user identities and masks are CSV input evidence; scoring, ranking, tie-breaking and bindings to arrays are not independently reproduced.',
                              'No arrays, checkpoint bytes, raw datasets, models, training, RM or final test were opened by this verifier.',
                              'Cache/reference/checkpoint pins are metadata bindings; array integrity and TRAIN KL values are not independently recomputed.',
                              'Conditional user-cluster confidence intervals are not independently resampled here; only endpoint structure and declared provenance are checked.',
                              'Three fitted seeds and the two prospective seeds are descriptive; this proof supplies no training-seed population interval or causal claim.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'), object_pairs_hook=duplicate_safe_object,
                         parse_constant=reject_constant)
    result = run(request)
    print(json.dumps({key: result[key] for key in ('state', 'verified', 'stage', 'rows')}))
