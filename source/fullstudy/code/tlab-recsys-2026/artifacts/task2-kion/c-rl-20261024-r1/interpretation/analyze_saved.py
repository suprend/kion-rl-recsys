"""I0 saved-output audit. Imports no ML/provider. Real execution belongs to root.

All ranking denominators include unknown/offpool targets. Token IDs map to raw
IDs through catalog_raw_ids[token-3]. Ranking ties use ascending raw item ID.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import time

ARMS = ('A', 'B', 'C', 'GPT', 'RRF')
BASE = Path('D:/tlab-recsys-2026/artifacts/task2-kion')
SELECTION_RULE = ('C-B gain, C-B loss, C-GPT gain, C-GPT loss; three unique rows '
                  'per cell, SHA256(c3-I0-cases-v1|row_index) ascending, '
                  'cell order as written; exclude previously chosen rows; '
                  'gain/loss uses reciprocal-rank@10; no outcome-based replacement')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def pin(path):
    path = Path(path).resolve()
    return {'path': path.as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def metric(rank):
    return {'map10': 1 / rank if 0 < rank <= 10 else 0.,
            'recall10': float(0 < rank <= 10),
            'ndcg10': 1 / math.log2(rank + 1) if 0 < rank <= 10 else 0.}


def describe(values):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return {'count': 0}
    return {'count': len(ordered), 'mean': math.fsum(ordered) / len(ordered),
            'min': ordered[0], 'p05': ordered[int(.05 * (len(ordered)-1))],
            'median': ordered[(len(ordered)-1)//2],
            'p95': ordered[int(.95 * (len(ordered)-1))], 'max': ordered[-1]}


def exposure(tops, weights, catalog_size):
    require(len(tops) == len(weights) and bool(tops), 'Empty/misaligned exposure')
    counts, first = Counter(), Counter()
    for items, weight in zip(tops, weights):
        require(len(items) == 10 and len(set(items)) == 10, 'Invalid top10')
        counts.update({int(item): weight for item in items})
        first[int(items[0])] += weight
    total = math.fsum(weights)
    hhi = math.fsum((value / (10 * total)) ** 2 for value in counts.values())
    return {'top1_max_item_share': max(first.values()) / total,
            'top1_most_frequent_token': min(first, key=lambda x: (-first[x], x)),
            'top10_hhi': hhi, 'effective_catalog_items': 1 / hhi,
            'unique_top10_items': len(counts),
            'catalog_coverage': len(counts) / catalog_size,
            'within_list_duplicate_fraction': 0.,
            'meaning': 'recommendation exposure, not actual impressions or watches'}


def compare_rows(records, reference):
    deltas = [row['C_rr10'] - row[reference + '_rr10'] for row in records]
    return {'denominator': len(records), 'gains': sum(x > 0 for x in deltas),
            'losses': sum(x < 0 for x in deltas), 'ties': sum(x == 0 for x in deltas),
            'mean_rr10_difference': math.fsum(deltas) / len(deltas),
            'target_rank_change_covered': describe([row[reference + '_rank'] - row['C_rank']
                                                   for row in records if row['covered']]),
            'mean_absolute_all_candidate_rank_displacement':
                describe([row['C_minus_' + reference + '_mean_abs_rank_shift'] for row in records]),
            'top10_overlap_fraction': describe([row['C_' + reference + '_top10_overlap'] for row in records])}


def select_cases(records):
    chosen, used, cells = [], set(), []
    for ref, direction in (('B', 'gain'), ('B', 'loss'), ('GPT', 'gain'), ('GPT', 'loss')):
        candidates = [row for row in records if row['row_index'] not in used and
                      ((row['C_rr10'] > row[ref + '_rr10']) if direction == 'gain' else
                       (row['C_rr10'] < row[ref + '_rr10']))]
        candidates.sort(key=lambda row: (hashlib.sha256(
            ('c3-I0-cases-v1|' + str(row['row_index'])).encode()).hexdigest(), row['row_index']))
        selected = candidates[:3]
        for row in selected:
            used.add(row['row_index'])
            chosen.append({'cell': 'C-' + ref + '-' + direction, 'row': row})
        cells.append({'cell': 'C-' + ref + '-' + direction, 'eligible_after_prior_cells': len(candidates),
                      'selected': len(selected), 'shortfall': 3-len(selected)})
    return {'rule': SELECTION_RULE, 'cells': cells, 'cases': chosen,
            'representativeness': 'fixed case studies enriched by outcome; not a population sample',
            'history_length_support_stratification': 'unavailable without validated saved history join'}


def old_audit(root):
    results = []
    for name in ('run-rl-r1', 'run-rl-recovered-r1'):
        worker = root / 'rl' / name / 'worker'
        path = worker / 'training' / 'screening.json'
        require(path.is_file(), 'Missing old screening receipt: ' + path.as_posix())
        doc = read(path)
        states = {entry['update']: entry['evaluation'] for entry in doc['state']['history']}
        require(0 in states and 1000 in states, 'Old run lacks update0/latest1000')
        require(states[0]['cohort_sha256'] == states[1000]['cohort_sha256'], 'Old cohort mismatch')
        require(states[0]['protocol'] == states[1000]['protocol'], 'Old protocol mismatch')
        selected = doc['state']['selected_update']
        require(selected in states, 'Missing selected screen state')
        # Metadata inventory, never load old feature/model/checkpoint arrays.
        output_files = sorted(path.as_posix() for path in worker.rglob('*') if path.is_file()
                              and ('top10' in path.name.lower() or 'scores' in path.name.lower()))
        results.append({'run': name, 'screening': pin(path), 'updates': doc['state']['updates'],
                        'selected_update': selected, 'update0': states[0], 'trained_latest1000': states[1000],
                        'selected': states[selected], 'latest_minus_update0_ndcg10':
                            states[1000]['policy']['ndcg@10'] - states[0]['policy']['ndcg@10'],
                        'selected_is_update0_by_receipt': selected == 0,
                        'selected_behavior_identity': 'implied by selection receipt; no new tensor/model replay',
                        'saved_policy_top10_score_file_inventory': output_files,
                        'additional_behavior_metrics': {'state': 'unavailable',
                            'reason': 'No complete matched update0/latest1000 policy predictions saved; '
                                      'screen-cache reference/base are not trained-policy outputs.'},
                        'reward_hacking': 'not established; heldout reward/proxy diagnostic unavailable',
                        'comparison_scope': 'old frozen unforced128 own512 screen; not current C natural1024'})
    return results


def rm_reuse(root):
    provenance_path = root / 'reports' / 'interpretation-provenance.json'
    request_path = root / 'auxiliary' / 'request-rm-r2.json'
    provenance, request = read(provenance_path), read(request_path)
    payload = root / 'auxiliary' / 'run-rm-r2' / 'result' / 'rm-interpretation.json'
    report = root / 'reports' / 'rm-interpretation.md'
    require(sha(payload) == provenance['sources_sha256'][payload.as_posix()], 'RM payload SHA differs')
    require(sha(report) == provenance['outputs_sha256'][report.name], 'RM report SHA differs')
    scorer = BASE / 'ready-pack-20261020' / 'rm-r1' / 'scorer' / 'manifest.json'
    require(sha(scorer) == request['input_pins'][scorer.as_posix()], 'Ready RM manifest SHA differs')
    rows = Path('D:/tlab-recsys-2026/data/processed/kion-reward-ready-20261020-r1/manifest.json')
    require(sha(rows) == request['input_pins'][rows.as_posix()], 'RM rows manifest SHA differs')
    return {'state': 'reused_closed_report_hash_verified_no_SHAP_or_model_load',
            'sources': [pin(x) for x in (provenance_path, request_path, payload, report, scorer, rows)],
            'scope': 'ready RM raw CatBoost output before calibration/clipping/fallback; not C or RL',
            'sample': '256 SHA-selected supported select rows Aug2-8; histories frozen before Aug2',
            'schema': '11227 numeric + 7 categorical; 24 namespaces',
            'excluded_unsupported_select_targets': 605,
            'leading_groups': ['candidate:local_dynamic', 'context:history'],
            'group_statistics': 'report preserves sum mean|column SHAP| and mean|sum group SHAP|',
            'limitations': 'selected supported observed watched pairs; not causal response/retention or untouched test'}


def load_arrays(np, manifest_path, check=None):
    doc = read(manifest_path)
    require(doc['state'] == 'completed' and doc['rows'] == 4096 and doc['candidate_width'] == 1024,
            'Not pinned completed4096x1024 baseline')
    require(doc['no_seen_filter'] and doc['no_target_injection'] and doc['unknown_and_offpool_are_zero'],
            'Candidate/denominator contract differs')
    require(doc['tie_break'] == 'score descending, raw item ID ascending', 'Tie policy differs')
    require(doc['metric_denominator'] == doc['rows'] and doc['source_columns'] == [12, 13], 'Head/denominator differs')
    require(set(doc['head_receipts']) == {'A', 'B', 'C'} and
            all(receipt == doc['head_receipts']['A'] for receipt in doc['head_receipts'].values()),
            'Head lineage receipts disagree')
    cleanup = read(manifest_path.with_name('cleanup.json'))
    require(cleanup.get('evaluation_completed') is True and
            cleanup.get('evidence', {}).get('teardown_completed') is True and
            cleanup.get('evidence', {}).get('errors') == [], 'Evaluation teardown not closed')
    arrays, sources = {}, []
    names = ('row_index', 'user_id', 'target_token', 'target_date', 'target_item_id',
             'known_target', 'candidate_coverage', 'catalog_raw_ids', 'pool_ids')
    for name in names + tuple(arm + suffix for arm in ARMS for suffix in ('_scores', '_top10')):
        if check:
            check(phase='saved-array-integrity', array=name)
        descriptor = doc['arrays'][name]
        path = manifest_path.parent / descriptor['path']
        require(path.resolve().parent == manifest_path.parent.resolve(), 'Unexpected array path')
        require(path.stat().st_size == descriptor['bytes'] and sha(path) == descriptor['sha256'],
                'Array identity differs: ' + name)
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(list(array.shape) == descriptor['shape'] and str(array.dtype) == descriptor['dtype'],
                'Array schema differs: ' + name)
        arrays[name] = array
        sources.append({'name': name, **descriptor})
    return doc, arrays, sources


def content_catalog(path):
    if path is None:
        return None
    cards = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item = int(row['item_id'])
            require(item not in cards, 'Duplicate item card')
            cards[item] = {key: row.get(key, '') for key in ('content_type', 'title')}
    return cards


def analyze(args, check=None):
    started = time.monotonic()
    manifest_path = args.evaluation.resolve()
    require(sha(manifest_path) == args.evaluation_sha256, 'Evaluation manifest SHA differs')
    output = args.output.resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent), 'Output must remain inside interpretation directory')
    for name in ('score-audit.json', 'cases.json', 'per-query.csv', 'REPORT.md'):
        require(not (output/name).exists(), 'Refusing to overwrite existing output '+name)
    if args.items_csv:
        require(bool(args.items_csv_sha256) and sha(args.items_csv) == args.items_csv_sha256, 'Items catalog SHA differs/missing')
    import numpy as np  # Runtime-only; synthetic kernels need no NumPy/ML dependency.
    doc, arrays, sources = load_arrays(np, manifest_path, check)
    n, width = doc['rows'], doc['candidate_width']
    require(len(set(map(int, arrays['row_index']))) == n, 'Duplicate query row ID')
    raw = arrays['catalog_raw_ids']
    require(len(set(map(int, raw))) == len(raw), 'Duplicate catalog raw ID')
    cards = content_catalog(args.items_csv)
    tops = {arm: [] for arm in ARMS}
    records, all_case_detail = [], {}
    residual_n, residual_sum, residual_square = 0, 0., 0.
    residual_min, residual_max = math.inf, -math.inf
    for index in range(n):
        if check and index % 64 == 0:
            check(phase='saved-score-audit', completed_rows=index, rows=n)
        pool = arrays['pool_ids'][index]
        target, known = int(arrays['target_token'][index]), bool(arrays['known_target'][index])
        require(known == (3 <= target < len(raw)+3), 'Known mask differs')
        require(len(np.unique(pool)) == width and bool(np.all((pool >= 3) & (pool < len(raw)+3))),
                'Duplicate/invalid pool')
        covered = known and bool(np.any(pool == target))
        require(covered == bool(arrays['candidate_coverage'][index]), 'Coverage differs')
        require(not known or int(arrays['target_item_id'][index]) == int(raw[target-3]), 'Target raw/token differs')
        positions = {int(token): j for j, token in enumerate(pool)}
        target_pos = positions.get(target) if covered else None
        row = {'row_index': int(arrays['row_index'][index]), 'user_id': int(arrays['user_id'][index]),
               'target_token': target, 'target_item_id': int(arrays['target_item_id'][index]),
               'target_date': int(arrays['target_date'][index]), 'known': known, 'covered': covered,
               'outcome_class': 'unknown' if not known else ('covered' if covered else 'offpool')}
        ranks, orders = {}, {}
        for arm in ARMS:
            score = arrays[arm + '_scores'][index]
            require(bool(np.isfinite(score).all()), 'Nonfinite saved scores')
            order = np.lexsort((raw[pool-3], -score))
            top = pool[order[:10]]
            require(bool(np.array_equal(top, arrays[arm+'_top10'][index])), 'Saved top10 mismatch ' + arm)
            rank = np.empty(width, dtype=np.int32)
            rank[order] = np.arange(1, width+1)
            ranks[arm], orders[arm] = rank, order
            target_rank = int(rank[target_pos]) if covered else 0
            row[arm+'_rank'], row[arm+'_rr10'] = target_rank, metric(target_rank)['map10']
            row[arm+'_boundary_margin_10_11'] = float(score[order[9]]) - float(score[order[10]])
            tops[arm].append(list(map(int, top)))
        residual = arrays['C_scores'][index].astype(np.float64) - arrays['GPT_scores'][index].astype(np.float64)
        residual_n += width
        residual_sum += float(residual.sum())
        residual_square += float(np.dot(residual, residual))
        residual_min, residual_max = min(residual_min, float(residual.min())), max(residual_max, float(residual.max()))
        row['residual_candidate_mean'] = float(residual.mean())
        for ref in ('GPT', 'B'):
            row['C_minus_'+ref+'_mean_abs_rank_shift'] = float(np.abs(ranks['C']-ranks[ref]).mean())
            row['C_'+ref+'_top10_overlap'] = len(set(tops['C'][-1]) & set(tops[ref][-1])) / 10
        records.append(row)
        # Small per-query details, never copy a full candidate-score matrix.
        details = {'top10_raw_ids': {arm: [int(raw[token-3]) for token in tops[arm][-1]] for arm in ARMS}}
        if covered:
            competitor = int(orders['C'][9])
            if competitor == target_pos:
                competitor = int(orders['C'][10])
            gpt_margin = float(arrays['GPT_scores'][index, target_pos]) - float(arrays['GPT_scores'][index, competitor])
            full_margin = float(arrays['C_scores'][index, target_pos]) - float(arrays['C_scores'][index, competitor])
            delta = float(residual[target_pos]-residual[competitor])
            require(math.isclose(gpt_margin+delta, full_margin, rel_tol=1e-10, abs_tol=1e-10), 'Residual arithmetic failed')
            details['target_boundary_pair'] = {'competitor_raw_id': int(raw[int(pool[competitor])-3]),
                 'competitor_rule': 'C rank10, or rank11 if target occupies rank10',
                 'GPT_margin': gpt_margin, 'residual_margin': delta, 'C_margin': full_margin,
                 'strict_sign_flip_by_residual': gpt_margin * full_margin < 0,
                 'target_GPT_score': float(arrays['GPT_scores'][index, target_pos]),
                 'target_saved_residual_C_minus_GPT': float(residual[target_pos]),
                 'target_C_score': float(arrays['C_scores'][index, target_pos])}
        all_case_detail[row['row_index']] = details
    frequencies = Counter(map(int, arrays['user_id']))
    query_weights = [1.] * n
    user_weights = [1/frequencies[int(user)] for user in arrays['user_id']]
    arms = {}
    for arm in ARMS:
        summary = {key: math.fsum(metric(row[arm+'_rank'])[key] for row in records)/n
                   for key in ('map10', 'recall10', 'ndcg10')}
        for key, value in summary.items():
            require(math.isclose(value, doc['metrics'][arm][key], abs_tol=1e-12, rel_tol=1e-12), 'Metric mismatch '+arm+key)
        content = {'state': 'unavailable', 'reason': 'No pinned items catalog provided'}
        if cards is not None:
            types = Counter()
            diversity = []
            for items in tops[arm]:
                row_types = [cards.get(int(raw[token-3]), {}).get('content_type', '') for token in items]
                row_types = [x if x in ('film', 'series') else 'missing_or_unknown' for x in row_types]
                types.update(row_types)
                diversity.append(len(set(row_types) - {'missing_or_unknown'}))
            content = {'state': 'available', 'recommendation_slots': dict(types),
                       'fractions': {key: value/(10*n) for key, value in types.items()},
                       'valid_type_count_per_list': describe(diversity),
                       'temporal_assumption': 'static items snapshot assumed valid at origin; dates not established'}
        arms[arm] = {**summary, 'query_weighted': exposure(tops[arm], query_weights, len(raw)),
                     'equal_user_weighted': exposure(tops[arm], user_weights, len(raw)), 'content_type': content}
    cases = select_cases(records)
    for case in cases['cases']:
        case.update(all_case_detail[case['row']['row_index']])
        if cards:
            case['target_title'] = cards.get(case['row']['target_item_id'], {}).get('title')
            case['C_top10_titles'] = [cards.get(item, {}).get('title') for item in case['top10_raw_ids']['C']]
    mean = residual_sum/residual_n
    report = {'contract': 'C3-saved-output-I0-v1', 'state': 'completed',
              'seconds': time.monotonic()-started, 'source': pin(Path(__file__)), 'evaluation': pin(manifest_path),
              'source_arrays': sources, 'lineage': {key: doc[key] for key in
                    ('frozen_teacher', 'feature_schema_sha256', 'head_receipts', 'selection', 'candidate_contract')},
              'rows': n, 'unique_users': len(frequencies), 'candidate_width': width,
              'target_outcome_counts': dict(Counter(row['outcome_class'] for row in records)),
              'tie_policy': doc['tie_break'], 'full_denominator': n, 'arms': arms,
              'comparisons': {'C-'+ref: compare_rows(records, ref) for ref in ('B', 'GPT')},
              'saved_residual_C_minus_GPT': {'scope': 'difference of saved float scores, not raw feature attribution',
                    'candidate_count': residual_n, 'mean': mean,
                    'std': math.sqrt(max(0., residual_square/residual_n-mean*mean)),
                    'min': residual_min, 'max': residual_max,
                    'per_query_means': describe([row['residual_candidate_mean'] for row in records])},
              'optional_sources': {'items_csv': pin(args.items_csv) if args.items_csv else None},
              'unavailable': {'repeat_in_observed_preorigin_history': 'No validated row-H63/rawID/date join supplied',
                    'novelty_and_head_tail': 'No pinned train-only/pre-origin counts; missing is not zero',
                    'runtime': 'No trustworthy catalog runtime supplied; target duration/watchpct not used',
                    'RM_reward': 'Saved score audit does not serve RM; current compatible RM handled separately',
                    'policy_KL_entropy': 'No policy normalization/temperature binding requested; no softmax guessed',
                    'raw_feature_or_neuron_attribution': 'I0 contains only saved scores; no model/provider access'},
              'old_RL_own_cohort_audit': old_audit(args.old_root), 'ready_RM_SHAP_reuse': rm_reuse(args.old_root),
              'limits': ['One seed/reused development cohort; no untouched test or long-term causal claims',
                         'C-B compares independently fitted heads; not decomposition of C features',
                         'Case studies cannot establish population feature importance'],
              'new_model_feature_or_SHAP_execution': False}
    output.mkdir(parents=True, exist_ok=True)
    write(output/'score-audit.json', report)
    write(output/'cases.json', cases)
    with (output/'per-query.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    lines = ['# C3 saved-output interpretation', '',
             'Saved4096-query natural1024 pool; frozen E2, A/B/C trained262144 examples×3epochs. '
             'Unknown/offpool remain misses. No new model, features, SHAP, or test reads.', '',
             '| Arm | MAP@10 | NDCG@10 | Recall@10 | top1 max share | top10 HHI | effective items | unique items |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, value in arms.items():
        e = value['query_weighted']
        lines.append(f"| {arm} | {value['map10']:.6f} | {value['ndcg10']:.6f} | {value['recall10']:.6f} | "
                     f"{e['top1_max_item_share']:.6f} | {e['top10_hhi']:.6f} | {e['effective_catalog_items']:.1f} | {e['unique_top10_items']} |")
    lines.extend(['', f"Retrieval boundary: {report['target_outcome_counts']}. Full denominator={n}.", ''])
    for key, value in report['comparisons'].items():
        lines.append(f"{key}: {value['gains']} gains, {value['losses']} losses, {value['ties']} ties; "
                     f"mean RR@10 difference {value['mean_rr10_difference']:+.6f}. "
                     f"Mean target rank improvement on covered rows {value['target_rank_change_covered']['mean']:+.2f}.")
    lines += ['', 'Equal-user exposure, full rank-displacement/margin summaries, residual arithmetic, '
              'content-type coverage and source pins: [score-audit.json](score-audit.json). '
              'All query outcomes: [per-query.csv](per-query.csv).', '',
              'Twelve fixed case studies (or explicitly recorded cell shortfall): [cases.json](cases.json). '
              + SELECTION_RULE + '. These outcome-enriched cases are not a population sample.', '']
    for old in report['old_RL_own_cohort_audit']:
        lines.append(f"Old {old['run']}: own512-query fixed128 screen NDCG "
                     f"{old['update0']['policy']['ndcg@10']:.6f}→{old['trained_latest1000']['policy']['ndcg@10']:.6f}; "
                     f"selected update={old['selected_update']}. Additional matched policy exposure unavailable; "
                     'saved reference/base cache does not recover latest policy. Proxy reward unavailable; reward hacking is not established.')
    lines += ['', 'Ready RM SHAP is reused with verified payload/report/scorer/rows SHA lineage. '
              'Its256 supported select rows,11227numeric+7categorical inputs and raw tree-output scale '
              'explain that RM only. Leading groups: candidate:local_dynamic and context:history. '
              'Report preserves both sum-absolute-column and absolute-net-group measures; '
              '605 unsupported select targets were excluded. It does not explain C or causal history benefit.', '',
              'Unavailable: history repeat, novelty/head-tail counts and trustworthy runtime; '
              'missing data were not imputed as zero. No raw-feature or neuron explanation was computed. '
              'Static catalog type metadata, if supplied, has an explicit temporal validity assumption. '
              'This reused development cohort and one training seed do not establish future-week quality or retention.', '']
    (output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return report


def run(request, check=None):
    """Native-wrapper seam; one bounded CPU job, no subprocess/ML/provider.

    Required: evaluation, evaluation_sha256, output. Optional: old_root,
    items_csv + items_csv_sha256. check receives phase/progress keyword fields.
    """
    values = {'evaluation': Path(request['evaluation']),
              'evaluation_sha256': request['evaluation_sha256'],
              'output': Path(request['output']),
              'old_root': Path(request.get('old_root', BASE/'night-20261023-r1')),
              'items_csv': Path(request['items_csv']) if request.get('items_csv') else None,
              'items_csv_sha256': request.get('items_csv_sha256')}
    return analyze(argparse.Namespace(**values), check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--evaluation-sha256', required=True)
    parser.add_argument('--old-root', type=Path, default=BASE/'night-20261023-r1')
    parser.add_argument('--items-csv', type=Path)
    parser.add_argument('--items-csv-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    require(args.execute, 'Real arrays are root-owned; supply --execute only from approved root launch')
    result = analyze(args)
    print(json.dumps({'state': result['state'], 'rows': result['rows'], 'seconds': result['seconds']}))


if __name__ == '__main__':
    main()
