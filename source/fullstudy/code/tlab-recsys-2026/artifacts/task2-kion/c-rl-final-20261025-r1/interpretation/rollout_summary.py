"""Strict saved-JSON five-step rollout summary. Stdlib only; root runs real data."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

HEADS = ('C', 'RL-completion', 'RL-residual')
MODES = ('greedy', 'categorical')
MASKS = ('none', 'exclude-generated')
STEPS = 5
POSITIONS = [253, 2, 30, 246, 204, 158, 96, 50, 128, 244, 255, 36, 12, 109, 171, 236]
ORIGINAL_ROWS = [158311, 1031, 20183, 153529, 127633, 101332, 55626, 29840,
                 79744, 153281, 158658, 22276, 7498, 62937, 108572, 147886]
STEP_METRICS = ('kl_nats', 'entropy_nats', 'unsupported_mass', 'expected_neutral_proxy',
                'expected_supported_reward_conditional', 'expected_supported_residual_conditional',
                'expected_supported_prior_conditional', 'expected_neutral_residual_proxy',
                'q_l2_drift', 'base_rmse_drift', 'selected_probability', 'selected_reward',
                'selected_residual', 'selected_fit_prior', 'selected_neutral_proxy', 'selected_neutral_residual_proxy')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (float, int) and math.isfinite(value), 'Nonfinite/nonnumeric '+name)
    return float(value)


def close(a, b):
    return math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=1e-7)


def mean(values):
    return math.fsum(values)/len(values) if values else None


def mean_with_denominator(values, total):
    valid = [float(value) for value in values if value is not None]
    return {'mean': mean(valid), 'denominator': len(valid), 'total': total}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pinned(entry, maximum_bytes=None):
    path = Path(entry['path']).resolve()
    require(path.is_file(), 'Missing source '+path.as_posix())
    size = path.stat().st_size
    require(maximum_bytes is None or size <= maximum_bytes, 'Source exceeds bounded JSON size')
    require(sha(path) == entry['sha256'], 'Source SHA differs '+path.as_posix())
    return path, {'path': path.as_posix(), 'sha256': entry['sha256'], 'bytes': size}


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def exposure(slates):
    counts = Counter(item for slate in slates for item in slate)
    first = Counter(slate[0] for slate in slates)
    require(bool(slates) and all(len(slate) == STEPS for slate in slates), 'Invalid slate exposure')
    denominator = STEPS*len(slates)
    hhi = math.fsum((count/denominator)**2 for count in counts.values())
    return {'slates': len(slates), 'recommendation_slots': denominator,
            'top1_max_item_share': max(first.values())/len(slates),
            'slate_slot_hhi': hhi, 'effective_catalog_items': 1/hhi,
            'unique_catalog_items': len(counts),
            'scope': 'generated recommendation slots, not observed impressions or watches'}


def step_exposure(items):
    counts = Counter(items)
    total = len(items)
    hhi = math.fsum((count/total)**2 for count in counts.values())
    return {'recommendation_slots': total, 'max_item_share': max(counts.values())/total,
            'hhi': hhi, 'effective_catalog_items': 1/hhi, 'unique_catalog_items': len(counts)}


def digest(value, name):
    require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
            'Invalid SHA256 '+name)
    return value


def metadata_pin(value, name):
    require(isinstance(value, dict) and isinstance(value.get('path'), str) and bool(value['path']),
            'Missing provenance pin '+name)
    digest(value.get('sha256'), name)


def validate_lineage(document):
    require(document.get('original_val_rows') == ORIGINAL_ROWS and document.get('trajectory_count') == 192 and
            document.get('step_count') == 960, 'Original rows/declared grid counts differ')
    for name in ('cache_sha256', 'reward_sha256', 'utility_sha256'):
        digest(document.get(name), name)
    for name in ('selection_pin', 'preregistration', 'source'):
        metadata_pin(document.get(name), name)
    pins = document.get('scientific_checkpoints')
    require(isinstance(pins, dict) and set(pins) == set(HEADS), 'Explicit scientific lineage required')
    for arm in HEADS:
        metadata_pin(pins[arm], arm)
    c_sha = pins['C']['sha256']
    require(pins['C'].get('step') is None, 'C must be original supervised checkpoint')
    for arm in HEADS[1:]:
        pin = pins[arm]
        metadata_pin(pin.get('training'), arm+' training')
        require(pin.get('step') == pin.get('updates') == 1000 and
                pin.get('C_checkpoint_sha256') == c_sha, 'Scientific latest1000/C ancestor differs')
    require(pins['RL-completion']['sha256'] != pins['RL-residual']['sha256'], 'Scientific checkpoints must be distinct')


def validate_document(document, positions, seed=20261024):
    """Return complete keyed trajectories and controls; reject partial Cartesian grid."""
    require(positions == POSITIONS and all(type(p) is int for p in positions),
            'Exact old label-blind sixteen-position order required')
    require(type(seed) is int and seed == 20261024, 'Original matched seed required')
    validate_lineage(document)
    require(document.get('contract') == 'kion-final3-liveprefix-grid-v1' and
            document.get('arms') == list(HEADS) and document.get('positions') == positions and
            document.get('seed') == seed, 'Scientific grid/selection contract differs')
    require(document.get('state') == 'completed' and document.get('five_step_training') is False and
            document.get('synthetic_prefix_only') is True and document.get('RM_observed_s0_fixed') is True,
            'Completed synthetic-prefix/fixed-observed-RM contract required')
    require(document.get('clock') == document.get('observed_history') == 'observed-s0-fixed',
            'Observed history/clock must remain frozen')
    require(len(document['results']) == 192 and len(document['RM_greedy_controls']) == 16,
            'Exactly192 trajectories and16 RM controls required')
    expected = {(p, h, m, mask) for p in positions for h in HEADS for m in MODES for mask in MASKS}
    found, controls, token_to_raw, raw_to_token = {}, {}, {}, {}
    candidate_bindings, observed_seen_bindings, baseline_bindings, reward_bindings = {}, {}, {}, {}
    producer_bindings, initial_bindings, residual_bindings, utility_bindings = {}, {}, {}, {}

    def bind(token, raw):
        require(type(token) is int and token >= 3 and type(raw) is int and raw >= 0, 'Invalid generated token/rawID')
        require(token_to_raw.get(token, raw) == raw and raw_to_token.get(raw, token) == token,
                'Inconsistent token-to-raw mapping')
        token_to_raw[token], raw_to_token[raw] = raw, token

    for outer in document['results']:
        require(type(outer['position']) is int, 'Integer trajectory position required')
        key = (outer['position'], outer['arm'], outer['selection'], outer['repeat_mask'])
        require(key in expected and key not in found, 'Duplicate/unexpected trajectory context')
        result = outer['result']
        require(result['contract'] == 'kion-final3-liveprefix-producer-shape-fixed-observed-rm-v1' and
                result['scope'] == 'synthetic-id-slate-fixed-observed-rm', 'Wrong trajectory contract')
        require(result['selection'] == key[2] and result['repeat_mask'] == key[3] and result['steps'] == STEPS
                and result['seed'] == seed+key[0], 'Inner trajectory context/seed differs')
        require(close(finite(result['gamma'], 'gamma'), .9) and close(finite(result['temperature'], 'temperature'), 1.),
                'Registered gamma=.9/temperature1 required')
        require(result['rm_state'] == result['raw1068'] == result['source13'] == 'observed-s0-fixed'
                and result['source12'] == 'live-prefix-gpt-logit' and result['frozen_tensor_versions_unchanged'] is True,
                'Frozen/live state scope differs')
        require(result['context_policy'] == 'BOS-plus-latest63-real-or-UNK-tokens' and
                result['encoder_peers'] == 'original-observed-s0-except-active-synthetic-prefix' and
                result['retrieval_projection'] == 'original-full-vocab-then-fixed-pool-gather',
                'Producer-prefix/full-vocabulary contract differs')
        size, slot = result['producer_size'], result['producer_slot']
        require(type(size) is int and 1 <= size <= 16 and type(slot) is int and 0 <= slot < size,
                'Exact original producer size/slot required')
        producer = (size, slot)
        require(producer_bindings.get(key[0], producer) == producer, 'Producer layout differs across trajectories')
        producer_bindings[key[0]] = producer
        ids, raw_ids, steps = result['generated_token_ids'], result['generated_raw_ids'], result['per_step']
        require(len(ids) == len(raw_ids) == len(steps) == STEPS, 'Exactly5 generated items/steps required')
        seen, neutral_return, raw_return, unsupported = set(), 0., 0., 0
        baseline = finite(result['initial_reference_supported_baseline'], 'initial supported baseline')
        require(0 <= baseline <= 1, 'Invalid supported baseline')
        require(close(baseline_bindings.get(key[0], baseline), baseline), 'Initial C baseline differs across matched contexts')
        baseline_bindings[key[0]] = baseline
        residual_baseline = finite(result['initial_reference_supported_residual_baseline'], 'residual baseline')
        require(-1 <= residual_baseline <= 1 and close(residual_bindings.get(key[0], residual_baseline), residual_baseline),
                'Residual baseline differs across matched contexts')
        residual_bindings[key[0]] = residual_baseline
        residual_neutral_return, residual_raw_return = 0., 0.
        for j, (token, raw, step) in enumerate(zip(ids, raw_ids, steps)):
            bind(token, raw)
            require(step['step'] == j+1 and step['generated_token_id'] == token and step['generated_raw_id'] == raw,
                    'Step/item binding differs')
            require(type(step['candidate_index']) is int and 0 <= step['candidate_index'] < 1024, 'Invalid candidate index')
            candidate_key = (key[0], step['candidate_index'])
            require(candidate_bindings.get(candidate_key, token) == token, 'Fixed-pool candidate binding differs')
            candidate_bindings[candidate_key] = token
            require(type(step['generated_repeat']) is bool and step['generated_repeat'] == (token in seen), 'Repeat flag differs')
            require(type(step['original_seen']) is bool and type(step['selected_supported']) is bool, 'Boolean outcome flags required')
            observed_key = (key[0], token)
            require(observed_seen_bindings.get(observed_key, step['original_seen']) == step['original_seen'],
                    'Observed-prefix seen flag differs across matched contexts')
            observed_seen_bindings[observed_key] = step['original_seen']
            require(key[3] != 'exclude-generated' or token not in seen, 'Generated-repeat exclusion violated')
            seen.add(token)
            require(step['producer_size'] == size and step['producer_slot'] == slot and
                    type(step['prefix_length']) is int and 1 <= step['prefix_length'] <= 64 and
                    type(step['encoder_trim_width']) is int and step['prefix_length'] <= step['encoder_trim_width'] <= 64 and
                    step['retrieval_projection'] == result['retrieval_projection'], 'Step producer/trim/projection differs')
            require(type(step['initial_C_cache_parity']) is dict if j == 0 else step['initial_C_cache_parity'] is None,
                    'Initial C cache parity evidence required exactly at s0')
            if j == 0:
                parity = step['initial_C_cache_parity']
                require(parity['atol'] == .003 and parity['rtol'] == .001 and
                        parity['baseline'] == 'fresh-live-C-on-identical-prefix' and type(parity['top10_exact']) is bool and
                        parity['score_tolerance_passed'] is True and parity['rank_tolerance_passed'] is True and
                        finite(parity['max_abs_error'], 'cache parity max error') >= 0,
                        'Old v2 cache parity gate/evidence differs')
                fresh_max = finite(parity['fresh_reference_maxabs'], 'reference score magnitude')
                require(fresh_max >= 0 and parity['max_abs_error'] <= .003+.001*fresh_max+1e-7,
                        'Cache score parity magnitude exceeds registered tolerance')
                initial = (step['prefix_length'], step['encoder_trim_width'], parity)
                require(initial_bindings.get(key[0], initial) == initial, 'Matched s0 parity/shape differs')
                initial_bindings[key[0]] = initial
            for metric in ('q_l2_drift', 'base_rmse_drift', 'cached_q_maxabs', 'cached_base_maxabs'):
                require(finite(step[metric], metric) >= 0, 'Invalid nonnegative '+metric)
            require(0 < finite(step['selected_probability'], 'selected probability') <= 1 and
                    math.isfinite(finite(step['selected_base_logit'], 'base logit')), 'Invalid selected score/probability')
            if j == 0:
                require(step['q_l2_drift'] == step['base_rmse_drift'] == 0, 's0 drift must be zero')
            if j:
                require(step['prefix_length'] == min(64, steps[j-1]['prefix_length']+1), 'Synthetic append prefix length differs')
            for metric in ('kl_nats', 'entropy_nats'):
                require(finite(step[metric], metric) >= -1e-7, 'Invalid nonnegative '+metric)
            allowed = 1024-j if key[3] == 'exclude-generated' else 1024
            require(step['entropy_nats'] <= math.log(allowed)+1e-7, 'Entropy exceeds active action support')
            if key[1] == 'C':
                require(abs(step['kl_nats']) <= 1e-7, 'C-to-identical-C KL must be zero')
            mass = finite(step['supported_mass'], 'supported_mass')
            unsupported_mass = finite(step['unsupported_mass'], 'unsupported_mass')
            require(0 <= mass <= 1+1e-7 and 0 <= unsupported_mass <= 1+1e-7 and close(mass+unsupported_mass, 1),
                    'Support probability mass differs')
            require(step['selected_probability'] <= (mass if step['selected_supported'] else unsupported_mass)+1e-7,
                    'Selected action probability exceeds its support-group mass')
            conditional = step['expected_supported_reward_conditional']
            require((conditional is None) == (mass == 0), 'Conditional reward availability differs')
            if conditional is not None:
                require(0 <= finite(conditional, 'conditional expected reward') <= 1, 'Conditional reward outside[0,1]')
            expected_neutral = finite(step['expected_neutral_proxy'], 'expected_neutral_proxy')
            require(-1-1e-7 <= expected_neutral <= 1+1e-7,
                    'Neutral expected proxy outside bounds')
            require(close(expected_neutral, mass*(conditional-baseline) if mass else 0),
                    'Expected neutral proxy arithmetic differs')
            residual_cond = step['expected_supported_residual_conditional']
            prior_cond = step['expected_supported_prior_conditional']
            require((residual_cond is None) == (prior_cond is None) == (mass == 0), 'Conditional utility availability differs')
            if mass:
                require(-1 <= finite(residual_cond, 'conditional residual') <= 1 and
                        0 <= finite(prior_cond, 'conditional prior') <= 1 and
                        close(residual_cond, conditional-prior_cond), 'Conditional residual/prior arithmetic differs')
            neutral_residual = finite(step['expected_neutral_residual_proxy'], 'expected neutral residual')
            require(close(neutral_residual, mass*(residual_cond-residual_baseline) if mass else 0),
                    'Expected neutral residual arithmetic differs')
            if key[1] == 'C' and j == 0:
                require(abs(expected_neutral) <= 1e-7 and abs(neutral_residual) <= 1e-7,
                        'C centered expected utilities at unmasked s0 must be zero')
            effective = finite(step['selected_neutral_proxy'], 'selected_neutral_proxy')
            effective_residual = finite(step['selected_neutral_residual_proxy'], 'selected neutral residual')
            reward = step['selected_reward']
            residual, prior = step['selected_residual'], step['selected_fit_prior']
            if step['selected_supported']:
                require(reward is not None and 0 <= finite(reward, 'selected_reward') <= 1, 'Invalid supported reward')
                require(close(effective, reward-baseline), 'Supported effective reward differs')
                raw_return += .9**j*reward
                require(-1 <= finite(residual, 'selected residual') <= 1 and 0 <= finite(prior, 'selected prior') <= 1 and
                        close(residual, reward-prior) and close(effective_residual, residual-residual_baseline),
                        'Supported residual/prior arithmetic differs')
                residual_raw_return += .9**j*residual
            else:
                require(reward is None and effective == 0, 'Unsupported action must have null raw/zero neutral proxy')
                require(residual is None and prior is None and effective_residual == 0,
                        'Unsupported residual/prior must be null; centered utility zero')
                unsupported += 1
            utility_binding = (residual, prior)
            require(utility_bindings.get(observed_key, utility_binding) == utility_binding, 'Fixed s0 utility/prior differs')
            utility_bindings[observed_key] = utility_binding
            reward_binding = (step['selected_supported'], reward)
            require(reward_bindings.get(observed_key, reward_binding) == reward_binding,
                    'Fixed observed-s0 RM support/reward differs across matched contexts')
            reward_bindings[observed_key] = reward_binding
            neutral_return += .9**j*effective
            residual_neutral_return += .9**j*effective_residual
        unique = len(set(ids))
        require(result['unique_generated'] == unique and result['generated_repeat_count'] == STEPS-unique,
                'Unique/repeat count differs')
        require(result['unsupported_selected'] == unsupported, 'Unsupported selected count differs')
        require(close(finite(result['fixed_neutral_proxy_return'], 'neutral return'), neutral_return) and
                close(finite(result['supported_completion_partial_sum'], 'partial raw return'), raw_return), 'Discounted return arithmetic differs')
        complete_raw = result['fixed_completion_proxy_return']
        require((complete_raw is None) == (unsupported > 0), 'Raw return requires all selected actions supported')
        if complete_raw is not None:
            require(close(finite(complete_raw, 'complete raw proxy return'), raw_return), 'Complete raw return differs')
        require(close(finite(result['fixed_neutral_residual_proxy_return'], 'neutral residual return'), residual_neutral_return) and
                close(finite(result['supported_residual_partial_sum'], 'partial residual return'), residual_raw_return),
                'Residual return arithmetic differs')
        residual_complete = result['fixed_residual_proxy_return']
        require((residual_complete is None) == (unsupported > 0), 'Residual return support differs')
        if residual_complete is not None:
            require(close(finite(residual_complete, 'residual return'), residual_raw_return), 'Complete residual return differs')
        final_prefix = result['final_prefix']
        require(isinstance(final_prefix, list) and len(final_prefix) == min(64, steps[0]['prefix_length']+STEPS) and
                final_prefix[0] == 1 and final_prefix[-STEPS:] == ids and
                all(type(v) is int and v >= 2 for v in final_prefix[1:]), 'Synthetic final prefix differs')
        found[key] = result
    require(set(found) == expected, 'Incomplete trajectory Cartesian grid')
    for control in document['RM_greedy_controls']:
        position = control['position']
        require(type(position) is int, 'Integer control position required')
        require(position in positions and position not in controls, 'Duplicate/unexpected RM control position')
        ids, raws, rewards, support = (control[k] for k in ('generated_ids', 'generated_raw_ids', 'rewards', 'supported'))
        require(len(ids) == len(raws) == len(rewards) == len(support) == STEPS and
                len(set(ids)) == len(set(raws)) == STEPS, 'RM-greedy control must contain5 unique items')
        for token, raw, reward, ok in zip(ids, raws, rewards, support):
            bind(token, raw)
            require(type(ok) is bool and (reward is None) == (not ok), 'RM control support/reward differs')
            if ok:
                require(0 <= finite(reward, 'RM reward') <= 1, 'Invalid RM control reward')
            binding = reward_bindings.get((position, token))
            require(binding is None or binding == (ok, reward), 'RM control/policy reward binding differs')
        expected_order = sorted(range(STEPS), key=lambda j: (-(rewards[j] if support[j] else -1), raws[j]))
        require(expected_order == list(range(STEPS)), 'RM-greedy control reward/rawID order differs')
        controls[position] = control
    require(set(controls) == set(positions), 'Incomplete RM control cohort')
    return found, controls


def summarize(document, positions, seed=20261024, titles=None):
    trajectories, controls = validate_document(document, positions, seed)
    groups = []
    for head in HEADS:
        for mode in MODES:
            for mask in MASKS:
                rows = [trajectories[p, head, mode, mask] for p in positions]
                groups.append({'arm': head, 'selection': mode, 'repeat_mask': mask, 'positions': len(rows),
                    'mean_unique_generated': mean([row['unique_generated'] for row in rows]),
                    'mean_generated_repeat_count': mean([row['generated_repeat_count'] for row in rows]),
                    'original_seen_fraction': mean([step['original_seen'] for row in rows for step in row['per_step']]),
                    'original_seen_fraction_basis': 'producer observed-prefix flags; not reconstructed sessions',
                    'exposure': exposure([row['generated_raw_ids'] for row in rows]),
                    'fixed_neutral_proxy_return': mean_with_denominator([row['fixed_neutral_proxy_return'] for row in rows], len(rows)),
                    'fixed_neutral_residual_proxy_return': mean_with_denominator(
                        [row['fixed_neutral_residual_proxy_return'] for row in rows], len(rows)),
                    'fixed_residual_proxy_return_all_supported_only': mean_with_denominator(
                        [row['fixed_residual_proxy_return'] for row in rows], len(rows)),
                    'fixed_completion_proxy_return_all_supported_only': mean_with_denominator(
                        [row['fixed_completion_proxy_return'] for row in rows], len(rows)),
                    'per_step': [{'step': j+1, **{metric: mean_with_denominator(
                        [row['per_step'][j][metric] for row in rows], len(rows)) for metric in STEP_METRICS},
                        'original_seen_fraction': mean([row['per_step'][j]['original_seen'] for row in rows]),
                        'generated_repeat_fraction': mean([row['per_step'][j]['generated_repeat'] for row in rows]),
                        'selected_supported_fraction': mean([row['per_step'][j]['selected_supported'] for row in rows]),
                        'exposure': step_exposure([row['generated_raw_ids'][j] for row in rows])}
                        for j in range(STEPS)]})
    comparisons = []
    for head in HEADS[1:]:
        for mode in MODES:
            for mask in MASKS:
                pairs = [(trajectories[p, 'C', mode, mask], trajectories[p, head, mode, mask]) for p in positions]
                comparisons.append({'arm': head, 'reference': 'C', 'selection': mode, 'repeat_mask': mask,
                    'matched_positions': len(pairs),
                    'mean_multiset_slate_overlap_fraction': mean([sum((Counter(a['generated_raw_ids']) &
                        Counter(b['generated_raw_ids'])).values())/STEPS for a, b in pairs]),
                    'mean_set_jaccard': mean([len(set(a['generated_raw_ids']) & set(b['generated_raw_ids']))/
                        len(set(a['generated_raw_ids']) | set(b['generated_raw_ids'])) for a, b in pairs]),
                    'exact_sequence_fraction': mean([a['generated_raw_ids'] == b['generated_raw_ids'] for a, b in pairs]),
                    'top1_same_fraction': mean([a['generated_raw_ids'][0] == b['generated_raw_ids'][0] for a, b in pairs]),
                    'neutral_proxy_return_difference': mean([b['fixed_neutral_proxy_return']-a['fixed_neutral_proxy_return'] for a, b in pairs]),
                    'neutral_residual_proxy_return_difference': mean([b['fixed_neutral_residual_proxy_return']-
                        a['fixed_neutral_residual_proxy_return'] for a,b in pairs]),
                    'mean_unique_difference': mean([b['unique_generated']-a['unique_generated'] for a, b in pairs])})
    control_rows = [controls[p] for p in positions]
    control_returns = [math.fsum(.9**j*reward for j, reward in enumerate(row['rewards']))
                       if all(row['supported']) else None for row in control_rows]
    examples = []
    # Owner supplies positions before outcomes; preserve that order regardless of input JSON ordering.
    for position in positions[:3]:
        slates = []
        for head in HEADS:
            for mode in MODES:
                for mask in MASKS:
                    row = trajectories[position, head, mode, mask]
                    slates.append({'arm': head, 'selection': mode, 'repeat_mask': mask,
                        'items': [{'raw_id': raw, 'title': titles.get(raw) if titles is not None else None}
                                  for raw in row['generated_raw_ids']]})
        examples.append({'position': position, 'slates': slates,
                         'RM_greedy_unique5': [{'raw_id': raw, 'title': titles.get(raw) if titles is not None else None}
                                              for raw in controls[position]['generated_raw_ids']]})
    return {'contract': 'kion-final3-saved-liveprefix-summary-v1', 'state': 'completed', 'trajectory_count': len(trajectories),
        'expected_positions': positions, 'position_count': len(positions), 'steps': STEPS, 'seed': seed,
        'group_summaries': groups, 'matched_C_comparisons': comparisons,
        'RM_greedy_unique5_control': {'definition': 'fixed observed-s0 RM-greedy unique top5, unsupported last; separate static control',
            'positions': len(control_rows), 'mean_unique_generated': 5.,
            'exposure': exposure([row['generated_raw_ids'] for row in control_rows]),
            'completion_proxy_return_all_supported_only': mean_with_denominator(control_returns, len(control_rows)),
            'policy_KL_entropy_original_seen': 'unavailable for this static control; no fabricated policy distribution'},
        'example_selection': 'first3 explicit preselected positions, in request order; all12 contexts at each; no outcome-based selection',
        'examples': examples,
        'limitations': ['Five-step inference diagnostic only; RL objective/training was H1, not five-step return optimization.',
            'Synthetic generated IDs enter live GPT prefix; raw1068/user/source13/RM state remain observed-s0-fixed.',
            'RM is a supported proxy, not observed/causal counterfactual completion, retention or satisfaction.',
            'Unknown outcomes are not negative labels; unsupported raw returns remain null and excluded with denominators.',
            'Original-seen flags describe available observed pre-origin prefix, not every lifetime interaction/session.',
            'Sixteen fixed positions and one seeded categorical trajectory per context are descriptive, not population inference.',
            'C and comparison policies can generate different prefixes; comparisons are matched initial states, not common later state distributions.',
            'No model, NumPy, provider, new feature computation or real-job launch in this summarizer.']}


def run(request, check=None):
    """Required rollout/items pins, exact expected_positions16 and fresh output."""
    path, rollout_pin = pinned(request['rollout'], maximum_bytes=20*1024*1024)
    require(path.suffix == '.json', 'Saved JSON metadata required')
    output = Path(request['output']).resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent) and not output.exists(), 'Fresh owned output required')
    if check:
        check(stage='saved-rollout-summary', phase='metadata-validation')
    document = read(path)
    positions = request['expected_positions']
    require(bool(request.get('items')), 'Pinned item catalog is required for named prechosen cases')
    catalog, catalog_pin = pinned(request['items'])
    titles = {}
    with catalog.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            item = int(row['item_id'])
            require(item not in titles, 'Duplicate catalog item')
            titles[item] = row.get('title', '').strip() or None
    result = summarize(document, positions, request.get('seed', 20261024), titles)
    for case in result['examples']:
        items = [item for slate in case['slates'] for item in slate['items']] + case['RM_greedy_unique5']
        require(all(item['title'] for item in items), 'Named case has missing catalog title')
    result.update(rollout=rollout_pin, items_catalog=catalog_pin,
        source={'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
        source_metadata={key: document[key] for key in ('contract', 'cache_sha256', 'reward_sha256', 'utility_sha256',
            'five_step_training', 'synthetic_prefix_only', 'RM_observed_s0_fixed')})
    output.mkdir(parents=True)
    write(output/'rollout-summary.json', result)
    lines = ['# Five-step live-prefix saved-output summary', '',
        'Validated192 complete trajectories:16 fixed positions×3heads×2selection modes×2repeat masks×5steps. '
        'This is inference from an H1-trained policy with fixed observed-s0 RM/raw features; no five-step training or causal outcome claim.', '',
        '| Head | Mode | Mask | mean unique | mean repeats | observed-prefix seen fraction | top1 max share | slate HHI | effective items | unique catalog | neutral proxy return | raw proxy return (all-supported n/16) |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in result['group_summaries']:
        e, raw = row['exposure'], row['fixed_completion_proxy_return_all_supported_only']
        raw_text = 'unavailable' if raw['mean'] is None else f"{raw['mean']:.6f}"
        lines.append(f"| {row['arm']} | {row['selection']} | {row['repeat_mask']} | {row['mean_unique_generated']:.3f} | "
            f"{row['mean_generated_repeat_count']:.3f} | {row['original_seen_fraction']:.4f} | {e['top1_max_item_share']:.4f} | "
            f"{e['slate_slot_hhi']:.6f} | {e['effective_catalog_items']:.2f} | {e['unique_catalog_items']} | "
            f"{row['fixed_neutral_proxy_return']['mean']:.6f} | {raw_text} ({raw['denominator']}/16) |")
    lines += ['', 'Per-step KL/entropy, unsupported probability mass, effective expected proxy and supported-conditional '
        'expected rewards with denominators; paired16-position overlaps; separate static RM-greedy unique5 control; '
        'and first3 predetermined-position slates with rawIDs/titles are in [rollout-summary.json](rollout-summary.json).', '',
        'A raw completion return is averaged only when all5 selected actions are supported. Unsupported selections have '
        'null raw reward and zero centered neutral proxy; this does not imply zero real-world satisfaction. '
        'Top1/slate concentration describes generated outputs, not impressions. Later prefixes differ across policies.', '',
        'Raw1068/source13 and observed RM state remain fixed while GPT q/base/source12 evolve. '
        'These synthetic sequences do not reconstruct watched sessions, user responses, retention or causal counterfactual outcomes. '
        'Sixteen fixed contexts and one categorical seed are descriptive only.']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8', newline='\n')
    if check:
        check(stage='saved-rollout-summary', phase='completed', trajectories=192)
    return result
