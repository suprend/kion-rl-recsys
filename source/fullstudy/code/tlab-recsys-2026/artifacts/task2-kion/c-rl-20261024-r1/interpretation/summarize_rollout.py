"""Strict saved-JSON five-step rollout summary. Stdlib only; root runs real data."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

HEADS = ('C', 'latest', 'selected')
MODES = ('greedy', 'categorical')
MASKS = ('none', 'exclude-generated')
STEPS = 5
STEP_METRICS = ('kl_nats', 'entropy_nats', 'unsupported_mass', 'expected_neutral_proxy',
                'expected_supported_reward_conditional')


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


def validate_document(document, positions, seed=20261024):
    """Return complete keyed trajectories and controls; reject partial Cartesian grid."""
    require(len(positions) == 16 and len(set(positions)) == 16 and
            all(type(p) is int and p >= 0 for p in positions), 'Exactly16 preselected unique positions required')
    require(type(seed) is int, 'Integer seed required')
    require(document.get('state') == 'completed' and document.get('five_step_training') is False and
            document.get('synthetic_prefix_only') is True and document.get('RM_observed_s0_fixed') is True,
            'Completed synthetic-prefix/fixed-observed-RM contract required')
    require(len(document['results']) == 192 and len(document['RM_greedy_controls']) == 16,
            'Exactly192 trajectories and16 RM controls required')
    expected = {(p, h, m, mask) for p in positions for h in HEADS for m in MODES for mask in MASKS}
    found, controls, token_to_raw, raw_to_token = {}, {}, {}, {}
    candidate_bindings, observed_seen_bindings, baseline_bindings, reward_bindings = {}, {}, {}, {}

    def bind(token, raw):
        require(type(token) is int and token >= 3 and type(raw) is int and raw >= 0, 'Invalid generated token/rawID')
        require(token_to_raw.get(token, raw) == raw and raw_to_token.get(raw, token) == token,
                'Inconsistent token-to-raw mapping')
        token_to_raw[token], raw_to_token[raw] = raw, token

    for outer in document['results']:
        require(type(outer['position']) is int, 'Integer trajectory position required')
        key = (outer['position'], outer['head'], outer['selection'], outer['repeat_mask'])
        require(key in expected and key not in found, 'Duplicate/unexpected trajectory context')
        result = outer['result']
        require(result['contract'] == 'kion-c3-liveprefix-fixed-observed-rm-v1' and
                result['scope'] == 'synthetic-id-slate-fixed-observed-rm', 'Wrong trajectory contract')
        require(result['selection'] == key[2] and result['repeat_mask'] == key[3] and result['steps'] == STEPS
                and result['seed'] == seed+key[0], 'Inner trajectory context/seed differs')
        require(close(finite(result['gamma'], 'gamma'), .9) and close(finite(result['temperature'], 'temperature'), 1.),
                'Registered gamma=.9/temperature1 required')
        require(result['rm_state'] == result['raw1068'] == result['source13'] == 'observed-s0-fixed'
                and result['source12'] == 'live-prefix-gpt-logit' and result['frozen_tensor_versions_unchanged'] is True,
                'Frozen/live state scope differs')
        ids, raw_ids, steps = result['generated_token_ids'], result['generated_raw_ids'], result['per_step']
        require(len(ids) == len(raw_ids) == len(steps) == STEPS, 'Exactly5 generated items/steps required')
        seen, neutral_return, raw_return, unsupported = set(), 0., 0., 0
        baseline = finite(result['initial_reference_supported_baseline'], 'initial supported baseline')
        require(0 <= baseline <= 1, 'Invalid supported baseline')
        require(close(baseline_bindings.get(key[0], baseline), baseline), 'Initial C baseline differs across matched contexts')
        baseline_bindings[key[0]] = baseline
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
            for metric in ('kl_nats', 'entropy_nats'):
                require(finite(step[metric], metric) >= -1e-7, 'Invalid nonnegative '+metric)
            mass = finite(step['supported_mass'], 'supported_mass')
            unsupported_mass = finite(step['unsupported_mass'], 'unsupported_mass')
            require(0 <= mass <= 1+1e-7 and 0 <= unsupported_mass <= 1+1e-7 and close(mass+unsupported_mass, 1),
                    'Support probability mass differs')
            conditional = step['expected_supported_reward_conditional']
            require((conditional is None) == (mass == 0), 'Conditional reward availability differs')
            if conditional is not None:
                require(0 <= finite(conditional, 'conditional expected reward') <= 1, 'Conditional reward outside[0,1]')
            expected_neutral = finite(step['expected_neutral_proxy'], 'expected_neutral_proxy')
            require(-1-1e-7 <= expected_neutral <= 1+1e-7,
                    'Neutral expected proxy outside bounds')
            require(close(expected_neutral, mass*(conditional-baseline) if mass else 0),
                    'Expected neutral proxy arithmetic differs')
            effective = finite(step['selected_neutral_proxy'], 'selected_neutral_proxy')
            reward = step['selected_reward']
            if step['selected_supported']:
                require(reward is not None and 0 <= finite(reward, 'selected_reward') <= 1, 'Invalid supported reward')
                require(close(effective, reward-baseline), 'Supported effective reward differs')
                raw_return += .9**j*reward
            else:
                require(reward is None and effective == 0, 'Unsupported action must have null raw/zero neutral proxy')
                unsupported += 1
            reward_binding = (step['selected_supported'], reward)
            require(reward_bindings.get(observed_key, reward_binding) == reward_binding,
                    'Fixed observed-s0 RM support/reward differs across matched contexts')
            reward_bindings[observed_key] = reward_binding
            neutral_return += .9**j*effective
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
                groups.append({'head': head, 'selection': mode, 'repeat_mask': mask, 'positions': len(rows),
                    'mean_unique_generated': mean([row['unique_generated'] for row in rows]),
                    'mean_generated_repeat_count': mean([row['generated_repeat_count'] for row in rows]),
                    'original_seen_fraction': mean([step['original_seen'] for row in rows for step in row['per_step']]),
                    'original_seen_fraction_basis': 'producer observed-prefix flags; not reconstructed sessions',
                    'exposure': exposure([row['generated_raw_ids'] for row in rows]),
                    'fixed_neutral_proxy_return': mean_with_denominator([row['fixed_neutral_proxy_return'] for row in rows], len(rows)),
                    'fixed_completion_proxy_return_all_supported_only': mean_with_denominator(
                        [row['fixed_completion_proxy_return'] for row in rows], len(rows)),
                    'per_step': [{'step': j+1, **{metric: mean_with_denominator(
                        [row['per_step'][j][metric] for row in rows], len(rows)) for metric in STEP_METRICS}}
                        for j in range(STEPS)]})
    comparisons = []
    for head in ('latest', 'selected'):
        for mode in MODES:
            for mask in MASKS:
                pairs = [(trajectories[p, 'C', mode, mask], trajectories[p, head, mode, mask]) for p in positions]
                comparisons.append({'head': head, 'reference': 'C', 'selection': mode, 'repeat_mask': mask,
                    'matched_positions': len(pairs),
                    'mean_multiset_slate_overlap_fraction': mean([sum((Counter(a['generated_raw_ids']) &
                        Counter(b['generated_raw_ids'])).values())/STEPS for a, b in pairs]),
                    'mean_set_jaccard': mean([len(set(a['generated_raw_ids']) & set(b['generated_raw_ids']))/
                        len(set(a['generated_raw_ids']) | set(b['generated_raw_ids'])) for a, b in pairs]),
                    'exact_sequence_fraction': mean([a['generated_raw_ids'] == b['generated_raw_ids'] for a, b in pairs]),
                    'top1_same_fraction': mean([a['generated_raw_ids'][0] == b['generated_raw_ids'][0] for a, b in pairs]),
                    'neutral_proxy_return_difference': mean([b['fixed_neutral_proxy_return']-a['fixed_neutral_proxy_return'] for a, b in pairs]),
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
                    slates.append({'head': head, 'selection': mode, 'repeat_mask': mask,
                        'items': [{'raw_id': raw, 'title': titles.get(raw) if titles is not None else None}
                                  for raw in row['generated_raw_ids']]})
        examples.append({'position': position, 'slates': slates,
                         'RM_greedy_unique5': [{'raw_id': raw, 'title': titles.get(raw) if titles is not None else None}
                                              for raw in controls[position]['generated_raw_ids']]})
    return {'contract': 'C3-saved-liveprefix-summary-v1', 'state': 'completed', 'trajectory_count': len(trajectories),
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
    """Required rollout pin, expected_positions16, output; optional seed/items pin."""
    path, rollout_pin = pinned(request['rollout'], maximum_bytes=20*1024*1024)
    require(path.suffix == '.json', 'Saved JSON metadata required')
    output = Path(request['output']).resolve()
    require(output.is_relative_to(Path(__file__).resolve().parent) and not output.exists(), 'Fresh owned output required')
    if check:
        check(stage='saved-rollout-summary', phase='metadata-validation')
    document = read(path)
    positions = request['expected_positions']
    titles, catalog_pin = None, None
    if request.get('items'):
        catalog, catalog_pin = pinned(request['items'])
        titles = {}
        with catalog.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                item = int(row['item_id'])
                require(item not in titles, 'Duplicate catalog item')
                titles[item] = row.get('title', '') or None
    result = summarize(document, positions, request.get('seed', 20261024), titles)
    result.update(rollout=rollout_pin, items_catalog=catalog_pin,
        source={'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
        source_metadata={key: document[key] for key in ('contract', 'cache_sha256', 'reward_sha256',
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
        lines.append(f"| {row['head']} | {row['selection']} | {row['repeat_mask']} | {row['mean_unique_generated']:.3f} | "
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
