"""Frozen observed-fit-prior utility. Import is stdlib-only; run is root-owned."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'kion-derived-completion-utility-v1'
PREREG = 'kion-C-residual-RL-preregistered-v1'
FORMULA = 'r_RM(s,i)-m_fit(i)'
PRIOR_SHA256 = 'd5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4'
POLICY_SHA256 = 'da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 8 << 20, 'Oversized metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def load_policy(request):
    declared = request['policy_module']
    if isinstance(declared, dict):
        require(declared.get('sha256') == POLICY_SHA256, 'Frozen policy module descriptor pin differs')
        declared = declared['path']
    path = Path(declared).resolve()
    require(sha(path) == POLICY_SHA256 == request['source_hashes'].get(path.as_posix()),
            'Exact immutable policy_v2 source pin required')
    spec = importlib.util.spec_from_file_location('_final25_frozen_policy', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_preregistration(request):
    require(sha(request['preregistration']) == request['preregistration_sha256'], 'Preregistration pin changed')
    value = read(request['preregistration'])
    required = dict(contract=PREREG, chosen_utility='rm-minus-frozen-item-mean', formula=FORMULA,
                    coefficient=1.0, beta=.5, temperature=1., sampler_seed=20261024,
                    checkpoint_steps=[0, 250, 500, 1000], start_from='original-C',
                    original_results_known=True, one_additional_fit_only=True,
                    primary_contrast='fixed-latest1000', no_gpt_unfreeze=True, no_rm_fit=True, no_sft=True)
    require(all(value.get(key) == expected for key, expected in required.items()), 'Residual preregistration differs')
    require(value.get('optimizer') == dict(name='Adam', updates=1000, batch_size=16, lr=1e-5,
                                         grad_clip=1, weight_decay=0, alpha_ce=0), 'Fixed optimizer required')
    return value


def validate_prior_records(ids, counts, means):
    """Small stdlib validation seam; no implicit raw-ID missing-value imputation."""
    require(len(ids) > 0 and len(ids) == len(counts) == len(means), 'Prior vector lengths differ')
    require(all(type(v) is int and v >= 0 for v in ids) and len(set(ids)) == len(ids), 'Invalid/duplicate raw item IDs')
    require(all(type(v) is int and v >= 0 for v in counts), 'Prior counts must be nonnegative integers')
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in means), 'Prior means outside [0,1]')


def residual_value(reward, prior, supported=True):
    require(math.isfinite(reward) and 0 <= reward <= 1 and math.isfinite(prior) and 0 <= prior <= 1,
            'Completion/prior outside [0,1]')
    return reward - prior if supported else 0.0


def pairwise_preference_changes(rewards, utilities):
    """Exact changed sign fraction, including ties, in O(k log k); unordered pairs."""
    require(len(rewards) == len(utilities), 'Preference vector lengths differ')
    n = len(rewards)
    if n < 2:
        return 0, 0
    coordinates = {v: i + 1 for i, v in enumerate(sorted(set(utilities)))}
    tree = [0] * (len(coordinates) + 1)
    def prefix(i):
        total = 0
        while i:
            total += tree[i]
            i -= i & -i
        return total
    def add(i):
        while i < len(tree):
            tree[i] += 1
            i += i & -i
    groups = {}
    for reward, utility in zip(rewards, utilities):
        groups.setdefault(reward, []).append(utility)
    changed, seen = 0, 0
    for reward in sorted(groups):
        values = groups[reward]
        # Distinct base rewards: a reversal OR utility tie changes the sign.
        changed += sum(seen - prefix(coordinates[u] - 1) for u in values)
        # Equal base rewards: distinct utilities change the sign.
        changed += len(values) * (len(values) - 1) // 2
        changed -= sum(k * (k - 1) // 2 for k in Counter(values).values())
        for u in values:
            add(coordinates[u])
        seen += len(values)
    return changed, n * (n - 1) // 2


def open_minimal_cache(spec, name, np, policy):
    """Only IDs/reference scores enter this data-only extraction; raw1068 stays unopened."""
    root = Path(spec['cache_dir']).resolve()
    require(sha(root / 'cache.json') == spec['cache_sha256'], 'Cache manifest pin changed')
    manifest = read(root / 'cache.json')
    require(manifest.get('contract') == policy.CONTRACT and manifest.get('state') == 'completed'
            and manifest.get('cohort') == name and manifest.get('rows') == (512 if name == 'train' else 256)
            and manifest.get('candidate_width') == 1024 and manifest.get('feature_schema_sha256') == policy.SCHEMA
            and manifest.get('source_columns') == policy.SOURCE and manifest.get('no_seen_filter') is True
            and manifest.get('no_target_injection') is True, 'Exact frozen cohort/schema required')
    arrays = {}
    try:
        for key in ('candidate_ids', 'candidate_raw_ids', 'reference_scores'):
            entry = manifest['arrays'][key]
            path = (root / entry['path']).resolve()
            require(path.is_relative_to(root) and path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                    'Changed minimal cache array: ' + key)
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            arrays[key] = value
            require(str(value.dtype) == entry['dtype'] and list(value.shape) == entry['shape']
                    and value.shape == (manifest['rows'], 1024), 'Cache header differs')
        return manifest, arrays
    except BaseException:
        policy.close_cache(arrays)
        raise


def scales(np, policy, reference_scores, reward, utility, prior, support):
    _, probs = policy.distribution(np, reference_scores)
    base_adv, active = policy.effective_advantage(np, probs, reward, support)
    new_adv, active_new = policy.effective_advantage(np, probs, utility, support)
    require(bool(np.array_equal(active, active_new)) and bool(active.any()), 'Supported train states required')
    ranges = [float(np.ptp(row[mask])) if bool(mask.any()) else 0. for row, mask in zip(prior, support)]
    changed = pairs = 0
    for r, u, mask in zip(reward, utility, support):
        c, p = pairwise_preference_changes(r[mask].tolist(), u[mask].tolist())
        changed += c
        pairs += p
    def describe(values, advantage):
        selected = values[support]
        adv = advantage[support]
        return dict(supported_min=float(selected.min()), supported_max=float(selected.max()),
                    supported_sd=float(selected.std()), advantage_min=float(adv.min()),
                    advantage_max=float(adv.max()), advantage_sd=float(adv.std()),
                    reference_weighted_advantage_rms=float(np.sqrt((probs * advantage**2).sum(1)[active].mean())))
    base, residual = describe(reward, base_adv), describe(utility, new_adv)
    base_rms = base['reference_weighted_advantage_rms']
    return dict(base_completion=base, derived_utility=residual,
                advantage_rms_ratio=residual['reference_weighted_advantage_rms'] / base_rms if base_rms else None,
                active_states=int(active.sum()), supported_pairs=int(support.sum()),
                prior_within_state_range_max=max(ranges), nondegenerate_prior=any(v > 0 for v in ranges),
                fraction_constant_within_state_priors=sum(v == 0 for v, ok in zip(ranges, active) if ok) / int(active.sum()),
                changed_pairwise_preferences=changed, supported_unordered_pairs=pairs,
                fraction_changed_pairwise_preferences=changed / pairs if pairs else None,
                rms_definition='sqrt(mean_active_states(sum_all_actions(p_C * centered_supported_utility^2))); unsupported advantage=0')


def utility_arrays(path, cache, manifest, np, policy):
    """Return (derived utility, observed-fit prior, unchanged support); never RM scores."""
    root = Path(path).resolve()
    meta = read(root / 'utility.json')
    require(meta.get('contract') == CONTRACT and meta.get('state') == 'completed'
            and meta.get('formula') == FORMULA and meta.get('range') == [-1, 1]
            and meta.get('coefficient') == 1.0 and meta.get('prior_archive', {}).get('sha256') == PRIOR_SHA256,
            'Invalid separate derived utility contract')
    require(sha(meta['cache']['path']) == meta['cache']['sha256'] and read(meta['cache']['path']) == manifest,
            'Utility cache binding differs')
    parent = meta['parent_reward']
    require(sha(parent['path']) == parent['sha256'], 'Parent reward changed')
    reward, support, _ = policy.reward_arrays(Path(parent['path']).parent, cache, manifest, np)
    require(sha(meta['support']['path']) == meta['support']['sha256'] == read(parent['path'])['files']['support.npy']['sha256'],
            'Utility support changed')
    loaded = []
    for name in ('utility.npy', 'prior.npy'):
        entry = meta['files'][name]
        file = (root / entry['path']).resolve()
        require(file.is_relative_to(root) and sha(file) == entry['sha256'], 'Derived utility payload changed')
        value = np.load(file, allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] == 'float64' and list(value.shape) == entry['shape']
                and value.shape == support.shape and bool(np.isfinite(value).all()), 'Invalid utility header/finite values')
        loaded.append(value)
    utility, prior = loaded
    require(bool(((prior >= 0) & (prior <= 1)).all()) and bool(((utility >= -1) & (utility <= 1)).all()),
            'Derived utility/prior range failed')
    require(bool(np.array_equal(utility, np.where(support, np.asarray(reward, dtype=np.float64) - prior, 0.))),
            'Derived formula or unsupported neutrality differs')
    return utility, prior, support


def run(request, check=None):
    policy = load_policy(request)
    check = policy.checked(request, check)
    validate_preregistration(request)
    require(set(request['cohorts']) == {'train', 'screen'}, 'Both train512 and screen256 must be built together')
    archive = request['prior_archive']
    require(archive['sha256'] == PRIOR_SHA256 and sha(archive['path']) == PRIOR_SHA256, 'Frozen prior archive pin changed')
    output = Path(request['utility_output']).resolve()
    require(output.is_relative_to(ROOT / 'reward') and not output.exists(), 'Fresh owned utility output required')
    import numpy as np
    check(stage='prior-data-only-extract')
    # NpzFile is lazy: ONLY these three fields are read; no model/CatBoost/provider load.
    with np.load(archive['path'], allow_pickle=False) as data:
        require({'item_ids', 'counts', 'item_mean'} <= set(data.files), 'Missing frozen prior vectors')
        ids, counts, means = (np.array(data[key], copy=True) for key in ('item_ids', 'counts', 'item_mean'))
    require(ids.ndim == counts.ndim == means.ndim == 1 and ids.dtype.kind in 'iu' and counts.dtype.kind in 'iu',
            'Invalid prior vector schema')
    validate_prior_records(ids.tolist(), counts.tolist(), means.tolist())
    lookup = {int(raw): i for i, raw in enumerate(ids)}
    output.mkdir(parents=True)
    summaries = {}
    for name, spec in request['cohorts'].items():
        check(stage='utility-cohort', cohort=name)
        require(sha(Path(spec['reward_dir']) / 'reward.json') == spec['reward_sha256'], 'Parent reward pin changed')
        manifest, arrays = open_minimal_cache(spec, name, np, policy)
        try:
            reward, support, _ = policy.reward_arrays(spec['reward_dir'], arrays, manifest, np)
            require(bool(np.isfinite(reward).all()) and bool(((reward >= 0) & (reward <= 1)).all()), 'Base completion outside [0,1]')
            raw = arrays['candidate_raw_ids']
            require(raw.dtype.kind in 'iu' and arrays['candidate_ids'].dtype.kind in 'iu', 'IDs must be integers')
            # Missing raw IDs are a hard error, never a count0/prior0 substitute.
            require(all(int(value) in lookup for value in np.unique(raw)), 'Raw ID missing from frozen prior catalog')
            indices = np.asarray([[lookup[int(value)] for value in row] for row in raw], dtype=np.int64)
            require(bool(np.array_equal(arrays['candidate_ids'], indices + 3)), 'Raw-ID/native-ID catalog mapping differs')
            require(bool(np.array_equal(counts[indices] > 0, support)), 'Frozen counts>0/support mismatch')
            prior = np.asarray(means[indices], dtype=np.float64)
            utility = np.where(support, np.asarray(reward, dtype=np.float64) - prior, 0.)
            require(bool(np.isfinite(utility).all()) and bool(((utility >= -1) & (utility <= 1)).all()), 'Invalid derived utility')
            summaries[name] = scales(np, policy, arrays['reference_scores'], reward, utility, prior, support)
            cohort = output / name
            cohort.mkdir()
            files = {}
            for filename, values in (('utility.npy', utility), ('prior.npy', prior)):
                path = cohort / filename
                with path.open('xb') as stream:
                    np.save(stream, values, allow_pickle=False)
                files[filename] = dict(path=filename, sha256=sha(path), bytes=path.stat().st_size,
                                       dtype=str(values.dtype), shape=list(values.shape))
            parent = Path(spec['reward_dir']).resolve() / 'reward.json'
            support_path = parent.parent / 'support.npy'
            write(cohort / 'utility.json', dict(contract=CONTRACT, state='completed', cohort=name,
                formula=FORMULA, coefficient=1., range=[-1, 1], files=files,
                cache=dict(path=(Path(spec['cache_dir']).resolve() / 'cache.json').as_posix(), sha256=spec['cache_sha256']),
                parent_reward=dict(path=parent.as_posix(), sha256=spec['reward_sha256']),
                support=dict(path=support_path.as_posix(), sha256=sha(support_path)),
                prior_archive=archive, prior_fields=['item_ids', 'counts', 'item_mean'],
                raw_native_mapping='candidate_native_id == archive_item_index + 3, exact for every pair',
                counts_positive_equals_original_support=True, unsupported_utility='storage sentinel 0; never observed label',
                preregistration_sha256=request['preregistration_sha256'], diagnostics=summaries[name]))
        finally:
            policy.close_cache(arrays)
    require(summaries['train']['nondegenerate_prior'], 'Degenerate prior: no substantive second training utility')
    record = dict(contract=CONTRACT, state='completed', both_cohorts_ready=True, prefit_gate_passed=True,
                  preregistration_sha256=request['preregistration_sha256'], cohorts=summaries,
                  utility_manifests={name: dict(path=(output / name / 'utility.json').as_posix(),
                    sha256=sha(output / name / 'utility.json')) for name in ('train', 'screen')},
                  prior_archive=archive, optimizer_created=False, model_loaded=False, provider_opened=False)
    check(stage='utility-complete')
    write(output / 'prefit.json', record)
    return record
