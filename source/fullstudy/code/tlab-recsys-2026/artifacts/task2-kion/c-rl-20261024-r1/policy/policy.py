"""Current C1070 frozen-encoder contextual bandit. All numerical entrypoints are root-owned.

Import is stdlib-only. No command runs implicitly, no supervised loss exists here.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
P = HERE.parents[1] / 'three-head-262k-20261024-r1'
CONTRACT = 'kion-current-C1070-natural1024-H1-v1'
SCHEMA = '71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d'
SOURCE = ['current_detached_gpt_logit', 'log1p_strict7d_count']
STEPS = (0, 250, 500, 1000)


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


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def load(request, name, path=None):
    path = Path(path or P / (name + '.py')).resolve()
    require(request['source_hashes'].get(path.as_posix()) == sha(path), 'Unpinned source: ' + str(path))
    spec = importlib.util.spec_from_file_location('_c_rl_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def checked(request, check=None):
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None, 'Aware deadline required')
    def run(**fields):
        require(datetime.now(timezone.utc) < deadline, 'Policy deadline exceeded')
        if check:
            check(**fields)
    run(stage='policy-request')
    return run


def precision(torch, request, device):
    mode = request.get('precision', 'cuda-fp16')
    require(mode in ('fp32', 'cuda-fp16'), 'Unknown precision')
    return torch.autocast('cuda', dtype=torch.float16) if mode == 'cuda-fp16' and str(device).startswith('cuda') else nullcontext()


def open_frozen_context(bank, request, check):
    """Adapt only the legacy reader stage; native operation authorization is unchanged.

    Old P bank's readonly validation-graph route is named ``evaluate``. All new
    operations reuse that frozen serving context; none may authorize graph builds.
    Keep the original graph path/source pins and let the original reader verify
    its exact existing path. Never pass the outer mutable request to that reader.
    """
    require(request.get('allow_validation_graph_build', False) is False,
            'Current C operations forbid validation graph builds')
    require(bool(request.get('validation_graph_cache')), 'Pinned existing validation graph required')
    legacy_request = dict(request, stage='evaluate', allow_validation_graph_build=False)
    return bank.open_context(legacy_request, check)


def groups(rows, origins, maximum=16):
    """Date-homogeneous provider calls, preserving selection order by scatter."""
    by_date = {}
    for position, row in enumerate(rows):
        by_date.setdefault(int(origins[row]), []).append((position, int(row)))
    for day in sorted(by_date):
        items = by_date[day]
        for start in range(0, len(items), maximum):
            yield items[start:start + maximum]


def rank(ids, scores, raw_ids, k=10):
    require(len(ids) == len(scores) == len(raw_ids), 'Ranking shape mismatch')
    require(all(math.isfinite(float(v)) for v in scores), 'Nonfinite rank score')
    return sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(raw_ids[i])))[:k]


def _descriptor(path, array):
    return dict(path=path.name, sha256=sha(path), bytes=path.stat().st_size,
                dtype=str(array.dtype), shape=list(array.shape))


def prepare_cache(request, check=None):
    """Build one declared train/screen cache; caller closes this stage before RM startup."""
    check = checked(request, check)
    bank, heads = load(request, 'bank'), load(request, 'heads')
    heads.preflight_heads(request)
    selection_path = Path(request['cohort_selection'])
    require(sha(selection_path) == request['cohort_selection_sha256'], 'Changed cohort selection')
    selection = read(selection_path)
    require(selection.get('exclusions_verified') is True, 'RM select/cal user/context exclusion proof required')
    name = request['cohort_name']
    require(name in ('train', 'screen'), 'Unknown cohort')
    spec = selection['cohorts'][name]
    rows, split = spec['row_indices'], spec['split']
    require(split == ('train' if name == 'train' else 'val'), 'Cohort split changed')
    require(0 < len(rows) <= (512 if name == 'train' else 256) and len(set(rows)) == len(rows), 'Invalid cohort cap/duplicates')
    output = Path(request['cache_dir'])
    require(not output.exists(), 'Fresh cache directory required')
    ctx = open_frozen_context(bank, request, check)
    maps = {}
    complete = False
    try:
        np, torch, teacher, provider = (ctx[k] for k in ('np', 'torch', 'model', 'provider'))
        train, table, candidates, device = (ctx[k] for k in ('train', 'table', 'candidates', 'device'))
        require(ctx['feature_schema_sha256'] == SCHEMA and ctx['source_columns'] == SOURCE, 'C schema/source mismatch')
        head = heads.load_heads(request, torch, device)['C'].eval()
        for param in head.parameters():
            param.requires_grad_(False)
        output.mkdir(parents=True)
        n = len(rows)
        shapes = {'row_indices': ('int64', (n,)), 'user_ids': ('int64', (n,)),
                  'origins': ('int32', (n,)), 'x': ('int64', (n, 64)), 'lengths': ('int64', (n,)),
                  'user_features': ('float32', (n, teacher.user_feature_dim)),
                  'target_ids': ('int32', (n,)), 'encoded': ('float32', (n, 256)),
                  'candidate_ids': ('int32', (n, 1024)), 'candidate_raw_ids': ('int64', (n, 1024)),
                  'candidate_sources': ('float32', (n, 1024, 14)), 'source_values': ('float32', (n, 1024, 2)),
                  'base_logits': ('float32', (n, 1024)), 'reference_scores': ('float32', (n, 1024)),
                  'raw_features': ('float32', (n, 1024, 1068)),
                  'item_table': ('float32', tuple(table.shape))}
        for key, (dtype, shape) in shapes.items():
            maps[key] = np.lib.format.open_memmap(output / (key + '.npy'), mode='w+', dtype=dtype, shape=shape)
        head_table = table.detach().float()
        maps['item_table'][:] = head_table.cpu().numpy()
        maps['row_indices'][:] = rows
        data = provider.splits[split]
        # Validation x is frozen as-of Aug9; target outcome dates Aug9-15 are not query clocks.
        bound_states = {int(state['row_index']): state for state in selection['states'][name]}
        require(set(bound_states) == set(rows), 'Selection state keys differ')
        origins = {row: int(bound_states[row]['origin']) for row in rows}
        if split == 'val':
            require(set(origins.values()) == {int(candidates.validation_origin)}, 'Validation origin must remain frozen Aug9')
        else:
            require(all(origins[row] == int(candidates.origins[row]) for row in rows), 'Train query origin differs')
        with torch.inference_mode():
            for batch in groups(rows, origins):
                check(stage='natural-cache-batch', cohort=name, completed=int(sum(1 for _ in batch)))
                dest, source_rows = zip(*batch)
                dest, source_rows = np.asarray(dest), np.asarray(source_rows)
                x, lengths, y, user = train.batch(provider, source_rows, split, device)
                padded_x = np.zeros((len(batch), 64), dtype=np.int64)
                padded_x[:, :x.shape[1]] = x.cpu().numpy()
                require(all(int(data['user_id'][row]) == int(bound_states[row]['user_id'])
                            and int(lengths[i]) == int(bound_states[row]['length']) for i, row in enumerate(source_rows)), 'Selected state identity differs')
                with train.legacy.autocast(device):
                    q = teacher.encode_context(x, lengths, user, candidate_table=table)
                    logits = teacher.retrieval_logits(q, table)
                ids, source = candidates.pool(source_rows, split, logits)
                require(ids.shape == (len(batch), 1024) and source.shape == (len(batch), 1024, 14), 'Natural1024 shape')
                require(bool(np.all((ids >= 3) & (ids < teacher.vocab_size))) and all(len(set(row)) == 1024 for row in ids.tolist()), 'Invalid candidate IDs')
                require(bool(np.array_equal(source[..., 12], logits.gather(1, torch.as_tensor(ids, device=device)).float().cpu().numpy())), 'GPT source12 differs from base')
                for key, values in [('x', padded_x), ('lengths', lengths.cpu().numpy()),
                                    ('user_features', user.float().cpu().numpy()),
                                    ('user_ids', data['user_id'][source_rows]), ('origins', [origins[int(row)] for row in source_rows]),
                                    ('target_ids', y.cpu().numpy()), ('encoded', q.float().cpu().numpy()),
                                    ('candidate_ids', ids), ('candidate_raw_ids', candidates.raw_item_ids(ids)),
                                    ('candidate_sources', source), ('source_values', source[..., 12:14])]:
                    maps[key][dest] = values
                for left in range(0, 1024, 64):
                    check(stage='natural-cache-chunk', candidate_column=left)
                    raw = provider.candidate_features(source_rows, ids[:, left:left+64], split)
                    require(raw.shape == (len(batch), 64, 1068) and bool(np.isfinite(raw).all()), 'Invalid raw1068')
                    tokens = torch.as_tensor(ids[:, left:left+64], device=device, dtype=torch.long)
                    base = logits.gather(1, tokens).float()
                    with precision(torch, request, device):
                        scores = head(q.float(), head_table[tokens], base,
                                      torch.as_tensor(raw, device=device, dtype=torch.float32),
                                      torch.as_tensor(source[:, left:left+64, 12:14], device=device))
                    require(bool(torch.isfinite(scores).all()), 'Nonfinite C cache scores')
                    maps['raw_features'][dest, left:left+64] = raw
                    maps['base_logits'][dest, left:left+64] = base.cpu().numpy()
                    maps['reference_scores'][dest, left:left+64] = scores.float().cpu().numpy()
                for array in maps.values():
                    array.flush()
        descriptors = {key: _descriptor(output / (key + '.npy'), value) for key, value in maps.items()}
        record = dict(contract=CONTRACT, state='completed', cohort=name, split=split, rows=n,
                      legacy_context_stage='evaluate', operation_stage=request.get('stage'),
                      candidate_width=1024, raw_width=1068, source_columns=SOURCE,
                      feature_schema_sha256=SCHEMA, temperature=1.0, precision=request.get('precision', 'cuda-fp16'),
                      no_target_injection=True, no_seen_filter=True, tie_break='score-desc/rawID-asc',
                      selection_sha256=sha(selection_path), teacher=ctx['teacher'],
                      C_checkpoint_sha256=sha(Path(request['head_dir']) / 'C/head.pt'),
                      arrays=descriptors, candidate_contract=ctx['candidate_contract'])
        complete = True
    finally:
        for value in maps.values():
            value.flush()
            value._mmap.close()
        cleanup = bank.close_context(ctx)
        require(cleanup.get('teardown_completed') is True and cleanup.get('errors') == [], 'Context teardown failed')
    require(complete, 'Incomplete cache')
    write(output / 'cache.json', record)
    return record


def open_cache(path, np):
    root = Path(path)
    manifest = read(root / 'cache.json')
    require(manifest.get('contract') == CONTRACT and manifest.get('state') == 'completed', 'Invalid cache contract')
    require(manifest.get('feature_schema_sha256') == SCHEMA and manifest.get('source_columns') == SOURCE
            and manifest.get('candidate_width') == 1024 and manifest.get('no_target_injection') is True
            and manifest.get('no_seen_filter') is True, 'Cache policy schema differs')
    arrays = {}
    for key, entry in manifest['arrays'].items():
        file = (root / entry['path']).resolve()
        require(file.is_relative_to(root.resolve()) and file.stat().st_size == entry['bytes'] and sha(file) == entry['sha256'], 'Changed cache array: ' + key)
        value = np.load(file, mmap_mode='r', allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] and list(value.shape) == entry['shape'], 'Changed NPY header')
        arrays[key] = value
    return manifest, arrays


def close_cache(arrays):
    for value in arrays.values():
        if getattr(value, '_mmap', None) is not None:
            value._mmap.close()


def reward_arrays(path, cache, manifest, np):
    """Bridge receipt pins exact observed state arrays and the completed cache manifest."""
    root = Path(path)
    metadata = read(root / 'reward.json')
    require(metadata.get('ready') is True and metadata.get('contract') == 'current-C-local-RM-r2-bridge-v1', 'Incomplete reward bridge')
    cache_path = Path(metadata['cache']['path'])
    require(sha(cache_path) == metadata['cache']['sha256'] and read(cache_path) == manifest, 'RM cache binding differs')
    for key in ('user_ids', 'origins', 'x', 'lengths', 'candidate_raw_ids'):
        entry = metadata['state_arrays'][key]
        require(sha(entry['path']) == entry['sha256'] == manifest['arrays'][key]['sha256'], 'RM state binding differs: ' + key)
    values = []
    for filename in ('rewards.npy', 'support.npy', 'fallback.npy'):
        entry, file = metadata['files'][filename], root / filename
        require(sha(file) == entry['sha256'], 'RM output changed')
        value = np.load(file, allow_pickle=False)
        require(str(value.dtype) == entry['dtype'] and list(value.shape) == entry['shape'], 'RM header differs')
        values.append(value)
    reward, support, fallback = values
    require(support.dtype == np.bool_ and fallback.dtype == np.bool_, 'RM support flags must be boolean')
    require(reward.shape == support.shape == fallback.shape == cache['candidate_ids'].shape, 'RM shape mismatch')
    require(bool(np.isfinite(reward[support]).all()) and bool(((reward[support] >= 0) & (reward[support] <= 1)).all()), 'Invalid supported calibrated rewards')
    require(not bool((support & fallback).any()), 'Fallback cannot be optimized')
    return reward, support, fallback


def distribution(np, scores, temperature=1.0):
    require(math.isfinite(temperature) and temperature > 0, 'Invalid temperature')
    scores = np.asarray(scores, dtype=np.float64) / temperature
    require(bool(np.isfinite(scores).all()), 'Nonfinite policy scores')
    scores -= scores.max(axis=1, keepdims=True)
    logp = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    return logp, np.exp(logp)


def effective_advantage(np, reference_probs, rewards, support):
    """Neutral unsupported actions retained in the normalization, never reward-zero labels."""
    mass = (reference_probs * support).sum(axis=1, keepdims=True)
    clean = np.where(support, rewards, 0.0)
    baseline = np.divide((reference_probs * clean).sum(axis=1, keepdims=True), mass,
                         out=np.zeros_like(mass), where=mass > 0)
    return np.where(support, clean - baseline, 0.0), mass[:, 0] > 0


def diagnostics(np, scores, arrays, rewards, support, reference_logp, temperature=1.0):
    logp, probs = distribution(np, scores, temperature)
    advantage, active = effective_advantage(np, np.exp(reference_logp), rewards, support)
    top = np.stack([rank(ids, s, raw) for ids, s, raw in zip(arrays['candidate_ids'], scores, arrays['candidate_raw_ids'])])
    top_ids = np.take_along_axis(arrays['candidate_ids'], top, axis=1)
    targets = arrays['target_ids']
    hits = top_ids == targets[:, None]
    support_mass = (probs * support).sum(1)
    supported_reward = (probs * np.where(support, rewards, 0.0)).sum(1)
    top_support = np.take_along_axis(support, top, axis=1)
    top_reward = np.take_along_axis(np.where(support, rewards, 0.0), top, axis=1)
    _, counts = np.unique(top_ids, return_counts=True)
    _, top1_counts = np.unique(top_ids[:, 0], return_counts=True)
    return dict(rows=len(scores), map10=float((hits/np.arange(1, 11)).sum(1).mean()),
                ndcg10=float((hits/np.log2(np.arange(2, 12))).sum(1).mean()), recall10=float(hits.any(1).mean()),
                expected_effective_reward=float((probs*advantage).sum(1).mean()),
                supported_conditional_raw_reward=float(supported_reward.sum()/support_mass.sum()) if support_mass.sum() else None,
                supported_probability_denominator=float(support_mass.sum()),
                supported_top10_raw_reward=float(top_reward.sum()/top_support.sum()) if top_support.sum() else None,
                supported_top10_denominator=int(top_support.sum()), unsupported_mass=float((1-support_mass).mean()),
                exact_KL=float((probs*(logp-reference_logp)).sum(1).mean()),
                entropy=float(-(probs*logp).sum(1).mean()),
                top1_max_share=float(top1_counts.max()/len(scores)), top10_HHI=float(((counts/counts.sum())**2).sum()),
                effective_catalog=float(1/((counts/counts.sum())**2).sum()), states_without_support=int((~active).sum()))


def inference_frontier(request, check=None):
    check = checked(request, check)
    prereg = read(request['preregistration'])
    require(sha(request['preregistration']) == request['preregistration_sha256'], 'Preregistration pin differs')
    require(prereg['frontier_betas'] == [0.1, 0.5] and prereg['temperature'] == 1.0, 'Predeclared frontier required')
    import numpy as np
    manifest, arrays = open_cache(request['cache_dir'], np)
    try:
        reward, support, fallback = reward_arrays(request['reward_dir'], arrays, manifest, np)
        logp0, probs0 = distribution(np, arrays['reference_scores'])
        advantage, _ = effective_advantage(np, probs0, reward, support)
        controls = {'C': np.asarray(arrays['reference_scores']),
                    'RM-only': np.where(support, reward, -1.0)}
        controls.update({f'tilt-beta{beta}': logp0 + advantage/beta for beta in prereg['frontier_betas']})
        results = {}
        for name, scores in controls.items():
            check(stage='inference-frontier', arm=name)
            results[name] = diagnostics(np, scores, arrays, reward, support, logp0)
        record = dict(contract=CONTRACT, state='completed', cache_sha256=sha(Path(request['cache_dir'])/'cache.json'),
                      reward_sha256=sha(Path(request['reward_dir'])/'reward.json'), preregistration_sha256=sha(request['preregistration']),
                      arms=results, cohort_rows=manifest['rows'], fallback_pairs=int(fallback.sum()),
                      RM_only_definition='supported reward, unsupported score -1; original pool retained',
                      optimizer_created=False)
        write(request['frontier_output'], record)
        return record
    finally:
        close_cache(arrays)


def tensor_batch(arrays, positions, table, torch, np, device):
    def take(key, dtype=None):
        return torch.as_tensor(np.array(arrays[key][positions], copy=True), device=device, dtype=dtype)
    ids = take('candidate_ids', torch.long)
    return take('encoded', torch.float32), table[ids].detach(), take('base_logits', torch.float32), take('raw_features', torch.float32), take('source_values', torch.float32)


def exact_loss(torch, scores, reference_scores, advantages, beta, active, temperature=1.0):
    """Zero-support states contribute neither reward nor KL updates."""
    logp = torch.log_softmax(scores.float()/temperature, dim=-1)
    logq = torch.log_softmax(reference_scores.detach().float()/temperature, dim=-1)
    probs = logp.exp()
    objective = -(probs*advantages.detach()).sum(-1) + beta*(probs*(logp-logq)).sum(-1)
    return (objective*active).sum()/active.sum().clamp_min(1)


def select_beta(prereg, frontier):
    """Frozen inference choice, independent of the subsequent optimization curve."""
    baseline = frontier['arms']['C']
    eligible = []
    gates = prereg['gates']
    for beta in prereg['frontier_betas']:
        arm = frontier['arms'][f'tilt-beta{beta}']
        if (arm['map10'] - baseline['map10'] >= gates['map_delta_min']
                and arm['exact_KL'] <= gates['KL_max']
                and arm['unsupported_mass'] - baseline['unsupported_mass'] <= gates['unsupported_mass_increase_max']
                and arm['top1_max_share'] - baseline['top1_max_share'] <= gates['top1_share_increase_max']):
            eligible.append((arm['expected_effective_reward'], beta))
    return max(eligible)[1] if eligible else 0.5


def normalized_preregistration(value):
    """Read the campaign's actual nested preregistration without rewriting its pin."""
    require(value.get('contract') == 'kion-C3-RL-preregistered-v1'
            and value.get('before_new_rewards_or_frontier') is True, 'Frozen campaign preregistration required')
    opt, gates = value['optimizer'], value['feasibility']
    require(opt['name'] == 'Adam' and opt['grad_clip'] == 1 and value['checkpoint_steps'] == list(STEPS), 'Optimizer contract differs')
    result = dict(value)
    result.update(updates=opt['updates'], batch_size=opt['batch_size'], weight_decay=opt['weight_decay'],
                  supervised_loss_weight=opt['alpha_ce'], learning_rate=opt['lr'], seed=value['sampler_seed'],
                  parity_atol=0.003, parity_rtol=0.001,
                  gates=dict(map_delta_min=gates['mean_MAP10_delta_min'], KL_max=gates['mean_KL_nats_max'],
                    unsupported_mass_increase_max=gates['unsupported_probability_mass_increase_max'],
                    top1_share_increase_max=gates['top1_item_max_share_increase_max']))
    return result


def head_scores(head, args, torch):
    """Preserve the original candidate chunk64 head arithmetic during training too."""
    q, candidates, base, raw, source = args
    return torch.cat([head(q, candidates[:,left:left+64], base[:,left:left+64],
                           raw[:,left:left+64], source[:,left:left+64])
                      for left in range(0, base.shape[1], 64)], dim=1)


def train(request, check=None):
    """Exact H1; bounded SGD cursor. No target labels ever enter the objective."""
    check = checked(request, check)
    prereg = normalized_preregistration(read(request['preregistration']))
    require(sha(request['preregistration']) == request['preregistration_sha256'], 'Preregistration pin differs')
    frontier = read(request['frontier_output'])
    require(frontier['state'] == 'completed' and sha(request['frontier_output']) == request['frontier_sha256'], 'Complete inference frontier required before optimizer')
    require(frontier['preregistration_sha256'] == request['preregistration_sha256'], 'Frontier preregistration differs')
    beta, updates = select_beta(prereg, frontier), prereg['updates']
    require(beta in prereg['frontier_betas'] and updates in (250, 500, 1000), 'One declared beta and bounded updates required')
    require(prereg['temperature'] == 1.0 and prereg['batch_size'] == 16 and prereg['weight_decay'] == 0
            and prereg['supervised_loss_weight'] == 0, 'Head-only B16, zero-decay, no CE contract required')
    if request.get('precision', 'cuda-fp16') == 'cuda-fp16':
        require(request.get('amp_validation_passed') is True, 'AMP needs root numerical parity/finite-gradient validation')
    import numpy as np
    import torch
    heads = load(request, 'heads')
    output = Path(request['rl_output'])
    require(not output.exists(), 'Fresh RL directory required')
    manifest, arrays = open_cache(request['cache_dir'], np)
    device = torch.device(request.get('device', 'cuda'))
    try:
        require(manifest['cohort'] == 'train' and manifest['precision'] == request.get('precision', 'cuda-fp16'), 'Matched training cache precision required')
        reward, support, _ = reward_arrays(request['reward_dir'], arrays, manifest, np)
        _, p0 = distribution(np, arrays['reference_scores'])
        advantages, active = effective_advantage(np, p0, reward, support)
        require(bool(active.any()), 'No supported train states')
        model = heads.load_heads(request, torch, device)['C'].eval()
        require(sha(Path(request['head_dir'])/'C/head.pt') == manifest['C_checkpoint_sha256'], 'C cache lineage differs')
        reference = copy.deepcopy(model).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        table = torch.as_tensor(np.array(arrays['item_table'], copy=True), device=device).detach()
        output.mkdir(parents=True)
        # Same head arithmetic, candidate IDs, source and precision; check all states.
        maximum_error = 0.0
        with torch.inference_mode():
            for start in range(0, len(active), 16):
                positions = list(range(start, min(start+16, len(active))))
                args = tensor_batch(arrays, positions, table, torch, np, device)
                with precision(torch, request, device):
                    scores, ref_scores = head_scores(model, args, torch), head_scores(reference, args, torch)
                require(torch.equal(scores, ref_scores), 'Update0 actor/reference mismatch')
                expected = np.asarray(arrays['reference_scores'][positions])
                actual = scores.float().cpu().numpy()
                maximum_error = max(maximum_error, float(np.abs(actual-expected).max()))
                require(bool(np.allclose(actual, expected, rtol=prereg['parity_rtol'], atol=prereg['parity_atol'])), 'Update0 cached C score mismatch')
                for i, pos in enumerate(positions):
                    require(rank(arrays['candidate_ids'][pos], actual[i], arrays['candidate_raw_ids'][pos]) ==
                            rank(arrays['candidate_ids'][pos], expected[i], arrays['candidate_raw_ids'][pos]), 'Update0 top10 differs')
        for p in model.parameters():
            p.requires_grad_(True)
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = dict(contract=CONTRACT, state='running', beta=beta, updates=0, checkpoints=[],
                      C_checkpoint_sha256=manifest['C_checkpoint_sha256'], cache_sha256=sha(Path(request['cache_dir'])/'cache.json'),
                      reward_sha256=sha(Path(request['reward_dir'])/'reward.json'), preregistration_sha256=request['preregistration_sha256'],
                      frontier_sha256=request['frontier_sha256'], update0_max_abs_error=maximum_error,
                      precision=request.get('precision', 'cuda-fp16'), objective='exact supported centered advantage minus beta KL',
                      frozen_encoder=True, supervised_updates=0, weight_decay=0, batch_size=16)
        def save(step):
            path = output / f'head-{step}.pt'
            torch.save(dict(contract=CONTRACT, step=step, state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            lineage=dict(record)), path)
            record['checkpoints'].append(dict(step=step, path=path.as_posix(), sha256=sha(path)))
        save(0)
        # Optimizer creation happens only after complete update0/source/schema closure.
        optimizer = torch.optim.Adam(model.parameters(), lr=prereg['learning_rate'], weight_decay=0)
        use_amp = request.get('precision', 'cuda-fp16') == 'cuda-fp16' and device.type == 'cuda'
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024)
        rng = random.Random(prereg['seed'])
        eligible = [i for i, ok in enumerate(active) if ok]
        history = []
        for step in range(1, updates+1):
            check(stage='exact-H1-update', step=step)
            positions = [rng.choice(eligible) for _ in range(16)]
            args = tensor_batch(arrays, positions, table, torch, np, device)
            refs = torch.as_tensor(np.array(arrays['reference_scores'][positions], copy=True), device=device)
            adv = torch.as_tensor(advantages[positions], device=device, dtype=torch.float32)
            valid = torch.ones(16, device=device)
            optimizer.zero_grad(set_to_none=True)
            with precision(torch, request, device):
                scores = head_scores(model, args, torch)
                loss = exact_loss(torch, scores, refs, adv, beta, valid)
            require(bool(torch.isfinite(loss)), 'Nonfinite RL loss; no blind retry')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite RL gradient')
            scaler.step(optimizer)
            scaler.update()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite updated parameters')
            record['updates'] = step
            if step % 25 == 0:
                history.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm)))
            if step in STEPS:
                save(step)
        require(all(torch.equal(v.detach().cpu(), initial[k]) for k,v in reference.state_dict().items()), 'Reference weights changed')
        record.update(state='completed', history=history,
                      parameter_l2_delta=math.sqrt(float(sum(((v.detach().cpu()-initial[k])**2).sum() for k,v in model.state_dict().items()))))
        write(output/'training.json', record)
        return record
    finally:
        close_cache(arrays)


def load_rl_head(request, torch, device, checkpoint):
    heads = load(request, 'heads')
    head = heads.load_heads(request, torch, device)['C']
    require(sha(checkpoint) == request['rl_checkpoint_sha256'], 'RL checkpoint pin changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(payload.get('contract') == CONTRACT and payload.get('step') in STEPS, 'Wrong RL checkpoint')
    require(payload['lineage']['C_checkpoint_sha256'] == sha(Path(request['head_dir'])/'C/head.pt'), 'Wrong C ancestor')
    head.load_state_dict(payload['state_dict'], strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def score_checkpoint(request, check=None):
    """Cheap screen cache scoring; all checkpoints reuse the one raw feature cache."""
    check = checked(request, check)
    import numpy as np
    import torch
    manifest, arrays = open_cache(request['cache_dir'], np)
    try:
        require(manifest['precision'] == request.get('precision', 'cuda-fp16'), 'Screen/cache precision differs')
        device = torch.device(request.get('device', 'cuda'))
        head = load_rl_head(request, torch, device, request['rl_checkpoint'])
        table = torch.as_tensor(np.array(arrays['item_table'], copy=True), device=device).detach()
        scores = np.empty(arrays['reference_scores'].shape, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, len(scores), 16):
                check(stage='checkpoint-screen', rows=start)
                pos = list(range(start, min(start+16, len(scores))))
                with precision(torch, request, device):
                    out = head_scores(head, tensor_batch(arrays, pos, table, torch, np, device), torch)
                scores[pos] = out.float().cpu().numpy()
        reward, support, _ = reward_arrays(request['reward_dir'], arrays, manifest, np)
        logp0, _ = distribution(np, arrays['reference_scores'])
        result = diagnostics(np, scores, arrays, reward, support, logp0)
        result.update(checkpoint_sha256=request['rl_checkpoint_sha256'], cache_sha256=sha(Path(request['cache_dir'])/'cache.json'))
        write(request['screen_output'], result)
        return result
    finally:
        close_cache(arrays)


def evaluate_selected(request, check=None):
    """Original fixed4096 evaluator protocol; score latest/selected in ONE raw pass.

    Reuse original selection, natural-pool generator, raw chunks and metric helpers.
    Original C is also recomputed for explicit cross-batch arithmetic parity evidence.
    """
    check = checked(request, check)
    bank, evaluator, heads = (load(request, name) for name in ('bank', 'evaluate', 'heads'))
    rows = evaluator.validate_selection(request['validation_selection'], request['validation_selection_sha256'])
    heads.preflight_heads(request)
    output = Path(request['full_evaluation_dir'])
    require(not output.exists(), 'Fresh full evaluation directory required')
    ctx = open_frozen_context(bank, request, check)
    complete = False
    try:
        np, torch, train, teacher, provider, candidates, table, device = (
            ctx[k] for k in ('np', 'torch', 'train', 'model', 'provider', 'candidates', 'table', 'device'))
        models = {'C-recomputed': heads.load_heads(request, torch, device)['C']}
        checkpoint_pins = {}
        for name, checkpoint in request['evaluation_checkpoints'].items():
            require(name in ('latest', 'selected'), 'Only declared latest/selected full evaluation')
            copy_request = dict(request, rl_checkpoint_sha256=checkpoint['sha256'])
            models[name] = load_rl_head(copy_request, torch, device, checkpoint['path'])
            checkpoint_pins[name] = checkpoint
        require('latest' in models, 'Actual latest checkpoint must be analyzed')
        for model in models.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        original_root = Path(request['original_evaluation_dir'])
        original_manifest = read(original_root/'evaluation.json')
        require(sha(original_root/'evaluation.json') == request['original_evaluation_sha256'], 'Original evaluation pin differs')
        original = {}
        for key in ('pool_ids', 'C_scores'):
            entry = original_manifest['arrays'][key]
            path = original_root/entry['path']
            require(sha(path) == entry['sha256'], 'Original evaluation payload differs')
            original[key] = np.load(path, mmap_mode='r', allow_pickle=False)
        output.mkdir(parents=True)
        scores = {name: np.empty((4096, 1024), np.float32) for name in models}
        tops = {name: np.empty((4096, 10), np.int32) for name in models}
        indices = np.asarray(rows, dtype=np.int64)
        data = provider.splits['val']
        targets = np.asarray(data['y'][indices])
        known = (targets >= 3) & (targets < teacher.vocab_size)
        coverage = np.zeros(4096, bool)
        head_table = table.detach().float()
        parity = dict(max_abs_error=0.0, score_tolerance_passed=True, top10_changed_rows=0,
                      changes_with_margin_above_tolerance=0, atol=request['parity_atol'], rtol=request['parity_rtol'])
        with torch.inference_mode():
            for start in range(0, 4096, 128):
                check(stage='selected-full4096', rows=start)
                batch_rows = indices[start:start+128]
                x, lengths, y, user = train.batch(provider, batch_rows, 'val', device)
                with train.legacy.autocast(device):
                    q = teacher.encode_context(x, lengths, user, candidate_table=table)
                    logits = teacher.retrieval_logits(q, table)
                ids, source = candidates.pool(batch_rows, 'val', logits)
                require(bool(np.array_equal(ids, original['pool_ids'][start:start+128])), 'Full evaluation candidate IDs changed')
                coverage[start:start+128] = (ids == targets[start:start+128, None]).any(1) & known[start:start+128]
                for left in range(0, 1024, 64):
                    check(stage='full4096-raw-chunk', rows=start, candidate_column=left)
                    raw = provider.candidate_features(batch_rows, ids[:, left:left+64], 'val')
                    token = torch.as_tensor(ids[:, left:left+64], device=device, dtype=torch.long)
                    base = logits.gather(1, token).float()
                    args = (q.float(), head_table[token], base, torch.as_tensor(raw, device=device, dtype=torch.float32),
                            torch.as_tensor(source[:, left:left+64, 12:14], device=device, dtype=torch.float32))
                    with precision(torch, request, device):
                        for name, model in models.items():
                            values = model(*args)
                            require(bool(torch.isfinite(values).all()), 'Nonfinite evaluation scores')
                            scores[name][start:start+128, left:left+64] = values.float().cpu().numpy()
                raw_ids = candidates.raw_item_ids(ids)
                expected, actual = original['C_scores'][start:start+128], scores['C-recomputed'][start:start+128]
                parity['max_abs_error'] = max(parity['max_abs_error'], float(np.abs(expected-actual).max()))
                parity['score_tolerance_passed'] &= bool(np.allclose(expected, actual, atol=request['parity_atol'], rtol=request['parity_rtol']))
                for i in range(len(ids)):
                    old_rank = rank(ids[i], expected[i], raw_ids[i])
                    new_rank = rank(ids[i], actual[i], raw_ids[i])
                    if old_rank != new_rank:
                        parity['top10_changed_rows'] += 1
                        # Any changed pair with a saved-score gap above both pointwise bounds is a protocol error.
                        for a, b in zip(old_rank, new_rank):
                            tolerance = 2*request['parity_atol'] + request['parity_rtol']*(abs(float(expected[i,a]))+abs(float(expected[i,b])))
                            if abs(float(expected[i,a]-expected[i,b])) > tolerance:
                                parity['changes_with_margin_above_tolerance'] += 1
                    for name in models:
                        tops[name][start+i] = ids[i, rank(ids[i], scores[name][start+i], raw_ids[i])]
        require(parity['score_tolerance_passed'] and parity['changes_with_margin_above_tolerance'] == 0, 'Original C parity failed; stop comparison')
        descriptors, metrics = {}, {}
        for name in models:
            descriptors[name+'_scores'] = evaluator._save(np, output, name+'_scores', scores[name])
            descriptors[name+'_top10'] = evaluator._save(np, output, name+'_top10', tops[name])
            metrics[name] = evaluator._summary(np, tops[name], targets, known, coverage)
        record = dict(contract=CONTRACT, state='completed', metric_denominator=4096, raw_feature_calls=512,
                      legacy_context_stage='evaluate', operation_stage=request.get('stage'),
                      latest_and_selected_share_raw_pass=True, original_C_parity=parity,
                      baseline_for_matched_comparison='C-recomputed' if parity['top10_changed_rows'] else 'saved-C-identical-ranks',
                      selection_sha256=sha(request['validation_selection']), original_evaluation_sha256=request['original_evaluation_sha256'],
                      checkpoints=checkpoint_pins, metrics=metrics, arrays=descriptors,
                      no_target_injection=True, no_seen_filter=True, final_test_reads=0)
        complete = True
    finally:
        for value in locals().get('original', {}).values():
            value._mmap.close()
        cleanup = bank.close_context(ctx)
        require(cleanup.get('teardown_completed') is True and cleanup.get('errors') == [], 'Full evaluation cleanup failed')
    require(complete, 'Incomplete full evaluation')
    write(output/'evaluation.json', record)
    return record


def rollout(request, check=None):
    """Root-owned cached initial state + live frozen GPT; RM remains original observed s0."""
    check = checked(request, check)
    bank = load(request, 'bank')
    live = load(request, 'liveprefix', HERE/'liveprefix.py')
    ctx = open_frozen_context(bank, request, check)
    arrays = {}
    try:
        np, torch, device = ctx['np'], ctx['torch'], ctx['device']
        manifest, arrays = open_cache(request['cache_dir'], np)
        require(manifest['precision'] == request.get('precision','cuda-fp16'), 'Rollout/cache head precision differs')
        ctx['head_autocast'] = lambda: precision(torch, request, device)
        reward, support, _ = reward_arrays(request['reward_dir'], arrays, manifest, np)
        heads = load(request, 'heads')
        reference = heads.load_heads(request, torch, device)['C']
        for p in reference.parameters():
            p.requires_grad_(False)
        models = {'C': reference}
        for name, checkpoint in request['rollout_checkpoints'].items():
            require(name in ('latest', 'selected'), 'Unexpected rollout checkpoint')
            models[name] = load_rl_head(dict(request, rl_checkpoint_sha256=checkpoint['sha256']), torch, device, checkpoint['path'])
        results, rm_controls = [], []
        for position in request['rollout_positions']:
            require(0 <= position < manifest['rows'], 'Rollout position outside frozen cache')
            state = {key: np.array(value[position], copy=True) for key, value in arrays.items() if key != 'item_table'}
            ordered = rank(state['candidate_ids'], np.where(support[position], reward[position], -1.0), state['candidate_raw_ids'], k=5)
            rm_controls.append(dict(position=position, definition='fixed observed s0 RM-greedy unique top5, unsupported last',
                generated_ids=[int(state['candidate_ids'][i]) for i in ordered],
                generated_raw_ids=[int(state['candidate_raw_ids'][i]) for i in ordered],
                rewards=[float(reward[position,i]) if support[position,i] else None for i in ordered],
                supported=[bool(support[position,i]) for i in ordered]))
            for name, head in models.items():
                for mode in request.get('rollout_selection', ['greedy', 'categorical']):
                    for mask in request.get('repeat_masks', ['none', 'exclude-generated']):
                        check(stage='liveprefix-five', position=position, head=name, selection=mode, repeat_mask=mask)
                        value = live.rollout_state(ctx, head, reference, state, reward[position], support[position],
                            repeat_mask=mask, selection=mode, seed=request['seed']+position, temperature=1.0, steps=5)
                        results.append(dict(position=position, head=name, selection=mode, repeat_mask=mask, result=value))
        record = dict(contract=CONTRACT, state='completed', cache_sha256=sha(Path(request['cache_dir'])/'cache.json'),
                      legacy_context_stage='evaluate', operation_stage=request.get('stage'),
                      reward_sha256=sha(Path(request['reward_dir'])/'reward.json'), results=results, RM_greedy_controls=rm_controls,
                      five_step_training=False, synthetic_prefix_only=True, RM_observed_s0_fixed=True)
        write(request['rollout_output'], record)
        return record
    finally:
        close_cache(arrays)
        cleanup = bank.close_context(ctx)
        require(cleanup.get('teardown_completed') is True and cleanup.get('errors') == [], 'Rollout cleanup failed')


def smoke(request, check=None):
    """Root-only tiny synthetic numerical proof; no providers/checkpoints/real data."""
    check = checked(request, check)
    import torch
    heads = load(request, 'heads')
    torch.set_num_threads(2)
    device = torch.device(request.get('device', 'cpu'))
    torch.manual_seed(20261024)
    head = heads.make_heads(torch, device=device)['C'].eval()
    reference = copy.deepcopy(head)
    for p in reference.parameters():
        p.requires_grad_(False)
    args = (torch.randn(2,256,device=device), torch.randn(2,7,256,device=device),
            torch.randn(2,7,device=device), torch.randn(2,7,1068,device=device), torch.randn(2,7,2,device=device))
    before = {k:v.detach().clone() for k,v in head.state_dict().items()}
    with precision(torch, request, device):
        initial = head(*args)
        refs = reference(*args).detach()
    require(torch.equal(initial, refs), 'Synthetic update0 copy mismatch')
    adv = torch.tensor([[.2,-.2,0,0,0,0,0],[-.1,.1,0,0,0,0,0]],device=device)
    loss = exact_loss(torch, initial, refs, adv, .5, torch.ones(2,device=device))
    loss.backward()
    require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in head.parameters()), 'Synthetic nonfinite gradient')
    require(any(p.grad is not None and bool((p.grad != 0).any()) for p in head.parameters()), 'Synthetic zero RL gradients')
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-5, weight_decay=0)
    optimizer.step()
    require(any(not torch.equal(value,before[key]) for key,value in head.state_dict().items()), 'No synthetic update')
    require(all(torch.equal(value,before[key]) for key,value in reference.state_dict().items()), 'Synthetic reference changed')
    logits = torch.tensor([[.2,-.1,.5]],device=device,requires_grad=True)
    reference_logits = logits.detach().clone()
    advantage = torch.tensor([[.1,-.2,0]],device=device)
    value = exact_loss(torch, logits, reference_logits, advantage, .5, torch.ones(1,device=device))
    gradient = torch.autograd.grad(value,logits)[0]
    h, errors = .001, []
    for i in range(3):
        plus, minus = logits.detach().clone(), logits.detach().clone()
        plus[0,i] += h
        minus[0,i] -= h
        finite = (exact_loss(torch,plus,reference_logits,advantage,.5,torch.ones(1,device=device))-
                  exact_loss(torch,minus,reference_logits,advantage,.5,torch.ones(1,device=device)))/(2*h)
        errors.append(abs(float(finite-gradient[0,i])))
    require(max(errors) < 2e-5, 'Exact H1 finite difference failed')
    record = dict(contract=CONTRACT, state='completed', synthetic_only=True, device=str(device),
                  precision=request.get('precision','cuda-fp16'), update0_exact=True, frozen_reference=True,
                  finite_difference_max_error=max(errors), nonzero_head_delta=True, real_data=False)
    write(request['smoke_output'], record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('cache', 'frontier', 'train', 'screen', 'evaluate', 'rollout', 'smoke'))
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    method = {'cache': prepare_cache, 'frontier': inference_frontier, 'train': train, 'screen': score_checkpoint,
              'evaluate': evaluate_selected, 'rollout': rollout, 'smoke': smoke}[args.stage]
    print(json.dumps(method(read(args.request)), ensure_ascii=False))


if __name__ == '__main__':
    main()
