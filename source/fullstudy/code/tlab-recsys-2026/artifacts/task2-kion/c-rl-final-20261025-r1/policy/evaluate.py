"""Root-owned scientific three-arm evaluation; import and unit tests are stdlib-only.

Numerical entrypoints are never called on import. Full inference retains the old
validation B128/chunk64 arithmetic and uses one shared raw-feature pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CONTRACT = 'kion-scientific3-evaluation-v1'
ARMS = ('C', 'RL-completion', 'RL-residual')
N, WIDTH, BATCH, CHUNK = 4096, 1024, 128, 64
TIE_BREAK = 'score descending, raw item ID ascending'


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
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def pinned(entry):
    path = Path(entry['path']).resolve()
    require(path.is_file() and sha(path) == entry['sha256'], 'Source pin differs: '+path.as_posix())
    return path


def module(entry, name):
    path = pinned(entry)
    spec = importlib.util.spec_from_file_location('_scientific3_'+name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def fresh_output(path):
    output = Path(path).resolve()
    require(output.is_relative_to(HERE) and output != HERE, 'Output must be below new policy directory')
    require(not output.exists(), 'Fresh output directory required; partial outputs are not resumed')
    return output


def checkpoint_receipts(request, c_sha):
    """Bind each scientific arm to its own actual completed latest1000 lineage."""
    entries = request['scientific_checkpoints']
    require(set(entries) == set(ARMS[1:]), 'Exactly two separately named scientific checkpoints required')
    result = {}
    for arm in ARMS[1:]:
        entry = entries[arm]
        checkpoint, receipt_path = pinned(entry), pinned(entry['training'])
        receipt = read(receipt_path)
        require(receipt.get('state') == 'completed' and receipt.get('updates') == 1000,
                'Completed actual latest1000 receipt required for '+arm)
        require(receipt.get('supervised_updates') == 0 and receipt.get('frozen_encoder') is True,
                'Frozen encoder/no supervised updates required for '+arm)
        require(receipt.get('C_checkpoint_sha256') == c_sha, 'C ancestor differs for '+arm)
        checkpoints = receipt['checkpoints']
        matching = [item for item in checkpoints if item['sha256'] == entry['sha256']
                    and Path(item['path']).resolve() == checkpoint]
        require(entry.get('step') == 1000 and len(matching) == 1 and matching[0]['step'] == 1000
                and max(item['step'] for item in checkpoints) == 1000, 'Not actual latest1000 for '+arm)
        result[arm] = {**entry, 'path': checkpoint.as_posix(), 'C_checkpoint_sha256': c_sha,
                       'training_contract': receipt.get('contract'), 'updates': 1000,
                       'utility': receipt.get('utility_formula', receipt.get('utility', receipt.get('objective')))}
    require(result[ARMS[1]]['sha256'] != result[ARMS[2]]['sha256'], 'Scientific checkpoints must be distinct')
    return result


def adapter_request(request):
    """Older residual/utility loaders accept path strings, not pin objects."""
    result = dict(request, source_hashes=dict(request['source_hashes']))
    for key in ('policy_module', 'utility_module'):
        if key not in request:
            continue
        entry = request[key]
        path = pinned(entry)
        previous = result['source_hashes'].get(path.as_posix(), entry['sha256'])
        require(previous == entry['sha256'], 'Conflicting source pin '+key)
        result[key] = path.as_posix()
        result['source_hashes'][path.as_posix()] = entry['sha256']
    return result


def load_scientific_heads(request, policy, torch, device):
    heads = policy.load(request, 'heads')
    c_path = Path(request['head_dir'])/'C/head.pt'
    pins = checkpoint_receipts(request, sha(c_path))
    models = {'C': heads.load_heads(request, torch, device)['C']}
    completion = pins['RL-completion']
    models['RL-completion'] = policy.load_rl_head(dict(request, rl_checkpoint_sha256=completion['sha256']),
                                                 torch, device, completion['path'])
    residual = module(request['residual_module'], 'residual')
    entry = pins['RL-residual']
    normalized = adapter_request(request)
    models['RL-residual'] = residual.load_head(dict(normalized, rl_checkpoint_sha256=entry['sha256']),
                                              torch, device, entry['path'])
    for model in models.values():
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    pins['C'] = {'path': c_path.resolve().as_posix(), 'sha256': sha(c_path),
                 'step': None, 'lineage': 'original supervised C; no RL updates'}
    return models, pins


def exact_legacy_parity(actual, expected):
    """No tolerance or rounding for the previously accepted cache-screen aggregates."""
    fields = tuple(actual)
    require(all(field in expected for field in fields), 'Legacy diagnostic fields missing')
    different = [field for field in fields if actual[field] != expected[field]]
    require(not different, 'Legacy screen exact parity failed: '+','.join(different))
    return {'passed': True, 'exact': True, 'fields': list(fields)}


def update_rank_parity(parity, expected, actual, ids, raw_ids, rank):
    """Original raw-ID ranking/margin rule; pointwise scores use NumPy's old gate."""
    for old, new, tokens, raw in zip(expected, actual, ids, raw_ids):
        require(len(old) == len(new) == len(tokens) == len(raw), 'Parity alignment differs')
        require(all(math.isfinite(float(v)) for v in old) and all(math.isfinite(float(v)) for v in new),
                'Nonfinite parity scores')
        old_rank, new_rank = rank(tokens, old, raw), rank(tokens, new, raw)
        if old_rank != new_rank:
            parity['top10_changed_rows'] += 1
            for a, b in zip(old_rank, new_rank):
                tolerance = 2*parity['atol']+parity['rtol']*(abs(float(old[a]))+abs(float(old[b])))
                if abs(float(old[a]-old[b])) > tolerance:
                    parity['changes_with_margin_above_tolerance'] += 1


def new_parity(request):
    atol, rtol = request['parity_atol'], request['parity_rtol']
    require(all(math.isfinite(v) and v >= 0 for v in (atol, rtol)), 'Invalid parity tolerances')
    return dict(max_abs_error=0., score_tolerance_passed=True, top10_changed_rows=0,
                changes_with_margin_above_tolerance=0, atol=atol, rtol=rtol)


def close_maps(arrays):
    for value in arrays.values():
        mmap = getattr(value, '_mmap', None)
        if mmap is not None:
            if getattr(value, 'flags', None) is not None and value.flags.writeable:
                value.flush()
            mmap.close()


def allocate(np, output, arrays, name, dtype, shape, values=None):
    arrays[name] = np.lib.format.open_memmap(output/(name+'.npy'), mode='w+', dtype=dtype, shape=shape)
    if values is not None:
        arrays[name][:] = values
    return arrays[name]


def describe(policy, output, arrays):
    for value in arrays.values():
        value.flush()
    return {key: policy._descriptor(output/(key+'.npy'), value) for key, value in arrays.items()}


def base_record(request, stage, n, checkpoints):
    return dict(contract=CONTRACT, state='completed', stage=stage, rows=n, metric_denominator=n,
                candidate_width=WIDTH, arms=list(ARMS), checkpoints=checkpoints,
                temperature=1., tie_break=TIE_BREAK, no_target_injection=True, no_seen_filter=True,
                unknown_and_offpool_are_zero=True, final_test_reads=0,
                source={'path': Path(__file__).resolve().as_posix(), 'sha256': sha(__file__)},
                policy_module=request['policy_module'], residual_module=request['residual_module'],
                utility_module=request.get('utility_module'), source_hashes=request['source_hashes'],
                precision=request.get('precision', 'cuda-fp16'))


def screen(request, check=None):
    """Cache-only scoring; provider, bank and encoder are not opened."""
    policy = module(request['policy_module'], 'policy')
    check = policy.checked(request, check)
    output = fresh_output(request['screen_evaluation_dir'])
    # Runtime configuration stays the same as the existing root-owned cache scorer.
    import numpy as np
    import torch
    manifest, cache = policy.open_cache(request['cache_dir'], np)
    arrays = {}
    complete = False
    try:
        require(manifest['cohort'] == 'screen' and manifest['split'] == 'val' and manifest['rows'] == 256,
                'Exactly original screen256 required')
        require(manifest['precision'] == request.get('precision', 'cuda-fp16'), 'Screen/cache precision differs')
        require(len(manifest['candidate_contract']['source_feature_names']) == 14, 'Source14 column names required')
        device = torch.device(request.get('device', 'cuda'))
        models, checkpoints = load_scientific_heads(request, policy, torch, device)
        require(manifest['C_checkpoint_sha256'] == checkpoints['C']['sha256'], 'Screen C ancestor differs')
        reward, support, fallback = policy.reward_arrays(request['reward_dir'], cache, manifest, np)
        utility_module = module(request['utility_module'], 'utility')
        utility, prior, utility_support = utility_module.utility_arrays(request['utility_dir'], cache, manifest, np, policy)
        require(bool(np.array_equal(support, utility_support)), 'Second utility support differs')
        require(utility.shape == prior.shape == reward.shape, 'Second utility/prior shape differs')
        utility_manifest = read(Path(request['utility_dir'])/'utility.json')
        n = 256
        output.mkdir(parents=True)
        table = torch.as_tensor(np.array(cache['item_table'], copy=True), device=device).detach()
        for arm in ARMS:
            allocate(np, output, arrays, arm+'_scores', 'float32', (n, WIDTH))
            allocate(np, output, arrays, arm+'_top10', 'int32', (n, 10))
        sources = {'row_index': ('row_indices', 'int64'), 'user_id': ('user_ids', 'int64'),
                   'target_token': ('target_ids', 'int32'), 'pool_ids': ('candidate_ids', 'int32'),
                   'candidate_raw_ids': ('candidate_raw_ids', 'int64'), 'history_tokens': ('x', 'int64'),
                   'candidate_sources': ('candidate_sources', 'float32')}
        for name, (source, dtype) in sources.items():
            allocate(np, output, arrays, name, dtype, cache[source].shape, cache[source])
        known = (cache['target_ids'] >= 3) & (cache['target_ids'] < len(cache['item_table']))
        coverage = known & (cache['candidate_ids'] == cache['target_ids'][:, None]).any(1)
        allocate(np, output, arrays, 'known_target', 'bool', (n,), known)
        allocate(np, output, arrays, 'candidate_coverage', 'bool', (n,), coverage)
        for name, value, dtype in [('raw_reward', reward, 'float64'), ('utility_residual', utility, 'float64'),
                                    ('utility_prior', prior, 'float64'), ('support', support, 'bool'),
                                    ('fallback', fallback, 'bool')]:
            allocate(np, output, arrays, name, dtype, value.shape, value)
        popularity = np.expm1(np.asarray(cache['source_values'][..., 1], dtype=np.float64))
        require(bool(np.isfinite(popularity).all()) and bool((popularity >= 0).all()), 'Invalid strict7d count')
        allocate(np, output, arrays, 'strict7d_count', 'float64', popularity.shape, popularity)
        layout = policy.producer_layout(cache['origins'])
        with torch.inference_mode():
            for start in range(0, n, 16):
                check(stage='scientific3-cache-screen', rows=start)
                positions = list(range(start, min(start+16, n)))
                shaped = policy.prepare_shaped_inputs(cache, positions, table, torch, np, device, layout)
                with policy.precision(torch, request, device):
                    for arm, head in models.items():
                        values = policy.score_shaped(head, shaped, torch)
                        arrays[arm+'_scores'][positions] = values.float().cpu().numpy()
        require(bool(np.array_equal(arrays['C_scores'], cache['reference_scores'])), 'C/cache exact score parity failed')
        reference_logp, reference_probs = policy.distribution(np, cache['reference_scores'])
        advantage, _ = policy.effective_advantage(np, reference_probs, utility, support)
        diagnostics, metrics, legacy_parity = {}, {}, {}
        for arm in ARMS:
            scores = arrays[arm+'_scores']
            for i in range(n):
                arrays[arm+'_top10'][i] = cache['candidate_ids'][i, policy.rank(cache['candidate_ids'][i], scores[i], cache['candidate_raw_ids'][i])]
            values = policy.diagnostics(np, scores, cache, reward, support, reference_logp)
            if arm in ('C', 'RL-completion'):
                legacy_entry = request['legacy_screen_outputs'][arm]
                legacy = read(pinned(legacy_entry))
                require(legacy['cache_sha256'] == sha(Path(request['cache_dir'])/'cache.json'), 'Legacy cache binding differs')
                if arm == 'RL-completion':
                    require(legacy['checkpoint_sha256'] == checkpoints[arm]['sha256'], 'Legacy completion checkpoint differs')
                legacy_parity[arm] = {**exact_legacy_parity(values, legacy), 'source': legacy_entry}
            logp, probabilities = policy.distribution(np, scores)
            values['expected_effective_completion_utility'] = values['expected_effective_reward']
            values['expected_effective_residual_utility'] = float((probabilities*advantage).sum(1).mean())
            values['expected_supported_raw_RM_numerator_per_query'] = float((probabilities*np.where(support, reward, 0.)).sum(1).mean())
            values['expected_supported_residual_numerator_per_query'] = float((probabilities*np.where(support, utility, 0.)).sum(1).mean())
            diagnostics[arm] = values
            hits = (arrays[arm+'_top10'] == cache['target_ids'][:, None]) & known[:, None]
            metrics[arm] = dict(rows=n, known_targets=int(known.sum()), unknown_target_misses=int((~known).sum()),
                                offpool_known_misses=int((known & ~coverage).sum()), candidate_coverage_count=int(coverage.sum()),
                                map10=float((hits/np.arange(1, 11)).sum(1).mean()),
                                ndcg10=float((hits/np.log2(np.arange(2, 12))).sum(1).mean()), recall10=float(hits.any(1).mean()))
        record = base_record(request, 'screen', n, checkpoints)
        record.update(arrays=describe(policy, output, arrays), metrics=metrics, diagnostics=diagnostics,
                      cache={'path': (Path(request['cache_dir'])/'cache.json').resolve().as_posix(),
                             'sha256': sha(Path(request['cache_dir'])/'cache.json')},
                      reward={'path': (Path(request['reward_dir'])/'reward.json').resolve().as_posix(),
                              'sha256': sha(Path(request['reward_dir'])/'reward.json')},
                      utility={'path': (Path(request['utility_dir'])/'utility.json').resolve().as_posix(),
                               'sha256': sha(Path(request['utility_dir'])/'utility.json'), 'metadata': utility_manifest},
                      selection_sha256=manifest['selection_sha256'], catalog_size=len(cache['item_table'])-3,
                      feature_schema_sha256=manifest['feature_schema_sha256'],
                      original_C_parity={'score_tolerance_passed': True, 'scores_identical': True,
                                        'top10_changed_rows': 0, 'changes_with_margin_above_tolerance': 0,
                                        'basis': 'immutable original screen cache C reference scores'},
                      legacy_screen_parity=legacy_parity, raw_feature_calls=0, provider_opened=False,
                      utility_support='supported actions centered by C conditional mean; unsupported advantage=0; full1024 normalization',
                      candidate_source_columns=manifest['candidate_contract']['source_feature_names'],
                      actor_source_columns=manifest['source_columns'])
        complete = True
    finally:
        try:
            close_maps(arrays)
        finally:
            policy.close_cache(cache)
    require(complete, 'Incomplete screen export')
    write(output/'evaluation.json', record)
    return record


def load_original(np, request):
    root = Path(request['original_evaluation_dir']).resolve()
    manifest_path = root/'evaluation.json'
    require(sha(manifest_path) == request['original_evaluation_sha256'], 'Original evaluation pin differs')
    manifest = read(manifest_path)
    require(manifest['state'] == 'completed' and manifest['rows'] == manifest['metric_denominator'] == N
            and manifest['candidate_width'] == WIDTH and manifest['no_seen_filter'] is True
            and manifest['no_target_injection'] is True and manifest['unknown_and_offpool_are_zero'] is True,
            'Original evaluation protocol differs')
    require(manifest['selection']['sha256'] == request['validation_selection_sha256'], 'Original selection binding differs')
    arrays = {}
    try:
        for name in ('row_index', 'user_id', 'target_token', 'known_target', 'candidate_coverage',
                     'pool_ids', 'catalog_raw_ids', 'C_scores'):
            entry = manifest['arrays'][name]
            path = (root/entry['path']).resolve()
            require(path.parent == root and path.suffix == '.npy', 'Nonlocal original array')
            require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], 'Original array pin differs '+name)
            value = np.load(path, mmap_mode='r', allow_pickle=False)
            arrays[name] = value
            require(list(value.shape) == entry['shape'] and str(value.dtype) == entry['dtype'], 'Original array header differs '+name)
        require(arrays['pool_ids'].shape == arrays['C_scores'].shape == (N, WIDTH), 'Original score/pool shape differs')
        require(all(arrays[name].shape == (N,) for name in ('row_index', 'user_id', 'target_token', 'known_target', 'candidate_coverage')),
                'Original binding shape differs')
        return manifest, arrays
    except BaseException:
        close_maps(arrays)
        raise


def full(request, check=None):
    """One original B128/chunk64 feature pass; stream scores/top10 for all arms."""
    policy = module(request['policy_module'], 'policy')
    check = policy.checked(request, check)
    bank, evaluator, heads = (policy.load(request, name) for name in ('bank', 'evaluate', 'heads'))
    rows = evaluator.validate_selection(request['validation_selection'], request['validation_selection_sha256'])
    require(len(rows) == N, 'Exactly original full4096 required')
    heads.preflight_heads(request)
    output = fresh_output(request['full_evaluation_dir'])
    ctx = policy.open_frozen_context(bank, request, check)
    original, arrays = {}, {}
    complete = False
    try:
        np, torch, train, teacher, provider, candidates, table, device = (
            ctx[key] for key in ('np', 'torch', 'train', 'model', 'provider', 'candidates', 'table', 'device'))
        require(ctx['feature_schema_sha256'] == policy.SCHEMA and ctx['source_columns'] == policy.SOURCE,
                'Current C schema/source mismatch')
        require(not teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'E2 must remain frozen eval')
        models, checkpoints = load_scientific_heads(request, policy, torch, device)
        original_manifest, original = load_original(np, request)
        require(original_manifest['head_files']['C']['sha256'] == checkpoints['C']['sha256'], 'Original C checkpoint differs')
        indices = np.asarray(rows, dtype=np.int64)
        require(bool(np.array_equal(indices, original['row_index'])), 'Original row order differs')
        data = provider.splits['val']
        targets = np.asarray(data['y'][indices])
        known = (targets >= 3) & (targets < teacher.vocab_size)
        raw_catalog = np.asarray(candidates.raw_item_ids(np.arange(3, teacher.vocab_size)), dtype=np.int64)
        require(bool(np.array_equal(targets, original['target_token'])) and bool(np.array_equal(known, original['known_target']))
                and bool(np.array_equal(data['user_id'][indices], original['user_id']))
                and bool(np.array_equal(raw_catalog, original['catalog_raw_ids'])), 'Original query/catalog binding differs')
        output.mkdir(parents=True)
        for name in ('row_index', 'user_id', 'target_token', 'known_target', 'catalog_raw_ids'):
            value = original[name]
            allocate(np, output, arrays, name, value.dtype, value.shape, value)
        allocate(np, output, arrays, 'pool_ids', 'int32', (N, WIDTH))
        allocate(np, output, arrays, 'candidate_raw_ids', 'int64', (N, WIDTH))
        allocate(np, output, arrays, 'candidate_coverage', 'bool', (N,))
        for arm in ARMS:
            allocate(np, output, arrays, arm+'_scores', 'float32', (N, WIDTH))
            allocate(np, output, arrays, arm+'_top10', 'int32', (N, 10))
        head_table = table.detach().float()
        parity, feature_calls = new_parity(request), 0
        with torch.inference_mode():
            for start in range(0, N, BATCH):
                check(stage='scientific3-full4096', rows=start, total=N)
                stop = start+BATCH
                batch_rows = indices[start:stop]
                x, lengths, y, user = train.batch(provider, batch_rows, 'val', device)
                with train.legacy.autocast(device):
                    q = teacher.encode_context(x, lengths, user, candidate_table=table)
                    logits = teacher.retrieval_logits(q, table)
                ids, source = candidates.pool(batch_rows, 'val', logits)
                require(ids.shape == (BATCH, WIDTH) and source.shape == (BATCH, WIDTH, 14), 'Natural1024/source14 shape differs')
                require(bool(np.array_equal(ids, original['pool_ids'][start:stop])), 'Original natural candidate pool differs')
                require(bool(np.array_equal(y.cpu().numpy(), targets[start:stop])), 'Batch target binding differs')
                require(bool(np.all((ids >= 3) & (ids < teacher.vocab_size)))
                        and all(len(set(row)) == WIDTH for row in ids.tolist()), 'Distinct known pool required')
                arrays['pool_ids'][start:stop] = ids
                arrays['candidate_coverage'][start:stop] = (ids == targets[start:stop, None]).any(1) & known[start:stop]
                for left in range(0, WIDTH, CHUNK):
                    check(stage='scientific3-full4096-raw-chunk', rows=start, candidate_column=left)
                    raw = provider.candidate_features(batch_rows, ids[:, left:left+CHUNK], 'val')
                    feature_calls += 1
                    require(raw.shape == (BATCH, CHUNK, 1068) and bool(np.isfinite(raw).all()), 'Raw1068 schema/values differ')
                    token = torch.as_tensor(ids[:, left:left+CHUNK], device=device, dtype=torch.long)
                    base = logits.gather(1, token).float()
                    args = (q.float(), head_table[token], base, torch.as_tensor(raw, device=device, dtype=torch.float32),
                            torch.as_tensor(source[:, left:left+CHUNK, 12:14], device=device, dtype=torch.float32))
                    with policy.precision(torch, request, device):
                        for arm, head in models.items():
                            values = head(*args)
                            require(tuple(values.shape) == (BATCH, CHUNK) and bool(torch.isfinite(values).all()), 'Invalid head score')
                            arrays[arm+'_scores'][start:stop, left:left+CHUNK] = values.float().cpu().numpy()
                    del raw, token, base, args, values
                raw_ids = candidates.raw_item_ids(ids)
                arrays['candidate_raw_ids'][start:stop] = raw_ids
                expected, actual = original['C_scores'][start:stop], arrays['C_scores'][start:stop]
                parity['max_abs_error'] = max(parity['max_abs_error'], float(np.abs(expected-actual).max()))
                parity['score_tolerance_passed'] &= bool(np.allclose(expected, actual, atol=request['parity_atol'], rtol=request['parity_rtol']))
                update_rank_parity(parity, expected, actual, ids, raw_ids, policy.rank)
                for arm in ARMS:
                    for i in range(BATCH):
                        arrays[arm+'_top10'][start+i] = ids[i, policy.rank(ids[i], arrays[arm+'_scores'][start+i], raw_ids[i])]
                for value in arrays.values():
                    value.flush()
                del q, logits, ids, source, x, lengths, y, user
        require(feature_calls == N//BATCH*(WIDTH//CHUNK) == 512, 'Must have exactly ONE full raw-feature pass')
        require(parity['score_tolerance_passed'] and parity['changes_with_margin_above_tolerance'] == 0,
                'Original C parity failed; stop comparison')
        require(bool(np.array_equal(arrays['candidate_coverage'], original['candidate_coverage'])), 'Original candidate coverage differs')
        require(not teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'Frozen E2 mode changed')
        metrics = {arm: evaluator._summary(np, arrays[arm+'_top10'], targets, known, arrays['candidate_coverage']) for arm in ARMS}
        record = base_record(request, 'full4096', N, checkpoints)
        record.update(arrays=describe(policy, output, arrays), metrics=metrics, original_C_parity=parity,
                      original_evaluation_sha256=request['original_evaluation_sha256'],
                      selection_sha256=request['validation_selection_sha256'], raw_feature_calls=feature_calls,
                      scientific_arms_share_raw_pass=True, provider_passes=1, batch_size=BATCH, candidate_chunk=CHUNK,
                      catalog_size=len(raw_catalog), baseline_for_matched_comparison='C recomputed in same shared-feature pass',
                      feature_schema_sha256=ctx['feature_schema_sha256'], actor_source_columns=ctx['source_columns'],
                      reward_proxy_available=False, legacy_context_stage='evaluate', operation_stage=request.get('stage'))
        complete = True
    finally:
        try:
            try:
                close_maps(arrays)
            finally:
                close_maps(original)
        finally:
            cleanup = bank.close_context(ctx)
            require(cleanup.get('teardown_completed') is True and cleanup.get('errors') == [], 'Full evaluation cleanup failed')
    require(complete, 'Incomplete full4096 evaluation')
    write(output/'evaluation.json', record)
    return record


def score(request, check=None):
    """Existing controller admission name for cache-only screen export."""
    return screen(request, check)


def evaluate(request, check=None):
    """Existing controller admission name for one shared full4096 pass."""
    return full(request, check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('screen', 'full'))
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    print(json.dumps({'screen': screen, 'full': full}[args.stage](read(args.request)), ensure_ascii=False))


if __name__ == '__main__':
    main()
