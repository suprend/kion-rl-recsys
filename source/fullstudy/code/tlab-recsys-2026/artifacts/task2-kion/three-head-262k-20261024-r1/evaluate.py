"""One frozen validation pass, shared natural1024 support, no import-time ML."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time

HERE = Path(__file__).resolve().parent
CONTRACT = 'kion-three-head-262k-shared4096-natural1024-v1'
SELECTION_CONTRACT = 'kion-three-head-shared4096-natural1024-v1'
HEAD_CONTRACT = 'kion-three-head-262k-head-v1'
ARMS = ('A', 'B', 'C', 'GPT', 'RRF')
ROWS, TOTAL_ROWS, SEED, WIDTH = 4096, 158760, 20261024, 1024
TRAIN_ROWS, EPOCHS, STEPS_PER_HEAD = 262144, 3, 6144


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    path = Path(path)
    require(path.stat().st_size <= 8 * 1024**2, 'JSON exceeds 8 MiB')
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def selection():
    """Label/head-independent row sample. Must be frozen before training."""
    return dict(contract=SELECTION_CONTRACT, split='val', population=TOTAL_ROWS, size=ROWS,
                seed=SEED, algorithm='sorted(random.Random(seed).sample(range(population),size))',
                row_indices=sorted(random.Random(SEED).sample(range(TOTAL_ROWS), ROWS)))


def freeze_selection(path):
    document = selection()
    path = Path(path)
    if path.exists():
        require(read_json(path) == document, 'Existing validation selection differs')
    else:
        write_json(path, document)
    return dict(path=path.resolve().as_posix(), sha256=sha(path))


def validate_selection(path, expected_sha=None):
    value = read_json(path)
    require(value == selection(), 'Validation selection must be the predeclared sample')
    if expected_sha:
        require(sha(path) == expected_sha, 'Validation selection pin differs')
    return value['row_indices']


def validate_completed_receipt(receipt):
    """Stdlib-only exact scale gate; the pilot/one-epoch fallback is forbidden."""
    require(receipt.get('contract') == HEAD_CONTRACT
            and receipt.get('status') == 'completed'
            and receipt.get('rows_per_epoch') == TRAIN_ROWS
            and receipt.get('batch_size') == 128
            and receipt.get('epochs_requested') == receipt.get('epochs_completed') == EPOCHS
            and receipt.get('steps_per_head') == STEPS_PER_HEAD
            and receipt.get('head_steps') == {name: STEPS_PER_HEAD for name in ('A', 'B', 'C')},
            'All fresh 262144-row heads must finish exactly 3 epochs and 6144 updates')


def row_metrics(top, target, known):
    rank = next((i for i, value in enumerate(top[:10], 1) if known and value == target), 0)
    return dict(map10=1.0/rank if rank else 0.0, recall10=float(bool(rank)),
                ndcg10=1.0/math.log2(rank+1) if rank else 0.0)


def rank_top(ids, scores, raw_ids, k=10):
    """Small stdlib oracle: descending score then ascending *raw* item ID."""
    require(len(ids) == len(scores) == len(raw_ids), 'Ranking alignment differs')
    require(all(math.isfinite(float(v)) for v in scores), 'Nonfinite rank score')
    require(len(set(ids)) == len(ids) and len(set(raw_ids)) == len(raw_ids), 'Duplicate catalog IDs')
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), int(raw_ids[i])))[:k]]


def rrf_row(source):
    """Stdlib oracle for exact source2048 rank recovery and fixed RRF60."""
    require(len(source) == 14, 'Source14 metadata required')
    total = 0.0
    for present, normalized, missing in zip(source[3:6], source[6:9], source[9:12]):
        require(present in (0, 1) and missing in (0, 1) and present + missing == 1, 'Inconsistent nomination flags')
        require(math.isfinite(float(normalized)), 'Nonfinite normalized rank')
        rank = round(float(normalized) * 2049)
        require(1 <= rank <= 2049 and bool(present) == (rank <= 2048), 'Invalid recovered nomination rank')
        total += float(present) / (60 + rank)
    return total


def rrf_scores(np, source):
    """CPU-only fixed fusion, evaluated once per natural-pool metadata batch."""
    require(source.ndim == 3 and source.shape[-1] == 14, 'Source14 metadata required')
    present, normalized, missing = source[..., 3:6], source[..., 6:9], source[..., 9:12]
    require(bool(np.isfinite(normalized).all()), 'Nonfinite normalized nomination ranks')
    require(bool(np.all((present == 0) | (present == 1))) and bool(np.all((missing == 0) | (missing == 1)))
            and bool(np.all(present + missing == 1)), 'Inconsistent nomination flags')
    ranks = np.rint(normalized * 2049).astype(np.int32)
    require(bool(np.all((ranks >= 1) & (ranks <= 2049)))
            and bool(np.all((present == 1) == (ranks <= 2048))), 'Invalid recovered nomination ranks')
    return (present.astype(np.float64) / (60 + ranks)).sum(axis=-1).astype(np.float32)


def load_module(name):
    path = HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('_pilot_eval_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _save(np, directory, name, values):
    path = directory / (name + '.npy')
    with path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path),
                dtype=str(values.dtype), shape=list(values.shape))


def _summary(np, top, target, known, coverage):
    hits = (top == target[:, None]) & known[:, None]
    return dict(rows=len(target), known_targets=int(known.sum()), unknown_target_misses=int((~known).sum()),
                candidate_coverage_count=int(coverage.sum()), candidate_coverage=float(coverage.mean()),
                offpool_known_misses=int((known & ~coverage).sum()),
                map10=float((hits / np.arange(1, 11)).sum(1).mean()),
                recall10=float(hits.any(1).mean()), ndcg10=float((hits / np.log2(np.arange(2, 12))).sum(1).mean()))


def evaluate(request, check=None):
    """Root-owned native entrypoint. Never calls train/optimizer or loads test."""
    check = check or (lambda **fields: None)
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    require(deadline.utcoffset() is not None, 'Aware deadline required')
    def checked(**fields):
        require(datetime.now(timezone.utc) < deadline, 'Evaluation deadline exceeded')
        check(**fields)
    selected_path = Path(request.get('validation_selection', HERE / 'validation-selection.json'))
    selected = validate_selection(selected_path, request.get('validation_selection_sha256'))
    output = Path(request['evaluation_dir']).resolve()
    require(not output.exists(), 'Fresh evaluation directory required; partial runs are not resumed')
    bank, head_module = load_module('bank'), load_module('heads')
    require(request.get('bank_manifest_sha256'), 'Pinned completed training bank required before validation')
    checked(stage='completed-head-preflight')
    training_receipt = head_module.preflight_heads(request)
    validate_completed_receipt(training_receipt)
    embedded_receipt = {key: value for key, value in training_receipt.items() if key != 'checkpoint_sha256'}
    # Runtime configure_paths rejects an already imported torch. The stdlib
    # preflight binds all three complete checkpoint hashes before provider open;
    # only the original configured runtime may decode those trusted payloads.
    ctx = bank.open_context(request, checked)
    completed = False
    try:
        np, torch = ctx['np'], ctx['torch']
        train, provider, teacher = ctx['train'], ctx['provider'], ctx['model']
        table, candidates, device = ctx['table'], ctx['candidates'], ctx['device']
        # Bank training stored frozen AMP values in fp32 arrays. Preserve exactly
        # that head arithmetic (especially fp32 base+residual) during inference.
        head_table = table.detach().float()
        require(len(provider.splits['val']['y']) == TOTAL_ROWS, 'Changed original validation population')
        require(not teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'E2 must be frozen eval')
        models = head_module.load_heads(request, torch, device)
        require(set(models) == {'A', 'B', 'C'}, 'All three completed heads required before validation')
        receipts = {name: getattr(model, 'pilot_receipt', None) for name, model in models.items()}
        require(all(isinstance(r, dict) and r for r in receipts.values()), 'Completed training receipts required')
        require(all(r == receipts['A'] for r in receipts.values()), 'Head training budgets/lineage differ')
        receipt = receipts['A']
        require(receipt == embedded_receipt, 'Loaded head receipt differs from preflight training metadata')
        validate_completed_receipt(receipt)
        require(receipt['teacher'] == ctx['teacher']
                and receipt['feature_schema_sha256'] == ctx['feature_schema_sha256']
                and receipt['source_columns'] == ctx['source_columns'], 'Evaluation context differs from frozen training bank')
        for model in models.values():
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        output.mkdir(parents=True)
        indices = np.asarray(selected, dtype=np.int64)
        data = provider.splits['val']
        arrays = dict(row_index=indices, user_id=np.asarray(data['user_id'][indices]),
                      target_item_id=np.asarray(data['target_item_id'][indices]),
                      target_date=np.asarray(data['target_date'][indices]),
                      target_token=np.asarray(data['y'][indices]))
        arrays['known_target'] = (arrays['target_token'] >= 3) & (arrays['target_token'] < teacher.vocab_size)
        arrays['candidate_coverage'] = np.zeros(ROWS, dtype=np.bool_)
        arrays['pool_ids'] = np.empty((ROWS, WIDTH), dtype=np.int32)
        arrays['catalog_raw_ids'] = np.asarray(candidates.raw_item_ids(np.arange(3, teacher.vocab_size)), dtype=np.int64)
        scores = {name: np.empty((ROWS, WIDTH), dtype=np.float32) for name in ARMS}
        tops = {name: np.empty((ROWS, 10), dtype=np.int32) for name in ARMS}
        started = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, ROWS, 128):
                checked(stage='shared-evaluation-batch', rows=start, total=ROWS)
                stop = min(start + 128, ROWS)
                rows = indices[start:stop]
                x, lengths, y, user = train.batch(provider, rows, 'val', device)
                with train.legacy.autocast(device):
                    q = teacher.encode_context(x, lengths, user, candidate_table=table)
                    logits = teacher.retrieval_logits(q, table)
                require(bool(torch.isfinite(logits[:, 3:]).all()), 'Nonfinite E2 logits')
                ids, source = candidates.pool(rows, 'val', logits)
                require(ids.shape == (len(rows), WIDTH) and source.shape == (len(rows), WIDTH, 14), 'Natural1024/source14 required')
                require(bool(np.all((ids >= 3) & (ids < teacher.vocab_size)))
                        and all(len(set(row)) == WIDTH for row in ids.tolist()), 'Distinct known candidates required')
                require(bool(np.array_equal(y.cpu().numpy(), arrays['target_token'][start:stop])), 'Row/target binding changed')
                arrays['pool_ids'][start:stop] = ids
                scores['RRF'][start:stop] = rrf_scores(np, source)
                arrays['candidate_coverage'][start:stop] = ((ids == arrays['target_token'][start:stop, None]).any(1)
                                                         & arrays['known_target'][start:stop])
                for left in range(0, WIDTH, 64):
                    checked(stage='shared-evaluation-feature-chunk', rows=start, candidate_column=left)
                    right = left + 64
                    # Expensive raw1068 is evaluated exactly once per shared chunk.
                    raw = provider.candidate_features(rows, ids[:, left:right], 'val')
                    require(raw.shape == (len(rows), right-left, 1068), 'Raw candidate schema changed')
                    selected_ids = torch.as_tensor(ids[:, left:right], device=device, dtype=torch.long)
                    raw_tensor = torch.as_tensor(raw, device=device, dtype=torch.float32)
                    source_tensor = torch.as_tensor(source[:, left:right, 12:14], device=device, dtype=torch.float32)
                    base = logits.gather(1, selected_ids).float()
                    with train.legacy.autocast(device):
                        values = head_module.score_heads(models, q.float(), head_table, selected_ids, base,
                                                         raw_tensor, source_tensor, torch)
                    require(set(values) == {'A', 'B', 'C'}, 'Missing scored arm')
                    values['GPT'] = base
                    for name, value in values.items():
                        require(tuple(value.shape) == (len(rows), right-left) and bool(torch.isfinite(value).all()), 'Invalid head scores')
                        scores[name][start:stop, left:right] = value.float().cpu().numpy()
                    del raw, raw_tensor, source_tensor, selected_ids, base, values
                raw_ids = candidates.raw_item_ids(ids)
                for name in ARMS:
                    tops[name][start:stop] = np.stack([ids[i, np.lexsort((raw_ids[i], -scores[name][start+i]))[:10]]
                                                      for i in range(len(rows))])
                del q, logits, ids, source, x, lengths, y, user
        train.legacy.synchronize(device)
        seconds = time.perf_counter() - started
        require(not teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'Frozen E2 mode changed')
        descriptors = {name: _save(np, output, name, values) for name, values in arrays.items()}
        metrics = {}
        for name in ARMS:
            checked(stage='shared-evaluation-save', arm=name)
            descriptors[name + '_scores'] = _save(np, output, name + '_scores', scores[name])
            descriptors[name + '_top10'] = _save(np, output, name + '_top10', tops[name])
            metrics[name] = _summary(np, tops[name], arrays['target_token'], arrays['known_target'], arrays['candidate_coverage'])
        head_dir = Path(request.get('head_dir', request['output_dir']))
        record = dict(contract=CONTRACT, state='completed', selection=dict(path=selected_path.resolve().as_posix(), sha256=sha(selected_path)),
                      rows=ROWS, population=TOTAL_ROWS, seed=SEED, candidate_width=WIDTH, batch_size=128,
                      candidate_chunk_size=64, raw_feature_calls=(ROWS//128)*(WIDTH//64),
                      arm_names=list(ARMS), source_columns=[12, 13], no_seen_filter=True, no_target_injection=True,
                      rrf=dict(constant=60, normalized_rank_columns=[6, 7, 8], nomination_columns=[3, 4, 5],
                               missing_columns=[9, 10, 11], rank_recovery='rint(normalized*2049)', missing_contribution=0),
                      unknown_and_offpool_are_zero=True, tie_break='score descending, raw item ID ascending',
                      frozen_teacher=ctx.get('teacher'), feature_schema_sha256=ctx.get('feature_schema_sha256'),
                      candidate_contract=ctx.get('candidate_contract'),
                      head_receipts=receipts, head_files={name: dict(path=(head_dir/name/'head.pt').resolve().as_posix(),
                      sha256=sha(head_dir/name/'head.pt')) for name in ('A', 'B', 'C')}, arrays=descriptors,
                      metrics=metrics, seconds=seconds, primary_comparison='C-B', metric_denominator=ROWS)
        write_json(output / 'evaluation.json', record)
        completed = True
        return record
    finally:
        cleanup = bank.close_context(ctx)
        if output.exists():
            write_json(output / 'cleanup.json', dict(evaluation_completed=completed, evidence=cleanup))
        require(cleanup.get('teardown_completed') is True and cleanup.get('errors') == [],
                'Evaluation context teardown failed: ' + repr(cleanup))


def run(request, check=None):
    return evaluate(request, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request')
    parser.add_argument('--freeze-selection')
    args = parser.parse_args()
    if args.freeze_selection:
        print(json.dumps(freeze_selection(args.freeze_selection)))
    else:
        require(args.request, '--request required')
        result = evaluate(read_json(args.request))
        print(json.dumps(dict(state=result['state'], metrics=result['metrics'])))


if __name__ == '__main__':
    main()
