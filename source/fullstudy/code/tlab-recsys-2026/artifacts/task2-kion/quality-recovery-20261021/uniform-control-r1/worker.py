"""Original fresh trainer with one constrained E1 sidecar and weight-state proof."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CHUNK_BYTES = 8 << 20


def state_digest(value, torch, np):
    """Typed device-independent bytes, at most8MiB tensor transfer per chunk."""
    digest = hashlib.sha256()

    def header(value):
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
        digest.update(len(payload).to_bytes(8, 'little'))
        digest.update(payload)

    def feed(value):
        if torch is not None and isinstance(value, torch.Tensor):
            if value.layout != torch.strided or not value.is_contiguous():
                raise ValueError('Expected contiguous dense frozen tensor for bounded identity hash')
            raw = value.detach().reshape(-1).view(torch.uint8)
            header({'kind': 'tensor', 'dtype': str(value.dtype), 'shape': list(value.shape), 'bytes': raw.numel()})
            for begin in range(0, raw.numel(), CHUNK_BYTES):
                cpu = raw[begin:begin + CHUNK_BYTES].to(device='cpu')
                digest.update(memoryview(cpu.numpy()))
        elif np is not None and isinstance(value, np.ndarray):
            if not value.flags.c_contiguous:
                raise ValueError('Expected contiguous RNG array')
            header({'kind': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape), 'bytes': value.nbytes})
            raw = memoryview(value).cast('B')
            for begin in range(0, len(raw), CHUNK_BYTES):
                digest.update(raw[begin:begin + CHUNK_BYTES])
        elif isinstance(value, dict):
            header({'kind': 'dict', 'length': len(value)})
            for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
                feed(key)
                feed(value[key])
        elif isinstance(value, (list, tuple)):
            header({'kind': type(value).__name__, 'length': len(value)})
            for item in value:
                feed(item)
        elif isinstance(value, bytes):
            header({'kind': 'bytes', 'length': len(value)})
            digest.update(value)
        elif value is None or type(value) in (str, bool, int, float):
            header({'kind': type(value).__name__, 'value': value})
        else:
            raise TypeError('Unexpected state identity leaf: ' + type(value).__name__)
    feed(value)
    return digest.hexdigest()


def replay_sidecar(p, train, provider, path, validation):
    np = train.np
    data, n = provider.splits['val'], 158760
    with np.load(path, allow_pickle=False) as archive:
        expected = set(train.ROW_KEYS) | {'row_index', 'target_token', 'known_target', 'top_items', 'retrieval_top_items',
            'rerank_top_items', 'retrieval_covered', 'hit_at_10', 'ndcg_at_10'}
        p.require(set(archive.files) == expected, 'exact original prediction NPZ schema')
        for key in train.ROW_KEYS:
            value = archive[key]
            p.require(value.dtype == data[key].dtype and value.shape == (n,) and np.array_equal(value, data[key]), 'full semantic row order: ' + key)
        targets, indices = archive['target_token'], archive['row_index']
        known = train.known_targets(data['y'], int(provider.metadata['vocab_size']))
        p.require(indices.dtype == np.int64 and np.array_equal(indices, np.arange(n)) and targets.dtype == data['y'].dtype
            and np.array_equal(targets, data['y']) and archive['known_target'].dtype == np.bool_
            and np.array_equal(archive['known_target'], known), 'full row indices/targets/known flags')
        result = {}
        for name in ('retrieval', 'rerank'):
            top = archive[name + '_top_items']
            p.require(top.shape == (n, 10) and top.dtype == np.int64 and np.all(top >= train.FIRST_ITEM)
                and np.all(top < int(provider.metadata['vocab_size']))
                and np.all(np.sort(top, axis=1)[:, 1:] != np.sort(top, axis=1)[:, :-1]), 'full unique known top10: ' + name)
            metrics = train._rank_metrics(top, targets, int(provider.metadata['vocab_size']))
            p.require(metrics == validation[name], 'exact CPU rank metric replay: ' + name)
            match = (top == targets[:, None]) & known[:, None]
            result[name] = {**metrics, 'map@10': float((match / np.arange(1, 11)).sum(1).mean())}
        p.require(np.array_equal(archive['top_items'], archive['rerank_top_items']), 'declared rerank top_items')
        match = (archive['top_items'] == targets[:, None]) & known[:, None]
        p.require(archive['hit_at_10'].dtype == np.bool_ and np.array_equal(archive['hit_at_10'], match.any(1))
            and np.array_equal(archive['ndcg_at_10'], (match / np.log2(np.arange(10) + 2)).sum(1)), 'serialized hit and NDCG vectors')
        covered = archive['retrieval_covered']
        p.require(covered.dtype == np.bool_ and covered.shape == (n,) and not np.any(covered & ~known)
            and float(covered.mean()) == validation['retrieval_coverage_all_rows']
            and float(covered.sum() / known.sum()) == validation['retrieval_coverage_known_rows'], 'serialized retrieval128 coverage')
        return {'rank_metrics_and_coverage_match': True, 'semantic_rows_targets_known_match': True,
            'examples': n, 'known_targets': int(known.sum()), 'unknown_target_misses': int((~known).sum()),
            'metrics': result, 'rowkeys_sha256': p.g.canonical({key: train.shared.array_hash(archive[key]) for key in train.ROW_KEYS}),
            'targets_sha256': train.shared.array_hash(targets), 'known_flags_sha256': train.shared.array_hash(known),
            'no_model_forward': True, 'cross_entropy_recomputed': False,
            'cross_entropy_provenance': 'original evaluator scalar; exact hook/selected/raw-metrics equality, not reconstructable from top10NPZ',
            'coverage_limit': 'checks serialized coverage and aggregates; full128 candidate IDs are not saved by original schema'}


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    cli = parser.parse_args(argv)
    request = json.loads(cli.request.read_text(encoding='utf-8'))
    path = HERE / 'control.py'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != request['source_hashes'][path.as_posix()]:
        raise ValueError('Exact reviewed control source changed')
    spec = importlib.util.spec_from_file_location('_uniform_worker_metadata', path)
    p = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = p
    exec(compile(payload, str(path), 'exec'), p.__dict__)
    p.require(request['specification'] == p.specification() and request['train_argv'] == p.train_argv(request['deadline_utc']), 'exact fresh source-fixed worker recipe')
    p.require(os.environ.get('KION_OUTPUT_DIR') == p.OUTPUT.as_posix()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == p.read(p.BASE_PLAN)['gpu_uuid'], 'original root-owned native environment')
    for path, digest in request['source_hashes'].items():
        p.require(p.sha(path) == digest, 'exact reviewed helper source')
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    p.require(deadline.utcoffset() is not None, 'aware absolute lease')
    adjunct = p.OUTPUT / 'adjunct'
    p.require(not adjunct.exists() and not (p.OUTPUT / 'model').exists(), 'fresh model and adjunct directories')
    adjunct.mkdir()
    sidecar = adjunct / 'predictions.npz'
    hook = None
    status = {'state': 'failed-unbound-sidecar', 'operation': p.KIND, 'schedule_completed': False, 'test_accessed': False}

    def budget():
        p.require(datetime.now(timezone.utc) < deadline and not (p.OUTPUT / 'stop-request.json').exists()
            and time.perf_counter() - started < 23910, 'finite native deadline/stop bound during provenance publication')

    try:
        budget()
        sys.path.insert(0, (p.COMMON / 'scripts').as_posix())
        source_hashes = p.read(p.BASE_PLAN)['source_hashes']
        runtime = p.load(p.COMMON / 'scripts/kion_ready_runtime.py', source_hashes['scripts/kion_ready_runtime.py'], '_uniform_accepted_bootstrap')
        runtime.configure_paths()  # Never call runpy dispatch after installing hooks.
        train = p.load(p.COMMON / 'scripts/kion_complete_train.py', source_hashes['scripts/kion_complete_train.py'], '_uniform_exact_frozen_train')
        hooks = p.load(p.HOOKS, request['source_hashes'][p.HOOKS.as_posix()], '_uniform_exact_singlepass_hooks')
        torch, np = train.torch, train.np

        def snapshot(model, provider, optimizer, scaler):
            budget()
            return {'weights': state_digest(model.state_dict(), torch, np),
                'optimizer': state_digest(optimizer.state_dict(), torch, np),
                'scaler': state_digest(scaler.state_dict(), torch, np),
                'rng': state_digest(train.legacy.rng_state(), torch, np)}

        hook = hooks.SinglePass(train, sidecar, snapshot)
        original_main, original_run = train.main, train.run
        hook.install()
        p.require(train.main is original_main and train.run is original_run
            and original_main.__globals__['run'] is original_run, 'actual unchanged original main invokes same run/globals')
        try:
            result = original_main(request['train_argv'])
        finally:
            hook.restore()
        budget()
        evidence = hook.finished()
        p.require(result['status'] == 'stopped' and result['reason'] == 'runtime_stop_after_validation'
            and result['predictions_status'] == 'deferred-runtime-stop', 'original raw stopped protocol')
        selected = p.read(p.OUTPUT / 'model/selected.json')
        config = p.read(p.OUTPUT / 'model/config.json')
        p.require(result['validation'] == selected['validation'] == hook.metrics, 'returned/selected/hook validation equality')
        saved = train.load_checkpoint(p.OUTPUT / 'model/selected-epoch0001.pt', config, map_location='cpu')
        p.require(saved['state'] == result['state'], 'committed selected checkpoint final E1 cursor/history')
        selected_digest = state_digest(saved['state_dict'], torch, np)
        p.require(selected_digest == hook.evaluated['weights']
            and state_digest(saved['optimizer'], torch, np) == hook.evaluated['optimizer']
            and state_digest(saved['scaler'], torch, np) == hook.evaluated['scaler']
            and state_digest(saved['rng'], torch, np) == hook.evaluated['rng'], 'actual committed checkpoint equals evaluated weights/optimizer/scaler/RNG')
        del saved
        budget()
        evidence.update(selected_checkpoint_weights_digest=selected_digest,
            selected_checkpoint_sha256=p.sha(p.OUTPUT / 'model/selected-epoch0001.pt'), sidecar_sha256=p.sha(sidecar),
            raw_completion_sha256=p.sha(p.OUTPUT / 'model/completion.json'), raw_metrics_sha256=p.sha(p.OUTPUT / 'model/metrics.json'),
            config_sha256=p.sha(p.OUTPUT / 'model/config.json'), source_hashes=request['source_hashes'],
            frozen_runner_sha256=source_hashes['scripts/kion_complete_train.py'], train_argv=request['train_argv'],
            entrypoint='accepted configure_paths -> exact imported train.main -> same run globals; no runpy dispatch')
        replay = replay_sidecar(p, train, hook.objects[1], sidecar, selected['validation'])
        p.g.write(adjunct / 'hook-evidence.json', evidence)
        p.g.write(adjunct / 'replay.json', replay)
        p.validate_result({'output': p.OUTPUT.as_posix()})
        budget()
        names = p.completion()['artifacts']
        status = {'state': 'completed', 'operation': p.KIND, 'scientific_training_state': 'stopped',
            'scientific_reason': 'runtime_stop_after_validation', 'schedule_completed': False, 'selected_epoch': 1,
            'updates': 27235, 'train_examples': 3477437, 'validation': selected['validation'], 'evaluate_calls': 1,
            'optimizer_updates_after_evaluation': 0, 'selected_weights_equal_evaluated': True,
            'sidecar_status': 'externally-bound-selectedE1-singlepass', 'selected_checkpoint_sha256': selected['sha256'],
            'evaluated_weights_digest': selected_digest, 'sidecar_sha256': p.sha(sidecar), 'test_accessed': False,
            'elapsed_seconds': time.perf_counter() - started, 'artifacts': names,
            'sha256': {name: p.sha(p.OUTPUT / name) for name in names}}
        budget()
        p.g.write(p.OUTPUT / 'singlepass-completion.json', status)
        budget()
    except BaseException as error:
        status = {'state': 'failed-unbound-sidecar', 'operation': p.KIND, 'schedule_completed': False,
            'test_accessed': False, 'error': type(error).__name__ + ': ' + str(error),
            'sidecar_present': sidecar.exists(), 'selected_checkpoint_present': (p.OUTPUT / 'model/selected-epoch0001.pt').exists(),
            'hook_attempt': hook.evidence if hook is not None else None, 'elapsed_seconds': time.perf_counter() - started,
            'selected_export_claim': False}
        p.g.write(p.OUTPUT / 'singlepass-completion.json', status)
        raise
    finally:
        if hook is not None:
            hook.restore()
    print(json.dumps({'state': status['state'], 'operation': p.KIND, 'schedule_completed': False}))


if __name__ == '__main__':
    main()
