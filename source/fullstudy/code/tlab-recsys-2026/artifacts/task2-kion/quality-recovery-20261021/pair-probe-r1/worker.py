"""Disposable36-update sampler performance fixture; frozen train.update only."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


class BudgetStop(RuntimeError):
    pass


def quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def apply_final_budget(report, reason, elapsed, stage):
    """A completed fixture cannot promote an exhausted finalization boundary."""
    if reason:
        error = reason + '; budget exhausted during ' + stage
        report.update(benchmark_match=False, no_match_reason=error, whole_worker_seconds=elapsed)
        return error
    return None


def validation_usage(provider_loaded=None):
    """Unknown until successful factory return; constructor physically loads val."""
    if provider_loaded is not None and provider_loaded is not True:
        raise ValueError('Provider validation load is unknown or confirmed true; never guess false')
    return {'validation_loaded_by_provider': provider_loaded, 'validation_evaluated': False,
        'validation_rows_used_for_updates': 0}


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.worker_request.read_text(encoding='utf-8'))
    path = HERE / 'probe.py'
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != request['source_hashes'][path.as_posix()]:
        raise ValueError('Exact reviewed probe bytes changed before import')
    spec = importlib.util.spec_from_file_location('_pair_probe_worker_metadata', path)
    p = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = p
    exec(compile(payload, str(path), 'exec'), p.__dict__)
    p.validate_spec(request['specification'])
    plan = p.g.bound(request['baseline_plan'])
    p.require(request['baseline_plan'] == {'path': p.BASE_PLAN.as_posix(), 'sha256': p.BASE_SHA}, 'exact immutable baseline plan')
    p.require(request['baseline_config'] == {'path': p.BASE_CONFIG.as_posix(), 'sha256': p.BASE_CONFIG_SHA}
        and p.sha(p.BASE_CONFIG) == p.BASE_CONFIG_SHA, 'exact original baseline config before any ML import')
    p.require(os.environ.get('KION_OUTPUT_DIR') == (p.OUTPUT).as_posix()
        and os.environ.get('CUDA_VISIBLE_DEVICES') == plan['gpu_uuid'], 'reviewed native owned worker environment')
    p.require(args.output.resolve() == p.OUTPUT / 'benchmark' and not args.output.exists()
        and args.stop_file.resolve() == p.OUTPUT / 'stop-request.json', 'fresh fixed benchmark output/stop boundary')
    for path, digest in request['source_hashes'].items():
        p.require(p.sha(path) == digest, 'exact execution helper source')
    deadline = datetime.fromisoformat(request['deadline_utc'].replace('Z', '+00:00'))
    p.require(deadline.utcoffset() is not None, 'timezone-aware finite worker deadline')
    args.output.mkdir()
    report = {'contract': p.KIND, 'benchmark_match': False, 'quality_claim': False,
        'accepted_updates': 0, 'records': [], 'origin_cell_orders': p.ORDERS,
        'provider_loads': 0, 'production_checkpoint': False, **validation_usage(), 'test_accessed': False,
        'whole_worker_started_utc': datetime.now(timezone.utc).isoformat(),
        'timing_includes': 'sync, frozen update, AMP retry, raw candidate instrumentation; shared model-independent caches and counterbalanced origin order',
        'statistical_limit': 'one36-update fixture; AB/BA/AB counterbalance is not an independent replicate or quality estimate'}
    fixture = {'origins': {}, 'origin_cell_orders': p.ORDERS, 'seed': 2026, 'candidate_fixture_hashes': []}
    provider, error, model, optimizer, scaler = None, None, None, None, None

    def budget_reason():
        if args.stop_file.exists():
            return 'native_stop_file'
        if datetime.now(timezone.utc) >= deadline:
            return 'absolute_deadline'
        if time.perf_counter() - started >= 810:
            return 'worker810s_boundary'
        return None

    def stop():
        reason = budget_reason()
        if reason:
            raise BudgetStop(reason + '; bounded900s operation, no fallback/retry')

    def publish():
        for name, value in (('report.json', report), ('fixture.json', fixture)):
            p.g.write(args.output / name, value)

    try:
        stop()
        cold = time.perf_counter()
        sys.path.insert(0, (p.COMMON / 'scripts').as_posix())
        runtime = p.load(p.COMMON / 'scripts/kion_ready_runtime.py', plan['source_hashes']['scripts/kion_ready_runtime.py'], '_pair_probe_accepted_bootstrap')
        runtime.configure_paths()
        train = p.load(p.COMMON / 'scripts/kion_complete_train.py', plan['source_hashes']['scripts/kion_complete_train.py'], '_pair_probe_frozen_functions')
        np, torch = train.np, train.torch
        p.require(torch.cuda.is_available(), 'explicit CUDA required')
        device = torch.device('cuda')
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        train_args = train.parser().parse_args(plan['jobs'][0]['argv'][2:])
        train_args.schedule_total_epochs = 5
        report['cold_runtime_import_seconds'] = time.perf_counter() - cold
        stop()
        cold = time.perf_counter()
        module, factory = train_args.provider.split(':', 1)
        provider = getattr(__import__(module, fromlist=[factory]), factory)(train_args)
        report['provider_loads'] = 1
        report.update(validation_usage(True))
        report['cold_provider_load_seconds'] = time.perf_counter() - cold
        schedule = train.TrainingSchedule(provider, train_args)
        p.require(schedule.contract() == p.read(request['baseline_config']['path'])['contract']['training_schedule'], 'exact fullrows/origin/length eligibility closure')
        expected_baseline = p.g.bound(request['baseline_config'])
        p.require(len(provider.feature_columns['item']) == 5736 and len(provider.feature_columns['user']) == 3252
            and len(provider.feature_columns['candidate']) == 1068, 'all73ready feature dimensions retained')
        for origin in p.ORIGINS:
            stop()
            rows = schedule.origin_rows[origin]
            groups = p.fixture_rows(rows, schedule.lengths[rows], origin)
            fixture['origins'][str(origin)] = {'rows': groups, 'labels': p.GROUPS,
                'row_hash': p.g.canonical(groups), 'eligible_origin_rows': int(len(rows)),
                'length_ranges': [{'min': int(np.min(schedule.lengths[group])), 'max': int(np.max(schedule.lengths[group])),
                    'mean': float(np.mean(schedule.lengths[group]))} for group in groups]}
        fixture['row_fixture_hash'] = p.g.canonical(fixture['origins'])
        publish()
        psutil = p.g.load('_pair_probe_existing_psutil', p.g.PSUTIL, p.g.HELPER_HASHES[p.g.PSUTIL.as_posix()]).load()
        process = psutil.Process()

        class Meter:
            def __init__(self, owner):
                self.owner, self.calls = owner, []

            def __getattr__(self, name):
                return getattr(self.owner, name)

            def candidate_features(self, rows, ids, split):
                p.require(split == 'train' and ids.shape == (128, 4) and len(rows) == 128, 'exactC4/B128 raw pair call')
                target = self.owner.splits['train']['y'][rows]
                p.require(np.array_equal(ids[:, 0], target) and np.all(ids[:, 1:] != target[:, None])
                    and np.all(np.sort(ids, axis=1)[:, 1:] != np.sort(ids, axis=1)[:, :-1]), 'forcedpositive plus3unique non-target candidates')
                value = self.owner.candidate_features(rows, ids, split)
                p.require(value.shape == (128, 4, 1068) and value.nbytes == 2187264, 'exact raw pair cell/byte count')
                self.calls.append({'candidate_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
                    'candidate_dtype': str(ids.dtype), 'candidate_shape': list(ids.shape),
                    'pair_cells': int(ids.size), 'raw_feature_bytes': int(value.nbytes)})
                return value

        cells = {cell['id']: cell for cell in p.CELLS}
        first_retrieval = {}
        for origin, cell_order in zip(p.ORIGINS, p.ORDERS):
            groups = fixture['origins'][str(origin)]['rows']
            for cell_id in cell_order:
                stop()
                cell = cells[cell_id]
                cold = time.perf_counter()
                train.random.seed(2026)
                np.random.seed(2026)
                torch.manual_seed(2026)
                torch.cuda.manual_seed_all(2026)
                train_args.hard_negatives, train_args.uniform_negatives = cell['hard_negatives'], cell['uniform_negatives']
                train_args.auxiliary_weight = 1.
                model = train.make_model(provider, train_args).to(device)
                actual_config = train.make_config(provider, model, train_args, device, schedule=schedule)
                expected_config = copy.deepcopy(expected_baseline)
                expected_config['contract']['training'].update(schedule_total_epochs=5,
                    hard_negatives=cell['hard_negatives'], uniform_negatives=cell['uniform_negatives'])
                p.require(actual_config == expected_config, 'exact original allfeatures/config apart from declared horizon/sampler')
                optimizer = torch.optim.AdamW(model.parameters(), lr=.0001, weight_decay=.01)
                scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024)
                torch.cuda.synchronize()
                initial_seconds = time.perf_counter() - cold
                torch.cuda.reset_peak_memory_stats(device)
                meter = Meter(provider)
                model.train()
                for local_update, (rows, label) in enumerate(zip(groups, p.GROUPS)):
                    stop()
                    for group in optimizer.param_groups:
                        group['lr'] = train.shared.learning_rate(local_update, 136175, .05, .0001)
                    meter.calls = []
                    scale_before = float(scaler.get_scale())
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    loss, retries = train.update(model, meter, [np.asarray(rows, np.int64)], optimizer, scaler, device,
                        hard_negatives=cell['hard_negatives'], uniform_negatives=cell['uniform_negatives'],
                        auxiliary_weight=1., candidate_chunk_size=64, collect_loss=True)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - tick
                    report['accepted_updates'] += 1
                    invariant = None
                    if local_update == 0:
                        previous = first_retrieval.setdefault(origin, loss['retrieval_ce'])
                        invariant = {'reference': previous, 'actual': loss['retrieval_ce'],
                            'tolerance': max(.01, abs(previous) * .001),
                            'passed': abs(previous - loss['retrieval_ce']) <= max(.01, abs(previous) * .001)}
                    record = {'origin': origin, 'cell': cell_id, 'local_update': local_update + 1,
                        'label': label, 'measured': local_update >= 2, 'rows_sha256': p.g.canonical(rows),
                        'update_wall_seconds': elapsed, 'model_optimizer_scaler_init_seconds': initial_seconds if local_update == 0 else None,
                        'loss': loss, 'amp_retries': retries, 'amp_scale_before': scale_before, 'amp_scale_after': float(scaler.get_scale()),
                        'first_retrieval_invariant': invariant,
                        'finite_accepted_update': True, 'rss_bytes': process.memory_info().rss,
                        'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                        'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'raw_pair_calls': list(meter.calls)}
                    report['records'].append(record)
                    fixture['candidate_fixture_hashes'].append({'origin': origin, 'cell': cell_id, 'update': local_update + 1,
                        'candidates': list(meter.calls)})
                    publish()
                    p.require(all(math.isfinite(v) for v in loss.values()) and len(meter.calls) == 1 + retries,
                        'finite accepted update and original cached singlechunk/retry pair-call count')
                    p.require(invariant is None or invariant['passed'], 'first retrievalCE invariant across fresh samefixture cells')
                del model, optimizer, scaler, meter
                model, optimizer, scaler = None, None, None
                gc.collect()
                torch.cuda.empty_cache()
        p.require(report['accepted_updates'] == 36 and report['provider_loads'] == 1, 'all exactly36 updates under single provider load')
        stop()
        report['benchmark_match'] = True
        report['first_retrieval_ce_by_origin'] = first_retrieval
    except BudgetStop as caught:
        error = str(caught)
        report['no_match_reason'] = error
    except BaseException as caught:
        error = type(caught).__name__ + ': ' + str(caught)
        report['no_match_reason'] = error
    finally:
        if provider is not None:
            provider.close()
        error = apply_final_budget(report, budget_reason(), time.perf_counter() - started, 'final update or cleanup') or error
        report['whole_worker_seconds'] = time.perf_counter() - started
        report['summary'] = []
        for cell in [cell['id'] for cell in p.CELLS]:
            for origin in p.ORIGINS:
                records = [r for r in report['records'] if r['cell'] == cell and r['origin'] == origin and r['measured']]
                values = [r['update_wall_seconds'] for r in records]
                report['summary'].append({'cell': cell, 'origin': origin, 'measured_updates': len(values),
                    'p50_seconds': quantile(values, .5), 'p90_seconds': quantile(values, .9)})
            for label in ('short', 'medium', 'long', 'measured-mixed'):
                records = [r for r in report['records'] if r['cell'] == cell and r['label'] == label and r['measured']]
                values = [r['update_wall_seconds'] for r in records]
                report['summary'].append({'cell': cell, 'length_group': label, 'measured_updates': len(values),
                    'p50_seconds': quantile(values, .5), 'p90_seconds': quantile(values, .9)})
        report['candidate_order_fixture_sha256'] = p.g.canonical(fixture['candidate_fixture_hashes'])
        report['fullE1_admission'] = 'not_admitted: exploratory36-update timing only; root must include fullvalidation/checkpoints/selectedexport and15%margin in any future fullE1 decision'
        publish()
        reason = budget_reason()
        if reason:
            error = apply_final_budget(report, reason, time.perf_counter() - started, 'report publication')
            publish()
        names = ['report.json', 'fixture.json']
        receipt = {'state': 'completed', 'operation': p.KIND,
            'benchmark_match': report['benchmark_match'], 'accepted_updates': report['accepted_updates'],
            'quality_claim': False, 'production_checkpoint': False,
            **validation_usage(report['validation_loaded_by_provider']), 'test_accessed': False,
            'no_match_reason': error, 'artifacts': names, 'sha256': {name: p.sha(args.output / name) for name in names}}
        p.write_new(args.output / 'completion.json', receipt)
        reason = budget_reason()
        if reason and report['benchmark_match']:
            error = apply_final_budget(report, reason, time.perf_counter() - started, 'final receipt publication')
            publish()
            receipt.update(benchmark_match=False, no_match_reason=error,
                sha256={name: p.sha(args.output / name) for name in names})
            p.g.write(args.output / 'completion.json', receipt)
    print(json.dumps({'state': 'completed_disposable_benchmark', 'benchmark_match': report['benchmark_match'], 'accepted_updates': report['accepted_updates']}))


if __name__ == '__main__':
    main()
