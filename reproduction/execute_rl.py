"""Execute-gated, cache-conditioned RL replay. Default is stdlib plan inspection.

This is a separately versioned orchestration port, not the historical policy's
run() function. Numerical kernels retain their original verified source bytes.
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
import os
from pathlib import Path, PurePosixPath
import random
import sys
import uuid

import train_rl as prep
import rl_bindings as bound

CONTRACT = 'kion-portable-cache-rl-execution-v1'
HERE = Path(__file__).resolve().parent
RUN_STATE = Path.home() / '.cache' / 'kion-rl-replay'
require = prep.require


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    path = prep.safe_path(path)
    require(path.stat().st_size <= 8 << 20, 'Oversized plan')
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def verify_digest(value, expected):
    data = dict(value)
    actual = data.pop('plan_sha256', None)
    require(actual == expected == prep.canonical_hash(data), 'Inspected plan digest differs')


def execution_plan(preparation, utility=None, seed=None):
    verify_digest(preparation, preparation.get('plan_sha256'))
    require(preparation.get('contract') == prep.CONTRACT, 'Historical preparation plan required')
    prep.validate_science(preparation['scientific_configuration'])
    require((utility is None) == (seed is None), 'Select both utility and seed, or all six fits')
    fits = [dict(utility=u, beta=b, seed=s) for u, b in prep.SCIENCE['utilities'].items()
            for s in prep.SCIENCE['seeds'] if utility is None or (u == utility and s == seed)]
    require(len(fits) in (1, 6), 'Unknown frozen fit')
    sources = {(HERE / name).as_posix(): sha(HERE / name) for name in
               ('execute_rl.py', 'train_rl.py', 'rl_bindings.py', 'rl_heads.py')}
    state = prep.safe_path(RUN_STATE)
    protected = [HERE.parent, prep.safe_path(preparation['output'])]
    protected += [prep.safe_path(e['physical']) for e in preparation['bindings']]
    require(all(state != p and not state.is_relative_to(p) and not p.is_relative_to(state)
                for p in protected), 'Shared run state overlaps inputs/output/repository')
    result = dict(contract=CONTRACT, preparation=preparation, fits=fits, run_state=state.as_posix(),
                  adapter_sources=sources, execution_supported=True,
                  training_launched=False, raw_reproduction_complete=False)
    result['plan_sha256'] = prep.canonical_hash(result)
    return result


def validate_plan(plan, inspected):
    verify_digest(plan, inspected)
    require(plan.get('contract') == CONTRACT, 'Execution plan contract differs')
    p = plan['preparation']
    fits = plan['fits']
    selection = {} if len(fits) == 6 else dict(utility=fits[0]['utility'], seed=fits[0]['seed'])
    require(execution_plan(p, **selection) == plan, 'Execution plan/source/science differs')
    roles = {e['role']: e for e in p['bindings']}
    cache = Path(roles['metadata:cache']['physical'])
    artifacts = cache.parents[3]
    policy_path = next(Path(k) for k in p['source_pins'] if k.endswith('/source/matched-kl/policy/policy.py'))
    fresh = prep.plan(artifacts, roles['metadata:bank']['physical'], p['output'], root=policy_path.parents[3])
    require(fresh == p, 'Preparation inputs or adapter changed since inspection')
    return bound.BindingResolver(p['bindings'])


class RuntimeGuard:
    """Mandatory cooperative checks at admission, every scoring batch and update.

    A single native CUDA operation is not preempted. Deadline overruns are rejected
    immediately after it returns. No retry, precision fallback or resumed fit.
    """
    def __init__(self, deadline, cancel_file, cap_gib, snapshot, now=None):
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
        require(self.deadline.utcoffset() is not None, 'Aware fresh deadline required')
        seconds = (self.deadline - self.now()).total_seconds()
        require(0 < seconds <= 86400, 'Fresh deadline must be within 24 hours')
        require(cap_gib in (4, 8), 'RAM cap must be 4 or 8 GiB')
        self.cap = cap_gib * 2**30
        self.cancel = prep.safe_path(cancel_file)
        self.snapshot = snapshot
        self.last = None

    def __call__(self, **stage):
        require(self.now() < self.deadline, 'Execution deadline exceeded')
        require(not self.cancel.exists(), 'Execution cancelled')
        stats = self.snapshot()
        require(stats['rss'] <= self.cap and stats['available'] >= 2 * 2**30,
                'RAM cap/reserve exceeded')
        require(stats.get('gpu_allocated', 0) <= self.cap, 'GPU allocation cap exceeded')
        self.last = dict(stage, resources=stats)


class ExclusiveRun:
    """O_EXCL reservation shared by every execution in this checkout.

    The append-only run ledger survives aborts. A crash leaves the lock in place;
    recovery requires explicit human inspection, never automatic stale stealing.
    """
    def __init__(self, directory, plan_sha256):
        self.lock = directory / '.rl-execution.lock'
        self.ledger = directory / '.rl-execution-ledger.jsonl'
        self.token = uuid.uuid4().hex
        self.plan_sha256 = plan_sha256

    def event(self, event, **fields):
        value = dict(event=event, reservation_id=self.token, plan_sha256=self.plan_sha256,
                     utc=datetime.now(timezone.utc).isoformat(), **fields)
        with self.ledger.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return value

    def __enter__(self):
        write_new(self.lock, dict(reservation_id=self.token, pid=os.getpid(), plan_sha256=self.plan_sha256))
        try:
            self.event('reserved')
        except BaseException:
            self.lock.unlink()
            raise
        return self

    def __exit__(self, kind, error, tb):
        try:
            self.event('aborted' if kind else 'completed', error=str(error) if error else None)
        finally:
            require(read(self.lock)['reservation_id'] == self.token, 'Execution lock ownership changed')
            self.lock.unlink()


def load_numeric(plan):
    relative = prep.SOURCE_BASE + 'c-rl-20261024-r1/policy/policy_v2.py'
    path = next(Path(p) for p in plan['preparation']['source_pins'] if p.endswith('/' + relative))
    prep.pinned_bytes(path, prep.SOURCE_PINS[relative])
    spec = importlib.util.spec_from_file_location('_portable_rl_verified_numeric', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def request_for(plan):
    entries = {e['role']: e for e in plan['preparation']['bindings']}
    logical = lambda name: entries['metadata:' + name]['logical']
    request = dict(precision='cuda-fp16', device='cuda', cpu_threads=4, reserve_gib=2)
    for name in ('cache', 'reward', 'utility', 'bank'):
        request[name + '_dir'] = str(PurePosixPath(logical(name)).parent)
    request['head_dir'] = str(PurePosixPath(logical('heads')).parent)
    for key in ('prefit', 'preregistration'):
        request['prefit_path' if key == 'prefit' else key] = logical(key)
        request[key + '_sha256'] = entries['metadata:' + key]['sha256']
    request['parity_proof'] = dict(path=logical('proof'), sha256=entries['metadata:proof']['sha256'])
    policy = prep.LOGICAL + 'c-rl-20261024-r1/policy/policy_v2.py'
    request['policy_module'] = policy
    request['source_hashes'] = {policy: prep.SOURCE_PINS[prep.SOURCE_BASE + 'c-rl-20261024-r1/policy/policy_v2.py']}
    return request


def score_all(model, reference, arrays, table, layout, policy, torch, np, device, guard, verify=False):
    scores = np.empty((512, 1024), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, 512, 16):
            guard(stage='parity' if verify else 'diagnostics', rows=start)
            positions = list(range(start, start + 16))
            shaped = policy.prepare_shaped_inputs(arrays, positions, table, torch, np, device, layout)
            with policy.precision(torch, {'precision': 'cuda-fp16'}, device):
                actual = policy.score_shaped(model, shaped, torch)
                if verify:
                    refs = policy.score_shaped(reference, shaped, torch)
                    require(torch.equal(actual, refs), 'Fresh actor/reference scores differ')
            values = actual.float().cpu().numpy()
            require(bool(np.isfinite(values).all()), 'Nonfinite scores')
            if verify:
                expected = np.asarray(arrays['reference_scores'][positions])
                require(bool(np.array_equal(values, expected)), 'Fresh original-C/cache exact parity failed')
                require(all(policy.rank(arrays['candidate_ids'][pos], values[i], arrays['candidate_raw_ids'][pos]) ==
                            policy.rank(arrays['candidate_ids'][pos], expected[i], arrays['candidate_raw_ids'][pos])
                            for i, pos in enumerate(positions)), 'Fresh top10 parity failed')
            scores[positions] = values
            guard(stage='score-batch-complete', rows=start + 16)
    return scores


def fit_one(plan, arm, output, request, bindings, manifest, arrays, rewards, derived, support,
            heads, policy, torch, np, guard, reservation):
    device = torch.device('cuda')
    record = dict(contract=CONTRACT, state='running', updates=0, checkpoints=[], **arm,
                  sampler_seed=arm['seed'], plan_sha256=plan['plan_sha256'],
                  adapter_sources=plan['adapter_sources'], frozen_sources=plan['preparation']['source_pins'],
                  scientific_configuration=prep.SCIENCE, start_from='original-C',
                  C_checkpoint_sha256=manifest['C_checkpoint_sha256'],
                  input_bindings=plan['preparation']['bindings'], fit_admission=reservation,
                  provider_opened=False, GPT_opened=False, RM_opened=False, final_test_reads=0,
                  raw_reproduction_complete=False)
    output.mkdir()
    try:
        guard(stage='load-original-heads')
        model = heads.load_original_heads(plan['preparation'], bindings, torch, device)['C'].eval()
        reference = copy.deepcopy(model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        table = torch.as_tensor(np.array(arrays['item_table'], copy=True), device=device).detach()
        layout = policy.producer_layout(arrays['origins'])
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        base_scores = score_all(model, reference, arrays, table, layout, policy, torch, np, device, guard, True)
        record.update(update0_actor_reference_exact=True, update0_cached_scores_exact=True,
                      update0_top10_exact=True, update0_rows=512, update0_max_abs_error=0.)
        logp0, probs0 = policy.distribution(np, arrays['reference_scores'])
        values = rewards if arm['utility'] == 'completion' else derived
        advantages, active = policy.effective_advantage(np, probs0, values, support)
        eligible = [i for i, ok in enumerate(active) if ok]
        require(bool(eligible), 'No supported train states')
        def diagnostics(scores):
            return {name: policy.diagnostics(np, scores, arrays, val, support, logp0)
                    for name, val in (('completion', rewards), ('residual', derived))}
        def save(step, scores=None):
            guard(stage='checkpoint', step=step)
            scores = scores if scores is not None else score_all(model, reference, arrays, table, layout,
                        policy, torch, np, device, guard)
            metrics = diagnostics(scores)
            path = output / f'head-{step}.pt'
            with path.open('xb') as stream:
                torch.save(dict(contract=CONTRACT, step=step,
                    state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    lineage=copy.deepcopy(record)), stream)
            record['checkpoints'].append(dict(step=step, path=path.as_posix(), sha256=sha(path), metrics=metrics))
            write_new(output / f'checkpoint-{step}.json', record['checkpoints'][-1])
        save(0, base_scores)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        guard(stage='before-optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
        scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024)
        rng, sampler_digest, history = random.Random(arm['seed']), hashlib.sha256(), []
        for step in range(1, 1001):
            guard(stage='update', step=step)
            positions = [rng.choice(eligible) for _ in range(16)]
            sampler_digest.update(json.dumps(positions, separators=(',', ':')).encode('ascii') + b'\n')
            shaped = policy.prepare_shaped_inputs(arrays, positions, table, torch, np, device, layout)
            refs = torch.as_tensor(np.array(arrays['reference_scores'][positions], copy=True), device=device)
            adv = torch.as_tensor(advantages[positions], device=device, dtype=torch.float32)
            valid = torch.ones(16, device=device)
            optimizer.zero_grad(set_to_none=True)
            with policy.precision(torch, request, device):
                scores = policy.score_shaped(model, shaped, torch)
                loss = policy.exact_loss(torch, scores, refs, adv, arm['beta'], valid)
            require(bool(torch.isfinite(loss)), 'Nonfinite loss; no retry')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite gradient')
            if step == 1:
                require(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters()), 'Zero first gradient')
            scale_before = scaler.get_scale()
            guard(stage='before-optimizer-step', step=step)
            scaler.step(optimizer)
            scaler.update()
            require(scaler.get_scale() >= scale_before, 'AMP skipped update; no retry')
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite updated parameters')
            record['updates'] = step
            guard(stage='update-complete', step=step)
            if step % 25 == 0:
                history.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm)))
            if step in prep.SCIENCE['checkpoint_steps']:
                save(step)
        require(all(torch.equal(v.detach().cpu(), initial[k]) for k, v in reference.state_dict().items()), 'Frozen reference changed')
        delta = math.sqrt(float(sum(((v.detach().cpu() - initial[k]) ** 2).sum() for k, v in model.state_dict().items())))
        require(math.isfinite(delta) and delta > 0, 'No finite nonzero parameter change')
        record.update(state='completed', actual_optimizer_updates=1000, history=history,
                      parameter_l2_delta=delta, reference_weights_unchanged=True,
                      sampler_positions_sha256=sampler_digest.hexdigest())
        guard(stage='fit-complete')
        write_new(output / 'training.json', record)
        return record
    except BaseException as exc:
        record.update(state='aborted', error=type(exc).__name__ + ': ' + str(exc), last_runtime_check=guard.last)
        write_new(output / 'aborted.json', record)
        raise


def execute(plan, inspected, deadline, cancel_file, cap_gib):
    bindings = validate_plan(plan, inspected)
    # The process-wide lease precedes imports, CUDA context and all model work.
    state = prep.safe_path(plan['run_state'])
    state.mkdir(parents=True, exist_ok=True)
    with ExclusiveRun(state, inspected) as lease:
        return _execute_locked(plan, inspected, deadline, cancel_file, cap_gib, bindings, lease)


def _execute_locked(plan, inspected, deadline, cancel_file, cap_gib, bindings, lease):
    # Dependencies and all numerical work are below the explicit CLI gate.
    import psutil
    import numpy as np
    import torch
    import rl_heads as heads
    process = psutil.Process()
    def snapshot():
        return dict(rss=process.memory_info().rss, available=psutil.virtual_memory().available,
                    gpu_allocated=torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0)
    guard = RuntimeGuard(deadline, cancel_file, cap_gib, snapshot)
    guard(stage='runtime-admission')
    require(torch.cuda.is_available(), 'CUDA required; CPU/FP32 fallback forbidden')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    torch.cuda.set_per_process_memory_fraction(min(1., cap_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    policy = load_numeric(plan)
    request = request_for(plan)
    request.update(deadline_utc=deadline, cap_gib=cap_gib)
    sources = plan['preparation']['source_pins']
    output = prep.safe_path(plan['preparation']['output'])
    cancel = prep.safe_path(cancel_file)
    protected = [Path(e['physical']) for e in plan['preparation']['bindings']] + [Path(p) for p in sources]
    require(all(cancel != p and not p.is_relative_to(cancel) for p in protected), 'Cancellation path overlaps inputs')
    with nullcontext(lease):
        output.mkdir()
        arrays = None
        receipt = dict(contract=CONTRACT, state='running', plan_sha256=inspected, fits=[],
                       deadline_utc=deadline, cancel_file=cancel.as_posix(), cap_gib=cap_gib,
                       reserve_gib=2, cpu_threads=4, training_launched=False, raw_reproduction_complete=False)
        try:
            for entry in plan['preparation']['bindings']:
                guard(stage='input-hash', role=entry['role'])
                bindings.sha_bound(entry['logical'])
            heads.validate_preregistration_bound(request, bindings)
            bound.validate_prefit_bound(request, bindings)
            manifest, arrays = bound.open_cache_bound(request['cache_dir'], np, bindings)
            heads.validate_historical_metadata(plan['preparation'], bindings, manifest)
            require(manifest['rows'] == 512 and manifest['precision'] == 'cuda-fp16' and manifest['cohort'] == 'train', 'Exact historical train512 required')
            reward, support, _ = bound.reward_arrays_bound(request['reward_dir'], arrays, manifest, np, bindings)
            derived, _, derived_support = bound.utility_arrays_bound(request['utility_dir'], arrays, manifest, np, bindings)
            require(bool(np.array_equal(support, derived_support)), 'Utility support differs')
            for ordinal, arm in enumerate(plan['fits'], 1):
                guard(stage='fit-reservation', ordinal=ordinal)
                reservation = lease.event('fit-reserved', fit_ordinal=ordinal, maximum_fits=len(plan['fits']), **arm)
                receipt['training_launched'] = True
                fit = fit_one(plan, arm, output / f"{arm['utility']}-{arm['seed']}", request, bindings,
                              manifest, arrays, reward, derived, support, heads, policy,
                              torch, np, guard, reservation)
                receipt['fits'].append(dict(arm=arm, state=fit['state'], updates=fit['updates']))
                lease.event('fit-completed', fit_ordinal=ordinal, **arm)
                torch.cuda.empty_cache()
            receipt['state'] = 'completed'
            guard(stage='run-complete')
            write_new(output / 'execution.json', receipt)
            return receipt
        except BaseException as exc:
            receipt.update(state='aborted', error=type(exc).__name__ + ': ' + str(exc), last_runtime_check=guard.last)
            write_new(output / 'aborted.json', receipt)
            raise
        finally:
            if arrays is not None:
                policy.close_cache(arrays)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True, help='Preparation JSON by default; execution JSON with execute flag')
    parser.add_argument('--plan-output', type=Path)
    parser.add_argument('--utility', choices=tuple(prep.SCIENCE['utilities']))
    parser.add_argument('--seed', type=int, choices=prep.SCIENCE['seeds'])
    parser.add_argument('--execute-training', action='store_true')
    parser.add_argument('--inspected-plan-sha256')
    parser.add_argument('--deadline-utc')
    parser.add_argument('--cancel-file', type=Path)
    parser.add_argument('--cap-gib', type=int, choices=(4, 8))
    args = parser.parse_args(argv)
    try:
        plan = read(args.plan)
        if args.execute_training:
            require(all((args.inspected_plan_sha256, args.deadline_utc, args.cancel_file, args.cap_gib)),
                    'Execution requires inspected digest, fresh deadline, cancellation path and RAM cap')
            require(args.utility is None and args.seed is None and args.plan_output is None, 'Execution uses inspected fit selection')
            result = execute(plan, args.inspected_plan_sha256, args.deadline_utc, args.cancel_file, args.cap_gib)
        else:
            require(not any((args.inspected_plan_sha256, args.deadline_utc, args.cancel_file, args.cap_gib)), 'Runtime flags require --execute-training')
            result = execution_plan(plan, args.utility, args.seed)
            if args.plan_output:
                destination = prep.plan_destination(args.plan_output, plan)
                require(destination != prep.safe_path(args.plan), 'Cannot overwrite preparation plan')
                write_new(destination, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'RL execution adapter: {exc}\n')


if __name__ == '__main__':
    main()
