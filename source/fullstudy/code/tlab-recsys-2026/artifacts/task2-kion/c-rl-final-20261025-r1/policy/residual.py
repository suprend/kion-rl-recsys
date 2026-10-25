"""One original-C -> residual H1 fit; explicit seams reuse immutable policy_v2 math.

Import is stdlib-only. Native admission, process/RAM budgets and scheduling are root-owned.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = 'kion-current-C1070-residual-H1-v1'
UTILITY_MODULE = ROOT / 'reward' / 'utility.py'
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


def load_utility(request):
    path = UTILITY_MODULE.resolve()
    require(request['source_hashes'].get(path.as_posix()) == sha(path), 'Unpinned root-local utility source')
    declared = request.get('utility_module', path)
    if isinstance(declared, dict):
        require(declared.get('sha256') == sha(path), 'Utility module descriptor pin differs')
        declared = declared['path']
    require(Path(declared).resolve() == path, 'Wrong root-local utility module')
    spec = importlib.util.spec_from_file_location('_final25_derived_utility', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_prefit(request, utility):
    require(sha(request['prefit_path']) == request['prefit_sha256'], 'Prefit diagnostics pin changed')
    receipt = read(request['prefit_path'])
    require(receipt.get('contract') == utility.CONTRACT and receipt.get('state') == 'completed'
            and receipt.get('both_cohorts_ready') is True and receipt.get('prefit_gate_passed') is True
            and receipt.get('preregistration_sha256') == request['preregistration_sha256']
            and receipt['cohorts']['train'].get('nondegenerate_prior') is True, 'Complete nondegenerate two-cohort prefit required')
    require(set(receipt['utility_manifests']) == {'train', 'screen'}, 'Both utility manifests required')
    for cohort, pin in receipt['utility_manifests'].items():
        require(sha(pin['path']) == pin['sha256'], 'Prefit utility manifest changed')
        meta = read(pin['path'])
        require(meta.get('contract') == utility.CONTRACT and meta.get('cohort') == cohort
                and meta.get('preregistration_sha256') == request['preregistration_sha256'], 'Prefit cohort binding differs')
    train = receipt['utility_manifests']['train']
    require(Path(train['path']).resolve() == Path(request['utility_dir']).resolve() / 'utility.json', 'Wrong train utility directory')
    return receipt


def validate_parity_proof(request, manifest, policy):
    pin = request['parity_proof']
    require(sha(pin['path']) == pin['sha256'], 'FP16 parity/gradient proof changed')
    proof = read(pin['path'])
    require(proof.get('state') == 'completed' and proof.get('contract') == policy.CONTRACT
            and proof.get('implementation') == policy.IMPLEMENTATION
            and proof.get('implementation_source_sha256') == policy.sha(policy.__file__)
            and proof.get('cache_sha256') == policy.sha(Path(request['cache_dir']) / 'cache.json')
            and proof.get('C_checkpoint_sha256') == manifest['C_checkpoint_sha256']
            and proof.get('states') == 512 and proof.get('all_actor_reference_exact') is True
            and proof.get('all_cached_allclose') is True and proof.get('all_cached_top10_exact') is True
            and proof.get('optimizer_created') is False and proof.get('learning_updates') == 0
            and proof.get('actor_reference_weights_unchanged') is True
            and proof['gradient_fixture'].get('logical_rows') == 16
            and proof['gradient_fixture'].get('dummy_output_gradient_max') == 0,
            'Exact immutable all512 producer-slot FP16 and gradient proof required')
    gradients = proof['gradient_fixture']['gradients']
    require(gradients and all(g['finite'] for g in gradients.values()) and any(g['nonzero'] for g in gradients.values()),
            'Finite nonzero head gradient proof required')


def sample_positions(rng, eligible):
    """Exactly old random.Random(seed).choice ordered eligible sequence, B16 with replacement."""
    return [rng.choice(eligible) for _ in range(16)]


def probe(request, check=None):
    """Explicit root-only reuse of frozen all512 parity/backward probe; no optimizer."""
    utility = load_utility(request)
    policy = utility.load_policy(request)
    path = Path(request['repair_output']).resolve()
    require(path.is_relative_to(ROOT / 'policy'), 'Parity proof output must be owned by new root')
    return policy.repair_probe(request, check)


def load_head(request, torch, device, checkpoint):
    """Load residual lineage without relabeling it as old selected/completion."""
    utility = load_utility(request)
    policy = utility.load_policy(request)
    heads = policy.load(request, 'heads')
    head = heads.load_heads(request, torch, device)['C']
    expected = request.get('residual_checkpoint_sha256', request.get('rl_checkpoint_sha256'))
    require(expected is not None and sha(checkpoint) == expected, 'Residual checkpoint pin changed')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    lineage = payload.get('lineage', {})
    require(payload.get('contract') == CONTRACT and payload.get('step') in STEPS
            and lineage.get('C_checkpoint_sha256') == sha(Path(request['head_dir']) / 'C/head.pt')
            and lineage.get('utility_contract') == utility.CONTRACT
            and lineage.get('utility_formula') == utility.FORMULA
            and lineage.get('start_from') == 'original-C' and lineage.get('beta') == .5,
            'Wrong residual checkpoint or original-C ancestor')
    head.load_state_dict(payload['state_dict'], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def train(request, check=None):
    utility = load_utility(request)
    policy = utility.load_policy(request)
    check = policy.checked(request, check)
    prereg = utility.validate_preregistration(request)
    prefit = validate_prefit(request, utility)
    output = Path(request['rl_output']).resolve()
    require(output == (ROOT / 'policy' / 'residual-r1').resolve() and not output.exists(),
            'Exactly one fresh residual-r1 fit namespace required; no resume or second fit')
    require(request.get('precision', 'cuda-fp16') == 'cuda-fp16' and request.get('amp_validation_passed') is True,
            'Same original CUDA FP16 precision and passed root validation required')
    import numpy as np
    import torch
    device = torch.device(request.get('device', 'cuda'))
    require(device.type == 'cuda', 'Actual CUDA FP16 residual fit required')
    heads = policy.load(request, 'heads')
    manifest, arrays = policy.open_cache(request['cache_dir'], np)
    try:
        require(manifest['cohort'] == 'train' and manifest['rows'] == 512 and manifest['precision'] == 'cuda-fp16',
                'Exact original train512 FP16 cache required')
        validate_parity_proof(request, manifest, policy)
        metadata = read(Path(request['utility_dir']) / 'utility.json')
        require(metadata['preregistration_sha256'] == request['preregistration_sha256'], 'Wrong utility preregistration')
        require(Path(metadata['parent_reward']['path']).resolve() == Path(request['reward_dir']).resolve() / 'reward.json',
                'Original reward directory differs')
        values, _, support = utility.utility_arrays(request['utility_dir'], arrays, manifest, np, policy)
        _, probs0 = policy.distribution(np, arrays['reference_scores'])
        advantages, active = policy.effective_advantage(np, probs0, values, support)
        require(bool(active.any()), 'No supported train states')
        model = heads.load_heads(request, torch, device)['C'].eval()
        require(sha(Path(request['head_dir']) / 'C/head.pt') == manifest['C_checkpoint_sha256'], 'Original-C cache ancestor differs')
        reference = copy.deepcopy(model).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        table = torch.as_tensor(np.array(arrays['item_table'], copy=True), device=device).detach()
        layout = policy.producer_layout(arrays['origins'])
        maximum_error = 0.
        # Explicit seam: use exact frozen packing/scoring/loss functions, no global replacement.
        with torch.inference_mode():
            for start in range(0, 512, 16):
                check(stage='residual-update0-parity', rows=start)
                positions = list(range(start, start + 16))
                shaped = policy.prepare_shaped_inputs(arrays, positions, table, torch, np, device, layout)
                with policy.precision(torch, request, device):
                    scores = policy.score_shaped(model, shaped, torch)
                    refs = policy.score_shaped(reference, shaped, torch)
                require(torch.equal(scores, refs), 'Update0 actor/reference mismatch')
                expected = np.asarray(arrays['reference_scores'][positions])
                actual = scores.float().cpu().numpy()
                maximum_error = max(maximum_error, float(np.abs(actual - expected).max()))
                require(bool(np.allclose(actual, expected, atol=.003, rtol=.001)), 'Update0 original-C cached score mismatch')
                require(all(policy.rank(arrays['candidate_ids'][pos], actual[i], arrays['candidate_raw_ids'][pos]) ==
                            policy.rank(arrays['candidate_ids'][pos], expected[i], arrays['candidate_raw_ids'][pos])
                            for i, pos in enumerate(positions)), 'Update0 top10 mismatch')
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        output.mkdir(parents=True)
        record = dict(contract=CONTRACT, state='running', implementation=policy.IMPLEMENTATION,
            start_from='original-C', beta=.5, updates=0, checkpoints=[], batch_size=16,
            learning_rate=1e-5, optimizer='Adam', weight_decay=0, grad_clip=1,
            precision='cuda-fp16', frozen_encoder=True, supervised_updates=0,
            C_checkpoint_sha256=manifest['C_checkpoint_sha256'], cache_sha256=sha(Path(request['cache_dir']) / 'cache.json'),
            utility_contract=utility.CONTRACT, utility_formula=utility.FORMULA,
            utility_sha256=sha(Path(request['utility_dir']) / 'utility.json'),
            reward_sha256=metadata['parent_reward']['sha256'], preregistration_sha256=request['preregistration_sha256'],
            prefit_sha256=request['prefit_sha256'], parity_proof=request['parity_proof'],
            immutable_policy_sha256=sha(policy.__file__), utility_source_sha256=sha(UTILITY_MODULE),
            sampler_seed=prereg['sampler_seed'], primary_contrast='fixed-latest1000',
            update0_actor_reference_exact=True, update0_top10_exact=True, update0_max_abs_error=maximum_error,
            loss_normalization='one mean over logical real B16 occurrences; no bucket means',
            objective='exact supported C-centered derived utility minus beta KL; all1024 actions normalized',
            provider_opened=False, GPT_opened=False, RM_opened=False, temperature=1.)
        def save(step):
            path = output / f'head-{step}.pt'
            torch.save(dict(contract=CONTRACT, step=step,
                state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                lineage=dict(record)), path)
            record['checkpoints'].append(dict(step=step, path=path.as_posix(), sha256=sha(path)))
        save(0)
        # Source/schema/prefit/all-state parity gates precede optimizer creation.
        check(stage='residual-before-optimizer')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
        scaler = torch.amp.GradScaler('cuda', enabled=True, init_scale=1024)
        rng = random.Random(prereg['sampler_seed'])
        eligible = [i for i, ok in enumerate(active) if ok]
        sampler_digest = hashlib.sha256()
        history = []
        for step in range(1, 1001):
            check(stage='residual-exact-H1-update', step=step)
            positions = sample_positions(rng, eligible)
            sampler_digest.update(json.dumps(positions, separators=(',', ':')).encode('ascii') + b'\n')
            shaped = policy.prepare_shaped_inputs(arrays, positions, table, torch, np, device, layout)
            refs = torch.as_tensor(np.array(arrays['reference_scores'][positions], copy=True), device=device)
            adv = torch.as_tensor(advantages[positions], device=device, dtype=torch.float32)
            valid = torch.ones(16, device=device)
            optimizer.zero_grad(set_to_none=True)
            with policy.precision(torch, request, device):
                scores = policy.score_shaped(model, shaped, torch)
                loss = policy.exact_loss(torch, scores, refs, adv, .5, valid)
            require(bool(torch.isfinite(loss)), 'Nonfinite residual loss; no retry')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite residual gradient')
            if step == 1:
                require(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters()), 'Zero first residual gradient')
            scaler.step(optimizer)
            scaler.update()
            require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite updated residual parameters')
            record['updates'] = step
            if step % 25 == 0:
                history.append(dict(step=step, loss=float(loss.detach()), grad_norm=float(norm)))
            if step in STEPS:
                save(step)
        require(all(torch.equal(v.detach().cpu(), initial[key]) for key, v in reference.state_dict().items()), 'Frozen C reference changed')
        delta = math.sqrt(float(sum(((v.detach().cpu() - initial[key])**2).sum() for key, v in model.state_dict().items())))
        require(math.isfinite(delta) and delta > 0, 'Missing actual residual parameter change')
        record.update(state='completed', history=history, parameter_l2_delta=delta,
                      sampler_positions_sha256=sampler_digest.hexdigest(),
                      actual_optimizer_updates=1000, reference_weights_unchanged=True)
        check(stage='residual-training-complete', updates=1000)
        policy.write(output / 'training.json', record)
        return record
    finally:
        policy.close_cache(arrays)


def run(request, check=None):
    return train(request, check)
