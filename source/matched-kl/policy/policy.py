"""Versioned original-C H1 experiments. Import is stdlib-only; real runs are root-owned."""
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
BASE = ROOT.parent
OLD = BASE / 'c-rl-20261024-r1'
FINAL = BASE / 'c-rl-final-20261025-r1'
CONTRACT = 'kion-current-C1070-kl-seeds-H1-v1'
STEPS = (0, 250, 500, 1000)
SEEDS = (20261024, 20261025, 20261026)
UTILITY_MODULE = FINAL / 'reward/utility.py'
POLICY_MODULE = OLD / 'policy/policy_v2.py'
RESIDUAL_MODULE = FINAL / 'policy/residual.py'
SOURCE_PINS = {
    POLICY_MODULE: 'da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba',
    UTILITY_MODULE: '4cffa8d876c84723326f035b825cafc520c4ad561d98e3851bfe38a5d3e1cd0f',
    RESIDUAL_MODULE: '781db1659305f2e6cc020f96cf52266201fa9f2ecfe21adde18bf6f16127b7f0',
}
FIXED = {
    'cache_dir': (OLD / 'policy/train-cache-r1/cache.json', 'e76c365d1245355d5314e383ba53499800eb3c099fa0af7de106cced5fad15e5'),
    'reward_dir': (OLD / 'reward/train512-r1/reward.json', 'e8047d12f8009bb422cc4956261a355dc10e05f42d73351ef76ccd5f7bc9ce05'),
    'utility_dir': (FINAL / 'reward/residual-r1/train/utility.json', 'e888cb47586b8ea2a4e5323a1051401e949f602ec36e43d36254684e80a320a5'),
    'prefit_path': (FINAL / 'reward/residual-r1/prefit.json', 'f1431dd1092a5d7ef96e2ed89d4c2889067cd3558fa0f478f29eff73825c3d97'),
    'preregistration': (FINAL / 'protocols/preregistration.json', '78c1568c35349394032333696ffce758b9ec1b7aea47edb9eb7023368e7ba59c'),
}
C_SHA = '9c42a75a0c3092ceef3ee8f94b5a62b9b69808a81a42be4ed8b1c3acec277ea7'
PROOF_SHA = 'dbdc6622bd620c15cbab77e2b12db4a241bb6e9577270fc0578b5da768d1e32b'


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


def parameters(request):
    require(request.get('utility') in ('completion', 'residual'), 'Explicit completion/residual utility required')
    beta = request.get('beta')
    require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0, 'Explicit positive finite beta required')
    require(type(request.get('seed')) is int and request['seed'] in SEEDS, 'One supported sampler seed required')
    expected_updates = 0 if request.get('mode') == 'evaluate_train512' else 1000
    for key, expected in dict(rl_updates=expected_updates, batch_size=16, learning_rate=1e-5,
                              optimizer='Adam', weight_decay=0, grad_clip=1,
                              supervised_updates=0, new_rm_fits=0, final_test_reads=0).items():
        require(request.get(key, expected) == expected, 'Fixed experiment parameter differs: ' + key)
    require(request.get('checkpoint_steps', list(STEPS)) == list(STEPS), 'Fixed checkpoints required')
    require(request.get('precision') == 'cuda-fp16' and request.get('device') == 'cuda', 'Original CUDA FP16 required')
    require(request.get('cpu_threads') == 4 and request.get('reserve_gib') == 2
            and request.get('cap_gib') in (4, 8), 'Root CPU/RAM bounds required')
    require(request.get('start_from', 'original-C') == 'original-C', 'Original-C start required')
    return request['utility'], float(beta), request['seed']


def load_frozen(request, path, name):
    path = path.resolve()
    digest = SOURCE_PINS[path]
    require(sha(path) == digest == request['source_hashes'].get(path.as_posix()), 'Frozen source pin changed: ' + str(path))
    spec = importlib.util.spec_from_file_location('_kl_seeds_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_dependencies(request):
    require(request['source_hashes'].get(Path(__file__).resolve().as_posix()) == sha(__file__), 'Own policy source unpinned')
    utility = load_frozen(request, UTILITY_MODULE, 'utility')
    residual = load_frozen(request, RESIDUAL_MODULE, 'residual')
    policy = load_frozen(request, POLICY_MODULE, 'policy')
    declared = request['policy_module']
    declared = declared['path'] if isinstance(declared, dict) else declared
    require(Path(declared).resolve() == POLICY_MODULE.resolve(), 'Wrong immutable policy module')
    for key, (path, expected) in FIXED.items():
        given = Path(request[key]).resolve()
        if key.endswith('_dir'):
            given /= path.name
        require(given == path.resolve() and sha(path) == expected, 'Exact frozen input required: ' + key)
        require(request['input_hashes'].get(path.resolve().as_posix()) == expected, 'Input metadata not pinned: ' + key)
    require(request['preregistration_sha256'] == FIXED['preregistration'][1]
            and request['prefit_sha256'] == FIXED['prefit_path'][1], 'Frozen utility provenance differs')
    require(Path(request['head_dir']).resolve() == (BASE / 'three-head-262k-20261024-r1/heads').resolve()
            and request['input_hashes'].get((Path(request['head_dir']) / 'C/head.pt').as_posix()) == C_SHA,
            'Exact original-C input pin required')
    proof = request['parity_proof']
    require(Path(proof['path']).resolve() == (OLD / 'policy/repair-proof-r1.json').resolve()
            and proof['sha256'] == PROOF_SHA, 'Exact producer-slot proof required')
    utility.validate_preregistration(request)  # Frozen provenance, not the new beta/seed experiment setting.
    residual.validate_prefit(request, utility)
    return utility, policy, residual


def owned_output(request, key, directory=False):
    path = Path(request[key]).resolve()
    require(path.is_relative_to((ROOT / 'policy').resolve()) and path != (ROOT / 'policy').resolve()
            and not path.exists(), 'Fresh experiment-owned output required')
    return path


def validate_cache(request, manifest, residual, policy):
    require(manifest['cohort'] == 'train' and manifest['rows'] == 512 and manifest['precision'] == 'cuda-fp16'
            and manifest['C_checkpoint_sha256'] == C_SHA, 'Exact original train512 FP16 cache required')
    residual.validate_parity_proof(request, manifest, policy)


def values_and_support(request, utility, policy, arrays, manifest, np):
    reward, support, fallback = policy.reward_arrays(request['reward_dir'], arrays, manifest, np)
    values, _, derived_support = utility.utility_arrays(request['utility_dir'], arrays, manifest, np, policy)
    require(bool(np.array_equal(support, derived_support)), 'Completion/residual support differs')
    return reward, values, support, fallback


def sample_positions(rng, eligible):
    return [rng.choice(eligible) for _ in range(16)]


def original_head(request, policy, torch, device):
    """Frozen supervised checkpoint metadata stays seed24/B128; RL sampler is separate."""
    heads = policy.load(request, 'heads')
    frozen_request = dict(request, seed=20261024, batch_size=128, epochs=3)
    return heads.load_heads(frozen_request, torch, device)['C']


def require_fit_admission(request, check):
    """Root owns the shared ledger and must bind each launched fit to a reservation."""
    admission = request.get('fit_admission', {})
    require(isinstance(admission, dict) and admission.get('maximum_new_fits') == 8
            and type(admission.get('fit_ordinal')) is int and 1 <= admission['fit_ordinal'] <= 8
            and isinstance(admission.get('reservation_id'), str) and bool(admission['reservation_id'])
            and isinstance(admission.get('ledger_path'), str) and bool(admission['ledger_path'])
            and isinstance(admission.get('ledger_sha256'), str) and len(admission['ledger_sha256']) == 64,
            'Root global fit-ledger reservation required')
    require(check is not None, 'Root runtime/fit admission callback required')
    check(stage='kl-seeds-fit-admission', fit_admission=admission)
    return admission


def load_checkpoint(request, descriptor, utility, policy, residual, torch, device):
    path = Path(descriptor['path']).resolve()
    require(path.is_relative_to(OLD / 'policy') or path.is_relative_to(FINAL / 'policy')
            or path.is_relative_to(ROOT / 'policy'), 'Checkpoint outside admitted lineage roots')
    require(sha(path) == descriptor['sha256'], 'Checkpoint pin changed')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    lineage = payload.get('lineage', {})
    contract = payload.get('contract')
    require(payload.get('step') in STEPS and payload['step'] == descriptor['step']
            and lineage.get('C_checkpoint_sha256') == C_SHA
            and lineage.get('cache_sha256') == FIXED['cache_dir'][1], 'Wrong checkpoint/original-C ancestor')
    require(contract in (policy.CONTRACT, residual.CONTRACT, CONTRACT), 'Unsupported checkpoint contract')
    expected_utility = 'completion' if contract == policy.CONTRACT else 'residual'
    if contract == CONTRACT:
        expected_utility = lineage.get('utility')
        require(expected_utility in ('completion', 'residual') and lineage.get('seed') in SEEDS
                and lineage.get('start_from') == 'original-C'
                and lineage.get('sampler_seed') == lineage.get('seed')
                and lineage.get('policy_source_sha256') == sha(__file__)
                and type(lineage.get('beta')) in (int, float) and math.isfinite(lineage['beta']) and lineage['beta'] > 0,
                'Wrong new experiment lineage')
    elif contract == residual.CONTRACT:
        require(lineage.get('utility_contract') == utility.CONTRACT and lineage.get('utility_formula') == utility.FORMULA
                and lineage.get('start_from') == 'original-C' and lineage.get('beta') == .5, 'Wrong frozen residual lineage')
    require(descriptor.get('utility') == expected_utility, 'Checkpoint utility declaration differs')
    head = original_head(request, policy, torch, device)
    head.load_state_dict(payload['state_dict'], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, payload


def load_head(request, torch, device, checkpoint):
    """Public new-lineage loader for screen/full evaluators; no train-cache opening."""
    beta, seed, chosen = request.get('beta'), request.get('seed'), request.get('utility')
    require(type(beta) in (int, float) and math.isfinite(beta) and beta > 0
            and type(seed) is int and seed in SEEDS and chosen in ('completion', 'residual'),
            'Explicit checkpoint beta/seed/utility required')
    require(request.get('sampler_seed', seed) == seed, 'Sampler seed declaration differs')
    require(request['source_hashes'].get(Path(__file__).resolve().as_posix()) == sha(__file__), 'Own source unpinned')
    policy = load_frozen(request, POLICY_MODULE, 'loader_policy')
    path = Path(checkpoint).resolve()
    require(path.is_relative_to(ROOT / 'policy')
            and sha(path) == request['rl_checkpoint_sha256'], 'New checkpoint pin/path differs')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    lineage = payload.get('lineage', {})
    require(payload.get('contract') == CONTRACT and payload.get('step') in STEPS
            and lineage.get('C_checkpoint_sha256') == C_SHA
            and lineage.get('cache_sha256') == FIXED['cache_dir'][1]
            and lineage.get('start_from') == 'original-C' and lineage.get('beta') == beta
            and lineage.get('utility') == chosen and lineage.get('seed') == seed
            and lineage.get('sampler_seed') == seed and lineage.get('policy_source_sha256') == sha(__file__),
            'New checkpoint experiment lineage differs')
    require(sha(Path(request['head_dir']) / 'C/head.pt') == C_SHA, 'Original C changed')
    head = original_head(request, policy, torch, device)
    head.load_state_dict(payload['state_dict'], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head


def evaluate_train512(request, check=None):
    """Cache-only all512 KL/support/reward diagnostics; no optimizer/provider/feature preparation."""
    parameters(request)
    utility, policy, residual = load_dependencies(request)
    check = policy.checked(request, check)
    output = owned_output(request, 'evaluation_output')
    descriptors = request['checkpoints']
    require(isinstance(descriptors, list) and 1 <= len(descriptors) <= 32, 'Bounded checkpoint list required')
    require(len({entry['name'] for entry in descriptors}) == len(descriptors), 'Duplicate checkpoint names')
    import numpy as np
    import torch
    device = torch.device('cuda')
    manifest, arrays = policy.open_cache(request['cache_dir'], np)
    try:
        validate_cache(request, manifest, residual, policy)
        reward, derived, support, fallback = values_and_support(request, utility, policy, arrays, manifest, np)
        logp0, _ = policy.distribution(np, arrays['reference_scores'])
        table = torch.as_tensor(np.array(arrays['item_table'], copy=True), device=device).detach()
        layout = policy.producer_layout(arrays['origins'])
        original = original_head(request, policy, torch, device).eval()
        reference = copy.deepcopy(original).eval()
        for model in (original, reference):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        def score_all(head, verify=False):
            scores = np.empty((512, 1024), dtype=np.float32)
            maximum_error = 0.
            with torch.inference_mode():
                for start in range(0, 512, 16):
                    check(stage='kl-seeds-train512-cache-score', rows=start)
                    positions = list(range(start, start + 16))
                    shaped = policy.prepare_shaped_inputs(arrays, positions, table, torch, np, device, layout)
                    with policy.precision(torch, request, device):
                        actual = policy.score_shaped(head, shaped, torch)
                        if verify:
                            refs = policy.score_shaped(reference, shaped, torch)
                            require(torch.equal(actual, refs), 'Update0 actor/reference mismatch')
                    values = actual.float().cpu().numpy()
                    require(bool(np.isfinite(values).all()), 'Nonfinite checkpoint scores')
                    scores[positions] = values
                    if verify:
                        expected = np.asarray(arrays['reference_scores'][positions])
                        maximum_error = max(maximum_error, float(np.abs(values - expected).max()))
                        require(bool(np.array_equal(values, expected)), 'Update0 cached score must be exact')
                        require(all(policy.rank(arrays['candidate_ids'][pos], values[i], arrays['candidate_raw_ids'][pos]) ==
                                    policy.rank(arrays['candidate_ids'][pos], expected[i], arrays['candidate_raw_ids'][pos])
                                    for i, pos in enumerate(positions)), 'Update0 top10 mismatch')
            return scores, maximum_error
        base_scores, error = score_all(original, True)
        def diagnostics(scores):
            return {name: policy.diagnostics(np, scores, arrays, values, support, logp0)
                    for name, values in (('completion', reward), ('residual', derived))}
        results = {'original-C': dict(step=0, metrics=diagnostics(base_scores))}
        for descriptor in descriptors:
            head, payload = load_checkpoint(request, descriptor, utility, policy, residual, torch, device)
            scores, _ = score_all(head, payload['step'] == 0)
            results[descriptor['name']] = dict(checkpoint=descriptor, contract=payload['contract'],
                                               lineage=payload['lineage'], metrics=diagnostics(scores))
            del head
        record = dict(contract=CONTRACT, state='completed', mode='evaluate_train512', rows=512,
            cache_sha256=FIXED['cache_dir'][1], reward_sha256=FIXED['reward_dir'][1], utility_sha256=FIXED['utility_dir'][1],
            results=results, update0_actor_reference_exact=True, update0_cached_scores_exact=True,
            update0_top10_exact=True, update0_max_abs_error=error, fallback_pairs=int(fallback.sum()),
            optimizer_created=False, learning_updates=0, provider_opened=False, GPT_opened=False,
            RM_opened=False, final_test_reads=0, raw_feature_calls=0,
            KL_definition='mean_all512 sum_all1024 pi*(log(pi)-log(C)), temperature=1',
            support_definition='unchanged frozen RM support; unsupported advantage=0; all1024 normalized')
        output.parent.mkdir(parents=True, exist_ok=True)
        policy.write(output, record)
        return record
    finally:
        policy.close_cache(arrays)


def train(request, check=None):
    chosen_utility, beta, seed = parameters(request)
    utility, policy, residual = load_dependencies(request)
    admission = require_fit_admission(request, check)
    check = policy.checked(request, check)
    output = owned_output(request, 'rl_output', directory=True)
    require(request.get('precision', 'cuda-fp16') == 'cuda-fp16' and request.get('amp_validation_passed') is True,
            'Same original CUDA FP16 precision and passed root validation required')
    import numpy as np
    import torch
    device = torch.device(request.get('device', 'cuda'))
    require(device.type == 'cuda', 'Actual CUDA FP16 residual fit required')
    manifest, arrays = policy.open_cache(request['cache_dir'], np)
    try:
        require(manifest['cohort'] == 'train' and manifest['rows'] == 512 and manifest['precision'] == 'cuda-fp16',
                'Exact original train512 FP16 cache required')
        validate_cache(request, manifest, residual, policy)
        metadata = read(Path(request['utility_dir']) / 'utility.json')
        require(metadata['preregistration_sha256'] == request['preregistration_sha256'], 'Wrong utility preregistration')
        require(Path(metadata['parent_reward']['path']).resolve() == Path(request['reward_dir']).resolve() / 'reward.json',
                'Original reward directory differs')
        reward, derived, support, _ = values_and_support(request, utility, policy, arrays, manifest, np)
        values = reward if chosen_utility == 'completion' else derived
        _, probs0 = policy.distribution(np, arrays['reference_scores'])
        advantages, active = policy.effective_advantage(np, probs0, values, support)
        require(bool(active.any()), 'No supported train states')
        model = original_head(request, policy, torch, device).eval()
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
                require(bool(np.array_equal(actual, expected)), 'Update0 original-C cached score must be exact')
                require(all(policy.rank(arrays['candidate_ids'][pos], actual[i], arrays['candidate_raw_ids'][pos]) ==
                            policy.rank(arrays['candidate_ids'][pos], expected[i], arrays['candidate_raw_ids'][pos])
                            for i, pos in enumerate(positions)), 'Update0 top10 mismatch')
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        output.mkdir(parents=True)
        record = dict(contract=CONTRACT, state='running', implementation=policy.IMPLEMENTATION,
            start_from='original-C', beta=beta, utility=chosen_utility, seed=seed, sampler_seed=seed,
            fit_admission=admission, updates=0, checkpoints=[], batch_size=16,
            learning_rate=1e-5, optimizer='Adam', weight_decay=0, grad_clip=1,
            precision='cuda-fp16', frozen_encoder=True, supervised_updates=0,
            C_checkpoint_sha256=manifest['C_checkpoint_sha256'], cache_sha256=sha(Path(request['cache_dir']) / 'cache.json'),
            utility_contract=utility.CONTRACT if chosen_utility == 'residual' else 'current-C-local-RM-r2-bridge-v1',
            utility_formula=utility.FORMULA if chosen_utility == 'residual' else 'r_RM(s,i)',
            utility_sha256=sha(Path(request['utility_dir']) / 'utility.json'),
            reward_sha256=metadata['parent_reward']['sha256'], preregistration_sha256=request['preregistration_sha256'],
            prefit_sha256=request['prefit_sha256'], parity_proof=request['parity_proof'],
            immutable_policy_sha256=sha(policy.__file__), utility_source_sha256=sha(UTILITY_MODULE),
            primary_contrast='fixed-latest1000',
            update0_actor_reference_exact=True, update0_top10_exact=True, update0_max_abs_error=maximum_error,
            loss_normalization='one mean over logical real B16 occurrences; no bucket means',
            objective='exact supported C-centered chosen utility minus beta KL; all1024 actions normalized',
            policy_source_sha256=sha(__file__), checkpoint_steps=list(STEPS), update0_cached_scores_exact=True,
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
        rng = random.Random(seed)
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
                loss = policy.exact_loss(torch, scores, refs, adv, beta, valid)
            require(bool(torch.isfinite(loss)), 'Nonfinite residual loss; no retry')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), 'Nonfinite residual gradient')
            if step == 1:
                require(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters()), 'Zero first residual gradient')
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            require(scaler.get_scale() >= scale_before, 'AMP skipped optimizer update; no retry')
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
    mode = request.get('mode')
    require(mode in ('train', 'evaluate_train512'), 'Explicit train/evaluate_train512 mode required')
    return train(request, check) if mode == 'train' else evaluate_train512(request, check)
