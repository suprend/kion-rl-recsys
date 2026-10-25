"""Root-admitted, inference-only synthetic ID autoregression over a fixed pool.

Import is stdlib-only. No provider/RM calls or target fields are used here.
The caller owns context admission, reward binding, checkpoint pins and teardown.
"""
from __future__ import annotations

import math
import random

CONTRACT = "kion-c3-liveprefix-producer-shape-fixed-observed-rm-v2"
POOL_WIDTH = 1024
CONTEXT_WIDTH = 64


def observed_prefix(tokens, length, context_width=CONTEXT_WIDTH):
    """Lengths include BOS/UNK; discard only verified right padding."""
    values = [int(value) for value in tokens]
    length = int(length)
    if not 1 <= length <= len(values) <= context_width:
        raise ValueError("Invalid observed context length/width")
    prefix = values[:length]
    if prefix[0] != 1 or any(value < 2 for value in prefix[1:]):
        raise ValueError("Expected BOS followed by real/UNK tokens")
    if any(value != 0 for value in values[length:]):
        raise ValueError("Observed context must be right padded with PAD=0")
    return prefix


def append_generated(prefix, token, context_width=CONTEXT_WIDTH):
    """Keep BOS conditioning and most recent context_width-1 ID/UNK tokens."""
    if context_width < 2 or not prefix or prefix[0] != 1 or int(token) < 3:
        raise ValueError("Invalid prefix or generated real-item token")
    return [1, *(list(prefix[1:]) + [int(token)])[-(context_width - 1):]]


def producer_prefix_batch(observed_peers, active_prefix, producer_size, producer_slot):
    """Original ordered peers; only the active slot gets the synthetic prefix."""
    if (type(producer_size) is not int or not 1 <= producer_size <= 16
            or type(producer_slot) is not int or not 0 <= producer_slot < producer_size
            or len(observed_peers) != producer_size):
        raise ValueError("Exact original producer B/slot required")
    prefixes = [list(row) for row in observed_peers]
    prefixes[producer_slot] = list(active_prefix)
    for prefix in prefixes:
        if not prefix or len(prefix) > CONTEXT_WIDTH:
            raise ValueError("Invalid producer prefix length")
        observed_prefix(prefix, len(prefix))
    lengths = [len(prefix) for prefix in prefixes]
    width = max(lengths)
    return [prefix + [0] * (width - len(prefix)) for prefix in prefixes], lengths


def distribution(logits, excluded=(), temperature=1.0):
    """Stable common support; return probabilities and finite log-probs."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    values = [float(value) for value in logits]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Finite nonempty logits required")
    excluded = set(excluded)
    if any(type(index) is not int or not 0 <= index < len(values) for index in excluded):
        raise ValueError("Invalid excluded candidate index")
    allowed = [i for i in range(len(values)) if i not in excluded]
    if not allowed:
        raise ValueError("Repeat mask exhausted the candidate pool")
    peak = max(values[i] for i in allowed)
    shifted = {i: (values[i] - peak) / temperature for i in allowed}
    if not all(math.isfinite(value) for value in shifted.values()):
        raise ValueError("Temperature causes nonfinite scaled logits")
    normalizer = math.log(math.fsum(math.exp(value) for value in shifted.values()))
    logp = [None] * len(values)
    probabilities = [0.0] * len(values)
    for i in allowed:
        logp[i] = shifted[i] - normalizer
        probabilities[i] = math.exp(logp[i])
    return probabilities, logp


def select_index(probabilities, selection, rng, raw_ids=None):
    if selection == "greedy":
        raw_ids = list(range(len(probabilities))) if raw_ids is None else raw_ids
        return min(range(len(probabilities)), key=lambda i: (-probabilities[i], raw_ids[i]))
    if selection != "categorical":
        raise ValueError("Selection must be greedy or categorical")
    point, cumulative = rng.random(), 0.0
    last = None
    for i, probability in enumerate(probabilities):
        if probability > 0:
            last = i
            cumulative += probability
            if point < cumulative:
                return i
    if last is None:
        raise ValueError("Empty categorical support")
    return last  # floating-point cumulative roundoff only


def policy_metrics(probabilities, logp, reference_logp, rewards, support, effective):
    mass = math.fsum(p for p, ok in zip(probabilities, support) if ok)
    supported_sum = math.fsum(p * reward for p, reward, ok in zip(probabilities, rewards, support) if ok)
    return {
        "kl_nats": math.fsum(p * (lp - lq) for p, lp, lq in zip(probabilities, logp, reference_logp) if p > 0),
        "entropy_nats": -math.fsum(p * lp for p, lp in zip(probabilities, logp) if p > 0),
        "supported_mass": mass,
        "unsupported_mass": math.fsum(p for p, ok in zip(probabilities, support) if not ok),
        "expected_supported_reward_conditional": supported_sum / mass if mass > 0 else None,
        "expected_neutral_proxy": math.fsum(p * reward for p, reward in zip(probabilities, effective)),
    }


def _versions(module):
    """Detect in-place writes without a second copy of the frozen GPT."""
    result = []
    for name, value in (*module.named_parameters(), *module.named_buffers()):
        try:
            version = value._version
        except RuntimeError:  # inference tensors have no version counter
            version = None
        result.append((name, id(value), version))
    return tuple(result)


def rollout_state(ctx, head, reference, state, rewards, support, *,
                  repeat_mask="exclude-generated", selection="greedy", seed=20261024,
                  temperature=1.0, steps=5, gamma=0.9):
    """Generate IDs, recomputing q/base/source12 on each common live prefix.

    ``state`` adds producer_x[B,64], producer_lengths[B],
    producer_user_features[B,U], producer_size and producer_slot to one cache
    row. ctx['score_live_head'] owns producer-B/slot padding and head64 scoring.
    The original peer encoder inputs and full-table retrieval shape are used
    at every step. All candidate raw features, source13,
    user features and RM arrays remain frozen at observed s0. No outcome is
    fabricated. Rewards are supported completion predictions, not labels.

    ``fixed_neutral_proxy_return`` discounts the fixed s0 effective reward:
    supported*(reward - initial_unmasked_reference_supported_mean). Unsupported
    actions contribute zero algebraically; their selected reward is reported
    null. Raw completion return is null if any selected action is unsupported.
    """
    if repeat_mask not in ("none", "exclude-generated"):
        raise ValueError("Unknown repeat mask")
    if selection not in ("greedy", "categorical") or type(seed) is not int:
        raise ValueError("Invalid selection/seed")
    if type(steps) is not int or not 1 <= steps <= 5:
        raise ValueError("Live-prefix diagnostic permits 1..5 steps")
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Invalid discount")
    distribution([0.0], temperature=temperature)
    torch, np = ctx["torch"], ctx["np"]
    model, table, device = ctx["model"], ctx["table"], ctx["device"]
    if model.context_length != CONTEXT_WIDTH:
        raise ValueError("Reviewed context64 model required")
    for module in (model, head, reference):
        if module.training:
            raise ValueError("All rollout modules must be in eval mode")
    for module in (model, reference):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError("Frozen GPT and reference parameters required")
    if table.requires_grad:
        raise ValueError("Frozen candidate table required")
    versions = [_versions(module) for module in (model, head, reference)]
    ids_np = np.asarray(state["candidate_ids"])
    raw_ids = [int(value) for value in state["candidate_raw_ids"]]
    if (ids_np.shape != (POOL_WIDTH,) or len(raw_ids) != POOL_WIDTH
            or not np.issubdtype(ids_np.dtype, np.integer)
            or len(set(ids_np.tolist())) != POOL_WIDTH
            or bool(np.any(ids_np < 3)) or bool(np.any(ids_np >= model.vocab_size))):
        raise ValueError("Fixed unique natural1024 real-token pool required")
    raw_np = np.asarray(state["raw_features"], dtype=np.float32)
    source_np = np.asarray(state["source_values"], dtype=np.float32)
    initial_q = np.asarray(state["encoded"], dtype=np.float32)
    initial_base = np.asarray(state["base_logits"], dtype=np.float32)
    if (raw_np.shape != (POOL_WIDTH, 1068) or source_np.shape != (POOL_WIDTH, 2)
            or initial_q.shape != (256,) or initial_base.shape != (POOL_WIDTH,)
            or not all(bool(np.isfinite(value).all()) for value in (raw_np, source_np, initial_q, initial_base))):
        raise ValueError("Invalid finite current-C cache row")
    if not np.array_equal(source_np[:, 0], initial_base):
        raise ValueError("Cached source12 must equal cached detached GPT logit")
    reward_values = [float(value) for value in rewards]
    support_values = list(support)
    if (len(reward_values) != POOL_WIDTH or len(support_values) != POOL_WIDTH
            or any(value not in (0, 1, False, True) for value in support_values)):
        raise ValueError("Aligned binary support/reward arrays required")
    support_values = [bool(value) for value in support_values]
    if any(not math.isfinite(value) or not 0 <= value <= 1
           for value, ok in zip(reward_values, support_values) if ok):
        raise ValueError("Supported completion predictions must lie in [0,1]")
    prefix = observed_prefix(state["x"], state["lengths"])
    original_seen = set(prefix[1:])
    user_values = np.asarray(state["user_features"], dtype=np.float32)
    if user_values.shape != (model.user_feature_dim,) or not bool(np.isfinite(user_values).all()):
        raise ValueError("Exact observed transformed user_features row required")
    producer_size, producer_slot = state['producer_size'], state['producer_slot']
    peer_x = np.asarray(state['producer_x'])
    peer_lengths = np.asarray(state['producer_lengths'])
    peer_user = np.asarray(state['producer_user_features'], dtype=np.float32)
    if (type(producer_size) is not int or not 1 <= producer_size <= 16
            or type(producer_slot) is not int or not 0 <= producer_slot < producer_size
            or peer_x.shape != (producer_size, CONTEXT_WIDTH)
            or peer_lengths.shape != (producer_size,)
            or not np.issubdtype(peer_x.dtype, np.integer)
            or not np.issubdtype(peer_lengths.dtype, np.integer)
            or peer_user.shape != (producer_size, model.user_feature_dim)
            or not bool(np.isfinite(peer_user).all())):
        raise ValueError("Invalid original producer encoder batch")
    observed_peers = [observed_prefix(row, length) for row, length in zip(peer_x, peer_lengths)]
    if (observed_peers[producer_slot] != prefix
            or not bool(np.array_equal(peer_user[producer_slot], user_values))
            or bool(np.any(peer_x >= model.vocab_size))):
        raise ValueError("Producer active slot differs from the selected observed state")
    if not callable(ctx.get('score_live_head')):
        raise ValueError("Producer-B/slot score_live_head callback required")
    if 'reference_scores' not in state:
        raise ValueError("Original cached C reference_scores required for initial parity")
    rng, generated, records = random.Random(seed), [], []
    selected_supported_rewards, proxy_return, completion_sum = [], 0.0, 0.0
    baseline = 0.0
    first_q, first_base = None, None
    check = ctx.get("check", lambda **fields: None)
    with torch.inference_mode():
        ids = torch.tensor(ids_np, device=device, dtype=torch.long).unsqueeze(0)
        selected_table = table.detach()[ids[0]]
        candidate_vectors = selected_table.float().unsqueeze(0)
        raw = torch.tensor(raw_np, device=device, dtype=torch.float32).unsqueeze(0)
        fixed_source = torch.tensor(source_np, device=device, dtype=torch.float32).unsqueeze(0)
        user = torch.tensor(peer_user, device=device, dtype=torch.float32)
        for step in range(steps):
            check(stage="liveprefix-step", step=step + 1, selection=selection, repeat_mask=repeat_mask)
            padded_peers, live_lengths = producer_prefix_batch(observed_peers, prefix, producer_size, producer_slot)
            x = torch.tensor(padded_peers, device=device, dtype=torch.long)
            lengths = torch.tensor(live_lengths, device=device, dtype=torch.long)
            with ctx["train"].legacy.autocast(device):
                q_group = model.encode_context(x, lengths, user, candidate_table=table)
                # Root-authorized canonical projection at EVERY step. Match
                # original producer M=B and N=vocab; do not regenerate the pool.
                retrieval_group = model.retrieval_logits(q_group, table)
                base = retrieval_group[producer_slot:producer_slot + 1].gather(1, ids).float()
            q = q_group[producer_slot:producer_slot + 1].float()
            source = fixed_source.clone()
            source[:, :, 0] = base
            def score(module):
                values = ctx['score_live_head'](module, q, candidate_vectors, base, raw, source,
                                                producer_size, producer_slot)
                if tuple(values.shape) != (1, POOL_WIDTH):
                    raise ValueError("Producer-shaped head callback must return selected slot [1,1024]")
                return values.float()[0].cpu().tolist()
            actor_logits = score(head)
            reference_logits = score(reference)
            initial_parity = None
            if step == 0:
                cached = np.asarray(state['reference_scores'], dtype=np.float32)
                fresh = np.asarray(reference_logits, dtype=np.float32)
                if cached.shape != (POOL_WIDTH,) or not bool(np.allclose(cached, fresh, atol=.003, rtol=.001)):
                    raise ValueError('Initial live C reference scores exceed registered cache tolerance')
                old_rank = sorted(range(POOL_WIDTH), key=lambda i: (-float(cached[i]),raw_ids[i]))[:10]
                new_rank = sorted(range(POOL_WIDTH), key=lambda i: (-float(fresh[i]),raw_ids[i]))[:10]
                if old_rank != new_rank:
                    for a,b in zip(old_rank,new_rank):
                        bound = .006 + .001*(abs(float(cached[a]))+abs(float(cached[b])))
                        if abs(float(cached[a]-cached[b])) > bound:
                            raise ValueError('Initial live C reference ranks differ above numerical tolerance')
                initial_parity = dict(max_abs_error=float(np.abs(cached-fresh).max()), top10_exact=old_rank==new_rank,
                    atol=.003,rtol=.001,baseline='fresh-live-C-on-identical-prefix')
            q_np, base_np = q[0].cpu().numpy(), base[0].cpu().numpy()
            if not bool(np.isfinite(q_np).all()) or not bool(np.isfinite(base_np).all()):
                raise FloatingPointError("Nonfinite live GPT outputs")
            excluded = set(generated) if repeat_mask == "exclude-generated" else set()
            probabilities, logp = distribution(actor_logits, excluded, temperature)
            ref_p, ref_logp = distribution(reference_logits, excluded, temperature)
            if step == 0:
                first_q, first_base = q_np.copy(), base_np.copy()
                mass = math.fsum(p for p, ok in zip(ref_p, support_values) if ok)
                baseline = (math.fsum(p * reward for p, reward, ok in zip(ref_p, reward_values, support_values) if ok)
                            / mass if mass else 0.0)
                effective = [reward - baseline if ok else 0.0 for reward, ok in zip(reward_values, support_values)]
            chosen = select_index(probabilities, selection, rng, raw_ids)
            token = int(ids_np[chosen])
            supported = support_values[chosen]
            reward = reward_values[chosen] if supported else None
            record = policy_metrics(probabilities, logp, ref_logp, reward_values, support_values, effective)
            record.update(step=step + 1, candidate_index=chosen, generated_token_id=token,
                generated_raw_id=raw_ids[chosen], selected_reward=reward, selected_supported=supported,
                selected_probability=probabilities[chosen], generated_repeat=chosen in generated,
                original_seen=token in original_seen, prefix_length=len(prefix),
                producer_size=producer_size, producer_slot=producer_slot,
                encoder_trim_width=len(padded_peers[0]),
                retrieval_projection='original-full-vocab-then-fixed-pool-gather',
                q_l2_drift=float(np.linalg.norm(q_np - first_q)),
                base_rmse_drift=float(np.sqrt(np.mean((base_np - first_base) ** 2))),
                cached_q_maxabs=float(np.max(np.abs(q_np - initial_q))),
                cached_base_maxabs=float(np.max(np.abs(base_np - initial_base))),
                initial_C_cache_parity=initial_parity,
                selected_base_logit=float(base_np[chosen]), selected_neutral_proxy=effective[chosen])
            records.append(record)
            generated.append(chosen)
            selected_supported_rewards.append(supported)
            proxy_return += gamma ** step * effective[chosen]
            if supported:
                completion_sum += gamma ** step * reward
            prefix = append_generated(prefix, token)
    if versions != [_versions(module) for module in (model, head, reference)]:
        raise RuntimeError("Rollout mutated model/head/reference tensors")
    return dict(contract=CONTRACT, scope="synthetic-id-slate-fixed-observed-rm",
        selection=selection, seed=seed, temperature=temperature, repeat_mask=repeat_mask,
        gamma=gamma, steps=steps, per_step=records,
        generated_token_ids=[int(ids_np[i]) for i in generated],
        generated_raw_ids=[raw_ids[i] for i in generated],
        unique_generated=len(set(generated)), generated_repeat_count=len(generated) - len(set(generated)),
        initial_reference_supported_baseline=baseline, fixed_neutral_proxy_return=proxy_return,
        fixed_completion_proxy_return=completion_sum if all(selected_supported_rewards) else None,
        supported_completion_partial_sum=completion_sum,
        unsupported_selected=sum(not value for value in selected_supported_rewards),
        rm_state="observed-s0-fixed", raw1068="observed-s0-fixed", source13="observed-s0-fixed",
        source12="live-prefix-gpt-logit", context_policy="BOS-plus-latest63-real-or-UNK-tokens",
        producer_size=producer_size, producer_slot=producer_slot,
        encoder_peers='original-observed-s0-except-active-synthetic-prefix',
        retrieval_projection='original-full-vocab-then-fixed-pool-gather',
        frozen_tensor_versions_unchanged=True, final_prefix=prefix)
