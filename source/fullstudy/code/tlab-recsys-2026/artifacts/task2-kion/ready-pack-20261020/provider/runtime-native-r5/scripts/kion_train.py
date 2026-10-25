"""Small, from-scratch GPT2 -> reward model -> one-step REINFORCE experiment.

Input is the chronological KION NPZ contract produced by kion_prepare.py.
Only validation is inspected during development; this command fixes its config
before loading test and writes one final test evaluation. No checkpoint download.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, UNK, FIRST_ITEM = 0, 1, 2, 3


def progress(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False, allow_nan=False), flush=True)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_split(path, vocab_size, feature_dim, context_length=None):
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in ("x", "lengths", "y", "reward", "user_features")}
    x, lengths = data["x"], data["lengths"]
    n = len(x)
    if x.ndim != 2 or (context_length is not None and x.shape[1] != context_length):
        raise ValueError(f"{path}: inconsistent context shape")
    if x.dtype != np.int64 or lengths.dtype != np.int64 or data["y"].dtype != np.int64:
        raise ValueError(f"{path}: token ids and lengths must be int64")
    if lengths.shape != (n,) or data["y"].shape != (n,) or data["reward"].shape != (n,):
        raise ValueError(f"{path}: inconsistent row shapes")
    if data["user_features"].shape != (n, feature_dim):
        raise ValueError(f"{path}: inconsistent user features")
    if np.any(lengths < 1) or np.any(lengths > x.shape[1]):
        raise ValueError(f"{path}: invalid context lengths")
    active = np.arange(x.shape[1])[None, :] < lengths[:, None]
    if np.any(x[~active] != PAD) or np.any(x[active] == PAD):
        raise ValueError(f"{path}: contexts must be right padded")
    if np.any(x < 0) or np.any(x >= vocab_size):
        raise ValueError(f"{path}: context ids outside vocabulary")
    if not np.isfinite(data["user_features"]).all():
        raise ValueError(f"{path}: non-finite user features")
    reward = data["reward"]
    if np.isinf(reward).any() or np.any((reward[np.isfinite(reward)] < 0) | (reward[np.isfinite(reward)] > 1)):
        raise ValueError(f"{path}: observed rewards must be in [0, 1]")
    return data


def known_targets(y, vocab_size):
    return (y >= FIRST_ITEM) & (y < vocab_size)


def observed_reward_rows(data, vocab_size):
    return np.flatnonzero(known_targets(data["y"], vocab_size) & np.isfinite(data["reward"]))


def tensor_batch(data, indices, device):
    return {key: torch.as_tensor(value[indices], device=device) for key, value in data.items()}


def minibatches(indices, batch_size, shuffle=False):
    indices = np.array(indices, copy=True)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def last_real_hidden(hidden, lengths):
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths - 1]


class NextItemGPT(nn.Module):
    def __init__(self, vocab_size, context_length, hidden_size=128, layers=2, heads=4, tied=True):
        super().__init__()
        # Lazy import lets reward/metric tests run without transformers installed.
        from transformers import GPT2Config, GPT2Model
        config = GPT2Config(vocab_size=vocab_size, n_positions=context_length,
                            n_ctx=context_length, n_embd=hidden_size, n_layer=layers,
                            n_head=heads, bos_token_id=BOS, eos_token_id=None,
                            pad_token_id=PAD, use_cache=False)
        self.backbone = GPT2Model(config)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        if tied:
            self.output.weight = self.backbone.wte.weight

    def forward(self, x, lengths):
        hidden = self.backbone(input_ids=x, attention_mask=x.ne(PAD), use_cache=False).last_hidden_state
        # Deliberately project only [B,H], never [B,T,H] -> [B,T,V].
        return self.output(last_real_hidden(hidden, lengths))


class RewardModel(nn.Module):
    """Inputs contain history, demographics, candidate id and fixed item metadata.

    No target watched_pct, total_dur, or future-event field enters forward().
    Embeddings are independent of the GPT; no supervised logits are reused.
    """
    def __init__(self, vocab_size, user_dim, item_features, embedding_dim=32):
        super().__init__()
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.candidate_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.register_buffer("item_features", torch.as_tensor(item_features, dtype=torch.float32))
        self.mlp = nn.Sequential(nn.Linear(embedding_dim * 2 + user_dim + item_features.shape[1], 128),
                                 nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode_context(self, x):
        mask = x.ge(FIRST_ITEM).unsqueeze(-1)
        return (self.context_embedding(x) * mask).sum(1) / mask.sum(1).clamp_min(1)

    def score_encoded(self, context, user_features, candidate):
        values = torch.cat((context, self.candidate_embedding(candidate),
                            user_features.float(), self.item_features[candidate]), dim=-1)
        return self.mlp(values).squeeze(-1)

    def forward(self, x, user_features, candidate):
        return self.score_encoded(self.encode_context(x), user_features, candidate)


def freeze(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def catalog_log_probs(logits):
    """Special tokens never belong to the action space; finite KL by slicing."""
    return F.log_softmax(logits[:, FIRST_ITEM:].float(), dim=-1)


def exact_kl(policy_log_probs, reference_log_probs):
    return (policy_log_probs.exp() * (policy_log_probs - reference_log_probs)).sum(-1).mean()


def reinforce_objective(log_probs, actions, rewards, reference_log_probs, kl_weight):
    selected = log_probs.gather(1, actions[:, None]).squeeze(1)
    # Leave-one-out batch baseline is action-independent for each example.
    # A raw same-batch mean shrinks the gradient; the LOO correction avoids it.
    baseline = ((rewards.sum() - rewards) / (len(rewards) - 1)) if len(rewards) > 1 else torch.zeros_like(rewards)
    policy_loss = -(selected * (rewards - baseline).detach()).mean()
    kl = exact_kl(log_probs, reference_log_probs)
    return policy_loss + kl_weight * kl, policy_loss, kl


def ranking_metrics(top_items, targets, vocab_size, cutoffs=(10, 20)):
    """All rows count, including unknown/unseen targets, which are misses."""
    n = len(targets)
    result = {"examples": n, "known_targets": int(known_targets(targets, vocab_size).sum()),
              "unknown_target_misses": int((~known_targets(targets, vocab_size)).sum())}
    for k in cutoffs:
        hits = top_items[:, :k] == targets[:, None]
        result[f"recall@{k}"] = float(hits.any(1).mean()) if n else None
        discounts = 1 / np.log2(np.arange(hits.shape[1]) + 2)
        result[f"ndcg@{k}"] = float((hits * discounts).sum(1).mean()) if n else None
    return result


def regression_metrics(truth, predicted):
    if not len(truth):
        return {"examples": 0, "mae": None, "rmse": None, "r2": None}
    truth, predicted = np.asarray(truth, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    error = truth - predicted
    centered = ((truth - truth.mean()) ** 2).sum()
    return {"examples": len(truth), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "r2": float(1 - (error ** 2).sum() / centered) if centered > 0 else None}


def reward_baselines(train, vocab_size, shrinkage=20.0):
    rows = observed_reward_rows(train, vocab_size)
    if not len(rows):
        raise ValueError("No observed known-target train rewards for reward model")
    y, rewards = train["y"][rows], train["reward"][rows]
    mean = float(rewards.mean())
    count = np.bincount(y, minlength=vocab_size)
    total = np.bincount(y, weights=rewards, minlength=vocab_size)
    return mean, (total + shrinkage * mean) / (count + shrinkage)


@torch.no_grad()
def evaluate_reward(model, data, vocab_size, batch_size, device, global_mean, item_mean):
    rows = observed_reward_rows(data, vocab_size)
    predictions = []
    for indices in minibatches(rows, batch_size):
        batch = tensor_batch(data, indices, device)
        predictions.extend(model(batch["x"], batch["user_features"], batch["y"]).cpu().tolist())
    truth = data["reward"][rows]
    return {"model": regression_metrics(truth, predictions),
            "train_global_mean": regression_metrics(truth, np.full(len(rows), global_mean)),
            "train_item_mean_shrunk_20": regression_metrics(truth, item_mean[data["y"][rows]]),
            "excluded_missing_reward_or_unknown_target": len(data["y"]) - len(rows),
            "unknown_target_rows": int((~known_targets(data["y"], vocab_size)).sum()),
            "missing_reward_rows": int((~np.isfinite(data["reward"])).sum()),
            "observed_reward_rows": int(np.isfinite(data["reward"]).sum()),
            "known_target_coverage_among_observed_rewards": float(len(rows) / np.isfinite(data["reward"]).sum())
            if np.isfinite(data["reward"]).any() else None}


@torch.no_grad()
def evaluate_ranking(model, reward_model, data, vocab_size, batch_size, device, popularity=None):
    model.eval() if model is not None else None
    top_size = min(20, vocab_size - FIRST_ITEM)
    all_top, all_rewards, all_repeated = [], [], []
    for indices in minibatches(np.arange(len(data["y"])), batch_size):
        batch = tensor_batch(data, indices, device)
        if popularity is None:
            scores = model(batch["x"], batch["lengths"])[:, FIRST_ITEM:]
            top = torch.topk(scores, top_size, dim=-1).indices + FIRST_ITEM
        else:
            # Stable item-id tie break; no validation/test popularity is used.
            order = np.argsort(-popularity[FIRST_ITEM:], kind="stable")[:top_size] + FIRST_ITEM
            top = torch.as_tensor(order.copy(), device=device)[None, :].expand(len(indices), -1)
        context = reward_model.encode_context(batch["x"])
        rewards = reward_model.score_encoded(
            context[:, None, :].expand(-1, top_size, -1).reshape(-1, context.shape[-1]),
            batch["user_features"][:, None, :].expand(-1, top_size, -1).reshape(len(indices) * top_size, batch["user_features"].shape[-1]),
            top.reshape(-1)).reshape(len(indices), top_size)
        repeated = (top[:, :, None] == batch["x"][:, None, :]).any(-1)
        all_top.append(top.cpu().numpy())
        all_rewards.append(rewards.cpu().numpy())
        all_repeated.append(repeated.cpu().numpy())
    top = np.concatenate(all_top) if all_top else np.empty((0, top_size), dtype=np.int64)
    result = ranking_metrics(top, data["y"], vocab_size)
    reward_values = np.concatenate(all_rewards) if all_rewards else np.empty((0, top_size))
    repeated = np.concatenate(all_repeated) if all_repeated else np.empty((0, top_size))
    for k in (10, 20):
        actual_k = min(k, top_size)
        result[f"actual_k@{k}"] = actual_k
        result[f"predicted_reward_uniform_top{k}"] = float(reward_values[:, :k].mean()) if len(top) else None
        result[f"catalog_coverage@{k}"] = float(len(np.unique(top[:, :k])) / (vocab_size - FIRST_ITEM))
        result[f"context_repetition_fraction@{k}"] = float(repeated[:, :k].mean()) if len(top) else None
    result["top1_max_item_share"] = float(np.bincount(top[:, 0], minlength=vocab_size).max() / len(top)) if len(top) else None
    return result


def append_items(x, lengths, items):
    x, lengths = x.clone(), lengths.clone()
    full = lengths.eq(x.shape[1])
    x[full] = torch.cat((x[full, 1:], items[full, None]), 1)
    short = ~full
    x[short, lengths[short]] = items[short]
    lengths[short] += 1
    return x, lengths


@torch.no_grad()
def hypothetical_rollout(model, reward_model, data, batch_size, device, steps):
    model.eval()
    rewards, repetition = [[] for _ in range(steps)], [[] for _ in range(steps)]
    for indices in minibatches(np.arange(len(data["y"])), batch_size):
        batch = tensor_batch(data, indices, device)
        x, lengths = batch["x"], batch["lengths"]
        for step in range(steps):
            candidate = model(x, lengths)[:, FIRST_ITEM:].argmax(-1) + FIRST_ITEM
            rewards[step].extend(reward_model(x, batch["user_features"], candidate).cpu().tolist())
            repetition[step].extend((x == candidate[:, None]).any(1).cpu().tolist())
            x, lengths = append_items(x, lengths, candidate)
    return {"kind": "hypothetical greedy self-generated contexts; not observed interactions",
            "steps": [{"step": i + 1, "predicted_reward": float(np.mean(rewards[i])) if rewards[i] else None,
                       "context_repetition": float(np.mean(repetition[i])) if repetition[i] else None}
                      for i in range(steps)]}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fit_supervised(model, train, epochs, batch_size, device, lr, stage, vocab_size):
    rows = np.flatnonzero(known_targets(train["y"], vocab_size)) if stage == "gpt" else observed_reward_rows(train, vocab_size)
    if not len(rows):
        raise ValueError(f"No eligible train rows for {stage}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for epoch in range(epochs):
        model.train()
        synchronize(device)
        start = time.perf_counter()
        loss_sum, seen = 0.0, 0
        for step, indices in enumerate(minibatches(rows, batch_size, shuffle=True), 1):
            batch = tensor_batch(train, indices, device)
            optimizer.zero_grad(set_to_none=True)
            if stage == "gpt":
                logits = model(batch["x"], batch["lengths"])
                loss = F.cross_entropy(logits[:, FIRST_ITEM:], batch["y"] - FIRST_ITEM)
            else:
                predicted = model(batch["x"], batch["user_features"], batch["y"])
                loss = F.mse_loss(predicted, batch["reward"].float())
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {stage} loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            seen += len(indices)
            if step % 100 == 0:
                progress("train_progress", stage=stage, epoch=epoch + 1, step=step, loss=loss_sum / seen)
        synchronize(device)
        elapsed = time.perf_counter() - start
        entry = {"epoch": epoch + 1, "loss": loss_sum / seen, "examples": seen,
                 "wall_seconds_excluding_imports_data": elapsed, "examples_per_second": seen / elapsed,
                 "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
        history.append(entry)
        progress("epoch_complete", stage=stage, **entry)
    model.eval()
    return history


def fit_rl(policy, reference, reward_model, train, steps, batch_size, device, lr, kl_weight, ce_weight):
    freeze(reference)
    freeze(reward_model)
    # Deterministic dropout-free reference and policy. eval() still permits grads.
    policy.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)
    history = []
    start = time.perf_counter()
    if steps and not len(train["y"]):
        raise ValueError("No training contexts for RL")
    for step in range(1, steps + 1):
        indices = np.random.randint(len(train["y"]), size=min(batch_size, len(train["y"])))
        batch = tensor_batch(train, indices, device)
        optimizer.zero_grad(set_to_none=True)
        logits = policy(batch["x"], batch["lengths"])
        log_probs = catalog_log_probs(logits)
        with torch.no_grad():
            ref_log_probs = catalog_log_probs(reference(batch["x"], batch["lengths"]))
            actions = torch.distributions.Categorical(logits=log_probs.detach()).sample()
            rewards = reward_model(batch["x"], batch["user_features"], actions + FIRST_ITEM)
        loss, policy_loss, kl = reinforce_objective(log_probs, actions, rewards, ref_log_probs, kl_weight)
        valid = (batch["y"] >= FIRST_ITEM) & (batch["y"] < logits.shape[1])
        ce = F.cross_entropy(logits[valid, FIRST_ITEM:], batch["y"][valid] - FIRST_ITEM) if valid.any() else logits.sum() * 0
        loss = loss + ce_weight * ce
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RL loss")
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            entry = {"step": step, "loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                     "sampled_predicted_reward": float(rewards.mean()), "exact_kl": float(kl.detach()),
                     "supervised_ce": float(ce.detach())}
            history.append(entry)
            progress("rl_progress", **entry)
    synchronize(device)
    return {"steps": history, "wall_seconds": time.perf_counter() - start,
            "objective": "one-step on-policy REINFORCE + exact KL(pi||frozen supervised GPT) + auxiliary supervised CE",
            "kl_weight": kl_weight, "auxiliary_supervised_ce_weight": ce_weight}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/kion-smoke"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/kion-smoke"))
    parser.add_argument("--gpt-epochs", type=int, default=1)
    parser.add_argument("--rm-epochs", type=int, default=2)
    parser.add_argument("--rl-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rl-batch-size", type=int, default=128)
    parser.add_argument("--gpt-lr", type=float, default=3e-4)
    parser.add_argument("--rm-lr", type=float, default=1e-3)
    parser.add_argument("--rl-lr", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--ce-weight", type=float, default=0.1)
    parser.add_argument("--rollout-steps", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--untied-embeddings", action="store_true")
    args = parser.parse_args(argv)
    if args.gpt_epochs < 1 or args.rm_epochs < 1 or args.rl_steps < 0 or min(args.batch_size, args.rl_batch_size) < 1:
        parser.error("epochs/batch sizes must be positive and RL steps nonnegative")
    if args.rollout_steps < 0 or min(args.kl_weight, args.ce_weight) < 0 or min(args.gpt_lr, args.rm_lr, args.rl_lr) <= 0:
        parser.error("invalid loss weights, rollout steps, or learning rates")
    return args


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "config.json").exists():
        raise FileExistsError("Choose a fresh --output; completed/partial runs are never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    vocab_size, user_dim = int(metadata["vocab_size"]), int(metadata["user_feature_dim"])
    if vocab_size <= FIRST_ITEM:
        raise ValueError("Empty known-item catalog")
    train = load_split(args.data / "train.npz", vocab_size, user_dim)
    context_length = train["x"].shape[1]
    val = load_split(args.data / "val.npz", vocab_size, user_dim, context_length)
    item_features = np.load(args.data / "item_features.npy", allow_pickle=False)
    if item_features.ndim != 2 or item_features.shape[0] != vocab_size or not np.isfinite(item_features).all():
        raise ValueError("Invalid train-available item features")
    config = {key: value.as_posix() if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({"architecture": {"model": "GPT2Model from scratch", "layers": 2, "hidden_size": 128,
                                    "heads": 4, "context_length": context_length, "tied_embeddings": not args.untied_embeddings},
                   "vocab_size": vocab_size, "known_catalog_size": vocab_size - FIRST_ITEM,
                   "user_feature_dim": user_dim, "item_feature_dim": item_features.shape[1],
                   "selection": "fixed epochs/hyperparameters; no checkpoint or hyperparameter selection on test",
                   "reward_inputs": ["context_item_ids", "demographic_features", "candidate_id", "train_available_content_genres"],
                   "reward_target": "watched_pct / 100, observed known targets only",
                   "evaluation": "all supplied contexts; unknown targets count as ranking misses; full prepared known catalog; no seen-item mask",
                   "popularity": "train target count only; stable item-id tie break; capped by prepared train vocabulary",
                   "warning": "first smoke run tests the pipeline; it does not establish meaningful quality improvement"})
    # Freeze the full experiment contract before any test file is opened.
    write_json(args.output / "config.json", config)
    runtime = {"torch_version": torch.__version__, "cuda_version": torch.version.cuda,
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "seed": args.seed}
    progress("run_started", **runtime, train_examples=len(train["y"]), validation_examples=len(val["y"]),
             known_catalog_size=vocab_size - FIRST_ITEM)
    gpt = NextItemGPT(vocab_size, context_length, tied=not args.untied_embeddings).to(device)
    gpt_history = fit_supervised(gpt, train, args.gpt_epochs, args.batch_size, device, args.gpt_lr, "gpt", vocab_size)
    torch.save({"state_dict": gpt.state_dict(), "config": config}, args.output / "supervised_gpt.pt")
    rm = RewardModel(vocab_size, user_dim, item_features).to(device)
    rm_history = fit_supervised(rm, train, args.rm_epochs, args.batch_size, device, args.rm_lr, "reward_model", vocab_size)
    freeze(rm)
    torch.save({"state_dict": rm.state_dict(), "config": config}, args.output / "reward_model.pt")
    global_mean, item_mean = reward_baselines(train, vocab_size)
    popularity = np.bincount(train["y"][known_targets(train["y"], vocab_size)], minlength=vocab_size)
    val_before = {"reward_model": evaluate_reward(rm, val, vocab_size, args.batch_size, device, global_mean, item_mean),
                  "supervised_gpt": evaluate_ranking(gpt, rm, val, vocab_size, args.batch_size, device),
                  "popularity": evaluate_ranking(None, rm, val, vocab_size, args.batch_size, device, popularity)}
    progress("validation_before_rl", metrics=val_before)
    reference = freeze(copy.deepcopy(gpt))
    rl_history = fit_rl(gpt, reference, rm, train, args.rl_steps, args.rl_batch_size, device,
                        args.rl_lr, args.kl_weight, args.ce_weight)
    torch.save({"state_dict": gpt.state_dict(), "config": config}, args.output / "rl_gpt.pt")
    val_before["rl_gpt"] = evaluate_ranking(gpt, rm, val, vocab_size, args.batch_size, device)
    if args.rollout_steps:
        val_before["hypothetical_rollout_supervised"] = hypothetical_rollout(reference, rm, val, args.batch_size, device, args.rollout_steps)
        val_before["hypothetical_rollout_rl"] = hypothetical_rollout(gpt, rm, val, args.batch_size, device, args.rollout_steps)
    progress("validation_complete", metrics=val_before)
    # One locked, final test pass across the identical supplied test contexts.
    test = load_split(args.data / "test.npz", vocab_size, user_dim, context_length)
    test_result = {"reward_model": evaluate_reward(rm, test, vocab_size, args.batch_size, device, global_mean, item_mean),
                   "supervised_gpt": evaluate_ranking(reference, rm, test, vocab_size, args.batch_size, device),
                   "rl_gpt": evaluate_ranking(gpt, rm, test, vocab_size, args.batch_size, device),
                   "popularity": evaluate_ranking(None, rm, test, vocab_size, args.batch_size, device, popularity)}
    if args.rollout_steps:
        test_result["hypothetical_rollout_supervised"] = hypothetical_rollout(reference, rm, test, args.batch_size, device, args.rollout_steps)
        test_result["hypothetical_rollout_rl"] = hypothetical_rollout(gpt, rm, test, args.batch_size, device, args.rollout_steps)
    runtime["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    metrics = {"runtime": runtime, "gpt_training": gpt_history, "reward_model_training": rm_history,
               "rl_training": rl_history, "validation": val_before, "test": test_result,
               "reward_train_global_mean": global_mean, "config": config}
    write_json(args.output / "metrics.json", metrics)
    progress("run_complete", output=args.output.as_posix(), runtime=runtime, test=test_result)


if __name__ == "__main__":
    main()
