"""Bounded REINFORCE bridge over frozen KION SFT and CPU reward scorers.

This is constrained slate generation, not an observed user environment. RM sees
the original state s0 at every step. Generated IDs condition the policy only.
No test split is loaded here. Real bundles must be admitted by the coordinator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, FIRST_ITEM = 0, 1, 3
FORMAT = "kion-policy-rl-v1"


def stable_hash(value: Any) -> str:
    """Hash complete typed contexts, including array content, not user ID alone."""
    digest = hashlib.sha256()

    def add(item):
        if isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("Object arrays are not stable reward context inputs")
            digest.update(b"array" + item.dtype.str.encode() + repr(item.shape).encode())
            if item.size:
                digest.update(memoryview(np.ascontiguousarray(item)).cast("B"))
        elif isinstance(item, torch.Tensor):
            add(item.detach().cpu().numpy())
        elif isinstance(item, Mapping):
            digest.update(b"map{")
            for key in sorted(item):
                if not isinstance(key, str):
                    raise ValueError("Context dictionary keys must be strings")
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence[")
            for child in item:
                add(child)
            digest.update(b"]")
        elif isinstance(item, (str, int, float, bool, type(None), np.generic)):
            scalar = item.item() if isinstance(item, np.generic) else item
            digest.update(json.dumps(scalar, ensure_ascii=False, allow_nan=False).encode("utf-8") + b";")
        else:
            raise TypeError(f"Unsupported context value {type(item).__name__}")
    add(value)
    return digest.hexdigest()


def model_hash(model: nn.Module) -> str:
    return stable_hash(dict(model.state_dict()))


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


@dataclass
class RLData:
    x: np.ndarray
    lengths: np.ndarray
    seen_tokens: Sequence[Sequence[int]]
    token_to_item: Mapping[int, int]
    reward_contexts: Sequence[Mapping[str, Any]]
    y: np.ndarray | None = None
    user_features: np.ndarray | None = None
    provenance: Mapping[str, Any] | None = None
    candidate_features_provider: Callable | None = None
    candidate_features_signature: str | None = None

    def validate(self, vocab: int, context_length: int, rollout_steps: int) -> None:
        n = len(self.x)
        if (self.x.dtype != np.int64 or self.x.shape != (n, context_length) or n < 1
                or self.lengths.dtype != np.int64 or self.lengths.shape != (n,)
                or np.any(self.lengths < 1) or np.any(self.lengths > context_length)):
            raise ValueError("RL contexts require nonempty int64 right-padded x/lengths")
        active = np.arange(context_length)[None, :] < self.lengths[:, None]
        if (np.any(self.x < 0) or np.any(self.x >= vocab) or np.any(self.x[:, 0] != BOS)
                or np.any(self.x[:, 1:] == BOS) or np.any(self.x[~active] != PAD)
                or np.any(self.x[active] == PAD)):
            raise ValueError("Invalid BOS, padding, or context token")
        if len(self.seen_tokens) != n or len(self.reward_contexts) != n:
            raise ValueError("Full observed history and reward contexts must align with x")
        if set(self.token_to_item) != set(range(FIRST_ITEM, vocab)):
            raise ValueError("Every real policy token needs one raw item ID")
        raw = list(self.token_to_item.values())
        if any(type(item) is not int or item < 0 for item in raw) or len(set(raw)) != len(raw):
            raise ValueError("Raw item mapping must be unique nonnegative integers")
        forbidden = {"y", "reward", "watched_pct", "total_dur", "target_item_id", "target_date"}
        for row, seen in enumerate(self.seen_tokens):
            if any(not isinstance(item, (int, np.integer)) or item < FIRST_ITEM or item >= vocab for item in seen):
                raise ValueError("Full seen history contains an invalid real token")
            seen_set = set(seen)
            visible = set(self.x[row, :self.lengths[row]]) - {PAD, BOS, 2}
            if not visible <= seen_set:
                raise ValueError("Full seen history omits a visible policy item")
            if vocab - FIRST_ITEM - len(seen_set) < rollout_steps:
                raise ValueError("Insufficient unseen candidates for a unique rollout")
            if forbidden.intersection(self.reward_contexts[row]):
                raise ValueError("Outcome/target fields cannot enter reward contexts")
            stable_hash(self.reward_contexts[row])
        if self.user_features is not None and (len(self.user_features) != n
                or self.user_features.dtype not in (np.float16, np.float32) or not np.isfinite(self.user_features).all()):
            raise ValueError("Policy features must be aligned finite float16/float32")
        if self.candidate_features_provider is not None and (not callable(self.candidate_features_provider)
                or not isinstance(self.candidate_features_signature, str) or not self.candidate_features_signature):
            raise ValueError("Dynamic candidate features require a pinned provider signature")
        if self.y is not None and (self.y.dtype != np.int64 or self.y.shape != (n,)
                or np.any(self.y < FIRST_ITEM) or np.any(self.y >= vocab)):
            raise ValueError("CE anchor targets must be known catalog tokens")

    def signature(self) -> str:
        return stable_hash({"x": self.x, "lengths": self.lengths,
                            "seen": self.seen_tokens, "mapping": {str(k): v for k, v in self.token_to_item.items()},
                            "reward_contexts": self.reward_contexts, "y": self.y,
                            "user_features": self.user_features, "provenance": self.provenance,
                            "candidate_features_signature": self.candidate_features_signature})

    def reward_inputs(self, indices, tokens):
        contexts = [dict(self.reward_contexts[row]) for row in indices]
        raw_ids = [self.token_to_item[int(token)] for token in tokens]
        if self.candidate_features_provider is not None:
            # Provider sees original row origin and candidate only, never prefix.
            features = self.candidate_features_provider(indices.copy(), tokens.copy())
            if (not isinstance(features, np.ndarray) or features.dtype != np.float32 or features.ndim != 2
                    or len(features) != len(indices) or np.isinf(features).any()):
                raise ValueError("Dynamic candidate features require float32 [batch,features] without infinity")
            for row, context in enumerate(contexts):
                context.update(candidate_numeric=features[row], candidate_item_id=raw_ids[row])
        return contexts, raw_ids


class PolicyAdapter(nn.Module):
    """Use the existing model; an optional forward_fn handles a newer facade.

    forward_fn(model, x, lengths, user_features) must return [batch,vocab]
    logits. It must keep all precomputed s0 history/profile features unchanged.
    """
    def __init__(self, model: nn.Module, *, vocab_size=None, context_length=None, forward_fn=None):
        super().__init__()
        self.model = model
        config = getattr(getattr(model, "backbone", None), "config", None)
        self.vocab_size = vocab_size if vocab_size is not None else config.vocab_size
        self.context_length = context_length if context_length is not None else config.n_positions
        self.forward_fn = forward_fn

    def forward(self, data: RLData, indices: np.ndarray, prefix: np.ndarray | None = None):
        device = next(self.model.parameters()).device
        x, lengths = policy_context(data, indices, self.context_length, prefix)
        x = torch.as_tensor(x, device=device)
        lengths = torch.as_tensor(lengths, device=device)
        user = (torch.as_tensor(data.user_features[indices], device=device, dtype=torch.float32)
                if data.user_features is not None else None)
        if self.forward_fn is not None:
            logits = self.forward_fn(self.model, x, lengths, user)
        else:
            mode = getattr(self.model, "feature_mode", "none")
            if mode == "basic-bundle" or mode == "local-pack":
                logits = self.model(x, lengths, user_features=user)
            elif mode == "sex":
                logits = self.model(x, lengths, sex=user)
            elif mode in ("age", "age-knownness"):
                logits = self.model(x, lengths, age=user)
            else:
                logits = self.model(x, lengths)
        if logits.shape != (len(indices), self.vocab_size):
            raise ValueError("Policy adapter must return full [batch,vocab] logits")
        return logits.float()


def policy_context(data, indices, context_length, prefix=None):
    """Append only to a temporary policy sequence; keep BOS when trimming."""
    x = np.zeros((len(indices), context_length), dtype=np.int64)
    lengths = np.empty(len(indices), dtype=np.int64)
    if prefix is not None and (prefix.dtype != np.int64 or prefix.ndim != 2 or len(prefix) != len(indices)):
        raise ValueError("Generated prefix must be an aligned int64 matrix")
    for local, row in enumerate(indices):
        history = data.x[row, 1:data.lengths[row]].tolist()
        generated = [] if prefix is None else prefix[local].tolist()
        sequence = [BOS] + (history + generated)[-(context_length - 1):] if context_length > 1 else [BOS]
        lengths[local] = len(sequence)
        x[local, :len(sequence)] = sequence
    return x, lengths


def candidate_mask(data, indices, vocab, prefix=None):
    result = np.ones((len(indices), vocab), dtype=bool)
    result[:, :FIRST_ITEM] = False
    for local, row in enumerate(indices):
        result[local, list(data.seen_tokens[row])] = False
        if prefix is not None:
            result[local, prefix[local]] = False
    if not result.any(1).all():
        raise ValueError("Unseen candidate support exhausted")
    return result


def masked_log_probs(logits, mask):
    mask = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
    if not torch.isfinite(logits[mask]).all():
        raise FloatingPointError("Non-finite candidate policy logits")
    return F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)


def exact_kl(log_probs, reference_log_probs, mask):
    """Forward KL(policy || frozen SFT), exact over the same unseen support."""
    mask = torch.as_tensor(mask, device=log_probs.device, dtype=torch.bool)
    delta = torch.where(mask, log_probs - reference_log_probs, 0.0)
    return (log_probs.exp() * delta).sum(-1)


class RewardCache:
    """Bounded CPU cache binding every context value and the entire scorer ID."""
    def __init__(self, scorer, max_entries=100_000, chunk_size=2048):
        if not isinstance(getattr(scorer, "signature", None), str) or not scorer.signature:
            raise ValueError("Frozen reward scorer requires a nonempty signature")
        if max_entries < 0 or chunk_size < 1:
            raise ValueError("Invalid reward cache bounds")
        self.scorer, self.signature = scorer, scorer.signature
        self.max_entries, self.chunk_size = max_entries, chunk_size
        self.values = {}
        self.calls = self.hits = self.requests = 0
        self.support_evaluated = self.unsupported = self.fallback = 0
        self.uncertainty_evaluated = 0
        self.uncertainty_sum = 0.0

    def diagnostic_state(self):
        return {name: getattr(self, name) for name in ("calls", "hits", "requests", "support_evaluated", "unsupported",
                                                      "fallback", "uncertainty_evaluated", "uncertainty_sum")}

    def restore_diagnostics(self, state):
        if set(state) != set(self.diagnostic_state()) or any(not math.isfinite(value) or value < 0 for value in state.values()):
            raise ValueError("Invalid saved reward diagnostics")
        for name, value in state.items():
            setattr(self, name, value)

    def score(self, contexts, candidates):
        if self.scorer.signature != self.signature:
            raise ValueError("Reward scorer signature changed during a run")
        keys = [(self.signature, stable_hash(context), int(candidate)) for context, candidate in zip(contexts, candidates)]
        if len(keys) != len(contexts) or len(keys) != len(candidates):
            raise ValueError("Reward contexts/candidates must align")
        missing = {}
        for row, key in enumerate(keys):
            if key not in self.values:
                missing.setdefault(key, row)
        pending = list(missing.items())
        fresh = {}
        for start in range(0, len(pending), self.chunk_size):
            chunk = pending[start:start + self.chunk_size]
            rows = [row for _, row in chunk]
            selected_contexts = [contexts[row] for row in rows]
            selected_candidates = np.asarray([candidates[row] for row in rows], dtype=np.int64)
            diagnose = getattr(self.scorer, "score_with_diagnostics", None)
            diagnostics = diagnose(selected_contexts, selected_candidates) if callable(diagnose) else None
            rewards = (diagnostics["rewards"] if diagnostics is not None
                       else self.scorer.score_batch(selected_contexts, selected_candidates))
            if isinstance(rewards, torch.Tensor):
                if rewards.device.type != "cpu" or rewards.requires_grad:
                    raise ValueError("Reward scorer must return detached CPU predictions")
                rewards = rewards.numpy()
            rewards = np.asarray(rewards, dtype=np.float64)
            if rewards.shape != (len(rows),) or not np.isfinite(rewards).all() or np.any((rewards < 0) | (rewards > 1)):
                raise ValueError("Reward scorer must apply its frozen clipping/fallback contract into [0,1]")
            support, fallback, uncertainty = None, None, None
            if diagnostics is not None:
                support = np.asarray(diagnostics["support"])
                fallback = np.asarray(diagnostics["fallback"])
                if support.dtype != bool or fallback.dtype != bool or support.shape != rewards.shape or fallback.shape != rewards.shape:
                    raise ValueError("Reward support/fallback diagnostics must be aligned boolean vectors")
                if diagnostics.get("uncertainty") is not None:
                    uncertainty = np.asarray(diagnostics["uncertainty"], dtype=np.float64)
                    if uncertainty.shape != rewards.shape or not np.isfinite(uncertainty).all() or np.any(uncertainty < 0):
                        raise ValueError("Reward uncertainty must be finite, nonnegative, and aligned")
            for local, ((key, _), reward) in enumerate(zip(chunk, rewards)):
                fresh[key] = (float(reward), bool(support[local]) if support is not None else None,
                              bool(fallback[local]) if fallback is not None else None,
                              float(uncertainty[local]) if uncertainty is not None else None)
            self.calls += 1
        records = [self.values[key] if key in self.values else fresh[key] for key in keys]
        result = np.asarray([record[0] for record in records])
        for _, support, fallback, uncertainty in records:
            self.support_evaluated += int(support is not None)
            self.unsupported += int(support is False)
            self.fallback += int(fallback is True)
            self.uncertainty_evaluated += int(uncertainty is not None)
            self.uncertainty_sum += uncertainty if uncertainty is not None else 0.0
        self.hits += len(keys) - len(missing)
        self.requests += len(keys)
        for key, reward in fresh.items():
            if self.max_entries:
                if len(self.values) >= self.max_entries:
                    self.values.pop(next(iter(self.values)))
                self.values[key] = reward
        return result


class ProcessRewardScorer:
    """One owned CPU subprocess for CatBoost when Torch uses another runtime.

    There is no network listener. The server must implement the versioned JSONL
    protocol and exit on EOF. Requests are serial, bounded, and timeout kills
    this subprocess; a backend error is never silently converted into reward.
    """
    protocol = "kion-reward-stdio-v1"

    def __init__(self, command: Sequence[str], expected_signature: str, *, cwd=None,
                 timeout=30.0, max_request_bytes=16_777_216, max_response_bytes=4_194_304):
        if (not command or any(not isinstance(arg, str) for arg in command) or not expected_signature
                or not math.isfinite(timeout) or timeout <= 0 or min(max_request_bytes, max_response_bytes) < 128):
            raise ValueError("Provide bounded native scorer command and selected signature")
        self.signature = expected_signature
        self.timeout, self.max_request_bytes, self.max_response_bytes = timeout, max_request_bytes, max_response_bytes
        self.request_id = 0
        self.lock = threading.Lock()
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        self.process = None
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1")
        try:
            self.process = subprocess.Popen(list(command), cwd=cwd, env=environment, shell=False,
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            hello = self._exchange(None)
            if hello.get("protocol") != self.protocol or hello.get("signature") != expected_signature:
                raise ValueError("Reward worker protocol/selected signature mismatch")
            self.metadata = hello
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _encode(payload):
        def encode(value):
            if isinstance(value, np.ndarray):
                return encode(value.tolist())
            if isinstance(value, np.generic):
                return encode(value.item())
            if isinstance(value, Mapping):
                return {key: encode(child) for key, child in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(child) for child in value]
            if isinstance(value, float) and not math.isfinite(value):
                if math.isnan(value):
                    # Server float-array coercion restores JSON null as NaN.
                    return None
                raise ValueError("Infinite numeric reward RPC input")
            if isinstance(value, (str, int, float, bool, type(None))):
                return value
            raise TypeError(f"Unsupported reward RPC value {type(value).__name__}")
        return (json.dumps(encode(payload), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

    def _exchange(self, payload):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("Reward worker is not running")
        encoded = None
        if payload is not None:
            encoded = self._encode(payload)
            if len(encoded) > self.max_request_bytes:
                raise ValueError("Reward request exceeds byte limit")
        completed = queue.Queue(maxsize=1)

        def communicate():
            try:
                if encoded is not None:
                    self.process.stdin.write(encoded)
                    self.process.stdin.flush()
                response = self.process.stdout.readline(self.max_response_bytes + 1)
                if not response or len(response) > self.max_response_bytes or not response.endswith(b"\n"):
                    raise RuntimeError("Reward worker returned empty, truncated, or oversized response")
                value = json.loads(response)
                if not isinstance(value, dict):
                    raise ValueError("Reward worker response must be an object")
                completed.put((value, None))
            except BaseException as error:
                completed.put((None, error))
        thread = threading.Thread(target=communicate, name="kion-reward-rpc", daemon=True)
        thread.start()
        try:
            value, error = completed.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            thread.join(timeout=1)
            raise TimeoutError("CPU reward worker request timed out") from None
        thread.join(timeout=1)
        if error is not None:
            self.close()
            raise RuntimeError(f"CPU reward worker protocol failed: {error}") from error
        return value

    def score_with_diagnostics(self, contexts, candidate_ids):
        if not self.lock.acquire(timeout=self.timeout):
            raise TimeoutError("CPU reward worker already has an outstanding request")
        try:
            try:
                contexts, candidate_ids = list(contexts), np.asarray(candidate_ids, dtype=np.int64)
                if len(contexts) != len(candidate_ids) or candidate_ids.ndim != 1:
                    raise ValueError("Reward RPC contexts/candidates must align")
                if not len(candidate_ids):
                    return {"rewards": np.empty(0, np.float64), "support": np.empty(0, bool),
                            "fallback": np.empty(0, bool), "uncertainty": None}
                pieces, pending = [], [(contexts, candidate_ids)]
                while pending:
                    chunk_contexts, chunk_candidates = pending.pop()
                    payload = {"protocol": self.protocol, "request_id": self.request_id + 1,
                               "contexts": chunk_contexts, "candidate_ids": chunk_candidates}
                    if len(self._encode(payload)) > self.max_request_bytes:
                        if len(chunk_candidates) <= 1:
                            raise ValueError("Single reward request exceeds byte limit")
                        middle = len(chunk_candidates) // 2
                        pending.extend(((chunk_contexts[middle:], chunk_candidates[middle:]),
                                        (chunk_contexts[:middle], chunk_candidates[:middle])))
                        continue
                    self.request_id += 1
                    response = self._exchange(payload)
                    if response.get("request_id") != self.request_id:
                        raise ValueError("Reward worker response ID mismatch")
                    if "error" in response:
                        raise RuntimeError(f"CPU reward backend error: {response['error']}")
                    result = {"rewards": np.asarray(response["rewards"], dtype=np.float64),
                              "support": np.asarray(response["support"]), "fallback": np.asarray(response["fallback"])}
                    if (result["support"].dtype != bool or result["fallback"].dtype != bool
                            or any(value.shape != (len(chunk_candidates),) for value in result.values())):
                        raise ValueError("Reward worker diagnostics have invalid shape/type")
                    result["uncertainty"] = (None if response.get("uncertainty") is None
                                             else np.asarray(response["uncertainty"], dtype=np.float64))
                    if "fit_target_support" in response:
                        result["fit_target_support"] = np.asarray(response["fit_target_support"], dtype=np.int64)
                    pieces.append(result)
                if any(set(piece) != set(pieces[0]) for piece in pieces):
                    raise ValueError("Reward worker changed diagnostic schema")
                combined = {}
                for name in pieces[0]:
                    if any(piece[name] is None for piece in pieces):
                        if any(piece[name] is not None for piece in pieces):
                            raise ValueError("Reward worker changed uncertainty availability")
                        combined[name] = None
                    else:
                        combined[name] = np.concatenate([piece[name] for piece in pieces])
                return combined
            except BaseException:
                self.close()
                raise
        finally:
            self.lock.release()

    def score_batch(self, contexts, candidate_ids):
        return self.score_with_diagnostics(contexts, candidate_ids)["rewards"]

    def close(self):
        process = self.process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        if not self.stderr.closed:
            self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def rollout(policy, reference, data, indices, steps, generator, reward_cache=None):
    actions = np.empty((len(indices), 0), dtype=np.int64)
    selected, divergences, rewards = [], [], []
    for _ in range(steps):
        mask = candidate_mask(data, indices, policy.vocab_size, actions)
        log_probs = masked_log_probs(policy(data, indices, actions), mask)
        with torch.no_grad():
            ref_probs = masked_log_probs(reference(data, indices, actions), mask)
        action = torch.multinomial(log_probs.detach().exp().cpu(), 1, generator=generator).squeeze(1)
        selected.append(log_probs.gather(1, action.to(log_probs.device)[:, None]).squeeze(1))
        divergences.append(exact_kl(log_probs, ref_probs, mask))
        token_ids = action.numpy().astype(np.int64)
        if reward_cache is not None:
            # The original s0 is deliberately reused. y is never passed here.
            contexts, raw_ids = data.reward_inputs(indices, token_ids)
            reward = reward_cache.score(contexts, raw_ids)
        else:
            reward = np.zeros(len(indices), dtype=np.float64)
        rewards.append(reward)
        actions = np.column_stack((actions, token_ids))
    return {"actions": actions, "selected_log_probs": torch.stack(selected, 1),
            "kl": torch.stack(divergences, 1), "rewards": np.column_stack(rewards)}


def reinforce_loss(selected_log_probs, rewards, baseline):
    # Future actions depend on the prefix, so use return-to-go for a slate.
    returns = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1).copy()
    advantage = torch.as_tensor(returns - baseline[None, :], dtype=selected_log_probs.dtype,
                                device=selected_log_probs.device).detach()
    return -(selected_log_probs * advantage).mean(), returns


def slate_kl_loss(selected_log_probs, conditional_kl):
    """Unbiased gradient of mean AR-slate forward KL, including prefix visits.

    Every conditional KL is integrated exactly over actions. Earlier sampled
    actions also change which future prefixes are visited; their score-function
    term must include those future KL costs, with a positive minimization sign.
    The returned scalar is a gradient surrogate, not the measured KL value.
    """
    future = torch.flip(torch.cumsum(torch.flip(conditional_kl.detach(), (1,)), 1), (1,)) - conditional_kl.detach()
    occupancy = (selected_log_probs * future).mean()
    return conditional_kl.mean() + occupancy, occupancy


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    batch_size: int = 32
    rollout_steps: int = 1
    lr: float = 1e-5
    kl_weight: float = 0.2
    ce_weight: float = 0.1
    baseline_decay: float = 0.9
    clip_grad: float = 1.0
    seed: int = 41
    mode: str = "rl"
    checkpoint_updates: int = 10
    cache_entries: int = 100_000
    reward_chunk_size: int = 2048

    def validate(self):
        for name in ("steps", "batch_size", "rollout_steps", "checkpoint_updates", "reward_chunk_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.rollout_steps > 5 or self.mode not in ("rl", "ce", "kl-ce"):
            raise ValueError("Use rl, ce, or kl-ce and at most five rollout steps")
        if (not all(math.isfinite(value) for value in (self.lr, self.kl_weight, self.ce_weight,
                self.baseline_decay, self.clip_grad)) or self.lr <= 0 or min(self.kl_weight, self.ce_weight) < 0
                or not 0 <= self.baseline_decay < 1 or self.clip_grad <= 0 or self.cache_entries < 0):
            raise ValueError("Invalid finite optimization or reward cache settings")
        if self.mode == "ce" and self.ce_weight <= 0 or self.mode == "kl-ce" and self.ce_weight + self.kl_weight <= 0:
            raise ValueError("A control requires a positive CE/KL objective")


def save_checkpoint(path, policy, optimizer, state, contract, rng, generator, *, rotate=True):
    temporary = path.with_name(path.name + ".tmp")
    payload = {"format": FORMAT, "contract": contract, "model": policy.state_dict(),
               "optimizer": optimizer.state_dict(), "state": state,
               "numpy_rng": rng.bit_generator.state, "sampling_rng": generator.get_state()}
    torch.save(payload, temporary)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    if loaded["contract"] != contract or loaded["state"] != state:
        raise ValueError("Checkpoint verification failed")
    if rotate and path.exists():
        # Commit a verified recoverable pair while latest is still intact. A
        # kill between replacing latest and its receipt leaves previous valid.
        load_checkpoint(path, contract)
        previous = path.with_name("previous.pt")
        for source, destination in ((path, previous), (path.with_suffix(".sha256.json"), previous.with_suffix(".sha256.json"))):
            copy = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(source, copy)
            os.replace(copy, destination)
        load_checkpoint(previous, contract)
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".sha256.json"), {"sha256": file_hash(path), "updates": state["updates"]})


def load_checkpoint(path, contract):
    receipt = json.loads(path.with_suffix(".sha256.json").read_text(encoding="utf-8"))
    if receipt["sha256"] != file_hash(path):
        raise ValueError("Checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != FORMAT or payload["contract"] != contract:
        raise ValueError("Checkpoint contract mismatch")
    if receipt["updates"] != payload["state"]["updates"]:
        raise ValueError("Checkpoint cursor mismatch")
    return payload


def train(policy, reference, data, scorer, output, config=TrainConfig(), *, resume=False,
          stop_file=None, max_updates=0, max_seconds=0, resume_from="latest"):
    """Train with a bounded budget; runtime caps do not alter resume contract."""
    config.validate()
    if max_updates < 0 or max_seconds < 0 or not math.isfinite(max_seconds):
        raise ValueError("Runtime bounds must be finite and nonnegative")
    if resume_from not in ("latest", "previous") or resume_from != "latest" and not resume:
        raise ValueError("Previous checkpoint recovery requires explicit resume")
    if any(parameter.requires_grad and parameter.dtype != torch.float32 for parameter in policy.parameters()):
        raise ValueError("RL optimizer requires FP32 trainable policy parameters; cast factory model.float()")
    if policy is reference or any(p.data_ptr() == q.data_ptr() for p in policy.parameters() for q in reference.parameters()):
        raise ValueError("Reference and policy must own independent parameters")
    if (policy.vocab_size, policy.context_length) != (reference.vocab_size, reference.context_length):
        raise ValueError("Policy/reference adapter layouts differ")
    data.validate(policy.vocab_size, policy.context_length, config.rollout_steps if config.mode != "ce" else 0)
    if config.ce_weight > 0 and data.y is None:
        raise ValueError("A positive CE anchor requires supervision")
    if config.mode == "rl" and scorer is None:
        raise ValueError("RL requires a selected frozen reward scorer")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise ValueError("A new RL run requires an empty output directory")
    policy.eval()  # Deterministic policy distributions; gradients remain enabled.
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    frozen_reference = model_hash(reference)
    contract = {"format": FORMAT, "training": asdict(config), "data_sha256": data.signature(),
                "reference_sha256": frozen_reference, "reward_signature": scorer.signature if config.mode == "rl" else None,
                "vocab_size": policy.vocab_size, "context_length": policy.context_length,
                "source_sha256": file_hash(Path(__file__)), "reward_state": "frozen_observed_s0",
                "candidate_mask": "full_observed_history_plus_generated_unique", "test_accessed": False}
    contract["initial_policy_sha256"] = (json.loads((output / "config.json").read_text(encoding="utf-8"))["initial_policy_sha256"]
                                         if resume else model_hash(policy))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=0)
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    state = {"updates": 0, "baseline": [0.0] * config.rollout_steps, "last_metrics": None}
    if resume:
        saved = load_checkpoint(output / (resume_from + ".pt"), contract)
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        rng.bit_generator.state = saved["numpy_rng"]
        generator.set_state(saved["sampling_rng"])
    else:
        atomic_json(output / "config.json", contract)
    cache = RewardCache(scorer, config.cache_entries, config.reward_chunk_size) if config.mode == "rl" else None
    if cache is not None and resume and "reward_diagnostics" in state:
        cache.restore_diagnostics(state["reward_diagnostics"])
    started = time.monotonic()
    reason = None
    recovery_publication = resume_from == "previous"

    def checkpoint():
        nonlocal recovery_publication
        save_checkpoint(output / "latest.pt", policy, optimizer, state, contract, rng, generator,
                        rotate=not recovery_publication)
        recovery_publication = False

    def emit(event, **values):
        payload = {"event": event, "updates": state["updates"], **values}
        atomic_json(output / "progress.json", payload)
        with (output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    emit("started", status="running", resumed=resume)
    while state["updates"] < config.steps:
        if stop_file is not None and Path(stop_file).exists():
            reason = "runtime_stop_file"
        elif max_updates and state["updates"] >= max_updates:
            reason = "runtime_max_updates"
        elif max_seconds and time.monotonic() - started >= max_seconds:
            reason = "runtime_max_seconds"
        if reason:
            break
        indices = rng.integers(0, len(data.x), size=config.batch_size, dtype=np.int64)
        optimizer.zero_grad(set_to_none=True)
        zero = next(policy.parameters()).sum() * 0
        pg_loss, kl_loss, ce_loss, returns = zero, zero, zero, None
        kl_objective, kl_occupancy = zero, zero
        trace = None
        if config.mode != "ce":
            trace = rollout(policy, reference, data, indices, config.rollout_steps, generator, cache)
            kl_loss = trace["kl"].mean()
            kl_objective, kl_occupancy = slate_kl_loss(trace["selected_log_probs"], trace["kl"])
            if config.mode == "rl":
                pg_loss, returns = reinforce_loss(trace["selected_log_probs"], trace["rewards"], np.asarray(state["baseline"]))
        if config.ce_weight > 0:
            logits = policy(data, indices)
            targets = torch.as_tensor(data.y[indices] - FIRST_ITEM, device=logits.device)
            ce_loss = F.cross_entropy(logits[:, FIRST_ITEM:], targets)
        loss = pg_loss + config.kl_weight * kl_objective + config.ce_weight * ce_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite RL objective; checkpoint cursor did not advance")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.clip_grad)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite RL gradients; checkpoint cursor did not advance")
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
            raise FloatingPointError("Optimizer produced non-finite policy weights; checkpoint cursor did not advance")
        if returns is not None:
            baseline = config.baseline_decay * np.asarray(state["baseline"]) + (1 - config.baseline_decay) * returns.mean(0)
            state["baseline"] = baseline.tolist()
        state["updates"] += 1
        if cache is not None:
            state["reward_diagnostics"] = cache.diagnostic_state()
        metrics = {"loss": float(loss.detach()), "policy_loss": float(pg_loss.detach()),
                   "exact_kl": float(kl_loss.detach()), "ce_loss": float(ce_loss.detach()),
                   "kl_prefix_occupancy_surrogate": float(kl_occupancy.detach()),
                   "grad_norm": float(norm), "mean_proxy_reward": float(trace["rewards"].mean()) if cache else None,
                   "reward_cache_requests": cache.requests if cache else 0,
                   "reward_cache_hits": cache.hits if cache else 0,
                   "reward_support_evaluated": cache.support_evaluated if cache else 0,
                   "reward_requests_unsupported": cache.unsupported if cache else 0,
                   "reward_requests_fallback": cache.fallback if cache else 0,
                   "reward_mean_uncertainty": (cache.uncertainty_sum / cache.uncertainty_evaluated
                                                if cache and cache.uncertainty_evaluated else None),
                   "interpretation": "frozen_RM_proxy_no_online_retention_evidence"}
        state["last_metrics"] = metrics
        if state["updates"] % config.checkpoint_updates == 0:
            checkpoint()
        emit("update", status="running", **metrics)
    if model_hash(reference) != frozen_reference or any(p.grad is not None for p in reference.parameters()):
        raise RuntimeError("Frozen SFT reference was modified")
    checkpoint()
    status = "completed" if state["updates"] >= config.steps else "stopped"
    result = {"status": status, "reason": reason, "updates": state["updates"], "mode": config.mode,
              "metrics": state["last_metrics"], "test_accessed": False, "reward_state": "frozen_observed_s0"}
    atomic_json(output / "metrics.json", result)
    artifacts = ["config.json", "latest.pt", "latest.sha256.json", "metrics.json"]
    receipt = {**result, "state": status, "artifacts": artifacts,
               "sha256": {name: file_hash(output / name) for name in artifacts}}
    atomic_json(output / "completion.json", receipt)
    emit("finished", **result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factory", required=True, help="module:function returning policy,reference,data,scorer; no downloads")
    parser.add_argument("--bundle", type=Path, required=True, help="Factory-specific pinned local artifact descriptor")
    parser.add_argument("--output", type=Path, default=os.environ.get("KION_OUTPUT_DIR"))
    parser.add_argument("--stop-file", type=Path, default=os.environ.get("KION_STOP_FILE"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", choices=("latest", "previous"), default="latest")
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--mode", choices=("rl", "ce", "kl-ce"), default="rl")
    for name, default in (("steps", 100), ("batch-size", 32), ("rollout-steps", 1), ("seed", 41), ("checkpoint-updates", 10)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("lr", 1e-5), ("kl-weight", 0.2), ("ce-weight", 0.1)):
        parser.add_argument("--" + name, type=float, default=default)
    args = parser.parse_args(argv)
    if args.output is None or args.cpu_threads < 1:
        raise ValueError("Provide output and positive CPU threads")
    torch.set_num_threads(args.cpu_threads)
    module, function = args.factory.split(":", 1)
    bundle = getattr(importlib.import_module(module), function)(args.bundle)
    if set(bundle) != {"policy", "reference", "data", "scorer"}:
        raise ValueError("Factory must return exactly policy,reference,data,scorer")
    config = TrainConfig(**{name: getattr(args, name) for name in
                           ("steps", "batch_size", "rollout_steps", "seed", "checkpoint_updates", "lr", "kl_weight", "ce_weight", "mode")})
    try:
        return train(**bundle, output=args.output, config=config, resume=args.resume, stop_file=args.stop_file,
                     max_updates=args.max_updates, max_seconds=args.max_seconds, resume_from=args.resume_from)
    finally:
        close = getattr(bundle["scorer"], "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
