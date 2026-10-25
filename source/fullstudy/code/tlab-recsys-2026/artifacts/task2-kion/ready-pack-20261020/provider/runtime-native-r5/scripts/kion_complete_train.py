"""New complete-package SFT runner; train/validation only, no downloads or test.

The provider owns immutable feature preparation and prefix caches. This module
owns torch models, optimizer state, exact retrieval CE and bounded reranking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as recompute_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_full_train as legacy
import kion_conditioned_train as shared
from kion_complete_model import CompleteGPT, sample_training_candidates
from kion_train import FIRST_ITEM, known_targets

ROW_KEYS = shared.ROW_KEYS
sha256, write_json = legacy.sha256, legacy.write_json
checkpoint, load_checkpoint = shared.checkpoint, shared.load_checkpoint
READY_SCOPE_ID = "ready-pack-20261020-r1"
READY_SCOPE_SHA256 = "a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962"
PAIR_FEATURE_CACHE_BYTE_CAP = 64 << 20


def _tensor(value, device, dtype):
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def batch(provider, rows, split, device):
    values = provider.batch(np.asarray(rows, np.int64), split)
    if not isinstance(values, (tuple, list)) or len(values) != 4:
        raise ValueError("provider.batch must return x,lengths,y,user_features")
    # Provider outputs are CPU arrays/tensors. Trim right padding before transfer,
    # so short histories do not execute all context positions on the accelerator.
    # Lengths include BOS/UNK and are the only authority for the retained prefix.
    width = int(values[1].max())
    values = (values[0][:, :width], *values[1:])
    return tuple(_tensor(value, device, torch.float32 if i == 3 else torch.long) for i, value in enumerate(values))


def _immutable_pair_snapshot(features, shape):
    if isinstance(features, torch.Tensor):
        if features.device.type != "cpu":
            raise ValueError("Provider pair snapshots require CPU features")
        features = features.detach().numpy()
    values = np.asarray(features, dtype=np.float32)
    if values.shape != shape or values.nbytes > PAIR_FEATURE_CACHE_BYTE_CAP:
        raise ValueError("Pair snapshot shape or 64MiB bound differs")
    # A bytes owner prevents caller mutation or re-enabling NumPy writeability.
    return np.frombuffer(values.tobytes(order="C"), dtype=np.float32).reshape(shape)


def pair_scores(model, provider, rows, split, encoded, table, logits, ids, chunk_size):
    """Recompute training chunks so raw pair inputs/head activations are bounded.

    Provider callbacks must be deterministic reads of their immutable contract.
    Autograd retains shared table/context/logits references and small chunk IDs,
    not every chunk's features/MLP inputs. One training chunk may retain one
    immutable CPU float32 snapshot up to 64 MiB, reused on backward; GPU inputs
    and head activations are still rematerialized. Multiple chunks keep no raw
    snapshot and replay the provider. Evaluation calls it once per chunk.
    """
    rows = np.asarray(rows, np.int64)
    shape = (len(rows), ids.shape[1], model.candidate_feature_dim)
    cache_single = (torch.is_grad_enabled() and ids.shape[1] <= chunk_size
                    and math.prod(shape) * 4 <= PAIR_FEATURE_CACHE_BYTE_CAP)
    cached = None

    def score_chunk(context, candidates, retrieval, selected):
        nonlocal cached
        if cache_single:
            if cached is None:
                cached = _immutable_pair_snapshot(
                    provider.candidate_features(rows, selected.detach().cpu().numpy(), split), shape)
            # Copy out of immutable CPU storage on every forward/recomputation.
            values = torch.tensor(cached, device=context.device, dtype=torch.float32)
        else:
            features = provider.candidate_features(rows, selected.detach().cpu().numpy(), split)
            values = _tensor(features, context.device, torch.float32)
        return model.score_candidates(context, selected, values, candidate_table=candidates, retrieval_logits=retrieval)

    result = []
    for start in range(0, ids.shape[1], chunk_size):
        selected = ids[:, start:start + chunk_size]
        if torch.is_grad_enabled():
            result.append(recompute_checkpoint(score_chunk, encoded, table, logits, selected,
                                               use_reentrant=False, preserve_rng_state=True))
        else:
            result.append(score_chunk(encoded, table, logits, selected))
    return torch.cat(result, dim=1)


def update(model, provider, microbatches, optimizer, scaler, device, *, hard_negatives=31,
           uniform_negatives=32, auxiliary_weight=1., candidate_chunk_size=64, collect_loss=True):
    if not microbatches or sum(map(len, microbatches)) == 0:
        raise ValueError("Cannot update an empty effective batch")
    if candidate_chunk_size < 1 or not math.isfinite(auxiliary_weight) or auxiliary_weight < 0:
        raise ValueError("Invalid candidate objective controls")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    count = sum(map(len, microbatches))
    for attempt in range(9):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        terms = []
        for rows in microbatches:
            x, lengths, y, user = batch(provider, rows, "train", device)
            with legacy.autocast(device):
                table = model.candidate_table()
                encoded = model.encode_context(x, lengths, user, candidate_table=table)
                logits = model.retrieval_logits(encoded, table)
                retrieval = F.cross_entropy(logits[:, FIRST_ITEM:].float(), y - FIRST_ITEM, reduction="sum") / count
                auxiliary = retrieval.new_zeros(())
                if auxiliary_weight:
                    candidates = sample_training_candidates(logits, y, hard_negatives=hard_negatives,
                                                            uniform_negatives=uniform_negatives)
                    scores = pair_scores(model, provider, rows, "train", encoded, table, logits, candidates, candidate_chunk_size)
                    auxiliary = F.cross_entropy(scores.float(), torch.zeros(len(y), device=device, dtype=torch.long), reduction="sum") / count
                loss = retrieval + auxiliary_weight * auxiliary
            scaler.scale(loss).backward()
            terms.append(torch.stack((loss.detach(), retrieval.detach(), auxiliary.detach())))
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
        values = torch.stack(terms).sum(0)
        finite_loss, finite_grad = torch.stack((values.isfinite().all(), norm.isfinite())).tolist()
        if not finite_loss:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite complete training loss; cursor did not advance")
        if finite_grad:
            scaler.step(optimizer)
            scaler.update()
            names = ("loss", "retrieval_ce", "sampled_auxiliary_ce")
            return ({key: value for key, value in zip(names, values.tolist())} if collect_loss else None), attempt
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled() or attempt == 8:
            raise FloatingPointError("Nonfinite complete gradients persisted after AMP retry")
        scaler.update(new_scale=scaler.get_scale() / 2)


def _rank_metrics(top, targets, vocab):
    known = known_targets(targets, vocab)
    hits = (top == targets[:, None]) & known[:, None]
    discount = 1 / np.log2(np.arange(top.shape[1]) + 2)
    return {"recall@10": float(hits.any(1).mean()), "ndcg@10": float((hits * discount).sum(1).mean())}


class EvaluationStopped(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def _evaluation_stop(should_stop):
    reason = should_stop() if should_stop is not None else None
    if reason:
        raise EvaluationStopped(reason)


@torch.no_grad()
def evaluate(model, provider, batch_size, device, *, retrieval_topk=128, candidate_chunk_size=64,
             scoring_mode="rerank", indices=None, predictions_path=None, should_stop=None):
    if scoring_mode not in ("retrieval", "rerank") or min(retrieval_topk, candidate_chunk_size, batch_size) < 1:
        raise ValueError("Invalid scoring controls")
    model.eval()
    data = provider.splits["val"]
    indices = np.arange(len(data["y"])) if indices is None else np.asarray(indices, np.int64)
    if not len(indices):
        raise ValueError("Cannot evaluate empty validation")
    _evaluation_stop(should_stop)
    width = min(retrieval_topk, model.vocab_size - FIRST_ITEM)
    k = min(10, width)
    if width < min(10, model.vocab_size - FIRST_ITEM):
        raise ValueError("retrieval_topk must cover the evaluated top 10")
    retrieval_tops, rerank_tops, coverages = [], [], []
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    # In eval mode, a learned table is constant for this complete pass.
    with legacy.autocast(device):
        table = model.candidate_table()
    for start in range(0, len(indices), batch_size):
        _evaluation_stop(should_stop)
        rows = indices[start:start + batch_size]
        x, lengths, y, user = batch(provider, rows, "val", device)
        with legacy.autocast(device):
            encoded = model.encode_context(x, lengths, user, candidate_table=table)
            logits = model.retrieval_logits(encoded, table)
            catalog = logits[:, FIRST_ITEM:].float()
            candidates = catalog.topk(width, dim=1).indices + FIRST_ITEM
            scores = pair_scores(model, provider, rows, "val", encoded, table, logits, candidates, candidate_chunk_size)
        known = (y >= FIRST_ITEM) & (y < model.vocab_size)
        ce = F.cross_entropy(catalog[known], y[known] - FIRST_ITEM, reduction="sum")
        totals += torch.stack((ce, known.sum(), (~torch.isfinite(catalog)).sum() + (~torch.isfinite(scores)).sum())).double()
        retrieval_tops.append(candidates[:, :k].cpu().numpy())
        rerank_tops.append(candidates.gather(1, scores.float().topk(k, dim=1).indices).cpu().numpy())
        coverages.append((candidates.eq(y[:, None]).any(1) & known).cpu().numpy())
    ce, known_count, nonfinite = totals.tolist()
    if nonfinite:
        raise FloatingPointError("Nonfinite complete evaluation logits")
    retrieval_top, rerank_top = np.concatenate(retrieval_tops), np.concatenate(rerank_tops)
    targets = data["y"][indices]
    retrieval_metrics = _rank_metrics(retrieval_top, targets, model.vocab_size)
    rerank_metrics = _rank_metrics(rerank_top, targets, model.vocab_size)
    metrics = {"examples": len(indices), "known_targets": int(known_count),
        "unknown_target_misses": len(indices) - int(known_count), "actual_k": k,
        "cross_entropy_known": ce / known_count if known_count else None,
        "retrieval_topk": width, "retrieval_coverage_all_rows": float(np.concatenate(coverages).mean()),
        "retrieval_coverage_known_rows": float(np.concatenate(coverages).sum() / known_count) if known_count else None,
        "retrieval": retrieval_metrics, "rerank": rerank_metrics, "scoring_mode": scoring_mode,
        **(rerank_metrics if scoring_mode == "rerank" else retrieval_metrics)}
    if predictions_path is not None:
        _evaluation_stop(should_stop)
        top = rerank_top if scoring_mode == "rerank" else retrieval_top
        match = top == targets[:, None]
        values = {key: data[key][indices] for key in ROW_KEYS}
        values.update(row_index=indices, target_token=targets, known_target=known_targets(targets, model.vocab_size),
            top_items=top, retrieval_top_items=retrieval_top, rerank_top_items=rerank_top,
            retrieval_covered=np.concatenate(coverages), hit_at_10=match.any(1),
            ndcg_at_10=(match / np.log2(np.arange(k) + 2)).sum(1))
        predictions_path = Path(predictions_path)
        temporary = predictions_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, predictions_path)
    return metrics


def make_model(provider, args):
    columns = provider.feature_columns
    return CompleteGPT(int(provider.metadata["vocab_size"]), int(provider.metadata["context"]),
        args.hidden_size, args.layers, args.heads, item_features=provider.item_features,
        user_feature_dim=len(columns["user"]), candidate_feature_dim=len(columns["candidate"]),
        feature_columns=columns, text_features=provider.text_features, text_variant=args.text_variant,
        people_catalog=provider.people_catalog, people_variants=tuple(args.people_variants),
        bottleneck=args.bottleneck, person_dim=args.person_dim, seed=args.seed,
        source_contract=provider.contract())


def make_config(provider, model, args, device, *, schedule=None):
    import transformers
    source_names = ("kion_complete_model.py", "kion_complete_train.py", "kion_train.py", "kion_full_train.py",
                    "kion_conditioned_train.py", "kion_people_embeddings.py", "kion_pack_people.py")
    training_keys = ("schedule_total_epochs", "batch_size", "accumulation", "lr", "warmup_ratio", "weight_decay",
        "seed", "patience", "bucket_size", "hard_negatives", "uniform_negatives", "auxiliary_weight",
        "candidate_chunk_size", "retrieval_topk", "scoring_mode", "allow_partial_integration", "training_order", "integration_mode")
    return {"contract": {"architecture": model.contract(), "provider": provider.contract(),
        "integration": integration_contract(provider, args),
        "training_schedule": (schedule if schedule is not None else TrainingSchedule(provider, args)).contract(),
        "training": {key: getattr(args, key) for key in training_keys},
        "sources_sha256": {name: sha256(Path(__file__).with_name(name)) for name in source_names},
        "objective": {"retrieval": "exact CE over every known token; specials excluded",
            "auxiliary": "sampled candidate CE; forced positive + unique detached hard negatives + uniform remainder without replacement; no importance correction",
            "evaluation": "unforced exact retrieval topK then pair rerank; all unknown targets are misses; no seen mask",
            "selection": "strict greater validation NDCG@10 of declared scoring_mode; patience"},
        "prediction_row_keys": list(ROW_KEYS)},
        "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "device": str(device), "cpu_threads": args.cpu_threads,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "parameters": sum(parameter.numel() for parameter in model.parameters())}


def integration_contract(provider, args):
    """Fail closed outside full, user-pinned scoped production or tiny partial."""
    declaration = provider.contract()
    status = declaration.get("status")
    mode = getattr(args, "integration_mode", "full")
    decision_path = getattr(args, "scope_decision", None)
    if mode == "scoped_production":
        if args.allow_partial_integration:
            raise ValueError("Scoped production cannot use the tiny partial integration flag")
        if decision_path is None or sha256(Path(decision_path)) != READY_SCOPE_SHA256:
            raise ValueError("Scoped production requires the exact pinned user scope decision")
        decision = json.loads(Path(decision_path).read_text(encoding="utf-8"))
        pin = {"path": Path(decision_path).resolve().as_posix(), "sha256": READY_SCOPE_SHA256}
        active = decision["active_families"]
        acceptance = declaration.get("family_acceptance", {})
        if (decision.get("contract") != "kion-user-approved-ready-scope-v1"
                or decision.get("scope_id") != READY_SCOPE_ID or len(active) != 8 or len(set(active)) != 8
                or status != "admitted_scoped" or declaration.get("integration_mode") != "scoped_production"
                or declaration.get("scope_id") != READY_SCOPE_ID or declaration.get("scope_decision") != pin
                or declaration.get("active_families") != active
                or set(acceptance) != set(active) or any(value != "accepted_scoped" for value in acceptance.values())
                or set(declaration.get("materialized_families", [])) != set(active)
                or declaration.get("full_package_admitted") is not False
                or declaration.get("full_row_fit_admitted") is not True):
            raise ValueError("Scoped production rejects scope/source/active-family admission mismatch")
        return {"mode": "scoped_production", "status": "admitted_scoped", "scope_id": READY_SCOPE_ID,
                "scope_decision": pin, "active_families": active,
                "full_package_admitted": False, "full_row_fit_admitted": True}
    if mode != "full" or decision_path is not None or status == "admitted_scoped":
        raise ValueError("Scoped provider requires explicit --integration-mode scoped_production and --scope-decision")
    if status == "cannot_train_full":
        if not args.allow_partial_integration:
            raise ValueError("Partial provider requires --allow-partial-integration; cannot_train_full")
        if any(len(split["y"]) > 4096 for split in provider.splits.values()):
            raise ValueError("Partial integration is capped at 4096 rows per split; cannot_train_full")
        return {"mode": "partial-integration", "status": "cannot_train_full", "full_feature_fit": False}
    acceptance = declaration.get("family_acceptance", {})
    from kion_complete_pack import REQUIRED_FAMILIES
    if (status != "admitted_full" or not acceptance
            or set(acceptance) != set(REQUIRED_FAMILIES)
            or any(value != "accepted_full" for value in acceptance.values())
            or set(acceptance) - set(declaration.get("materialized_families", []))):
        raise ValueError("Full provider rejects missing/unreviewed/pilot family admission")
    return {"mode": "admitted-full-package", "status": "admitted_full", "full_package_admitted": True}


class TrainingSchedule:
    """Deterministic row/update plan with an explicit origin-aware cursor.

    Origin grouping changes SGD order and the number/weight of partial updates.
    It is a separate experiment contract, never a legacy performance alias.
    """
    MODES = ("legacy-bucketed", "origin-grouped-v1")

    def __init__(self, provider, args):
        self.mode, self.seed = args.training_order, args.seed
        if self.mode not in self.MODES:
            raise ValueError("Unknown complete training order")
        data = provider.splits["train"]
        self.lengths = data["lengths"]
        self.eligible = np.flatnonzero(known_targets(data["y"], int(provider.metadata["vocab_size"])))
        if not len(self.eligible):
            raise ValueError("No known training targets")
        self.microbatch, self.accumulation, self.bucket_size = args.batch_size, args.accumulation, args.bucket_size
        self.effective = self.microbatch * self.accumulation
        self.origins = None
        self.origin_rows = {}
        self.origin_counts = {}
        if self.mode == "origin-grouped-v1":
            if not callable(getattr(provider, "origin_keys", None)):
                raise ValueError("origin-grouped-v1 requires public provider.origin_keys(split); no fallback")
            values = np.asarray(provider.origin_keys("train"))
            if values.dtype != np.int64 or values.shape != data["y"].shape or np.any(values < 1):
                raise ValueError("origin_keys requires aligned positive int64 CPU day ordinals")
            self.origins = np.array(values, copy=True)
            self.origins.setflags(write=False)
            order = self.eligible[np.argsort(self.origins[self.eligible], kind="stable")]
            keys, starts, counts = np.unique(self.origins[order], return_index=True, return_counts=True)
            order.setflags(write=False)
            for key, start, count in zip(keys, starts, counts):
                self.origin_rows[int(key)] = order[start:start + count]
                self.origin_counts[int(key)] = int(count)
            self.updates_per_epoch = sum(math.ceil(count / self.effective) for count in self.origin_counts.values())
        else:
            self.updates_per_epoch = math.ceil(len(self.eligible) / self.effective)

    def contract(self):
        return {"contract": "kion-training-order-v1", "mode": self.mode, "seed": self.seed,
            "eligible_rows_sha256": shared.array_hash(self.eligible), "eligible_count": len(self.eligible),
            "origin_keys_sha256": shared.array_hash(self.origins) if self.origins is not None else None,
            "origin_counts": {str(key): value for key, value in self.origin_counts.items()},
            "microbatch": self.microbatch, "accumulation": self.accumulation, "bucket_size": self.bucket_size,
            "updates_per_epoch": self.updates_per_epoch,
            "origin_shuffle": "SeedSequence(seed,epoch,0xD47E); permutation of sorted origins" if self.origins is not None else None,
            "within_origin_shuffle": "legacy length buckets with SeedSequence(seed,epoch,origin,0xB47C) derived uint32 seed" if self.origins is not None else None,
            "tail_policy": "one partial effective update per origin; never mix origins" if self.origins is not None else "one partial effective update per epoch",
            "cursor": "epoch + consumed row offset + completed group_index within epoch"}

    def _origin_order(self, epoch):
        keys = np.array(sorted(self.origin_rows), dtype=np.int64)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0xD47E]))
        return rng.permutation(keys)

    def _completed_groups(self, epoch, offset):
        if self.mode == "legacy-bucketed":
            if offset != len(self.eligible) and offset % self.effective:
                raise ValueError("Checkpoint cursor is inside an optimizer update")
            return math.ceil(offset / self.effective)
        consumed, completed = 0, 0
        for origin in self._origin_order(epoch):
            count = self.origin_counts[int(origin)]
            if offset >= consumed + count:
                consumed += count
                completed += math.ceil(count / self.effective)
                continue
            local = offset - consumed
            if local < 0 or local % self.effective:
                raise ValueError("Checkpoint cursor is inside an origin optimizer update")
            return completed + local // self.effective
        if offset != consumed:
            raise ValueError("Checkpoint offset differs from origin row counts")
        return completed

    def validate_cursor(self, state, total_epochs):
        epoch, offset = state["epoch"], state["offset"]
        if not 0 <= epoch <= total_epochs or not 0 <= offset <= len(self.eligible):
            raise ValueError("Checkpoint cursor is outside the training order contract")
        if epoch == total_epochs and offset or len(state["history"]) != epoch:
            raise ValueError("Checkpoint epoch/validation cursor mismatch")
        completed = self._completed_groups(epoch, offset)
        if (state.get("group_index") != completed
                or state["updates"] != epoch * self.updates_per_epoch + completed
                or state["examples"] != epoch * len(self.eligible) + offset):
            raise ValueError("Checkpoint origin group/update/example cursor mismatch")

    def groups(self, epoch, offset=0):
        completed = self._completed_groups(epoch, offset)
        if self.mode == "legacy-bucketed":
            order = legacy.epoch_order(self.eligible, self.lengths, self.seed, epoch, self.microbatch, self.bucket_size)
            for before, microbatches in legacy.groups(order, self.microbatch, self.accumulation, offset):
                yield before, microbatches, completed, None
                completed += 1
            return
        consumed = 0
        for origin in self._origin_order(epoch):
            origin = int(origin)
            rows = self.origin_rows[origin]
            if offset >= consumed + len(rows):
                consumed += len(rows)
                continue
            local = max(0, offset - consumed)
            seed = int(np.random.SeedSequence([self.seed, epoch, origin, 0xB47C]).generate_state(1, dtype=np.uint32)[0])
            order = legacy.epoch_order(rows, self.lengths, seed, 0, self.microbatch, self.bucket_size)
            for before, microbatches in legacy.groups(order, self.microbatch, self.accumulation, local):
                yield consumed + before, microbatches, completed, origin
                completed += 1
            consumed += len(rows)


class EpochTiming:
    """Volatile wall-clock observation; never part of checkpoint training state."""
    def __init__(self, epoch, state):
        self.epoch = epoch + 1
        self.start_offset, self.start_updates = state["offset"], state["updates"]
        self.started = time.perf_counter()
        self.train_seconds = None
        self.validation_seconds = 0.
        self.checkpoint_seconds = 0.
        self.train_complete = self.validation_complete = False
        self.end_offset = None

    def finish_train(self, state, device, *, complete):
        legacy.synchronize(device)
        if self.train_seconds is None:
            self.train_seconds = max(0., time.perf_counter() - self.started - self.checkpoint_seconds)
        self.train_complete = complete
        self.end_offset = state["offset"]

    def checkpoint(self, callback, device):
        # Complete pending optimizer work before attributing publication time.
        legacy.synchronize(device)
        started = time.perf_counter()
        result = callback()
        legacy.synchronize(device)
        self.checkpoint_seconds += time.perf_counter() - started
        return result

    def validation(self, callback, device):
        legacy.synchronize(device)
        started = time.perf_counter()
        try:
            result = callback()
            self.validation_complete = True
            return result
        finally:
            legacy.synchronize(device)
            self.validation_seconds += time.perf_counter() - started

    def fields(self, state):
        train = self.train_seconds
        if train is None:
            train = max(0., time.perf_counter() - self.started - self.checkpoint_seconds)
        scope = ("resumed-segment" if self.start_offset else
                 "complete-epoch" if self.train_complete and self.validation_complete else "partial-segment")
        return {"timed_epoch": self.epoch, "epoch_timing_scope": scope,
            "epoch_segment_start_offset": self.start_offset,
            "epoch_segment_end_offset": self.end_offset if self.end_offset is not None else state["offset"],
            "epoch_segment_updates": state["updates"] - self.start_updates,
            "epoch_train_wall_seconds": train, "epoch_validation_wall_seconds": self.validation_seconds,
            "epoch_checkpoint_wall_seconds": self.checkpoint_seconds,
            "epoch_train_complete": self.train_complete, "epoch_validation_complete": self.validation_complete}


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(_state_equal(left[key], right[key]) for key in left)
    return left == right


def publish_selected(path, model, optimizer, scaler, state, config):
    if not path.exists():
        return checkpoint(path, model, optimizer, scaler, state, config)
    receipt = path.with_suffix(".json")
    saved = (load_checkpoint(path, config) if receipt.exists() else torch.load(path, map_location="cpu", weights_only=False))
    legacy.validate_resume(saved["config"], config)
    if not _state_equal(saved["state_dict"], model.state_dict()) or saved["state"] != state:
        raise ValueError("Existing selected complete checkpoint conflicts with replay")
    digest = sha256(path)
    if not receipt.exists():
        write_json(receipt, {"sha256": digest, "epoch": state["epoch"], "offset": state["offset"], "updates": state["updates"]})
    return digest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--provider", default="kion_complete_pack:load_training_provider")
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--feature-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    result.add_argument("--allow-partial-integration", action="store_true",
                        help="Explicit tiny incomplete-package integration only; cannot_train_full")
    result.add_argument("--integration-mode", choices=("full", "scoped_production"), default="full",
                        help="Full all-agreed package or exact user-pinned ready scope with full rows")
    result.add_argument("--scope-decision", type=Path,
                        help="Exact user-approved decision required only for scoped_production")
    result.add_argument("--training-order", choices=TrainingSchedule.MODES, default="origin-grouped-v1",
                        help="Explicit SGD order/cursor contract; origin grouping keeps one feature snapshot per update")
    result.add_argument("--text-variant", choices=("all", "none", "title", "title_orig", "description"), default="all")
    result.add_argument("--people-variants", nargs="*", choices=("actor-mean", "director-mean", "actor-top3-concat"),
                        default=["actor-mean", "director-mean", "actor-top3-concat"])
    result.add_argument("--scoring-mode", choices=("retrieval", "rerank"), default="rerank")
    for name, default in (("schedule-total-epochs", 5), ("batch-size", 64), ("accumulation", 2),
        ("hidden-size", 256), ("layers", 4), ("heads", 4), ("bottleneck", 32), ("person-dim", 32),
        ("seed", 2026), ("patience", 2), ("bucket-size", 8192), ("cpu-threads", 2),
        ("checkpoint-updates", 1000), ("log-updates", 100), ("max-updates", 0), ("stop-after-validation", 0),
        ("retrieval-topk", 128), ("candidate-chunk-size", 64), ("hard-negatives", 31), ("uniform-negatives", 32)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--weight-decay", type=float, default=.01)
    result.add_argument("--warmup-ratio", type=float, default=.05)
    result.add_argument("--auxiliary-weight", type=float, default=1.)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--resume-from", choices=("latest", "previous"), default="latest")
    result.add_argument("--stop-file", type=Path)
    result.add_argument("--deadline-utc")
    return result


def run(args, provider):
    session_started = time.perf_counter()
    integration = integration_contract(provider, args)
    positive = ("schedule_total_epochs", "batch_size", "accumulation", "hidden_size", "layers", "heads",
        "bottleneck", "person_dim", "patience", "cpu_threads", "checkpoint_updates", "log_updates",
        "retrieval_topk", "candidate_chunk_size")
    if any(getattr(args, key) < 1 for key in positive):
        raise ValueError("Complete schedule/dimensions must be positive")
    if (args.hidden_size % args.heads or args.schedule_total_epochs > 5
            or min(args.seed, args.bucket_size, args.max_updates, args.stop_after_validation, args.hard_negatives, args.uniform_negatives) < 0
            or args.hard_negatives + args.uniform_negatives < 1
            or not math.isfinite(args.lr) or args.lr <= 0
            or not all(math.isfinite(value) and value >= 0 for value in (args.weight_decay, args.auxiliary_weight))
            or not 0 <= args.warmup_ratio < 1):
        raise ValueError("Invalid complete training controls")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")) if args.deadline_utc else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError("Deadline requires timezone")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA explicitly requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a fresh complete run output or --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    schedule = TrainingSchedule(provider, args)
    model = make_model(provider, args).to(device)
    config = make_config(provider, model, args, device, schedule=schedule)
    if args.resume:
        old = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        legacy.validate_resume(old, config)
        config = old
    else:
        write_json(args.output / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    state = {"epoch": 0, "offset": 0, "updates": 0, "examples": 0, "best_ndcg": -1., "bad_epochs": 0,
        "history": [], "amp_overflow_retries": 0, "selected_epoch": None, "group_index": 0}
    if args.resume:
        # torch Optimizer.load_state_dict casts active state to each parameter's
        # device. Keep the whole checkpoint (especially immutable buffers) off GPU.
        saved = load_checkpoint(args.output / (args.resume_from + ".pt"), config, map_location="cpu")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        legacy.restore_rng(saved["rng"])
        del saved
    eligible = schedule.eligible
    schedule.validate_cursor(state, args.schedule_total_epochs)
    total_updates = schedule.updates_per_epoch * args.schedule_total_epochs
    started, initial_examples = time.perf_counter(), state["examples"]
    runner_initialization_seconds = started - session_started
    recovery = args.resume and args.resume_from == "previous"
    timing = None

    def emit(event, **extra):
        elapsed = time.perf_counter() - started
        payload = {"event": event, "updated_utc": datetime.now(timezone.utc).isoformat(), **state,
            "elapsed_seconds": elapsed, "examples_per_second": (state["examples"] - initial_examples) / max(elapsed, 1e-9),
            "total_session_elapsed": time.perf_counter() - session_started,
            "runner_initialization_wall_seconds": runner_initialization_seconds,
            **(timing.fields(state) if timing is not None else {}), **extra}
        write_json(args.output / "progress.json", payload)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    def stop_reason():
        if args.stop_file and args.stop_file.exists():
            return "runtime_stop_file"
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            return "runtime_deadline"
        if args.max_updates and state["updates"] >= args.max_updates:
            return "runtime_max_updates"
        if args.stop_after_validation and state["epoch"] >= args.stop_after_validation:
            return "runtime_stop_after_validation"
        return None

    def save_latest():
        nonlocal recovery
        schedule.validate_cursor(state, args.schedule_total_epochs)
        callback = lambda: checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config, rotate=not recovery)
        timing.checkpoint(callback, device) if timing is not None else callback()
        recovery = False

    stopped = None
    emit("started", status="running", train_examples=len(eligible), schedule_total_updates=total_updates)
    for epoch in range(state["epoch"], args.schedule_total_epochs):
        if state["bad_epochs"] >= args.patience:
            break
        timing = EpochTiming(epoch, state)
        stopped = stop_reason()
        if stopped:
            timing.finish_train(state, device, complete=state["offset"] == len(eligible))
            save_latest()
            break
        model.train()
        for offset, microbatches, group_index, origin in schedule.groups(epoch, state["offset"]):
            for group in optimizer.param_groups:
                group["lr"] = shared.learning_rate(state["updates"], total_updates, args.warmup_ratio, args.lr)
            logging = state["updates"] == 0 or (state["updates"] + 1) % args.log_updates == 0
            losses, retries = update(model, provider, microbatches, optimizer, scaler, device,
                hard_negatives=args.hard_negatives, uniform_negatives=args.uniform_negatives,
                auxiliary_weight=args.auxiliary_weight, candidate_chunk_size=args.candidate_chunk_size, collect_loss=logging)
            count = sum(map(len, microbatches))
            state.update(updates=state["updates"] + 1, offset=offset + count, examples=state["examples"] + count,
                         amp_overflow_retries=state["amp_overflow_retries"] + retries, group_index=group_index + 1)
            if logging:
                emit("training", status="running", **losses, learning_rate=optimizer.param_groups[0]["lr"], training_origin=origin)
            if state["updates"] % args.checkpoint_updates == 0:
                save_latest()
            stopped = stop_reason()
            if stopped:
                timing.finish_train(state, device, complete=state["offset"] == len(eligible))
                save_latest()
                break
        if stopped:
            break
        timing.finish_train(state, device, complete=True)
        try:
            metrics = timing.validation(lambda: evaluate(model, provider, args.batch_size, device, retrieval_topk=args.retrieval_topk,
                candidate_chunk_size=args.candidate_chunk_size, scoring_mode=args.scoring_mode, should_stop=stop_reason), device)
        except EvaluationStopped as error:
            stopped = error.reason
            save_latest()
            break
        state["history"].append({"epoch": epoch + 1, "validation": metrics})
        improved = metrics["ndcg@10"] > state["best_ndcg"]
        state.update(epoch=epoch + 1, offset=0, group_index=0, bad_epochs=0 if improved else state["bad_epochs"] + 1)
        if improved:
            state.update(best_ndcg=metrics["ndcg@10"], selected_epoch=epoch + 1)
            path = args.output / f"selected-epoch{epoch + 1:04d}.pt"

            def publish():
                digest = publish_selected(path, model, optimizer, scaler, state, config)
                shared._atomic_copy(path, args.output / "best.pt")
                write_json(args.output / "best.json", {"sha256": digest})
                write_json(args.output / "selected.json", {"file": path.name, "sha256": digest, "epoch": epoch + 1, "validation": metrics})

            timing.checkpoint(publish, device)
        save_latest()
        emit("validation", status="running", validation=metrics)
    complete = state["epoch"] >= args.schedule_total_epochs or state["bad_epochs"] >= args.patience
    status, selected_metrics = ("completed" if complete else "stopped"), None
    stopped = stopped or stop_reason()
    prediction_status = "no-selected-checkpoint"
    if state["selected_epoch"] is not None:
        selected_metrics = state["history"][state["selected_epoch"] - 1]["validation"]
        prediction_status = "deferred-runtime-stop"
        if not stopped:
            path = args.output / f"selected-epoch{state['selected_epoch']:04d}.pt"
            saved = load_checkpoint(path, config, map_location="cpu")
            model.load_state_dict(saved["state_dict"])
            expected_metrics = saved["state"]["history"][-1]["validation"]
            del saved
            try:
                selected_metrics = evaluate(model, provider, args.batch_size, device, retrieval_topk=args.retrieval_topk,
                    candidate_chunk_size=args.candidate_chunk_size, scoring_mode=args.scoring_mode,
                    predictions_path=args.output / "predictions.npz", should_stop=stop_reason)
                prediction_status = "exported-selected-checkpoint"
                if selected_metrics != expected_metrics:
                    raise ValueError("Selected complete predictions do not reproduce validation")
            except EvaluationStopped as error:
                stopped = error.reason
    result = {"status": status, "reason": stopped, "state": state, "validation": selected_metrics,
              "predictions_status": prediction_status, "integration": integration, "test_accessed": False}
    write_json(args.output / "metrics.json", result)
    artifacts = [path.name for path in args.output.iterdir() if path.name in
                 ("latest.pt", "previous.pt", "best.pt", "selected.json", "predictions.npz", "metrics.json") or path.name.startswith("selected-epoch")]
    write_json(args.output / "completion.json", {"state": status, "updates": state["updates"],
        "selected_epoch": state["selected_epoch"], "artifacts": artifacts,
        "sha256": {name: sha256(args.output / name) for name in artifacts}, "validation": selected_metrics,
        "predictions_status": prediction_status, "integration": integration,
        "test_accessed": False, "learned_status": "optimizer-updated" if state["updates"] else "initialization-only"})
    emit("finished", status=status, reason=stopped)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    module_name, factory_name = args.provider.split(":", 1)
    provider = getattr(importlib.import_module(module_name), factory_name)(args)
    try:
        return run(args, provider)
    finally:
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    main()
