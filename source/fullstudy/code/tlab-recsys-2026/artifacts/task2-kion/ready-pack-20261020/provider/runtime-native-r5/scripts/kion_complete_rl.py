"""Offline CompleteGPT -> frozen complete-RM bridge; no fitting or downloads.

The policy is explicitly full-catalog retrieval + pair scores, unlike SFT's
retrieval-topK reranker. Feature queries always retain the observed train row.
Recommended IDs condition a temporary GPT working sequence only.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import date
import importlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint as recompute

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_policy_rl as rl
import kion_complete_model as architecture
import kion_complete_pack as complete_pack
import kion_local_pack_rl as legacy_rules
import kion_reward_boosting as rm
import kion_reward_data as observed

CONTRACT = "kion-complete-rl-v1"
RM_FEATURE_CONTRACT = "kion-complete-reward-features-v1"
FIT_CAP = date(2021, 8, 2).toordinal()
VAL_ORIGIN = date(2021, 8, 9).toordinal()
EVAL_CONTRACT = "kion-complete-heldout-rl-v1"
COHORT_CONTRACT = "kion-complete-canonical-val-cohort-v1"
SCORING = "full_catalog_retrieval_plus_pair"
ANCHOR = "full_composite_logged_ce_not_complete_sft_total_objective"
READY_SCOPE_SHA256 = "a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962"
READY_SCOPE_ID = "ready-pack-20261020-r1"
SOURCES = ("kion_complete_rl.py", "kion_policy_rl.py", "hm_job.py", "kion_complete_model.py",
           "kion_complete_train.py", "kion_complete_pack.py", "kion_local_pack_rl.py",
           "kion_reward_boosting.py", "kion_reward_data.py", "kion_complete_reward_data.py")
FILES = ("sft_config", "sft_selection", "sft_completion", "sft_checkpoint", "sft_checkpoint_receipt",
         "provider_manifest", "metadata", "rm_bundle_manifest", "rm_model_manifest")


class WaitingArtifact(RuntimeError):
    """A selected, learned, complete artifact is required; random fallback is forbidden."""


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pin(path):
    path = Path(path).resolve(strict=True)
    return {"path": path.as_posix(), "sha256": rl.file_hash(path)}


def _checked_pin(pin):
    if (not isinstance(pin, dict) or set(pin) != {"path", "sha256"}
            or not Path(pin["path"]).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", pin["sha256"])):
        raise ValueError("Explicit absolute file SHA256 pin required")
    path = Path(pin["path"])
    if rl.file_hash(path) != pin["sha256"]:
        raise ValueError("Artifact/source changed: " + path.as_posix())
    return path


def _scope_authority(pin):
    """Only the explicit user-approved ready decision authorizes scoped fitting."""
    path = _checked_pin(pin)
    decision = _json(path)
    if (pin != _pin(path) or pin["sha256"] != READY_SCOPE_SHA256
            or decision.get("contract") != "kion-user-approved-ready-scope-v1"
            or decision.get("scope_id") != READY_SCOPE_ID):
        raise ValueError("Scoped RL requires the exact approved ready scope decision")
    return decision


def _scoped_identity(declaration, pin, *, mode_key="integration_mode", provider=False):
    decision = _scope_authority(pin)
    identity = {"scope_id": decision["scope_id"], "scope_decision": pin,
                "active_families": decision["active_families"]}
    if (declaration.get(mode_key) != "scoped_production" or declaration.get("status") != "admitted_scoped"
            or declaration.get("full_package_admitted") is not False
            or declaration.get("full_row_fit_admitted") is not True
            or any(declaration.get(key) != value for key, value in identity.items())):
        raise ValueError("Scoped production identity/flags differ from the approved ready package")
    if provider and (declaration.get("additional_ready_components") != decision["additional_ready_components"]
            or declaration.get("family_acceptance") != dict.fromkeys(decision["active_families"], "accepted_scoped")
            or declaration.get("materialized_families") != decision["active_families"]):
        raise ValueError("Scoped provider must materialize exactly the accepted ready families/components")
    return identity


def production_admission(declaration, integration, *, scope_decision=None, fixture_mode=False):
    """Keep full/fixture gates; scoped production is a separate pinned authority."""
    if fixture_mode:
        return {"mode": "explicit_fixture_only", "production_admitted": False}
    if declaration.get("status") == "admitted_scoped" or integration.get("status") == "admitted_scoped":
        if scope_decision is None:
            raise ValueError("Scoped production requires an explicit scope decision")
        pin = _pin(scope_decision)
        identity = _scoped_identity(declaration, pin, provider=True)
        if _scoped_identity(integration, pin, mode_key="mode") != identity:
            raise ValueError("Selected SFT/provider scoped identities differ")
        return {"mode": "scoped_production", **identity, "full_package_admitted": False,
                "full_row_fit_admitted": True}
    if (scope_decision is not None or declaration.get("status") != "admitted_full"
            or integration.get("full_package_admitted") is not True):
        raise ValueError("Production complete RL refuses partial/pilot feature or SFT admission")
    return {"mode": "admitted_full", "full_package_admitted": True}


def reward_admission(declaration, admission, *, scope_decision=None):
    """The refreshed RM must use the same approved scope, with its own Aug2 cap."""
    if admission["mode"] == "scoped_production":
        identity = _scoped_identity(declaration, _pin(scope_decision), provider=True)
        if any(admission[key] != value for key, value in identity.items()):
            raise ValueError("Ready RM and SFT scoped identities differ")
    elif declaration.get("status") != "admitted_full":
        raise ValueError("Full-package policy requires a refreshed full-package RM")


@dataclass(frozen=True)
class ActionBudget:
    batch_rows: int = 2
    candidate_chunk_size: int = 32
    max_pair_cells_per_forward: int = 262144
    max_forward_seconds: float = 60.0
    max_chunk_seconds: float = 30.0

    def validate(self):
        for name, upper in (("batch_rows", 64), ("candidate_chunk_size", 4096),
                            ("max_pair_cells_per_forward", 2_000_000)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in 1..{upper}")
        for name, upper in (("max_forward_seconds", 60), ("max_chunk_seconds", 30)):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f"{name} requires a finite value in (0,{upper}]")


@dataclass
class CompleteRLData(rl.RLData):
    provider_rows: np.ndarray | None = None
    provider_signature: str | None = None
    policy_contract: dict | None = None

    def validate(self, vocab, context_length, rollout_steps):
        super().validate(vocab, context_length, rollout_steps)
        if (not isinstance(self.provider_rows, np.ndarray) or self.provider_rows.dtype != np.int64
                or self.provider_rows.shape != (len(self.x),) or np.any(self.provider_rows < 0)
                or not self.provider_signature or self.policy_contract is None):
            raise ValueError("Complete RL requires immutable original provider rows and policy identity")

    def signature(self):
        return rl.stable_hash({"base": super().signature(), "provider_rows": self.provider_rows,
                               "provider_signature": self.provider_signature, "policy": self.policy_contract})


@dataclass
class CompleteValidationData(CompleteRLData):
    """Observed validation only; it can never be sent to the RL optimizer."""
    provider_split: str = "val"

    def validate(self, vocab, context_length, rollout_steps):
        if rollout_steps != 0 or self.provider_split != "val" or self.y is not None:
            raise ValueError("Validation rows are evaluation-only; no optimization or CE labels")
        super().validate(vocab, context_length, 0)

    def signature(self):
        return rl.stable_hash({"observed": super().signature(), "split": self.provider_split,
                               "role": "heldout_evaluation_only"})


class CompletePolicyAdapter(rl.PolicyAdapter):
    """Exact composite logits, deterministic bounded features and recomputation.

    The shared provider is owned externally. Only the torch model is copied for
    reference: provider memmaps/caches must never be deep-copied or moved to CUDA.
    """
    def __init__(self, model, provider, budget=ActionBudget()):
        budget.validate()
        if not isinstance(model, architecture.CompleteGPT):
            raise ValueError("Complete adapter refuses history-only/old SFT models")
        declaration = provider.contract()
        if (model.contract()["source_contract"] != declaration
                or model.feature_columns != provider.feature_columns):
            raise ValueError("Live complete provider differs from the model's independent source/schema binding")
        super().__init__(model, vocab_size=model.vocab_size, context_length=model.context_length)
        self.provider, self.budget = provider, budget
        self.provider_signature = rl.stable_hash(declaration)
        self.observations = {"forwards": 0, "provider_calls": 0, "pair_cells": 0,
                             "feature_seconds": 0.0, "forward_seconds": 0.0}

    def get_extra_state(self):
        return {"contract": CONTRACT, "policy_scoring": SCORING, "ce_anchor": ANCHOR,
                "provider_signature": self.provider_signature, "budget": asdict(self.budget),
                "scoring_source_sha256": rl.file_hash(Path(__file__))}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError("Complete RL adapter source/policy/budget changed")

    def _chunk_width(self, count):
        self.budget.validate()
        limits = self.provider.contract().get("limits", {})
        max_rows = getattr(self.provider, "max_batch_rows", limits.get("max_batch_rows", 0))
        max_cells = getattr(self.provider, "max_pair_cells", limits.get("max_pair_cells", 0))
        max_bytes = getattr(self.provider, "max_pair_bytes", limits.get("max_pair_bytes", 0))
        if not 1 <= count <= min(self.budget.batch_rows, max_rows):
            raise ValueError("Requested rough batch exceeds the admitted complete-provider/RL budget")
        if count * (self.vocab_size - 3) > self.budget.max_pair_cells_per_forward:
            raise ValueError("Exact full-catalog work exceeds max_pair_cells_per_forward; no topK fallback")
        width = min(4096, self.budget.candidate_chunk_size, max_cells // count)
        if self.model.candidate_feature_dim:
            width = min(width, max_bytes // (4 * count * self.model.candidate_feature_dim))
        if width < 1:
            raise ValueError("No feasible candidate chunk within provider cell/byte budget")
        return width

    def forward(self, data, indices, prefix=None):
        if (not isinstance(data, CompleteRLData) or data.provider_signature != self.provider_signature
                or data.policy_contract != self.get_extra_state()):
            raise ValueError("Complete adapter requires exactly bound observed rows and scoring contract")
        indices = np.asarray(indices)
        if indices.dtype != np.int64 or indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(data.x)):
            raise ValueError("Invalid local RL indices")
        chunk_width = self._chunk_width(len(indices))
        rows = data.provider_rows[indices].copy()
        rows.flags.writeable = False
        split = "val" if isinstance(data, CompleteValidationData) else "train"
        device = next(self.model.parameters()).device
        working_x, working_lengths = rl.policy_context(data, indices, self.context_length, prefix)
        width = int(working_lengths.max())
        x = torch.as_tensor(working_x[:, :width], device=device)
        lengths = torch.as_tensor(working_lengths, device=device)
        user = torch.as_tensor(data.user_features[indices], dtype=torch.float32, device=device)
        started = time.perf_counter()
        table = self.model.candidate_table()
        encoded = self.model.encode_context(x, lengths, user, candidate_table=table)
        retrieval = self.model.retrieval_logits(encoded, table)
        # No target is forced and no retrieval topK is used. Every known token
        # gets the SAME composite formula before one full-catalog normalization.
        ids = torch.arange(3, self.vocab_size, device=device).expand(len(indices), -1)
        compute_seconds = 0.0

        def score_chunk(context, candidates, base, selected):
            nonlocal compute_seconds
            tick = time.perf_counter()
            feature_tick = tick
            values = self.provider.candidate_features(rows, selected.detach().cpu().numpy(), split)
            feature_seconds = time.perf_counter() - feature_tick
            expected = (*selected.shape, self.model.candidate_feature_dim)
            if (not isinstance(values, np.ndarray) or values.dtype != np.float32 or values.shape != expected
                    or not np.isfinite(values).all()):
                raise ValueError("Complete original-row candidate feature shape/dtype/finiteness mismatch")
            self.observations["provider_calls"] += 1
            self.observations["pair_cells"] += selected.numel()
            self.observations["feature_seconds"] += feature_seconds
            if feature_seconds > self.budget.max_chunk_seconds:
                raise TimeoutError("Complete candidate callback exceeded its cooperative chunk budget")
            result = self.model.score_candidates(context, selected, torch.as_tensor(values, device=device),
                                                 candidate_table=candidates, retrieval_logits=base)
            spent = time.perf_counter() - tick
            compute_seconds += spent
            if spent > self.budget.max_chunk_seconds or compute_seconds > 2 * self.budget.max_forward_seconds:
                raise TimeoutError("Complete candidate computation exceeded its measured chunk/recompute budget")
            return result

        scores = []
        for begin in range(0, ids.shape[1], chunk_width):
            if time.perf_counter() - started > self.budget.max_forward_seconds:
                raise TimeoutError("Exact complete full-catalog forward exceeded its cooperative time budget")
            selected = ids[:, begin:begin + chunk_width]
            score = (recompute(score_chunk, encoded, table, retrieval, selected, use_reentrant=False,
                               preserve_rng_state=True) if torch.is_grad_enabled()
                     else score_chunk(encoded, table, retrieval, selected))
            scores.append(score)
        elapsed = time.perf_counter() - started
        if elapsed > self.budget.max_forward_seconds:
            raise TimeoutError("Exact complete full-catalog forward exceeded its cooperative time budget")
        self.observations["forwards"] += 1
        self.observations["forward_seconds"] += elapsed
        return torch.cat((retrieval[:, :3], torch.cat(scores, 1)), 1).float()


class ResourceOwner:
    """The scorer key owns both provider lifetimes through the final backward."""
    def __init__(self, scorer, *providers):
        self.scorer, self.providers, self.closed = scorer, providers, False
        self.preflight_report = None

    @property
    def signature(self):
        return self.scorer.signature

    def score_batch(self, contexts, candidates):
        return self.scorer.score_batch(contexts, candidates)

    def score_with_diagnostics(self, contexts, candidates):
        return self.scorer.score_with_diagnostics(contexts, candidates)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            close_resources(self.scorer, *self.providers)
        finally:
            self.providers = ()


def close_resources(*resources):
    """Attempt every unique cleanup, even after another owner's close fails."""
    errors = []
    for index, resource in enumerate(resources):
        if resource is None or any(resource is old for old in resources[:index]):
            continue
        try:
            resource.close()
        except BaseException as error:
            errors.append(error)
    if errors:
        raise errors[0]


def load_selected_complete(provider, run_dir, *, fixture_mode=False, scope_decision=None):
    """Load only a published optimizer-updated CompleteGPT, never old/random SFT."""
    run_dir = Path(run_dir)
    required = [run_dir / name for name in ("config.json", "selected.json", "completion.json")]
    missing = [path.as_posix() for path in required if not path.is_file()]
    if missing:
        raise WaitingArtifact("Missing learned complete checkpoint receipts: " + ", ".join(missing))
    config, selected, completion = map(_json, required)
    contract = config.get("contract", {})
    saved_architecture = contract.get("architecture", {})
    if saved_architecture.get("contract") != architecture.CONTRACT:
        raise ValueError("Old/history-only SFT checkpoint is incompatible with CompleteGPT")
    declaration = provider.contract()
    if (contract.get("provider") != declaration or saved_architecture.get("source_contract") != declaration
            or saved_architecture.get("feature_columns") != provider.feature_columns):
        raise ValueError("Selected CompleteGPT does not bind the current dynamic provider independently")
    production_admission(declaration, contract.get("integration", {}), scope_decision=scope_decision,
                         fixture_mode=fixture_mode)
    for name, digest in contract.get("sources_sha256", {}).items():
        if Path(name).name != name or rl.file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError("Selected complete model/trainer source changed")
    if not {"kion_complete_model.py", "kion_complete_train.py"} <= set(contract.get("sources_sha256", {})):
        raise ValueError("Complete selected constructor/trainer source pins missing")
    name = selected.get("file")
    if not isinstance(name, str) or re.fullmatch(r"selected-epoch[0-9]+\.pt", name) is None:
        raise ValueError("A published selected CompleteGPT epoch is required")
    path = run_dir / name
    if not path.is_file():
        raise WaitingArtifact("Selected learned CompleteGPT weights are not present: " + path.as_posix())
    if selected.get("sha256") != rl.file_hash(path) or _json(path.with_suffix(".json")).get("sha256") != selected["sha256"]:
        raise ValueError("Complete selected checkpoint checksum receipts disagree")
    if (completion.get("state") != "completed" or completion.get("learned_status") != "optimizer-updated"
            or completion.get("selected_epoch") != selected.get("epoch")
            or type(selected.get("epoch")) is not int or selected["epoch"] < 1):
        raise WaitingArtifact("Complete learned run has not published completed selected artifacts")
    if completion.get("sha256", {}).get(name) != selected["sha256"]:
        raise ValueError("Completed complete run does not receipt the selected learned checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("config", {}).get("contract") != contract or payload.get("state", {}).get("updates", 0) < 1
            or payload["state"].get("selected_epoch") != selected.get("epoch")):
        raise ValueError("Complete checkpoint is initialization-only or bound to another run")
    with torch.random.fork_rng(devices=[]):
        model = architecture.CompleteGPT.from_config(config, item_features=provider.item_features,
            text_features=provider.text_features, people_catalog=provider.people_catalog)
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.cpu().float().eval().requires_grad_(True)
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite CompleteGPT selected parameters")
    return model, config


def join_fit_rows(provider, fit, maximum, *, max_seconds=60.0):
    """Label-blind exact original-key join; missing SFT first-watch rows are counted."""
    if type(maximum) is not int or not 1 <= maximum <= 100000:
        raise ValueError("Bounded fit-context row cap required")
    users, items, dates = (np.asarray(fit[key]) for key in ("user_id", "item_id", "target_date"))
    if any(value.dtype != np.int64 or value.shape != users.shape for value in (users, items, dates)):
        raise ValueError("RM fit row keys must be aligned int64")
    if np.any(dates >= FIT_CAP) or np.any(dates < 1) or not len(users):
        raise ValueError("RL optimization contexts require observed dates strictly before Aug2")
    keys = list(zip(users.tolist(), items.tolist(), dates.tolist()))
    if len(set(keys)) != len(keys):
        raise ValueError("RM fit keys must be unique")
    wanted, found = {key: index for index, key in enumerate(keys)}, {}
    train, started = provider.splits["train"], time.perf_counter()
    for begin in range(0, len(train["y"]), 65536):
        if time.perf_counter() - started > max_seconds:
            raise TimeoutError("Complete/RM original row-key join exceeded its bounded probe budget")
        end = min(begin + 65536, len(train["y"]))
        for row in range(begin, end):
            key = tuple(int(train[name][row]) for name in ("user_id", "target_item_id", "target_date"))
            if key in wanted:
                rm_row = wanted[key]
                if rm_row in found:
                    raise ValueError("Complete SFT row key is duplicated")
                found[rm_row] = row
    selected = sorted(found)[:maximum]
    if not selected:
        raise WaitingArtifact("No exact observed RM-fit context joins the complete train rows")
    return np.asarray([found[index] for index in selected], np.int64), np.asarray(selected, np.int64), {
        "rm_fit_rows": len(keys), "matched_rows": len(found), "selected_rows": len(selected),
        "not_in_complete_train": len(keys) - len(found), "selection": "first matched frozen RM-fit order; label blind"}


def assemble_bundle(provider, model, provider_rows, *, reward_contexts, candidate_provider,
                    candidate_signature, seen_tokens, eligible_tokens, scorer, provenance,
                    budget=ActionBudget(), reward_provider=None):
    """Concrete in-process seam shared by production loading and tiny CPU probes."""
    rows = np.asarray(provider_rows)
    if rows.dtype != np.int64 or rows.ndim != 1 or not len(rows):
        raise ValueError("Original complete train row indices must be nonempty int64")
    if (np.any(rows >= len(provider.splits["train"]["y"]))
            or np.any(provider.splits["train"]["target_date"][rows] >= FIT_CAP)):
        raise ValueError("Complete RL requires original train origins strictly before Aug2")
    policy = CompletePolicyAdapter(model, provider, budget)
    reference = CompletePolicyAdapter(copy.deepcopy(model).eval().requires_grad_(False), provider, budget)
    x, lengths, targets, features = [], [], [], []
    setup_started = time.perf_counter()
    for begin in range(0, len(rows), budget.batch_rows):
        if time.perf_counter() - setup_started > 60:
            raise TimeoutError("Observed Complete user-feature materialization exceeded its cooperative setup budget")
        values = provider.batch(rows[begin:begin + budget.batch_rows], "train")
        arrays = [value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value) for value in values]
        padded = np.zeros((len(arrays[0]), model.context_length), np.int64)
        padded[:, :arrays[0].shape[1]] = arrays[0]
        x.append(padded)
        lengths.append(arrays[1].astype(np.int64))
        targets.append(arrays[2].astype(np.int64))
        features.append(arrays[3].astype(np.float32))
    mapping = observed.checked_mapping(provider.metadata)
    data = CompleteRLData(x=np.concatenate(x), lengths=np.concatenate(lengths), y=np.concatenate(targets),
        user_features=np.concatenate(features), token_to_item={token: raw for raw, token in mapping.items()},
        reward_contexts=reward_contexts, candidate_features_provider=candidate_provider,
        candidate_features_signature=candidate_signature, seen_tokens=seen_tokens, eligible_tokens=eligible_tokens,
        provider_rows=rows.copy(), provider_signature=policy.provider_signature, policy_contract=policy.get_extra_state(),
        provenance={"contract": CONTRACT, "pins": provenance, "policy_scoring": SCORING, "ce_anchor": ANCHOR,
                    "features": "observed original rows only; recommended IDs affect GPT working sequence only",
                    "optimizer": "inherited REINFORCE EMA-prior baseline; AdamW weight_decay=0",
                    "baseline_rule": "evaluate update0 using this full-composite adapter and the same support"})
    data.validate(model.vocab_size, model.context_length, 1)
    for value in (data.x, data.lengths, data.y, data.user_features, data.provider_rows):
        value.flags.writeable = False
    owner = ResourceOwner(scorer, provider, *(() if reward_provider is None else (reward_provider,)))
    return {"policy": policy, "reference": reference, "data": data, "scorer": owner}


def preflight(bundle):
    """Measure actual chosen rough-batch callback/compute cost without fitting."""
    policy, reference, data = (bundle[key] for key in ("policy", "reference", "data"))
    count = min(policy.budget.batch_rows, len(data.x))
    indices = np.arange(count, dtype=np.int64)
    prefix = np.empty((count, 0), dtype=np.int64)
    started = time.perf_counter()
    policy.eval()
    reference.eval()
    with torch.no_grad():
        mask = rl.candidate_mask(data, indices, policy.vocab_size, prefix)
        p = rl.masked_log_probs(policy(data, indices, prefix), mask)
        q = rl.masked_log_probs(reference(data, indices, prefix), mask)
        divergence = rl.exact_kl(p, q, mask)
    if not torch.isfinite(divergence).all() or divergence.abs().max() > 1e-6:
        raise ValueError("Initial complete policy/reference distributions do not reproduce")
    report = {"state": "measured_not_fitted", "policy_scoring": SCORING, "batch_rows": count,
        "catalog_actions": policy.vocab_size - 3, "policy": dict(policy.observations),
        "reference": dict(reference.observations), "wall_seconds": time.perf_counter() - started,
        "initial_exact_kl_max_abs": float(divergence.abs().max()), "budget": asdict(policy.budget),
        "model_sha256": rl.model_hash(policy), "data_sha256": data.signature(), "device": "cpu",
        "admission": data.provenance.get("pins", {}).get("admission"),
        "hard_deadline": "external owned supervisor required; in-callback guards are cooperative",
        "not_online_retention_evidence": True}
    bundle["scorer"].preflight_report = report
    return report


def status_receipt(sft_dir, provider_dir, rm_bundle, rm_artifact):
    required = [Path(sft_dir) / name for name in ("config.json", "selected.json", "completion.json")]
    required += [Path(provider_dir) / "manifest.json", Path(rm_bundle) / "manifest.json", Path(rm_artifact) / "manifest.json"]
    missing = [path.resolve().as_posix() for path in required if not path.is_file()]
    if required[0].is_file() and _json(required[0]).get("contract", {}).get("architecture", {}).get("contract") != architecture.CONTRACT:
        raise ValueError("Old SFT checkpoint cannot become a complete RL artifact")
    if not missing:
        config, selected = _json(required[0]), _json(required[1])
        if config.get("contract", {}).get("architecture", {}).get("contract") != architecture.CONTRACT:
            raise ValueError("Old SFT checkpoint cannot become a complete RL artifact")
        name = selected.get("file", "")
        if re.fullmatch(r"selected-epoch[0-9]+\.pt", name) is None:
            raise ValueError("Selected complete checkpoint path is invalid")
        if not (Path(sft_dir) / name).is_file():
            missing.append((Path(sft_dir) / name).resolve().as_posix())
    dependencies = {}
    for name, root, files in (("complete_gpt", sft_dir, ("config.json", "selected.json", "completion.json")),
                             ("complete_sft_features", provider_dir, ("manifest.json",)),
                             ("complete_rm_data", rm_bundle, ("manifest.json",)),
                             ("complete_rm_scorer", rm_artifact, ("manifest.json",))):
        dependencies[name] = {"path": Path(root).resolve().as_posix(), "state": "planned_missing" if any(
            not (Path(root) / name).is_file() for name in files) else "present_requires_full_admission"}
    return {"contract": CONTRACT, "state": "waiting_artifact" if missing else "requires_full_admission",
            "missing": missing, "policy_scoring": SCORING, "ce_anchor": ANCHOR,
            "dependencies": dependencies,
            "fit_status": "no_complete_RL_fit", "random_fallback": False,
            "scope": "offline bridge preparation; source contracts are not fitted weights"}


class RMCandidates:
    def __init__(self, provider, users, origins):
        self.provider, self.users, self.origins = provider, users.copy(), origins.copy()

    def __call__(self, indices, tokens):
        origins = self.origins[indices]
        return self.provider.candidate_features(tokens, origins, history_cutoffs=origins,
                                                user_ids=self.users[indices]).astype(np.float32)


def _read_descriptor(descriptor_path):
    """Shared strict artifact/source admission; this does not start a worker."""
    value = _json(descriptor_path)
    expected_fields = {"contract", "state", "policy_scoring", "ce_anchor", "files", "sources", "max_rows",
                       "budget", "catboost_python", "rpc_timeout_seconds"}
    if value.get("contract") != CONTRACT or value.get("state") != "pinned_requires_admission":
        raise WaitingArtifact("A pinned learned CompleteGPT/complete-RM descriptor is not ready")
    if "scope_decision" in value:
        expected_fields.add("scope_decision")
    if (set(value) != expected_fields or set(value["files"]) != set(FILES)
            or value["policy_scoring"] != SCORING or value["ce_anchor"] != ANCHOR
            or type(value["max_rows"]) is not int or not 1 <= value["max_rows"] <= 100000
            or not Path(value["catboost_python"]).is_absolute() or not Path(value["catboost_python"]).is_file()
            or not math.isfinite(value["rpc_timeout_seconds"]) or not 0 < value["rpc_timeout_seconds"] <= 60):
        raise ValueError("Invalid exact complete policy/artifact/bounded-runtime descriptor")
    paths = {name: _checked_pin(pin) for name, pin in value["files"].items()}
    scope_decision = None
    if "scope_decision" in value:
        _scope_authority(value["scope_decision"])
        scope_decision = _checked_pin(value["scope_decision"])
    scope_options = {} if scope_decision is None else {"scope_decision": scope_decision}
    for name, pin in value["sources"].items():
        if _checked_pin(pin).resolve() != Path(__file__).with_name(name).resolve():
            raise ValueError("Complete RL source path mismatch")
    if set(value["sources"]) != set(SOURCES):
        raise ValueError("Complete RL dependency source census changed")
    run_dir = paths["sft_config"].parent
    for name, relative in (("sft_selection", "selected.json"), ("sft_completion", "completion.json")):
        if paths[name] != run_dir / relative:
            raise ValueError("Selected complete run file path binding differs")
    if (paths["sft_checkpoint"].parent != run_dir
            or paths["sft_checkpoint_receipt"] != paths["sft_checkpoint"].with_suffix(".json")
            or _json(paths["sft_selection"]).get("file") != paths["sft_checkpoint"].name):
        raise ValueError("Complete selected epoch path binding differs")
    budget = ActionBudget(**value["budget"])
    budget.validate()
    return value, paths, run_dir, budget, scope_options


def build_factory(descriptor_path):
    """Generic RL CLI factory; production requires refreshed complete RM."""
    value, paths, run_dir, budget, scope_options = _read_descriptor(descriptor_path)
    provider = reward_provider = scorer = None
    try:
        # Guard before expensive provider indices or a CPU RPC process.
        check = status_receipt(run_dir, paths["provider_manifest"].parent,
                               paths["rm_bundle_manifest"].parent, paths["rm_model_manifest"].parent)
        if check["state"] != "requires_full_admission":
            raise WaitingArtifact(json.dumps(check))
        provider = complete_pack.CompleteProvider.load(paths["metadata"].parent, paths["provider_manifest"].parent,
                                                       **scope_options)
        model, config = load_selected_complete(provider, run_dir, **scope_options)
        admission = production_admission(provider.contract(), config["contract"].get("integration", {}),
                                         **scope_options)
        complete_reward = importlib.import_module("kion_complete_reward_data")
        roles, catalog, bundle_manifest = complete_reward.load_complete_reward_bundle(paths["rm_bundle_manifest"].parent,
                                                                                      **scope_options)
        reward_provider = complete_reward.load_reward_provider_from_bundle(paths["rm_bundle_manifest"].parent,
                                                                            **scope_options)
        if (bundle_manifest.get("feature_pack", {}).get("contract") != RM_FEATURE_CONTRACT
                or bundle_manifest["feature_pack"].get("metadata_sha256") != rl.file_hash(paths["metadata"])
                or reward_provider.fit_cutoff != "2021-08-02"
                or reward_provider.user_columns != catalog.context_names
                or reward_provider.candidate_columns != catalog.item_names):
            raise ValueError("Production RL requires the refreshed all-family Aug2 RM schema")
        reward_admission(bundle_manifest["feature_pack"], admission, **scope_options)
        if reward_provider.contract() != bundle_manifest["feature_pack"]:
            raise ValueError("Refreshed RM does not bind its independently reconstructed provider")
        frozen = legacy_rules._frozen_catboost(paths["rm_model_manifest"], rl.file_hash(Path(rm.__file__)),
                                               rl.file_hash(paths["rm_bundle_manifest"]))
        if frozen.get("catalog") != catalog.describe():
            raise ValueError("Complete frozen RM bundle/model schemas differ")
        eligible, pool = legacy_rules._action_pool(paths["rm_model_manifest"], frozen, "supported_fit_only")
        mapping = observed.checked_mapping(provider.metadata)
        if catalog.mapping != {str(raw): token for raw, token in mapping.items()}:
            raise ValueError("Complete policy and frozen RM raw vocabulary differ")
        provider_rows, reward_rows, joined = join_fit_rows(provider, roles["fit"], value["max_rows"])
        fit = roles["fit"]
        users, origins = fit["user_id"][reward_rows], fit["history_cutoff"][reward_rows]
        root = paths["rm_bundle_manifest"].parent
        observed_x = np.load(root / "fit.x.npy", mmap_mode="r", allow_pickle=False)[reward_rows].copy()
        observed_lengths = np.load(root / "fit.lengths.npy", mmap_mode="r", allow_pickle=False)[reward_rows].copy()
        raw = observed.RawRows.from_pack(reward_provider, mapping)
        expected_x, expected_lengths, _ = raw.history_batch(users, origins)
        original = provider.splits["train"]
        if (not np.array_equal(observed_x, expected_x) or not np.array_equal(observed_lengths, expected_lengths)
                or not np.array_equal(observed_x, original["x"][provider_rows])
                or not np.array_equal(observed_lengths, original["lengths"][provider_rows])):
            raise ValueError("Complete/RM rows do not share the exact original observed H63 state")
        numeric = np.asarray(fit["numeric"][reward_rows], np.float32)
        for begin in range(0, len(users), budget.batch_rows):
            sl = slice(begin, begin + budget.batch_rows)
            actual = reward_provider.user_features(users[sl], origins[sl], observed_x[sl], observed_lengths[sl],
                                                   history_cutoffs=origins[sl])
            if not np.array_equal(actual, numeric[sl]):
                raise ValueError("Prepared complete RM state differs from its frozen typed provider")
        contexts = tuple({"user_id": int(users[i]), "numeric": numeric[i],
                          "categorical": tuple(fit["categorical"][reward_rows[i]]),
                          "schema_signature": catalog.schema_signature} for i in range(len(users)))
        command = [value["catboost_python"], Path(rm.__file__).resolve().as_posix(), "serve", "--artifact",
                   paths["rm_model_manifest"].parent.as_posix(), "--expected-mapping-sha256", catalog.mapping_sha256]
        scorer = rl.ProcessRewardScorer(command, frozen["signature"], timeout=value["rpc_timeout_seconds"])
        if any(scorer.metadata.get(key) != expected for key, expected in (
                ("schema_signature", catalog.schema_signature), ("context_numeric_dim", len(catalog.context_names)),
                ("candidate_numeric_dim", len(catalog.item_names)))):
            raise ValueError("Complete frozen CPU reward worker handshake schema differs")
        bundle = assemble_bundle(provider, model, provider_rows, reward_contexts=contexts,
            candidate_provider=RMCandidates(reward_provider, users, origins),
            candidate_signature=rl.stable_hash({"descriptor": value, "users": users, "origins": origins}),
            seen_tokens=legacy_rules._full_seen(raw, users, origins), eligible_tokens=eligible, scorer=scorer,
            provenance={"descriptor": value, "join": joined, "action_pool": pool, "admission": admission}, budget=budget,
            reward_provider=reward_provider)
        preflight(bundle)
        return bundle
    except BaseException as error:
        try:
            close_resources(scorer, reward_provider, provider)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


def validation_cohort(provider, *, max_rows=256, seed=2026):
    """Freeze a label-blind subset of canonical validation identities, never test."""
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000 or type(seed) is not int or seed < 0:
        raise ValueError("A bounded validation row count and nonnegative seed are required")
    data = provider.splits["val"]
    total = len(data["y"])
    if not total:
        raise ValueError("Canonical validation is empty")
    rows = np.sort(np.random.default_rng(seed).choice(total, min(max_rows, total), replace=False)).astype(np.int64)
    origins = np.asarray(provider.origin_keys("val"))[rows]
    if (origins.dtype != np.int64 or not np.all(origins == VAL_ORIGIN)
            or np.any(data["target_date"][rows] < origins)):
        raise ValueError("Validation must retain the original Aug9 query clock and later observed targets")
    keys = {name: np.asarray(data[name][rows]) for name in
            ("user_id", "target_item_id", "target_date", "y", "x", "lengths")}
    return {"contract": COHORT_CONTRACT, "split": "val", "selection": "seeded_without_replacement_label_blind",
        "seed": seed, "max_rows": max_rows, "canonical_rows": total, "row_indices": rows.tolist(),
        "row_identity_sha256": rl.stable_hash(keys), "provider_signature": rl.stable_hash(provider.contract()),
        "query_origin": "2021-08-09", "unknown_targets": "misses_in_all_row_denominators"}


def checked_validation_cohort(provider, cohort):
    if (not isinstance(cohort, dict) or cohort.get("contract") != COHORT_CONTRACT
            or validation_cohort(provider, max_rows=cohort.get("max_rows"), seed=cohort.get("seed")) != cohort):
        raise ValueError("Frozen canonical validation cohort/provider/row identity changed")
    return np.asarray(cohort["row_indices"], np.int64)


def validation_data(adapter, provider, rows, seen_tokens, eligible_tokens):
    values = provider.batch(rows, "val")
    x, lengths, _, user = [value.detach().cpu().numpy() if isinstance(value, torch.Tensor)
                           else np.asarray(value) for value in values]
    padded = np.zeros((len(rows), adapter.context_length), np.int64)
    padded[:, :x.shape[1]] = x
    original = provider.splits["val"]
    if (not np.array_equal(padded, original["x"][rows])
            or not np.array_equal(lengths, original["lengths"][rows])):
        raise ValueError("Heldout provider batch must preserve the exact original observed validation H63")
    mapping = observed.checked_mapping(provider.metadata)
    data = CompleteValidationData(x=padded, lengths=lengths.astype(np.int64),
        user_features=user.astype(np.float32), y=None, seen_tokens=seen_tokens,
        token_to_item={token: raw for raw, token in mapping.items()}, reward_contexts=tuple({} for _ in rows),
        eligible_tokens=eligible_tokens, provider_rows=rows.copy(), provider_signature=adapter.provider_signature,
        policy_contract=adapter.get_extra_state(), provenance={"role": "canonical_heldout_val", "query_origin": "2021-08-09"})
    data.validate(adapter.vocab_size, adapter.context_length, 0)
    return data


def _ranking_metrics(top, targets, vocab, cutoff):
    """One observed relevant target per row: AP is reciprocal hit rank."""
    known = (targets >= 3) & (targets < vocab)
    hit = (top[:, :cutoff] == targets[:, None]) & known[:, None]
    ranks = np.arange(hit.shape[1]) + 1
    return {"recall@" + str(cutoff): hit.any(1).astype(np.float64),
            "ndcg@" + str(cutoff): (hit / np.log2(ranks + 1)).sum(1),
            "map@" + str(cutoff): (hit / ranks).sum(1)}


def _sft_comparison(model, provider, rows, batch_size, chunk_size, should_stop):
    """Call the unchanged retrieval128/rerank evaluator, on this exact cohort."""
    from kion_complete_train import evaluate
    with tempfile.TemporaryDirectory(prefix="kion-val-comparison-") as directory:
        predictions = Path(directory) / "validation-predictions.npz"
        result = evaluate(model, provider, batch_size, next(model.parameters()).device,
            retrieval_topk=128, candidate_chunk_size=chunk_size, scoring_mode="rerank", indices=rows,
            predictions_path=predictions, should_stop=should_stop)
        with np.load(predictions, allow_pickle=False) as values:
            top = values["top_items"]
        result["map@10"] = float(_ranking_metrics(top, provider.splits["val"]["y"][rows], model.vocab_size, 10)["map@10"].mean())
    return result


@torch.no_grad()
def evaluate_heldout(policy, reference, provider, cohort, *, eligible_tokens, full_seen,
                     max_seconds=120.0, include_sft_comparison=True):
    """Paired heldout full-composite policy metrics; no RM query or fitting."""
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError("A finite heldout evaluation budget in (0,3600] seconds is required")
    if (policy.provider is not provider or reference.provider is not provider
            or policy.get_extra_state() != reference.get_extra_state() or policy is reference
            or any(p.data_ptr() == q.data_ptr() for p in policy.parameters() for q in reference.parameters())):
        raise ValueError("Paired validation needs independent models and one identical provider/scoring budget")
    rl.execution_device(policy, reference)
    rows = checked_validation_cohort(provider, cohort)
    policy.eval()
    reference.eval()
    hashes = {"policy": rl.model_hash(policy), "reference": rl.model_hash(reference)}
    started = time.perf_counter()

    def should_stop():
        return "heldout_evaluation_time_budget" if time.perf_counter() - started > max_seconds else None

    accumulators = {name: {"metrics": {}, "entropy": 0., "effective_support": 0., "ce": 0.,
        "top10_counts": np.zeros(policy.vocab_size, np.int64), "top1_counts": np.zeros(policy.vocab_size, np.int64)}
        for name in hashes}
    known_count = excluded_targets = 0
    kl_sum = kl_max = 0.
    minimum_support, maximum_support = policy.vocab_size, 0
    for begin in range(0, len(rows), policy.budget.batch_rows):
        if should_stop():
            raise TimeoutError(should_stop())
        selected = rows[begin:begin + policy.budget.batch_rows]
        data = validation_data(policy, provider, selected, full_seen(selected), eligible_tokens)
        local = np.arange(len(selected), dtype=np.int64)
        mask = rl.candidate_mask(data, local, policy.vocab_size)
        support = mask.sum(1)
        minimum_support, maximum_support = min(minimum_support, int(support.min())), max(maximum_support, int(support.max()))
        targets = np.asarray(provider.splits["val"]["y"][selected])
        known = (targets >= 3) & (targets < policy.vocab_size)
        known_count += int(known.sum())
        excluded_targets += sum(not mask[i, int(target)] for i, target in enumerate(targets) if known[i])
        logits = {"policy": policy(data, local), "reference": reference(data, local)}
        logs = {name: rl.masked_log_probs(value, mask) for name, value in logits.items()}
        divergence = rl.exact_kl(logs["policy"], logs["reference"], mask)
        if not torch.isfinite(divergence).all() or divergence.min() < -1e-5:
            raise FloatingPointError("Nonfinite or negative heldout exact KL")
        kl_sum += float(divergence.sum())
        kl_max = max(kl_max, float(divergence.max()))
        for name, value in logits.items():
            if not torch.isfinite(value[:, 3:]).all():
                raise FloatingPointError("Nonfinite full-composite heldout catalog logits")
            accumulator = accumulators[name]
            device_mask = torch.as_tensor(mask, device=value.device)
            ranked = torch.argsort(value.masked_fill(~device_mask, -torch.inf), descending=True, stable=True)[:, :min(20, policy.vocab_size - 3)]
            valid = device_mask.gather(1, ranked)
            top = ranked.masked_fill(~valid, -1).cpu().numpy()
            for cutoff in (10, 20):
                for metric, per_row in _ranking_metrics(top, targets, policy.vocab_size, cutoff).items():
                    accumulator["metrics"][metric] = accumulator["metrics"].get(metric, 0.) + float(per_row.sum())
            log = logs[name]
            entropy = -torch.where(device_mask, log.exp() * log, 0.).sum(1)
            accumulator["entropy"] += float(entropy.sum())
            accumulator["effective_support"] += float(entropy.exp().sum())
            if known.any():
                known_tensor = torch.as_tensor(known, device=value.device)
                expected = torch.as_tensor(targets[known] - 3, device=value.device)
                accumulator["ce"] += float(torch.nn.functional.cross_entropy(value[known_tensor, 3:], expected, reduction="sum"))
            for key, selected_top in (("top10_counts", top[:, :10]), ("top1_counts", top[:, :1])):
                ids = selected_top[selected_top >= 3]
                accumulator[key] += np.bincount(ids, minlength=policy.vocab_size)
    metrics = {}
    eligible_count = int(np.asarray(eligible_tokens).sum())
    for name, accumulator in accumulators.items():
        counts, first = accumulator["top10_counts"], accumulator["top1_counts"]
        exposures = int(counts.sum())
        metrics[name] = {**{key: value / len(rows) for key, value in accumulator["metrics"].items()},
            "mean_entropy": accumulator["entropy"] / len(rows),
            "mean_effective_support": accumulator["effective_support"] / len(rows),
            "full_known_catalog_ce": accumulator["ce"] / known_count if known_count else None,
            "top10_unique_items": int((counts > 0).sum()),
            "top10_eligible_catalog_coverage": float((counts > 0).sum() / eligible_count),
            "max_top1_item_share": float(first.max() / len(rows)),
            "top10_exposure_hhi": float(((counts / exposures) ** 2).sum()) if exposures else None}
    sft_comparison = None
    if include_sft_comparison:
        sft_comparison = {name: _sft_comparison(adapter.model, provider, rows, policy.budget.batch_rows,
            policy._chunk_width(min(len(rows), policy.budget.batch_rows)), should_stop)
            for name, adapter in (("reference", reference), ("policy", policy))}
    if should_stop():
        raise TimeoutError(should_stop())
    if hashes != {"policy": rl.model_hash(policy), "reference": rl.model_hash(reference)}:
        raise RuntimeError("Evaluation mutated policy/reference weights")
    return {"contract": EVAL_CONTRACT, "state": "evaluated_not_fitted", "cohort": cohort,
        "examples": len(rows), "known_targets": known_count, "unknown_target_misses": len(rows) - known_count,
        "known_targets_excluded_by_action_support": int(excluded_targets), "test_accessed": False,
        "full_composite": {"protocol": SCORING + "; frozen_fit_support_minus_full_observed_seen; no_forced_target",
            "tie_rule": "descending_logit_then_ascending_token", "reference": metrics["reference"], "policy": metrics["policy"],
            "policy_minus_reference": {key: metrics["policy"][key] - metrics["reference"][key]
                for key in ("map@10", "recall@10", "ndcg@10", "map@20", "recall@20", "ndcg@20")},
            "mean_exact_kl_policy_reference": kl_sum / len(rows), "max_exact_kl_policy_reference": kl_max,
            "min_action_support": minimum_support, "max_action_support": maximum_support,
            "eligible_tokens_sha256": rl.stable_hash(eligible_tokens)},
        "sft_top128_comparison": {"protocol": "unchanged_complete_train_retrieval128_rerank_no_seenmask_no_forced_target",
            "same_canonical_cohort": True, "metrics": sft_comparison} if include_sft_comparison else {"state": "not_requested"},
        "heldout_RM_reward": {"state": "unavailable", "reason": "typed_RM_query_cap_Aug2_cannot_score_Aug9_state; no_date_clamping_or_fabricated_features"},
        "RM_fit_cutoff_exclusive": "2021-08-02", "policy_query_origin": "2021-08-09",
        "model_sha256": hashes, "wall_seconds": time.perf_counter() - started,
        "uncertainty": "fixed_cohort_point_metrics; no_significance_or_online_quality_claim",
        "not_online_retention_evidence": True, "hard_deadline": "external_owned_supervisor_required"}


def _evaluation_resources(training_bundle):
    """Load learned SFT and support receipts only; never start an RM RPC worker."""
    value, paths, run_dir, budget, scope_options = _read_descriptor(training_bundle)
    provider = None
    try:
        provider = complete_pack.CompleteProvider.load(paths["metadata"].parent, paths["provider_manifest"].parent, **scope_options)
        model, config = load_selected_complete(provider, run_dir, **scope_options)
        admission = production_admission(provider.contract(), config["contract"].get("integration", {}), **scope_options)
        feature_pack = _json(paths["rm_bundle_manifest"]).get("feature_pack", {})
        if (feature_pack.get("contract") != RM_FEATURE_CONTRACT or feature_pack.get("role") != "rm"
                or feature_pack.get("fit_cutoff_exclusive") != "2021-08-02"
                or feature_pack.get("metadata_sha256") != rl.file_hash(paths["metadata"])):
            raise ValueError("Heldout action pool needs the refreshed typed Aug2 RM receipt")
        reward_admission(feature_pack, admission, **scope_options)
        frozen = legacy_rules._frozen_catboost(paths["rm_model_manifest"], rl.file_hash(Path(rm.__file__)),
                                               rl.file_hash(paths["rm_bundle_manifest"]))
        eligible, pool = legacy_rules._action_pool(paths["rm_model_manifest"], frozen, "supported_fit_only")
        mapping = observed.checked_mapping(provider.metadata)
        if frozen.get("catalog", {}).get("mapping_sha256") != rm.canonical_hash({str(raw): token for raw, token in mapping.items()}):
            raise ValueError("Heldout policy/RM action mapping differs")
        reference = CompletePolicyAdapter(model.eval().requires_grad_(False), provider, budget)
        policy = CompletePolicyAdapter(copy.deepcopy(model).eval(), provider, budget)
        return {"provider": provider, "policy": policy, "reference": reference, "eligible": eligible,
                "pool": pool, "frozen": frozen, "admission": admission}
    except BaseException as error:
        try:
            close_resources(provider)
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise


def _admit_rl_checkpoint(resources, checkpoint_path):
    path = Path(checkpoint_path).resolve(strict=True)
    if path.name not in ("latest.pt", "previous.pt"):
        raise ValueError("Explicit verified latest/previous RL checkpoint is required")
    configuration = path.parent / "config.json"
    contract = _json(configuration)
    expected = {"reference_sha256": rl.model_hash(resources["reference"]), "reward_state": "frozen_observed_s0",
        "source_sha256": rl.file_hash(Path(rl.__file__)), "owned_process_helper_sha256": rl.file_hash(Path(rl.__file__).with_name("hm_job.py")),
        "vocab_size": resources["policy"].vocab_size, "context_length": resources["policy"].context_length}
    mode = contract.get("training", {}).get("mode")
    if (any(contract.get(key) != value for key, value in expected.items()) or mode not in ("rl", "ce", "kl-ce")
            or contract.get("initial_policy_sha256") != expected["reference_sha256"]
            or contract.get("reward_signature") != (resources["frozen"]["signature"] if mode == "rl" else None)
            or contract.get("candidate_pool", {}).get("mask_sha256") != rl.stable_hash(resources["eligible"])):
        raise ValueError("RL checkpoint reference/reward/source/action-pool admission differs")
    saved = rl.load_checkpoint(path, contract)
    if saved.get("state", {}).get("updates", 0) < 1:
        raise ValueError("Heldout RL evaluation requires an optimizer-updated checkpoint")
    resources["policy"].load_state_dict(saved["model"], strict=True)
    if any(not torch.isfinite(p).all() for p in resources["policy"].parameters()):
        raise ValueError("Nonfinite restored RL policy")
    return {"checkpoint": _pin(path), "receipt": _pin(path.with_suffix(".sha256.json")),
        "configuration": _pin(configuration), "updates": saved["state"]["updates"], "mode": mode,
        "training_data_identity": "pinned_checkpoint_contract_not_reconstructed_for_heldout_eval"}


def prepare_evaluation(training_bundle, output, *, rl_checkpoint=None, max_rows=256, seed=2026):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new heldout evaluation admission path")
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000 or type(seed) is not int or seed < 0:
        raise ValueError("A bounded validation row count and nonnegative seed are required")
    resources = _evaluation_resources(training_bundle)
    try:
        checkpoint = None if rl_checkpoint is None else _admit_rl_checkpoint(resources, rl_checkpoint)
        value = {"contract": EVAL_CONTRACT, "state": "pinned_requires_evaluation", "training_bundle": _pin(training_bundle),
            "evaluator_source": _pin(Path(__file__)), "cohort": validation_cohort(resources["provider"], max_rows=max_rows, seed=seed),
            "rl_policy": checkpoint, "policy_sha256": rl.model_hash(resources["policy"]),
            "reference_sha256": rl.model_hash(resources["reference"]), "admission": resources["admission"],
            "action_pool": resources["pool"], "baseline_only": checkpoint is None, "test_accessed": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        rl.atomic_json(output, value)
        return value
    finally:
        resources["provider"].close()


def run_evaluation(admission_path, *, device="cpu", max_seconds=120.):
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 3600:
        raise ValueError("A finite heldout evaluation budget in (0,3600] seconds is required")
    value = _json(admission_path)
    fields = {"contract", "state", "training_bundle", "evaluator_source", "cohort", "rl_policy", "policy_sha256",
              "reference_sha256", "admission", "action_pool", "baseline_only", "test_accessed"}
    if (set(value) != fields or value.get("contract") != EVAL_CONTRACT or value.get("state") != "pinned_requires_evaluation"
            or value.get("test_accessed") is not False or value.get("baseline_only") != (value.get("rl_policy") is None)):
        raise ValueError("Invalid explicit heldout evaluation admission")
    if device not in ("cpu", "cuda"):
        raise ValueError("Evaluation device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation explicitly requested but unavailable")
    if _checked_pin(value["evaluator_source"]).resolve() != Path(__file__).resolve():
        raise ValueError("Heldout evaluator source path differs")
    resources = _evaluation_resources(_checked_pin(value["training_bundle"]))
    try:
        if value["rl_policy"] is not None:
            checkpoint = value["rl_policy"]
            for name in ("checkpoint", "receipt", "configuration"):
                _checked_pin(checkpoint[name])
            if _admit_rl_checkpoint(resources, checkpoint["checkpoint"]["path"]) != checkpoint:
                raise ValueError("Pinned RL checkpoint admission changed")
        if (value["admission"] != resources["admission"] or value["action_pool"] != resources["pool"]
                or value["policy_sha256"] != rl.model_hash(resources["policy"])
                or value["reference_sha256"] != rl.model_hash(resources["reference"])):
            raise ValueError("Heldout policy/reference/pool identity changed")
        raw = observed.RawRows.from_pack(resources["provider"], observed.checked_mapping(resources["provider"].metadata))

        def full_seen(rows):
            data = resources["provider"].splits["val"]
            origins = resources["provider"].origin_keys("val")[rows]
            return legacy_rules._full_seen(raw, data["user_id"][rows], origins)

        resources["policy"].to(device)
        resources["reference"].to(device)
        result = evaluate_heldout(resources["policy"], resources["reference"], resources["provider"], value["cohort"],
            eligible_tokens=resources["eligible"], full_seen=full_seen, max_seconds=max_seconds)
        result.update(admission=_pin(admission_path), policy_checkpoint=value["rl_policy"], baseline_only=value["baseline_only"], device=device)
        return result
    finally:
        resources["provider"].close()


def prepare_descriptor(*, sft_dir, provider_dir, data_dir, rm_bundle, rm_artifact, catboost_python,
                       output, max_rows=4096, budget=ActionBudget(), rpc_timeout_seconds=60., planned_rl_output=None,
                       scope_decision=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a fresh bridge descriptor/receipt path")
    budget.validate()
    scope_pin = None
    if scope_decision is not None:
        scope_pin = _pin(scope_decision)
        _scope_authority(scope_pin)
    status = status_receipt(sft_dir, provider_dir, rm_bundle, rm_artifact)
    if scope_pin is not None:
        status["admission_requested"] = {"mode": "scoped_production", "scope_decision": scope_pin,
                                          "scope_id": READY_SCOPE_ID, "full_package_admitted": False}
    if planned_rl_output is not None:
        status["dependencies"]["future_rl_run"] = {"path": Path(planned_rl_output).resolve().as_posix(),
            "state": "planned_missing", "not_created_by_bridge": True}
    if status["state"] == "waiting_artifact":
        output.parent.mkdir(parents=True, exist_ok=True)
        rl.atomic_json(output, status)
        return status
    selected = _json(Path(sft_dir) / "selected.json")
    paths = {"sft_config": Path(sft_dir) / "config.json", "sft_selection": Path(sft_dir) / "selected.json",
             "sft_completion": Path(sft_dir) / "completion.json", "sft_checkpoint": Path(sft_dir) / selected["file"],
             "sft_checkpoint_receipt": (Path(sft_dir) / selected["file"]).with_suffix(".json"),
             "provider_manifest": Path(provider_dir) / "manifest.json", "metadata": Path(data_dir) / "metadata.json",
             "rm_bundle_manifest": Path(rm_bundle) / "manifest.json", "rm_model_manifest": Path(rm_artifact) / "manifest.json"}
    value = {"contract": CONTRACT, "state": "pinned_requires_admission", "policy_scoring": SCORING,
        "ce_anchor": ANCHOR, "files": {name: _pin(path) for name, path in paths.items()},
        "sources": {name: _pin(Path(__file__).with_name(name)) for name in SOURCES}, "max_rows": max_rows,
        "budget": asdict(budget), "catboost_python": Path(catboost_python).resolve(strict=True).as_posix(),
        "rpc_timeout_seconds": rpc_timeout_seconds}
    if scope_pin is not None:
        value["scope_decision"] = scope_pin
    if type(max_rows) is not int or not 1 <= max_rows <= 100000 or not math.isfinite(rpc_timeout_seconds) or not 0 < rpc_timeout_seconds <= 60:
        raise ValueError("Bounded context/RPC controls required")
    output.parent.mkdir(parents=True, exist_ok=True)
    rl.atomic_json(output, value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Pin learned artifacts or publish honest waiting_artifact receipt")
    for name in ("sft-dir", "provider-dir", "data-dir", "rm-bundle", "rm-artifact", "catboost-python", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--max-rows", type=int, default=4096)
    prepare.add_argument("--batch-rows", type=int, default=2)
    prepare.add_argument("--candidate-chunk-size", type=int, default=32)
    prepare.add_argument("--planned-rl-output", type=Path, help="Receipt only; never creates an RL model directory")
    prepare.add_argument("--scope-decision", type=Path, help="Explicit pinned authority for ready scoped production")
    probe = commands.add_parser("probe", help="CPU bounded inference/callback-cost measurement; no optimizer")
    probe.add_argument("--bundle", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True)
    prepare_eval = commands.add_parser("prepare-evaluation", help="Pin canonical heldout cohort and an explicit learned RL checkpoint")
    prepare_eval.add_argument("--bundle", type=Path, required=True)
    prepare_eval.add_argument("--output", type=Path, required=True)
    prepare_eval.add_argument("--rl-checkpoint", type=Path, help="Omit only for an explicit S0 versus S0 baseline")
    prepare_eval.add_argument("--max-rows", type=int, default=256)
    prepare_eval.add_argument("--seed", type=int, default=2026)
    evaluate = commands.add_parser("evaluate", help="Paired full-composite validation plus separate unchanged SFT top128 comparison")
    evaluate.add_argument("--admission", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    evaluate.add_argument("--max-seconds", type=float, default=120.)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.command == "prepare":
        values = vars(args).copy()
        for key in ("command", "batch_rows", "candidate_chunk_size"):
            values.pop(key)
        result = prepare_descriptor(**values, budget=ActionBudget(batch_rows=args.batch_rows,
                                                                 candidate_chunk_size=args.candidate_chunk_size))
    elif args.command == "prepare-evaluation":
        result = prepare_evaluation(args.bundle, args.output, rl_checkpoint=args.rl_checkpoint,
                                    max_rows=args.max_rows, seed=args.seed)
    elif args.command == "evaluate":
        if args.output.exists():
            raise FileExistsError("Choose a fresh heldout evaluation report path")
        result = run_evaluation(args.admission, device=args.device, max_seconds=args.max_seconds)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rl.atomic_json(args.output, result)
    else:
        if args.output.exists():
            raise FileExistsError("Choose a new cost receipt path")
        bundle = build_factory(args.bundle)
        try:
            result = bundle["scorer"].preflight_report
            args.output.parent.mkdir(parents=True, exist_ok=True)
            rl.atomic_json(args.output, result)
        finally:
            bundle["scorer"].close()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    main()
