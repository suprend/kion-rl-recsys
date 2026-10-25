"""Pinned, CPU-only factory for local-pack SFT -> frozen CatBoost -> policy RL.

``prepare`` writes a descriptor only after selected learned SFT and CatBoost
artifacts exist. ``build_factory`` never trains or substitutes random weights.
Observed fit contexts come from the RM bundle; generated prefixes reach GPT only.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_local_pack as local
import kion_pack_train as sft
import kion_policy_rl as rl
import kion_reward_boosting as rm
import kion_reward_data as observed

CONTRACT = "kion-local-pack-rl-v2"
SOURCE_NAMES = (
    "kion_local_pack_rl.py", "kion_policy_rl.py", "kion_pack_train.py", "kion_train.py",
    "kion_full_train.py", "kion_conditioned_train.py", "kion_local_pack.py",
    "kion_local_features.py", "kion_pack_static.py", "kion_pack_people.py",
    "kion_pack_history.py", "kion_pack_dynamic.py", "kion_pack_semantics.py",
    "kion_basic_bundle_features.py", "kion_sex_features.py", "kion_reward_data.py",
    "kion_reward_boosting.py", "hm_job.py",
)
FILE_NAMES = {
    "sft_config", "sft_checkpoint", "sft_selection", "sft_cache_manifest",
    "rm_bundle_manifest", "rm_model_manifest", "metadata", "basic_npz", "basic_manifest",
    "raw.items.csv", "raw.users.csv", "raw.interactions.csv",
}


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pin(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Pinned artifact must be a file: " + path.as_posix())
    return {"path": path.as_posix(), "sha256": rm.file_hash(path)}


def _checked_pin(value, label):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or not isinstance(value["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Invalid absolute SHA256 pin: " + label)
    path = Path(value["path"]).resolve(strict=True)
    if not path.is_file() or rm.file_hash(path) != value["sha256"]:
        raise ValueError("Artifact SHA256 mismatch: " + label)
    return path


def _check_source_paths(value):
    """Validate every path-bearing provenance pin, including series annotation."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["path"])
            if not path.is_absolute():
                raise ValueError("Feature provenance path must be absolute")
            if rm.file_hash(path) != value["sha256"]:
                raise ValueError("Feature source SHA256 mismatch: " + path.as_posix())
        for child in value.values():
            _check_source_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_source_paths(child)


def _frozen_catboost(manifest_path, source_sha256, bundle_sha256):
    """Require the actual frozen learned files before descriptor publication."""
    manifest_path = Path(manifest_path)
    frozen = _json(manifest_path)
    unsigned = {key: child for key, child in frozen.items() if key != "signature"}
    if (frozen.get("signature") != rm.canonical_hash(unsigned) or frozen.get("contract") != rm.CONTRACT
            or frozen.get("backend") != "catboost_cpu" or frozen.get("source_sha256") != source_sha256
            or frozen.get("report", {}).get("bundle_manifest_sha256") != bundle_sha256):
        raise ValueError("Fitted CatBoost model/bundle/source identity mismatch")
    if set(frozen.get("files", {})) != {"model.cbm", "scorer.npz"}:
        raise ValueError("Fitted CatBoost model files are missing")
    for name, digest in frozen["files"].items():
        if rm.file_hash(manifest_path.parent / name) != digest:
            raise ValueError("Frozen CatBoost file checksum mismatch")
    return frozen


def _action_pool(manifest_path, frozen, mode):
    """Fit-count eligibility is fixed before policy sampling or heldout scoring."""
    if mode not in ("supported_fit_only", "allow_unsupported_stress"):
        raise ValueError("Choose supported-fit production or explicit unsupported stress mode")
    path = Path(manifest_path).parent / "scorer.npz"
    # The manifest/file hash was verified by _frozen_catboost. Read only the
    # small ID/count vectors rather than copying the large candidate matrix.
    with np.load(path, allow_pickle=False) as values:
        ids, counts = values["item_ids"], values["counts"]
    if rm.file_hash(path) != frozen["files"]["scorer.npz"]:
        raise ValueError("Frozen RM support arrays changed while reading")
    ids = rm.checked_ids(ids, "frozen support item IDs")
    if (counts.dtype.kind not in "iu" or counts.shape != ids.shape or np.any(counts < 0)
            or not len(ids) or len(np.unique(ids)) != len(ids)):
        raise ValueError("Frozen RM target-support counts must be aligned nonnegative integers")
    mapping = {str(int(item)): token for token, item in enumerate(ids, 3)}
    if rm.canonical_hash(mapping) != frozen["catalog"]["mapping_sha256"]:
        raise ValueError("Frozen RM support counts and full action mapping differ")
    supported = counts > 0
    eligible = np.zeros(len(ids) + 3, bool)
    eligible[3:] = True if mode == "allow_unsupported_stress" else supported
    if not eligible.any():
        raise ValueError("RM has no supported production candidates")
    cold = ~supported
    receipt = {"mode": mode, "support_definition": "frozen_RM_fit_target_count_gt_zero",
               "reward_signature": frozen["signature"], "serving_source_sha256": frozen["source_sha256"],
               "scorer_arrays_sha256": frozen["files"]["scorer.npz"],
               "mapping_sha256": frozen["catalog"]["mapping_sha256"],
               "fit_target_counts_sha256": rm.array_hash(counts),
               "eligible_tokens_sha256": rm.array_hash(eligible),
               "catalog_items": len(ids), "supported_fit_items": int(supported.sum()),
               "unsupported_fit_items": int(cold.sum()), "eligible_items": int(eligible.sum()),
               "excluded_catalog_items": int(len(ids) - eligible.sum()),
               "excluded_catalog_fraction": float((len(ids) - eligible.sum()) / len(ids)),
               "unsupported_catalog_fraction": float(cold.mean()),
               "unsupported_tokens": (np.flatnonzero(cold) + 3).tolist(),
               "unsupported_item_ids": ids[cold].tolist(),
               "unsupported_items_allowed": mode == "allow_unsupported_stress",
               "cold_coverage": {}}
    # These frozen, observed diagnostics quantify excluded coverage. They do not
    # set the support threshold, transform rewards, or select policy actions.
    for role in ("select", "calibration"):
        report = frozen.get("report", {}).get("selected", {}).get(role, {})
        total = report.get("metrics", {}).get("n")
        segment = report.get("segments", {}).get("item_zero_support", {})
        n = segment.get("n")
        if type(total) is not int or type(n) is not int or not 0 <= n <= total or total < 1:
            raise ValueError("Frozen RM lacks observed cold coverage diagnostics: " + role)
        receipt["cold_coverage"][role] = {
            "diagnostic_observed_rows": total, "unsupported_target_rows": n,
            "unsupported_target_fraction": n / total,
            "excluded_target_rows": n if mode == "supported_fit_only" else 0,
            "unsupported_mean_prediction": segment.get("mean_prediction"),
            "unsupported_mean_observed": segment.get("mean_observed"),
            "interpretation": "frozen_observed_RM_diagnostic_not_counterfactual_or_online_evidence"}
    eligible.flags.writeable = False
    return eligible, receipt


def _producer_sources(bundle_manifest):
    """Immutable data producer pins are distinct from the current RM server."""
    pins = {"rm_data_builder_sha256": bundle_manifest.get("builder_source_sha256"),
            "rm_data_boosting_sha256": bundle_manifest.get("catalog", {}).get("provenance", {}).get("source_pins", {}).get("boosting")}
    if any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in pins.values()):
        raise ValueError("Immutable RM data producer source pins are missing")
    return pins


def prepare_descriptor(*, sft_dir, rm_bundle, rm_artifact, raw_dir, metadata,
                       basic_npz, catboost_python, output, max_rows=4096,
                       rollout_steps=1, rpc_timeout=60.0, allow_unsupported_stress=False):
    """Pin existing artifacts; no model initialization, data build, or training."""
    output = Path(output)
    if type(allow_unsupported_stress) is not bool:
        raise ValueError("Unsupported stress mode requires an explicit boolean flag")
    if output.exists():
        raise FileExistsError("Choose a new descriptor path")
    sft_dir, rm_bundle, rm_artifact, raw_dir = map(Path, (sft_dir, rm_bundle, rm_artifact, raw_dir))
    config = _json(sft_dir / "config.json")
    selected = _json(sft_dir / "selected.json")
    name = selected.get("file")
    if not isinstance(name, str) or Path(name).name != name or not re.fullmatch(r"selected-epoch[0-9]+\.pt", name):
        raise ValueError("SFT requires its published selected epoch checkpoint")
    feature_dir = Path(config["feature_dir"])
    cache = _json(feature_dir / "manifest.json")
    paths = {"sft_config": sft_dir / "config.json", "sft_checkpoint": sft_dir / name,
             "sft_selection": sft_dir / "selected.json", "sft_cache_manifest": feature_dir / "manifest.json",
             "rm_bundle_manifest": rm_bundle / "manifest.json", "rm_model_manifest": rm_artifact / "manifest.json",
             "metadata": Path(metadata), "basic_npz": Path(basic_npz),
             "basic_manifest": Path(basic_npz).with_name("manifest.json")}
    paths.update({"raw." + name: raw_dir / name for name in ("items.csv", "users.csv", "interactions.csv")})
    value = {"contract": CONTRACT, "files": {name: _pin(path) for name, path in paths.items()},
             "sources": {name: _pin(Path(__file__).with_name(name)) for name in SOURCE_NAMES},
             "feature_cache_outputs": {name: _pin(feature_dir / name) for name in cache["outputs"]},
             "contexts": {"role": "fit", "max_rows": max_rows, "rollout_steps": rollout_steps},
             "runtime": {"catboost_python": Path(catboost_python).resolve(strict=True).as_posix(),
                         "rpc_timeout_seconds": rpc_timeout}}
    value["producer_sources"] = _producer_sources(_json(paths["rm_bundle_manifest"]))
    frozen = _frozen_catboost(paths["rm_model_manifest"], value["sources"]["kion_reward_boosting.py"]["sha256"],
                              value["files"]["rm_bundle_manifest"]["sha256"])
    _, value["action_pool"] = _action_pool(paths["rm_model_manifest"], frozen,
                                           "allow_unsupported_stress" if allow_unsupported_stress else "supported_fit_only")
    _validate_descriptor(value)
    # Early admission catches missing learned artifacts without loading Torch/CatBoost.
    if selected.get("sha256") != value["files"]["sft_checkpoint"]["sha256"]:
        raise ValueError("SFT selected checkpoint SHA mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    rm.write_json(output, value)
    return value


def _validate_descriptor(value):
    if (not isinstance(value, dict) or set(value) != {"contract", "files", "sources", "feature_cache_outputs", "contexts", "runtime", "action_pool", "producer_sources"}
            or value["contract"] != CONTRACT or set(value["files"]) != FILE_NAMES
            or set(value["sources"]) != set(SOURCE_NAMES)):
        raise ValueError("Unknown or incomplete local-pack RL descriptor")
    if (not isinstance(value["action_pool"], dict)
            or value["action_pool"].get("mode") not in ("supported_fit_only", "allow_unsupported_stress")):
        raise ValueError("A pinned production/stress action pool is required")
    if (not isinstance(value["producer_sources"], dict)
            or set(value["producer_sources"]) != {"rm_data_builder_sha256", "rm_data_boosting_sha256"}
            or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                   for digest in value["producer_sources"].values())):
        raise ValueError("Explicit immutable RM producer source pins are required")
    contexts, runtime = value["contexts"], value["runtime"]
    if (set(contexts) != {"role", "max_rows", "rollout_steps"} or contexts["role"] != "fit"
            or any(type(contexts[k]) is not int or not 1 <= contexts[k] <= limit
                   for k, limit in (("max_rows", 100000), ("rollout_steps", 5)))):
        raise ValueError("Bounded observed pre-Aug2 fit contexts are required")
    if (set(runtime) != {"catboost_python", "rpc_timeout_seconds"}
            or not isinstance(runtime["catboost_python"], str) or not Path(runtime["catboost_python"]).is_absolute()
            or type(runtime["rpc_timeout_seconds"]) not in (int, float)
            or not math.isfinite(runtime["rpc_timeout_seconds"]) or not 0 < runtime["rpc_timeout_seconds"] <= 300):
        raise ValueError("Provide an explicit local CatBoost interpreter and bounded RPC timeout")
    if not Path(runtime["catboost_python"]).is_file():
        raise FileNotFoundError("CatBoost interpreter is missing")
    paths = {name: _checked_pin(pin, name) for name, pin in value["files"].items()}
    for name, pin in value["sources"].items():
        if _checked_pin(pin, name) != Path(__file__).with_name(name).resolve():
            raise ValueError("Factory source must be the pinned common implementation: " + name)
    if (not isinstance(value["feature_cache_outputs"], dict)
            or not {"item_features.npy", "train_user_features.npy", "val_user_features.npy"} <= set(value["feature_cache_outputs"])):
        raise ValueError("Incomplete SFT feature cache pins")
    for name, pin in value["feature_cache_outputs"].items():
        if Path(name).name != name or _checked_pin(pin, name) != paths["sft_cache_manifest"].parent / name:
            raise ValueError("SFT cache file path differs from manifest")
    raw = {paths["raw." + name].parent for name in ("items.csv", "users.csv", "interactions.csv")}
    if len(raw) != 1 or any(paths["raw." + name].name != name for name in ("items.csv", "users.csv", "interactions.csv")):
        raise ValueError("Raw feature files must share their original directory")
    return paths


def _sft_model(value, paths):
    config, cache, selected = map(_json, (paths["sft_config"], paths["sft_cache_manifest"], paths["sft_selection"]))
    contract, architecture = config["contract"], config["contract"]["architecture"]
    if (architecture.get("feature_mode") != "local-pack" or cache.get("contract") != local.CONTRACT
            or cache.get("state") != "completed" or cache.get("cache_dtype") != "float16"
            or cache.get("batch_transfer_dtype") != "float32"
            or contract.get("feature_manifest_sha256") != value["files"]["sft_cache_manifest"]["sha256"]):
        raise ValueError("SFT must bind the completed float16 -> float32 local-pack cache")
    columns = cache["feature_columns"]
    if (architecture["feature_columns"] != columns or architecture["schema_hash"] != cache["schema_hash"]
            or cache["schema_hash"] != rm.canonical_hash(columns)):
        raise ValueError("SFT cache/architecture schema mismatch")
    if (set(cache["outputs"]) != set(value["feature_cache_outputs"])
            or contract["features_sha256"] != {name: pin["sha256"] for name, pin in value["feature_cache_outputs"].items()}
            or any(cache["outputs"][name]["sha256"] != pin["sha256"] for name, pin in value["feature_cache_outputs"].items())):
        raise ValueError("SFT feature cache checksum binding mismatch")
    if (contract["inputs_sha256"]["metadata.json"] != value["files"]["metadata"]["sha256"]
            or any(cache["inputs"][name]["sha256"] != digest for name, digest in contract["inputs_sha256"].items())
            or any(value["sources"][name]["sha256"] != digest for name, digest in contract["sources_sha256"].items())):
        raise ValueError("SFT metadata/common source binding mismatch")
    _check_source_paths(cache.get("provenance", {}))
    if (selected.get("file") != paths["sft_checkpoint"].name
            or selected.get("sha256") != value["files"]["sft_checkpoint"]["sha256"]
            or type(selected.get("epoch")) is not int or selected["epoch"] < 1):
        raise ValueError("SFT selected checkpoint receipt mismatch")
    saved = torch.load(paths["sft_checkpoint"], map_location="cpu", weights_only=False)
    if (not isinstance(saved, dict) or rm.canonical_hash(saved["config"]["contract"]) != rm.canonical_hash(contract)
            or type(saved["state"].get("updates")) is not int or saved["state"]["updates"] < 1
            or saved["state"].get("selected_epoch") != selected["epoch"]):
        raise ValueError("SFT requires selected learned weights from the pinned config")
    items = np.load(paths["sft_cache_manifest"].parent / "item_features.npy", allow_pickle=False)
    with torch.random.fork_rng(devices=[]):
        model = sft.NextItemGPT.from_config(config, items)
    model.load_state_dict(saved["state_dict"], strict=True)
    model = model.cpu().float()
    if (not np.array_equal(model.item_features.detach().numpy(), items)
            or any(not torch.isfinite(tensor).all() for tensor in model.state_dict().values() if tensor.is_floating_point())):
        raise ValueError("SFT has non-finite tensors or a different frozen item feature matrix")
    model.requires_grad_(True)
    if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("RL policy parameters must be finite trainable CPU FP32")
    return model, config, cache


def _full_seen(raw, users, cutoffs):
    """Full actual known-item history, including observations outside GPT H63."""
    result = []
    for user, cutoff in zip(users, cutoffs):
        start = np.searchsorted(raw.users, user, side="left")
        end = np.searchsorted(raw.keys, int(user) * observed.MAX_ORDINAL + int(cutoff), side="left")
        result.append(tuple(int(token) for token in np.unique(raw.tokens[start:end]) if token >= 3))
    return tuple(result)


class CandidateProvider:
    """The only candidate-feature route: original row index + new item token."""
    def __init__(self, pack, users, origins, signature):
        self.pack = pack
        self.users, self.origins = users.copy(), origins.copy()
        self.users.flags.writeable = self.origins.flags.writeable = False
        self.signature = signature

    def __call__(self, indices, tokens):
        indices, tokens = np.asarray(indices), np.asarray(tokens)
        if (indices.dtype != np.int64 or tokens.dtype != np.int64 or indices.ndim != 1
                or tokens.shape != indices.shape or np.any(indices < 0) or np.any(indices >= len(self.users))
                or np.any(tokens < 3)):
            raise ValueError("Dynamic candidate requests require original row indices and known item tokens")
        dates = self.origins[indices]
        values = self.pack.candidate_features(tokens, dates, history_cutoffs=dates, user_ids=self.users[indices])
        if (values.dtype != np.float32 or values.shape != (len(indices), len(self.pack.candidate_columns))
                or not np.isfinite(values).all()):
            raise ValueError("Invalid fixed-origin candidate feature block")
        return values


def build_factory(descriptor_path):
    """Return the exact policy/reference/data/scorer contract, with no GPU use."""
    value = _json(descriptor_path)
    paths = _validate_descriptor(value)
    model, config, cache = _sft_model(value, paths)
    roles, catalog, bundle_manifest = rm.load_bundle(paths["rm_bundle_manifest"].parent)
    if value["producer_sources"] != _producer_sources(bundle_manifest):
        raise ValueError("Immutable RM data producer source identity mismatch")
    if (bundle_manifest.get("builder_source_sha256") != value["sources"]["kion_reward_data.py"]["sha256"]
            or catalog.provenance.get("candidate_features_dynamic") is not True):
        raise ValueError("RM bundle requires the pinned dynamic observed-context builder")
    _check_source_paths(catalog.provenance)
    source_keys = {"metadata": "metadata", "basic_npz": "basic_npz", "builder": "kion_reward_data.py",
                   "items": "raw.items.csv",
                   "users": "raw.users.csv", "interactions": "raw.interactions.csv"}
    for key, name in source_keys.items():
        pin = value["sources" if name in SOURCE_NAMES else "files"][name]
        if catalog.provenance["source_pins"].get(key) != pin["sha256"]:
            raise ValueError("RM bundle common source binding mismatch: " + key)
    frozen = _frozen_catboost(paths["rm_model_manifest"], value["sources"]["kion_reward_boosting.py"]["sha256"],
                              value["files"]["rm_bundle_manifest"]["sha256"])
    signature = frozen["signature"]
    if frozen.get("catalog") != catalog.describe():
        raise ValueError("Fitted CatBoost model/bundle schema identity mismatch")
    eligible_tokens, action_pool = _action_pool(paths["rm_model_manifest"], frozen, value["action_pool"]["mode"])
    if value["action_pool"] != action_pool:
        raise ValueError("Pinned action pool/support/coverage receipt mismatch")
    metadata = _json(paths["metadata"])
    mapping = observed.checked_mapping(metadata)
    if (catalog.mapping != {str(key): token for key, token in mapping.items()}
            or model.backbone.config.vocab_size != len(mapping) + 3):
        raise ValueError("Full policy/RM mapping identity mismatch")
    pack = local.FeaturePack.build(paths["raw.items.csv"].parent, paths["metadata"], paths["basic_npz"],
                                   fit_cutoff="2021-08-02", include_relative=True)
    _check_source_paths(pack.provenance)
    if (pack.fit_cutoff != "2021-08-02" or pack.user_columns != catalog.context_names
            or pack.candidate_columns != catalog.item_names or pack.item_columns != cache["feature_columns"]["item"]
            or pack.user_columns != cache["feature_columns"]["user"]
            or not np.array_equal(pack.item_features, model.item_features.detach().numpy())):
        raise ValueError("Common feature facade/SFT/RM schema or item feature mismatch")
    raw = observed.RawRows.from_pack(pack, mapping)
    n = min(value["contexts"]["max_rows"], len(roles["fit"]["user_id"]))
    rows = roles["fit"]
    users, origins = rows["user_id"][:n].copy(), rows["history_cutoff"][:n].copy()
    sidecars = bundle_manifest.get("sidecars", {})
    if not {"fit.x.npy", "fit.lengths.npy"} <= set(sidecars):
        raise ValueError("Prepared observed RM policy contexts are missing")
    root = paths["rm_bundle_manifest"].parent
    x = np.load(root / "fit.x.npy", allow_pickle=False)[:n].copy()
    lengths = np.load(root / "fit.lengths.npy", allow_pickle=False)[:n].copy()
    if np.any(origins >= date(2021, 8, 2).toordinal()):
        raise ValueError("RL contexts must be observed strictly before Aug2")
    expected_x, expected_lengths, _ = raw.history_batch(users, origins)
    if not np.array_equal(x, expected_x) or not np.array_equal(lengths, expected_lengths):
        raise ValueError("RM context sidecar differs from strict raw observed history")
    numeric = np.asarray(rows["numeric"][:n], dtype=np.float32).copy()
    # Chunked verification avoids an extra large N x U feature allocation.
    for begin in range(0, n, 256):
        sl = slice(begin, min(n, begin + 256))
        actual = pack.user_features(users[sl], origins[sl], x[sl], lengths[sl], history_cutoffs=origins[sl])
        if not np.array_equal(actual, numeric[sl]):
            raise ValueError("Prepared full-FP32 RM context differs from common feature facade")
    if not np.isfinite(numeric).all() or np.any(np.abs(numeric) > np.finfo(np.float16).max):
        raise ValueError("Policy user features cannot be represented safely by the SFT float16 cache")
    policy_features = numeric.astype(np.float16).astype(np.float32)
    contexts = tuple({"user_id": int(users[row]), "numeric": numeric[row],
                      "categorical": tuple(str(c) for c in rows["categorical"][row]),
                      "schema_signature": catalog.schema_signature} for row in range(n))
    provider_signature = rl.stable_hash({"descriptor": value, "users": users, "origins": origins,
                                        "candidate_schema": catalog.schema_signature})
    provider = CandidateProvider(pack, users, origins, provider_signature)
    targets = np.fromiter((mapping[int(item)] for item in rows["item_id"][:n]), np.int64, count=n)
    if not eligible_tokens[targets].all():
        raise ValueError("Observed fit CE anchors must belong to the declared action pool")
    data = rl.RLData(x=x, lengths=lengths, seen_tokens=_full_seen(raw, users, origins),
                     token_to_item={token: item for item, token in mapping.items()}, reward_contexts=contexts,
                     y=targets, user_features=policy_features, eligible_tokens=eligible_tokens,
                     candidate_features_provider=provider,
                     candidate_features_signature=provider_signature,
                     provenance={"contract": CONTRACT, "descriptor_sha256": rm.file_hash(descriptor_path),
                                 "pins": value, "observed_role": "fit", "rows": n,
                                 "action_pool": action_pool,
                                 "reward_state": "fixed_observed_s0", "test_accessed": False,
                                 "policy_user_rounding": "float32(float16(original_float32))"})
    for array in (x, lengths, targets, numeric, policy_features):
        array.flags.writeable = False
    for context in contexts:
        context["numeric"].flags.writeable = False
    data.validate(model.backbone.config.vocab_size, config["contract"]["architecture"]["context_length"],
                  value["contexts"]["rollout_steps"])
    policy = rl.PolicyAdapter(model)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    command = [value["runtime"]["catboost_python"], Path(__file__).with_name("kion_reward_boosting.py").as_posix(),
               "serve", "--artifact", paths["rm_model_manifest"].parent.as_posix(),
               "--expected-mapping-sha256", catalog.mapping_sha256]
    scorer = rl.ProcessRewardScorer(command, signature, cwd=Path(__file__).resolve().parents[1],
                                    timeout=value["runtime"]["rpc_timeout_seconds"])
    if any(scorer.metadata.get(key) != expected for key, expected in (
            ("schema_signature", catalog.schema_signature), ("context_numeric_dim", len(catalog.context_names)),
            ("candidate_numeric_dim", len(catalog.item_names)))):
        scorer.close()
        raise ValueError("CPU reward worker schema handshake mismatch")
    return {"policy": policy, "reference": reference, "data": data, "scorer": scorer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Pin existing learned artifacts; does not train")
    for name in ("sft-dir", "rm-bundle", "rm-artifact", "raw-dir", "metadata", "basic-npz", "catboost-python", "output"):
        prepare.add_argument("--" + name, type=Path, required=True)
    prepare.add_argument("--max-rows", type=int, default=4096)
    prepare.add_argument("--rollout-steps", type=int, default=1)
    prepare.add_argument("--rpc-timeout", type=float, default=60.)
    prepare.add_argument("--allow-unsupported-stress", action="store_true",
                         help="Create a separate stress descriptor allowing zero-fit-support candidates")
    check = commands.add_parser("check", help="Verify pinned factory on CPU, then close the scorer")
    check.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        values = vars(args).copy()
        values.pop("command")
        value = prepare_descriptor(**values)
        print(json.dumps({"descriptor": args.output.resolve().as_posix(), "contract": value["contract"],
                          "action_pool": value["action_pool"]}))
    else:
        bundle = build_factory(args.bundle)
        try:
            print(json.dumps({"state": "ready", "rows": len(bundle["data"].x),
                              "reward_signature": bundle["scorer"].signature, "device": "cpu",
                              "action_pool": bundle["data"].provenance["action_pool"],
                              "test_accessed": False}))
        finally:
            bundle["scorer"].close()


if __name__ == "__main__":
    main()
