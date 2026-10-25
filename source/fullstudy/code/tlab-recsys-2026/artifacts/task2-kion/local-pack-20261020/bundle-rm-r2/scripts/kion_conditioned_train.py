"""Isolated, resumable KION singleton conditioning; never reads the test split.

Age is a named seven-category snapshot, added only at the existing BOS. Content
features affect history embeddings only; candidates always use the tied ID head.
Legacy ordering, effective batches, CE denominator and cosine are reused exactly.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_full_train as legacy
from kion_train import NextItemGPT, BOS, PAD, FIRST_ITEM, known_targets, last_real_hidden

AGE_CATEGORIES = ("__UNKNOWN__", "age_18_24", "age_25_34", "age_35_44",
                  "age_45_54", "age_55_64", "age_65_inf")
SEX_CATEGORIES = ("sex_male", "sex_female", "sex_unknown")
PREPARED_SEX_CATEGORIES = ("__UNKNOWN__", "Ж", "М")
SEX_HELPER_SHA256 = "e5f992093b53839d425ccd436dba0404862ad912b795a390d60ff6c9a2576de1"
MODES = ("none", "age", "age-knownness", "content-type", "release-year", "sex", "basic-bundle")
EXECUTION_MODES = ("legacy", "optimized")
BATCH_TRANSFERS = ("direct", "packed", "pinned-packed")
BUNDLE_GROUP_NAMES = ("age", "sex", "genres", "content_type", "release_year", "countries", "age_rating")
ROW_KEYS = ("user_id", "target_item_id", "target_date")
sha256, write_json = legacy.sha256, legacy.write_json
groups, epoch_order = legacy.groups, legacy.epoch_order


def validate_age(age, count):
    if age.dtype != np.float32 or age.shape != (count, len(AGE_CATEGORIES)):
        raise ValueError("Age must be float32 N x 7 in the named categorical layout")
    if not np.isfinite(age).all() or np.any((age != 0) & (age != 1)) or np.any(age.sum(1) != 1):
        raise ValueError("Age must be finite, binary, exactly one-hot including unknown")
    return age


def age_knownness(age):
    """Same adapter and parameter budget; known categories collapse to column 1."""
    result = np.zeros_like(age)
    result[:, 0] = age[:, 0]
    result[:, 1] = 1 - age[:, 0]
    return result


def validate_sex(sex, count):
    if sex.dtype != np.float32 or sex.shape != (count, len(SEX_CATEGORIES)):
        raise ValueError("Sex must be float32 N x 3 in the named categorical layout")
    if not np.isfinite(sex).all() or np.any((sex != 0) & (sex != 1)) or np.any(sex.sum(1) != 1):
        raise ValueError("Sex must be finite, binary, exactly one-hot including unknown")
    return sex


def validate_item_features(features, vocab, mode):
    if features.dtype != np.float32 or features.shape != (vocab, 2):
        raise ValueError("Item features must be float32 vocab x 2")
    if not np.isfinite(features).all() or np.any(features[:FIRST_ITEM]):
        raise ValueError("Item features must be finite and special rows must be zero")
    if mode == "content-type":
        if np.any((features != 0) & (features != 1)) or np.any(features.sum(1) > 1):
            raise ValueError("Content type must be [film, series] with implicit unknown")
    elif mode == "release-year":
        missing = features[FIRST_ITEM:, 1]
        if np.any((missing != 0) & (missing != 1)) or np.any(features[FIRST_ITEM:, 0][missing == 1] != 0):
            raise ValueError("Release year must contain [train z-score, binary missing]")
    else:
        raise ValueError("Unknown item conditioning mode")
    return features


def load_data(path, vocab, context, require_keys=False, unique_keys=True):
    data = legacy.load_data(path, vocab, context)
    x = data["x"]
    if np.any(x[:, 0] != BOS) or np.any(x[:, 1:] == BOS):
        raise ValueError("Exactly one BOS must occupy the existing position zero")
    if require_keys:
        with np.load(path, allow_pickle=False) as source:
            for name in ROW_KEYS:
                value = source[name]
                if value.dtype != np.int64 or value.shape != data["y"].shape:
                    raise ValueError("Validation row keys must be aligned int64 vectors")
                data[name] = value
        if unique_keys:
            keys = np.rec.fromarrays([data[name] for name in ROW_KEYS], names=ROW_KEYS)
            if len(np.unique(keys)) != len(keys):
                raise ValueError("Validation row keys must be one-to-one")
    return data


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def array_hash(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def normalized_drop_groups(names):
    if len(names) != len(set(names)):
        raise ValueError("Duplicate drop feature group")
    if set(names) - set(BUNDLE_GROUP_NAMES):
        raise ValueError("Unknown drop feature group")
    return tuple(name for name in BUNDLE_GROUP_NAMES if name in names)


def validate_bundle_manifest(path, feature_path, metadata, inputs, train, val, item_features):
    # This helper is loaded only by the new mode; legacy models keep their old
    # imports, feature budgets and random-number consumption.
    import kion_basic_bundle_features as builder
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or manifest.get("builder_contract") != "kion-basic-bundle-v1":
        raise ValueError("Unsupported basic-bundle manifest contract")
    if manifest.get("profile_availability") != "static_snapshot_unverified":
        raise ValueError("Bundle manifest must declare snapshot availability")
    categories = metadata.get("user_categories", {})
    if (manifest.get("age_categories") != list(AGE_CATEGORIES) or categories.get("age") != list(AGE_CATEGORIES)
            or manifest.get("sex_categories") != list(SEX_CATEGORIES)
            or manifest.get("prepared_sex_categories") != list(PREPARED_SEX_CATEGORIES)
            or categories.get("sex") != list(PREPARED_SEX_CATEGORIES)
            or list(categories) != ["age", "income", "sex", "kids_flg"]
            or metadata.get("user_feature_dim") != sum(map(len, categories.values()))):
        raise ValueError("Bundle named profile category layout mismatch")
    if (manifest.get("group_order") != list(BUNDLE_GROUP_NAMES)
            or manifest.get("metadata_experiment") != "retrospective"
            or manifest.get("item_metadata_availability") != "static_snapshot_unverified"
            or manifest.get("age_rating_review") != {"21.0": "unresolved raw category; retained separately without remapping"}
            or manifest.get("cutoffs") != {"train": "<2021-08-09", "validation": "2021-08-09..2021-08-15", "fit": "2021-08-09"}):
        raise ValueError("Bundle named group/snapshot/fit declaration mismatch")
    vocabulary = manifest["vocabulary"]
    if (vocabulary["size"] != metadata["vocab_size"] or vocabulary["special_tokens"] != {"PAD": 0, "BOS": 1, "UNK": 2}
            or vocabulary["sha256"] != canonical_hash(metadata["item_id_to_token"])):
        raise ValueError("Bundle vocabulary mapping mismatch")
    for split, data in (("train", train), ("val", val)):
        binding = manifest["inputs"][split]
        key_hash = canonical_hash({name: array_hash(data[name]) for name in ROW_KEYS})
        if (binding["sha256"] != inputs[split + ".npz"] or binding["rows"] != len(data["y"])
                or binding["user_id_sha256"] != array_hash(data["user_id"]) or binding["row_keys_sha256"] != key_hash):
            raise ValueError("Bundle row alignment/input binding mismatch")
    if manifest["inputs"]["metadata"]["sha256"] != inputs["metadata.json"]:
        raise ValueError("Bundle metadata binding mismatch")
    feature = manifest["features"]
    if feature.get("file") != "features.npz" or feature["sha256"] != sha256(feature_path):
        raise ValueError("Bundle feature file hash mismatch")
    arrays = {"train_user_features": train["bundle_user"], "val_user_features": val["bundle_user"],
              "item_features": item_features}
    if set(feature["arrays"]) != set(arrays):
        raise ValueError("Bundle array declaration mismatch")
    for name, value in arrays.items():
        declaration = feature["arrays"][name]
        if (declaration["shape"] != list(value.shape) or declaration["dtype"] != "float32"
                or value.dtype != np.float32 or declaration["sha256"] != array_hash(value)):
            raise ValueError("Bundle array layout/dtype/row-order hash mismatch")
    for split, data in (("train", train), ("val", val)):
        if data["bundle_user"].shape != (len(data["y"]), 10):
            raise ValueError("Bundle profile must be aligned float32 N x 10")
        builder.validate_bundle_layout(data["bundle_user"], item_features, manifest["groups"])
    if item_features.shape[0] != metadata["vocab_size"]:
        raise ValueError("Bundle item vocabulary shape mismatch")
    fit = manifest["fit"]
    item_ids = sorted(map(int, metadata["item_id_to_token"]))
    for name in ("genres", "countries"):
        declaration = fit[name]
        if (declaration.get("role") != "SFT" or declaration.get("cutoff") != "2021-08-09"
                or declaration.get("scope") != "unique training-vocabulary item IDs"
                or declaration.get("item_count") != len(item_ids)
                or declaration.get("item_ids_sha256") != canonical_hash(item_ids)
                or declaration.get("categories_sha256") != canonical_hash(manifest["groups"][name]["columns"])):
            raise ValueError("Bundle train-only category fit declaration mismatch")
    year_fit, year_group = fit["release_year"], manifest["groups"]["release_year"]
    year_known = item_features[:, year_group["start"] + 1] == 0
    year_ids = sorted(int(item_id) for item_id, token in metadata["item_id_to_token"].items() if year_known[token])
    if (year_fit.get("role") != "SFT" or year_fit.get("cutoff") != "2021-08-09"
            or year_fit.get("scope") != "unique training-vocabulary item IDs with valid known year"
            or year_fit.get("known_count") != len(year_ids) or not year_ids
            or year_fit.get("item_ids_sha256") != canonical_hash(year_ids)
            or not all(isinstance(year_fit.get(name), (int, float)) and math.isfinite(year_fit[name])
                       for name in ("mean", "population_std", "scale"))
            or year_fit["population_std"] < 0 or year_fit["scale"] != (year_fit["population_std"] or 1)
            or year_fit.get("zero_std_fallback") != 1.0):
        raise ValueError("Invalid bundle train-only release-year transform contract")
    # Bind both the declared transformation implementation and its raw sources.
    # Never follow arbitrary split names supplied by a manifest (in particular
    # test); the only input roles accepted here are the builder's fixed roles.
    source_inputs = manifest["inputs"]
    required = {"users", "items", "metadata", "train", "val", "builder", "local_helper", "sex_helper"}
    if not required <= set(source_inputs) or set(source_inputs) - required - {"raw_manifest"}:
        raise ValueError("Bundle source input roles mismatch")
    import kion_sex_features as sex_builder
    for name, module in (("builder", builder), ("local_helper", builder.local), ("sex_helper", sex_builder)):
        source = Path(module.__file__).resolve()
        if (source_inputs[name]["sha256"] != sha256(source)
                or name in builder.HELPER_PINS and source_inputs[name]["sha256"] != builder.HELPER_PINS[name]):
            raise ValueError("Bundle builder/helper source hash mismatch")
    for name in ("users", "items", "raw_manifest"):
        if name in source_inputs:
            source = Path(source_inputs[name]["path"])
            expected_name = {"users": "users.csv", "items": "items.csv", "raw_manifest": "manifest.json"}[name]
            if source.name != expected_name or not source.is_file() or sha256(source) != source_inputs[name]["sha256"]:
                raise ValueError("Bundle raw source hash mismatch")
    return manifest


def validate_feature_manifest(path, feature_path, mode, metadata, inputs, train, val, item_features):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    builder_contract = "kion-sex-sidecars-v1" if mode == "sex" else "kion-local-sidecars-v1"
    if manifest.get("format_version") != 1 or manifest.get("builder_contract") != builder_contract:
        raise ValueError("Unsupported feature manifest contract")
    if manifest.get("profile_availability") != "static_snapshot_unverified":
        raise ValueError("Feature manifest must declare snapshot availability")
    vocabulary = manifest["vocabulary"]
    if (vocabulary["size"] != metadata["vocab_size"] or vocabulary["special_tokens"] != {"PAD": 0, "BOS": 1, "UNK": 2}
            or vocabulary["sha256"] != canonical_hash(metadata["item_id_to_token"])):
        raise ValueError("Feature vocabulary mapping mismatch")
    for split, data in (("train", train), ("val", val)):
        binding = manifest["inputs"][split]
        key_hash = canonical_hash({name: array_hash(data[name]) for name in ROW_KEYS})
        if (binding["sha256"] != inputs[split + ".npz"] or binding["rows"] != len(data["y"])
                or binding["user_id_sha256"] != array_hash(data["user_id"]) or binding["row_keys_sha256"] != key_hash):
            raise ValueError("Feature row alignment/input binding mismatch")
    if manifest["inputs"]["metadata"]["sha256"] != inputs["metadata.json"]:
        raise ValueError("Feature metadata binding mismatch")
    key = "age" if mode.startswith("age") else mode.replace("-", "_")
    feature = manifest["features"][key]
    if feature["sha256"] != sha256(feature_path):
        raise ValueError("Feature sidecar file hash mismatch")
    if mode == "sex":
        categories = metadata.get("user_categories", {})
        if (manifest.get("sex_categories") != list(SEX_CATEGORIES) or feature["categories"] != list(SEX_CATEGORIES)
                or manifest.get("prepared_sex_categories") != list(PREPARED_SEX_CATEGORIES)
                or categories.get("sex") != list(PREPARED_SEX_CATEGORIES)
                or list(categories) != ["age", "income", "sex", "kids_flg"]
                or metadata.get("user_feature_dim") != sum(map(len, categories.values()))):
            raise ValueError("Named sex category layout mismatch")
        if (set(manifest["features"]) != {"sex"} or manifest.get("metadata_experiment") != "retrospective"
                or manifest.get("safety", {}).get("feature_fields") != ["users.sex"]
                or manifest["inputs"]["helper"]["sha256"] != SEX_HELPER_SHA256):
            raise ValueError("Sex-only builder/source contract mismatch")
        for split, data in (("train", train), ("val", val)):
            declaration = feature["arrays"][split + "_user_sex"]
            if (declaration["shape"] != list(data["sex"].shape) or declaration["dtype"] != "float32"
                    or declaration["sha256"] != array_hash(data["sex"])):
                raise ValueError("Sex array/row-order hash mismatch")
    elif mode.startswith("age"):
        if (manifest.get("age_categories") != list(AGE_CATEGORIES) or feature["categories"] != list(AGE_CATEGORIES)
                or metadata.get("user_categories", {}).get("age") != list(AGE_CATEGORIES)):
            raise ValueError("Named age category layout mismatch")
        for split, data in (("train", train), ("val", val)):
            declaration = feature["arrays"][split + "_user_age"]
            if (declaration["shape"] != list(data["age"].shape) or declaration["dtype"] != "float32"
                    or declaration["sha256"] != array_hash(data["age"])):
                raise ValueError("Age array/row-order hash mismatch")
    else:
        columns = ["film", "series"] if mode == "content-type" else ["normalized_year", "missing"]
        if (feature["shape"] != list(item_features.shape) or feature["dtype"] != "float32"
                or feature["columns"] != columns or feature["array_sha256"] != array_hash(item_features)):
            raise ValueError("Named item feature layout/hash mismatch")
        if mode == "release-year":
            fit = feature["fit"]
            if (fit["role"] != "SFT" or fit["cutoff"] != "2021-08-09" or fit["known_count"] < 1
                    or not all(math.isfinite(fit[name]) for name in ("mean", "population_std", "scale"))
                    or fit["population_std"] < 0 or fit["scale"] <= 0
                    or fit["scale"] != (fit["population_std"] or 1)):
                raise ValueError("Invalid train-only release-year transform contract")
    return manifest


def batch(data, indices, device):
    if isinstance(data, _ValidatedExecutionData) and data.batcher is not None:
        return data.batcher.batch(indices)
    x, lengths, y = legacy.batch(data, indices, device)
    field = "bundle_user" if "bundle_user" in data else ("sex" if "sex" in data else "age")
    user = torch.as_tensor(data[field][indices], device=device) if field in data else None
    return x, lengths, y, user


class _ValidatedExecutionData(Mapping):
    """Private runner-owned CPU arrays; public model calls still validate profiles.

    The CLI constructs this only after input and feature manifest checks. The
    factory below also validates values, rather than trusting writeable flags as
    evidence of provenance. Read-only ownership prevents accidental mutation in
    this runner; this is not a security boundary against hostile Python code.
    """
    def __init__(self, values, feature_mode):
        self._values = dict(values)
        self.feature_mode = feature_mode
        self.batcher = None
        for value in self._values.values():
            value.setflags(write=False)

    def __getitem__(self, key):
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def _validated_execution_data(data, model):
    """Validate all CPU training/evaluation inputs before enabling trusted calls.

    The caller must separately bind external sidecars to their manifest. No
    CUDA tensor is inspected here, and no RNG is consumed.
    """
    x, lengths, y = (data[name] for name in ("x", "lengths", "y"))
    context, vocab = model.backbone.config.n_positions, model.backbone.config.vocab_size
    if (any(not isinstance(value, np.ndarray) or value.dtype != np.int64 for value in (x, lengths, y))
            or x.ndim != 2 or x.shape[1] != context or not len(x)
            or lengths.shape != (len(x),) or y.shape != (len(x),)):
        raise ValueError("Optimized execution needs aligned int64 CPU tokens, lengths and targets")
    if np.any(lengths < 1) or np.any(lengths > context) or np.any(y < 0) or np.any(y >= vocab):
        raise ValueError("Optimized execution has invalid lengths or targets")
    for start in range(0, len(x), 16384):
        block, size = x[start:start + 16384], lengths[start:start + 16384]
        active = np.arange(context)[None, :] < size[:, None]
        if (np.any(block < 0) or np.any(block >= vocab) or np.any(block[~active] != PAD)
                or np.any(block[active] == PAD) or np.any(block[:, 0] != BOS) or np.any(block[:, 1:] == BOS)):
            raise ValueError("Optimized execution has invalid tokens, BOS or right padding")
    expected_profile = ("bundle_user" if model.feature_mode == "basic-bundle" else
                        "sex" if model.feature_mode == "sex" else "age" if model.feature_mode.startswith("age") else None)
    present = {name for name in ("age", "sex", "bundle_user") if name in data}
    if present != ({expected_profile} if expected_profile else set()):
        raise ValueError("Optimized execution profile does not match the model")
    if expected_profile == "bundle_user":
        user = data["bundle_user"]
        if not isinstance(user, np.ndarray) or user.shape != (len(x), 10):
            raise ValueError("Bundle profile must be aligned float32 N x 10")
        validate_age(user[:, :7], len(x))
        validate_sex(user[:, 7:], len(x))
    elif expected_profile:
        profile = data[expected_profile]
        if not isinstance(profile, np.ndarray):
            raise ValueError("Optimized execution requires a CPU numpy profile")
        (validate_sex if expected_profile == "sex" else validate_age)(profile, len(x))
    return _ValidatedExecutionData(data, model.feature_mode)


class ConditionedGPT(NextItemGPT):
    def __init__(self, vocab_size, context_length, hidden_size=256, layers=4, heads=4,
                 feature_mode="none", item_features=None, bundle_groups=None, drop_feature_groups=()):
        if feature_mode not in MODES:
            raise ValueError("Unknown feature mode")
        dropped = normalized_drop_groups(drop_feature_groups)
        if feature_mode != "basic-bundle" and (bundle_groups is not None or dropped):
            raise ValueError("Feature groups are only supported by basic-bundle")
        if feature_mode == "basic-bundle":
            if item_features is None or bundle_groups is None:
                raise ValueError("Basic bundle needs item features and named groups")
            import kion_basic_bundle_features as builder
            user_probe = np.zeros((1, 10), np.float32)
            user_probe[0, [0, 9]] = 1
            builder.validate_bundle_layout(user_probe, np.asarray(item_features), bundle_groups)
            if len(item_features) != vocab_size:
                raise ValueError("Bundle item vocabulary shape mismatch")
        elif feature_mode in ("content-type", "release-year"):
            if item_features is None:
                raise ValueError("Item mode needs its singleton sidecar")
            validate_item_features(np.asarray(item_features), vocab_size, feature_mode)
        elif item_features is not None:
            raise ValueError("Only item modes accept item features")
        super().__init__(vocab_size, context_length, hidden_size, layers, heads)
        self.feature_mode = feature_mode
        if feature_mode == "none":
            return
        if feature_mode == "basic-bundle":
            self.bundle_groups = bundle_groups
            self.drop_feature_groups = dropped
            self.register_buffer("item_features", torch.as_tensor(item_features).clone())
            user_mask, item_mask = torch.ones(10), torch.ones(item_features.shape[1])
            for name in dropped:
                group = bundle_groups[name]
                (user_mask if group["scope"] == "user" else item_mask)[group["start"]:group["stop"]] = 0
            self.register_buffer("bundle_user_mask", user_mask)
            self.register_buffer("bundle_item_mask", item_mask)
            with torch.random.fork_rng(devices=[]):
                self.user_feature_projection = nn.Linear(10, hidden_size, bias=False)
                nn.init.normal_(self.user_feature_projection.weight, mean=0, std=.02)
                self.item_feature_projection = nn.Linear(item_features.shape[1], hidden_size, bias=False)
                nn.init.normal_(self.item_feature_projection.weight, mean=0, std=.02)
            return
        dim = 3 if feature_mode == "sex" else (7 if feature_mode in ("age", "age-knownness") else 2)
        if item_features is not None:
            self.register_buffer("item_features", torch.as_tensor(item_features).clone())
        # New parameters must not alter baseline initialization/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.feature_projection = nn.Linear(dim, hidden_size, bias=False)
            nn.init.normal_(self.feature_projection.weight, mean=0, std=.02)

    def conditioned_embeddings(self, x, age=None, sex=None, user_features=None):
        return self._conditioned_embeddings(x, age, sex, user_features, validate_profile=True)

    def _conditioned_embeddings(self, x, age=None, sex=None, user_features=None, *, validate_profile):
        embeddings = self.backbone.wte(x)
        if self.feature_mode == "basic-bundle":
            if age is not None or sex is not None:
                raise ValueError("Basic bundle requires its combined user_features block")
            if user_features is None or user_features.shape != (len(x), 10):
                raise ValueError("Bundle profile conditioning is required in train and evaluation")
            if validate_profile and (not torch.isfinite(user_features).all() or torch.any((user_features != 0) & (user_features != 1))
                    or torch.any(user_features[:, :7].sum(1) != 1) or torch.any(user_features[:, 7:].sum(1) != 1)):
                raise ValueError("Bundle age and sex must be finite exactly one-hot")
            profile = self.user_feature_projection(user_features * self.bundle_user_mask)
            history = self.item_feature_projection(self.item_features[x] * self.bundle_item_mask)
            return embeddings + x.eq(BOS).unsqueeze(-1) * profile.unsqueeze(1) + x.ge(FIRST_ITEM).unsqueeze(-1) * history
        if self.feature_mode == "sex":
            if age is not None:
                raise ValueError("Sex-only mode does not accept an age block")
            if sex is None or sex.shape != (len(x), 3):
                raise ValueError("Sex conditioning is required in train and evaluation")
            if validate_profile and (not torch.isfinite(sex).all() or torch.any((sex != 0) & (sex != 1)) or torch.any(sex.sum(1) != 1)):
                raise ValueError("Sex must be finite exactly one-hot")
            return embeddings + x.eq(BOS).unsqueeze(-1) * self.feature_projection(sex).unsqueeze(1)
        if self.feature_mode in ("age", "age-knownness"):
            if age is None or age.shape != (len(x), 7):
                raise ValueError("Age conditioning is required in train and evaluation")
            if validate_profile and (not torch.isfinite(age).all() or torch.any((age != 0) & (age != 1)) or torch.any(age.sum(1) != 1)):
                raise ValueError("Age must be finite exactly one-hot")
            if self.feature_mode == "age-knownness":
                age = torch.stack((age[:, 0], 1 - age[:, 0], *[torch.zeros_like(age[:, 0])] * 5), dim=1)
            return embeddings + x.eq(BOS).unsqueeze(-1) * self.feature_projection(age).unsqueeze(1)
        if self.feature_mode in ("content-type", "release-year"):
            return embeddings + self.feature_projection(self.item_features[x])
        return embeddings

    def forward(self, x, lengths, age=None, sex=None, user_features=None):
        if self.feature_mode == "none":
            return super().forward(x, lengths)
        hidden = self.backbone(inputs_embeds=self.conditioned_embeddings(x, age, sex, user_features),
                               attention_mask=x.ne(PAD), use_cache=False).last_hidden_state
        return self.output(last_real_hidden(hidden, lengths))


def conditioned_forward(model, x, lengths, user, *, validated_data=None):
    if validated_data is not None:
        if not isinstance(validated_data, _ValidatedExecutionData) or validated_data.feature_mode != model.feature_mode:
            raise ValueError("Trusted forward requires validated execution data for this feature mode")
        if model.feature_mode != "none":
            # Only the private runner adapter bypasses immutable CPU-validated
            # profile value checks. Public forward/conditioned_embeddings retain
            # their checks; the algebra and dropout-bearing backbone are shared.
            values = {"user_features": user} if model.feature_mode == "basic-bundle" else (
                {"sex": user} if model.feature_mode == "sex" else {"age": user})
            hidden = model.backbone(inputs_embeds=model._conditioned_embeddings(x, **values, validate_profile=False),
                                    attention_mask=x.ne(PAD), use_cache=False).last_hidden_state
            return model.output(last_real_hidden(hidden, lengths))
    if model.feature_mode == "basic-bundle":
        return model(x, lengths, user_features=user)
    return model(x, lengths, sex=user) if model.feature_mode == "sex" else model(x, lengths, user)


def _update_attempt(model, data, microbatches, optimizer, scaler, device, clip):
    optimizer.zero_grad(set_to_none=True)
    count, total_loss = sum(map(len, microbatches)), 0.0
    for indices in microbatches:
        x, lengths, y, age = batch(data, indices, device)
        with legacy.autocast(device):
            logits = conditioned_forward(model, x, lengths, age)
            loss = F.cross_entropy(logits[:, FIRST_ITEM:].float(), y - FIRST_ITEM, reduction="sum") / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; cursor did not advance")
        scaler.scale(loss).backward()
        total_loss += float(loss.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise legacy.GradientOverflow()
    scaler.step(optimizer)
    scaler.update()
    return total_loss


def _optimized_update_attempt(model, data, microbatches, optimizer, scaler, device, clip, collect_loss):
    optimizer.zero_grad(set_to_none=True)
    count, losses = sum(map(len, microbatches)), []
    for indices in microbatches:
        x, lengths, y, user = batch(data, indices, device)
        with legacy.autocast(device):
            logits = conditioned_forward(model, x, lengths, user, validated_data=data)
            loss = F.cross_entropy(logits[:, FIRST_ITEM:].float(), y - FIRST_ITEM, reduction="sum") / count
        scaler.scale(loss).backward()
        losses.append(loss.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    # One combined host read per attempt, before optimizer commit. Inspect loss
    # independently: an infinite constant added to CE has finite gradients.
    losses_finite = torch.stack([torch.isfinite(loss) for loss in losses]).all()
    finite_loss, finite_grad = torch.stack((losses_finite, torch.isfinite(norm))).tolist()
    if not finite_loss:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("Non-finite training loss; cursor did not advance")
    if not finite_grad:
        optimizer.zero_grad(set_to_none=True)
        raise legacy.GradientOverflow()
    scaler.step(optimizer)
    scaler.update()
    # Preserve the old Python double sum, with just one tensor transfer at log
    # boundaries. Non-logging updates do not read detached losses on the host.
    return sum(torch.stack(losses).tolist()) if collect_loss else None


def update(model, data, microbatches, optimizer, scaler, device, clip=1.0, *, execution_mode="legacy", collect_loss=True):
    if execution_mode not in EXECUTION_MODES:
        raise ValueError("Unknown execution mode")
    if execution_mode == "optimized" and not isinstance(data, _ValidatedExecutionData):
        raise ValueError("Optimized update requires validated CPU execution data")
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    overflows = 0
    for attempt in range(9):
        if attempt:
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        try:
            if execution_mode == "optimized":
                return _optimized_update_attempt(model, data, microbatches, optimizer, scaler, device, clip, collect_loss), overflows
            return _update_attempt(model, data, microbatches, optimizer, scaler, device, clip), overflows
        except legacy.GradientOverflow:
            overflows += 1
            if not scaler.is_enabled() or attempt == 8:
                raise FloatingPointError("Non-finite gradients persisted after AMP backoff")
            scaler.update(new_scale=scaler.get_scale() / 2)


@torch.no_grad()
def evaluate(model, data, vocab, batch_size, device, indices=None, predictions_path=None, *, execution_mode="legacy"):
    if execution_mode not in EXECUTION_MODES or execution_mode == "optimized" and not isinstance(data, _ValidatedExecutionData):
        raise ValueError("Optimized evaluation requires validated CPU execution data")
    model.eval()
    indices = np.arange(len(data["y"])) if indices is None else indices
    hits = ndcg = ce = 0.0
    known_count = unknown = 0
    k = min(10, vocab - FIRST_ITEM)
    tops, row_hits, row_ndcg = [], [], []
    for start in range(0, len(indices), batch_size):
        x, lengths, y, age = batch(data, indices[start:start + batch_size], device)
        with legacy.autocast(device):
            logits = conditioned_forward(model, x, lengths, age,
                validated_data=data if execution_mode == "optimized" else None)[:, FIRST_ITEM:].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        known = (y >= FIRST_ITEM) & (y < vocab)
        known_count += int(known.sum())
        unknown += int((~known).sum())
        if known.any():
            ce += float(F.cross_entropy(logits[known], y[known] - FIRST_ITEM, reduction="sum"))
        top = logits.topk(k, dim=1).indices + FIRST_ITEM
        match = top.eq(y[:, None]) & known[:, None]
        per_hit = match.any(dim=1)
        discounts = 1 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)
        per_ndcg = (match * discounts).sum(1)
        hits += float(per_hit.sum())
        ndcg += float((match * discounts).sum())
        if predictions_path is not None:
            tops.append(top.cpu().numpy())
            row_hits.append(per_hit.cpu().numpy())
            row_ndcg.append(per_ndcg.cpu().numpy())
    metrics = {"examples": len(indices), "known_targets": known_count, "unknown_target_misses": unknown,
               "cross_entropy_known": ce / known_count if known_count else None,
               "recall@10": hits / len(indices), "ndcg@10": ndcg / len(indices), "actual_k": k}
    if predictions_path is not None:
        values = {name: data[name][indices] for name in ROW_KEYS}
        values.update(row_index=indices, target_token=data["y"][indices],
                      known_target=known_targets(data["y"][indices], vocab), top_items=np.concatenate(tops),
                      hit_at_10=np.concatenate(row_hits), ndcg_at_10=np.concatenate(row_ndcg))
        temporary = predictions_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, predictions_path)
    return metrics


def learning_rate(step, total_updates, warmup_ratio, rate):
    warmup = max(1, int(total_updates * warmup_ratio))
    factor = ((step + 1) / warmup if step < warmup else
              .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup))))
    return rate * factor


def validate_cursor(state, eligible_count, effective_batch, total_epochs):
    epoch, offset = state["epoch"], state["offset"]
    if not 0 <= epoch <= total_epochs or not 0 <= offset <= eligible_count:
        raise ValueError("Checkpoint cursor is outside the training contract")
    if offset != eligible_count and offset % effective_batch:
        raise ValueError("Checkpoint cursor is inside an optimizer update")
    if epoch == total_epochs and offset or len(state["history"]) != epoch:
        raise ValueError("Checkpoint epoch/validation cursor mismatch")
    expected_updates = epoch * math.ceil(eligible_count / effective_batch) + math.ceil(offset / effective_batch)
    if state["updates"] != expected_updates or state["examples"] != epoch * eligible_count + offset:
        raise ValueError("Checkpoint update/example cursor mismatch")


def _atomic_copy(source, destination):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open("rb") as incoming, temporary.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    os.replace(temporary, destination)


def checkpoint(path, model, optimizer, scaler, state, config, rotate=False):
    """Verify a closed temporary checkpoint before publication; retain previous."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "state": state, "rng": legacy.rng_state(), "config": config}
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    legacy.validate_resume(verified["config"], config)
    if verified["state"] != state or set(verified) != set(payload):
        raise ValueError("Temporary checkpoint verification failed")
    del verified
    digest = sha256(temporary)
    if rotate and path.exists():
        # Only the previously committed, hash-matched checkpoint is eligible.
        load_checkpoint(path, config, map_location="cpu", load_weights=False)
        previous = path.with_name("previous.pt")
        _atomic_copy(path, previous)
        write_json(previous.with_suffix(".json"), {"sha256": sha256(previous)})
    os.replace(temporary, path)
    write_json(path.with_suffix(".json"), {"sha256": digest, "epoch": state["epoch"], "offset": state["offset"],
                                         "updates": state["updates"]})
    return digest


def load_checkpoint(path, config, map_location="cpu", load_weights=True):
    receipt = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(path) != receipt["sha256"]:
        raise ValueError("Checkpoint checksum mismatch; explicit previous recovery is required")
    saved = torch.load(path, map_location=map_location, weights_only=False)
    legacy.validate_resume(saved["config"], config)
    if load_weights:
        return saved
    return None


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, default=Path(os.environ["KION_OUTPUT_DIR"]) if "KION_OUTPUT_DIR" in os.environ else None)
    result.add_argument("--feature-mode", choices=MODES, default="none")
    result.add_argument("--execution-mode", choices=EXECUTION_MODES, default="legacy",
                        help="Explicit optimized host synchronization path; CPU parity tested, exact CUDA parity unverified")
    result.add_argument("--batch-transfer", choices=BATCH_TRANSFERS, default="direct",
                        help="Opt-in packed transfer; pinned-packed requests bounded reusable pinned staging")
    result.add_argument("--age-features", type=Path)
    result.add_argument("--sex-features", type=Path)
    result.add_argument("--content-type-features", type=Path)
    result.add_argument("--release-year-features", type=Path)
    result.add_argument("--bundle-features", type=Path)
    result.add_argument("--drop-feature-group", action="append", default=[], choices=BUNDLE_GROUP_NAMES,
                        help="Zero one named input group before projection; repeat for multiple groups")
    result.add_argument("--feature-manifest", type=Path)
    result.add_argument("--device", default="auto")
    for name, default in (("schedule-total-epochs", 5), ("batch-size", 64), ("accumulation", 2),
                          ("hidden-size", 256), ("layers", 4), ("heads", 4), ("seed", 2026),
                          ("patience", 2), ("validation-limit", 0), ("max-updates", 0),
                          ("checkpoint-updates", 1000), ("log-updates", 100), ("bucket-size", 8192),
                          ("cpu-threads", 4), ("stop-after-validation", 0)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--warmup-ratio", type=float, default=.05)
    result.add_argument("--weight-decay", type=float, default=.01)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--resume-from", choices=("latest", "previous"), default="latest")
    result.add_argument("--stop-file", type=Path, default=Path(os.environ["KION_STOP_FILE"]) if "KION_STOP_FILE" in os.environ else None)
    result.add_argument("--evaluate-only", action="store_true", help="Requires --resume; exports the selected validation checkpoint")
    result.add_argument("--legacy-checkpoint", type=Path, help="ID-only validation export from a pinned legacy full checkpoint")
    return result


def export_legacy(args, config, model, val, val_indices, vocab, device):
    """Evaluation bridge only; a legacy checkpoint never becomes a new resume."""
    saved = torch.load(args.legacy_checkpoint, map_location=device, weights_only=False)
    old = saved["config"]["contract"]
    current = config["contract"]
    if old["inputs_sha256"] != current["inputs_sha256"]:
        raise ValueError("Legacy checkpoint data mapping/input mismatch")
    for name in ("hidden_size", "layers", "heads", "context_length", "vocab_size", "tied_embeddings"):
        if old["architecture"][name] != current["architecture"][name]:
            raise ValueError("Legacy checkpoint architecture mismatch")
    if old["architecture"].get("feature_mode", "none") != "none":
        raise ValueError("Legacy export bridge only supports the ID baseline")
    for name, digest in old["sources_sha256"].items():
        if current["sources_sha256"].get(name) != digest:
            raise ValueError("Legacy source pin mismatch")
    model.load_state_dict(saved["state_dict"], strict=True)
    epoch = saved["state"]["epoch"]
    del saved
    write_json(args.output / "config.json", config)
    metrics = evaluate(model, val, vocab, args.batch_size, device, val_indices, args.output / "predictions.npz",
                       execution_mode=args.execution_mode)
    result = {"status": "completed", "mode": "legacy_validation_export", "selected_epoch": epoch, "validation": metrics,
              "checkpoint": args.legacy_checkpoint.resolve().as_posix(), "checkpoint_sha256": sha256(args.legacy_checkpoint)}
    write_json(args.output / "metrics.json", result)
    artifacts = ["predictions.npz", "metrics.json", "config.json"]
    write_json(args.output / "completion.json", {"state": "completed", "validation_rows": metrics["examples"],
        "metrics": {"ndcg10": metrics["ndcg@10"], "recall10": metrics["recall@10"], "ce": metrics["cross_entropy_known"]},
        "artifacts": artifacts, "sha256": {name: sha256(args.output / name) for name in artifacts},
        "selected_epoch": epoch, "legacy_checkpoint_sha256": result["checkpoint_sha256"], "test_accessed": False})
    return result


def _feature_paths(args):
    paths = {"age": args.age_features, "age-knownness": args.age_features,
             "content-type": args.content_type_features, "release-year": args.release_year_features,
             "sex": args.sex_features, "basic-bundle": args.bundle_features}
    selected = paths.get(args.feature_mode)
    supplied = [path for path in (args.age_features, args.content_type_features, args.release_year_features,
                                  args.sex_features, args.bundle_features) if path is not None]
    if (args.feature_mode != "none" and selected is None) or len(supplied) != int(selected is not None):
        raise ValueError("Supply exactly the single sidecar selected by feature-mode")
    if args.feature_mode == "none" and args.feature_manifest is not None:
        raise ValueError("ID baseline does not consume a feature manifest")
    if selected is not None and args.feature_manifest is None:
        raise ValueError("A named, row-bound --feature-manifest is required")
    if args.drop_feature_group and args.feature_mode != "basic-bundle":
        raise ValueError("Drop feature groups require basic-bundle")
    args.drop_feature_group = normalized_drop_groups(args.drop_feature_group)
    return selected


def _main(argv, resources):
    args = parser().parse_args(argv)
    for name in ("schedule_total_epochs", "batch_size", "accumulation", "hidden_size", "layers", "heads",
                 "patience", "checkpoint_updates", "log_updates", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.output is None or args.hidden_size % args.heads or not 0 <= args.warmup_ratio < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid output, architecture, or optimization settings")
    if min(args.validation_limit, args.max_updates, args.bucket_size, args.stop_after_validation) < 0:
        raise ValueError("Limits must be nonnegative")
    if (args.evaluate_only and not args.legacy_checkpoint or args.resume_from != "latest") and not args.resume:
        raise ValueError("Evaluation-only/previous recovery requires --resume")
    if args.legacy_checkpoint and (not args.evaluate_only or args.resume or args.feature_mode != "none" or args.validation_limit):
        raise ValueError("Legacy checkpoint export requires a fresh ID-only full-validation evaluation output")
    feature_path = _feature_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.iterdir()):
        raise ValueError("New run requires an empty output directory")
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    vocab, context = metadata["vocab_size"], metadata["context"]
    if vocab <= FIRST_ITEM:
        raise ValueError("Vocabulary needs real item candidates")
    inputs = {name: sha256(args.data / name) for name in ("metadata.json", "train.npz", "val.npz")}
    feature_hashes = {"sidecar": sha256(feature_path)} if feature_path else {}
    if args.feature_manifest:
        feature_hashes["manifest"] = sha256(args.feature_manifest)
    sources = {name: sha256(Path(__file__).with_name(name)) for name in
               ("kion_conditioned_train.py", "kion_full_train.py", "kion_train.py", "kion_prepare.py")}
    if args.feature_mode == "basic-bundle":
        for name in ("kion_basic_bundle_features.py", "kion_local_features.py", "kion_sex_features.py"):
            sources[name] = sha256(Path(__file__).with_name(name))
    if args.batch_transfer != "direct":
        sources["kion_cuda_batches.py"] = sha256(Path(__file__).with_name("kion_cuda_batches.py"))
    import transformers
    contract = {"architecture": {"model": "GPT2Model from scratch", "hidden_size": args.hidden_size,
                "layers": args.layers, "heads": args.heads, "context_length": context, "vocab_size": vocab,
                "tied_embeddings": True, "feature_mode": args.feature_mode,
                "conditioning": "age at existing BOS; singleton content on history; tied ID candidate head",
                "age_categories": list(AGE_CATEGORIES) if args.feature_mode.startswith("age") else None,
                "profile_availability": "static_snapshot_unverified"},
                "training": {key: getattr(args, key) for key in ("schedule_total_epochs", "batch_size", "accumulation",
                    "lr", "warmup_ratio", "weight_decay", "seed", "patience", "validation_limit", "bucket_size",
                    "execution_mode", "batch_transfer")},
                "inputs_sha256": inputs, "sources_sha256": sources, "features_sha256": feature_hashes,
                "objective": "CE over known train catalog; all validation rows including unknown misses",
                "checkpoint_selection": "maximum validation NDCG@10, strict greater; patience on full rows",
                "prediction_row_keys": list(ROW_KEYS)}
    if args.feature_mode == "sex":
        contract["architecture"]["sex_categories"] = list(SEX_CATEGORIES)
        contract["architecture"]["conditioning"] = "sex at existing BOS only; tied ID candidate head"
        contract["architecture"]["profile_identity"] = "account profile; actual viewer identity unverified"
    config = {"contract": contract, "data": args.data.resolve().as_posix(),
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                "cuda": torch.version.cuda, "device": str(device), "cpu_threads": args.cpu_threads,
                "torch_num_threads": torch.get_num_threads(),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    if args.resume and args.feature_mode != "basic-bundle":
        old_config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        legacy.validate_resume(old_config, config)
        config = old_config
    train = load_data(args.data / "train.npz", vocab, context, require_keys=feature_path is not None, unique_keys=False)
    val = load_data(args.data / "val.npz", vocab, context, require_keys=True)
    item_features = None
    bundle_manifest = None
    if args.feature_mode == "basic-bundle":
        with np.load(feature_path, allow_pickle=False) as source:
            if set(source.files) != {"train_user_features", "val_user_features", "item_features"}:
                raise ValueError("Bundle requires exactly its three named arrays")
            train["bundle_user"], val["bundle_user"] = source["train_user_features"], source["val_user_features"]
            item_features = source["item_features"]
        bundle_manifest = validate_bundle_manifest(args.feature_manifest, feature_path, metadata, inputs, train, val, item_features)
        contract["architecture"].update(
            conditioning="age+sex at existing BOS; concatenated item features on real history items; tied ID candidate head",
            age_categories=list(AGE_CATEGORIES), sex_categories=list(SEX_CATEGORIES),
            bundle_contract=bundle_manifest["builder_contract"], feature_groups=bundle_manifest["groups"],
            dropped_feature_groups=list(args.drop_feature_group),
            feature_group_mask={name: int(name not in args.drop_feature_group) for name in BUNDLE_GROUP_NAMES},
            user_feature_dim=10, item_feature_dim=item_features.shape[1],
            feature_array_hashes={name: declaration["sha256"] for name, declaration in bundle_manifest["features"]["arrays"].items()})
        if args.resume:
            old_config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
            legacy.validate_resume(old_config, config)
            config = old_config
    elif args.feature_mode.startswith("age"):
        with np.load(feature_path, allow_pickle=False) as source:
            train["age"] = validate_age(source["train_user_age"], len(train["y"]))
            val["age"] = validate_age(source["val_user_age"], len(val["y"]))
    elif args.feature_mode == "sex":
        with np.load(feature_path, allow_pickle=False) as source:
            train["sex"] = validate_sex(source["train_user_sex"], len(train["y"]))
            val["sex"] = validate_sex(source["val_user_sex"], len(val["y"]))
    elif feature_path:
        item_features = validate_item_features(np.load(feature_path, allow_pickle=False), vocab, args.feature_mode)
    if args.feature_manifest and args.feature_mode != "basic-bundle":
        validate_feature_manifest(args.feature_manifest, feature_path, args.feature_mode, metadata, inputs, train, val, item_features)
    eligible = np.flatnonzero(known_targets(train["y"], vocab))
    if not len(eligible):
        raise ValueError("No known training targets")
    val_indices = np.arange(len(val["y"]))
    if args.validation_limit and args.validation_limit < len(val_indices):
        val_indices = np.sort(np.random.default_rng(args.seed).choice(val_indices, args.validation_limit, replace=False))
    model = ConditionedGPT(vocab, context, args.hidden_size, args.layers, args.heads, args.feature_mode, item_features,
                           bundle_groups=bundle_manifest["groups"] if bundle_manifest else None,
                           drop_feature_groups=args.drop_feature_group).to(device)
    if args.execution_mode == "optimized" or args.batch_transfer != "direct":
        train, val = (_validated_execution_data(data, model) for data in (train, val))
        if args.batch_transfer != "direct":
            from kion_cuda_batches import PackedBatcher
            for data in (train, val):
                data.batcher = resources.enter_context(PackedBatcher(data, device, max_batch_size=args.batch_size,
                    pin_memory=args.batch_transfer == "pinned-packed"))
    config["parameters"] = sum(p.numel() for p in model.parameters())
    if args.legacy_checkpoint:
        return export_legacy(args, config, model, val, val_indices, vocab, device)
    if not args.resume:
        write_json(args.output / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    state = {"epoch": 0, "offset": 0, "updates": 0, "examples": 0, "best_ndcg": -1.0,
             "bad_epochs": 0, "history": [], "amp_overflow_retries": 0, "selected_epoch": None}
    if args.resume:
        saved = load_checkpoint(args.output / (args.resume_from + ".pt"), config, map_location=device)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        legacy.restore_rng(saved["rng"])
        del saved
    effective_batch = args.batch_size * args.accumulation
    validate_cursor(state, len(eligible), effective_batch, args.schedule_total_epochs)
    total_updates = math.ceil(len(eligible) / effective_batch) * args.schedule_total_epochs
    started = time.perf_counter()
    initial_examples = state["examples"]
    recovery_publication = args.resume and args.resume_from == "previous"

    def emit(event, **values):
        legacy.synchronize(device)
        elapsed = time.perf_counter() - started
        speed = (state["examples"] - initial_examples) / elapsed if elapsed else 0
        payload = {"event": event, "updated_utc": datetime.now(timezone.utc).isoformat(), **state,
                   "elapsed_seconds": elapsed, "examples_per_second": speed,
                   "eta_seconds": max(0, (total_updates - state["updates"]) * effective_batch / speed) if speed else None,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, **values}
        write_json(args.output / "progress.json", payload)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    def save_latest():
        nonlocal recovery_publication
        validate_cursor(state, len(eligible), effective_batch, args.schedule_total_epochs)
        digest = checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config, rotate=not recovery_publication)
        recovery_publication = False
        return digest

    def stop_reason():
        if args.stop_file is not None and args.stop_file.exists():
            return "runtime_stop_file"
        if args.max_updates and state["updates"] >= args.max_updates:
            return "runtime_max_updates"
        if args.stop_after_validation and state["epoch"] >= args.stop_after_validation:
            return "runtime_stop_after_validation"
        return None

    emit("started", status="running", train_examples=len(eligible), validation_examples=len(val_indices),
         schedule_total_updates=total_updates)
    stopped = None
    for epoch in range(state["epoch"], args.schedule_total_epochs):
        if args.evaluate_only or state["bad_epochs"] >= args.patience:
            break
        stopped = stop_reason()
        if stopped:
            save_latest()
            break
        order = epoch_order(eligible, train["lengths"], args.seed, epoch, args.batch_size, args.bucket_size)
        legacy.synchronize(device)
        epoch_started = time.perf_counter()
        epoch_initial_examples = state["examples"]
        resumed_segment = state["offset"] > 0
        model.train()
        for offset, microbatches in groups(order, args.batch_size, args.accumulation, state["offset"]):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(state["updates"], total_updates, args.warmup_ratio, args.lr)
            logging_update = (state["updates"] + 1) % args.log_updates == 0 or state["updates"] == 0
            loss, overflows = update(model, train, microbatches, optimizer, scaler, device,
                                    execution_mode=args.execution_mode, collect_loss=logging_update)
            count = sum(map(len, microbatches))
            state.update(updates=state["updates"] + 1, offset=offset + count, examples=state["examples"] + count,
                         amp_overflow_retries=state["amp_overflow_retries"] + overflows)
            if state["updates"] % args.log_updates == 0 or state["updates"] == 1:
                emit("training", status="running", loss=loss, learning_rate=optimizer.param_groups[0]["lr"])
            if state["updates"] % args.checkpoint_updates == 0:
                save_latest()
            # A stop is honored only after a full successful optimizer update.
            stopped = stop_reason()
            if stopped:
                save_latest()
                break
        if stopped:
            break
        legacy.synchronize(device)
        train_wall = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics = evaluate(model, val, vocab, args.batch_size, device, val_indices, execution_mode=args.execution_mode)
        legacy.synchronize(device)
        state["history"].append({"epoch": epoch + 1, "validation": metrics,
            "train_wall_seconds_including_periodic_checkpoints_and_logs": train_wall,
            "train_timing_is_resumed_segment": resumed_segment,
            "train_examples_per_second": (state["examples"] - epoch_initial_examples) / train_wall,
            "validation_wall_seconds": time.perf_counter() - validation_started})
        improved = metrics["ndcg@10"] > state["best_ndcg"]
        state.update(epoch=epoch + 1, offset=0, bad_epochs=0 if improved else state["bad_epochs"] + 1)
        if improved:
            state["best_ndcg"] = metrics["ndcg@10"]
            state["selected_epoch"] = epoch + 1
            selected = args.output / f"selected-epoch{epoch + 1:04d}.pt"
            if selected.exists():
                # Replaying from previous may reach an already published epoch.
                # A missing checksum receipt means publication was interrupted.
                # Verify and adopt those bytes; never rewrite a selected snapshot.
                receipt_path = selected.with_suffix(".json")
                committed = receipt_path.exists()
                digest = sha256(selected)
                prior = (load_checkpoint(selected, config, map_location=device) if committed else
                         torch.load(selected, map_location=device, weights_only=False))
                legacy.validate_resume(prior["config"], config)
                same_weights = all(torch.equal(value, prior["state_dict"][name]) for name, value in model.state_dict().items())
                if (not same_weights or prior["state"]["history"][-1]["validation"] != metrics
                        or any(prior["state"][name] != state[name] for name in ("epoch", "offset", "updates", "examples"))
                        or sha256(selected) != digest):
                    raise ValueError("Selected snapshot conflicts with replay; use a new output")
                if not committed:
                    if set(prior) != {"state_dict", "optimizer", "scaler", "state", "rng", "config"}:
                        raise ValueError("Orphan selected checkpoint payload is incomplete")
                    write_json(receipt_path, {"sha256": digest, "epoch": state["epoch"], "offset": state["offset"],
                                              "updates": state["updates"]})
                del prior
            else:
                digest = checkpoint(selected, model, optimizer, scaler, state, config)
            _atomic_copy(selected, args.output / "best.pt")
            write_json(args.output / "best.json", {"sha256": digest})
            write_json(args.output / "selected.json", {"file": selected.name, "sha256": digest, "epoch": epoch + 1,
                                                       "validation": metrics})
        save_latest()
        emit("validation", status="running", validation=metrics)
        stopped = stop_reason()
        if stopped:
            break
    early_stopped = state["bad_epochs"] >= args.patience
    finished = state["epoch"] >= args.schedule_total_epochs or early_stopped
    status = "completed" if finished else "stopped"
    selected_metrics = None
    selected = None
    if state["selected_epoch"] is not None:
        selected_path = args.output / f"selected-epoch{state['selected_epoch']:04d}.pt"
        saved = load_checkpoint(selected_path, config, map_location=device)
        selected = {"file": selected_path.name, "sha256": sha256(selected_path), "epoch": state["selected_epoch"],
                    "validation": saved["state"]["history"][-1]["validation"]}
        write_json(args.output / "selected.json", selected)
        if not (args.output / "best.pt").exists() or sha256(args.output / "best.pt") != selected["sha256"]:
            _atomic_copy(selected_path, args.output / "best.pt")
            write_json(args.output / "best.json", {"sha256": selected["sha256"]})
        model.load_state_dict(saved["state_dict"])
        del saved
        export_started = time.perf_counter()
        selected_metrics = evaluate(model, val, vocab, args.batch_size, device, val_indices, args.output / "predictions.npz",
                                    execution_mode=args.execution_mode)
        if selected_metrics != selected["validation"]:
            raise ValueError("Selected predictions do not reproduce validation metrics")
        emit("exported", status=status, prediction_sha256=sha256(args.output / "predictions.npz"),
             export_wall_seconds=time.perf_counter() - export_started)
    elif args.evaluate_only:
        raise ValueError("No validation-selected checkpoint is available")
    result = {"status": status, "reason": stopped, "early_stopped": early_stopped, "state": state,
              "selected_epoch": selected["epoch"] if selected else None, "validation": selected_metrics}
    write_json(args.output / "metrics.json", result)
    artifacts = [name for name in ("latest.pt", "previous.pt", "best.pt", "selected.json", "predictions.npz", "metrics.json")
                 if (args.output / name).exists() and (selected is not None or name not in ("best.pt", "selected.json", "predictions.npz"))]
    if selected:
        artifacts.append(selected["file"])
    completion = {"state": status, "updates": state["updates"], "schedule_total_epochs": args.schedule_total_epochs,
                  "validation_rows": selected_metrics["examples"] if selected_metrics else 0,
                  "metrics": ({"ndcg10": selected_metrics["ndcg@10"], "recall10": selected_metrics["recall@10"],
                               "ce": selected_metrics["cross_entropy_known"]} if selected_metrics else {}),
                  "artifacts": artifacts, "sha256": {name: sha256(args.output / name) for name in artifacts},
                  "reason": stopped, "selected_epoch": result["selected_epoch"], "test_accessed": False}
    write_json(args.output / "completion.json", completion)
    emit("finished", status=status, early_stopped=early_stopped, reason=stopped)
    return result


def main(argv=None):
    # ExitStack also drains packed transfers when a validation/training/checkpoint
    # exception escapes. Legacy/direct creates no batchers and preserves its path.
    with ExitStack() as resources:
        return _main(argv, resources)


if __name__ == "__main__":
    main()
