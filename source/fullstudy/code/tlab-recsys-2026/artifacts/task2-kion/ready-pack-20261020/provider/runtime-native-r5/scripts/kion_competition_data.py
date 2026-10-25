"""Build CPU-only KION competition contexts using the selected model's contract.

The official sample supplies the complete user set and row order. Profiles are
the same retrospective snapshot used in training. Item IDs, categories, source
helpers, and the model configuration stay frozen; no feature or model is fitted.
Only user/item IDs and last-watch dates are projected from interactions.
"""
from __future__ import annotations

import argparse
from array import array
import csv
from datetime import date
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


CONTRACT = "kion-competition-contexts-v1"
CONTEXT = 64
CUTOFF_EXCLUSIVE = "2021-08-23"
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
MAX_ID = np.iinfo(np.int64).max
SOURCE_NAMES = {
    "conditioned_helper": "kion_conditioned_train.py",
    "full_helper": "kion_full_train.py",
    "train_helper": "kion_train.py",
    "prepare_helper": "kion_prepare.py",
    "bundle_helper": "kion_basic_bundle_features.py",
    "local_helper": "kion_local_features.py",
    "sex_helper": "kion_sex_features.py",
}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_hash(values):
    header = json.dumps({"dtype": values.dtype.str, "shape": list(values.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(np.ascontiguousarray(values)).cast("B"))
    return digest.hexdigest()


def _read_json(path):
    # Duplicate keys in a contract or provenance document are ambiguous.
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def _strict_id(value, label):
    text = value.strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}: {text!r}") from error
    if parsed < 0 or parsed > MAX_ID or str(parsed) != text:
        raise ValueError(f"Invalid {label}: {text!r}")
    return parsed


def _csv_projection(path, required):
    """Validate CSV structure and return only explicitly requested columns."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        headers = next(reader, None)
        if not headers or len(headers) != len(set(headers)) or not set(required).issubset(headers):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        indices = [headers.index(field) for field in required]
        for number, row in enumerate(reader, 2):
            if len(row) != len(headers):
                raise ValueError(f"Malformed CSV row {number}: {path}")
            yield tuple(row[index] for index in indices)


def read_sample(path):
    ids, seen = [], set()
    for (text,) in _csv_projection(path, ("user_id",)):
        user_id = _strict_id(text, "sample user_id")
        if user_id in seen:
            raise ValueError(f"Duplicate sample user_id: {user_id}")
        seen.add(user_id)
        ids.append(user_id)
    if not ids:
        raise ValueError("Official sample is empty")
    return np.asarray(ids, dtype=np.int64)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_helpers(paths):
    """Import checked snapshots without reusing another task's helper module."""
    local = _load_module(paths["local_helper"], "_competition_frozen_local")
    previous_module = sys.modules.get("kion_local_features")
    previous_path = list(sys.path)
    try:
        sys.modules["kion_local_features"] = local
        sex = _load_module(paths["sex_helper"], "_competition_frozen_sex")
    finally:
        sys.path[:] = previous_path
        if previous_module is None:
            sys.modules.pop("kion_local_features", None)
        else:
            sys.modules["kion_local_features"] = previous_module
    return local, sex


def _validate_model(config, metadata_hash, paths, hashes, local, sex):
    contract = config.get("contract", {})
    architecture = contract.get("architecture", {})
    if (architecture.get("feature_mode") != "basic-bundle"
            or architecture.get("bundle_contract") != "kion-basic-bundle-v1"
            or architecture.get("context_length") != CONTEXT
            or architecture.get("user_feature_dim") != 10
            or architecture.get("dropped_feature_groups") != []):
        raise ValueError("Expected the full seven-group context64 basic-bundle model")
    names = ("age", "sex", "genres", "content_type", "release_year", "countries", "age_rating")
    if architecture.get("feature_group_mask") != {name: 1 for name in names}:
        raise ValueError("Model must enable all seven feature groups")
    groups = architecture.get("feature_groups", {})
    if list(groups) != list(names):
        raise ValueError("Model feature-group order differs from the frozen contract")
    expected_user = {
        "age": {"scope": "user", "start": 0, "stop": 7, "columns": list(local.AGE_CATEGORIES)},
        "sex": {"scope": "user", "start": 7, "stop": 10, "columns": list(sex.SEX_CATEGORIES)},
    }
    if (any(groups.get(name) != entry for name, entry in expected_user.items())
            or architecture.get("age_categories") != local.AGE_CATEGORIES
            or architecture.get("sex_categories") != sex.SEX_CATEGORIES):
        raise ValueError("Model's profile feature order differs from frozen helpers")
    if contract.get("inputs_sha256", {}).get("metadata.json") != metadata_hash:
        raise ValueError("Metadata hash differs from the model training contract")
    pins = contract.get("sources_sha256", {})
    for key, filename in SOURCE_NAMES.items():
        if pins.get(filename) != hashes[key]:
            raise ValueError(f"Frozen model source pin mismatch: {paths[key].name}")
    return architecture, expected_user


def _verify_raw_origin(path, hashes, paths):
    manifest = _read_json(path)
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError("Raw manifest has no files list")
    for key in ("interactions", "users"):
        matches = [entry for entry in files if isinstance(entry, dict) and entry.get("file") == paths[key].name]
        if (len(matches) != 1 or matches[0].get("sha256") != hashes[key]
                or matches[0].get("bytes") != paths[key].stat().st_size):
            raise ValueError(f"Raw {key} hash/size differs from source manifest")
    return {"status": "source_manifest_verified_for_used_raw_files", "url": manifest.get("url"),
            "revision": manifest.get("tree_sha"), "archive_sha256": manifest.get("archive_sha256")}


def _verify_sample_origin(path, sample_path, sample_hash, user_ids):
    manifest = _read_json(path)
    if (manifest.get("sha256") != sample_hash
            or manifest.get("bytes") != sample_path.stat().st_size
            or manifest.get("rows") != len(user_ids)
            or not isinstance(manifest.get("source"), str) or not manifest["source"]):
        raise ValueError("Official sample differs from its source manifest")
    for field, observed in (("first_ids", user_ids[:5]), ("last_ids", user_ids[-5:])):
        if field in manifest and manifest[field] != observed.tolist():
            raise ValueError(f"Sample source manifest {field} disagree")
    return {"status": "sample_source_manifest_verified", "url": manifest["source"]}


def read_histories(path, user_ids, mapping):
    selected = set(map(int, user_ids))
    users, items, days = array("q"), array("q"), array("i")
    cutoff = date.fromisoformat(CUTOFF_EXCLUSIVE).toordinal()
    dates = {}
    total = eligible = selected_total = 0
    min_day = max_day = eligible_min = eligible_max = None
    for raw_user, raw_item, raw_day in _csv_projection(path, ("user_id", "item_id", "last_watch_dt")):
        uid, iid = _strict_id(raw_user, "interaction user_id"), _strict_id(raw_item, "interaction item_id")
        if raw_day not in dates:
            parsed = date.fromisoformat(raw_day)
            if parsed.isoformat() != raw_day:
                raise ValueError(f"Noncanonical interaction date: {raw_day!r}")
            dates[raw_day] = parsed.toordinal()
        day = dates[raw_day]
        total += 1
        min_day = day if min_day is None else min(min_day, day)
        max_day = day if max_day is None else max(max_day, day)
        if day < cutoff:
            eligible += 1
            eligible_min = day if eligible_min is None else min(eligible_min, day)
            eligible_max = day if eligible_max is None else max(eligible_max, day)
        if uid in selected:
            selected_total += 1
            if day < cutoff:
                users.append(uid)
                items.append(iid)
                days.append(day)
    if total == 0:
        raise ValueError("Empty interactions")
    users = np.frombuffer(users, np.int64).copy()
    items = np.frombuffer(items, np.int64).copy()
    days = np.frombuffer(days, np.int32).copy()
    order = np.lexsort((items, days, users))
    users, items, days = users[order], items[order], days[order]
    tokens = np.fromiter((mapping.get(str(int(item)), SPECIAL_TOKENS["UNK"]) for item in items),
                         dtype=np.int64, count=len(items))
    def iso(ordinal):
        return date.fromordinal(ordinal).isoformat() if ordinal is not None else None
    facts = {"rows": total, "eligible_rows": eligible, "excluded_on_or_after_cutoff_rows": total - eligible,
             "selected_user_rows": selected_total, "selected_eligible_rows": len(users),
             "selected_excluded_on_or_after_cutoff_rows": selected_total - len(users),
             "selected_eligible_oov_rows": int(np.count_nonzero(tokens == SPECIAL_TOKENS["UNK"]))}
    ranges = {"raw_min": iso(min_day), "raw_max": iso(max_day),
              "eligible_min": iso(eligible_min), "eligible_max": iso(eligible_max),
              "selected_eligible_min": iso(int(days.min())) if len(days) else None,
              "selected_eligible_max": iso(int(days.max())) if len(days) else None}
    return users, tokens, facts, ranges


def make_contexts(user_ids, history_users, history_tokens):
    x = np.zeros((len(user_ids), CONTEXT), dtype=np.int64)
    x[:, 0] = SPECIAL_TOKENS["BOS"]
    lengths = np.ones(len(user_ids), dtype=np.int64)
    row_lookup = {int(uid): row for row, uid in enumerate(user_ids)}
    unique, begins, counts = np.unique(history_users, return_index=True, return_counts=True)
    for uid, begin, count in zip(unique, begins, counts):
        row = row_lookup[int(uid)]
        recent = history_tokens[begin + max(0, count - CONTEXT + 1):begin + count]
        lengths[row] += len(recent)
        x[row, 1:lengths[row]] = recent
    history = x[:, 1:]
    coverage = {"sample_users": len(user_ids), "history_users": len(unique),
                "cold_users": len(user_ids) - len(unique),
                "retained_history_rows": int(np.sum(lengths - 1)),
                "truncated_history_users": int(np.count_nonzero(counts > CONTEXT - 1)),
                "retained_oov_rows": int(np.count_nonzero(history == SPECIAL_TOKENS["UNK"])),
                "retained_oov_users": int(np.count_nonzero(np.any(history == SPECIAL_TOKENS["UNK"], axis=1))),
                "retained_known_rows": int(np.count_nonzero(history >= 3)),
                "users_with_known_history": int(np.count_nonzero(np.any(history >= 3, axis=1))),
                "minimum_length": int(lengths.min()), "maximum_length": int(lengths.max())}
    return x, lengths, coverage


def build_competition_data(sample, interactions, users, metadata, bundle_scripts, model_config, output,
                           raw_manifest=None, sample_manifest=None, sample_sha256=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    paths = {name: Path(value).resolve() for name, value in {
        "sample": sample, "interactions": interactions, "users": users,
        "metadata": metadata, "model_config": model_config, "builder": __file__}.items()}
    source_dir = Path(bundle_scripts).resolve()
    paths.update({key: source_dir / filename for key, filename in SOURCE_NAMES.items()})
    paths["raw_manifest"] = Path(raw_manifest or paths["interactions"].parent / "manifest.json").resolve()
    if sample_manifest is not None:
        paths["sample_manifest"] = Path(sample_manifest).resolve()
    for path in paths.values():
        if output == path or path.parent == output or path.parent in output.parents:
            raise ValueError("Output must be outside all read-only input directories")
    hashes = {key: file_hash(path) for key, path in paths.items()}
    if sample_sha256 is not None and sample_sha256 != hashes["sample"]:
        raise ValueError("Official sample SHA-256 differs from the expected source hash")
    config = _read_json(paths["model_config"])
    # Validate source hashes before executing either frozen helper.
    for key, filename in SOURCE_NAMES.items():
        if config.get("contract", {}).get("sources_sha256", {}).get(filename) != hashes[key]:
            raise ValueError(f"Frozen model source pin mismatch: {filename}")
    local, sex = _load_helpers(paths)
    architecture, user_groups = _validate_model(config, hashes["metadata"], paths, hashes, local, sex)
    _read_json(paths["metadata"])
    prepared_metadata, mapping = sex.read_metadata(paths["metadata"])
    if (prepared_metadata.get("context") != CONTEXT
            or prepared_metadata.get("special_tokens") != SPECIAL_TOKENS
            or prepared_metadata.get("vocab_size") != architecture.get("vocab_size")):
        raise ValueError("Prepared vocabulary/context differs from the model")
    origin = _verify_raw_origin(paths["raw_manifest"], hashes, paths)
    user_ids = read_sample(paths["sample"])
    sample_origin = {"status": "local_sample_hashed", "url": None}
    if "sample_manifest" in paths:
        sample_origin = _verify_sample_origin(paths["sample_manifest"], paths["sample"], hashes["sample"], user_ids)
    profiles = local.read_fields(paths["users"], "user_id", ("age", "sex"))
    ages, age_coverage = local.age_features({uid: (row[0],) for uid, row in profiles.items()}, user_ids)
    sexes, sex_coverage = sex.sex_features({uid: (row[1],) for uid, row in profiles.items()}, user_ids)
    user_features = np.concatenate((ages, sexes), axis=1)
    history_users, tokens, interaction_facts, ranges = read_histories(paths["interactions"], user_ids, mapping)
    x, lengths, history_coverage = make_contexts(user_ids, history_users, tokens)
    arrays = {"x": x, "lengths": lengths, "user_ids": user_ids, "user_features": user_features}
    if (np.any(x < 0) or np.any(x >= prepared_metadata["vocab_size"])
            or user_features.dtype != np.dtype("float32") or user_features.shape != (len(user_ids), 10)
            or not np.isfinite(user_features).all()
            or np.any(user_features[:, :7].sum(axis=1) != 1)
            or np.any(user_features[:, 7:].sum(axis=1) != 1)):
        raise ValueError("Generated inference arrays violate the frozen contract")
    inputs = {key: {"path": path.as_posix(), "sha256": hashes[key], "bytes": path.stat().st_size}
              for key, path in paths.items()}
    manifest = {"format_version": 1, "builder_contract": CONTRACT,
                "kind": "competition_inference", "context": CONTEXT,
                "cutoff_exclusive": CUTOFF_EXCLUSIVE, "special_tokens": SPECIAL_TOKENS,
                "vocab_size": prepared_metadata["vocab_size"],
                "item_id_to_token_sha256": canonical_hash(mapping), "user_feature_groups": user_groups,
                "inputs": inputs, "source_origin": origin, "sample_origin": sample_origin,
                "sample_user_ids_sha256": array_hash(user_ids), "raw_date_range": ranges,
                "coverage": {"interactions": interaction_facts, "history": history_coverage,
                             "age": age_coverage, "sex": sex_coverage},
                "history_order": "user_id, last_watch_dt, item_id; same-day item order is a storage convention",
                "history_encoding": "BOS=1 then last63 mapped tokens; OOV=2; PAD=0 on right; cold users BOS only",
                "profile_availability": "static_snapshot_unverified",
                "feature_fit": "none; immutable training vocabulary and frozen age/sex transform",
                "interaction_fields_used": ["user_id", "item_id", "last_watch_dt"]}
    # A failed validation leaves no output. The final directory rename is atomic
    # on the native Windows filesystem; existing destinations are never replaced.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building-", dir=output.parent) as staging:
        staging = Path(staging)
        np.savez_compressed(staging / "inference.npz", **arrays)
        manifest["inference"] = {"file": "inference.npz", "sha256": file_hash(staging / "inference.npz"),
                                 "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                                  "sha256": array_hash(value)} for key, value in arrays.items()}}
        if any(not path.is_file() or file_hash(path) != hashes[key] for key, path in paths.items()):
            raise ValueError("Input/source changed while building competition contexts")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                encoding="utf-8", newline="\n")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
        os.rename(staging, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "interactions", "users", "metadata", "bundle-scripts", "model-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-sha256")
    args = parser.parse_args()
    manifest = build_competition_data(**vars(args))
    print(json.dumps({"output": args.output.resolve().as_posix(), "coverage": manifest["coverage"],
                      "inference_sha256": manifest["inference"]["sha256"],
                      "manifest_sha256": file_hash(args.output / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
