"""Build the frozen seven-group KION basic bundle using local train/val only.

Profiles and item metadata are retrospective snapshots with unverified event-time
availability. Target keys bind rows; outcomes, interactions and test are not read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_local_features as local
import kion_sex_features as sex


CONTRACT = "kion-basic-bundle-v1"
GROUP_NAMES = ("age", "sex", "genres", "content_type", "release_year", "countries", "age_rating")
UNKNOWN = "__UNKNOWN__"
CONTENT_TYPE_CATEGORIES = ["film", "series", UNKNOWN]
AGE_RATING_CATEGORIES = ["0.0", "6.0", "12.0", "16.0", "18.0", "21.0", UNKNOWN]
YEAR_COLUMNS = ["normalized_year", "missing"]
EXCLUDED_GENRES = frozenset(("популярное", "рекомендуем"))
ITEM_FIELDS = ("genres", "content_type", "release_year", "countries", "age_rating")
HELPER_PINS = {
    "local_helper": "e5f992093b53839d425ccd436dba0404862ad912b795a390d60ff6c9a2576de1",
    "sex_helper": "e524c68af1ad327c3e83959aa4503be17aabdc700beaa24793831d09a152a905",
}


def parse_tags(text, excluded=frozenset()):
    """Match the preparer's comma/strip semantics, preserving spelling and case."""
    tokens = [part.strip() for part in text.split(",") if part.strip()]
    return sorted(set(tokens) - excluded), tokens


def fit_tag_categories(items, mapping, field_index, excluded=frozenset()):
    vocabulary = set()
    for item_id in mapping:
        row = items.get(int(item_id))
        tags, _ = parse_tags(row[field_index] if row is not None else "", excluded)
        vocabulary.update(tag for tag in tags if tag != UNKNOWN)
    return [UNKNOWN] + sorted(vocabulary)


def tag_features(items, mapping, size, field_index, categories, excluded=frozenset()):
    """Transform against a frozen dictionary; unseen tags activate unknown."""
    lookup = {name: index for index, name in enumerate(categories)}
    if not categories or categories[0] != UNKNOWN or len(lookup) != len(categories):
        raise ValueError("Invalid frozen tag categories")
    features = np.zeros((size, len(categories)), np.float32)
    coverage = Counter({key: 0 for key in (
        "known_items", "unknown_items", "empty_items", "unseen_tag_items",
        "raw_unknown_tag_items", "excluded_only_items", "duplicate_tag_items",
        "missing_item_rows", "active_assignments")})
    for item_id, token in mapping.items():
        row = items.get(int(item_id))
        text = row[field_index] if row is not None else ""
        tags, original = parse_tags(text, excluded)
        coverage["missing_item_rows"] += row is None
        coverage["empty_items"] += not original
        coverage["excluded_only_items"] += bool(original) and not tags
        coverage["duplicate_tag_items"] += len(original) != len(set(original))
        coverage["raw_unknown_tag_items"] += UNKNOWN in tags
        coverage["unseen_tag_items"] += any(tag not in lookup for tag in tags)
        for tag in tags or [UNKNOWN]:
            features[token, lookup.get(tag, 0)] = 1
        coverage["known_items"] += bool(features[token, 1:].any())
        coverage["unknown_items"] += bool(features[token, 0])
        coverage["active_assignments"] += int(features[token].sum())
    return features, dict(coverage)


def categorical_features(items, mapping, size, field_index, categories):
    lookup = {name: index for index, name in enumerate(categories)}
    features = np.zeros((size, len(categories)), np.float32)
    unknown_index = lookup[UNKNOWN]
    coverage = Counter({key: 0 for key in (
        "known_items", "unknown_items", "empty_items", "invalid_nonblank_items", "missing_item_rows")})
    counts = Counter({name: 0 for name in categories})
    for item_id, token in mapping.items():
        row = items.get(int(item_id))
        text = row[field_index] if row is not None else ""
        index = lookup.get(text, unknown_index)
        features[token, index] = 1
        coverage["missing_item_rows"] += row is None
        coverage["empty_items"] += not text
        coverage["invalid_nonblank_items"] += bool(text) and index == unknown_index
        coverage["known_items"] += index != unknown_index
        coverage["unknown_items"] += index == unknown_index
        counts[categories[index]] += 1
    return features, {**dict(coverage), "category_counts": dict(counts)}


def validate_bundle_layout(user_features, item_features, groups):
    """Reject ambiguous layout and invalid encodings before training or export."""
    if not isinstance(groups, dict) or tuple(groups) != GROUP_NAMES:
        raise ValueError("Bundle groups must use the seven canonical names and order")
    arrays = {"user": user_features, "item": item_features}
    for scope, values in arrays.items():
        if not isinstance(values, np.ndarray) or values.dtype != np.dtype("float32") or values.ndim != 2:
            raise ValueError(f"{scope} features must be two-dimensional float32")
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{scope} features must be nonempty and finite")
    if item_features.shape[0] <= 3 or np.any(item_features[:3]):
        raise ValueError("PAD/BOS/UNK item rows must be all zero with real items after them")
    offsets = {"user": 0, "item": 0}
    fixed_columns = {"age": local.AGE_CATEGORIES, "sex": sex.SEX_CATEGORIES,
                     "content_type": CONTENT_TYPE_CATEGORIES, "release_year": YEAR_COLUMNS,
                     "age_rating": AGE_RATING_CATEGORIES}
    for name in GROUP_NAMES:
        entry = groups[name]
        scope = "user" if name in ("age", "sex") else "item"
        if not isinstance(entry, dict) or set(entry) != {"scope", "start", "stop", "columns"}:
            raise ValueError(f"Invalid group schema: {name}")
        start, stop, columns = entry["start"], entry["stop"], entry["columns"]
        if (entry["scope"] != scope or isinstance(start, bool) or not isinstance(start, int)
                or isinstance(stop, bool) or not isinstance(stop, int)
                or start != offsets[scope] or stop <= start or stop > arrays[scope].shape[1]
                or not isinstance(columns, list) or len(columns) != stop - start
                or any(not isinstance(value, str) or not value for value in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f"Invalid/noncontiguous group layout: {name}")
        if name in fixed_columns and columns != fixed_columns[name]:
            raise ValueError(f"Unexpected named columns: {name}")
        if name in ("genres", "countries") and (columns[0] != UNKNOWN or columns[1:] != sorted(columns[1:])):
            raise ValueError(f"Invalid frozen tag dictionary: {name}")
        offsets[scope] = stop
        values = arrays[scope][:, start:stop]
        real = values[3:] if scope == "item" else values
        if name == "release_year":
            if not np.isin(real[:, 1], (0, 1)).all() or np.any(real[real[:, 1] == 1, 0]):
                raise ValueError("Invalid release-year missing encoding")
        elif not np.isin(real, (0, 1)).all():
            raise ValueError(f"Nonbinary categorical features: {name}")
        elif name in ("genres", "countries"):
            if np.any(real.sum(axis=1) < 1):
                raise ValueError(f"Missing explicit unknown/category: {name}")
        elif np.any(real.sum(axis=1) != 1):
            raise ValueError(f"Expected exactly one category per row: {name}")
    if offsets != {scope: array.shape[1] for scope, array in arrays.items()} or user_features.shape[1] != 10:
        raise ValueError("Bundle groups must cover every feature column exactly")


def source_origin(raw, manifest_bytes, hashes):
    if manifest_bytes is None:
        return {"status": "local_source_origin_unavailable"}
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    for name, key in (("users.csv", "users"), ("items.csv", "items")):
        entries = [entry for entry in manifest.get("files", []) if entry.get("file") == name]
        if len(entries) != 1 or entries[0].get("sha256") != hashes[key]:
            raise ValueError(f"Raw {name} hash disagrees with source manifest")
    return {"status": "source_manifest_verified_for_used_raw_files",
            "path": (raw / "manifest.json").as_posix(),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "url": manifest.get("url"),
            "revision": manifest.get("tree_sha"), "archive_sha256": manifest.get("archive_sha256")}


def build(raw: Path, prepared: Path, output: Path):
    started, cpu_started = time.perf_counter(), time.process_time()
    raw, prepared, output = (Path(path).resolve() for path in (raw, prepared, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if output == raw or raw in output.parents or output == prepared or prepared in output.parents:
        raise ValueError("Output must be outside read-only raw/prepared inputs")
    paths = {"users": raw / "users.csv", "items": raw / "items.csv",
             "metadata": prepared / "metadata.json", "train": prepared / "train.npz",
             "val": prepared / "val.npz", "builder": Path(__file__).resolve(),
             "local_helper": Path(local.__file__).resolve(), "sex_helper": Path(sex.__file__).resolve()}
    for key, expected in HELPER_PINS.items():
        if local.file_hash(paths[key]) != expected:
            raise ValueError(f"Reviewed {key} source pin mismatch")
    origin_path = raw / "manifest.json"
    try:
        origin_bytes = origin_path.read_bytes()
    except FileNotFoundError:
        origin_bytes = None
    hashes = {name: local.file_hash(path) for name, path in paths.items()}
    if origin_bytes is not None:
        paths["raw_manifest"] = origin_path
        hashes["raw_manifest"] = hashlib.sha256(origin_bytes).hexdigest()
    metadata, mapping = sex.read_metadata(paths["metadata"])
    profiles = local.read_fields(paths["users"], "user_id", ("age", "sex"))
    items = local.read_fields(paths["items"], "item_id", ITEM_FIELDS)
    origin = source_origin(raw, origin_bytes, hashes)
    inputs = {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()}
    arrays, coverage, groups = {}, {}, {}
    age_profiles = {user_id: (row[0],) for user_id, row in profiles.items()}
    sex_profiles = {user_id: (row[1],) for user_id, row in profiles.items()}
    coverage["age"], coverage["sex"] = {}, {}
    for split in ("train", "val"):
        user_ids, facts = local.read_split(paths[split], metadata, split)
        inputs[split].update(facts)
        ages, coverage["age"][split] = local.age_features(age_profiles, user_ids)
        sexes, coverage["sex"][split] = sex.sex_features(sex_profiles, user_ids)
        arrays[split + "_user_features"] = np.concatenate((ages, sexes), axis=1)
    for name, start, stop, columns in (("age", 0, 7, local.AGE_CATEGORIES), ("sex", 7, 10, sex.SEX_CATEGORIES)):
        groups[name] = {"scope": "user", "start": start, "stop": stop, "columns": list(columns)}
    size = metadata["vocab_size"]
    item_blocks, fit = [], {}
    # Unique train-vocabulary cards, without target frequency weights or future IDs.
    fit_ids_hash = local.canonical_hash(sorted(int(item_id) for item_id in mapping))
    for name in ITEM_FIELDS:
        index = ITEM_FIELDS.index(name)
        if name in ("genres", "countries"):
            excluded = EXCLUDED_GENRES if name == "genres" else frozenset()
            columns = fit_tag_categories(items, mapping, index, excluded)
            values, coverage[name] = tag_features(items, mapping, size, index, columns, excluded)
            fit[name] = {"role": "SFT", "cutoff": local.TRAIN_CUTOFF,
                         "scope": "unique training-vocabulary item IDs", "item_count": len(mapping),
                         "item_ids_sha256": fit_ids_hash, "categories_sha256": local.canonical_hash(columns)}
        elif name == "release_year":
            year_items = {item_id: (row[1], row[2]) for item_id, row in items.items()}
            _, values, year_coverage, fit[name] = local.item_features(year_items, mapping, size)
            coverage[name] = {key: value for key, value in year_coverage.items() if key.startswith("release_year_")}
            coverage[name]["missing_item_rows"] = int(sum(int(item_id) not in items for item_id in mapping))
            columns = YEAR_COLUMNS
        else:
            columns = CONTENT_TYPE_CATEGORIES if name == "content_type" else AGE_RATING_CATEGORIES
            values, coverage[name] = categorical_features(items, mapping, size, index, columns)
        start = sum(block.shape[1] for block in item_blocks)
        groups[name] = {"scope": "item", "start": start, "stop": start + len(columns), "columns": list(columns)}
        item_blocks.append(values)
    arrays["item_features"] = np.concatenate(item_blocks, axis=1)
    for split in ("train", "val"):
        validate_bundle_layout(arrays[split + "_user_features"], arrays["item_features"], groups)
    # Bind the versions actually consumed, including both imported source files.
    if (origin_path.exists() != (origin_bytes is not None) or
            any(not path.is_file() or local.file_hash(path) != hashes[name] for name, path in paths.items())):
        raise ValueError("Input/source changed while building basic bundle")
    try:
        revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "features.npz", **arrays)
    manifest = {"format_version": 1, "builder_contract": CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(), "code_revision": revision,
        "inputs": inputs, "source_origin": origin,
        "features": {"file": "features.npz", "sha256": local.file_hash(output / "features.npz"),
            "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": local.array_hash(value)}
                       for name, value in arrays.items()}},
        "groups": groups, "group_order": list(GROUP_NAMES), "fit": fit,
        "age_categories": local.AGE_CATEGORIES, "sex_categories": sex.SEX_CATEGORIES,
        "prepared_sex_categories": sex.PREPARED_SEX_CATEGORIES,
        "profile_availability": "static_snapshot_unverified", "metadata_experiment": "retrospective",
        "item_metadata_availability": "static_snapshot_unverified",
        "profile_identity": "account profile; actual viewer identity unverified",
        "age_rating_review": {"21.0": "unresolved raw category; retained separately without remapping"},
        "normalization": {"version": "raw-comma-strip-deduplicate-v1", "case_and_spelling": "preserved",
            "genres_excluded": sorted(EXCLUDED_GENRES), "rare_categories": "retained",
            "unseen_tags": "activate explicit unknown, retaining any known tags",
            "age_rating": "exact stripped raw labels; unrecognized nonblank labels use unknown",
            "raw_values": "unchanged in hash-bound input CSVs"},
        "vocabulary": {"size": size, "sha256": local.canonical_hash(mapping), "special_tokens": local.SPECIAL_TOKENS},
        "cutoffs": {"train": "<2021-08-09", "validation": "2021-08-09..2021-08-15", "fit": local.TRAIN_CUTOFF},
        "coverage": {**coverage, "raw_users": len(profiles), "raw_items": len(items),
            "training_vocabulary_items": len(mapping), "missing_item_rows": int(sum(int(iid) not in items for iid in mapping))},
        "safety": {"read_splits": ["train", "val"],
            "feature_fields": ["users.age", "users.sex"] + ["items." + name for name in ITEM_FIELDS],
            "special_item_rows": "PAD/BOS/UNK all zero", "target_keys": "alignment only; never features or fit weights",
            "helper_source_sha256": dict(HELPER_PINS), "test_opened_or_hashed": False,
            "interactions_opened_or_hashed": False, "outcomes_or_other_demographics_used": False},
        "timing": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/kion/full"))
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/kion-full-context64"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/kion-basic-bundle-v1"))
    args = parser.parse_args(argv)
    manifest = build(args.raw, args.prepared, args.output)
    print(json.dumps({"output": args.output.resolve().as_posix(), "coverage": manifest["coverage"],
        "timing": manifest["timing"], "manifest_sha256": local.file_hash(args.output / "manifest.json")},
        ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
