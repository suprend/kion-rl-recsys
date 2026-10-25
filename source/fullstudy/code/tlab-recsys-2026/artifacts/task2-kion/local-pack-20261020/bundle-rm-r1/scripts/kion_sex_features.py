"""Build only named profile-sex BOS sidecars from raw users.sex, never test.

The profile is a retrospective account snapshot with unverified availability at
event time. Missing profiles, empty values and unseen labels share one unknown
model category; their diagnostic counts remain separate. No inference of sex.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_local_features as local

CONTRACT = "kion-sex-sidecars-v1"
SEX_CATEGORIES = ["sex_male", "sex_female", "sex_unknown"]
PREPARED_SEX_CATEGORIES = ["__UNKNOWN__", "Ж", "М"]
HELPER_SHA256 = "e5f992093b53839d425ccd436dba0404862ad912b795a390d60ff6c9a2576de1"
RAW_TO_CATEGORY = {"М": 0, "Ж": 1}


def read_metadata(path):
    metadata, mapping = local.read_metadata(path)
    categories = metadata.get("user_categories", {})
    if list(categories) != ["age", "income", "sex", "kids_flg"]:
        raise ValueError("Unexpected named prepared profile block order")
    if categories["sex"] != PREPARED_SEX_CATEGORIES:
        raise ValueError("Prepared sex category layout differs from approved schema")
    if metadata.get("user_feature_dim") != sum(map(len, categories.values())):
        raise ValueError("Prepared profile schema dimension mismatch")
    return metadata, mapping


def source_origin(raw, manifest_bytes, users_hash):
    """Only users.csv provenance is relevant; never open/hash items.csv."""
    if manifest_bytes is None:
        return {"status": "local_source_origin_unavailable"}
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    entries = [entry for entry in manifest.get("files", []) if entry.get("file") == "users.csv"]
    if len(entries) != 1 or entries[0].get("sha256") != users_hash:
        raise ValueError("Raw users.csv hash disagrees with source manifest")
    return {"status": "source_manifest_verified_for_users_only", "path": (raw / "manifest.json").as_posix(),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "url": manifest.get("url"),
            "revision": manifest.get("tree_sha"), "archive_sha256": manifest.get("archive_sha256")}


def sex_features(profiles, user_ids):
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    category = np.full(len(unique_users), 2, np.int64)
    reasons = []
    for index, user_id in enumerate(unique_users):
        profile = profiles.get(int(user_id))
        if profile is None:
            reason = "profile_missing"
        elif not profile[0]:
            reason = "sex_empty"
        elif profile[0] not in RAW_TO_CATEGORY:
            reason = "sex_invalid_nonblank"
        else:
            category[index] = RAW_TO_CATEGORY[profile[0]]
            reason = SEX_CATEGORIES[category[index]]
        reasons.append(reason)
    unique_counts = Counter(reasons)
    _, weights = np.unique(inverse, return_counts=True)
    row_counts = Counter()
    for reason, weight in zip(reasons, weights):
        row_counts[reason] += int(weight)
    coverage = {"rows": len(user_ids), "unique_users": len(unique_users)}
    for reason in ("profile_missing", "sex_empty", "sex_invalid_nonblank", "sex_male", "sex_female"):
        coverage[reason + "_users"] = int(unique_counts[reason])
        coverage[reason + "_rows"] = int(row_counts[reason])
    coverage["sex_known_users"] = int(np.count_nonzero(category != 2))
    coverage["sex_unknown_users"] = int(np.count_nonzero(category == 2))
    coverage["sex_known_rows"] = int(np.count_nonzero(category[inverse] != 2))
    coverage["sex_unknown_rows"] = len(user_ids) - coverage["sex_known_rows"]
    return np.eye(3, dtype=np.float32)[category[inverse]], coverage


def build(raw: Path, prepared: Path, output: Path):
    raw, prepared, output = (Path(path).resolve() for path in (raw, prepared, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if output == raw or raw in output.parents or output == prepared or prepared in output.parents:
        raise ValueError("Output must be outside read-only raw/prepared inputs")
    helper = Path(local.__file__).resolve()
    if local.file_hash(helper) != HELPER_SHA256:
        raise ValueError("Reviewed local feature helper source pin mismatch")
    paths = {"users": raw / "users.csv", "metadata": prepared / "metadata.json",
             "train": prepared / "train.npz", "val": prepared / "val.npz",
             "builder": Path(__file__).resolve(), "helper": helper}
    origin_path = raw / "manifest.json"
    try:
        origin_bytes = origin_path.read_bytes()
    except FileNotFoundError:
        origin_bytes = None
    hashes = {name: local.file_hash(path) for name, path in paths.items()}
    if origin_bytes is not None:
        paths["raw_manifest"] = origin_path
        hashes["raw_manifest"] = hashlib.sha256(origin_bytes).hexdigest()
    metadata, mapping = read_metadata(paths["metadata"])
    profiles = local.read_fields(paths["users"], "user_id", ("sex",))
    origin = source_origin(raw, origin_bytes, hashes["users"])
    inputs = {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()}
    features, coverage = {}, {}
    for split in ("train", "val"):
        user_ids, facts = local.read_split(paths[split], metadata, split)
        inputs[split].update(facts)
        features[split + "_user_sex"], coverage[split] = sex_features(profiles, user_ids)
    if (origin_path.exists() != (origin_bytes is not None) or
            any(not path.is_file() or local.file_hash(path) != hashes[name] for name, path in paths.items())):
        raise ValueError("Input/source changed while building sex features")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "sex.npz", **features)
    manifest = {"format_version": 1, "builder_contract": CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(), "inputs": inputs, "source_origin": origin,
        "features": {"sex": {"file": "sex.npz", "sha256": local.file_hash(output / "sex.npz"),
            "categories": SEX_CATEGORIES, "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": local.array_hash(value)} for name, value in features.items()},
            "unknown_encoding": "own one-hot column; profile missing, empty and unseen raw label merged"}},
        "sex_categories": SEX_CATEGORIES, "prepared_sex_categories": PREPARED_SEX_CATEGORIES,
        "profile_availability": "static_snapshot_unverified", "metadata_experiment": "retrospective",
        "profile_identity": "account profile; actual viewer identity unverified",
        "vocabulary": {"size": metadata["vocab_size"], "sha256": local.canonical_hash(mapping),
                       "special_tokens": local.SPECIAL_TOKENS},
        "cutoffs": {"train": "<2021-08-09", "validation": "2021-08-09..2021-08-15", "fit": local.TRAIN_CUTOFF},
        "coverage": {"sex": coverage, "raw_users": len(profiles)},
        "safety": {"read_splits": ["train", "val"], "feature_fields": ["users.sex"],
            "target_keys": "alignment only; never features or fit weights", "helper_source_sha256": HELPER_SHA256,
            "test_opened_or_hashed": False, "interactions_opened_or_hashed": False,
            "items_opened_or_hashed": False, "outcomes_or_other_demographics_used": False}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/kion/full"))
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/kion-full-context64"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/kion-sex-features-v1"))
    args = parser.parse_args(argv)
    manifest = build(args.raw, args.prepared, args.output)
    print(json.dumps({"output": args.output.resolve().as_posix(), "coverage": manifest["coverage"]},
                     ensure_ascii=False, indent=2))
    return manifest


if __name__ == "__main__":
    main()
