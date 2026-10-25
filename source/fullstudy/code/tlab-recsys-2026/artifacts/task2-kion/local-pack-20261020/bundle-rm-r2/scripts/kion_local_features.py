"""Build frozen, row-aligned KION age/content/year sidecars without reading test.

Only raw users.age and items.content_type/release_year are feature inputs. The
provided metadata is a retrospective snapshot, not historically verified as-of
data. This builder never opens interactions.csv, test.npz, or user_features.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np


AGE_CATEGORIES = ["__UNKNOWN__", "age_18_24", "age_25_34", "age_35_44",
                  "age_45_54", "age_55_64", "age_65_inf"]
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
TRAIN_CUTOFF = "2021-08-09"
VAL_STOP = "2021-08-16"
CONTRACT = "kion-local-sidecars-v1"


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    """Bind both interpretation and C-order bytes, including duplicate rows."""
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def strict_id(value, label):
    text = str(value).strip()
    try:
        result = int(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error
    if result < 0 or str(result) != text:
        raise ValueError(f"Invalid {label}")
    return result


def read_fields(path, id_field, fields):
    """Project allowed fields only; reject ambiguous keys even outside fit IDs."""
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"Duplicate or missing CSV headers: {Path(path).name}")
        if not set((id_field, *fields)).issubset(reader.fieldnames):
            raise ValueError(f"Missing required CSV fields: {Path(path).name}")
        for row in reader:
            if None in row or any(row.get(field) is None for field in (id_field, *fields)):
                raise ValueError(f"Malformed CSV row: {Path(path).name}")
            key = strict_id(row[id_field], id_field)
            if key in result:
                raise ValueError(f"Duplicate {id_field}: {Path(path).name}")
            result[key] = tuple(row[field].strip() for field in fields)
    return result


def read_metadata(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1 or metadata.get("special_tokens") != SPECIAL_TOKENS:
        raise ValueError("Unsupported metadata/special-token contract")
    if metadata.get("split_dates", {}).get("train") != "<2021-08-09" or metadata.get("split_dates", {}).get("val") != "2021-08-09..2021-08-15":
        raise ValueError("Unexpected train/validation cutoffs")
    if metadata.get("user_categories", {}).get("age") != AGE_CATEGORIES:
        raise ValueError("Age category order differs from approved metadata schema")
    size = metadata.get("vocab_size")
    mapping = metadata.get("item_id_to_token")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 3 or not isinstance(mapping, dict):
        raise ValueError("Invalid vocabulary")
    checked = {}
    for item_id, token in mapping.items():
        strict_id(item_id, "vocabulary item_id")
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("Invalid item token")
        checked[item_id] = token
    if len(checked) != size - 3 or sorted(checked.values()) != list(range(3, size)):
        raise ValueError("Vocabulary tokens must be one-to-one and contiguous after specials")
    return metadata, checked


def read_split(path, metadata, split):
    # Explicit members avoid accidentally loading x, y, outcomes, demographics or
    # a future test member. The target keys bind alignment; they never fit features.
    with np.load(path, allow_pickle=False) as archive:
        keys = [archive[field] for field in ("user_id", "target_item_id", "target_date")]
    if any(a.ndim != 1 or a.dtype != np.dtype("int64") for a in keys):
        raise ValueError(f"{split} row keys must be one-dimensional int64")
    count = len(keys[0])
    if not count or any(len(a) != count for a in keys):
        raise ValueError(f"{split} row-key lengths differ or are empty")
    if metadata.get("splits", {}).get(split, {}).get("saved") != count:
        raise ValueError(f"{split} rows disagree with metadata")
    users, items, days = keys
    if np.any(users < 0) or np.any(items < 0):
        raise ValueError(f"{split} negative user/item ID")
    cutoff, stop = date.fromisoformat(TRAIN_CUTOFF).toordinal(), date.fromisoformat(VAL_STOP).toordinal()
    valid = ((days > 0) & (days < cutoff)) if split == "train" else ((days >= cutoff) & (days < stop))
    if not valid.all():
        raise ValueError(f"{split} target date outside fixed split")
    return users, {
        "rows": count, "unique_users": int(len(np.unique(users))),
        "user_id_sha256": array_hash(users),
        "row_keys_sha256": canonical_hash({name: array_hash(a) for name, a in
                                            zip(("user_id", "target_item_id", "target_date"), keys)}),
    }


def age_features(profiles, user_ids):
    categories = {value: n for n, value in enumerate(AGE_CATEGORIES)}
    reasons = ("profile_missing", "age_empty", "age_invalid_nonblank", *AGE_CATEGORIES[1:])
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    category = np.zeros(len(unique_users), np.int64)
    reason_ids = np.empty(len(unique_users), np.int64)
    for n, user_id in enumerate(unique_users):
        profile = profiles.get(int(user_id))
        if profile is None:
            reason_ids[n] = 0
        elif not profile[0]:
            reason_ids[n] = 1
        elif profile[0] not in categories or profile[0] == "__UNKNOWN__":
            reason_ids[n] = 2
        else:
            category[n] = categories[profile[0]]
            reason_ids[n] = 2 + category[n]
    row_category = category[inverse]
    row_weights = np.bincount(inverse, minlength=len(unique_users))
    user_counts = np.bincount(reason_ids, minlength=len(reasons))
    row_counts = np.bincount(reason_ids, weights=row_weights, minlength=len(reasons))
    coverage = Counter()
    for n, reason in enumerate(reasons):
        coverage[reason + "_rows"] = int(row_counts[n])
        coverage[reason + "_users"] = int(user_counts[n])
    values = np.eye(len(AGE_CATEGORIES), dtype=np.float32)[row_category]
    coverage.update(rows=len(user_ids), unique_users=len(unique_users),
                    age_known_rows=int(np.count_nonzero(row_category)),
                    age_unknown_rows=int(np.count_nonzero(row_category == 0)),
                    age_known_users=int(np.count_nonzero(category)),
                    age_unknown_users=int(np.count_nonzero(category == 0)))
    return values, dict(coverage)


def parse_year(text):
    if not text:
        return None, "empty"
    try:
        value = float(text)
    except ValueError:
        return None, "invalid_nonblank"
    if not math.isfinite(value) or value != math.floor(value) or not 1 <= value <= 9999:
        return None, "invalid_nonblank"
    return int(value), "known"


def item_features(items, mapping, size):
    types = np.zeros((size, 2), np.float32)
    years = np.zeros((size, 2), np.float32)
    coverage = Counter()
    known_years, known_tokens, fit_ids = [], [], []
    for item_id, token in sorted(mapping.items(), key=lambda pair: pair[1]):
        row = items.get(int(item_id))
        if row is None:
            coverage["missing_item_rows"] += 1
            content_type, raw_year = "", ""
        else:
            content_type, raw_year = row
        if content_type in ("film", "series"):
            types[token, ("film", "series").index(content_type)] = 1
            coverage[content_type] += 1
        else:
            coverage["content_type_empty" if not content_type else "content_type_invalid_nonblank"] += 1
        year, reason = parse_year(raw_year)
        coverage["release_year_" + reason] += 1
        if year is None:
            years[token, 1] = 1
        else:
            known_years.append(year)
            known_tokens.append(token)
            fit_ids.append(int(item_id))
    if not known_years:
        raise ValueError("No valid release years in training vocabulary")
    numeric = np.asarray(known_years, dtype=np.float64)
    mean, std = float(numeric.mean()), float(numeric.std(ddof=0))
    scale = std if std > 0 else 1.0
    years[np.asarray(known_tokens), 0] = ((numeric - mean) / scale).astype(np.float32)
    fit = {"role": "SFT", "cutoff": TRAIN_CUTOFF, "scope": "unique training-vocabulary item IDs with valid known year",
           "known_count": len(known_years), "mean": mean, "population_std": std, "scale": scale,
           "zero_std_fallback": 1.0, "year_validity": "finite integer calendar year 1..9999",
           "item_ids_sha256": canonical_hash(sorted(fit_ids))}
    return types, years, dict(coverage), fit


def source_origin(raw, manifest_bytes):
    path = raw / "manifest.json"
    if manifest_bytes is None:
        return {"status": "local_source_origin_unavailable"}
    # Parse and bind exactly the same snapshot. A later read must never label
    # different, unverified origin bytes as verified provenance.
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    expected = {entry["file"]: entry["sha256"] for entry in manifest.get("files", [])}
    for name in ("users.csv", "items.csv"):
        if name not in expected or expected[name] != file_hash(raw / name):
            raise ValueError(f"Raw {name} hash disagrees with source manifest")
    return {"status": "source_manifest_verified_for_used_raw_files", "path": path.as_posix(),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(), "url": manifest.get("url"), "revision": manifest.get("tree_sha"),
            "archive_sha256": manifest.get("archive_sha256")}


def build(raw: Path, prepared: Path, output: Path):
    raw, prepared, output = (Path(p).resolve() for p in (raw, prepared, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    if output == raw or raw in output.parents or output == prepared or prepared in output.parents:
        raise ValueError("Output must be outside read-only raw/prepared inputs")
    paths = {"users": raw / "users.csv", "items": raw / "items.csv", "metadata": prepared / "metadata.json",
             "train": prepared / "train.npz", "val": prepared / "val.npz", "builder": Path(__file__).resolve()}
    origin_path = raw / "manifest.json"
    try:
        origin_bytes = origin_path.read_bytes()
    except FileNotFoundError:
        origin_bytes = None
    hashes = {name: file_hash(path) for name, path in paths.items()}
    if origin_bytes is not None:
        paths["raw_manifest"] = origin_path
        hashes["raw_manifest"] = hashlib.sha256(origin_bytes).hexdigest()
    metadata, mapping = read_metadata(paths["metadata"])
    profiles = read_fields(paths["users"], "user_id", ("age",))
    items = read_fields(paths["items"], "item_id", ("content_type", "release_year"))
    origin = source_origin(raw, origin_bytes)
    inputs = {name: {"path": path.as_posix(), "sha256": hashes[name]} for name, path in paths.items()}
    age, age_coverage = {}, {}
    for split in ("train", "val"):
        user_ids, facts = read_split(paths[split], metadata, split)
        inputs[split].update(facts)
        age[f"{split}_user_age"], age_coverage[split] = age_features(profiles, user_ids)
    types, years, item_coverage, fit = item_features(items, mapping, metadata["vocab_size"])
    # Bind the version actually consumed, not a file replaced during preparation.
    if (origin_path.exists() != (origin_bytes is not None) or
            any(not path.is_file() or file_hash(path) != hashes[name] for name, path in paths.items())):
        raise ValueError("Input/source changed while building features")
    try:
        revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "age.npz", **age)
    np.save(output / "item_content_type.npy", types, allow_pickle=False)
    np.save(output / "item_release_year.npy", years, allow_pickle=False)
    feature_manifest = {
        "age": {"file": "age.npz", "sha256": file_hash(output / "age.npz"), "categories": AGE_CATEGORIES,
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_hash(value)}
                           for key, value in age.items()},
                "knownness_control": "unknown=age[:,0]; known=1-age[:,0]; no other profile fields"},
    }
    for name, filename, values, columns in (
        ("content_type", "item_content_type.npy", types, ["film", "series"]),
        ("release_year", "item_release_year.npy", years, ["normalized_year", "missing"]),
    ):
        feature_manifest[name] = {"file": filename, "sha256": file_hash(output / filename),
                                  "shape": list(values.shape), "dtype": str(values.dtype),
                                  "array_sha256": array_hash(values), "columns": columns}
    feature_manifest["content_type"]["unknown_encoding"] = "all_zero; implicit unknown for real items"
    feature_manifest["release_year"]["fit"] = fit
    manifest = {
        "format_version": 1, "builder_contract": CONTRACT,
        "created_utc": datetime.now(timezone.utc).isoformat(), "code_revision": revision,
        "inputs": inputs, "source_origin": origin, "features": feature_manifest,
        "age_categories": AGE_CATEGORIES, "profile_availability": "static_snapshot_unverified",
        "metadata_experiment": "retrospective", "item_metadata_availability": "static_snapshot_unverified",
        "vocabulary": {"size": metadata["vocab_size"], "sha256": canonical_hash(mapping),
                       "special_tokens": SPECIAL_TOKENS},
        "cutoffs": {"train": "<2021-08-09", "validation": "2021-08-09..2021-08-15", "fit": TRAIN_CUTOFF},
        "coverage": {"age": age_coverage, "items": item_coverage,
                     "raw_users": len(profiles), "raw_items": len(items)},
        "safety": {"read_splits": ["train", "val"], "feature_fields": ["users.age", "items.content_type", "items.release_year"],
                   "special_item_rows": "PAD/BOS/UNK all zero", "target_keys": "alignment only; never features or fit weights",
                   "test_opened_or_hashed": False, "interactions_opened_or_hashed": False},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8", newline="\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/kion/full"))
    parser.add_argument("--prepared", type=Path, default=Path("data/processed/kion-full-context64"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/kion-night-features-v1"))
    args = parser.parse_args()
    manifest = build(args.raw, args.prepared, args.output)
    print(json.dumps({"output": args.output.as_posix(), "coverage": manifest["coverage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
