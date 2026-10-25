"""Build the CPU-only all-public-history exclusion mask for KION submissions.

The fixed vocabulary comes from training metadata. The sample defines the exact
user order, including cold users. Only user/item IDs and last-watch dates are
read from interactions; outcomes and model weights are never used.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np


CONTRACT = "kion-competition-seen-v1"
CUTOFF_EXCLUSIVE = "2021-08-23"
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
SAMPLE_SHA256 = "97f45058710308417a348098452dae7b91795e568504809f966582a375446db8"
INTERACTIONS_SHA256 = "201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf"
METADATA_SHA256 = "f02acaa9a5e5532e79b060279ba8a5027f268f202132e85ebab97cd4912fbd11"
SAMPLE_USERS = 193113
COMMON_PATH = Path(__file__).with_name("kion_competition_data.py").resolve()
_spec = importlib.util.spec_from_file_location("_competition_seen_common", COMMON_PATH)
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)
file_hash, array_hash, canonical_hash = common.file_hash, common.array_hash, common.canonical_hash


def read_metadata(path):
    metadata = common._read_json(path)
    size, mapping = metadata.get("vocab_size"), metadata.get("item_id_to_token")
    if (metadata.get("format_version") != 1 or metadata.get("context") != 64
            or metadata.get("special_tokens") != SPECIAL_TOKENS
            or type(size) is not int or size <= 3 or not isinstance(mapping, dict)):
        raise ValueError("Invalid fixed metadata/vocabulary contract")
    for item_id, token in mapping.items():
        common._strict_id(item_id, "vocabulary item_id")
        if type(token) is not int:
            raise ValueError("Invalid vocabulary token")
    if len(mapping) != size - 3 or sorted(mapping.values()) != list(range(3, size)):
        raise ValueError("Vocabulary must map each real item to one token >=3")
    return metadata, mapping


def make_seen_csr(user_ids, history_users, history_tokens):
    """Deduplicate all known user/token pairs and store them in sample order."""
    if len(history_users) != len(history_tokens):
        raise ValueError("History user/token lengths differ")
    known = history_tokens >= 3
    order = np.argsort(user_ids)
    sorted_users = user_ids[order]
    positions = np.searchsorted(sorted_users, history_users[known])
    if (np.any(positions >= len(user_ids))
            or not np.array_equal(sorted_users[positions], history_users[known])):
        raise ValueError("History contains users outside official sample")
    rows, tokens = order[positions], history_tokens[known]
    pair_order = np.lexsort((tokens, rows))
    rows, tokens = rows[pair_order], tokens[pair_order]
    keep = np.ones(len(rows), dtype=bool)
    keep[1:] = (rows[1:] != rows[:-1]) | (tokens[1:] != tokens[:-1])
    rows, tokens = rows[keep], tokens[keep]
    counts = np.bincount(rows, minlength=len(user_ids))
    indptr = np.zeros(len(user_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    _, history_counts = np.unique(history_users, return_counts=True)
    history_user_count = len(history_counts)
    coverage = {
        "sample_users": len(user_ids), "history_users": history_user_count,
        "cold_users": len(user_ids) - history_user_count,
        "users_with_known_history": int(np.count_nonzero(counts)),
        "users_with_only_oov_history": history_user_count - int(np.count_nonzero(counts)),
        "users_with_oov_history": len(np.unique(history_users[~known])),
        "users_with_history_over63": int(np.count_nonzero(history_counts > 63)),
        "known_history_rows": int(np.count_nonzero(known)),
        "oov_history_rows": int(np.count_nonzero(~known)),
        "distinct_known_user_item_pairs": len(tokens),
        "deduplicated_known_pair_rows": int(np.count_nonzero(known)) - len(tokens),
        "minimum_seen_items": int(counts.min()), "maximum_seen_items": int(counts.max()),
    }
    return {"user_ids": user_ids.copy(), "indptr": indptr, "token_ids": tokens.copy()}, coverage


def validate_arrays(arrays, user_ids, vocab_size):
    if set(arrays) != {"user_ids", "indptr", "token_ids"}:
        raise ValueError("Seen cache must contain exactly user_ids, indptr, token_ids")
    if any(value.dtype != np.dtype("int64") or value.ndim != 1 for value in arrays.values()):
        raise ValueError("Seen arrays must be one-dimensional int64")
    if not np.array_equal(arrays["user_ids"], user_ids):
        raise ValueError("Seen user IDs/order differ from official sample")
    pointers, tokens = arrays["indptr"], arrays["token_ids"]
    if (len(pointers) != len(user_ids) + 1 or pointers[0] != 0
            or pointers[-1] != len(tokens) or np.any(pointers[1:] < pointers[:-1])
            or np.any(tokens < 3) or np.any(tokens >= vocab_size)):
        raise ValueError("Invalid seen CSR pointers or real-token range")
    # Adjacent tokens must increase within a row. Boundaries may repeat/decrease.
    boundary = np.zeros(max(0, len(tokens) - 1), dtype=bool)
    ends = pointers[1:-1]
    ends = ends[(ends > 0) & (ends < len(tokens))]
    boundary[ends - 1] = True
    if np.any((tokens[1:] <= tokens[:-1]) & ~boundary):
        raise ValueError("Seen tokens must be unique and sorted within each user")


def _raw_origin(path, interaction_path, interaction_hash):
    value = common._read_json(path)
    matches = [entry for entry in value.get("files", [])
               if isinstance(entry, dict) and entry.get("file") == interaction_path.name]
    if (len(matches) != 1 or matches[0].get("sha256") != interaction_hash
            or matches[0].get("bytes") != interaction_path.stat().st_size):
        raise ValueError("Raw interactions differ from source manifest")
    return {"status": "source_manifest_verified_for_interactions_only", "url": value.get("url"),
            "revision": value.get("tree_sha"), "archive_sha256": value.get("archive_sha256")}


def build_competition_seen(*, sample, interactions, metadata, output, raw_manifest=None,
                           sample_manifest=None, sample_sha256=None, interactions_sha256=None,
                           metadata_sha256=None, expected_users=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    paths = {key: Path(value).resolve() for key, value in {
        "sample": sample, "interactions": interactions, "metadata": metadata,
        "builder": __file__, "common_helper": COMMON_PATH}.items()}
    paths["raw_manifest"] = Path(raw_manifest or paths["interactions"].with_name("manifest.json")).resolve()
    paths["sample_manifest"] = Path(sample_manifest or paths["sample"].with_suffix(".source.json")).resolve()
    if any(output == path or output == path.parent or path.parent in output.parents for path in paths.values()):
        raise ValueError("Output must be outside read-only input/source directories")
    hashes = {key: file_hash(path) for key, path in paths.items()}
    for key, expected in (("sample", sample_sha256), ("interactions", interactions_sha256),
                          ("metadata", metadata_sha256)):
        if expected is not None and hashes[key] != expected:
            raise ValueError(f"Expected {key} SHA-256 differs from input")
    meta, mapping = read_metadata(paths["metadata"])
    users = common.read_sample(paths["sample"])
    if expected_users is not None and len(users) != expected_users:
        raise ValueError("Official sample user count differs from expected count")
    origin = _raw_origin(paths["raw_manifest"], paths["interactions"], hashes["interactions"])
    sample_origin = common._verify_sample_origin(paths["sample_manifest"], paths["sample"], hashes["sample"], users)
    history_users, tokens, facts, ranges = common.read_histories(paths["interactions"], users, mapping)
    arrays, coverage = make_seen_csr(users, history_users, tokens)
    validate_arrays(arrays, users, meta["vocab_size"])
    manifest = {
        "format_version": 1, "builder_contract": CONTRACT, "kind": "competition_seen",
        "cutoff_exclusive": CUTOFF_EXCLUSIVE, "special_tokens": SPECIAL_TOKENS,
        "vocab_size": meta["vocab_size"], "item_id_to_token_sha256": canonical_hash(mapping),
        "sample_user_ids_sha256": array_hash(users),
        "inputs": {key: {"path": path.as_posix(), "sha256": hashes[key], "bytes": path.stat().st_size}
                   for key, path in paths.items()},
        "source_origin": origin, "sample_origin": sample_origin, "raw_date_range": ranges,
        "coverage": {"interactions": facts, "seen": coverage},
        "encoding": "sample-ordered CSR; all distinct known tokens >=3 before cutoff, sorted per user; OOV omitted",
        "history_window": "all public history before cutoff; no 63-item truncation",
        "feature_fit": "none; immutable training vocabulary",
        "interaction_fields_used": ["user_id", "item_id", "last_watch_dt"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".building-", dir=output.parent) as staging:
        staging = Path(staging)
        np.savez_compressed(staging / "seen.npz", **arrays)
        manifest["seen"] = {"file": "seen.npz", "sha256": file_hash(staging / "seen.npz"),
                            "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                             "sha256": array_hash(value)} for key, value in arrays.items()}}
        if any(not path.is_file() or file_hash(path) != hashes[key] for key, path in paths.items()):
            raise ValueError("Input/source changed while building seen mask")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8", newline="\n")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
        os.rename(staging, output)
    return manifest


def preflight_seen(*, data, sample, metadata):
    """Check source/artifact pins and CSR invariants before either export mode."""
    paths = {"data": Path(data).resolve(), "sample": Path(sample).resolve(), "metadata": Path(metadata).resolve()}
    paths.update(cache=paths["data"] / "seen.npz", manifest=paths["data"] / "manifest.json")
    manifest = common._read_json(paths["manifest"])
    meta, mapping = read_metadata(paths["metadata"])
    users = common.read_sample(paths["sample"])
    if (manifest.get("format_version") != 1 or manifest.get("builder_contract") != CONTRACT
            or manifest.get("kind") != "competition_seen" or manifest.get("cutoff_exclusive") != CUTOFF_EXCLUSIVE
            or manifest.get("special_tokens") != SPECIAL_TOKENS or manifest.get("vocab_size") != meta["vocab_size"]
            or manifest.get("item_id_to_token_sha256") != canonical_hash(mapping)
            or manifest.get("sample_user_ids_sha256") != array_hash(users)
            or manifest.get("interaction_fields_used") != ["user_id", "item_id", "last_watch_dt"]):
        raise ValueError("Seen cache metadata/cutoff/sample contract mismatch")
    hashes = {key: file_hash(paths[key]) for key in ("sample", "metadata", "cache", "manifest")}
    expected_sources = {"sample": paths["sample"], "metadata": paths["metadata"]}
    executing_sources = {"builder": Path(__file__).resolve(), "common_helper": COMMON_PATH}
    inputs = manifest.get("inputs", {})
    if set(inputs) != {"sample", "interactions", "metadata", "raw_manifest", "sample_manifest", "builder", "common_helper"}:
        raise ValueError("Seen cache input provenance is incomplete")
    for key, declaration in inputs.items():
        source = Path(declaration["path"]).resolve()
        if key in expected_sources and source != expected_sources[key]:
            raise ValueError(f"Seen source path mismatch: {key}")
        if key in executing_sources and declaration.get("sha256") != file_hash(executing_sources[key]):
            raise ValueError(f"Seen executing source hash mismatch: {key}")
        if declaration.get("sha256") != file_hash(source) or declaration.get("bytes") != source.stat().st_size:
            raise ValueError(f"Seen source hash/size mismatch: {key}")
    _raw_origin(Path(inputs["raw_manifest"]["path"]), Path(inputs["interactions"]["path"]), inputs["interactions"]["sha256"])
    common._verify_sample_origin(Path(inputs["sample_manifest"]["path"]), paths["sample"], hashes["sample"], users)
    declaration = manifest.get("seen", {})
    if declaration.get("file") != "seen.npz" or declaration.get("sha256") != hashes["cache"]:
        raise ValueError("Seen artifact hash mismatch")
    with np.load(paths["cache"], allow_pickle=False) as archive:
        arrays = dict(archive)
    validate_arrays(arrays, users, meta["vocab_size"])
    if set(declaration.get("arrays", {})) != set(arrays):
        raise ValueError("Seen array declarations are incomplete")
    for key, values in arrays.items():
        declared = declaration["arrays"][key]
        if (declared.get("shape") != list(values.shape) or declared.get("dtype") != str(values.dtype)
                or declared.get("sha256") != array_hash(values)):
            raise ValueError(f"Seen array declaration/hash mismatch: {key}")
    return {"paths": paths, "hashes": hashes, "manifest": manifest, "arrays": arrays}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sample", "interactions", "metadata", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path)
    parser.add_argument("--sample-manifest", type=Path)
    parser.add_argument("--sample-sha256", default=SAMPLE_SHA256)
    parser.add_argument("--interactions-sha256", default=INTERACTIONS_SHA256)
    parser.add_argument("--metadata-sha256", default=METADATA_SHA256)
    parser.add_argument("--expected-users", type=int, default=SAMPLE_USERS)
    args = parser.parse_args()
    manifest = build_competition_seen(**vars(args))
    print(json.dumps({"output": args.output.resolve().as_posix(), "coverage": manifest["coverage"],
                      "seen_sha256": manifest["seen"]["sha256"],
                      "manifest_sha256": file_hash(args.output / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
