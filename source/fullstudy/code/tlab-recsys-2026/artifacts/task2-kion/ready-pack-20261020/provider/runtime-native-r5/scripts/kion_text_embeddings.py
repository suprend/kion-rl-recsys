"""Frozen local E5 item caches and exact visible-H63 text profiles.

No downloader, outcome fitting, test-file discovery, or external inference API.
Each field has its own immutable cache; callers explicitly select a field.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

CONTRACT = "kion-text-embeddings-v1"
PROFILE_CONTRACT = "kion-personal-text-h63-v1"
FIELDS = ("title", "title_orig", "description")
SPECIALS = {"PAD": 0, "BOS": 1, "UNK": 2}
MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
DIMENSION = 384
EPSILON = 1e-12
NORM_ATOL = 1e-4
PROFILE_DATE_BOUNDS = {
    "train": (1, date(2021, 8, 9).toordinal()),
    "val": (date(2021, 8, 9).toordinal(), date(2021, 8, 16).toordinal()),
    "test": (date(2021, 8, 16).toordinal(), date(2021, 8, 23).toordinal()),
}
MODEL_FILES = {
    "config.json": {"size": 655, "sha256": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"},
    "model.safetensors": {"size": 470641600, "sha256": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"},
    "tokenizer.json": {"size": 17082730, "sha256": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"},
    "tokenizer_config.json": {"size": 443, "sha256": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"},
    "special_tokens_map.json": {"size": 167, "sha256": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"},
}
PREPROCESSING = {
    "version": "nfc-unicode-whitespace-v1", "null": "empty", "unicode": "NFC",
    "whitespace": "collapse_unicode_whitespace_then_strip", "prefix": "query: ",
    "pooling": "last_hidden_masked_mean_including_special_tokens_excluding_pad",
    "normalization": "L2_float32", "max_length": 512, "truncation_side": "right",
    "special_tokens": "tokenizer_default_add_special_tokens_true", "storage_dtype": "float32",
    "norm_atol": NORM_ATOL, "metadata_availability": "retrospective_catalog_snapshot_not_historical_asof",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_hash(value):
    value = np.asarray(value)
    result = hashlib.sha256(json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                                      sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8", newline="\n")


def normalize_text(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Text must be a string or null")
    return " ".join(unicodedata.normalize("NFC", value).split())


def _mapping(metadata):
    if metadata.get("special_tokens") != SPECIALS:
        raise ValueError("Special-token mapping mismatch")
    size, mapping = metadata.get("vocab_size"), metadata.get("item_id_to_token")
    if isinstance(size, bool) or not isinstance(size, int) or size < 4 or not isinstance(mapping, dict):
        raise ValueError("Invalid vocabulary")
    result = []
    for item_id, token in mapping.items():
        if not isinstance(item_id, str) or re.fullmatch(r"0|[1-9][0-9]*", item_id) is None:
            raise ValueError("Noncanonical item ID")
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("Invalid vocabulary token")
        result.append((token, item_id))
    result.sort()
    if [token for token, _ in result] != list(range(3, size)):
        raise ValueError("Vocabulary must be contiguous and one-to-one after specials")
    return [item_id for _, item_id in result]


def plan_catalog(items_path, metadata_path, field):
    """Read only selected catalogue field and metadata; never read histories/outcomes."""
    if field not in FIELDS:
        raise ValueError("Choose exactly one approved text field")
    items_path, metadata_path = Path(items_path), Path(metadata_path)
    pins = {name: {"path": path.resolve().as_posix(), "sha256": file_hash(path)}
            for name, path in (("items", items_path), ("metadata", metadata_path))}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ordered = _mapping(metadata)
    texts = {}
    with items_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames)
                or not {"item_id", field}.issubset(reader.fieldnames)):
            raise ValueError("Missing/duplicate catalogue headers")
        for row in reader:
            item_id, text = row.get("item_id"), row.get(field)
            if (None in row or item_id is None or text is None
                    or re.fullmatch(r"0|[1-9][0-9]*", item_id) is None or item_id in texts):
                raise ValueError("Malformed/duplicate catalogue item")
            texts[item_id] = text
    if not set(ordered).issubset(texts):
        raise ValueError("Train-known catalogue item missing from source")
    raw = [texts[item_id] for item_id in ordered]
    normalized = [normalize_text(value) for value in raw]
    for pin in pins.values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("Source changed during catalogue planning")
    return {"field": field, "ordered_item_ids": ordered, "vocab_size": metadata["vocab_size"],
            "vocabulary_sha256": canonical_hash({"ordered_item_ids": ordered, "special_tokens": SPECIALS}),
            "source_pins": pins, "raw_text_sha256": canonical_hash(raw),
            "normalized_text_sha256": canonical_hash(normalized), "texts": normalized,
            "raw_catalog_rows": len(texts),
            "empty_counts": {"raw_empty": sum(not value for value in raw),
                             "normalized_empty": sum(not value for value in normalized),
                             "whitespace_only": sum(bool(a) and not b for a, b in zip(raw, normalized))}}


def validate_local_model(directory):
    """Exact local-only artifact admission, before importing any model libraries."""
    directory = Path(directory)
    for name, expected in MODEL_FILES.items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != expected["size"] or file_hash(path) != expected["sha256"]:
            raise ValueError("Missing or mismatched pinned model file: " + name)
    return {"model_id": MODEL_ID, "revision": REVISION, "tokenizer_revision": REVISION,
            "files": MODEL_FILES, "dimension": DIMENSION, "production_encoder": True}


def masked_mean_l2(hidden, attention_mask):
    """Small CPU reference for the same PAD-excluded float32 pooling contract."""
    hidden, mask = np.asarray(hidden, np.float32), np.asarray(attention_mask)
    if (hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not np.isfinite(hidden).all()
            or not np.isin(mask, (0, 1)).all() or np.any(mask.sum(axis=1) == 0)):
        raise ValueError("Invalid hidden states/attention mask")
    mean = np.where(mask[..., None].astype(bool), hidden, 0).sum(axis=1, dtype=np.float32)
    mean /= mask.sum(axis=1, dtype=np.float32)[:, None]
    norms = np.linalg.norm(mean, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= EPSILON):
        raise ValueError("Encoder produced nonfinite/zero embedding")
    return (mean / norms[:, None]).astype(np.float32)


class E5Encoder:
    """Production frozen encoder. Model/tokenizer loading cannot access network."""
    def __init__(self, local_model_dir, *, device="cpu", compute_dtype="float32", threads=2):
        self.spec = validate_local_model(local_model_dir)
        if device not in ("cpu", "cuda") or compute_dtype not in ("float32", "float16"):
            raise ValueError("Unsupported device/precision")
        if compute_dtype == "float16" and device != "cuda":
            raise ValueError("Float16 encoder requires CUDA")
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 2:
            raise ValueError("Encoder threads must be 1 or 2")
        import torch
        from transformers import AutoModel, AutoTokenizer
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir), use_fast=True,
                                                     local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise ValueError("Pinned fast tokenizer required")
        self.tokenizer.truncation_side, self.tokenizer.padding_side = "right", "right"
        dtype = torch.float16 if compute_dtype == "float16" else torch.float32
        self.model = AutoModel.from_pretrained(str(local_model_dir), local_files_only=True,
                                              trust_remote_code=False, use_safetensors=True,
                                              torch_dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != DIMENSION:
            raise ValueError("Encoder dimension mismatch")
        self.spec.update({"compute_dtype": compute_dtype, "device": device,
                          "pooling_dtype": "float32", "threads": threads,
                          "library_versions": {name: importlib.metadata.version(name)
                                               for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy")}})

    def encode(self, prefixed_texts):
        if not prefixed_texts:
            return np.empty((0, DIMENSION), np.float32), np.empty(0, np.int32)
        if any(not text.startswith("query: ") or not text[7:] for text in prefixed_texts):
            raise ValueError("Encoder expects nonempty query-prefixed texts")
        lengths = np.asarray([len(row) for row in self.tokenizer(prefixed_texts, truncation=False,
                              padding=False, add_special_tokens=True)["input_ids"]], np.int32)
        batch = self.tokenizer(prefixed_texts, max_length=512, padding=True, truncation=True,
                               add_special_tokens=True, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state.float()
            mask = batch["attention_mask"].bool()
            mean = hidden.masked_fill(~mask[..., None], 0).sum(dim=1) / mask.sum(dim=1)[:, None]
            norms = self.torch.linalg.vector_norm(mean, dim=1)
            if not self.torch.isfinite(mean).all() or self.torch.any(norms <= EPSILON):
                raise ValueError("Encoder produced invalid vectors")
            vectors = (mean / norms[:, None]).cpu().numpy().astype(np.float32)
        return vectors, lengths


def _descriptor(plan, encoder_spec):
    return {"contract": CONTRACT, **{key: plan[key] for key in (
            "field", "ordered_item_ids", "vocab_size", "vocabulary_sha256", "source_pins",
            "raw_text_sha256", "normalized_text_sha256", "raw_catalog_rows", "empty_counts")},
            "preprocessing": PREPROCESSING, "encoder": encoder_spec,
            "code_sha256": file_hash(__file__)}


class ItemCache:
    def __init__(self, embeddings, missing, token_lengths, truncated, manifest):
        self.embeddings, self.missing = embeddings, missing
        self.token_lengths, self.truncated, self.manifest = token_lengths, truncated, manifest
        _validate_arrays(self)
        for values in (embeddings, missing, token_lengths, truncated):
            values.flags.writeable = False


def _validate_arrays(cache):
    vectors, missing, lengths, truncated = cache.embeddings, cache.missing, cache.token_lengths, cache.truncated
    size = cache.manifest["vocab_size"]
    if (vectors.shape != (size, DIMENSION) or vectors.dtype != np.float32
            or missing.shape != (size,) or missing.dtype != bool
            or lengths.shape != (size,) or lengths.dtype != np.int32
            or truncated.shape != (size,) or truncated.dtype != bool
            or not np.isfinite(vectors).all() or np.any(lengths < 0)
            or not missing[:3].all() or np.any(vectors[missing] != 0)
            or np.any(lengths[missing] != 0) or not np.array_equal(truncated, lengths > 512)
            or np.any(lengths[~missing] < 1)
            or not np.allclose(np.linalg.norm(vectors[~missing], axis=1), 1, rtol=0, atol=NORM_ATOL)):
        raise ValueError("Invalid text cache arrays, missing masks, or norms")


def build_item_cache(items_path, metadata_path, field, output, encoder, *, batch_size=16):
    """Encode each nonempty field independently; encoder injection is for CPU tests."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Positive batch_size required")
    plan = plan_catalog(items_path, metadata_path, field)
    if encoder.spec.get("dimension") != DIMENSION or not encoder.spec.get("revision"):
        raise ValueError("Encoder dimension and pinned revision required")
    descriptor = _descriptor(plan, encoder.spec)
    signature = canonical_hash(descriptor)
    target = Path(output) / signature
    if target.exists():
        return load_item_cache(target, expected_signature=signature, expected_vocabulary_sha256=plan["vocabulary_sha256"],
                               allow_test_encoder=not encoder.spec.get("production_encoder", False))
    size = plan["vocab_size"]
    vectors, missing = np.zeros((size, DIMENSION), np.float32), np.ones(size, bool)
    lengths = np.zeros(size, np.int32)
    present = [index for index, text in enumerate(plan["texts"]) if text]
    for start in range(0, len(present), batch_size):
        indices = present[start:start + batch_size]
        result, token_lengths = encoder.encode(["query: " + plan["texts"][index] for index in indices])
        result, token_lengths = np.asarray(result), np.asarray(token_lengths)
        if (result.shape != (len(indices), DIMENSION) or result.dtype != np.float32
                or token_lengths.shape != (len(indices),) or token_lengths.dtype.kind not in "iu"
                or np.any(token_lengths < 1) or np.any(token_lengths > np.iinfo(np.int32).max)):
            raise ValueError("Invalid encoder output shape/dtype/token lengths")
        tokens = np.asarray(indices) + 3
        vectors[tokens], lengths[tokens], missing[tokens] = result, token_lengths, False
    truncated = lengths > 512
    manifest = {**descriptor, "signature": signature, "state": "completed",
                "coverage": {"universe": size - 3, "present": len(present),
                             "missing": size - 3 - len(present), "truncated": int(truncated.sum())}}
    result = ItemCache(vectors, missing, lengths, truncated, manifest)
    for pin in plan["source_pins"].values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("Catalogue source changed during encoding")
    target.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, array in (("embeddings", vectors), ("missing", missing), ("token_lengths", lengths), ("truncated", truncated)):
        path = target / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        outputs[path.name] = {"sha256": file_hash(path), "shape": list(array.shape), "dtype": str(array.dtype)}
    manifest["outputs"] = outputs
    write_json(target / "manifest.json", manifest)
    return result


def load_item_cache(directory, *, expected_signature=None, expected_vocabulary_sha256=None, allow_test_encoder=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != CONTRACT or manifest.get("state") != "completed":
        raise ValueError("Incomplete/unsupported text cache")
    descriptor = {key: value for key, value in manifest.items() if key not in ("signature", "state", "coverage", "outputs")}
    if canonical_hash(descriptor) != manifest.get("signature"):
        raise ValueError("Text cache signature mismatch")
    if expected_signature is not None and manifest["signature"] != expected_signature:
        raise ValueError("Unexpected text cache signature")
    if expected_vocabulary_sha256 is not None and manifest["vocabulary_sha256"] != expected_vocabulary_sha256:
        raise ValueError("Text vocabulary mapping mismatch")
    if not allow_test_encoder and not manifest["encoder"].get("production_encoder", False):
        raise ValueError("Synthetic fixture encoder cannot be admitted as production cache")
    encoder = manifest["encoder"]
    if encoder.get("production_encoder", False) and any(encoder.get(key) != expected for key, expected in (
            ("model_id", MODEL_ID), ("revision", REVISION), ("tokenizer_revision", REVISION),
            ("files", MODEL_FILES), ("dimension", DIMENSION))):
        raise ValueError("Unsupported production encoder pins")
    if manifest["field"] not in FIELDS or manifest["preprocessing"] != PREPROCESSING:
        raise ValueError("Unsupported text preprocessing/field")
    ordered = manifest["ordered_item_ids"]
    checked = _mapping({"special_tokens": SPECIALS, "vocab_size": manifest["vocab_size"],
                        "item_id_to_token": {item_id: token for token, item_id in enumerate(ordered, 3)}})
    if checked != ordered or manifest["vocabulary_sha256"] != canonical_hash({"ordered_item_ids": ordered, "special_tokens": SPECIALS}):
        raise ValueError("Corrupt vocabulary declaration")
    arrays = {}
    for name in ("embeddings", "missing", "token_lengths", "truncated"):
        path = directory / (name + ".npy")
        pin = manifest["outputs"][path.name]
        if file_hash(path) != pin["sha256"]:
            raise ValueError("Text cache array hash mismatch: " + path.name)
        arrays[name] = np.load(path, allow_pickle=False)
        if list(arrays[name].shape) != pin["shape"] or str(arrays[name].dtype) != pin["dtype"]:
            raise ValueError("Text cache array schema mismatch")
        arrays[name].flags.writeable = False
    return ItemCache(**arrays, manifest=manifest)


def personal_profile(cache, x, lengths):
    """Mean+L2 of exactly x[1:length], retaining UNK slots in denominator n."""
    x, lengths = np.asarray(x), np.asarray(lengths)
    if (x.ndim != 2 or not 1 <= x.shape[1] <= 64 or x.dtype.kind not in "iu"
            or lengths.shape != (len(x),) or lengths.dtype.kind not in "iu"
            or np.any(lengths < 1) or np.any(lengths > x.shape[1])
            or np.any(x < 0) or np.any(x >= len(cache.embeddings)) or np.any(x[:, 0] != SPECIALS["BOS"])):
        raise ValueError("Invalid H63 tokens/lengths/BOS")
    positions = np.arange(x.shape[1])[None, :]
    visible = (positions > 0) & (positions < lengths[:, None])
    if np.any(x[(positions >= lengths[:, None])] != 0) or np.any(x[visible] < SPECIALS["UNK"]):
        raise ValueError("Visible H63 specials or nonzero padding")
    known = visible & ~cache.missing[x]
    n, s = (lengths - 1).astype(np.int32), known.sum(axis=1).astype(np.int32)
    mean = (cache.embeddings[x] * known[..., None]).sum(axis=1, dtype=np.float32)
    mean /= np.maximum(s, 1)[:, None]
    norm = np.linalg.norm(mean, axis=1).astype(np.float32)
    valid = (s > 0) & (norm > EPSILON)
    profile = np.zeros_like(mean)
    np.divide(mean, norm[:, None], out=profile, where=valid[:, None])
    coverage = np.zeros(len(x), np.float32)
    missing_share = np.zeros(len(x), np.float32)
    np.divide(s, n, out=coverage, where=n > 0)
    np.divide(n - s, n, out=missing_share, where=n > 0)
    return {"p": profile, "n": n, "s": s, "coverage": coverage, "missing_share": missing_share,
            "empty_history": n == 0, "profile_missing": s == 0,
            "profile_zero_norm": (s > 0) & ~valid, "profile_valid": valid, "mean_norm": norm}


def build_profile_sidecar(cache, split_npz, output, *, split, batch_size=512, profile_storage_dtype="float16"):
    """Consume prepared states verbatim; role cutoff provenance is explicit.

    The caller owns the preparer's strict-past guarantee. No interactions, target
    outcomes, y, reward, artificial append, or older raw-ID lookup is performed.
    """
    if (split not in PROFILE_DATE_BOUNDS or isinstance(batch_size, bool)
            or not isinstance(batch_size, int) or batch_size < 1
            or profile_storage_dtype not in ("float16", "float32")):
        raise ValueError("Explicit supported split and positive batch size required")
    split_npz, output = Path(split_npz), Path(output)
    if output.exists():
        raise ValueError("Profile output directory must be new")
    before = file_hash(split_npz)
    with np.load(split_npz, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("x", "lengths", "user_id", "target_item_id", "target_date")}
    if data["x"].ndim != 2:
        raise ValueError("Profile x must be a rank-two token array")
    rows = data["x"].shape[0]
    if data["lengths"].shape != (rows,):
        raise ValueError("Profile lengths must be an aligned rank-one array")
    for key in ("user_id", "target_item_id", "target_date"):
        values = data[key]
        if (values.shape != (rows,) or values.dtype.kind not in "iu"
                or np.any(values < 0) or np.any(values > np.iinfo(np.int64).max)):
            raise ValueError("Profile row key requires aligned nonnegative int64-range integers: " + key)
    first, stop = PROFILE_DATE_BOUNDS[split]
    if np.any(data["target_date"] < first) or np.any(data["target_date"] >= stop):
        raise ValueError("Profile target_date outside declared prepared split")
    # Validate even empty inputs and every active/padded history before mkdir.
    # Bounded calls retain the same public H63 validation without allocating Bx63x384.
    templates = personal_profile(cache, data["x"][:0], data["lengths"][:0])
    for begin in range(0, rows, batch_size):
        personal_profile(cache, data["x"][begin:begin + batch_size], data["lengths"][begin:begin + batch_size])
    row_hashes = {key: array_hash(value) for key, value in data.items()}
    output.mkdir(parents=True, exist_ok=False)
    files, mappings = {}, {}
    for name, template in templates.items():
        path = output / (name + ".npy")
        dtype = profile_storage_dtype if name == "p" else template.dtype
        mappings[name] = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(rows, *template.shape[1:]))
    try:
        for begin in range(0, rows, batch_size):
            end = min(rows, begin + batch_size)
            block = personal_profile(cache, data["x"][begin:end], data["lengths"][begin:end])
            for name, values in mappings.items():
                values[begin:end] = block[name]
        for name, values in mappings.items():
            path = output / (name + ".npy")
            values.flush()
            files[path.name] = {"sha256": file_hash(path), "shape": list(values.shape), "dtype": str(values.dtype)}
    finally:
        for values in mappings.values():
            values._mmap.close()
    if file_hash(split_npz) != before:
        raise ValueError("Split changed during profile generation")
    manifest = {"contract": PROFILE_CONTRACT, "state": "completed", "field": cache.manifest["field"],
                "item_cache_signature": cache.manifest["signature"],
                "item_cache_manifest_sha256": canonical_hash(cache.manifest),
                "dimension": DIMENSION, "norm_atol": NORM_ATOL,
                "encoder": cache.manifest["encoder"], "preprocessing": cache.manifest["preprocessing"],
                "source_pins": cache.manifest["source_pins"],
                "vocabulary_sha256": cache.manifest["vocabulary_sha256"], "epsilon": EPSILON,
                "accumulation_dtype": "float32", "profile_storage_dtype": profile_storage_dtype, "split": split,
                "rounding": "profile p rounded once after float32 mean+L2; masks/counts retain exact types",
                "source": {"path": split_npz.resolve().as_posix(), "sha256": before},
                "row_hashes": row_hashes, "row_identity_sha256": canonical_hash(row_hashes),
                "origin": "strict_before_target_day" if split == "train" else {"val": "2021-08-09", "test": "2021-08-16"}[split],
                "row_key_validation": "rank1_integer_nonnegative_int64_bounds; target_date_in_declared_split",
                "history_contract": "prepared_exact_x[1:lengths];cap63_before_text_filter;UNK_counts;duplicates_retained",
                "cutoff_validation": "upstream_preparer_required; no_history_timestamps_in_split",
                "outputs": files, "code_sha256": file_hash(__file__)}
    manifest["signature"] = canonical_hash(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    build = commands.add_parser("encode")
    for command in (plan, build):
        command.add_argument("--items", type=Path, required=True)
        command.add_argument("--metadata", type=Path, required=True)
        command.add_argument("--field", choices=FIELDS, required=True)
    build.add_argument("--local-model", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    build.add_argument("--compute-dtype", choices=("float32", "float16"), default="float32")
    build.add_argument("--batch-size", type=int, default=16)
    profile = commands.add_parser("profile")
    profile.add_argument("--cache", type=Path, required=True)
    profile.add_argument("--split-npz", type=Path, required=True)
    profile.add_argument("--split", choices=("train", "val", "test"), required=True)
    profile.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = plan_catalog(args.items, args.metadata, args.field)
        texts = result.pop("texts")
        result["coverage"] = {"universe": len(texts), "missing": sum(not value for value in texts)}
        result["model_acquisition"] = {"model_id": MODEL_ID, "revision": REVISION,
                                       "files": MODEL_FILES, "total_bytes": sum(p["size"] for p in MODEL_FILES.values())}
    elif args.command == "encode":
        encoder = E5Encoder(args.local_model, device=args.device, compute_dtype=args.compute_dtype)
        result = build_item_cache(args.items, args.metadata, args.field, args.output, encoder, batch_size=args.batch_size).manifest
    else:
        result = build_profile_sidecar(load_item_cache(args.cache), args.split_npz, args.output, split=args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
