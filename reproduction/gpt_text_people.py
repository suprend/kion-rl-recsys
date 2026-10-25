"""Portable text and people readers. Import is stdlib-only; no encoder/model.
Explicit calls execute reads, never planning. Historical source identities remain
separate from this adapter contract; np is an explicitly supplied backend.
"""
from collections import Counter
import copy
import re
import unicodedata
from gpt_provider import canonical_hash, logical_child, _close_array
ADAPTER_CONTRACT = "gpt-portable-text-people-v1"

CONTRACT = "kion-text-embeddings-v1"

FIELDS = ("title", "title_orig", "description")

SPECIALS = {"PAD": 0, "BOS": 1, "UNK": 2}

MODEL_ID = "intfloat/multilingual-e5-small"

REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"

DIMENSION = 384

NORM_ATOL = 1e-4

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

class ItemCache:
    def __init__(self, embeddings, missing, token_lengths, truncated, manifest, *, np):
        self.embeddings, self.missing = embeddings, missing
        self.token_lengths, self.truncated, self.manifest = token_lengths, truncated, manifest
        _validate_arrays(self, np=np)
        for values in (embeddings, missing, token_lengths, truncated):
            values.flags.writeable = False

def _validate_arrays(cache, *, np):
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

def load_item_cache(io, directory, *, source_logical, np, expected_manifest_sha256=None, expected_signature=None, expected_vocabulary_sha256=None, allow_test_encoder=False):
    manifest = io.read_json(logical_child(directory, "manifest.json"), expected=expected_manifest_sha256)
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
    io.verify(source_logical, "source", manifest["code_sha256"])
    if set(manifest["source_pins"]) != {"items", "metadata"}:
        raise ValueError("Exact text source census required")
    for name, pin in manifest["source_pins"].items():
        io.pin(directory, pin, "metadata" if name == "metadata" else "payload")
    arrays = {}
    try:
        for name in ("embeddings", "missing", "token_lengths", "truncated"):
            filename = name + ".npy"
            pin = manifest["outputs"][filename]
            path = io.verify(logical_child(directory, filename), "payload", pin["sha256"])
            io.check()
            arrays[name] = np.load(path, allow_pickle=False)
            if list(arrays[name].shape) != pin["shape"] or str(arrays[name].dtype) != pin["dtype"]:
                raise ValueError("Text cache array schema mismatch")
            arrays[name].flags.writeable = False
        io.check()
        return ItemCache(**arrays, manifest=manifest, np=np)
    except BaseException:
        for array in arrays.values():
            _close_array(array)
        raise


SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}

def parse_people(value):
    """Comma-separated NFKC/casefold keys; no aliases, transliteration or ё/е merge."""
    return tuple(sorted({" ".join(unicodedata.normalize("NFKC", part).casefold().split())
                         for part in str(value or "").split(",") if part.strip()}))

def _metadata(value):
    meta = dict(value)
    mapping, size = meta.get("item_id_to_token"), meta.get("vocab_size")
    if meta.get("special_tokens", SPECIAL_TOKENS) != SPECIAL_TOKENS:
        raise ValueError("Unexpected special tokens")
    if (not isinstance(mapping, dict) or isinstance(size, bool) or not isinstance(size, int) or size < 3
            or any(not isinstance(k, str) or not k or isinstance(v, bool) or not isinstance(v, int)
                   for k, v in mapping.items())
            or sorted(mapping.values()) != list(range(3, size))):
        raise ValueError("Expected unique contiguous train-known item tokens after PAD/BOS/UNK")
    return meta, mapping

PEOPLE_CONTRACT = "kion-learned-people-history-v1"
PAD, MISSING, RARE = 0, 1, 2

NORMALIZER = "personal-familiarity-key-v1"

FIRST_ITEM = 3

def _signature(catalog):
    return canonical_hash({key: value for key, value in catalog.items() if key != "signature"})

def _rebuild_people_catalog(items, metadata, *, rare_threshold=2, source_identity=None):
    """Fit only U, preserving normalized person identity before rare mapping.

    ``rare_threshold=2`` is an explicit initial experiment configuration, not a
    selected optimum. Other thresholds require separate run signatures. A stable
    person ID is its index in ascending Unicode normalized-key order; embedding
    rows 0/1/2 are PAD/MISSING/RARE and retained identities start at row 3.
    """
    meta, mapping = _metadata(metadata)
    if isinstance(rare_threshold, bool) or not isinstance(rare_threshold, int) or rare_threshold < 1:
        raise ValueError("rare_threshold must be a positive integer")
    items = {str(key): value for key, value in items.items()}
    size = meta["vocab_size"]
    roles = {}
    raw_values = {role: [""] * size for role in ("actors", "directors")}
    for role in raw_values:
        per_item = [[] for _ in range(size)]
        counts = Counter()
        for item_id, token in mapping.items():
            value = items.get(item_id, {}).get(role, "")
            raw_values[role][token] = "" if value is None else str(value)
            keys = parse_people(value)
            per_item[token] = list(keys)
            counts.update(keys)
        keys = sorted(counts)
        lookup = {key: index for index, key in enumerate(keys)}
        retained = [key for key in keys if counts[key] >= rare_threshold]
        embedding_lookup = {key: index + FIRST_ITEM for index, key in enumerate(retained)}
        person_to_embedding = [embedding_lookup.get(key, RARE) for key in keys]
        person_ids = [[lookup[key] for key in row] for row in per_item]
        embedding_ids = [[person_to_embedding[index] for index in row] if row else
                         ([MISSING] if token >= FIRST_ITEM else [])
                         for token, row in enumerate(person_ids)]
        top3_person_ids, top3_embedding_ids, occupancy = [], [], []
        for token, row in enumerate(person_ids):
            selected = sorted(row, key=lambda index: (-counts[keys[index]], index))[:3]
            top3_person_ids.append(selected)
            ids = [person_to_embedding[index] for index in selected]
            if not ids and token >= FIRST_ITEM:
                ids = [MISSING]
            occupancy.append([int(index < len(ids)) for index in range(3)])
            top3_embedding_ids.append(ids + [PAD] * (3 - len(ids)))
        roles[role] = {"keys": keys, "card_counts": [counts[key] for key in keys],
            "person_to_embedding": person_to_embedding, "embedding_keys": ["<PAD>", "<MISSING>", "<RARE>"] + retained,
            "item_person_ids": person_ids, "item_embedding_ids": embedding_ids,
            "top3_person_ids": top3_person_ids, "top3_embedding_ids": top3_embedding_ids,
            "top3_occupancy": occupancy, "item_missing": [int(token >= 3 and not row) for token, row in enumerate(person_ids)],
            "item_rare_count": [sum(person_to_embedding[index] == RARE for index in row) for row in person_ids],
            "mapping_sha256": canonical_hash({"keys": keys, "person_to_embedding": person_to_embedding}),
            "ranking_sha256": canonical_hash(top3_person_ids),
            "frequency_unit": "distinct train-known catalogue card, current card included",
            "ranking": "descending train-catalogue card count; ascending stable numeric person ID"}
    catalog = {"contract": PEOPLE_CONTRACT, "normalizer": NORMALIZER,
        "unicode_version": unicodedata.unidata_version, "rare_threshold": rare_threshold,
        "vocab_size": size, "item_id_to_token": mapping, "special_tokens": SPECIAL_TOKENS,
        "roles": roles, "raw_values": raw_values, "source_identity": copy.deepcopy(source_identity or {}),
        "source_pins": {},
        "provenance": {"fit_scope": "metadata.item_id_to_token only; PAD/BOS/UNK excluded",
            "metadata_availability": "retrospective static snapshot; historical known_at unverified",
            "top3_score": "fixed train-catalogue card frequency; not behavioural popularity or billing",
            "unknown_person": "RARE after frozen mapping; distinct source identities retained",
            "aliases": "none; no transliteration, homonym, e/yo or work merging",
            "empty_cast": "learned MISSING; top3 [MISSING,PAD,PAD] with [1,0,0] mask",
            "special_items": "zero branch even for UNK", "interactions_read": False,
            "evaluation_fit": False, "outcomes_read": False}}
    catalog["signature"] = _signature(catalog)
    return catalog

def validate_catalog(catalog, *, expected_signature=None, expected_sources=None):
    """Reject malformed/tampered sidecars before constructing trainable modules."""
    if not isinstance(catalog, dict) or catalog.get("contract") != PEOPLE_CONTRACT:
        raise ValueError("Unknown people catalogue contract")
    if catalog.get("signature") != _signature(catalog):
        raise ValueError("People catalogue signature mismatch")
    if expected_signature is not None and catalog["signature"] != expected_signature:
        raise ValueError("People catalogue does not match checkpoint/run")
    if expected_sources is not None and catalog["source_identity"] != expected_sources:
        raise ValueError("People catalogue source identity mismatch")
    # Re-derive every fit quantity from the preserved U-only raw credits. This
    # validates masks/rare multiplicities/ranking, not only shape or a hash.
    _, mapping = _metadata(catalog)
    size = catalog["vocab_size"]
    if set(catalog.get("roles", {})) != {"actors", "directors"}:
        raise ValueError("Expected independent actor and director roles")
    raw = catalog.get("raw_values", {})
    if (set(raw) != {"actors", "directors"} or any(not isinstance(raw[role], list)
            or len(raw[role]) != size or any(not isinstance(value, str) for value in raw[role])
            or any(raw[role][:3]) for role in raw)):
        raise ValueError("Invalid preserved raw credit rows")
    items = {item_id: {role: raw[role][token] for role in raw} for item_id, token in mapping.items()}
    rebuilt = _rebuild_people_catalog(items, catalog, rare_threshold=catalog["rare_threshold"],
                                   source_identity=catalog["source_identity"])
    if rebuilt["roles"] != catalog["roles"] or catalog.get("normalizer") != NORMALIZER:
        raise ValueError("Malformed people mapping, frequencies, ranking or masks")
    return catalog

def load_catalog(io, logical, *, source_bindings, expected_manifest_sha256=None,
                 expected_signature=None, expected_sources=None):
    """Read and rederive preserved credit mappings; no trainable module build.

    source_bindings maps retained source basenames to explicit logical identities.
    Provider vocabulary equality remains the composer's responsibility.
    """
    catalog = io.read_json(logical, expected=expected_manifest_sha256, signed=True)
    names = {"kion_people_embeddings.py", "kion_pack_people.py"}
    if set(source_bindings) != names or set(catalog["source_pins"]) != names:
        raise ValueError("Exact people source census required")
    for name in sorted(names):
        io.verify(source_bindings[name], "source", catalog["source_pins"][name])
    if set(catalog["source_identity"]) != {"items.csv", "metadata"}:
        raise ValueError("Exact people input census required")
    root = logical.rsplit("/", 1)[0]
    for name, pin in catalog["source_identity"].items():
        io.pin(root, pin, "metadata" if name == "metadata" else "payload")
    io.check()
    return validate_catalog(catalog, expected_signature=expected_signature,
                            expected_sources=expected_sources)
