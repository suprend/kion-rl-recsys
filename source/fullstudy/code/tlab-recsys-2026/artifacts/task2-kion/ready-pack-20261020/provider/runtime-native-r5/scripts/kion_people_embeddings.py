"""Frozen train-catalogue credit maps and three learned history-only adapters.

No interactions, labels, external identities, downloads or accelerator operations
are used by the builder. Actor top-3 uses fixed train-catalogue card frequency,
not audience popularity. Static credits are retrospective metadata. This module
does not claim that initial random vectors are learned or improve ranking.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_pack_people as people

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # Mapping/CSR preparation also works in the base PAS runtime.
    torch = nn = F = None

CONTRACT = "kion-learned-people-history-v1"
NORMALIZER = "personal-familiarity-key-v1"
VARIANTS = {"director-mean": ("directors", "mean"),
            "actor-mean": ("actors", "mean"),
            "actor-top3-concat": ("actors", "top3-concat")}
PAD, MISSING, RARE = 0, 1, 2
FIRST_ITEM = 3
canonical_hash = people.canonical_hash


def _source_pins():
    return {Path(module).name: people.file_hash(module)
            for module in (__file__, people.__file__)}


def _signature(catalog):
    return canonical_hash({key: value for key, value in catalog.items() if key != "signature"})


def build_people_catalog(items, metadata, *, rare_threshold=2, source_identity=None):
    """Fit only U, preserving normalized person identity before rare mapping.

    ``rare_threshold=2`` is an explicit initial experiment configuration, not a
    selected optimum. Other thresholds require separate run signatures. A stable
    person ID is its index in ascending Unicode normalized-key order; embedding
    rows 0/1/2 are PAD/MISSING/RARE and retained identities start at row 3.
    """
    meta, mapping = people._metadata(metadata)
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
            keys = people.parse_people(value)
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
    catalog = {"contract": CONTRACT, "normalizer": NORMALIZER,
        "unicode_version": unicodedata.unidata_version, "rare_threshold": rare_threshold,
        "vocab_size": size, "item_id_to_token": mapping, "special_tokens": people.SPECIAL_TOKENS,
        "roles": roles, "raw_values": raw_values, "source_identity": copy.deepcopy(source_identity or {}),
        "source_pins": _source_pins(),
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
    if not isinstance(catalog, dict) or catalog.get("contract") != CONTRACT:
        raise ValueError("Unknown people catalogue contract")
    if catalog.get("signature") != _signature(catalog):
        raise ValueError("People catalogue signature mismatch")
    if expected_signature is not None and catalog["signature"] != expected_signature:
        raise ValueError("People catalogue does not match checkpoint/run")
    if expected_sources is not None and catalog["source_identity"] != expected_sources:
        raise ValueError("People catalogue source identity mismatch")
    # Re-derive every fit quantity from the preserved U-only raw credits. This
    # validates masks/rare multiplicities/ranking, not only shape or a hash.
    _, mapping = people._metadata(catalog)
    size = catalog["vocab_size"]
    if set(catalog.get("roles", {})) != {"actors", "directors"}:
        raise ValueError("Expected independent actor and director roles")
    raw = catalog.get("raw_values", {})
    if (set(raw) != {"actors", "directors"} or any(not isinstance(raw[role], list)
            or len(raw[role]) != size or any(not isinstance(value, str) for value in raw[role])
            or any(raw[role][:3]) for role in raw)):
        raise ValueError("Invalid preserved raw credit rows")
    items = {item_id: {role: raw[role][token] for role in raw} for item_id, token in mapping.items()}
    rebuilt = build_people_catalog(items, catalog, rare_threshold=catalog["rare_threshold"],
                                   source_identity=catalog["source_identity"])
    if rebuilt["roles"] != catalog["roles"] or catalog.get("normalizer") != NORMALIZER:
        raise ValueError("Malformed people mapping, frequencies, ranking or masks")
    return catalog


def exact_key_csr(catalog, role):
    """Uncollapsed exact keys for personal crosses; RARE never creates a match."""
    validate_catalog(catalog)
    if role not in ("actors", "directors"):
        raise ValueError("role must be actors or directors")
    rows = catalog["roles"][role]["item_person_ids"]
    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
    return {"offsets": np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths))),
            "entities": np.asarray([index for row in rows for index in row], dtype=np.int64),
            "key_columns": list(catalog["roles"][role]["keys"])}


def build(raw_dir, metadata, *, rare_threshold=2):
    """Read local CSV/metadata only; pin bytes and reject mutation while building."""
    path = Path(raw_dir) / "items.csv"
    before = people.file_hash(path)
    metadata_path = Path(metadata) if isinstance(metadata, (str, Path)) else None
    metadata_sha = people.file_hash(metadata_path) if metadata_path is not None else canonical_hash(metadata)
    # The accepted parser validates IDs/CSV schema and retains the raw credits.
    items = people.read_items(path)
    result = build_people_catalog(items, metadata, rare_threshold=rare_threshold,
        source_identity={"items.csv": {"path": path.resolve().as_posix(), "sha256": before},
            "metadata": {"path": metadata_path.resolve().as_posix() if metadata_path else None,
                         "sha256": metadata_sha}})
    if people.file_hash(path) != before or (metadata_path is not None and people.file_hash(metadata_path) != metadata_sha):
        raise ValueError("People input changed during build")
    return result


def write_catalog(path, catalog):
    validate_catalog(catalog)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False, sort_keys=True, allow_nan=False,
                              separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")


def load_catalog(path, *, expected_signature=None, expected_sources=None):
    catalog = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_catalog(catalog, expected_signature=expected_signature, expected_sources=expected_sources)


class PeopleEmbeddingAdapter(nn.Module if nn is not None else object):
    """Add ``adapter(x)`` to history token embeddings before one GPT forward.

    Parameters are ordinary trainable nn.Embedding/Linear weights. The caller
    includes them in the same CE optimizer as the GPT; this class has no detached
    feature cache and never modifies candidate output embeddings. Mean pooling
    uses fused EmbeddingBag for all catalogue cards per forward, then indexes x.
    This bounds intermediates by catalogue credits, not B*T*maximum_cast*d.
    """
    def __init__(self, catalog, variant, *, hidden_size=256, person_dim=32, seed=42, enabled=True):
        if torch is None:
            raise RuntimeError("PeopleEmbeddingAdapter requires an existing PyTorch runtime")
        super().__init__()
        validate_catalog(catalog)
        if catalog["source_pins"] != _source_pins():
            raise ValueError("People builder/parser source pins changed; rebuild sidecar")
        if variant not in VARIANTS:
            raise ValueError("Expected one of the three separately declared people experiments")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
               for value in (hidden_size, person_dim)):
            raise ValueError("Embedding dimensions must be positive integers")
        if not isinstance(enabled, bool) or isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("enabled must be boolean and seed integer")
        role, pooling = VARIANTS[variant]
        self.variant, self.role, self.pooling = variant, role, pooling
        self.hidden_size, self.person_dim, self.enabled = hidden_size, person_dim, enabled
        self.catalog_signature, self.source_identity = catalog["signature"], copy.deepcopy(catalog["source_identity"])
        self.source_pins = copy.deepcopy(catalog["source_pins"])
        self.seed, self.vocab_size = seed, catalog["vocab_size"]
        data = catalog["roles"][role]
        rows = data["item_embedding_ids"]
        self.register_buffer("flat_embedding_ids", torch.tensor([index for row in rows for index in row], dtype=torch.long))
        self.register_buffer("item_offsets", torch.tensor([0] + list(np.cumsum([len(row) for row in rows])), dtype=torch.long))
        self.register_buffer("top3_ids", torch.tensor(data["top3_embedding_ids"], dtype=torch.long))
        self.register_buffer("top3_occupancy", torch.tensor(data["top3_occupancy"], dtype=torch.bool))
        self.register_buffer("missing_mask", torch.tensor(data["item_missing"], dtype=torch.bool))
        self.register_buffer("rare_count", torch.tensor(data["item_rare_count"], dtype=torch.long))
        # Constructors/init must not shift baseline weights, data order or dropout.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.table = nn.Embedding(len(data["embedding_keys"]), person_dim, padding_idx=PAD)
            self.projection = nn.Linear(person_dim * (3 if pooling == "top3-concat" else 1), hidden_size, bias=False)
            nn.init.normal_(self.table.weight, mean=0, std=.02, generator=generator)
            nn.init.normal_(self.projection.weight, mean=0, std=.02, generator=generator)
            with torch.no_grad():
                self.table.weight[PAD].zero_()
        self.rare_threshold = catalog["rare_threshold"]
        self.retained_people = len(data["embedding_keys"]) - 3

    def contract(self):
        return {"contract": CONTRACT, "variant": self.variant, "role": self.role, "pooling": self.pooling,
            "catalog_signature": self.catalog_signature, "source_identity": copy.deepcopy(self.source_identity),
            "source_pins": copy.deepcopy(self.source_pins), "hidden_size": self.hidden_size, "person_dim": self.person_dim,
            "rare_threshold": self.rare_threshold, "seed": self.seed, "enabled": self.enabled,
            "integration": "additive history-only; tied ID-only candidate output unchanged",
            "capacity_matched_between_aggregators": False,
            "projection_input_dim": self.projection.in_features,
            "trainable_parameters": sum(p.numel() for p in self.parameters()),
            "table_parameters": self.table.weight.numel(), "projection_parameters": self.projection.weight.numel(),
            "pad_policy": "fixed zero/no gradient; slots masked before concatenation",
            "learned_status": "trainable initialization; learned only after evidenced CE optimizer updates"}

    def get_extra_state(self):
        return self.contract()

    def set_extra_state(self, state):
        if state != self.contract():
            raise RuntimeError("People checkpoint catalog/source/variant/architecture mismatch")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if state_dict.get(prefix + "_extra_state") != self.contract():
            raise RuntimeError("People checkpoint requires identical source/catalog/variant contract")
        for name in ("flat_embedding_ids", "item_offsets", "top3_ids", "top3_occupancy", "missing_mask", "rare_count"):
            incoming, expected = state_dict.get(prefix + name), getattr(self, name)
            if (incoming is None or incoming.dtype != expected.dtype or incoming.shape != expected.shape
                    or not torch.equal(incoming.to(expected.device), expected)):
                raise RuntimeError("People checkpoint frozen mapping/mask mismatch: " + name)
        weight = state_dict.get(prefix + "table.weight")
        if weight is not None and (not torch.isfinite(weight).all() or torch.any(weight[PAD] != 0)):
            raise RuntimeError("Invalid people checkpoint table/PAD row")
        projection = state_dict.get(prefix + "projection.weight")
        if projection is not None and not torch.isfinite(projection).all():
            raise RuntimeError("Nonfinite people checkpoint projection")
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, item_tokens):
        if item_tokens.dtype != torch.long or item_tokens.device != self.table.weight.device:
            raise ValueError("People token IDs must be int64 on the adapter device")
        if torch.any(item_tokens < 0) or torch.any(item_tokens >= self.vocab_size):
            raise ValueError("People item token outside frozen vocabulary")
        if not self.enabled:
            return self.table.weight.new_zeros((*item_tokens.shape, self.hidden_size))
        if self.pooling == "mean":
            pooled = F.embedding_bag(self.flat_embedding_ids, self.table.weight, self.item_offsets,
                mode="mean", include_last_offset=True, padding_idx=PAD)
            values = pooled[item_tokens]
        else:
            # Repeated RARE slots remain separate; zero PAD cannot dilute them.
            values = (self.table(self.top3_ids[item_tokens]) * self.top3_occupancy[item_tokens].unsqueeze(-1)).flatten(-2)
        return self.projection(values) * item_tokens.ge(FIRST_ITEM).unsqueeze(-1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rare-threshold", type=int, default=2)
    args = parser.parse_args(argv)
    catalog = build(args.raw, args.metadata, rare_threshold=args.rare_threshold)
    write_catalog(args.output, catalog)
    print(json.dumps({"signature": catalog["signature"], "vocab_size": catalog["vocab_size"],
        "roles": {role: {"unique_people": len(data["keys"]),
            "retained_people": len(data["embedding_keys"]) - 3,
            "missing_cards": sum(data["item_missing"]),
            "rare_credits": sum(data["item_rare_count"])} for role, data in catalog["roles"].items()}}))


if __name__ == "__main__":
    main()
