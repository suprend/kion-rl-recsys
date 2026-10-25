"""Complete-package GPT: exact catalog retrieval and bounded pair-aware scoring.

This is a new architecture, separate from the frozen history-only experiments.
Only immutable feature matrices are cached. The trainable candidate tower is
recomputed once per microbatch and shared with the history input embeddings.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kion_train import NextItemGPT as IDGPT, BOS, PAD, FIRST_ITEM, last_real_hidden
from kion_conditioned_train import canonical_hash, array_hash

CONTRACT = "kion-complete-gpt-v1"
TEXT_FIELDS = ("title", "title_orig", "description")
TEXT_VARIANTS = ("all", "none", *TEXT_FIELDS)


def _matrix(values, rows, name):
    values = np.asarray(values)
    if (values.ndim != 2 or len(values) != rows or values.dtype.kind != "f"
            or not np.isfinite(values).all() or np.any(values[:FIRST_ITEM] != 0)):
        raise ValueError(f"{name} requires finite floating V x D with zero special rows")
    return np.asarray(values, np.float32)


def _columns(columns, widths):
    if not isinstance(columns, dict) or set(columns) != set(widths):
        raise ValueError("Named item, user and candidate feature columns are required")
    result = copy.deepcopy(columns)
    for name, width in widths.items():
        names = result[name]
        if (not isinstance(names, list) or len(names) != width or len(set(names)) != width
                or any(not isinstance(value, str) or not value for value in names)):
            raise ValueError(f"Invalid named {name} feature schema")
    return result


def _tower(input_dim, rank, hidden_size):
    return nn.Sequential(nn.Linear(input_dim, rank, bias=False),
                         nn.Linear(rank, hidden_size, bias=False))


class CompleteGPT(IDGPT):
    """All static candidate branches participate in exact retrieval CE.

    Pair features are row/candidate-specific and only evaluated for bounded
    candidate sets. They never expand into a B x V x feature_dim tensor.
    """
    def __init__(self, vocab_size, context_length, hidden_size=256, layers=4, heads=4, *,
                 item_features, user_feature_dim, candidate_feature_dim, feature_columns,
                 text_features=None, text_variant="all", people_catalog=None,
                 people_variants=("actor-mean", "director-mean", "actor-top3-concat"), bottleneck=32,
                 person_dim=32, seed=2026, schema_hash=None, source_contract=None):
        widths = (user_feature_dim, candidate_feature_dim, bottleneck, person_dim)
        if (any(isinstance(n, bool) or not isinstance(n, int) for n in widths)
                or min(user_feature_dim, candidate_feature_dim) < 0
                or min(bottleneck, person_dim) < 1 or vocab_size <= FIRST_ITEM):
            raise ValueError("Invalid complete model dimensions")
        items = _matrix(item_features, vocab_size, "item_features")
        columns = _columns(feature_columns, {"item": items.shape[1], "user": user_feature_dim,
                                             "candidate": candidate_feature_dim})
        self.schema_hash = canonical_hash(columns)
        if schema_hash is not None and schema_hash != self.schema_hash:
            raise ValueError("Complete model schema hash mismatch")
        if text_variant not in TEXT_VARIANTS:
            raise ValueError("Unknown complete text variant")
        text_features = {} if text_features is None else text_features
        if set(text_features) - set(TEXT_FIELDS):
            raise ValueError("Unknown text field")
        selected = TEXT_FIELDS if text_variant == "all" else (() if text_variant == "none" else (text_variant,))
        if not set(selected).issubset(text_features):
            raise ValueError("Selected text fields are missing; use text_variant='none' explicitly for ablation")
        texts = {field: _matrix(text_features[field], vocab_size, field) for field in selected}
        if len(set(people_variants)) != len(people_variants):
            raise ValueError("Duplicate people branches")
        if people_variants and people_catalog is None:
            raise ValueError("Selected people branches require a frozen catalog")
        super().__init__(vocab_size, context_length, hidden_size, layers, heads, tied=True)
        self.vocab_size, self.context_length = vocab_size, context_length
        self.user_feature_dim, self.candidate_feature_dim = user_feature_dim, candidate_feature_dim
        self.feature_columns = columns
        self.register_buffer("item_features", torch.tensor(items))
        self.item_tower = _tower(items.shape[1], bottleneck, hidden_size) if items.shape[1] else None
        self.user_projection = nn.Linear(user_feature_dim, hidden_size, bias=False) if user_feature_dim else None
        self.text_towers = nn.ModuleDict()
        for field, values in texts.items():
            self.register_buffer("text_" + field, torch.tensor(values))
            self.text_towers[field] = _tower(values.shape[1], bottleneck, hidden_size)
        self.people = nn.ModuleDict()
        if people_variants:
            from kion_people_embeddings import PeopleEmbeddingAdapter
            if people_catalog["vocab_size"] != vocab_size:
                raise ValueError("People vocabulary differs from the complete package")
            for variant in people_variants:
                self.people[variant] = PeopleEmbeddingAdapter(people_catalog, variant, hidden_size=hidden_size,
                                                               person_dim=person_dim, seed=seed)
        self.pair_projection = nn.Linear(candidate_feature_dim, bottleneck, bias=False) if candidate_feature_dim else None
        self.pair_head = nn.Sequential(nn.Linear(3 * hidden_size + (bottleneck if candidate_feature_dim else 0), hidden_size),
                                       nn.GELU(), nn.Linear(hidden_size, 1))
        self._contract = {"contract": CONTRACT, "vocab_size": vocab_size, "context_length": context_length,
            "hidden_size": hidden_size, "layers": layers, "heads": heads, "bottleneck": bottleneck,
            "person_dim": person_dim, "seed": seed, "schema_hash": self.schema_hash,
            "feature_columns": columns, "item_features_sha256": array_hash(items),
            "text_variant": text_variant, "text_features_sha256": {key: array_hash(value) for key, value in texts.items()},
            "people_variants": list(people_variants),
            "people_catalog_signature": people_catalog["signature"] if people_catalog is not None else None,
            "integration": "complete history+candidate shared tower; bounded pair head; distinct from isolated experiments",
            "source_contract": copy.deepcopy(source_contract or {})}

    def contract(self):
        return copy.deepcopy(self._contract)

    @classmethod
    def from_config(cls, config, *, item_features, text_features=None, people_catalog=None):
        """Recreate a checkpoint architecture with independently admitted sidecars."""
        wrapper = config["contract"] if isinstance(config.get("contract"), dict) else config
        architecture = wrapper.get("architecture", wrapper)
        columns = architecture["feature_columns"]
        model = cls(**{key: architecture[key] for key in
            ("vocab_size", "context_length", "hidden_size", "layers", "heads", "bottleneck", "person_dim", "seed")},
            item_features=item_features, user_feature_dim=len(columns["user"]),
            candidate_feature_dim=len(columns["candidate"]), feature_columns=columns,
            text_features=text_features, text_variant=architecture["text_variant"], people_catalog=people_catalog,
            people_variants=tuple(architecture["people_variants"]), schema_hash=architecture["schema_hash"],
            source_contract=architecture["source_contract"])
        if model.contract() != architecture:
            raise ValueError("Complete reconstruction sidecars do not reproduce checkpoint architecture")
        return model

    def get_extra_state(self):
        return self.contract()

    def set_extra_state(self, state):
        if state != self.contract():
            raise RuntimeError("Complete checkpoint source/schema/architecture mismatch")

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if state_dict.get(prefix + "_extra_state") != self.contract():
            raise RuntimeError("Complete checkpoint requires identical frozen features, source and schema")
        # Do not silently replace sidecar bytes even if an external receipt was forged.
        for name in ("item_features", *("text_" + field for field in self.text_towers)):
            incoming = state_dict.get(prefix + name)
            if incoming is None or not torch.equal(incoming.cpu(), getattr(self, name).cpu()):
                raise RuntimeError("Complete checkpoint immutable feature buffer mismatch: " + name)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def candidate_table(self):
        table = self.backbone.wte.weight
        residual = torch.zeros_like(table)
        if self.item_tower is not None:
            residual = residual + self.item_tower(self.item_features)
        for field, tower in self.text_towers.items():
            residual = residual + tower(getattr(self, "text_" + field))
        if self.people:
            tokens = torch.arange(self.vocab_size, device=table.device)
            for adapter in self.people.values():
                residual = residual + adapter(tokens)
        mask = torch.arange(self.vocab_size, device=table.device).ge(FIRST_ITEM).unsqueeze(-1)
        return table + residual * mask

    def encode_context(self, x, lengths, user_features=None, *, candidate_table=None):
        if (x.dtype != torch.long or lengths.dtype != torch.long or lengths.shape != (len(x),)
                or x.ndim != 2 or x.shape[1] > self.context_length or torch.any(lengths < 1)
                or torch.any(lengths > x.shape[1])):
            raise ValueError("Invalid complete GPT context")
        if self.user_feature_dim:
            if (user_features is None or user_features.shape != (len(x), self.user_feature_dim)
                    or not user_features.is_floating_point() or not torch.isfinite(user_features).all()):
                raise ValueError("Aligned finite user feature rows are required")
        elif user_features is not None and user_features.shape != (len(x), 0):
            raise ValueError("Unexpected user feature width")
        table = self.candidate_table() if candidate_table is None else candidate_table
        embeddings = table[x]
        if self.user_projection is not None:
            embeddings = embeddings + x.eq(BOS).unsqueeze(-1) * self.user_projection(user_features.float()).unsqueeze(1)
        hidden = self.backbone(inputs_embeds=embeddings, attention_mask=x.ne(PAD), use_cache=False).last_hidden_state
        return last_real_hidden(hidden, lengths)

    def retrieval_logits(self, encoded, candidate_table=None):
        table = self.candidate_table() if candidate_table is None else candidate_table
        return F.linear(encoded, table)

    def score_candidates(self, encoded, candidate_ids, pair_features=None, *, candidate_table=None,
                         retrieval_logits=None):
        if (candidate_ids.dtype != torch.long or candidate_ids.ndim != 2
                or candidate_ids.shape[0] != len(encoded) or torch.any(candidate_ids < FIRST_ITEM)
                or torch.any(candidate_ids >= self.vocab_size)):
            raise ValueError("Candidates require known int64 B x C token IDs")
        shape = (*candidate_ids.shape, self.candidate_feature_dim)
        if self.candidate_feature_dim:
            if (pair_features is None or pair_features.shape != shape or not pair_features.is_floating_point()
                    or not torch.isfinite(pair_features).all()):
                raise ValueError("Aligned finite B x C x candidate_dim features are required")
        elif pair_features is not None and pair_features.shape != shape:
            raise ValueError("Unexpected pair feature width")
        table = self.candidate_table() if candidate_table is None else candidate_table
        candidates = table[candidate_ids]
        contexts = encoded.unsqueeze(1).expand_as(candidates)
        parts = [contexts, candidates, contexts * candidates]
        if self.pair_projection is not None:
            parts.append(self.pair_projection(pair_features.float()))
        residual = self.pair_head(torch.cat(parts, dim=-1)).squeeze(-1)
        base = ((contexts * candidates).sum(-1) if retrieval_logits is None else
                retrieval_logits.gather(1, candidate_ids))
        return base + residual

    def forward(self, x, lengths, user_features=None):
        table = self.candidate_table()
        return self.retrieval_logits(self.encode_context(x, lengths, user_features, candidate_table=table), table)


def sample_training_candidates(logits, targets, *, hard_negatives=31, uniform_negatives=32):
    """Unique candidates: forced positive, detached hard negatives, uniform remainder.

    Uniform negatives are sampled without replacement from known tokens excluding
    the positive and hard set. This trains an auxiliary sampled ranking loss; it
    is deliberately not an importance-corrected estimator of full-catalog CE.
    """
    if min(hard_negatives, uniform_negatives) < 0 or hard_negatives + uniform_negatives < 1:
        raise ValueError("At least one nonnegative configured negative count is required")
    if targets.shape != (len(logits),) or torch.any(targets < FIRST_ITEM) or torch.any(targets >= logits.shape[1]):
        raise ValueError("Auxiliary sampling requires known targets")
    known = logits[:, FIRST_ITEM:].detach().clone()
    local_targets = targets - FIRST_ITEM
    known.scatter_(1, local_targets[:, None], -torch.inf)
    hard_count = min(hard_negatives, known.shape[1] - 1)
    hard = known.topk(hard_count, dim=1).indices if hard_count else targets.new_empty((len(targets), 0))
    noise = torch.rand(known.shape, device=logits.device)
    noise.scatter_(1, local_targets[:, None], -1.)
    noise.scatter_(1, hard, -1.)
    uniform_count = min(uniform_negatives, known.shape[1] - 1 - hard_count)
    uniform = noise.topk(uniform_count, dim=1).indices if uniform_count else targets.new_empty((len(targets), 0))
    return torch.cat((targets[:, None], hard + FIRST_ITEM, uniform + FIRST_ITEM), dim=1)
