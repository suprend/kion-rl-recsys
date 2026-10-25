"""Explicit H63 user/candidate compatibility, independently of the GPT head.

Only local arrays/CSV are consumed. Metadata and pretrained text are retrospective;
the strict last-pair clock does not establish historical metadata availability.
The caller owns fusion/training. No target, outcome or candidate append is used.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_pack_history as history


CONTRACT = "kion-personal-candidates-v1"
KEY_VERSION = "personal-familiarity-key-v1"
ROLES = ("actor", "director", "studio", "franchise")
EPSILON = 1e-12
TIE_TOLERANCE = 1e-12
FALLBACK = {"accepted": 0, "no_metadata": 1, "short_history": 2,
            "few_distinct": 3, "empty_cluster": 4, "not_converged": 5,
            "small_cluster": 6, "low_coherence": 7, "not_separated": 8,
            "small_gain": 9, "k2_rejected": 10}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def exact_keys(value):
    """One vote per card/key; normalization never resolves aliases/identities."""
    return tuple(sorted({key for part in str(value or "").split(",")
        if (key := " ".join(unicodedata.normalize("NFKC", part).casefold().split()))}))


def _unit(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > EPSILON)


def _lex_order(values):
    return np.lexsort(tuple(values[:, i] for i in range(values.shape[1] - 1, -1, -1)))


def _first_tie_min(values):
    # Rows are already lexicographically ordered, including duplicates.
    return int(np.flatnonzero(values <= values.min() + TIE_TOLERANCE)[0])


def _fit(z, k):
    center = _unit(z.sum(axis=0))
    seeds = [_first_tie_min(z @ center)]
    while len(seeds) < k:
        seeds.append(_first_tie_min(np.max(z @ z[seeds].T, axis=1)))
    centers = z[seeds].copy()
    previous = None
    for _ in range(10):
        similarity = z @ centers.T
        best = similarity.max(axis=1, keepdims=True)
        membership = np.argmax(similarity >= best - TIE_TOLERANCE, axis=1)
        counts = np.bincount(membership, minlength=k)
        if np.any(counts == 0):
            return None, "empty_cluster"
        centers = _unit(np.stack([z[membership == slot].sum(axis=0) for slot in range(k)]))
        if np.any(np.linalg.norm(centers, axis=1) <= EPSILON):
            return None, "empty_cluster"
        if previous is not None and np.array_equal(membership, previous):
            order = _lex_order(centers)
            inverse = np.empty(k, np.int64)
            inverse[order] = np.arange(k)
            return (centers[order], inverse[membership]), "accepted"
        previous = membership
    return None, "not_converged"


def partition_interests(incidence, max_interests=2):
    """Deterministic float64 spherical K1/adaptive K2/K3 reference.

    K3 uses deterministic farthest-first seeds and requires accepted K2, n>=12,
    min3, coherence>=.6, all pair cosines<=.8 and incremental J3-J2>=.10.
    This declared extension deliberately preserves the proposed intermediate gate.
    """
    if max_interests not in (1, 2, 3) or isinstance(max_interests, bool):
        raise ValueError("max_interests must be 1, 2 or 3")
    a = np.asarray(incidence, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all() or not np.isin(a, (0, 1)).all():
        raise ValueError("Genre incidence must be a finite binary matrix")
    a = a[np.any(a != 0, axis=1)]
    a = a[_lex_order(_unit(a))] if len(a) else a
    z = _unit(a)
    n, width = a.shape
    result = {"centers": np.zeros((3, width), np.float64),
              "incidence": np.zeros((3, width), np.float64),
              "support": np.zeros(3, np.int32), "mass": np.zeros(3, np.float64),
              "coherence": np.zeros(3, np.float64), "valid": np.zeros(3, bool),
              "k_eff": 0, "n_known": n, "reason": FALLBACK["no_metadata"],
              "objective": 0., "gain": 0., "max_pair_cosine": 0.}
    if not n:
        return result
    centers, membership = _unit(z.sum(axis=0))[None, :], np.zeros(n, np.int64)
    objective = float(np.mean(z @ centers[0]))
    reason, gain = "accepted", 0.
    for k in range(2, max_interests + 1):
        if n < (8 if k == 2 else 12):
            reason = "short_history"
            break
        if len(np.unique(z, axis=0)) < k:
            reason = "few_distinct"
            break
        fitted, reason = _fit(z, k)
        if fitted is None:
            break
        trial, assignment = fitted
        counts = np.bincount(assignment, minlength=k)
        coherence = np.array([np.mean(z[assignment == slot] @ trial[slot]) for slot in range(k)])
        separated = (trial @ trial.T)[np.triu_indices(k, 1)]
        trial_objective = float(np.mean(np.sum(z * trial[assignment], axis=1)))
        reason = ("small_cluster" if np.any(counts < 3) else
                  "low_coherence" if np.any(coherence < .6 - TIE_TOLERANCE) else
                  "not_separated" if np.any(separated > .8 + TIE_TOLERANCE) else
                  "small_gain" if trial_objective - objective < .10 - TIE_TOLERANCE else "accepted")
        if reason != "accepted":
            break
        centers, membership = trial, assignment
        gain, objective = trial_objective - objective, trial_objective
    k = len(centers)
    counts = np.bincount(membership, minlength=k)
    result["centers"][:k] = centers
    result["incidence"][:k] = np.stack([a[membership == slot].mean(axis=0) for slot in range(k)])
    result["support"][:k] = counts
    result["mass"][:k] = counts / n
    result["coherence"][:k] = [np.mean(z[membership == slot] @ centers[slot]) for slot in range(k)]
    result["valid"][:k] = True
    result.update(k_eff=k, reason=FALLBACK[reason], objective=objective, gain=gain,
                  max_pair_cosine=float(np.max((centers @ centers.T)[np.triu_indices(k, 1)])) if k > 1 else 0.)
    return result


def multi_interest_scores(centers, valid, mass, candidates, *, method="max", temperature=.2):
    """Symmetric [B,K,d] x [C,d] or aligned [B,C,d] score.

    Weighted normalized LSE uses support masses, with a stable reduction; masked
    slots never enter a reduction. All invalid and zero candidate rows return 0.
    """
    p, c = np.asarray(centers), np.asarray(candidates)
    valid, mass = np.asarray(valid), np.asarray(mass)
    if (p.ndim != 3 or c.ndim not in (2, 3) or p.shape[2] != c.shape[-1]
            or (c.ndim == 3 and c.shape[0] != p.shape[0])
            or valid.shape != p.shape[:2] or valid.dtype.kind != "b" or mass.shape != valid.shape
            or not np.isfinite(p).all() or not np.isfinite(c).all() or not np.isfinite(mass).all()
            or np.any(mass < 0) or np.any(mass[~valid] != 0)
            or np.any(valid & (mass <= 0))
            or not np.allclose(mass.sum(axis=1), valid.any(axis=1), atol=1e-7)):
        raise ValueError("Malformed interest/candidate arrays")
    if method not in ("max", "lse", "attention") or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid multi-interest aggregation")
    pnorm, cnorm = np.linalg.norm(p, axis=-1), np.linalg.norm(c, axis=-1)
    if (not np.allclose(pnorm[valid], 1., atol=1e-7, rtol=0) or np.any(p[~valid] != 0)
            or not np.allclose(cnorm[cnorm > EPSILON], 1., atol=1e-7, rtol=0)):
        raise ValueError("Cosine inputs require unit valid profiles/candidates and zero invalid profiles")
    scores = (np.einsum("bkd,cd->bkc", p, c, optimize=True) if c.ndim == 2
              else np.matmul(p, np.swapaxes(c, 1, 2)))
    candidate_valid = cnorm[None, :] > EPSILON if c.ndim == 2 else cnorm > EPSILON
    pair_valid = valid.any(axis=1)[:, None] & candidate_valid
    safe = np.where(valid[:, :, None], scores, -np.inf)
    maximum = np.max(safe, axis=1, initial=-np.inf)
    maximum = np.where(np.isfinite(maximum), maximum, 0.)
    if method == "max":
        output = maximum
    else:
        weights = np.exp(np.where(valid[:, :, None], (scores - maximum[:, None, :]) / temperature,
                                  -np.inf)) * mass[:, :, None]
        denominator = weights.sum(axis=1)
        if method == "lse":
            output = maximum + temperature * np.log(np.where(denominator > 0, denominator, 1.))
        else:
            output = np.divide((weights * scores).sum(axis=1), denominator,
                               out=np.zeros_like(denominator), where=denominator > 0)
    return np.where(pair_valid, output, 0.).astype(np.float32), pair_valid


class QueryProfiles:
    """Caller-owned immutable-by-contract query state; contains no target keys."""
    def __init__(self, **values):
        self.__dict__.update(values)


def _relation(value, columns, size):
    if isinstance(value, dict):
        columns = value.get("key_columns", columns)
        offsets, entities = np.asarray(value["offsets"]), np.asarray(value["entities"])
        if (offsets.shape != (size + 1,) or offsets.dtype.kind not in "iu"
                or entities.ndim != 1 or entities.dtype.kind not in "iu"
                or offsets[0] != 0 or offsets[-1] != len(entities)
                or np.any(offsets < 0) or np.any(offsets > len(entities))
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("Invalid exact-key CSR")
        rows = [entities[offsets[i]:offsets[i + 1]].astype(np.int64, copy=True) for i in range(size)]
    else:
        matrix = np.asarray(value)
        if matrix.ndim != 2 or matrix.shape[0] != size or not np.isin(matrix, (0, 1)).all():
            raise ValueError("Exact-key matrix must be binary [vocab,key]")
        rows = [np.flatnonzero(row) for row in matrix]
        if columns is None:
            raise ValueError("Ordered exact-key columns are required")
        if len(columns) != matrix.shape[1]:
            raise ValueError("Exact-key matrix/column mismatch")
    if (columns is None or len(set(columns)) != len(columns)
            or any(not isinstance(key, str) or not key for key in columns)
            or any(len(row) and (np.any(row < 0) or np.any(row >= len(columns))
                                 or len(np.unique(row)) != len(row)) for row in rows)
            or any(len(row) for row in rows[:3])):
        raise ValueError("Invalid exact-key identity mapping/special rows")
    return rows, list(columns)


class PersonalCandidateIndex:
    """User preparation once; pair or catalog-chunk compatibility thereafter."""
    def __init__(self, history_index, *, key_matrices=None, key_columns=None,
                 text_matrices=None, text_missing=None, genre_matrix=None,
                 genre_columns=None, representation_manifest=None):
        self.history = history_index
        self._owner = object()
        self.vocab_size = history_index.vocab_size
        size = self.vocab_size
        if genre_matrix is None:
            genre_columns = list(history_index.categories["genres"])
            genre_matrix = np.zeros((size, len(genre_columns)), np.float64)
            for token in range(size):
                labels = history_index.labels["genres"][token]
                genre_matrix[token, labels[labels >= 0]] = 1
        a = np.asarray(genre_matrix, dtype=np.float64).copy()
        if (a.ndim != 2 or a.shape[0] != size or genre_columns is None
                or a.shape[1] != len(genre_columns) or len(set(genre_columns)) != len(genre_columns)
                or not np.isin(a, (0, 1)).all() or np.any(a[:3] != 0)):
            raise ValueError("Invalid genre incidence/schema/special rows")
        for i, column in enumerate(genre_columns):
            if column in history.EXCLUDED_GENRES:
                a[:, i] = 0
        self.genre_incidence, self.genre_columns = a, list(genre_columns)
        self.genre_vectors = _unit(a)
        self.key_rows, self.key_columns = {}, {}
        key_matrices, key_columns = key_matrices or {}, key_columns or {}
        if set(key_matrices) - set(ROLES) or set(key_columns) - set(ROLES):
            raise ValueError("Unknown familiarity role")
        for role in ROLES:
            value = key_matrices.get(role, np.zeros((size, 0), np.uint8))
            self.key_rows[role], self.key_columns[role] = _relation(value, key_columns.get(role, []), size)
        self.text_matrices, self.text_missing = {}, {}
        if set(text_missing or {}) != set(text_matrices or {}):
            raise ValueError("Every text matrix requires an explicit missing mask")
        for field, matrix in sorted((text_matrices or {}).items()):
            if field not in ("title", "title_orig", "description"):
                raise ValueError("Unsupported text field")
            z, missing = np.asarray(matrix), np.asarray(text_missing[field])
            if (z.ndim != 2 or z.shape[0] != size or z.shape[1] < 1 or z.dtype.kind != "f"
                    or not np.isfinite(z).all() or missing.shape != (size,) or missing.dtype.kind != "b"
                    or not missing[:3].all() or np.any(z[missing] != 0)
                    or not np.allclose(np.linalg.norm(z[~missing], axis=1), 1., atol=1e-4, rtol=0)):
                raise ValueError("Text cache must have unit known rows, zero missing/specials and boolean masks")
            self.text_matrices[field], self.text_missing[field] = z.copy(), missing.copy()
        # Historical popularity is evaluated before each history row's own day.
        known = history_index.tokens >= 3
        self.pop_keys = np.sort(history_index.tokens[known].astype(np.int64) * history.MAX_ORDINAL
                                + history_index.days[known])
        self.pop_base = np.searchsorted(self.pop_keys, np.arange(size, dtype=np.int64) * history.MAX_ORDINAL)
        self.global_days = np.sort(history_index.days)
        self.columns, self.groups = [], {}
        self._layout()
        self.user_columns = ["history_pop." + name for name in ("mean", "median", "n", "valid_n", "known_share",
            "positive_n", "global_empty_n", "empty_history", "profile_missing")]
        self.provenance = {"contract": CONTRACT, "history_cap": 63,
            "clock": "strict raw history day < explicit cutoff <= query origin; frozen origins supplied by caller",
            "popularity_clock": "history row baseline < min(row day, role fit cutoff); all raw retained pairs",
            "sort": "user,date,raw_item_id then cap63; UNK retained and never backfilled",
            "metadata_availability_mode": "static_retrospective_catalog_snapshot",
            "normalizer": KEY_VERSION, "unicode_version": unicodedata.unidata_version,
            "fit_cutoff_exclusive": history_index.fit_cutoff,
            "mapping_sha256": canonical_hash(history_index.mapping),
            "genre_columns": self.genre_columns, "key_columns": self.key_columns,
            "text_fields": list(self.text_matrices), "text_epsilon": EPSILON,
            "text_norm_atol": 1e-4, "text_accumulator": "float64", "partition": {"iterations": 10, "min_n": {"2": 8, "3": 12},
                "min_support": 3, "min_coherence": .6, "max_center_cosine": .8, "min_incremental_gain": .10,
                "dtype": "float64 reference", "tie_tolerance": TIE_TOLERANCE,
                "k3_requires_accepted_k2": True, "fallback_reasons": FALLBACK},
            "representation_manifest": representation_manifest or {}, "columns": self.columns,
            "user_columns": self.user_columns,
            "history_source_pins": history_index.source_pins,
            "genre_incidence_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "key_incidence_sha256": {role: hashlib.sha256(b"".join(np.asarray(row, np.int64).tobytes()
                + b";" for row in self.key_rows[role])).hexdigest() for role in ROLES},
            "text_sha256": {field: hashlib.sha256(z.tobytes()).hexdigest() for field, z in self.text_matrices.items()}}
        self.schema_hash = canonical_hash(self.provenance)

    def _add(self, group, suffixes):
        start = len(self.columns)
        self.columns.extend(f"{group}.{name}" for name in suffixes)
        self.groups[group] = {"start": start, "stop": len(self.columns)}

    def _layout(self):
        for role in ROLES:
            self._add("familiarity_" + role, ["mean_count", "max_count", "matched_fraction", "candidate_key_count",
                "matched_key_count", "candidate_metadata_missing", "n_known", "distinct_history_keys", "coverage",
                "missing_share", "empty_profile"])
        self._add("familiarity", ["n", "empty_history"])
        self._add("director_novelty", ["seen_any", "unseen_fraction", "missing", "history_n", "history_known_n",
            "candidate_key_n", "candidate_metadata_missing", "empty_history", "history_metadata_missing"])
        for k in (1, 2, 3):
            self._add(f"interest_k{k}", ["max_cosine", "lse_cosine", "cross_valid", "candidate_missing",
                "k_eff", "history_n", "known_n", "coverage", "profile_missing", "fallback_reason",
                "objective", "gain", "max_pair_cosine", "min_cluster_support", "missing_share", "empty_history", "has_unseen"])
        for field in self.text_matrices:
            self._add("text_" + field, ["cosine", "cross_valid", "candidate_missing", "history_n", "known_n",
                "coverage", "missing_share", "empty_history", "profile_missing", "profile_zero_norm", "profile_valid", "mean_norm"])

    def prepare_queries(self, user_ids, query_dates, *, x=None, lengths=None, history_cutoffs=None):
        idx, mask, tokens, n, origins = self.history._history(user_ids, query_dates, history_cutoffs, x, lengths)
        count = len(n)
        interests = {k: [] for k in (1, 2, 3)}
        keys, known_n, distinct_n = {}, {}, {}
        for role in ROLES:
            keys[role], known_n[role], distinct_n[role] = [], np.zeros(count, np.int32), np.zeros(count, np.int32)
        for row in range(count):
            selected = tokens[row, mask[row]]
            for k in interests:
                interests[k].append(partition_interests(self.genre_incidence[selected], k))
            for role in ROLES:
                rows = [self.key_rows[role][token] for token in selected]
                counter = Counter(int(key) for values in rows for key in values)
                keys[role].append(counter)
                known_n[role][row], distinct_n[role][row] = sum(bool(len(values)) for values in rows), len(counter)
        text = {}
        for field, z in self.text_matrices.items():
            valid = mask & ~self.text_missing[field][tokens]
            support = valid.sum(axis=1)
            mean = np.divide(np.where(valid[:, :, None], z[tokens], 0).sum(axis=1, dtype=np.float64),
                support[:, None], out=np.zeros((count, z.shape[1]), np.float64), where=support[:, None] > 0)
            norms = np.linalg.norm(mean, axis=1)
            text[field] = {"profile": _unit(mean), "n_known": support, "mean_norm": norms,
                "valid": (support > 0) & (norms > EPSILON), "zero_norm": (support > 0) & (norms <= EPSILON)}
        days = self.history.days[idx] if len(self.history.days) else np.zeros_like(idx)
        bounds = np.minimum(days, self.history.fit_cutoff)
        pop_n = np.searchsorted(self.pop_keys, tokens.astype(np.int64) * history.MAX_ORDINAL + bounds,
                                side="left") - self.pop_base[tokens]
        pop_valid = mask & (tokens >= 3)
        support = pop_valid.sum(axis=1)
        pop = np.log1p(np.where(pop_valid, pop_n, 0))
        user_features = np.zeros((count, len(self.user_columns)), np.float32)
        user_features[:, 0] = np.divide(pop.sum(axis=1), support, out=np.zeros(count), where=support > 0)
        for row in range(count):
            if support[row]:
                user_features[row, 1] = np.median(pop[row, pop_valid[row]])
        user_features[:, 2:] = np.column_stack((n, support, np.divide(support, n, out=np.zeros(count), where=n > 0),
            ((pop_n > 0) & pop_valid).sum(axis=1),
            ((np.searchsorted(self.global_days, bounds, side="left") == 0) & pop_valid).sum(axis=1), n == 0, support == 0))
        partial = np.any(self.history.unseen["genres"][tokens] & mask, axis=1)
        query_cutoffs = origins.copy() if history_cutoffs is None else np.asarray(
            [history._day(value) for value in history_cutoffs], np.int64)
        return QueryProfiles(owner_schema_hash=self.schema_hash, owner=self._owner, n=n, origins=origins,
            user_ids=history._ids(user_ids, "query user IDs").copy(), query_cutoffs=query_cutoffs,
            tokens=tokens, mask=mask, genre_has_unseen=partial,
            interests=interests, keys=keys, known_n=known_n, distinct_n=distinct_n, text=text, user_features=user_features)

    def _check_profiles(self, profiles):
        if (not isinstance(profiles, QueryProfiles) or profiles.owner_schema_hash != self.schema_hash
                or profiles.owner is not self._owner):
            raise ValueError("Query profile schema/representation mismatch")

    def concatenate_profiles(self, states):
        """Copy prepared batches in order, retaining duplicates and all query fields.

        No history lookup or profile fitting is repeated. The nonempty iterable
        may contain variable-sized (including zero-row) batches from this exact
        index. Arrays, Counters and interest dictionaries never alias the inputs
        or another repeated row. Candidate scoring remains bounded by [sum B,C].
        """
        states = tuple(states)
        if not states:
            raise ValueError("At least one prepared query batch is required")
        fields = {"n", "origins", "user_ids", "query_cutoffs", "tokens", "mask", "genre_has_unseen",
                  "interests", "keys", "known_n", "distinct_n", "text", "user_features"}
        interest_shapes = {"centers": (3, len(self.genre_columns)), "incidence": (3, len(self.genre_columns)),
            "support": (3,), "mass": (3,), "coherence": (3,), "valid": (3,),
            **{name: () for name in ("k_eff", "n_known", "reason", "objective", "gain", "max_pair_cosine")}}

        def array(value, shape, kinds):
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype.kind not in kinds:
                raise ValueError("Malformed prepared query array shape/dtype")

        for state in states:
            if (not isinstance(state, QueryProfiles)
                    or set(vars(state)) != fields | {"owner", "owner_schema_hash"}):
                raise ValueError("Incomplete or unexpected prepared query fields")
            self._check_profiles(state)
            if not isinstance(state.n, np.ndarray) or state.n.ndim != 1:
                raise ValueError("Prepared history counts must be a vector")
            b = len(state.n)
            for name in ("n", "origins", "user_ids", "query_cutoffs"):
                array(getattr(state, name), (b,), "iu")
            array(state.tokens, (b, 63), "iu")
            array(state.mask, (b, 63), "b")
            array(state.genre_has_unseen, (b,), "b")
            array(state.user_features, (b, len(self.user_columns)), "f")
            expected_mask = np.arange(63)[None, :] < state.n[:, None]
            if (np.any(state.n < 0) or np.any(state.n > 63) or np.any(state.user_ids < 0)
                    or np.any(state.query_cutoffs > state.origins) or not np.array_equal(state.mask, expected_mask)
                    or np.any(state.tokens[~state.mask] != 0) or np.any(state.tokens[state.mask] < 2)
                    or np.any(state.tokens >= self.vocab_size)):
                raise ValueError("Prepared query history/clock fields disagree")
            for name in ("keys", "known_n", "distinct_n"):
                if not isinstance(getattr(state, name), dict) or set(getattr(state, name)) != set(ROLES):
                    raise ValueError("Prepared familiarity roles disagree")
            for role in ROLES:
                counters = state.keys[role]
                if (not isinstance(counters, list) or len(counters) != b
                        or any(not isinstance(counter, Counter) for counter in counters)):
                    raise ValueError("Prepared familiarity rows disagree")
                array(state.known_n[role], (b,), "iu")
                array(state.distinct_n[role], (b,), "iu")
            if not isinstance(state.interests, dict) or set(state.interests) != {1, 2, 3}:
                raise ValueError("Prepared interest modes disagree")
            for values in state.interests.values():
                if not isinstance(values, list) or len(values) != b:
                    raise ValueError("Prepared interest rows disagree")
                for value in values:
                    if not isinstance(value, dict) or set(value) != set(interest_shapes):
                        raise ValueError("Prepared interest fields disagree")
                    if any(np.shape(value[name]) != shape for name, shape in interest_shapes.items()):
                        raise ValueError("Prepared interest shapes disagree")
            if not isinstance(state.text, dict) or set(state.text) != set(self.text_matrices):
                raise ValueError("Prepared text representations disagree")
            for field, value in state.text.items():
                if (not isinstance(value, dict)
                        or set(value) != {"profile", "n_known", "mean_norm", "valid", "zero_norm"}):
                    raise ValueError("Prepared text fields disagree")
                array(value["profile"], (b, self.text_matrices[field].shape[1]), "f")
                array(value["n_known"], (b,), "iu")
                array(value["mean_norm"], (b,), "f")
                array(value["valid"], (b,), "b")
                array(value["zero_norm"], (b,), "b")

        def join(values):
            first = values[0]
            if isinstance(first, np.ndarray):
                if any(value.dtype != first.dtype for value in values):
                    raise ValueError("Prepared array dtypes disagree")
                return np.concatenate(values, axis=0)
            if isinstance(first, dict):
                return {key: join([value[key] for value in values]) for key in first}
            # Per-row interest dictionaries and Counters are independent even
            # when the same source row occurs more than once in this batch.
            return [deepcopy(row) for value in values for row in value]

        return QueryProfiles(owner_schema_hash=self.schema_hash, owner=self._owner,
            **{name: join([getattr(state, name) for state in states]) for name in fields})

    def candidate_features(self, profiles, item_tokens, *, max_interests=None):
        """Shared [C] or aligned [B,C] real tokens -> float32 [B,C,D].

        The aligned path never constructs a batch-wide candidate union.
        max_interests=1/2/3 keeps a matched 17-column interest block; None emits
        all declared blocks, convenient for a caller-controlled ablation mask.
        """
        self._check_profiles(profiles)
        tokens = np.asarray(item_tokens)
        b = len(profiles.n)
        if (tokens.ndim not in (1, 2) or (tokens.ndim == 2 and tokens.shape[0] != b)
                or tokens.dtype.kind not in "iu" or np.any(tokens < 3)
                or np.any(tokens >= self.vocab_size)):
            raise ValueError("Candidates must be shared [C] or aligned [B,C] real mapped item tokens")
        if max_interests is not None and (max_interests not in (1, 2, 3) or isinstance(max_interests, bool)):
            raise ValueError("max_interests must be 1, 2 or 3")
        aligned = tokens.ndim == 2
        c = tokens.shape[-1]
        output = np.zeros((b, c, len(self.columns)), np.float32)
        def put(group, values):
            sl = self.groups[group]
            output[:, :, sl["start"]:sl["stop"]] = np.asarray(values).transpose(1, 2, 0)
        def broad(values):
            return np.broadcast_to(np.asarray(values)[:, None], (b, c))
        n = profiles.n
        director_matched = None
        for role in ROLES:
            rows = ([self.key_rows[role][token] for token in tokens.ravel()] if aligned
                    else [self.key_rows[role][token] for token in tokens])
            m = np.array([len(row) for row in rows], np.int64)
            m = m.reshape(b, c) if aligned else np.broadcast_to(m, (b, c))
            mean, maximum, matched = (np.zeros((b, c), np.float64) for _ in range(3))
            for row, counter in enumerate(profiles.keys[role]):
                candidate_rows = rows[row * c:(row + 1) * c] if aligned else rows
                for col, candidate in enumerate(candidate_rows):
                    values = [counter[int(key)] for key in candidate]
                    if values:
                        mean[row, col], maximum[row, col] = np.mean(values), max(values)
                        matched[row, col] = np.count_nonzero(values)
            nf, df = profiles.known_n[role], profiles.distinct_n[role]
            fraction = np.divide(matched, m, out=np.zeros_like(matched), where=m > 0)
            put("familiarity_" + role, [mean, maximum, fraction, m, matched,
                m == 0, broad(nf), broad(df),
                broad(np.divide(nf, n, out=np.zeros(b), where=n > 0)),
                broad(np.divide(n - nf, n, out=np.zeros(b), where=n > 0)), broad(nf == 0)])
            if role == "director":
                director_matched = matched, fraction, m, nf
        put("familiarity", [broad(n), broad(n == 0)])
        matched, fraction, m, nf = director_matched
        valid = (nf[:, None] > 0) & (m > 0)
        put("director_novelty", [(matched > 0) & valid, np.where(valid, 1 - fraction, 0), ~valid,
            broad(n), broad(nf), m, m == 0, broad(n == 0), broad(nf == 0)])
        for k in (1, 2, 3):
            states = profiles.interests[k]
            centers = np.stack([state["centers"] for state in states]) if b else np.zeros((0, 3, len(self.genre_columns)))
            masks = np.array([state["valid"] for state in states], bool).reshape(b, 3)
            mass = np.array([state["mass"] for state in states]).reshape(b, 3)
            score, cross_valid = multi_interest_scores(centers, masks, mass, self.genre_vectors[tokens])
            lse, _ = multi_interest_scores(centers, masks, mass, self.genre_vectors[tokens], method="lse")
            ng = np.array([state["n_known"] for state in states])
            candidate_missing = ~np.any(self.genre_incidence[tokens], axis=-1)
            values = [score, lse, cross_valid, candidate_missing if aligned else np.broadcast_to(candidate_missing, (b, c))]
            values += [broad(np.array([state[name] for state in states])) for name in ("k_eff",)]
            values += [broad(n), broad(ng), broad(np.divide(ng, n, out=np.zeros(b), where=n > 0)), broad(ng == 0)]
            values += [broad(np.array([state[name] for state in states])) for name in ("reason", "objective", "gain", "max_pair_cosine")]
            values += [broad(np.array([min(state["support"][state["valid"]], default=0) for state in states]))]
            values += [broad(np.divide(n - ng, n, out=np.zeros(b), where=n > 0)), broad(n == 0), broad(profiles.genre_has_unseen)]
            put(f"interest_k{k}", values)
        for field, state in profiles.text.items():
            missing = self.text_missing[field][tokens]
            cross_valid = state["valid"][:, None] & ~missing
            vectors = self.text_matrices[field][tokens]
            cosine = (np.matmul(state["profile"][:, None, :], np.swapaxes(vectors, 1, 2))[:, 0, :]
                      if aligned else state["profile"] @ vectors.T)
            support = state["n_known"]
            put("text_" + field, [np.where(cross_valid, cosine, 0), cross_valid, np.broadcast_to(missing, (b, c)),
                broad(n), broad(support), broad(np.divide(support, n, out=np.zeros(b), where=n > 0)),
                broad(np.divide(n - support, n, out=np.zeros(b), where=n > 0)), broad(n == 0), broad(support == 0),
                broad(state["zero_norm"]), broad(state["valid"]), broad(state["mean_norm"])])
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite compatibility output")
        if max_interests is not None:
            return output[:, :, self.selected_indices(max_interests)]
        return output

    def selected_indices(self, max_interests):
        if max_interests not in (1, 2, 3) or isinstance(max_interests, bool):
            raise ValueError("max_interests must be 1, 2 or 3")
        return [i for i, name in enumerate(self.columns) if not name.startswith("interest_k")
                or name.startswith(f"interest_k{max_interests}.")]

    def pair_features(self, user_ids, query_dates, item_tokens, *, x=None, lengths=None, history_cutoffs=None,
                      profiles=None, max_interests=None):
        """Paired float32 [B,D], without constructing a B x B candidate tensor."""
        profiles = profiles if profiles is not None else self.prepare_queries(user_ids, query_dates,
            x=x, lengths=lengths, history_cutoffs=history_cutoffs)
        self._check_profiles(profiles)
        users = history._ids(user_ids, "query user IDs")
        origins = np.asarray([history._day(value) for value in query_dates], np.int64)
        cutoffs = origins if history_cutoffs is None else np.asarray(
            [history._day(value) for value in history_cutoffs], np.int64)
        if (not np.array_equal(users, profiles.user_ids) or not np.array_equal(origins, profiles.origins)
                or not np.array_equal(cutoffs, profiles.query_cutoffs)):
            raise ValueError("Cached query profiles disagree with supplied user/origin/cutoff identity")
        if x is not None or lengths is not None:
            self.history._history(users, origins, cutoffs, x=x, lengths=lengths)
        candidates = np.asarray(item_tokens)
        if candidates.shape != (len(profiles.n),):
            raise ValueError("One candidate per query is required")
        if candidates.dtype.kind not in "iu" or np.any(candidates < 3) or np.any(candidates >= self.vocab_size):
            raise ValueError("Paired candidates must be real integer mapped tokens")
        return self.candidate_features(profiles, candidates[:, None], max_interests=max_interests)[:, 0, :]

    def _one_query(self, profiles, row):
        return QueryProfiles(owner_schema_hash=self.schema_hash, owner=self._owner, n=profiles.n[row:row + 1],
            user_ids=profiles.user_ids[row:row + 1], origins=profiles.origins[row:row + 1],
            query_cutoffs=profiles.query_cutoffs[row:row + 1],
            tokens=profiles.tokens[row:row + 1], mask=profiles.mask[row:row + 1],
            user_features=profiles.user_features[row:row + 1],
            genre_has_unseen=profiles.genre_has_unseen[row:row + 1],
            interests={k: states[row:row + 1] for k, states in profiles.interests.items()},
            keys={role: values[row:row + 1] for role, values in profiles.keys.items()},
            known_n={role: values[row:row + 1] for role, values in profiles.known_n.items()},
            distinct_n={role: values[row:row + 1] for role, values in profiles.distinct_n.items()},
            text={field: {name: values[row:row + 1] for name, values in state.items()} for field, state in profiles.text.items()})

    def candidate_pairs(self, profiles, item_tokens, *, max_interests=None):
        """Per-query candidate matrix [B,C] -> [B,C,D] in one bounded call."""
        self._check_profiles(profiles)
        tokens = np.asarray(item_tokens)
        if (tokens.ndim != 2 or tokens.shape[0] != len(profiles.n) or tokens.dtype.kind not in "iu"
                or np.any(tokens < 3) or np.any(tokens >= self.vocab_size)):
            raise ValueError("Candidate pairs require a [batch,candidates] real-token integer matrix")
        return self.candidate_features(profiles, tokens, max_interests=max_interests)

    def iter_candidate_features(self, profiles, item_tokens=None, *, chunk_size=256, max_interests=None):
        """Yield (start,stop,[B,chunk,D]); no full [B,V,D] allocation."""
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        tokens = np.arange(3, self.vocab_size) if item_tokens is None else np.asarray(item_tokens)
        # Validate even an empty candidate vector.
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 3) or np.any(tokens >= self.vocab_size):
            raise ValueError("Invalid candidate vector")
        self._check_profiles(profiles)
        for start in range(0, len(tokens), chunk_size):
            stop = min(start + chunk_size, len(tokens))
            yield start, stop, self.candidate_features(profiles, tokens[start:stop], max_interests=max_interests)


def build_index(raw_dir, metadata, *, history_index=None, fit_cutoff="2021-08-09", text_matrices=None,
                text_missing=None, genre_matrix=None, genre_columns=None, representation_manifest=None):
    """Local metadata adapter; injected text has already been obtained/validated."""
    raw_dir = Path(raw_dir)
    meta = json.loads(Path(metadata).read_text(encoding="utf-8")) if isinstance(metadata, (str, Path)) else metadata
    index = history_index or history.build_history_index(raw_dir, meta, fit_cutoff=fit_cutoff,
        schema_item_ids=meta["item_id_to_token"].keys())
    if index.mapping != {int(key): value for key, value in meta["item_id_to_token"].items()}:
        raise ValueError("History and metadata mappings differ")
    if genre_matrix is None and "item_categories" in meta:
        genre_columns = list(meta["item_categories"]["genres"])
        genre_matrix = np.zeros((index.vocab_size, len(genre_columns)), np.float64)
        lookup = {name: i for i, name in enumerate(genre_columns)}
        for old, name in enumerate(index.categories["genres"]):
            if name not in lookup:
                raise ValueError("History genre schema exceeds the fixed prepared vocabulary")
            genre_matrix[:, lookup[name]] = np.any(index.labels["genres"] == old, axis=1)
    token_keys = {role: [() for _ in range(index.vocab_size)] for role in ROLES}
    for iid, actors, directors, studios in history._csv_rows(raw_dir / "items.csv", ("item_id", "actors", "directors", "studios")):
        token = index.mapping.get(int(iid))
        if token is not None:
            for role, raw in zip(ROLES[:3], (actors, directors, studios)):
                token_keys[role][token] = exact_keys(raw)
    explicit = meta.get("series_annotation_path")
    candidates = [Path(explicit)] if explicit else [parent / "configs/datasets/kion-series-v1.json" for parent in (raw_dir, *raw_dir.parents)]
    series_path = next((path for path in candidates if path.is_file()), None)
    if series_path is None:
        raise ValueError("Confirmed franchise annotation is required; absence is not an empty annotation")
    annotation = json.loads(series_path.read_text(encoding="utf-8"))
    seen_ids = set()
    for series in annotation["series"]:
        if series.get("status") != "confirmed":
            continue
        key = series["id"]
        if not isinstance(key, str) or not key or key in seen_ids:
            raise ValueError("Invalid/duplicate confirmed franchise key")
        seen_ids.add(key)
        for member in series["members"]:
            if not isinstance(member, dict) or member.get("status") != "confirmed":
                raise ValueError("Only explicitly confirmed franchise memberships are permitted")
            iid = member["item_id"] if isinstance(member, dict) else member
            token = index.mapping.get(int(iid))
            if token is not None:
                token_keys["franchise"][token] = tuple(sorted(set(token_keys["franchise"][token]) | {key}))
    matrices = {}
    for role, rows in token_keys.items():
        columns = sorted({key for row in rows for key in row})
        mapping = {key: pos for pos, key in enumerate(columns)}
        matrices[role] = {"offsets": np.r_[0, np.cumsum([len(row) for row in rows])].astype(np.int64),
            "entities": np.array([mapping[key] for row in rows for key in row], np.int64), "key_columns": columns}
    manifest = dict(representation_manifest or {})
    manifest.update(series_annotation_sha256=hashlib.sha256(series_path.read_bytes()).hexdigest(),
                    items_sha256=hashlib.sha256((raw_dir / "items.csv").read_bytes()).hexdigest())
    return PersonalCandidateIndex(index, key_matrices=matrices, text_matrices=text_matrices,
        text_missing=text_missing, genre_matrix=genre_matrix, genre_columns=genre_columns,
        representation_manifest=manifest)
