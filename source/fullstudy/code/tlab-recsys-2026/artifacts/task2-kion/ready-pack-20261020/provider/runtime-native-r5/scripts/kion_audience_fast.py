"""Bounded exact acceleration of the reviewed 119-coordinate audience contract.

The core/index are read-only. Both explicit co-view kernels count the same full
snapshot intersections; query user is then removed analytically from counts,
degrees, priors and frozen viewer-profile sufficient statistics. No fitted data.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_audience_features as core


CONTRACT = "kion-audience-fast-exact-v1"
ATOL = 2e-6
RTOL = 2e-6


def _cosine(a, b):
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.einsum("ij,ij->i", a, b), norms, out=np.zeros(len(a)), where=norms > 0)


class FastAudienceIndex:
    def __init__(self, index, *, pair_mode="search", max_pair_cache=250_000,
                 max_gram_cells=2_000_000, max_pair_work=50_000_000):
        if pair_mode not in ("search", "gram"):
            raise ValueError("pair_mode must explicitly be search or gram")
        if any(type(v) is not int or v < 1 for v in (max_pair_cache, max_gram_cells, max_pair_work)):
            raise ValueError("Positive exact-kernel budgets required")
        self.index, self.pair_mode = index, pair_mode
        self.max_pair_cache, self.max_gram_cells, self.max_pair_work = max_pair_cache, max_gram_cells, max_pair_work
        self.columns, self.groups = index.columns, index.groups
        self.column_index = index.column_index
        self._snapshot_ref = None
        self._pairs = OrderedDict()
        self._incidence = None
        self.timings = {"snapshot_seconds": 0., "fast_setup_seconds": 0., "pair_seconds": 0.,
                        "pair_cache_hits": 0, "pair_cache_misses": 0, "gram_cells": 0}
        self.provenance = {"contract": CONTRACT, "core_contract": core.CONTRACT,
            "pair_mode": pair_mode, "atol": ATOL, "rtol": RTOL,
            "max_pair_cache": max_pair_cache, "max_gram_cells": max_gram_cells,
            "max_pair_work": max_pair_work, "formula_change": False,
            "scipy": "required only for explicit gram mode; no alternate-mode fallback"}

    def _snapshot(self, cutoff):
        before = time.perf_counter()
        snapshot = self.index._snapshot(cutoff)
        self.timings["snapshot_seconds"] += time.perf_counter() - before
        if snapshot is self._snapshot_ref:
            return snapshot
        before = time.perf_counter()
        self._snapshot_ref = snapshot
        self._pairs.clear()
        self._incidence = None
        # First/second distinct informative profile lengths and endpoint counts
        # permit exact min/max after deleting one viewer, without per-row copies.
        self._extrema = np.zeros((self.index.vocab_size, 6), np.int64)
        for token, (sizes, counts) in enumerate(snapshot["lengths_by_item"]):
            if len(sizes):
                self._extrema[token] = [sizes[0], sizes[1] if len(sizes) > 1 else 0, counts[0],
                    sizes[-1], sizes[-2] if len(sizes) > 1 else 0, counts[-1]]
        if self.pair_mode == "gram":
            from scipy.sparse import csc_matrix
            # O(E) CSC uses the core's sorted audience indices directly. int32
            # ones/counts avoid int8 overflow; viewer count is bounded by int32.
            self._incidence = csc_matrix((np.ones(len(snapshot["viewers"]), np.int32),
                snapshot["viewers"], snapshot["offsets"]),
                shape=(len(self.index.unique_users), self.index.vocab_size), copy=False)
        self.timings["fast_setup_seconds"] += time.perf_counter() - before
        return snapshot

    def _pair_counts(self, candidates, histories, valid, snapshot):
        """Full-background c_ij. User exclusion happens once at the caller."""
        output = np.zeros(valid.shape, dtype=np.int64)
        if not valid.any():
            return output
        before = time.perf_counter()
        vocab = self.index.vocab_size
        pairs = np.minimum(candidates[:, None], histories) * vocab + np.maximum(candidates[:, None], histories)
        unique, inverse = np.unique(pairs[valid], return_inverse=True)
        counts = np.zeros(len(unique), dtype=np.int64)
        missing = []
        for position, key in enumerate(unique):
            found = self._pairs.get(int(key))
            if found is None:
                missing.append(position)
            else:
                counts[position] = found
        self.timings["pair_cache_hits"] += len(unique) - len(missing)
        self.timings["pair_cache_misses"] += len(missing)
        if missing:
            needed = unique[missing]
            left, right = np.divmod(needed, vocab)
            if self.pair_mode == "gram":
                # Canonical pair keys are symmetric, but sorting each pair by
                # token can spread random candidates across both Gram axes.
                # Orient one axis to actual query-history items to keep the
                # requested rectangle narrow without dropping any needed pair.
                history_items = np.unique(histories[valid])
                left_is_history = np.isin(left, history_items)
                gram_left = np.where(left_is_history, right, left)
                gram_right = np.where(left_is_history, left, right)
                item_left, l_inverse = np.unique(gram_left, return_inverse=True)
                item_right, r_inverse = np.unique(gram_right, return_inverse=True)
                cells = len(item_left) * len(item_right)
                if cells > self.max_gram_cells:
                    raise ValueError("Exact Gram rectangle budget exceeded; reduce candidate batch")
                # Request only a bounded rectangle, never full catalog squared.
                gram = (self._incidence[:, item_left].T @ self._incidence[:, item_right]).toarray()
                fresh = gram[l_inverse, r_inverse].astype(np.int64)
                self.timings["gram_cells"] += cells
            else:
                fresh = np.zeros(len(needed), dtype=np.int64)
                offsets, viewers = snapshot["offsets"], snapshot["viewers"]
                work = 0
                for position, (i, j) in enumerate(zip(left, right)):
                    a = viewers[offsets[i]:offsets[i + 1]]
                    b = viewers[offsets[j]:offsets[j + 1]]
                    if len(a) > len(b):
                        a, b = b, a
                    work += len(a)
                    if work > self.max_pair_work:
                        raise ValueError("Exact binary-search pair budget exceeded; reduce candidate batch")
                    if len(a):
                        locations = np.searchsorted(b, a)
                        usable = locations < len(b)
                        fresh[position] = np.count_nonzero(b[locations[usable]] == a[usable])
            for position, value, key in zip(missing, fresh, needed):
                counts[position] = value
                self._pairs[int(key)] = int(value)
                if len(self._pairs) > self.max_pair_cache:
                    self._pairs.popitem(last=False)
        output[valid] = counts[inverse]
        self.timings["pair_seconds"] += time.perf_counter() - before
        return output

    @staticmethod
    def _dates(values, size, name):
        raw = np.asarray(values)
        result = np.full(size, core._day(raw.item()), np.int64) if raw.ndim == 0 else np.array([core._day(v) for v in raw])
        if result.shape != (size,):
            raise ValueError(name + " must align")
        return result

    def features(self, item_tokens, query_dates, *, user_ids, role="sft_train", x=None, lengths=None,
                 history_cutoffs=None):
        if role not in core.ROLES:
            raise ValueError("Unknown feature role")
        index = self.index
        tokens, users = core._ids(item_tokens, "item_tokens"), core._ids(user_ids, "user_ids")
        n = len(tokens)
        if len(users) != n or n > 4096:
            raise ValueError("Aligned query rows, maximum 4096, required")
        dates = self._dates(query_dates, n, "query dates")
        cuts = [None] * n if history_cutoffs is None else self._dates(history_cutoffs, n, "history cutoffs")
        if (x is None) != (lengths is None):
            raise ValueError("Both x and lengths required")
        if x is not None:
            x, lengths = np.asarray(x), np.asarray(lengths)
            if (x.shape != (n, 64) or x.dtype.kind not in "iu" or lengths.shape != (n,)
                    or lengths.dtype.kind not in "iu" or np.any(lengths < 1) or np.any(lengths > 64)):
                raise ValueError("Invalid BOS+H63 input")
        result = np.zeros((n, len(self.columns)), np.float32)
        result[tokens >= index.vocab_size, index.column_index["audience.unknown_item"]] = 1
        selected = np.flatnonzero((tokens >= 3) & (tokens < index.vocab_size))
        if not len(selected):
            return result
        bounds = np.array([index._bounds(int(dates[r]), role, cuts[r]) for r in selected], np.int64)
        for cutoff in np.unique(bounds[:, 0]):
            positions = np.flatnonzero(bounds[:, 0] == cutoff)
            rows = selected[positions]
            result[rows] = self._group(tokens[rows], users[rows], bounds[positions, 1], int(cutoff),
                None if x is None else x[rows], None if lengths is None else lengths[rows])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite fast audience output")
        return result

    def _group(self, tokens, users, state_cuts, cutoff, x, lengths):
        index = self.index
        snapshot = self._snapshot(cutoff)
        n, width = len(tokens), len(index.genres)
        out = np.zeros((n, len(self.columns)), np.float32)
        put = lambda name, values: out.__setitem__((slice(None), self.column_index[name]), values)
        query_codes, query_found = index._codes(users)
        positions = np.searchsorted(index.unique_users, users)
        found = positions < len(index.unique_users)
        if len(index.unique_users):
            found &= index.unique_users[np.minimum(positions, len(index.unique_users) - 1)] == users
        positions = np.where(found, positions, 0)
        active, member = np.zeros(n, bool), np.zeros(n, bool)
        global_degree = np.zeros(n, np.int64)
        global_known = np.zeros(n, np.int64)
        global_sum = np.zeros((n, width), np.float64)
        hist_n, hist_known_n, hist_empty_n, hist_missing_n = (np.zeros(n, np.int64) for _ in range(4))
        hist_sum = np.zeros((n, width), np.float64)
        repeat = np.zeros(n, bool)
        histories = np.zeros((n, 63), np.int64)
        history_members = np.zeros((n, 63), bool)
        pairs = np.column_stack((users, state_cuts))
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        global_histories = {}
        for key, (uid, state_cut) in enumerate(unique):
            rows = np.flatnonzero(inverse == key)
            h, upos = index._history(int(uid), int(state_cut))
            if x is not None:
                for row in rows:
                    size = int(lengths[row])
                    if x[row, 0] != 1 or np.any(x[row, size:] != 0) or not np.array_equal(x[row, 1:size], h):
                        raise ValueError("Provided context differs from strict-date retained H63")
            if int(uid) not in global_histories:
                if upos is None:
                    global_histories[int(uid)] = np.empty(0, np.int32)
                else:
                    left, right = index.user_offsets[upos:upos + 2]
                    end = left + np.searchsorted(index.days[left:right], cutoff, side="left")
                    gh = index.tokens[left:end]
                    global_histories[int(uid)] = np.unique(gh[gh >= 3])
            gh = global_histories[int(uid)]
            if upos is not None:
                active[rows] = snapshot["degrees"][upos] > 0
                global_degree[rows] = snapshot["degrees"][upos]
                global_known[rows] = snapshot["known_n"][upos]
                global_sum[rows] = snapshot["sums"][upos]
            member[rows] = np.isin(tokens[rows], gh)
            known = index.genre_known[h]
            hist_n[rows], hist_known_n[rows] = len(h), int(known.sum())
            hist_empty_n[rows], hist_missing_n[rows] = int(index.genre_empty[h].sum()), int(index.genre_missing[h].sum())
            hist_sum[rows] = index.genre_vectors[h[known]].sum(axis=0)
            repeat[rows] = np.isin(tokens[rows], h)
            other = np.unique(h[h >= 3])
            histories[rows, :len(other)] = other
            history_members[rows, :len(other)] = np.isin(other, gh)
        ni = snapshot["item_n"][tokens] - member
        put("audience.item_n", ni)
        put("audience.active_users_n", snapshot["active_users"] - active)
        put("audience.global_pairs_n", snapshot["global_pairs"] - global_degree)
        put("audience.query_global_member", member)
        put("audience.no_item_support", ni == 0)
        put("audience.global_cutoff_ordinal", cutoff)
        put("audience.query_state_cutoff_ordinal", state_cuts)
        for col, field in enumerate(core.FIELDS):
            k = len(index.categories[field])
            counts = snapshot["demography"][field][tokens].copy()
            codes = query_codes[:, col].astype(np.int64)
            category = np.where(codes >= 0, codes, k - codes - 1)
            counts[np.arange(n), category] -= member
            priors = np.broadcast_to(snapshot["priors"][field], (n, k)).copy()
            correct = active & (codes >= 0)
            priors[np.flatnonzero(correct), codes[correct]] -= 1
            support = priors.sum(axis=1)
            prior = np.divide(priors, support[:, None], out=np.full((n, k), 1 / k), where=support[:, None] > 0)
            known_n = counts[:, :k].sum(axis=1)
            probability = (counts[:, :k] + index.alpha * prior) / (known_n[:, None] + index.alpha)
            start = self.groups["demography_" + field]["start"]
            out[:, start:start + k] = probability
            out[:, start + k:start + 2 * k] = prior
            safe = np.maximum(codes, 0)
            match = probability[np.arange(n), safe]
            lift = np.log((match + 1e-12) / (prior[np.arange(n), safe] + 1e-12))
            prefix = "demography_" + field + "."
            for suffix, values in (("match", np.where(codes >= 0, match, 0)),
                ("log_lift", np.where(codes >= 0, lift, 0)), ("known_n", known_n),
                ("unknown_field_n", counts[:, k]), ("unseen_category_n", counts[:, k + 1]),
                ("no_user_row_n", counts[:, k + 2]),
                ("known_fraction", np.divide(known_n, ni, out=np.zeros(n), where=ni > 0)),
                ("global_known_n", support), ("no_known", known_n == 0), ("no_global_known", support == 0),
                ("query_field_missing", codes == -1), ("query_category_unseen", codes == -2),
                ("query_user_row_missing", ~query_found)):
                put(prefix + suffix, values)
        genre = index.genre_vectors[tokens]
        self_other = global_known - index.genre_known[tokens]
        contributed = member & (self_other > 0)
        informative = snapshot["informative_n"][tokens] - contributed
        taste_sum = snapshot["taste_sums"][tokens].copy()
        taste_sum[contributed] -= (global_sum[contributed] - genre[contributed]) / self_other[contributed, None]
        audience_profile = np.divide(taste_sum, informative[:, None], out=np.zeros_like(taste_sum), where=informative[:, None] > 0)
        total_other = snapshot["other_sum"][tokens] - np.where(contributed, self_other, 0)
        q_n = hist_known_n - repeat * index.genre_known[tokens]
        q_sum = hist_sum - repeat[:, None] * genre
        q = np.divide(q_sum, q_n[:, None], out=np.zeros_like(q_sum), where=q_n[:, None] > 0)
        ext = self._extrema[tokens]
        minimum = np.where(contributed & (self_other == ext[:, 0]) & (ext[:, 2] == 1), ext[:, 1], ext[:, 0])
        maximum = np.where(contributed & (self_other == ext[:, 3]) & (ext[:, 5] == 1), ext[:, 4], ext[:, 3])
        qh_n = hist_n - repeat
        for suffix, values in (("cosine", _cosine(q, audience_profile)),
            ("direct_genre_cosine", _cosine(q, genre)), ("informative_viewers_n", informative),
            ("audience_n", ni), ("coverage", np.divide(informative, ni, out=np.zeros(n), where=ni > 0)),
            ("other_known_items_mean", np.divide(total_other, informative, out=np.zeros(n), where=informative > 0)),
            ("other_known_items_min", np.where(informative > 0, minimum, 0)),
            ("other_known_items_max", np.where(informative > 0, maximum, 0)),
            ("query_history_n", qh_n), ("query_known_n", q_n),
            ("query_known_fraction", np.divide(q_n, qh_n, out=np.zeros(n), where=qh_n > 0)),
            ("query_genre_empty_n", hist_empty_n - repeat * index.genre_empty[tokens]),
            ("query_genre_missing_n", hist_missing_n - repeat * index.genre_missing[tokens]),
            ("query_candidate_repeat", repeat), ("no_audience_profile", informative == 0),
            ("no_query_profile", q_n == 0), ("cosine_valid", (informative > 0) & (q_n > 0)),
            ("candidate_genre_empty", index.genre_empty[tokens]), ("candidate_genre_missing", index.genre_missing[tokens])):
            put("taste." + suffix, values)
        valid = (histories >= 3) & (histories != tokens[:, None])
        cij = self._pair_counts(tokens, histories, valid, snapshot)
        cij -= (member[:, None] & history_members & valid)
        nj = snapshot["item_n"][histories] - history_members
        denominators = np.sqrt(ni[:, None] * nj)
        weights = np.divide(cij, denominators, out=np.zeros(cij.shape, np.float64), where=(denominators > 0) & valid)
        weights *= np.divide(cij, cij + index.coview_lambda, out=np.zeros_like(weights), where=valid)
        neighbor_n = valid.sum(axis=1)
        supported = ((weights > 0) & valid).sum(axis=1)
        for suffix, values in (("mean", np.divide(weights.sum(axis=1), neighbor_n, out=np.zeros(n), where=neighbor_n > 0)),
            ("max", weights.max(axis=1)), ("weight_sum", weights.sum(axis=1)), ("history_n", neighbor_n),
            ("supported_neighbors_n", supported),
            ("history_degree_mean", np.divide(np.where(valid, nj, 0).sum(axis=1), neighbor_n, out=np.zeros(n), where=neighbor_n > 0)),
            ("no_evidence", supported == 0), ("no_history", neighbor_n == 0)):
            put("coview." + suffix, values)
        return out

    def iter_features(self, item_tokens, query_dates, *, user_ids, batch_size=4096, **kwargs):
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be in [1,4096]")
        tokens, users = core._ids(item_tokens, "item_tokens"), core._ids(user_ids, "user_ids")
        n = len(tokens)
        if len(users) != n:
            raise ValueError("Query rows must align")
        dates = self._dates(query_dates, n, "query dates")
        for name in ("x", "lengths", "history_cutoffs"):
            value = kwargs.get(name)
            if value is None:
                continue
            shape = np.shape(value)
            expected = (n, 64) if name == "x" else (n,)
            if shape != expected and not (name == "history_cutoffs" and not shape):
                raise ValueError(name + " must align before batching")
        for begin in range(0, n, batch_size):
            end = min(begin + batch_size, n)
            options = {name: (np.asarray(value)[begin:end] if name in ("x", "lengths", "history_cutoffs") and value is not None and np.ndim(value) else value)
                       for name, value in kwargs.items()}
            yield begin, end, self.features(tokens[begin:end], dates[begin:end], user_ids=users[begin:end], **options)
