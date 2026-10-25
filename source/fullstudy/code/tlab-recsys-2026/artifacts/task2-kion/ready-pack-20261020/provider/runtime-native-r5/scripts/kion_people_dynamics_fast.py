"""Exact NumPy CPU adapter for the frozen people-dynamics feature contract.

The source index is unchanged. This adapter batches role/metric reductions and
keeps the original strict-past person statistics, priors, masks and schema.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np

from scripts import kion_people_dynamics as core


CORE_SHA256 = "6bf97bb98c8fdf0cb858db18066862d3a94cb107334bcb14c74e340f2943911c"


def _summaries(raw):
    """All eight original reductions along people, omitting only None values."""
    values = np.asarray(raw, np.float64)
    if values.ndim != 2:
        raise ValueError("Expected person by metric matrix")
    valid = ~np.isnan(values)
    counts = valid.sum(axis=0)
    result = np.zeros((len(core.AGGREGATES), values.shape[1]), np.float64)
    if not len(values):
        return result, counts
    # Metrics with equal availability can be packed into contiguous rows. This
    # preserves NumPy's 1-D reduction order (including tiny nonzero standard
    # deviations of constant floating values), while batching the dispatch.
    transposed, available = values.T, valid.T
    for count in np.unique(counts[counts > 0]):
        columns = np.flatnonzero(counts == count)
        selected = transposed[columns]
        packed = selected[available[columns]].reshape(len(columns), int(count))
        result[0, columns] = np.mean(packed, axis=1)
        result[1, columns] = np.max(packed, axis=1)
        result[2, columns] = np.median(packed, axis=1)
        result[3, columns] = np.std(packed, axis=1)
        result[4:7, columns] = np.quantile(packed, [.25, .75, .90], axis=1, method="linear")
        result[7, columns] = np.mean(np.sort(packed, axis=1)[:, -3:], axis=1)
    return result, counts


class FastPeopleDynamicsIndex(core.PeopleDynamicsIndex):
    """Drop-in index preserving all coordinates, with bounded original caches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prepare_layout()

    @classmethod
    def from_index(cls, index):
        if core._file_hash(Path(core.__file__)) != CORE_SHA256:
            raise ValueError("Frozen people core source hash mismatch")
        result = cls.__new__(cls)
        result.__dict__.update(index.__dict__)
        result._day_cache, result._feature_cache = OrderedDict(), OrderedDict()
        result._prepare_layout()
        return result

    @classmethod
    def load(cls, output_dir, *, verify_sources=True, expected_manifest_sha256=None,
             expected_mapping_sha256=None):
        if core._file_hash(Path(core.__file__)) != CORE_SHA256:
            raise ValueError("Frozen people core source hash mismatch")
        if (expected_manifest_sha256 is not None
                and core._file_hash(Path(output_dir) / "manifest.json") != expected_manifest_sha256):
            raise ValueError("Pinned people index manifest hash mismatch")
        result = super().load(output_dir, verify_sources=verify_sources)
        if (expected_mapping_sha256 is not None
                and core._hash(result.mapping) != expected_mapping_sha256):
            raise ValueError("Pinned people index mapping hash mismatch")
        return result

    def _prepare_layout(self):
        self._window_cache = OrderedDict()
        self._window_cache_limit = 4096
        self._role_layout = {}
        self._log_person = np.array([
            (metric.startswith("related_") and metric.endswith(("_pairs", "_distinct_users")))
            or metric.endswith(("_format_n_film", "_format_n_series"))
            for metric in core.METRICS])
        for role in core.ROLES:
            prefix = "people_" + role + "_"
            source, target, logs, masks = [], [], [], []
            for m, metric in enumerate(core.METRICS):
                for agg in core._aggregates(role, metric):
                    source.append((core.AGGREGATES.index(agg), m))
                    target.append(self._column[prefix + metric + "_" + agg])
                    logs.append(role == "directors" and metric in core.DIRECTOR_LOG_METRICS)
                masks.append([self._column[prefix + metric + "_" + name]
                    for name in ("available_count", "available_fraction", "missing")])
            self._role_layout[role] = {
                "source": tuple(np.array(v, np.int64) for v in zip(*source)),
                "target": np.array(target, np.int64), "log": np.array(logs, bool),
                "masks": np.asarray(masks, np.int64),
                "identity": np.array([self._column[prefix + "identity_" + str(i)]
                    for i in range(len(self.top_people[role]))], np.int64)}

    def _window(self, works, left, right):
        # Identical portfolios occur for many cast members. Reuse only the exact
        # work set and half-open interval; no audience subtraction approximation.
        key = (tuple(map(int, works)), int(left), int(right))
        if key not in self._window_cache:
            self._window_cache[key] = super()._window(works, left, right)
        self._window_cache.move_to_end(key)
        result = self._window_cache[key]
        while len(self._window_cache) > self._window_cache_limit:
            self._window_cache.popitem(last=False)
        return result

    def _context(self, work, cutoff, token):
        """Reuse date/year reference populations; preserve literal LOO means."""
        day_state = self._date_state(cutoff)
        cache = day_state.setdefault("fast_references", {})
        stats = day_state["stats"]
        state = dict(day_state)
        state["candidate_attributes"] = self.token_attributes[token]
        state["quality_references"], state["specialization_priors"] = {}, {}
        if "catalog" not in cache:
            observed = stats[:, core.USERS] > 0
            kinds = {}
            for kind, incidence in (("genre", self.genre_incidence), ("format", self.format_incidence)):
                available = observed & (incidence.sum(axis=1) > 0)
                kinds[kind] = (incidence[available].sum(axis=0), int(available.sum()), available)
            cache["catalog"] = kinds
        # Keep the original ordered, strided summation for the non-integer
        # completion total; subtraction could change floating standard deviations.
        ref = np.arange(len(self.work_groups))
        values = stats[ref[ref != work]]
        valid = values[:, core.VALID].sum()
        observed = values[values[:, core.USERS] > 0, core.USERS]
        state["quality_references"]["catalog"] = (
            float(values[:, core.SUM].sum() / valid) if valid else None,
            float(np.quantile(observed, .90, method="linear")) if len(observed) else None)
        state["specialization_priors"]["catalog"] = {}
        for kind, incidence in (("genre", self.genre_incidence), ("format", self.format_incidence)):
            total, count, available = cache["catalog"][kind]
            excluded = bool(available[work])
            n = count - excluded
            state["specialization_priors"]["catalog"][kind] = (
                (total - incidence[work] * excluded) / n if n else None)
        year = int(self.years[work])
        if year not in cache:
            ref = np.flatnonzero((self.years > 0) & (self.years < year))
            values = stats[ref]
            valid = values[:, core.VALID].sum()
            observed = values[values[:, core.USERS] > 0, core.USERS]
            quality = (float(values[:, core.SUM].sum() / valid) if valid else None,
                float(np.quantile(observed, .90, method="linear")) if len(observed) else None)
            profile_ref = ref[(stats[ref, core.USERS] > 0)
                & (self.years[ref] < date.fromordinal(cutoff).year)]
            priors = {}
            for kind, incidence in (("genre", self.genre_incidence), ("format", self.format_incidence)):
                available = profile_ref[incidence[profile_ref].sum(axis=1) > 0]
                priors[kind] = incidence[available].mean(axis=0) if len(available) else None
            cache[year] = quality, priors
        state["quality_references"]["prior_release"], state["specialization_priors"]["prior_release"] = cache[year]
        return state

    def _row(self, token, day):
        row = np.zeros(len(self.columns), np.float32)
        work = int(self.token_work[token])
        state = self._context(work, day, token)
        thresholds = self._thresholds(work, day, state)
        row[self._column["people_known_item"]] = 1
        row[self._column["people_candidate_year_missing"]] = self.years[work] == 0
        for role in core.ROLES:
            names, rel = self.work_groups[work][role], self.relations[role]
            people, portfolios = [], {}
            for name in names:
                person = rel["lookup"][name]
                left, right = rel["offsets"][person:person + 2]
                portfolio = tuple(map(int, rel["works"][left:right]))
                if portfolio not in portfolios:
                    portfolios[portfolio] = self._person(role, person, work, day, state, thresholds)
                people.append(portfolios[portfolio])
            prefix, layout = "people_" + role + "_", self._role_layout[role]
            row[self._column[prefix + "listed_count"]] = len(names)
            row[self._column[prefix + "metadata_missing"]] = not names
            row[self._column[prefix + "topn_coverage"]] = (
                sum(name in self.top_people[role] for name in names) / len(names) if names else 0)
            raw = np.array([[person[metric] for metric in core.METRICS] for person in people],
                np.float64).reshape(len(people), len(core.METRICS))
            raw[:, self._log_person] = np.log1p(raw[:, self._log_person])
            reduced, counts = _summaries(raw)
            aggregate_values = reduced[layout["source"]]
            aggregate_values[layout["log"]] = np.log1p(aggregate_values[layout["log"]])
            row[layout["target"]] = aggregate_values
            row[layout["masks"][:, 0]] = counts
            row[layout["masks"][:, 1]] = counts / len(names) if names else 0
            row[layout["masks"][:, 2]] = counts == 0
            if role == "actors":
                for tier in core.TIERS:
                    count = sum(person["tier"] == tier for person in people)
                    for field, value in (("count", count),
                            ("fraction", count / len(names) if names else 0), ("has", count > 0)):
                        row[self._column[prefix + "tier_" + tier + "_" + field]] = value
                for field, value in (("thresholds_missing", thresholds["q90"] is None),
                        ("q90", thresholds["q90"] or 0), ("q99", thresholds["q99"] or 0),
                        ("threshold_pool_count", thresholds["pool_count"])):
                    row[self._column["people_actors_" + field]] = value
            row[layout["identity"]] = [name in names for name in self.top_people[role]]
        return row

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        tokens = np.asarray(item_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("item_tokens must be nonnegative integer vector")
        n = len(tokens)
        def dates(values):
            values = np.asarray(values)
            if values.ndim == 0:
                return np.full(n, core._day(values.item()), np.int64)
            if values.shape != (n,):
                raise ValueError("Dates must be scalar or one per token")
            return np.array([core._day(value) for value in values], np.int64)
        cutoffs = np.minimum(dates(query_dates), self.fit_cutoff)
        if history_cutoffs is not None:
            cutoffs = np.minimum(cutoffs, dates(history_cutoffs))
        if user_ids is not None and np.asarray(user_ids).shape != (n,):
            raise ValueError("user_ids must have one value per token")
        output = np.zeros((n, len(self.columns)), np.float32)
        output[tokens >= self.vocab_size, self._column["people_unknown_item"]] = 1
        # Build batch positions once: the original rescanned all rows per key.
        positions = {}
        for position, (token, day) in enumerate(zip(tokens, cutoffs)):
            if 3 <= token < self.vocab_size:
                positions.setdefault((int(token), int(day)), []).append(position)
        for key in sorted(positions, key=lambda pair: (pair[1], pair[0])):
            if key not in self._feature_cache:
                self._feature_cache[key] = self._row(*key)
            self._feature_cache.move_to_end(key)
            output[positions[key]] = self._feature_cache[key]
            while len(self._feature_cache) > self.parameters["cache_rows"]:
                self._feature_cache.popitem(last=False)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite people feature values")
        return output


PeopleDynamicsIndex = FastPeopleDynamicsIndex


def build(*args, **kwargs):
    return FastPeopleDynamicsIndex.from_index(core.build(*args, **kwargs))


build_index = build
