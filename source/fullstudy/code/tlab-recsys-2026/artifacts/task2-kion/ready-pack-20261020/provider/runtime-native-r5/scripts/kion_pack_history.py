"""Strict-date H63 features for the local KION package (CPU, bounded batches).

KION is a last-user/item-pair snapshot. Earlier last_watch_dt is an explicit
availability assumption for its aggregated outcome, not proof of online receipt.
Metadata is retrospective. No target outcome, reward or total_dur is consumed.
Callers must pass the query origin (Aug 9 for frozen validation), never infer it
from a future response date. Outputs are float32; iter_features avoids an N x D
allocation. Unknown tokens retain raw dates/outcomes but never regain metadata.
"""
from __future__ import annotations

from array import array
import csv
from datetime import date
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

import numpy as np


CONTRACT = "kion-local-history-v1"
UNKNOWN = "__UNKNOWN__"
MAX_ORDINAL = date.max.toordinal() + 1
EXCLUDED_GENRES = frozenset((UNKNOWN, "популярное", "рекомендуем"))
AGE_RATINGS = ("0.0", "6.0", "12.0", "16.0", "18.0", "21.0")
THRESHOLDS = ("le10", "ge20", "ge50", "ge80", "ge95")
BASELINE_ALPHA = 1.0
BASELINE_BETA = 1.0
BASELINE_C0 = 0.5
RELATIVE_LAMBDA = 1.0


def _day(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        result = int(value)
        if 1 <= result < MAX_ORDINAL:
            return result
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid ISO query/history date: {value!r}") from error
        if parsed.isoformat() == value:
            return parsed.toordinal()
    raise ValueError(f"Invalid ISO/ordinal query/history date: {value!r}")


def _ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or np.any(values < 0):
        raise ValueError(f"{name} must be a one-dimensional nonnegative integer array")
    if values.size and int(values.max()) > np.iinfo(np.int64).max // MAX_ORDINAL - 1:
        raise ValueError(f"{name} too large for the exact user/day index")
    return values.astype(np.int64, copy=False)


def _mapping_int(value, name):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        result = int(value)
    elif isinstance(value, str):
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError(f"Invalid {name}: {value!r}") from error
        if str(result) != value:
            raise ValueError(f"Noncanonical {name}: {value!r}")
    else:
        raise ValueError(f"Invalid {name}: {value!r}")
    if result < 0 or result > np.iinfo(np.int64).max // MAX_ORDINAL - 1:
        raise ValueError(f"Out-of-range {name}: {value!r}")
    return result


def _csv_rows(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(set(header)) != len(header) or not set(fields).issubset(header):
            raise ValueError(f"Missing/duplicate CSV columns: {path}")
        columns = [header.index(field) for field in fields]
        for line, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row {line}: {path}")
            yield tuple(row[index] for index in columns)


def parse_outcome(text):
    """Return normalized value, status (0 valid/1 missing/2 invalid), exact bits."""
    raw = text.strip()
    if raw.lower() in ("", "nan", "none", "null"):
        return 0.0, 1, 0
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return 0.0, 2, 0
    if not value.is_finite() or not 0 <= value <= 100:
        return 0.0, 2, 0
    bits = int(value <= 10)
    for index, threshold in enumerate((20, 50, 80, 95), 1):
        bits |= int(value >= threshold) << index
    return float(value / 100), 0, bits


def _year(text):
    try:
        number = Decimal(text.strip())
    except InvalidOperation:
        return None
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        return None
    return int(number)


def _tags(text, excluded=frozenset()):
    return set(part.strip() for part in text.split(",") if part.strip()) - excluded


def _label_table(token_labels, categories, vocab_size):
    lookup = {label: index for index, label in enumerate(categories)}
    width = max((len(labels & lookup.keys()) for labels in token_labels.values()), default=0)
    table = np.full((vocab_size, max(width, 1)), -1, dtype=np.int16)
    known = np.zeros(vocab_size, dtype=bool)
    unseen = np.zeros(vocab_size, dtype=bool)
    for token, labels in token_labels.items():
        indices = sorted(lookup[label] for label in labels if label in lookup)
        table[token, :len(indices)] = indices
        known[token] = bool(indices)
        unseen[token] = bool(labels - lookup.keys())
    return table, known, unseen


class HistoryIndex:
    """One compact raw index; feature batches and query origins are independent.

    ``from_csv(..., include_relative=True)`` adds nested, other-user strict-past
    residual profiles. Their hyperparameters are fixed constants above, not fit.
    ``fit_cutoff`` independently freezes their priors (RM uses Aug 2).
    ``schema_item_ids`` optionally restricts the metadata vocabulary/bin fit for
    an RM role. Token visibility still comes only from item_id_to_token.
    """

    @classmethod
    def from_csv(cls, interactions, items, item_id_to_token, *, fit_cutoff="2021-08-09",
                 include_relative=False, schema_item_ids=None):
        source_paths = {"interactions": Path(interactions), "items": Path(items), "history_builder": Path(__file__)}
        source_pins = {}
        for name, path in source_paths.items():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            source_pins[name] = {"path": path.resolve().as_posix(), "sha256": digest}
        users, item_ids, days = array("q"), array("q"), array("i")
        outcomes, statuses, bits = array("f"), array("B"), array("B")
        date_cache = {}
        for raw_user, raw_item, raw_date, raw_pct in _csv_rows(
                interactions, ("user_id", "item_id", "last_watch_dt", "watched_pct")):
            try:
                uid, iid = int(raw_user), int(raw_item)
            except ValueError as error:
                raise ValueError("Invalid interaction ID") from error
            if str(uid) != raw_user or str(iid) != raw_item or min(uid, iid) < 0:
                raise ValueError("Noncanonical interaction ID")
            users.append(uid)
            item_ids.append(iid)
            if raw_date not in date_cache:
                date_cache[raw_date] = _day(raw_date)
            days.append(date_cache[raw_date])
            outcome, status, threshold_bits = parse_outcome(raw_pct)
            outcomes.append(outcome)
            statuses.append(status)
            bits.append(threshold_bits)
        result = cls(np.asarray(users, np.int64), np.asarray(item_ids, np.int64),
                   np.asarray(days, np.int32), np.asarray(outcomes, np.float32),
                   np.asarray(statuses, np.uint8), np.asarray(bits, np.uint8),
                   items, item_id_to_token, fit_cutoff=fit_cutoff,
                   include_relative=include_relative, schema_item_ids=schema_item_ids)
        for name, path in source_paths.items():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != source_pins[name]["sha256"]:
                raise ValueError(f"Source changed while building history: {path}")
        result.source_pins = source_pins
        result.provenance["source_pins"] = result.source_pins
        return result

    def __init__(self, users, item_ids, days, outcomes, statuses, bits, items,
                 item_id_to_token, *, fit_cutoff="2021-08-09", include_relative=False,
                 schema_item_ids=None):
        users, item_ids = _ids(users, "users"), _ids(item_ids, "items")
        arrays = (users, item_ids, np.asarray(days), np.asarray(outcomes), np.asarray(statuses), np.asarray(bits))
        if any(value.ndim != 1 or len(value) != len(users) for value in arrays):
            raise ValueError("Raw interaction arrays must align")
        if arrays[2].dtype.kind not in "iu" or np.any((arrays[2] < 1) | (arrays[2] >= MAX_ORDINAL)):
            raise ValueError("Invalid raw ordinal date")
        if (np.any(~np.isfinite(arrays[3])) or np.any((arrays[3] < 0) | (arrays[3] > 1))
                or not np.isin(statuses, (0, 1, 2)).all()
                or np.any((arrays[5] < 0) | (arrays[5] > 31))):
            raise ValueError("Invalid parsed outcome/status/threshold arrays")
        pair_order = np.lexsort((item_ids, users))
        if len(users) > 1 and np.any((users[pair_order[1:]] == users[pair_order[:-1]])
                                    & (item_ids[pair_order[1:]] == item_ids[pair_order[:-1]])):
            raise ValueError("Duplicate raw user-item pair")
        del pair_order
        order = np.lexsort((item_ids, arrays[2], users))
        self.users, self.items, self.days, self.outcomes, self.statuses, self.bits = (
            np.asarray(value[order], dtype=dtype) for value, dtype in zip(
                arrays, (np.int64, np.int64, np.int32, np.float32, np.uint8, np.uint8)))
        self.fit_cutoff = _day(fit_cutoff)
        self.include_relative = bool(include_relative)
        mapping = {_mapping_int(iid, "mapping raw item ID"): _mapping_int(token, "mapping token")
                   for iid, token in item_id_to_token.items()}
        if (len(mapping) != len(item_id_to_token) or any(token < 3 for token in mapping.values())
                or set(mapping.values()) != set(range(3, len(mapping) + 3))):
            raise ValueError("item_id_to_token must be a contiguous unique real-item mapping")
        self.mapping = mapping
        self.vocab_size = len(mapping) + 3
        self.tokens = np.fromiter((mapping.get(int(iid), 2) for iid in self.items),
                                  dtype=np.int32, count=len(self.items))
        self.keys = self.users * MAX_ORDINAL + self.days
        self.unique_users, self.user_starts = np.unique(self.users, return_index=True)
        self.user_stops = np.r_[self.user_starts[1:], len(self.users)]
        fields = ("item_id", "genres", "content_type", "release_year", "countries", "age_rating")
        if isinstance(items, (str, Path)):
            cards = {}
            for iid, *values in _csv_rows(items, fields):
                raw_id = int(iid)
                if str(raw_id) != iid or raw_id < 0 or raw_id in cards:
                    raise ValueError("Invalid/duplicate metadata item ID")
                cards[raw_id] = dict(zip(fields[1:], values))
        else:
            cards = {_mapping_int(key, "metadata raw item ID"): dict(value) for key, value in items.items()}
            if len(cards) != len(items):
                raise ValueError("Duplicate normalized metadata item ID")
        fit_ids = (set(map(int, np.unique(self.items[self.days < self.fit_cutoff]))) & set(mapping)
                   if schema_item_ids is None else {_mapping_int(value, "schema-fit raw item ID") for value in schema_item_ids})
        if not fit_ids.issubset(mapping):
            raise ValueError("Schema fit IDs must be visible mapped items")
        self.categories = {
            "genres": sorted(set().union(*(_tags(cards.get(iid, {}).get("genres", ""), EXCLUDED_GENRES)
                                           for iid in fit_ids))) if fit_ids else [],
            "countries": sorted(set().union(*(_tags(cards.get(iid, {}).get("countries", ""), {UNKNOWN})
                                              for iid in fit_ids))) if fit_ids else [],
            "content_type": ["film", "series"], "age_rating": list(AGE_RATINGS),
        }
        years = [_year(cards.get(iid, {}).get("release_year", "")) for iid in fit_ids]
        years = [year for year in years if year is not None]
        if not years:
            raise ValueError("No valid release years in metadata schema-fit IDs")
        if max(years) // 10 - min(years) // 10 > 10000:
            raise ValueError("Metadata decade schema exceeds the bounded feature budget")
        self.categories["decade"] = list(range(min(years) // 10 * 10, max(years) // 10 * 10 + 1, 10))
        self.labels, self.known, self.unseen = {}, {}, {}
        self.year_valid = np.zeros(self.vocab_size, bool)
        self.year_outside = np.zeros(self.vocab_size, bool)
        for field, categories in self.categories.items():
            token_labels = {}
            for iid, token in mapping.items():
                text = cards.get(iid, {}).get(field, "")
                if field in ("genres", "countries"):
                    labels = _tags(text, EXCLUDED_GENRES if field == "genres" else {UNKNOWN})
                    # Sentinel with known labels is retained as partial unknown.
                    if field == "genres" and UNKNOWN in _tags(text):
                        labels.add(UNKNOWN)
                elif field == "decade":
                    year = _year(cards.get(iid, {}).get("release_year", ""))
                    self.year_valid[token] = year is not None
                    labels = {year // 10 * 10} if year is not None else set()
                    self.year_outside[token] = bool(labels - set(categories))
                else:
                    labels = {text.strip()} if text.strip() else set()
                token_labels[token] = labels
            self.labels[field], self.known[field], self.unseen[field] = _label_table(
                token_labels, categories, self.vocab_size)
        self.types = np.zeros(self.vocab_size, np.int8) - 1
        for iid, token in mapping.items():
            content_type = cards.get(iid, {}).get("content_type", "").strip()
            if content_type in ("film", "series"):
                self.types[token] = int(content_type == "series")
        self.columns, self.groups = [], {}
        self._layout()
        self.relative_support = None
        self.residuals = self._relative_residuals() if include_relative else None
        self.provenance = {
            "contract": CONTRACT, "history_cap": 63, "sort": "user_id,date,raw_item_id",
            "history_filter": "date < history_cutoff <= query_origin",
            "metadata_visibility": "mapped token >=3 only; unknown raw metadata never restored",
            "outcome_availability": "assumed at earlier last_watch_dt; aggregated snapshot receipt unknown",
            "metadata_availability": "retrospective snapshot; online-as-of unverified",
            "fit_cutoff_exclusive": date.fromordinal(self.fit_cutoff).isoformat(),
            "schema_fit_item_count": len(fit_ids), "categories": self.categories,
            "outcome_parser": "Decimal finite [0,100], missing empty/nan/none/null; exact threshold bits",
            "relative_baseline": {"enabled": bool(include_relative), "alpha": BASELINE_ALPHA,
                                  "beta": BASELINE_BETA, "c0": BASELINE_C0,
                                  "centered_lambda": RELATIVE_LAMBDA,
                                  "clock": "other users d < min(history_row_day,fit_cutoff)"},
            "raw_rows": len(self.users), "columns": self.columns, "groups": self.groups,
        }
        self.source_pins = {}

    def _add(self, group, suffixes):
        start = len(self.columns)
        self.columns.extend(f"{group}.{value}" for value in suffixes)
        self.groups[group] = {"start": start, "stop": len(self.columns)}

    def _layout(self):
        for field in ("genres", "countries"):
            self._add("profile_" + field, [*map(str, self.categories[field]), "n", "n_known", "coverage",
                                          "unknown_share", "empty_history", "empty_profile", "has_unseen"])
        self._add("profile_type", ["series_fraction", "n", "n_known", "empty_history", "empty_profile"])
        self._add("profile_decade", [*map(str, self.categories["decade"]), "outside_share", "n", "n_valid",
                                     "empty_history", "empty_profile"])
        self._add("profile_age_rating", [*AGE_RATINGS, "n", "n_known", "empty_history", "empty_profile"])
        self._add("temporal", ["recency_log1p_days", "empty_history", "cards_1d", "dates_1d",
                               "cards_7d", "dates_7d", "cards_30d", "dates_30d",
                               "gap_median_log1p_days", "n_gaps", "gap_missing", "gap_cv_log1p", "cv_missing"])
        self._add("completion", ["mean", *THRESHOLDS, "n_valid", "n_missing", "n_invalid", "empty"])
        for group in ("completion_film", "completion_series", "completion_7d", "completion_30d"):
            self._add(group, ["mean", "ge80", "le10", "n_valid", "empty"])
        if self.include_relative:
            self._add("relative", ["mean", "n_valid", "empty"])
            for field in ("genres", "content_type", "countries", "decade"):
                self._add("relative_" + field,
                          [f"{label}.{stat}" for label in self.categories[field] for stat in ("z", "n_over63", "empty")])

    def _relative_residuals(self):
        """Nested prefixes and whole-user exclusion, computed once per role cap."""
        count = len(self.users)
        if not count:
            self.relative_support = {name: np.zeros(0, np.int32) for name in ("item", "type", "global")}
            return np.zeros(0, np.float32)
        unique_items, item_index = np.unique(self.items, return_inverse=True)
        # The raw dataset covers 180 days. Avoid unbounded dense calendar grids.
        observed_days = np.unique(self.days[self.days < self.fit_cutoff])
        day_slots = np.searchsorted(observed_days, self.days)
        boundary_slots = np.searchsorted(observed_days, np.minimum(self.days, self.fit_cutoff))
        shape = (len(unique_items), len(observed_days) + 1)
        if shape[0] * shape[1] > 50_000_000:
            raise ValueError("Relative item/calendar prefix grid exceeds the bounded CPU memory budget")
        n = np.zeros(shape, np.int32)
        sums = np.zeros(shape, np.float64)
        valid = (self.statuses == 0) & (self.days < self.fit_cutoff)
        np.add.at(n, (item_index[valid], day_slots[valid] + 1), 1)
        np.add.at(sums, (item_index[valid], day_slots[valid] + 1), self.outcomes[valid])
        np.cumsum(n, axis=1, out=n)
        np.cumsum(sums, axis=1, out=sums)
        global_n, global_sum = n.sum(axis=0, dtype=np.int64), sums.sum(axis=0)
        item_n, item_sum = n[item_index, boundary_slots], sums[item_index, boundary_slots]
        del n, sums
        prior_boundary = self.users * MAX_ORDINAL + np.minimum(self.days, self.fit_cutoff)
        own_ends = np.searchsorted(self.keys, prior_boundary, side="left")
        starts = self.user_starts[np.searchsorted(self.unique_users, self.users)]

        def own_prefix(mask):
            prefix_n = np.r_[0, np.cumsum(mask, dtype=np.int64)]
            prefix_sum = np.r_[0., np.cumsum(np.where(mask, self.outcomes, 0.), dtype=np.float64)]
            return prefix_n[own_ends] - prefix_n[starts], prefix_sum[own_ends] - prefix_sum[starts]

        own_n, own_sum = own_prefix(self.statuses == 0)
        other_n = global_n[boundary_slots] - own_n
        other_sum = global_sum[boundary_slots] - own_sum
        self.relative_support = {"item": item_n.astype(np.int32), "global": other_n.astype(np.int32),
                                 "type": np.zeros(count, np.int32)}
        global_mean = np.divide(other_sum, other_n, out=np.full(count, BASELINE_C0), where=other_n > 0)
        prior = global_mean.copy()
        row_types = self.types[self.tokens]
        for content_type in (0, 1):
            type_mask = (self.statuses == 0) & (row_types == content_type)
            allowed = type_mask & (self.days < self.fit_cutoff)
            type_n = np.bincount(day_slots[allowed] + 1, minlength=len(observed_days) + 1).cumsum()
            type_sum = np.bincount(day_slots[allowed] + 1, weights=self.outcomes[allowed],
                                   minlength=len(observed_days) + 1).cumsum()
            own_n, own_sum = own_prefix(type_mask)
            other_n = type_n[boundary_slots] - own_n
            other_sum = type_sum[boundary_slots] - own_sum
            chosen = row_types == content_type
            self.relative_support["type"][chosen] = other_n[chosen]
            prior[chosen] = ((other_sum + BASELINE_BETA * global_mean) / (other_n + BASELINE_BETA))[chosen]
        # A unique raw user-item pair has no other own row in its item prefix.
        baseline = (item_sum + BASELINE_ALPHA * prior) / (item_n + BASELINE_ALPHA)
        residuals = np.where(self.statuses == 0, self.outcomes - baseline, 0.)
        if not np.isfinite(residuals).all() or np.any(np.abs(residuals) > 1.000001):
            raise ValueError("Invalid strict-past relative baseline")
        return residuals.astype(np.float32)

    def _history(self, user_ids, query_dates, history_cutoffs=None, x=None, lengths=None):
        user_ids = _ids(user_ids, "query user IDs")
        query_dates = np.asarray([_day(value) for value in query_dates], np.int64)
        cutoffs = query_dates if history_cutoffs is None else np.asarray([_day(value) for value in history_cutoffs], np.int64)
        if query_dates.shape != user_ids.shape or cutoffs.shape != user_ids.shape or np.any(cutoffs > query_dates):
            raise ValueError("Query origins/cutoffs must align and history_cutoff <= origin")
        positions = np.searchsorted(self.unique_users, user_ids)
        found = positions < len(self.unique_users)
        if len(self.unique_users):
            found &= self.unique_users[np.minimum(positions, len(self.unique_users) - 1)] == user_ids
        starts = np.zeros(len(user_ids), np.int64)
        if np.any(found):
            starts[found] = self.user_starts[positions[found]]
        ends = np.searchsorted(self.keys, user_ids * MAX_ORDINAL + cutoffs, side="left")
        ends[~found] = starts[~found]
        begins = np.maximum(starts, ends - 63)
        n = ends - begins
        indices = begins[:, None] + np.arange(63)[None, :]
        mask = indices < ends[:, None]
        indices[~mask] = 0
        if len(self.users):
            tokens = np.where(mask, self.tokens[indices], 0)
        else:
            tokens = np.zeros_like(indices, np.int32)
        if x is not None or lengths is not None:
            if x is None or lengths is None:
                raise ValueError("Both x and lengths are required for H63 alignment")
            x, lengths = np.asarray(x), np.asarray(lengths)
            expected = np.column_stack((np.ones(len(n), np.int32), tokens))
            if (x.shape != expected.shape or x.dtype.kind not in "iu" or lengths.shape != n.shape
                    or lengths.dtype.kind not in "iu" or not np.array_equal(lengths, n + 1)
                    or not np.array_equal(x, expected)):
                raise ValueError("Prepared x/lengths differ from exact strict-date raw H63")
        return indices, mask, tokens, n, query_dates

    def features(self, user_ids, query_dates, *, x=None, lengths=None, history_cutoffs=None):
        """Return float32 [batch,D]; target keys/outcomes are deliberately absent."""
        idx, mask, tokens, n, origins = self._history(user_ids, query_dates, history_cutoffs, x, lengths)
        batch = len(n)
        out = np.zeros((batch, len(self.columns)), np.float32)
        if not batch:
            return out
        empty = n == 0

        def put(group, values):
            sl = self.groups[group]
            values = np.asarray(values)
            if values.shape != (batch, sl["stop"] - sl["start"]):
                raise AssertionError(f"Internal feature layout mismatch: {group}")
            out[:, sl["start"]:sl["stop"]] = values

        def counts(field, row_weights=None):
            labels = self.labels[field][tokens]
            valid = (labels >= 0) & mask[:, :, None]
            k = len(self.categories[field])
            if not k:
                return np.zeros((batch, 0), np.float64)
            bins = np.broadcast_to(np.arange(batch)[:, None, None] * k, labels.shape) + labels
            weights = None if row_weights is None else np.broadcast_to(row_weights[:, :, None], labels.shape)[valid]
            # NumPy returns int64 for an empty weighted bincount; normalize here
            # so cold/all-UNK residual batches retain a floating divide buffer.
            return np.bincount(bins[valid], weights=weights, minlength=batch * k).reshape(batch, k).astype(np.float64, copy=False)

        for field in ("genres", "countries"):
            known_n = np.sum(self.known[field][tokens] & mask, axis=1)
            fractions = np.divide(counts(field), known_n[:, None], out=np.zeros((batch, len(self.categories[field]))),
                                  where=known_n[:, None] > 0)
            coverage = np.divide(known_n, n, out=np.zeros(batch), where=n > 0)
            unknown = np.divide(n - known_n, n, out=np.zeros(batch), where=n > 0)
            put("profile_" + field, np.column_stack((fractions, n, known_n, coverage, unknown, empty, known_n == 0,
                                                    np.any(self.unseen[field][tokens] & mask, axis=1))))
        type_counts = counts("content_type")
        type_n = type_counts.sum(axis=1)
        series = np.divide(type_counts[:, 1], type_n, out=np.zeros(batch), where=type_n > 0)
        put("profile_type", np.column_stack((series, n, type_n, empty, type_n == 0)))
        year_n = np.sum(self.year_valid[tokens] & mask, axis=1)
        year_counts = counts("decade")
        year_fractions = np.divide(year_counts, year_n[:, None], out=np.zeros_like(year_counts, dtype=float), where=year_n[:, None] > 0)
        outside = np.divide(np.sum(self.year_outside[tokens] & mask, axis=1), year_n,
                            out=np.zeros(batch), where=year_n > 0)
        put("profile_decade", np.column_stack((year_fractions, outside, n, year_n, empty, year_n == 0)))
        age_counts = counts("age_rating")
        age_n = age_counts.sum(axis=1)
        age_fractions = np.divide(age_counts, age_n[:, None], out=np.zeros_like(age_counts, dtype=float), where=age_n[:, None] > 0)
        put("profile_age_rating", np.column_stack((age_fractions, n, age_n, empty, age_n == 0)))
        dates = np.where(mask, self.days[idx], 0) if len(self.days) else np.zeros_like(idx)
        distinct = mask & np.column_stack((np.ones(batch, bool), dates[:, 1:] != dates[:, :-1]))
        last = np.max(dates, axis=1)
        recency = np.where(empty, 0., np.log1p(origins - last))
        temporal = [recency, empty]
        windows = {}
        for window in (1, 7, 30):
            inside = mask & (dates >= origins[:, None] - window)
            windows[window] = inside
            temporal.extend((inside.sum(axis=1), (inside & distinct).sum(axis=1)))
        gaps = dates[:, 1:] - dates[:, :-1]
        gap_mask = distinct[:, 1:] & mask[:, :-1]
        gap_n = gap_mask.sum(axis=1)
        sorted_gaps = np.sort(np.where(gap_mask, gaps, np.inf), axis=1)
        median = (sorted_gaps[np.arange(batch), np.maximum(gap_n - 1, 0) // 2]
                  + sorted_gaps[np.arange(batch), gap_n // 2]) / 2
        median = np.where(gap_n > 0, median, 0.)
        gap_sum = np.where(gap_mask, gaps, 0).sum(axis=1)
        gap_mean = np.divide(gap_sum, gap_n, out=np.zeros(batch), where=gap_n > 0)
        gap_var = np.divide(np.where(gap_mask, (gaps - gap_mean[:, None]) ** 2, 0).sum(axis=1), gap_n,
                            out=np.zeros(batch), where=gap_n > 0)
        cv = np.divide(np.sqrt(gap_var), gap_mean, out=np.zeros(batch), where=gap_n >= 3)
        temporal.extend((np.log1p(median), gap_n, gap_n == 0, np.log1p(cv), gap_n < 3))
        put("temporal", np.column_stack(temporal))
        values = self.outcomes[idx] if len(self.outcomes) else np.zeros_like(idx, float)
        status = self.statuses[idx] if len(self.statuses) else np.ones_like(idx)
        flags = self.bits[idx] if len(self.bits) else np.zeros_like(idx)
        valid = mask & (status == 0)

        def completion(chosen, full=False):
            selected = chosen & valid
            support = selected.sum(axis=1)
            mean = np.divide(np.where(selected, values, 0).sum(axis=1, dtype=np.float64), support,
                             out=np.zeros(batch), where=support > 0)
            bit_positions = range(5) if full else (3, 0)
            result = [mean]
            for bit in bit_positions:
                numerator = (((flags & (1 << bit)) != 0) & selected).sum(axis=1)
                result.append(np.divide(numerator, support, out=np.zeros(batch), where=support > 0))
            result.append(support)
            if full:
                result.extend(((chosen & (status == 1)).sum(axis=1), (chosen & (status == 2)).sum(axis=1)))
            result.append(support == 0)
            return np.column_stack(result)

        put("completion", completion(mask, True))
        for name, chosen in (("film", self.types[tokens] == 0), ("series", self.types[tokens] == 1),
                             ("7d", windows[7]), ("30d", windows[30])):
            put("completion_" + name, completion(chosen & mask))
        if self.include_relative:
            residual = self.residuals[idx] if len(self.residuals) else np.zeros_like(idx, float)
            support = valid.sum(axis=1)
            overall = np.divide(np.where(valid, residual, 0).sum(axis=1, dtype=np.float64), support,
                                out=np.zeros(batch), where=support > 0)
            put("relative", np.column_stack((overall, support, support == 0)))
            for field in ("genres", "content_type", "countries", "decade"):
                group_n = counts(field, valid.astype(float))
                group_sum = counts(field, np.where(valid, residual, 0.))
                group_mean = np.divide(group_sum, group_n, out=np.zeros_like(group_sum), where=group_n > 0)
                z = np.where(group_n > 0, group_n / (group_n + RELATIVE_LAMBDA) * (group_mean - overall[:, None]), 0.)
                put("relative_" + field, np.stack((z, group_n / 63., group_n == 0), axis=2).reshape(batch, -1))
        if not np.isfinite(out).all():
            raise ValueError("Nonfinite H63 feature output")
        return out

    def iter_features(self, user_ids, query_dates, *, batch_size=1024, x=None, lengths=None, history_cutoffs=None):
        """Yield (start,stop,float32 block); write to a caller-owned memmap/file."""
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        user_ids = _ids(user_ids, "query user IDs")
        size = len(user_ids)
        if len(query_dates) != size or (history_cutoffs is not None and len(history_cutoffs) != size):
            raise ValueError("All query dates/cutoffs must align before batching")
        if (x is None) != (lengths is None):
            raise ValueError("Both x and lengths are required for H63 alignment")
        if x is not None and (np.shape(x) != (size, 64) or np.shape(lengths) != (size,)):
            raise ValueError("All x/lengths rows must align before batching")
        for start in range(0, len(user_ids), batch_size):
            stop = min(len(user_ids), start + batch_size)
            yield start, stop, self.features(user_ids[start:stop], query_dates[start:stop],
                x=None if x is None else x[start:stop], lengths=None if lengths is None else lengths[start:stop],
                history_cutoffs=None if history_cutoffs is None else history_cutoffs[start:stop])

    def history_max_dates(self, user_ids, query_dates, *, history_cutoffs=None):
        """Maximum retained ordinal date per row; zero explicitly denotes cold."""
        idx, mask, _, _, _ = self._history(user_ids, query_dates, history_cutoffs)
        if not len(self.days):
            return np.zeros(len(user_ids), np.int32)
        return np.where(mask, self.days[idx], 0).max(axis=1).astype(np.int32)

    def historical_sidecar(self, user_ids, query_dates, *, history_cutoffs=None):
        """Bounded diagnostic H63 raw join; padding is distinguished by mask.

        Outcomes here belong exclusively to past retained rows. This diagnostic
        surface must not be concatenated as target/candidate input features.
        """
        idx, mask, tokens, _, _ = self._history(user_ids, query_dates, history_cutoffs)
        result = {"mask": mask, "tokens": tokens}
        for name, values in (("raw_item_id", self.items), ("date", self.days),
                             ("outcome", self.outcomes), ("outcome_status", self.statuses)):
            result[name] = np.where(mask, values[idx], 0) if len(values) else np.zeros_like(idx)
        if self.include_relative:
            result["residual"] = np.where(mask, self.residuals[idx], 0) if len(self.residuals) else np.zeros_like(idx, float)
            for name, values in self.relative_support.items():
                result["baseline_n_" + name] = np.where(mask, values[idx], 0) if len(values) else np.zeros_like(idx)
            result["baseline_type_unknown"] = mask & (self.types[tokens] < 0)
            result["baseline_prior_only"] = mask & (result["baseline_n_item"] == 0)
        return result


def build_history_index(raw_dir, metadata, *, fit_cutoff="2021-08-09", include_relative=False,
                        schema_item_ids=None):
    """Convenience facade; metadata can be a mapping or loaded metadata.json dict."""
    mapping = metadata.get("item_id_to_token", metadata)
    raw_dir = Path(raw_dir)
    return HistoryIndex.from_csv(raw_dir / "interactions.csv", raw_dir / "items.csv", mapping,
                                 fit_cutoff=fit_cutoff, include_relative=include_relative,
                                 schema_item_ids=schema_item_ids)
