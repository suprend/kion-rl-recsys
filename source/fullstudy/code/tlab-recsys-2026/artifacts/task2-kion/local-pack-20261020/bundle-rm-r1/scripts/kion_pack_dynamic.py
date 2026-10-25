"""Compact strict-date KION item and related-interest features (NumPy/CPU).

The source contains one retained aggregate per user/item, not an event log.
Historical prefixes cannot recover earlier events replaced by a later last date.
All outcome priors use the same strict prefix as the query. Metadata/curated
series are retrospective; names are normalized strings, not resolved people.
Related counts exclude the candidate card; no unverified edition merge is made.
"""
from __future__ import annotations

from array import array
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


CONTRACT = "kion-local-pack-dynamic-v1"
DAY_BASE = date.max.toordinal() + 1
WINDOWS = (1, 7, 30)
ROLES = ("actors", "directors", "studios", "series")
AGGREGATES = ("mean", "max", "median", "top3_mean")
THRESHOLDS = ("le10", "ge20", "ge50", "ge80", "ge95")
ALPHA = 10.0
WEEKDAY_BETA = 7.0
POP_ALPHA = 1.0
TREND_LAMBDA = 10.0
TREND_CLIP = 3.0
SOURCE_START = date(2021, 3, 13).toordinal()
# pair, observed, sum(y), missing, invalid, threshold counts, active days, weekdays
PAIR, OBS, SUM, MISSING, INVALID = range(5)
ACTIVE, WEEKDAY = 10, 11
N_STATS = 18


def _day(value):
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if 1 <= int(value) < DAY_BASE:
            return int(value)
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"Invalid ISO/ordinal date: {value!r}") from error
        if parsed.isoformat() == value:
            return parsed.toordinal()
    raise ValueError(f"Invalid ISO/ordinal date: {value!r}")


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _tags(value):
    return tuple(sorted({" ".join(unicodedata.normalize("NFKC", part).casefold().split())
                         for part in str(value or "").split(",") if part.strip()}))


def _read_csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)) or not set(required).issubset(header):
            raise ValueError(f"Missing/duplicate CSV headers: {path}")
        positions = [header.index(key) for key in required]
        for line, values in enumerate(reader, 2):
            if len(values) != len(header):
                raise ValueError(f"Malformed CSV row {line}: {path}")
            yield tuple(values[position] for position in positions)


def _metadata(value):
    meta = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    mapping, size = meta.get("item_id_to_token"), meta.get("vocab_size")
    if meta.get("special_tokens", {"PAD": 0, "BOS": 1, "UNK": 2}) != {"PAD": 0, "BOS": 1, "UNK": 2}:
        raise ValueError("Unexpected special token mapping")
    if (not isinstance(mapping, dict) or not isinstance(size, int) or isinstance(size, bool)
            or size < 3 or any(not isinstance(key, str) or not key or not isinstance(token, int)
                              or isinstance(token, bool) for key, token in mapping.items())
            or sorted(mapping.values()) != list(range(3, size))):
        raise ValueError("Expected unique contiguous item tokens after PAD/BOS/UNK")
    return meta, mapping


class _Prefix:
    """Sparse entity/day cumulative table; no date x entity dense allocation."""

    def __init__(self, keys, values, size):
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        order = np.argsort(keys, kind="stable")
        keys, values = keys[order], values[order]
        if len(keys):
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
            values = np.add.reduceat(values, starts, axis=0)
            keys = keys[starts]
        self.keys = keys
        self.cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
        self.base = np.searchsorted(keys, np.arange(size, dtype=np.int64) * DAY_BASE)
        self.width = values.shape[1]

    def before(self, entities, dates):
        entities = np.asarray(entities, dtype=np.int64)
        dates = np.clip(np.asarray(dates, dtype=np.int64), 1, DAY_BASE - 1)
        positions = np.searchsorted(self.keys, entities * DAY_BASE + dates, side="left")
        return self.cumulative[positions] - self.cumulative[self.base[entities]]

    def window(self, entities, dates, days=None):
        result = self.before(entities, dates)
        return result if days is None else result - self.before(entities, dates - days)

    @property
    def nbytes(self):
        return self.keys.nbytes + self.cumulative.nbytes + self.base.nbytes


def _parse_outcome(raw):
    text = raw.strip()
    if text.casefold() in ("", "nan", "none", "null"):
        return 0.0, 1
    try:
        value = float(text)
    except ValueError:
        return 0.0, 2
    if not np.isfinite(value) or not 0 <= value <= 100:
        return 0.0, 2
    return value / 100, 0


def _interaction_table(path, mapping, size, fit_cutoff):
    tokens, days, outcomes, states, all_pairs = (array("i"), array("i"), array("d"), array("b"), array("q"))
    date_cache = {}
    source_rows = excluded = 0
    for user_text, item_id, day_text, outcome_text in _read_csv(path,
            ("user_id", "item_id", "last_watch_dt", "watched_pct")):
        source_rows += 1
        if day_text not in date_cache:
            date_cache[day_text] = _day(day_text)
        day = date_cache[day_text]
        token = mapping.get(item_id)
        if token is None:
            excluded += 1
            continue
        try:
            user = int(user_text)
        except ValueError as error:
            raise ValueError("Invalid user_id") from error
        if user < 0 or user > (np.iinfo(np.int64).max - token) // max(size, 1):
            raise ValueError("Invalid/overflow user_id")
        all_pairs.append(user * size + token)
        if day >= fit_cutoff:
            continue
        value, state = _parse_outcome(outcome_text)
        tokens.append(token)
        days.append(day)
        outcomes.append(value)
        states.append(state)
    pair_keys = np.frombuffer(all_pairs, dtype=np.int64)
    pair_keys.sort()
    if len(pair_keys) > 1 and np.any(pair_keys[1:] == pair_keys[:-1]):
        raise ValueError("Duplicate user/item retained pair in interactions.csv")
    del pair_keys, all_pairs
    tokens = np.frombuffer(tokens, dtype=np.int32)
    days = np.frombuffer(days, dtype=np.int32)
    values = np.frombuffer(outcomes, dtype=np.float64)
    states = np.frombuffer(states, dtype=np.int8)
    keys = tokens.astype(np.int64) * DAY_BASE + days
    order = np.argsort(keys, kind="stable")
    keys, days, values, states = keys[order], days[order], values[order], states[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1] if len(keys) else np.array([], dtype=int)
    grouped = np.zeros((len(starts), N_STATS), dtype=np.float64)
    if len(starts):
        valid = states == 0
        grouped[:, PAIR] = np.diff(np.r_[starts, len(keys)])
        grouped[:, OBS] = np.add.reduceat(valid.astype(np.int64), starts)
        grouped[:, SUM] = np.add.reduceat(values, starts)
        grouped[:, MISSING] = np.add.reduceat((states == 1).astype(np.int64), starts)
        grouped[:, INVALID] = np.add.reduceat((states == 2).astype(np.int64), starts)
        for offset, selected in enumerate((values <= .10, values >= .20, values >= .50,
                                           values >= .80, values >= .95), 5):
            grouped[:, offset] = np.add.reduceat((valid & selected).astype(np.int64), starts)
        grouped[:, ACTIVE] = 1
        grouped[np.arange(len(starts)), WEEKDAY + (days[starts] - 1) % 7] = grouped[:, PAIR]
    return keys[starts], grouped, {"source_rows": source_rows, "excluded_unmapped_rows": excluded,
                                  "indexed_rows": int(grouped[:, PAIR].sum())}


def _series_path(raw_dir, meta):
    explicit = meta.get("series_annotation_path")
    if explicit is not None:
        return Path(explicit)
    for parent in (Path(raw_dir), *Path(raw_dir).parents):
        candidate = parent / "configs/datasets/kion-series-v1.json"
        if candidate.is_file():
            return candidate
    return None


def _relations(raw_dir, meta, mapping, size):
    by_token = {role: [()] * size for role in ROLES}
    seen = set()
    for item_id, actors, directors, studios in _read_csv(Path(raw_dir) / "items.csv",
                                                       ("item_id", "actors", "directors", "studios")):
        if not item_id or item_id in seen:
            raise ValueError("Missing/duplicate item_id in items.csv")
        seen.add(item_id)
        token = mapping.get(item_id)
        if token is not None:
            for role, value in zip(ROLES, (actors, directors, studios)):
                by_token[role][token] = _tags(value)
    path = _series_path(raw_dir, meta)
    if path is not None:
        annotation = json.loads(path.read_text(encoding="utf-8"))
        groups = annotation.get("series")
        if not isinstance(groups, list):
            raise ValueError("Malformed confirmed series annotation")
        group_ids = set()
        for group in groups:
            if group.get("status") != "confirmed":
                continue
            group_id = group.get("id")
            if not isinstance(group_id, str) or not group_id or group_id in group_ids:
                raise ValueError("Missing/duplicate confirmed series ID")
            group_ids.add(group_id)
            for member in group.get("members", []):
                if member.get("status") == "confirmed":
                    token = mapping.get(str(member.get("item_id")))
                    if token is not None:
                        by_token["series"][token] = tuple(sorted(set(by_token["series"][token]) | {group_id}))
    tables = {}
    for role in ROLES:
        labels = sorted({label for values in by_token[role] for label in values})
        lookup = {label: i for i, label in enumerate(labels)}
        offsets = np.r_[0, np.cumsum([len(values) for values in by_token[role]])].astype(np.int64)
        entities = np.array([lookup[label] for values in by_token[role] for label in values], dtype=np.int32)
        card_counts = np.bincount(entities, minlength=len(labels))
        tables[role] = {"offsets": offsets, "entities": entities, "card_counts": card_counts,
                        "size": len(labels)}
    return tables, path


def _entity_prefix(item_keys, item_counts, relation):
    """One small calendar scratch vector per entity; only active days retained."""
    size = relation["size"]
    if not len(item_keys) or not size:
        return _Prefix(np.array([], dtype=np.int64), np.empty((0, 1)), size)
    tokens, days = np.divmod(item_keys, DAY_BASE)
    start_day, end_day = int(days.min()), int(days.max()) + 1
    item_starts = np.searchsorted(tokens, np.arange(len(relation["offsets"])))
    edge_tokens = np.repeat(np.arange(len(relation["offsets"]) - 1), np.diff(relation["offsets"]))
    entity_order = np.argsort(relation["entities"], kind="stable")
    entity_tokens = edge_tokens[entity_order]
    counts = relation["card_counts"]
    offsets = np.r_[0, np.cumsum(counts)]
    key_blocks, value_blocks = [], []
    for entity in range(size):
        daily = np.zeros(end_day - start_day, dtype=np.float64)
        for token in entity_tokens[offsets[entity]:offsets[entity + 1]]:
            left, right = item_starts[token:token + 2]
            daily[days[left:right] - start_day] += item_counts[left:right]
        active = np.flatnonzero(daily)
        key_blocks.append(entity * DAY_BASE + active + start_day)
        value_blocks.append(daily[active])
    return _Prefix(np.concatenate(key_blocks), np.concatenate(value_blocks), size)


def _schema():
    columns, groups = [], {}

    def add(group, names):
        start = len(columns)
        columns.extend(names)
        groups[group] = {"start": start, "end": len(columns)}

    add("identity", ["dynamic_unknown_item", "dynamic_known_item", "dynamic_has_history"])
    popularity = ["pop_all_pairs_log", "pop_all_distinct_users_log"]
    for window in WINDOWS:
        popularity.extend(f"pop_{window}d_{name}" for name in
            ("pairs_log", "distinct_users_log", "share", "previous_pairs_log", "growth",
             "support_pair_log", "global_pairs_log", "previous_global_pairs_log", "growth_observed",
             "growth_supported", "coverage_fraction", "full_window"))
    add("item_popularity", popularity)
    completion = []
    for name in ("all", "1d", "7d", "30d"):
        completion.extend(f"completion_{name}_{field}" for field in
            ("mean", *THRESHOLDS, "pair_count_log", "observed_count_log", "missing_count_log",
             "invalid_count_log", "missing_share", "invalid_share", "no_pair_support",
             "no_observed_support", "no_global_support", "prior_fallback_cumulative", "active_days_log"))
    add("item_completion", completion)
    add("item_weekday", [f"weekday_share_{day}" for day in range(7)] +
        [f"weekday_pair_count_{day}_log" for day in range(7)] +
        [f"weekday_global_share_{day}" for day in range(7)] +
        ["weekday_pair_support_log", "weekday_global_support_log", "weekday_no_item_support", "weekday_no_global_support"])
    for role in ROLES:
        names = [f"related_{role}_{name}" for name in
            ("listed_count_log", "metadata_missing", "other_cards_mean_log", "other_cards_max_log",
             "entity_data_fraction", "unresolved_entities_fraction", "distinct_users_available", "edition_exclusion_unverified")]
        for window in WINDOWS:
            for metric in ("pairs_log", "previous_pairs_log", "growth", "pair_support_log"):
                names.extend(f"related_{role}_{window}d_{metric}_{aggregation}" for aggregation in AGGREGATES)
            names.extend(f"related_{role}_{window}d_{name}" for name in
                ("growth_observed", "growth_supported_fraction", "empty_global_window"))
        add(f"related_interest_{role}", names)
    return columns, groups


def _history_inputs(x, lengths, batch_size):
    x, lengths = np.asarray(x), np.asarray(lengths)
    if (x.ndim != 2 or x.dtype.kind not in "iu" or np.any(x < 0)
            or lengths.shape != (len(x),) or lengths.dtype.kind not in "iu"
            or np.any(lengths < 0) or np.any(lengths > x.shape[1])
            or not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("Invalid padded history, lengths or batch_size")
    return x, lengths


def _reduce_history(matrix, valid, lengths):
    """Shared aggregation for both strict per-row queries and frozen day banks."""
    n, context, width = matrix.shape
    result = np.zeros((n, 3 * width + 3), dtype=np.float32)
    matrix = np.where(valid[..., None], matrix, 0)
    counts = valid.sum(axis=1)
    result[:, :width] = matrix.sum(axis=1) / np.maximum(counts[:, None], 1)
    result[:, width:2 * width] = np.max(np.where(valid[..., None], matrix, -np.inf), axis=1, initial=-np.inf)
    result[counts == 0, width:2 * width] = 0
    last = np.max(np.where(valid, np.arange(context), -1), axis=1, initial=-1)
    rows = np.flatnonzero(last >= 0)
    result[rows, 2 * width:3 * width] = matrix[rows, last[rows]]
    result[:, -3] = np.log1p(counts)
    result[:, -2] = counts / np.maximum(lengths, 1)
    result[:, -1] = counts == 0
    return result


class DynamicIndex:
    def __init__(self, item, global_prefix, relations, entity_prefixes, size, fit_cutoff, provenance):
        self.item, self.global_prefix = item, global_prefix
        self.relations, self.entity_prefixes = relations, entity_prefixes
        self.vocab_size, self.fit_cutoff = size, fit_cutoff
        self.columns, self.groups = _schema()
        self.provenance = provenance
        self._columns = {name: index for index, name in enumerate(self.columns)}
        self.history_columns = [f"dynamic_history_{agg}_{name}" for agg in ("mean", "max", "last")
                                for name in self.columns] + ["dynamic_history_known_count_log", "dynamic_history_known_fraction", "dynamic_history_empty"]
        self.history_groups = {"dynamic_history": {"start": 0, "end": len(self.history_columns)}}

    def _put(self, output, name, value):
        output[:, self._columns[name]] = value

    def _dates(self, values, n, name):
        source = np.asarray(values)
        if source.ndim == 0:
            return np.full(n, _day(source.item()), dtype=np.int64)
        if source.shape != (n,):
            raise ValueError(f"{name} must be scalar or have one value per item")
        return np.fromiter((_day(value) for value in source), dtype=np.int64, count=n)

    def _tokens(self, values):
        tokens = np.asarray(values)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
            raise ValueError("item_tokens must be a one-dimensional integer array")
        if np.any(tokens < 0) or np.any(tokens > np.iinfo(np.int64).max // DAY_BASE):
            raise ValueError("item token out of range")
        return tokens.astype(np.int64, copy=False)

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        """Float32 [N,D], cutoff=min(query origin, explicit freeze, fit cap).

        user_ids is accepted for facade parity; these are global item statistics,
        so legitimate earlier query-user pairs remain in the global background.
        PAD/BOS/UNK are all-zero; out-of-vocabulary tokens get unknown_item=1.
        """
        tokens = self._tokens(item_tokens)
        n = len(tokens)
        dates = np.minimum(self._dates(query_dates, n, "query_dates"), self.fit_cutoff)
        if history_cutoffs is not None:
            dates = np.minimum(dates, self._dates(history_cutoffs, n, "history_cutoffs"))
        if user_ids is not None and np.asarray(user_ids).shape != (n,):
            raise ValueError("user_ids must have one value per item")
        output = np.zeros((n, len(self.columns)), dtype=np.float32)
        unknown = tokens >= self.vocab_size
        output[unknown, self._columns["dynamic_unknown_item"]] = 1
        selected = np.flatnonzero((tokens >= 3) & ~unknown)
        if not len(selected):
            return output
        # Deduplication is essential for repeated H63 history tokens in a batch.
        keys, inverse = np.unique(tokens[selected] * DAY_BASE + dates[selected], return_inverse=True)
        work_tokens, work_dates = np.divmod(keys, DAY_BASE)
        values = self._known_features(work_tokens, work_dates)
        output[selected] = values[inverse]
        return output

    def _known_features(self, tokens, dates):
        output = np.zeros((len(tokens), len(self.columns)), dtype=np.float32)
        put = lambda name, value: self._put(output, name, value)
        all_item = self.item.before(tokens, dates)
        zero = np.zeros(len(tokens), dtype=np.int64)
        all_global = self.global_prefix.before(zero, dates)
        put("dynamic_known_item", 1)
        put("dynamic_has_history", all_item[:, PAIR] > 0)
        put("pop_all_pairs_log", np.log1p(all_item[:, PAIR]))
        put("pop_all_distinct_users_log", np.log1p(all_item[:, PAIR]))
        cached = {None: (all_item, all_global)}
        for window in WINDOWS:
            current = self.item.window(tokens, dates, window)
            previous = self.item.window(tokens, dates - window, window)
            glob = self.global_prefix.window(zero, dates, window)
            glob_prev = self.global_prefix.window(zero, dates - window, window)
            cached[window] = current, glob
            count, prev_count = current[:, PAIR], previous[:, PAIR]
            g, gp = glob[:, PAIR], glob_prev[:, PAIR]
            share = (count + POP_ALPHA) / (g + POP_ALPHA * max(self.vocab_size - 3, 1))
            prev_share = (prev_count + POP_ALPHA) / (gp + POP_ALPHA * max(self.vocab_size - 3, 1))
            support = count + prev_count
            observed = dates - SOURCE_START >= 2 * window
            growth = support / (support + TREND_LAMBDA) * np.clip(np.log(share / prev_share), -TREND_CLIP, TREND_CLIP)
            fields = (np.log1p(count), np.log1p(count), share, np.log1p(prev_count),
                      growth * observed, np.log1p(support), np.log1p(g), np.log1p(gp),
                      observed, support > 0, np.clip(dates - SOURCE_START, 0, window) / window, dates - SOURCE_START >= window)
            for name, value in zip(("pairs_log", "distinct_users_log", "share", "previous_pairs_log", "growth",
                    "support_pair_log", "global_pairs_log", "previous_global_pairs_log", "growth_observed",
                    "growth_supported", "coverage_fraction", "full_window"), fields):
                put(f"pop_{window}d_{name}", value)
        # Prefix-only priors; recent empty windows fall back to this same cutoff.
        constants = np.array([.5, .1, .8, .5, .2, .05])
        global_cum_prior = np.divide(all_global[:, [SUM, 5, 6, 7, 8, 9]], all_global[:, [OBS]],
            out=np.broadcast_to(constants, (len(tokens), 6)).copy(), where=all_global[:, [OBS]] > 0)
        for window, (current, glob) in cached.items():
            name = "all" if window is None else f"{window}d"
            prior = np.divide(glob[:, [SUM, 5, 6, 7, 8, 9]], glob[:, [OBS]],
                              out=global_cum_prior.copy(), where=glob[:, [OBS]] > 0)
            smooth = (current[:, [SUM, 5, 6, 7, 8, 9]] + ALPHA * prior) / (current[:, [OBS]] + ALPHA)
            for offset, field in enumerate(("mean", *THRESHOLDS)):
                put(f"completion_{name}_{field}", smooth[:, offset])
            pair = current[:, PAIR]
            vals = (np.log1p(pair), np.log1p(current[:, OBS]), np.log1p(current[:, MISSING]),
                    np.log1p(current[:, INVALID]), current[:, MISSING] / np.maximum(pair, 1),
                    current[:, INVALID] / np.maximum(pair, 1), pair == 0, current[:, OBS] == 0,
                    glob[:, OBS] == 0, (glob[:, OBS] == 0) & (all_global[:, OBS] > 0), np.log1p(current[:, ACTIVE]))
            for field, value in zip(("pair_count_log", "observed_count_log", "missing_count_log", "invalid_count_log",
                    "missing_share", "invalid_share", "no_pair_support", "no_observed_support", "no_global_support",
                    "prior_fallback_cumulative", "active_days_log"), vals):
                put(f"completion_{name}_{field}", value)
        prior = (all_global[:, WEEKDAY:] + WEEKDAY_BETA / 7) / (all_global[:, [PAIR]] + WEEKDAY_BETA)
        weekdays = (all_item[:, WEEKDAY:] + ALPHA * prior) / (all_item[:, [PAIR]] + ALPHA)
        for day in range(7):
            put(f"weekday_share_{day}", weekdays[:, day])
            put(f"weekday_pair_count_{day}_log", np.log1p(all_item[:, WEEKDAY + day]))
            put(f"weekday_global_share_{day}", prior[:, day])
        for field, value in zip(("pair_support_log", "global_support_log", "no_item_support", "no_global_support"),
                (np.log1p(all_item[:, PAIR]), np.log1p(all_global[:, PAIR]), all_item[:, PAIR] == 0, all_global[:, PAIR] == 0)):
            put(f"weekday_{field}", value)
        self._related(output, tokens, dates, cached)
        return output

    def _related(self, output, tokens, dates, cached):
        n = len(tokens)
        zero = np.zeros(n, dtype=np.int64)
        for role in ROLES:
            relation, prefix = self.relations[role], self.entity_prefixes[role]
            starts, ends = relation["offsets"][tokens], relation["offsets"][tokens + 1]
            lengths = ends - starts
            owner = np.repeat(np.arange(n), lengths)
            take = np.repeat(starts, lengths) + np.arange(len(owner)) - np.repeat(np.cumsum(lengths) - lengths, lengths)
            entities = relation["entities"][take]
            put = lambda field, value: self._put(output, f"related_{role}_{field}", value)
            put("listed_count_log", np.log1p(lengths))
            put("metadata_missing", lengths == 0)
            # Exact normalized raw names have no verified external identity IDs.
            put("unresolved_entities_fraction", (lengths > 0) if role != "series" else 0)
            put("edition_exclusion_unverified", 1)
            if not len(owner):
                continue
            offsets = np.r_[0, np.cumsum(lengths)]

            def aggregate(values):
                result = np.zeros((n, 4))
                rows = np.flatnonzero(lengths)
                order = np.lexsort((values, owner))
                sorted_values = values[order]
                result[:, 0] = np.bincount(owner, weights=values, minlength=n) / np.maximum(lengths, 1)
                result[rows, 1] = sorted_values[offsets[rows + 1] - 1]
                result[rows, 2] = (sorted_values[offsets[rows] + lengths[rows] // 2]
                    + sorted_values[offsets[rows] + (lengths[rows] - 1) // 2]) / 2
                top3 = np.arange(len(owner)) >= offsets[owner + 1] - 3
                result[:, 3] = np.bincount(owner[top3], weights=sorted_values[top3], minlength=n) / np.maximum(np.minimum(lengths, 3), 1)
                return result

            card_counts = relation["card_counts"][entities] - 1
            summarized = aggregate(card_counts)
            put("other_cards_mean_log", np.log1p(summarized[:, 0]))
            put("other_cards_max_log", np.log1p(summarized[:, 1]))
            observed_other_pairs = prefix.before(entities, dates[owner])[:, 0] - cached[None][0][owner, PAIR]
            put("entity_data_fraction", np.bincount(owner, weights=observed_other_pairs > 0, minlength=n) / np.maximum(lengths, 1))
            for window in WINDOWS:
                item_current, global_current = cached[window]
                previous_item = self.item.window(tokens, dates - window, window)[:, PAIR]
                previous_global = self.global_prefix.window(zero, dates - window, window)[:, PAIR]
                current = prefix.window(entities, dates[owner], window)[:, 0] - item_current[owner, PAIR]
                previous = prefix.window(entities, dates[owner] - window, window)[:, 0] - previous_item[owner]
                g = global_current[owner, PAIR] - item_current[owner, PAIR]
                gp = previous_global[owner] - previous_item[owner]
                support = current + previous
                q = (current + POP_ALPHA) / (g + 2 * POP_ALPHA)
                qp = (previous + POP_ALPHA) / (gp + 2 * POP_ALPHA)
                observed = dates - SOURCE_START >= 2 * window
                growth = support / (support + TREND_LAMBDA) * np.clip(np.log(q / qp), -TREND_CLIP, TREND_CLIP)
                growth *= observed[owner]
                for metric, value in (("pairs_log", np.log1p(current)), ("previous_pairs_log", np.log1p(previous)),
                                      ("growth", growth), ("pair_support_log", np.log1p(support))):
                    summary = aggregate(value)
                    for index, agg in enumerate(AGGREGATES):
                        put(f"{window}d_{metric}_{agg}", summary[:, index])
                put(f"{window}d_growth_observed", observed & (lengths > 0))
                put(f"{window}d_growth_supported_fraction", np.bincount(owner, weights=support > 0, minlength=n) / np.maximum(lengths, 1))
                put(f"{window}d_empty_global_window", (global_current[:, PAIR] - item_current[:, PAIR] == 0) & (lengths > 0))

    def history_features(self, x, lengths, query_dates, history_cutoffs=None, *, batch_size=128):
        """Mean/max/last item vectors over valid H63 positions, in bounded chunks."""
        x, lengths = _history_inputs(x, lengths, batch_size)
        dates = self._dates(query_dates, len(x), "query_dates")
        cutoffs = dates if history_cutoffs is None else self._dates(history_cutoffs, len(x), "history_cutoffs")
        result = np.zeros((len(x), len(self.history_columns)), dtype=np.float32)
        width = len(self.columns)
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            tokens = x[start:end]
            valid = (np.arange(x.shape[1])[None, :] < lengths[start:end, None]) & (tokens >= 3) & (tokens < self.vocab_size)
            matrix = self.features(tokens.ravel(), np.repeat(dates[start:end], x.shape[1]),
                np.repeat(cutoffs[start:end], x.shape[1])).reshape(end - start, x.shape[1], width)
            result[start:end] = _reduce_history(matrix, valid, lengths[start:end])
        return result

    def day_bank(self, query_date, history_cutoff=None, *, batch_size=4096):
        """One [vocab,D] float32 bank; caller owns its query/cutoff association.

        Generate one effective day at a time and discard it after all matching
        rows have been written. This avoids an entire date x item x D cache.
        """
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        query_date = _day(query_date)
        history_cutoff = None if history_cutoff is None else _day(history_cutoff)
        result = np.empty((self.vocab_size, len(self.columns)), dtype=np.float32)
        for start in range(0, self.vocab_size, batch_size):
            end = min(start + batch_size, self.vocab_size)
            result[start:end] = self.features(np.arange(start, end), query_date, history_cutoff)
        return result

    @staticmethod
    def history_features_from_bank(x, lengths, bank, *, batch_size=128):
        """Bounded mean/max/last aggregation using a caller's matching day bank.

        The caller must group by the effective cutoff used to build ``bank``;
        this helper deliberately has no date inference or mutable cache state.
        Specials, out-of-vocabulary and out-of-length positions do not contribute.
        Cold known items contribute their valid prior/mask vectors.
        """
        x, lengths = _history_inputs(x, lengths, batch_size)
        bank = np.asarray(bank)
        if bank.ndim != 2 or bank.shape[0] < 3 or bank.dtype.kind != "f":
            raise ValueError("bank must be a floating [vocab,feature] array with specials")
        size, width = bank.shape
        result = np.zeros((len(x), 3 * width + 3), dtype=np.float32)
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            tokens = x[start:end]
            valid = (np.arange(x.shape[1])[None, :] < lengths[start:end, None]) & (tokens >= 3) & (tokens < size)
            safe_tokens = np.where(valid, tokens, 0).astype(np.int64)
            result[start:end] = _reduce_history(bank[safe_tokens], valid, lengths[start:end])
        return result


def build_index(raw_dir, metadata, *, fit_cutoff):
    """Read local CSVs once and build sparse prefixes, capped strictly at fit_cutoff."""
    raw_dir = Path(raw_dir)
    meta, mapping = _metadata(metadata)
    cutoff, size = _day(fit_cutoff), meta["vocab_size"]
    relations, series_path = _relations(raw_dir, meta, mapping, size)
    keys, values, diagnostics = _interaction_table(raw_dir / "interactions.csv", mapping, size, cutoff)
    item = _Prefix(keys, values, size)
    global_prefix = _Prefix(keys % DAY_BASE, values, 1)
    prefixes = {role: _entity_prefix(keys, values[:, PAIR], relation) for role, relation in relations.items()}
    provenance = {"contract": CONTRACT, "fit_cutoff_exclusive": date.fromordinal(cutoff).isoformat(),
        "effective_cutoff": "min(query_date, supplied_history_cutoff, fit_cutoff); strict last_watch_dt < cutoff",
        "source_unit": "retained aggregated user-item pair; not event history or exposure",
        "temporal_mode": "retrospective metadata; reconstructed strict last-pair prefixes",
        "mapping_sha256": _canonical_hash(mapping), "interactions_sha256": _hash_file(raw_dir / "interactions.csv"),
        "items_sha256": _hash_file(raw_dir / "items.csv"), "series_sha256": _hash_file(series_path) if series_path else None,
        "parameters": {"completion_alpha": ALPHA, "weekday_alpha": ALPHA, "weekday_beta": WEEKDAY_BETA,
                       "pop_alpha": POP_ALPHA, "trend_lambda": TREND_LAMBDA, "trend_clip": TREND_CLIP,
                       "completion_constants": [.5, .1, .8, .5, .2, .05]},
        "source_calendar_start": "2021-03-13; declared dataset interval, not proof of complete logging",
        "related_self_exclusion": "candidate raw item only; no verified edition equivalences supplied",
        "related_distinct_users": "unavailable; all related support explicitly counts retained pairs",
        "metadata_identity": "NFKC/casefold exact names, deduplicated within role; unresolved aliases not merged",
        "related_entity_data_fraction": "fraction of listed entities with strict-prefix observed other-card retained pairs",
        "priors": "window-specific strict prefix; empty window -> same-cutoff cumulative -> fixed constants",
        "thresholds_pct": {"lower": [10], "upper": [20, 50, 80, 95]}, **diagnostics}
    provenance["index_bytes"] = item.nbytes + global_prefix.nbytes + sum(prefix.nbytes for prefix in prefixes.values())
    provenance["sparse_item_days"] = len(item.keys)
    provenance["sparse_entity_days"] = {role: len(prefix.keys) for role, prefix in prefixes.items()}
    return DynamicIndex(item, global_prefix, relations, prefixes, size, cutoff, provenance)
