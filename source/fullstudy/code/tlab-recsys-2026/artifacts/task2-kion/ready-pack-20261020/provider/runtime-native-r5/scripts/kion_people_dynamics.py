"""Strict-past, leave-one-work-out people features; NumPy and local CSV only.

People are normalized credit strings, not resolved real-world identities.  Dates
describe retained last user/item aggregates, not a recovered event history.
The catalogue is retrospective; behavioral statistics always use strict dates.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict
import csv
from copy import deepcopy
from datetime import date
import hashlib
import json
from pathlib import Path
import time
import unicodedata

import numpy as np


CONTRACT = "kion-people-dynamics-v1"
DAY_BASE = date.max.toordinal() + 1
ROLE_CAPS = {"sft": "2021-08-09", "rm": "2021-08-02"}
ROLES = ("actors", "directors")
AGGREGATES = ("mean", "max", "median", "std", "q25", "q75", "q90", "top3_mean")
TIERS = ("low_evidence", "ordinary", "high", "very_high")
SOURCE_START = date(2021, 3, 13).toordinal()
# Work prefix: distinct users, valid user means, sum(user means), raw missing,
# raw invalid, retained user/item pairs. Edition outcomes are averaged per user.
USERS, VALID, SUM, MISSING, INVALID, PAIRS = range(6)
POPULARITY = ("catalog_work_count", "other_work_count", "observed_work_count",
    "observation_count", "audience_union", "work_popularity_mean",
    "work_popularity_max", "work_popularity_median", "work_popularity_std",
    "work_popularity_q25", "work_popularity_q75", "work_popularity_q90",
    "other_work_mean_viewers", "observed_work_fraction", "actor_score",
    "no_other_work", "no_behavioral_support", "low_evidence")
EXPERIENCE = ("credits_before_work", "films_before_work", "series_before_work",
    "credits_prev_3y", "credits_prev_5y", "years_since_first_known_credit",
    "years_since_previous_known_credit", "distinct_prior_release_years",
    "same_year_other_credits", "unknown_year_credits", "valid_year_fraction")
QUALITY = ("mean_log_viewers", "median_log_viewers", "mean_completion",
    "high_completion_fraction", "low_completion_fraction", "popular_fraction",
    "supported_work_fraction", "observed_work_count", "valid_user_count",
    "missing_outcome_count", "invalid_outcome_count", "prior_missing")
RELATED = tuple(f"related_{w}d_{field}" for w in (7, 30) for field in
    ("pairs", "previous_pairs", "distinct_users", "previous_distinct_users",
     "growth", "growth_observed", "growth_supported", "empty_global_window",
     "previous_empty_global_window", "observed_days", "full_window"))
GENRES = ("драмы", "комедии", "мелодрамы", "романтика", "детективы", "триллеры",
    "боевики", "криминал", "приключения", "фантастика", "фэнтези", "ужасы",
    "мистика", "военные", "исторические", "историческое", "биография",
    "вестерн", "мюзиклы", "сказки", "катастрофы", "фильм-нуар")
PROFILE_SUPPORT = ("n_all", "n_valid", "n_unknown", "known_fraction", "support_weight",
    "profile_missing", "prior_missing", "profile_supported", "prior_only", "history_unavailable", "candidate_missing")
SPECIALIZATION = tuple(f"specialization_{mode}_{kind}_{field}"
    for mode in ("catalog", "prior_release") for kind, fields in
    (("genre", (*("share_" + str(i) for i in range(len(GENRES))), "match", *PROFILE_SUPPORT)),
     ("format", ("film_share", "series_share", "n_film", "n_series", "match", *PROFILE_SUPPORT))) for field in fields)
METRICS = (*POPULARITY, *("experience_" + x for x in EXPERIENCE),
    *(f"quality_{mode}_{x}" for mode in ("catalog", "prior_release") for x in QUALITY), *RELATED, *SPECIALIZATION)
DIRECTOR_LOG_METRICS = set(POPULARITY[:13]) - {"observed_work_fraction"}


def _aggregates(role, metric):
    if role == "directors":
        return ("mean", "max", "median") if metric in POPULARITY else ("mean", "max", "median", "top3_mean")
    return AGGREGATES


def _day(value):
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if 1 <= int(value) < DAY_BASE:
            return int(value)
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() == value:
                return parsed.toordinal()
        except ValueError:
            pass
    raise ValueError(f"Expected ISO day or ordinal: {value!r}")


def _people(value):
    return tuple(sorted({" ".join(unicodedata.normalize("NFKC", part).casefold().split())
        for part in str(value or "").split(",") if part.strip()}))


def _year(value):
    try:
        number = float(value)
        return int(number) if np.isfinite(number) and number.is_integer() and 1 <= number <= 9999 else None
    except (ValueError, TypeError):
        return None


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _file_hash(path, limit=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        if limit is None:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        else:
            # Hash exactly the consumed CSV header and records in a bounded pilot.
            # Raw CSV used here has one record per line; canonical parsed rows are
            # hashed separately by the reader, so this is only a source fingerprint.
            for _ in range(limit + 1):
                digest.update(stream.readline())
    return digest.hexdigest()


def _source_pin(path):
    """Fingerprint an entire source file, detecting mutation during hashing."""
    path = Path(path).resolve()
    before = path.stat()
    digest = _file_hash(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Source changed while hashing: {path.as_posix()}")
    return {"kind": "file", "path": path.as_posix(), "sha256": digest, "size_bytes": after.st_size}


def _source_pins(raw_dir, metadata):
    pins = {"items": _source_pin(Path(raw_dir) / "items.csv"),
        "interactions": _source_pin(Path(raw_dir) / "interactions.csv"),
        "parser": _source_pin(Path(__file__))}
    if isinstance(metadata, (str, Path)):
        pins["metadata"] = _source_pin(metadata)
    else:
        snapshot = deepcopy(dict(metadata))
        pins["metadata"] = {"kind": "canonical_inline_json", "sha256": _hash(snapshot), "value": snapshot}
    return pins


def _assert_sources_unchanged(expected):
    for name, pin in expected.items():
        if pin["kind"] != "file":
            if pin["kind"] == "canonical_inline_json" and _hash(pin["value"]) != pin["sha256"]:
                raise ValueError("Inline metadata snapshot hash mismatch")
            continue
        actual = _source_pin(pin["path"])
        if actual != pin:
            raise ValueError(f"Source changed since index build: {name}: {pin['path']}")


def _rows(path, required, max_rows=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)) or not set(required).issubset(header):
            raise ValueError(f"Missing/duplicate headers: {path}")
        positions = [header.index(key) for key in required]
        for number, row in enumerate(reader):
            if max_rows is not None and number >= max_rows:
                break
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row {number + 2}: {path}")
            yield tuple(row[position] for position in positions)


class _Prefix:
    """Sparse entity/day deltas, reduced before cumulative summation."""
    def __init__(self, keys, values, size):
        keys, values = np.asarray(keys, np.int64), np.asarray(values, np.float64)
        if values.ndim == 1:
            values = values[:, None]
        order = np.argsort(keys, kind="stable")
        keys, values = keys[order], values[order]
        if len(keys):
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
            keys, values = keys[starts], np.add.reduceat(values, starts, axis=0)
        self.keys = keys
        self.cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
        self.base = np.searchsorted(keys, np.arange(size, dtype=np.int64) * DAY_BASE)

    def before(self, entities, day):
        entities = np.asarray(entities, np.int64)
        day = np.clip(np.asarray(day, np.int64), 1, DAY_BASE - 1)
        position = np.searchsorted(self.keys, entities * DAY_BASE + day, side="left")
        return self.cumulative[position] - self.cumulative[self.base[entities]]

    @property
    def nbytes(self):
        return self.keys.nbytes + self.cumulative.nbytes + self.base.nbytes


def _catalogue(raw_dir, metadata, confirmed_work_ids):
    meta = json.loads(Path(metadata).read_text(encoding="utf-8")) if isinstance(metadata, (str, Path)) else deepcopy(dict(metadata))
    mapping, size = meta.get("item_id_to_token"), meta.get("vocab_size")
    if (not isinstance(mapping, dict) or not isinstance(size, int) or isinstance(size, bool)
            or size < 3 or any(not isinstance(k, str) or not k or not isinstance(v, int)
                              or isinstance(v, bool) for k, v in mapping.items())
            or sorted(mapping.values()) != list(range(3, size))
            or meta.get("special_tokens", {"PAD": 0, "BOS": 1, "UNK": 2}) != {"PAD": 0, "BOS": 1, "UNK": 2}):
        raise ValueError("Expected contiguous unique train-known item mapping after PAD/BOS/UNK")
    confirmed = {str(k): str(v) for k, v in (confirmed_work_ids or {}).items()}
    if any(not k.strip() or not v.strip() for k, v in confirmed.items()):
        raise ValueError("Confirmed work IDs must be nonempty")
    items, seen = {}, set()
    for item_id, actors, directors, year, kind in _rows(Path(raw_dir) / "items.csv",
            ("item_id", "actors", "directors", "release_year", "content_type")):
        if not item_id.strip() or item_id in seen:
            raise ValueError("Missing/duplicate items.csv item ID")
        seen.add(item_id)
        if item_id in mapping:
            items[item_id] = {"actors": _people(actors), "directors": _people(directors),
                "year": _year(year), "type": kind if kind in ("film", "series") else None}
    genres = {item: sorted(set(_people(value)) & set(GENRES)) for item, value in
              _rows(Path(raw_dir) / "items.csv", ("item_id", "genres")) if item in mapping}
    token_attributes = [{} for _ in range(size)]
    groups, lookup = [], {}
    token_work = np.full(size, -1, dtype=np.int32)
    for item_id, token in sorted(mapping.items(), key=lambda pair: pair[1]):
        key = ("work", confirmed[item_id]) if item_id in confirmed else ("card", item_id)
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append({"cards": [], "actors": set(), "directors": set(), "years": set(), "types": set(), "genres": set()})
        work = lookup[key]
        token_work[token] = work
        row = items.get(item_id, {"actors": (), "directors": (), "year": None, "type": None})
        group = groups[work]
        group["cards"].append(item_id)
        for role in ROLES:
            group[role].update(row[role])
        group["years"].add(row["year"])
        group["types"].add(row["type"])
        group["genres"].update(genres.get(item_id, []))
        token_attributes[token] = {"type": row["type"], "genres": genres.get(item_id, [])}
    for group in groups:
        for role in ROLES:
            group[role] = sorted(group[role])
        group["year"] = next(iter(group["years"])) if len(group["years"]) == 1 else None
        group["type"] = next(iter(group["types"])) if len(group["types"]) == 1 else None
        del group["years"], group["types"]
        group["genres"] = sorted(group["genres"])
    relations = {}
    for role in ROLES:
        labels = sorted({person for group in groups for person in group[role]})
        person_ids = {person: i for i, person in enumerate(labels)}
        portfolios = [[] for _ in labels]
        for work, group in enumerate(groups):
            for person in group[role]:
                portfolios[person_ids[person]].append(work)
        offsets = np.r_[0, np.cumsum([len(works) for works in portfolios])].astype(np.int64)
        works = np.array([work for portfolio in portfolios for work in portfolio], np.int32)
        relations[role] = {"labels": labels, "lookup": person_ids, "offsets": offsets,
            "works": works, "owners": np.repeat(np.arange(len(labels)), np.diff(offsets))}
    return meta, mapping, token_work, groups, relations, token_attributes, len(set(mapping) - set(items))


def _interactions(path, mapping, token_work, cutoff, max_rows):
    users, tokens, days, outcomes, states = array("q"), array("i"), array("i"), array("d"), array("b")
    seen_users, seen_tokens = array("q"), array("i")
    date_cache, source_rows, excluded = {}, 0, 0
    fingerprint = hashlib.sha256()
    for user_text, item, day_text, outcome in _rows(path,
            ("user_id", "item_id", "last_watch_dt", "watched_pct"), max_rows):
        source_rows += 1
        fingerprint.update(json.dumps((user_text, item, day_text, outcome), ensure_ascii=False).encode("utf-8") + b"\n")
        if day_text not in date_cache:
            date_cache[day_text] = _day(day_text)
        day = date_cache[day_text]
        token = mapping.get(item)
        if token is None:
            excluded += 1
            continue
        try:
            user = int(user_text)
            if not 0 <= user <= np.iinfo(np.int64).max:
                raise ValueError
        except ValueError as error:
            raise ValueError("Invalid user_id") from error
        seen_users.append(user)
        seen_tokens.append(token)
        if day >= cutoff:
            continue
        text = outcome.strip()
        state, value = 0, 0.0
        if text.casefold() in ("", "nan", "none", "null"):
            state = 1
        else:
            try:
                value = float(text)
                if not np.isfinite(value) or not 0 <= value <= 100:
                    state, value = 2, 0.0
            except ValueError:
                state = 2
        users.append(user)
        tokens.append(token)
        days.append(day)
        outcomes.append(value / 100)
        states.append(state)
    check_users, check_tokens = np.asarray(seen_users), np.asarray(seen_tokens)
    order = np.lexsort((check_tokens, check_users))
    if len(order) > 1 and np.any((check_users[order[1:]] == check_users[order[:-1]]) &
                                 (check_tokens[order[1:]] == check_tokens[order[:-1]])):
        raise ValueError("Duplicate retained user/item pair")
    work = token_work[np.asarray(tokens)]
    users, days = np.asarray(users, np.int64), np.asarray(days, np.int32)
    values, states = np.asarray(outcomes, np.float64), np.asarray(states, np.int8)
    order = np.lexsort((days, users, work))
    work, users, days, values, states = (x[order] for x in (work, users, days, values, states))
    n = len(work)
    start = np.r_[True, (work[1:] != work[:-1]) | (users[1:] != users[:-1])] if n else np.array([], bool)
    valid = states == 0
    cumulative_n, cumulative_sum = np.cumsum(valid), np.cumsum(values)
    starts = np.flatnonzero(start)
    owners = np.cumsum(start) - 1
    base_n = np.r_[0, cumulative_n[:-1]][starts] if n else np.array([])
    base_sum = np.r_[0, cumulative_sum[:-1]][starts] if n else np.array([])
    user_n, user_sum = cumulative_n - base_n[owners], cumulative_sum - base_sum[owners]
    means = np.divide(user_sum, user_n, out=np.zeros(n), where=user_n > 0)
    previous_mean = np.r_[0, means[:-1]] if n else np.array([])
    previous_n = np.r_[0, user_n[:-1]] if n else np.array([])
    previous_mean[start], previous_n[start] = 0, 0
    delta = np.zeros((n, 6))
    delta[:, USERS] = start
    delta[:, VALID] = (user_n > 0) & (previous_n == 0)
    delta[:, SUM] = means - previous_mean
    delta[:, MISSING], delta[:, INVALID], delta[:, PAIRS] = states == 1, states == 2, 1
    prefix = _Prefix(work.astype(np.int64) * DAY_BASE + days, delta, int(token_work.max()) + 1)
    first = {"work": work[starts], "users": users[starts], "days": days[starts]}
    event_order = np.lexsort((users, days, work))
    events = {"work": work[event_order], "users": users[event_order], "days": days[event_order]}
    return prefix, first, events, {"source_rows_read": source_rows, "indexed_rows": n,
        "excluded_unmapped_rows": excluded, "unique_work_users": len(starts),
        "parsed_rows_sha256": fingerprint.hexdigest(), "max_rows": max_rows,
        "duplicate_check_scope": "all mapped rows read, including after fit cap"}


def _union_prefixes(relation, first, work_count):
    """Exact union excluding one work, using each user's two earliest works.

    Each portfolio is processed separately. Only reduced person/day and
    person-work/day deltas survive; no all-people x user x work edge table.
    """
    work_offsets = np.searchsorted(first["work"], np.arange(work_count + 1))
    union_keys, union_values, correction_keys, correction_values = [], [], [], []
    for person in range(len(relation["labels"])):
        left, right = relation["offsets"][person:person + 2]
        works = relation["works"][left:right]
        take = [np.arange(work_offsets[w], work_offsets[w + 1]) for w in works if work_offsets[w + 1] > work_offsets[w]]
        if not take:
            continue
        take = np.concatenate(take)
        order = np.lexsort((first["work"][take], first["days"][take], first["users"][take]))
        take = take[order]
        users, days, work = (first[field][take] for field in ("users", "days", "work"))
        starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
        ends = np.r_[starts[1:], len(users)]
        first_days = days[starts]
        unique, counts = np.unique(first_days, return_counts=True)
        union_keys.append(person * DAY_BASE + unique.astype(np.int64))
        union_values.append(counts)
        edges = left + np.searchsorted(works, work[starts])
        keys = edges * DAY_BASE + first_days
        has_second = starts + 1 < ends
        keys = np.r_[keys, edges[has_second] * DAY_BASE + days[starts[has_second] + 1]]
        values = np.r_[np.ones(len(starts)), -np.ones(has_second.sum())]
        unique, inverse = np.unique(keys, return_inverse=True)
        values = np.bincount(inverse, weights=values)
        keep = values != 0
        correction_keys.append(unique[keep])
        correction_values.append(values[keep])
    cat = lambda values, dtype: np.concatenate(values).astype(dtype) if values else np.array([], dtype=dtype)
    return (_Prefix(cat(union_keys, np.int64), cat(union_values, np.float64), len(relation["labels"])),
            _Prefix(cat(correction_keys, np.int64), cat(correction_values, np.float64), len(relation["works"])))


def _summarize(values):
    values = np.asarray([value for value in values if value is not None], np.float64)
    if not len(values):
        return [0.0] * len(AGGREGATES), 0
    return [float(np.mean(values)), float(np.max(values)), float(np.median(values)),
        float(np.std(values)), *np.quantile(values, [.25, .75, .90], method="linear").tolist(),
        float(np.mean(np.sort(values)[-3:]))], len(values)


class PeopleDynamicsIndex:
    def __init__(self, mapping, token_work, groups, relations, work_prefix, unions,
                 corrections, events, fit_cutoff, parameters, provenance, token_attributes):
        self.mapping, self.token_work, self.work_groups = mapping, token_work, groups
        self.relations, self.work_prefix = relations, work_prefix
        self.unions, self.corrections, self.events = unions, corrections, events
        self.fit_cutoff, self.parameters, self.provenance = fit_cutoff, parameters, provenance
        self.token_attributes = token_attributes
        self.vocab_size = len(token_work)
        self.years = np.array([group["year"] or 0 for group in groups], np.int32)
        self.genre_incidence = np.array([[genre in group["genres"] for genre in GENRES] for group in groups], np.float64).reshape(len(groups), len(GENRES))
        self.format_incidence = np.array([[group["type"] == kind for kind in ("film", "series")] for group in groups], np.float64).reshape(len(groups), 2)
        self.event_offsets = np.searchsorted(events["work"], np.arange(len(groups) + 1))
        self._day_cache = OrderedDict()
        self._feature_cache = OrderedDict()
        self.top_people = {}
        for role in ROLES:
            rel = relations[role]
            credit_counts = np.diff(rel["offsets"])
            order = sorted(range(len(rel["labels"])), key=lambda p: (-int(credit_counts[p]), rel["labels"][p]))
            self.top_people[role] = [rel["labels"][p] for p in order[:parameters["top_" + role]]]
        self.columns, self.groups = self._schema()
        self._column = {name: i for i, name in enumerate(self.columns)}

    def _schema(self):
        columns = ["people_known_item", "people_unknown_item", "people_candidate_year_missing"]
        groups = {"people_identity": {"start": 0, "end": 3}}
        for role in ROLES:
            start = len(columns)
            columns += [f"people_{role}_{field}" for field in ("listed_count", "metadata_missing", "topn_coverage")]
            for metric in METRICS:
                columns += [f"people_{role}_{metric}_{agg}" for agg in _aggregates(role, metric)]
                columns += [f"people_{role}_{metric}_{field}" for field in ("available_count", "available_fraction", "missing")]
            if role == "actors":
                columns += [f"people_actors_tier_{tier}_{field}" for tier in TIERS for field in ("count", "fraction", "has")]
                columns += ["people_actors_thresholds_missing", "people_actors_q90", "people_actors_q99", "people_actors_threshold_pool_count"]
            columns += [f"people_{role}_identity_{i}" for i in range(len(self.top_people[role]))]
            groups["people_dynamics_" + role] = {"start": start, "end": len(columns)}
        return columns, groups

    def _date_state(self, cutoff):
        if cutoff in self._day_cache:
            self._day_cache.move_to_end(cutoff)
            return self._day_cache[cutoff]
        stats = self.work_prefix.before(np.arange(len(self.work_groups)), cutoff)
        state = {"stats": stats}
        state["window_counts"] = {}
        for window in (7, 30):
            last = self.work_prefix.before(np.arange(len(self.work_groups)), cutoff - window)[:, PAIRS]
            previous = self.work_prefix.before(np.arange(len(self.work_groups)), cutoff - 2 * window)[:, PAIRS]
            state["window_counts"][window] = stats[:, PAIRS] - last, last - previous
        for role, rel in self.relations.items():
            state[role] = (np.bincount(rel["owners"], weights=stats[rel["works"], USERS], minlength=len(rel["labels"])),
                np.bincount(rel["owners"], weights=stats[rel["works"], USERS] > 0, minlength=len(rel["labels"])))
        self._day_cache[cutoff] = state
        while len(self._day_cache) > self.parameters["cache_days"]:
            self._day_cache.popitem(last=False)
        return state

    def _context(self, work, cutoff, token):
        state = dict(self._date_state(cutoff))
        state["candidate_attributes"] = self.token_attributes[token]
        state["quality_references"] = {}
        state["specialization_priors"] = {}
        for mode in ("catalog", "prior_release"):
            ref = np.arange(len(self.work_groups))
            ref = ref[ref != work]
            if mode == "prior_release":
                ref = ref[(self.years[ref] > 0) & (self.years[ref] < self.years[work])]
            values = state["stats"][ref]
            valid = values[:, VALID].sum()
            observed = values[values[:, USERS] > 0, USERS]
            state["quality_references"][mode] = (
                float(values[:, SUM].sum() / valid) if valid else None,
                float(np.quantile(observed, .90, method="linear")) if len(observed) else None)
            # Prior fit uses only works observed in this role-capped past prefix.
            profile_ref = ref[state["stats"][ref, USERS] > 0]
            if mode == "prior_release":
                profile_ref = profile_ref[self.years[profile_ref] < date.fromordinal(cutoff).year]
            state["specialization_priors"][mode] = {}
            for kind, incidence in (("genre", self.genre_incidence), ("format", self.format_incidence)):
                available = profile_ref[incidence[profile_ref].sum(axis=1) > 0]
                state["specialization_priors"][mode][kind] = incidence[available].mean(axis=0) if len(available) else None
        return state

    def _thresholds(self, work, cutoff, state):
        stats = state["stats"]
        total, observed = stats[:, USERS].sum() - stats[work, USERS], np.count_nonzero(stats[:, USERS]) - (stats[work, USERS] > 0)
        mu = total / observed if observed else None
        sums, counts = (value.copy() for value in state["actors"])
        rel = self.relations["actors"]
        people = [rel["lookup"][person] for person in self.work_groups[work]["actors"]]
        sums[people] -= stats[work, USERS]
        counts[people] -= stats[work, USERS] > 0
        supported = counts >= self.parameters["min_works"]
        alpha = self.parameters["alpha"]
        scores = np.log1p((sums[supported] + alpha * mu) / (counts[supported] + alpha)) if mu is not None else np.array([])
        thresholds = np.quantile(scores, [.90, .99], method="linear") if len(scores) else None
        return {"mu": mu, "q90": None if thresholds is None else float(thresholds[0]),
            "q99": None if thresholds is None else float(thresholds[1]), "pool_count": len(scores)}

    def _window(self, works, left, right):
        users, count = [], 0
        for work in works:
            start, stop = self.event_offsets[work:work + 2]
            days = self.events["days"][start:stop]
            a, b = np.searchsorted(days, [left, right], side="left")
            count += int(b - a)
            if b > a:
                users.append(self.events["users"][start + a:start + b])
        return count, len(np.unique(np.concatenate(users))) if users else 0

    def _person(self, role, person, work, cutoff, state, thresholds):
        rel, stats = self.relations[role], state["stats"]
        start, stop = rel["offsets"][person:person + 2]
        all_works = rel["works"][start:stop]
        others = all_works[all_works != work]
        edge = start + int(np.searchsorted(all_works, work))
        union = float(self.unions[role].before([person], cutoff)[0, 0] - self.corrections[role].before([edge], cutoff)[0, 0])
        n = stats[others, USERS]
        observed = n[n > 0]
        k, support = len(others), len(observed)
        result = {"catalog_work_count": len(all_works) if role == "directors" else k,
            "other_work_count": k, "observed_work_count": support, "observation_count": float(n.sum()),
            "audience_union": max(0.0, union), "other_work_mean_viewers": float(n.mean()) if k else None,
            "observed_work_fraction": support / k if k else None, "no_other_work": int(k == 0),
            "no_behavioral_support": int(support == 0), "low_evidence": int(support < self.parameters["min_works"])}
        for name, value in zip(("mean", "max", "median", "std", "q25", "q75", "q90"), _summarize(observed)[0][:7]):
            result["work_popularity_" + name] = value if support else None
        mu, alpha = thresholds["mu"], self.parameters["alpha"]
        result["actor_score"] = float(np.log1p((n.sum() + alpha * mu) / (support + alpha))) if mu is not None else None
        if result["low_evidence"] or thresholds["q90"] is None:
            result["tier"] = "low_evidence"
        elif result["actor_score"] >= thresholds["q99"]:
            result["tier"] = "very_high"
        elif result["actor_score"] >= thresholds["q90"]:
            result["tier"] = "high"
        else:
            result["tier"] = "ordinary"
        current_year, years = self.years[work], self.years[others]
        prior = others[(years > 0) & (years < current_year)] if current_year else np.array([], np.int32)
        py = self.years[prior]
        experience = {"credits_before_work": len(prior),
            "films_before_work": sum(self.work_groups[w]["type"] == "film" for w in prior),
            "series_before_work": sum(self.work_groups[w]["type"] == "series" for w in prior),
            "credits_prev_3y": int(np.count_nonzero(py >= current_year - 3)),
            "credits_prev_5y": int(np.count_nonzero(py >= current_year - 5)),
            "years_since_first_known_credit": int(current_year - py.min()) if len(py) else 0,
            "years_since_previous_known_credit": int(current_year - py.max()) if len(py) else None,
            "distinct_prior_release_years": len(np.unique(py)),
            "same_year_other_credits": int(np.count_nonzero(years == current_year)),
            "unknown_year_credits": int(np.count_nonzero(years == 0)),
            "valid_year_fraction": float(np.mean(years > 0)) if k else None}
        for name, value in experience.items():
            result["experience_" + name] = value if current_year or name in ("unknown_year_credits", "valid_year_fraction") else None
        for mode, works in (("catalog", others), ("prior_release", prior)):
            quality = self._quality(works, state, mode)
            for name, value in quality.items():
                result[f"quality_{mode}_{name}"] = value if mode == "catalog" or current_year else None
            profile_works = others if mode == "catalog" else prior[self.years[prior] < date.fromordinal(cutoff).year]
            for key, value in self._specialization(profile_works, work, state, mode).items():
                result[f"specialization_{mode}_{key}"] = value
        for window in (7, 30):
            count, users = self._window(others, cutoff - window, cutoff)
            previous, previous_users = self._window(others, cutoff - 2 * window, cutoff - window)
            # Global pair denominators can use the sparse work prefix; user-union
            # denominator is deliberately not substituted for retained pairs.
            current_counts, previous_counts = state["window_counts"][window]
            now = current_counts.sum() - current_counts[work]
            before = previous_counts.sum() - previous_counts[work]
            observed_growth = cutoff - SOURCE_START >= 2 * window
            s = count + previous
            growth = s / (s + self.parameters["trend_lambda"]) * np.clip(
                np.log((count + 1) / (now + 2)) - np.log((previous + 1) / (before + 2)),
                -self.parameters["trend_clip"], self.parameters["trend_clip"])
            fields = {"pairs": count, "previous_pairs": previous, "distinct_users": users,
                "previous_distinct_users": previous_users, "growth": float(growth) if observed_growth else 0.0,
                "growth_observed": int(observed_growth), "growth_supported": int(s >= self.parameters["min_related_pairs"]),
                "empty_global_window": int(now == 0), "previous_empty_global_window": int(before == 0),
                "observed_days": min(window, max(0, cutoff - SOURCE_START)),
                "full_window": int(cutoff - SOURCE_START >= window)}
            if not fields["growth_supported"]:
                fields["growth"] = 0.0
            result.update({f"related_{window}d_{key}": value for key, value in fields.items()})
        return result

    def _specialization(self, works, candidate, state, mode):
        unavailable = mode == "prior_release" and self.years[candidate] == 0
        alpha, result = self.parameters["alpha"], {}
        for kind, incidence in (("genre", self.genre_incidence), ("format", self.format_incidence)):
            valid = incidence[works].sum(axis=1) > 0
            n = int(valid.sum())
            prior = state["specialization_priors"][mode][kind]
            counts = incidence[works[valid]].sum(axis=0)
            profile = ((counts + alpha * prior) / (n + alpha)) if prior is not None else (counts / n if n else None)
            fields = {"n_all": len(works), "n_valid": n, "n_unknown": len(works) - n,
                "known_fraction": n / len(works) if len(works) else None,
                "support_weight": n / (n + alpha), "profile_missing": int(n == 0),
                "prior_missing": int(prior is None), "profile_supported": int(n >= self.parameters["min_works"]),
                "prior_only": int(n == 0 and prior is not None), "history_unavailable": int(unavailable),
                "candidate_missing": int(not state["candidate_attributes"]["genres"]) if kind == "genre" else int(state["candidate_attributes"]["type"] is None)}
            for key, value in fields.items():
                result[kind + "_" + key] = value if not unavailable or key in ("history_unavailable", "candidate_missing") else None
            if kind == "genre":
                for i in range(len(GENRES)):
                    result[f"genre_share_{i}"] = float(profile[i]) if profile is not None and not unavailable else None
                candidate_genres = state["candidate_attributes"]["genres"]
                indexes = [i for i, genre in enumerate(GENRES) if genre in candidate_genres]
                result["genre_match"] = float(profile[indexes].mean()) if profile is not None and indexes and not unavailable else None
            else:
                for i, key in enumerate(("film_share", "series_share")):
                    result["format_" + key] = float(profile[i]) if profile is not None and not unavailable else None
                result["format_n_film"] = int(counts[0]) if not unavailable else None
                result["format_n_series"] = int(counts[1]) if not unavailable else None
                kind_index = {"film": 0, "series": 1}.get(state["candidate_attributes"]["type"])
                result["format_match"] = float(profile[kind_index]) if profile is not None and kind_index is not None and not unavailable else None
        return result

    def _quality(self, works, state, mode):
        stats = state["stats"]
        mu, pop_threshold = state["quality_references"][mode]
        values = stats[works]
        observed = values[:, USERS] > 0
        supported = values[:, VALID] >= self.parameters["min_outcome_users"]
        means = np.divide(values[:, SUM], values[:, VALID], out=np.zeros(len(works)), where=values[:, VALID] > 0)
        completion = ((values[:, SUM] + self.parameters["alpha"] * mu) / (values[:, VALID] + self.parameters["alpha"])) if mu is not None else np.where(values[:, VALID] > 0, means, np.nan)
        q = completion[observed & np.isfinite(completion)]
        logs = np.log1p(values[observed, USERS])
        return {"mean_log_viewers": float(logs.mean()) if len(logs) else None,
            "median_log_viewers": float(np.median(logs)) if len(logs) else None,
            "mean_completion": float(q.mean()) if len(q) else None,
            "high_completion_fraction": float(np.mean(completion[supported] >= self.parameters["high_completion"])) if supported.any() else None,
            "low_completion_fraction": float(np.mean(completion[supported] <= self.parameters["low_completion"])) if supported.any() else None,
            "popular_fraction": float(np.mean(values[observed, USERS] >= pop_threshold)) if observed.any() and pop_threshold is not None else None,
            "supported_work_fraction": float(np.mean(supported)) if len(works) else None,
            "observed_work_count": int(observed.sum()), "valid_user_count": float(values[:, VALID].sum()),
            "missing_outcome_count": float(values[:, MISSING].sum()), "invalid_outcome_count": float(values[:, INVALID].sum()),
            "prior_missing": int(mu is None)}

    def person_statistics(self, role, person, candidate_token, cutoff):
        """Raw, nullable, auditable per-person values; candidate must credit person."""
        if role not in ROLES:
            raise ValueError("role must be actors or directors")
        if isinstance(candidate_token, (bool, np.bool_)) or not isinstance(candidate_token, (int, np.integer)) or not 3 <= candidate_token < self.vocab_size:
            raise ValueError("candidate_token must be a known nonspecial item")
        names = _people(person)
        if len(names) != 1:
            raise ValueError("Expected one normalized person name")
        person = names[0]
        work = int(self.token_work[candidate_token])
        if person not in self.work_groups[work][role]:
            raise ValueError("Candidate does not credit requested person in this role")
        cutoff = min(_day(cutoff), self.fit_cutoff)
        state = self._context(work, cutoff, int(candidate_token))
        return self._person(role, self.relations[role]["lookup"][person], work, cutoff,
                            state, self._thresholds(work, cutoff, state))

    def specialization_statistics(self, role, person, candidate_token, cutoff, *, mode="catalog"):
        """Nullable smoothed genre/format shares, match and support for one person."""
        if mode not in ("catalog", "prior_release"):
            raise ValueError("mode must be catalog or prior_release")
        values = self.person_statistics(role, person, candidate_token, cutoff)
        prefix = "specialization_" + mode + "_"
        return {key[len(prefix):]: value for key, value in values.items() if key.startswith(prefix)}

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        tokens = np.asarray(item_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0):
            raise ValueError("item_tokens must be nonnegative integer vector")
        n = len(tokens)
        def dates(values):
            values = np.asarray(values)
            if values.ndim == 0:
                return np.full(n, _day(values.item()), np.int64)
            if values.shape != (n,):
                raise ValueError("Dates must be scalar or one per token")
            return np.array([_day(value) for value in values], np.int64)
        cutoff = np.minimum(dates(query_dates), self.fit_cutoff)
        if history_cutoffs is not None:
            cutoff = np.minimum(cutoff, dates(history_cutoffs))
        if user_ids is not None and np.asarray(user_ids).shape != (n,):
            raise ValueError("user_ids must have one value per token")
        output = np.zeros((n, len(self.columns)), np.float32)
        output[tokens >= self.vocab_size, self._column["people_unknown_item"]] = 1
        pairs = sorted({(int(token), int(day)) for token, day in zip(tokens, cutoff) if 3 <= token < self.vocab_size}, key=lambda pair: (pair[1], pair[0]))
        for token, day in pairs:
            cache_key = (token, day)
            if cache_key in self._feature_cache:
                self._feature_cache.move_to_end(cache_key)
                output[(tokens == token) & (cutoff == day)] = self._feature_cache[cache_key]
                continue
            row = np.zeros(len(self.columns), np.float32)
            put = lambda name, value: row.__setitem__(self._column[name], value)
            work = int(self.token_work[token])
            state = self._context(work, day, token)
            thresholds = self._thresholds(work, day, state)
            put("people_known_item", 1)
            put("people_candidate_year_missing", self.years[work] == 0)
            for role in ROLES:
                names, rel = self.work_groups[work][role], self.relations[role]
                people = [self._person(role, rel["lookup"][name], work, day, state, thresholds) for name in names]
                prefix = "people_" + role + "_"
                put(prefix + "listed_count", len(names))
                put(prefix + "metadata_missing", not names)
                put(prefix + "topn_coverage", sum(name in self.top_people[role] for name in names) / len(names) if names else 0)
                for metric in METRICS:
                    raw = [person[metric] for person in people]
                    # Related log counts are aggregated after a per-person log,
                    # while director audience/counts are logged after aggregation.
                    if ((metric.startswith("related_") and metric.endswith(("_pairs", "_distinct_users")))
                            or metric.endswith(("_format_n_film", "_format_n_series"))):
                        raw = [None if value is None else np.log1p(value) for value in raw]
                    values, available = _summarize(raw)
                    for agg in _aggregates(role, metric):
                        value = values[AGGREGATES.index(agg)]
                        if role == "directors" and metric in DIRECTOR_LOG_METRICS:
                            value = np.log1p(value)
                        put(prefix + metric + "_" + agg, value)
                    put(prefix + metric + "_available_count", available)
                    put(prefix + metric + "_available_fraction", available / len(names) if names else 0)
                    put(prefix + metric + "_missing", available == 0)
                if role == "actors":
                    for tier in TIERS:
                        count = sum(person["tier"] == tier for person in people)
                        for field, value in (("count", count), ("fraction", count / len(names) if names else 0), ("has", count > 0)):
                            put(prefix + "tier_" + tier + "_" + field, value)
                    put("people_actors_thresholds_missing", thresholds["q90"] is None)
                    put("people_actors_q90", thresholds["q90"] or 0)
                    put("people_actors_q99", thresholds["q99"] or 0)
                    put("people_actors_threshold_pool_count", thresholds["pool_count"])
                for i, name in enumerate(self.top_people[role]):
                    put(prefix + "identity_" + str(i), name in names)
            output[(tokens == token) & (cutoff == day)] = row
            self._feature_cache[cache_key] = row
            while len(self._feature_cache) > self.parameters["cache_rows"]:
                self._feature_cache.popitem(last=False)
        if not np.isfinite(output).all():
            raise AssertionError("Nonfinite people feature values")
        return output

    @property
    def nbytes(self):
        prefixes = [self.work_prefix, *self.unions.values(), *self.corrections.values()]
        return sum(prefix.nbytes for prefix in prefixes) + self.token_work.nbytes + self.years.nbytes + self.genre_incidence.nbytes + self.format_incidence.nbytes + self.event_offsets.nbytes + sum(value.nbytes for value in self.events.values()) + sum(rel[key].nbytes for rel in self.relations.values() for key in ("offsets", "works", "owners"))

    def save(self, output_dir):
        """Persist compact arrays and a self-contained manifest; no pickle."""
        _assert_sources_unchanged(self.provenance["source_pins"])
        if _hash(self.mapping) != self.provenance["mapping_sha256"]:
            raise ValueError("Index mapping changed since index build")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {"token_work": self.token_work, **{"events_" + k: v for k, v in self.events.items()}}
        prefixes = {"work": self.work_prefix, **{"union_" + k: v for k, v in self.unions.items()},
                    **{"correction_" + k: v for k, v in self.corrections.items()}}
        for name, prefix in prefixes.items():
            for key in ("keys", "cumulative", "base"):
                arrays[name + "_" + key] = getattr(prefix, key)
        for role, rel in self.relations.items():
            for key in ("offsets", "works", "owners"):
                arrays[role + "_" + key] = rel[key]
        np.savez_compressed(output_dir / "index.npz", **arrays)
        manifest = {"contract": CONTRACT, "mapping": self.mapping, "work_groups": self.work_groups,
            "relation_labels": {role: rel["labels"] for role, rel in self.relations.items()},
            "token_attributes": self.token_attributes,
            "fit_cutoff": self.fit_cutoff, "parameters": self.parameters, "provenance": self.provenance,
            "columns": self.columns, "groups": self.groups, "top_people": self.top_people,
            "index_array_bytes": self.nbytes, "index_npz_sha256": _file_hash(output_dir / "index.npz")}
        (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        return manifest

    @classmethod
    def load(cls, output_dir, *, verify_sources=True):
        output_dir = Path(output_dir)
        meta = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if meta.get("contract") != CONTRACT or _file_hash(output_dir / "index.npz") != meta["index_npz_sha256"]:
            raise ValueError("Index contract/hash mismatch")
        if _hash(meta["mapping"]) != meta["provenance"]["mapping_sha256"]:
            raise ValueError("Index mapping/hash mismatch")
        if verify_sources:
            pins = meta.get("provenance", {}).get("source_pins")
            if not pins:
                raise ValueError("Index has no verified full-source provenance")
            _assert_sources_unchanged(pins)
            if _file_hash(Path(__file__)) != pins["parser"]["sha256"]:
                raise ValueError("Executing parser does not match index builder source")
        with np.load(output_dir / "index.npz", allow_pickle=False) as archive:
            def prefix(name):
                result = _Prefix.__new__(_Prefix)
                for key in ("keys", "cumulative", "base"):
                    setattr(result, key, archive[name + "_" + key].copy())
                return result
            relations = {}
            for role in ROLES:
                labels = meta["relation_labels"][role]
                relations[role] = {"labels": labels, "lookup": {name: i for i, name in enumerate(labels)},
                    **{key: archive[role + "_" + key].copy() for key in ("offsets", "works", "owners")}}
            result = cls(meta["mapping"], archive["token_work"].copy(), meta["work_groups"], relations,
                prefix("work"), {role: prefix("union_" + role) for role in ROLES},
                {role: prefix("correction_" + role) for role in ROLES},
                {key: archive["events_" + key].copy() for key in ("work", "users", "days")},
                meta["fit_cutoff"], meta["parameters"], meta["provenance"], meta["token_attributes"])
        if result.columns != meta["columns"]:
            raise ValueError("Index schema mismatch")
        return result


def build(raw_dir, metadata, *, role="sft", fit_cutoff=None, confirmed_work_ids=None,
          top_actors=100, top_directors=100, max_rows=None, min_works=3, alpha=5.0,
          min_outcome_users=3, min_related_pairs=1, cache_days=2, cache_rows=512):
    """Build CPU index, strict capped SFT/RM prefixes; max_rows is a pilot only.

    Tier quantiles and popularity/completion priors are fitted on each query's
    strict prefix, excluding the current work also from the reference pool.
    Top-N identities use catalogue credits only and do not use outcomes.
    """
    if role not in ROLE_CAPS:
        raise ValueError("role must be sft or rm")
    parameters = {"top_actors": top_actors, "top_directors": top_directors,
        "min_works": min_works, "alpha": alpha, "min_outcome_users": min_outcome_users,
        "min_related_pairs": min_related_pairs, "cache_days": cache_days, "cache_rows": cache_rows,
        "high_completion": .8, "low_completion": .2, "trend_lambda": 10.0, "trend_clip": 3.0}
    for name in ("top_actors", "top_directors", "min_works", "min_outcome_users", "min_related_pairs", "cache_days", "cache_rows"):
        value = parameters[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if name.startswith("top_") else 1):
            raise ValueError(f"Invalid {name}")
    if isinstance(alpha, bool) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive finite")
    if max_rows is not None and (not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1):
        raise ValueError("max_rows must be a positive integer")
    started = time.perf_counter()
    cap = _day(ROLE_CAPS[role])
    cutoff = cap if fit_cutoff is None else min(cap, _day(fit_cutoff))
    raw_dir = Path(raw_dir)
    source_pins_before = _source_pins(raw_dir, metadata)
    meta, mapping, token_work, groups, relations, token_attributes, missing_items = _catalogue(raw_dir, metadata, confirmed_work_ids)
    work_prefix, first, events, diagnostics = _interactions(raw_dir / "interactions.csv", mapping, token_work, cutoff, max_rows)
    unions, corrections = {}, {}
    for name in ROLES:
        unions[name], corrections[name] = _union_prefixes(relations[name], first, len(groups))
    source_pins_after = _source_pins(raw_dir, metadata)
    if source_pins_after != source_pins_before:
        raise ValueError("Source changed during people index build")
    provenance = {"contract": CONTRACT, "fit_role": role, "fit_cutoff_exclusive": date.fromordinal(cutoff).isoformat(),
        "effective_cutoff": "min(query day, explicit history cutoff, role cap, fit cutoff); strict < whole day",
        "source_unit": "retained last user-item aggregate; previous events censored; no exposure/cause claim",
        "metadata_temporal_mode": "retrospective snapshot; historical credits availability unknown",
        "person_identity": "NFKC/casefold/whitespace/per-role dedup; no aliases or external entity resolution",
        "universe_item_ids": sorted(mapping), "mapping_sha256": _hash(mapping),
        "universe_sha256": _hash(sorted(mapping)), "count_unit": "confirmed_work" if confirmed_work_ids else "card",
        "confirmed_work_map_sha256": _hash({str(k): str(v) for k, v in (confirmed_work_ids or {}).items()}),
        "items_sha256": source_pins_before["items"]["sha256"],
        "interactions_sha256": source_pins_before["interactions"]["sha256"],
        "metadata_sha256": source_pins_before["metadata"]["sha256"],
        "parser_sha256": source_pins_before["parser"]["sha256"],
        "source_provenance_version": 2, "source_pins": source_pins_before,
        "source_pins_after": source_pins_after,
        "source_scan_scope": {"fingerprints": "entire source files before and after build",
            "parsed_rows": "all CSV records" if max_rows is None else "first max_rows CSV records",
            "max_rows": max_rows, "complete_interactions_parse": max_rows is None},
        "missing_item_rows": missing_items,
        "work_attribute_conflict": "unknown, including known/missing disagreement",
        "work_user_outcomes": "mean valid edition outcomes per user as of cutoff, then equal-user mean per work",
        "tier_fit": "candidate-work LOO; per-query strict-prefix shrunk-mean score; supported actors q90/q99 linear quantile; ties same tier",
        "threshold_parameters": {"min_works": min_works, "alpha": alpha, "quantiles": [.90, .99], "method": "linear"},
        "quality_fit": "candidate-work LOO; user-work weighted prefix prior; popularity observed-work q90",
        "numeric_transforms": "raw nullable person audit; actor equal-person mean/max/median/population std/linear q25/q75/q90/top3; director popularity mean/max/median counts log1p after aggregation; related pair/user counts log1p before aggregation; masked zero fill",
        "related_units": "retained user-item pairs, separate exact distinct user unions; global pair denominator excludes candidate work",
        "external_facts": "not loaded; needs a separate provenance-bearing input",
        "specialization": {"taxonomy": "genre-semantic-v1-proposal", "genres": list(GENRES),
            "alpha": alpha, "prior": "equal-work incidence among strict-prefix observed fit-known works, current work excluded",
            "catalog": "all other U metadata works; retrospective",
            "prior_release": "year < candidate year and year < cutoff calendar year; historical metadata availability unknown",
            "empty_prior": "unsmoothed profile if n>0; unavailable if n=0; prior-only fallback explicitly masked"},
        "not_implemented": ["external ratings/awards/biography", "historical metadata known_at", "model training"],
        "pilot_only": max_rows is not None, **diagnostics}
    index = PeopleDynamicsIndex(mapping, token_work, groups, relations, work_prefix, unions, corrections, events, cutoff, parameters, provenance, token_attributes)
    provenance["index_array_bytes"] = index.nbytes
    provenance["build_seconds"] = time.perf_counter() - started
    provenance["sparse_work_days"] = len(work_prefix.keys)
    provenance["sparse_union_days"] = {name: len(prefix.keys) for name, prefix in unions.items()}
    provenance["sparse_correction_days"] = {name: len(prefix.keys) for name, prefix in corrections.items()}
    provenance["top_people_sha256"] = _hash(index.top_people)
    return index


build_index = build
