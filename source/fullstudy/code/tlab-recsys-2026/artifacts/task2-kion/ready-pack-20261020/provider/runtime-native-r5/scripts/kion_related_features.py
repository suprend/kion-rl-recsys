"""Exact retained-pair audience unions and interval-valued related releases.

Only studios and reviewed franchise memberships live here. Actors/directors are
owned by kion_people_dynamics. Catalogue facts are retrospective; behaviors use
strict prefix dates. No event stream, historical metadata or complete filmography
is inferred from a retained last-user/item snapshot.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
import calendar
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np

CONTRACT = "kion-related-interest-v1"
ROLE_CAPS = {"sft": "2021-08-09", "rm": "2021-08-02"}
SOURCE_START = date(2021, 3, 13).toordinal()
KINDS = ("studios", "franchises")
AGGREGATES = ("mean", "max", "median", "top3_mean")
WINDOW_FIELDS = ("pairs", "distinct_viewers", "pairs_log1p", "distinct_viewers_log1p",
    "previous_pairs", "previous_distinct_viewers", "observed_works", "global_pairs",
    "global_distinct_viewers", "previous_global_pairs", "previous_global_distinct_viewers",
    "pair_growth", "viewer_growth", "growth_supported", "growth_observed", "observed_days",
    "full_window", "empty_global", "previous_empty_global")
RELEASE_FIELDS = tuple(f"{f}_{w}d" for w in (7, 30, 60, 90) for f in
    ("count_lower", "count_upper", "observed_count_upper", "observed_exact_day_count", "has_lower", "has_upper", "count_exact", "window_known")) + (
    "known_date_works", "unknown_date_works", "exact_date_works", "interval_date_works",
    "history_complete", "recency_lower_days", "recency_upper_days", "recency_bounds_known",
    "days_since_latest", "recency_exact_known", "observed_recency_lower_days", "observed_recency_upper_days",
    "observed_recency_bounds_known", "release_scope_unknown")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_pin(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.as_posix(), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _json(value):
    return json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else value


def _day(value):
    if isinstance(value, date):
        return value.toordinal()
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if 1 <= int(value) <= date.max.toordinal():
            return int(value)
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
            if parsed.isoformat() == value:
                return parsed.toordinal()
        except ValueError:
            pass
    raise ValueError("Expected ISO calendar day or ordinal")


def date_interval(raw, precision=None):
    """Inclusive calendar-day bounds; a partial date is never an exact day."""
    if raw is None or str(raw).strip().casefold() in ("", "nan", "none"):
        return None
    text = str(raw).strip()
    if re.fullmatch(r"\d{4}\.0", text):
        text = text[:4]
    if precision == "year" or (precision is None and re.fullmatch(r"\d{4}", text)):
        year = int(text[:4])
        return date(year, 1, 1).toordinal(), date(year, 12, 31).toordinal(), "year"
    if precision == "month" or (precision is None and re.fullmatch(r"\d{4}-\d{2}", text)):
        year, month = map(int, text[:7].split("-"))
        return date(year, month, 1).toordinal(), date(year, month, calendar.monthrange(year, month)[1]).toordinal(), "month"
    if precision not in (None, "day"):
        raise ValueError("Unsupported release precision")
    day = _day(text)
    return day, day, "day"


def _clock_day(value):
    if value is None:
        return None
    if isinstance(value, str) and "T" in value:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            raise ValueError("Source clock timestamp requires timezone")
        return stamp.astimezone(timezone.utc).date().toordinal()
    return _day(value)


def parse_studios(value):
    """Raw comma-delimited credit strings, exact NFKC/case/space normalization."""
    if value is None or str(value).strip().casefold() in ("", "nan", "none"):
        return ()
    return tuple(sorted({" ".join(unicodedata.normalize("NFKC", x).casefold().split())
        for x in str(value).split(",") if x.strip()}))


def _metadata(value):
    meta = _json(value)
    mapping, size = meta["item_id_to_token"], meta["vocab_size"]
    if (not isinstance(size, int) or isinstance(size, bool) or size < 3
        or any(not isinstance(k, str) or not k or not isinstance(v, int) or isinstance(v, bool) for k, v in mapping.items())
        or sorted(mapping.values()) != list(range(3, size))
        or meta.get("special_tokens", {"PAD": 0, "BOS": 1, "UNK": 2}) != {"PAD": 0, "BOS": 1, "UNK": 2}):
        raise ValueError("Expected contiguous unique train-known mapping after PAD/BOS/UNK")
    return meta


def _catalogue(meta, cards, series_annotation, reviewed_annotations, graph, confirmed_work_ids, aliases,
               map_authoritative):
    mapping, size = meta["item_id_to_token"], meta["vocab_size"]
    if map_authoritative and not set(map(str, confirmed_work_ids)).issubset(mapping):
        raise ValueError("Authoritative work map must be restricted to the exact U")
    parent = {iid: iid for iid in mapping}
    confirmed = set()
    def find(iid):
        while parent[iid] != iid:
            parent[iid] = parent[parent[iid]]
            iid = parent[iid]
        return iid
    def merge(ids):
        ids = [str(x) for x in ids if str(x) in mapping]
        if not ids:
            return
        base = min(find(x) for x in ids)
        for iid in ids:
            parent[find(iid)] = base
        confirmed.update(ids)
    grouped = defaultdict(list)
    for iid, wid in (confirmed_work_ids or {}).items():
        if not str(wid).strip():
            raise ValueError("Empty confirmed work ID")
        grouped[str(wid)].append(str(iid))
    for ids in grouped.values():
        merge(ids)
    accepted = (_json(reviewed_annotations) or {}).get("accepted_annotations", [])
    for record in accepted:
        if record.get("status") != "accepted_annotation" or not record.get("independent_review_id"):
            raise ValueError("Unreviewed annotation is not consumable")
        if not map_authoritative and record.get("relation_type") == "same_work":
            merge(record["item_ids"])
    graph = _json(graph) or {}
    if graph:
        if graph.get("contract") != "kion-franchise-progress-v1":
            raise ValueError("Unrecognized work graph contract")
        if graph.get("graph_sha256") and graph["graph_sha256"] != canonical_hash({k: v for k, v in graph.items() if k != "graph_sha256"}):
            raise ValueError("Work graph hash mismatch")
        if any(not link.get("evidence_ids") or any(e not in graph.get("evidence", {}) for e in link["evidence_ids"])
            for link in graph.get("item_work_links", [])):
            raise ValueError("Work link lacks evidence")
    graph_groups = defaultdict(list)
    for link in graph.get("item_work_links", []) if not map_authoritative else []:
        if link.get("status") == "verified" and link.get("evidence_ids"):
            graph_groups[link["work_id"]].append(str(link["item_id"]))
    for ids in graph_groups.values():
        merge(ids)
    roots = sorted({find(x) for x in mapping})
    root_id = {x: i for i, x in enumerate(roots)}
    token_work = np.full(size, -1, np.int32)
    works = [{"cards": [], "confirmed": False, "years": []} for _ in roots]
    item_entities = {kind: [set() for _ in range(size)] for kind in KINDS}
    for iid, token in mapping.items():
        wid = root_id[find(iid)]
        token_work[token] = wid
        works[wid]["cards"].append(iid)
        works[wid]["confirmed"] |= iid in confirmed
        row = cards.get(iid, {})
        item_entities["studios"][token].update(parse_studios(row.get("studios")))
        interval = date_interval(row.get("release_year"))
        if interval:
            works[wid]["years"].append(interval)
    aliases = dict(aliases or {})
    def canonical(key):
        seen = set()
        while key in aliases:
            if key in seen:
                raise ValueError("Franchise alias cycle")
            seen.add(key)
            key = aliases[key]
        return key
    def members(key, ids):
        key = canonical(key)
        for iid in ids:
            if str(iid) in mapping:
                item_entities["franchises"][mapping[str(iid)]].add(key)
    for group in (_json(series_annotation) or {}).get("series", []):
        if group.get("status") != "confirmed":
            continue
        members("series:" + group["id"], [m["item_id"] for m in group["members"] if m.get("status") == "confirmed"])
    for record in accepted:
        relation = record.get("relation_type")
        if relation in ("series_member", "universe_member"):
            family = "series" if relation == "series_member" else "universe"
            key = record["canonical_series_key" if family == "series" else "canonical_universe_key"]
            if not key or key == "unknown":
                raise ValueError("Membership lacks canonical key")
            members(family + ":" + key, record["item_ids"])
    entities, portfolios = {}, {}
    for kind in KINDS:
        work_labels = [set() for _ in works]
        for token in range(3, size):
            work_labels[int(token_work[token])].update(item_entities[kind][token])
        labels = sorted({label for labels in work_labels for label in labels})
        entities[kind] = [sorted(labels) for labels in work_labels]
        portfolios[kind] = {label: [i for i, labels in enumerate(work_labels) if label in labels] for label in labels}
    for work in works:
        # Disagreeing edition years remain a wide interval, never an arbitrary date.
        years = work.pop("years")
        work["release_interval"] = [min(x[0] for x in years), max(x[1] for x in years), "year"] if years else None
    item_entities = {kind: [sorted(labels) for labels in rows] for kind, rows in item_entities.items()}
    return token_work, works, entities, portfolios, item_entities


class RelatedInterestIndex:
    def __init__(self, metadata, cards, users, tokens, days, *, series_annotation=None,
                 reviewed_annotations=None, graph=None, confirmed_work_ids=None, franchise_aliases=None,
                 map_authoritative=False,
                 release_records=None, release_scope="catalog_original_release_year", temporal_mode="retrospective_reconstruction",
                 role="sft", fit_cutoff=None, alpha=1.0, shrinkage=10.0, growth_clip=4.0,
                 min_pairs=3, min_viewers=2, cache_days=2, provenance=None):
        meta = _metadata(metadata)
        if role not in ROLE_CAPS:
            raise ValueError("Unknown feature role")
        if not isinstance(map_authoritative, bool) or map_authoritative and not confirmed_work_ids:
            raise ValueError("Authoritative identity requires an explicit nonempty confirmed work map")
        self.fit_cutoff = _day(fit_cutoff or ROLE_CAPS[role])
        if self.fit_cutoff > _day(ROLE_CAPS[role]):
            raise ValueError("Feature fit cutoff exceeds role cap")
        if temporal_mode not in ("retrospective_reconstruction", "historical_as_of"):
            raise ValueError("Explicit temporal mode required")
        if temporal_mode == "historical_as_of":
            raise ValueError("Historical mode needs a separately clock-filtered relation catalogue; raw studios/year and current annotations are retrospective")
        if not all(np.isfinite(x) and x > 0 for x in (alpha, shrinkage, growth_clip)):
            raise ValueError("Positive finite smoothing parameters required")
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (min_pairs, min_viewers, cache_days)):
            raise ValueError("Positive integer support/cache limits required")
        self.mapping, self.vocab_size = meta["item_id_to_token"], meta["vocab_size"]
        self.parameters = dict(alpha=float(alpha), shrinkage=float(shrinkage), growth_clip=float(growth_clip),
            min_pairs=min_pairs, min_viewers=min_viewers, cache_days=cache_days)
        self.role, self.temporal_mode, self.release_scope = role, temporal_mode, release_scope
        self.token_work, self.works, self.entities, self.portfolios, self.item_entities = _catalogue(meta, cards, series_annotation,
            reviewed_annotations, graph, confirmed_work_ids, franchise_aliases, map_authoritative)
        users, tokens, days = np.asarray(users), np.asarray(tokens), np.asarray(days)
        if (any(a.ndim != 1 or a.dtype.kind not in "iu" for a in (users, tokens, days))
            or not len(users) == len(tokens) == len(days) or np.any(users < 0)
            or np.any(tokens < 0) or np.any(tokens >= self.vocab_size)
            or np.any(days < 1) or np.any(days > date.max.toordinal())):
            raise ValueError("Aligned nonnegative integer event arrays required")
        take = np.flatnonzero((tokens >= 3) & (days < self.fit_cutoff))
        # One sorted compact retained-pair stream. No catalog x date x user bank.
        order = np.lexsort((days[take], tokens[take]))
        take = take[order]
        self.users, self.tokens, self.days = (a[take].astype(dtype, copy=False) for a, dtype in
            ((users, np.int64), (tokens, np.int32), (days, np.int32)))
        self._finish_arrays()
        self.releases = [[] for _ in self.works]
        if release_scope == "catalog_original_release_year":
            for i, work in enumerate(self.works):
                if work["release_interval"]:
                    lo, hi, precision = work["release_interval"]
                    self.releases[i].append(dict(lower=lo, upper=hi, precision=precision, known_at=None,
                        source="raw.items.release_year", complete=False, scope=release_scope))
        for record in _json(release_records) or []:
            if (record.get("status") not in ("actual", "source_asserted_publication")
                or record.get("match_status") not in ("confirmed", "confirmed_accepted_work_identity")
                or not record.get("source") or not record.get("evidence_id") or not record.get("scope")
                or record.get("release_unit") != "work"):
                raise ValueError("Release requires actual confirmed work-level match, evidence, source and scope")
            if record["status"] == "source_asserted_publication" and (
                record["scope"] != "reported_work_publication_claim" or record.get("complete")
                or record.get("global_earliest_known")):
                raise ValueError("Source publication assertion cannot become a complete territorial/global release history")
            if record["scope"] != release_scope:
                continue
            iid = str(record["item_id"])
            if iid not in self.mapping:
                continue
            interval = date_interval(record.get("date"), record.get("precision"))
            if interval is None:
                continue
            lo, hi, precision = interval
            source_interval = record.get("valid_time_interval") or record.get("raw_record", {}).get("valid_time_interval")
            if source_interval:
                if not isinstance(source_interval, (list, tuple)) or len(source_interval) != 2:
                    raise ValueError("Source release time interval needs two bounds")
                lo, hi = map(_clock_day, source_interval)
                if lo is None or hi is None or lo > hi:
                    raise ValueError("Invalid source release time interval")
            self.releases[int(self.token_work[self.mapping[iid]])].append(dict(lower=lo, upper=hi,
                precision=precision, known_at=_clock_day(record.get("known_at")),
                source=record["source"], evidence_id=record["evidence_id"], complete=bool(record.get("complete", False)),
                scope=release_scope, raw_record=record))
        self._columns()
        self.provenance = dict(provenance or {})
        self.provenance.update(contract=CONTRACT, role=role, fit_cutoff=self.fit_cutoff, temporal_mode=temporal_mode,
            release_scope=release_scope, retained_pairs=len(self.users), mapping_sha256=canonical_hash(self.mapping),
            schema_sha256=canonical_hash(self.columns), parameters=self.parameters,
            source_unit="retained last user-item pair; prior visits censored; no event/exposure/causal claim",
            query_exclusion="optional user_ids removes query user from numerator and global denominator; explicit mask",
            work_exclusion="candidate and all reviewed same-work editions removed before every union and denominator",
            identity="raw normalized studio credit; reviewed named franchise; unresolved cards stay distinct",
            work_map_authoritative=map_authoritative,
            candidate_membership_policy="raw-card credits and explicit reviewed card membership only; identity map supplies exclusion/deduplication, never candidate membership",
            release_policy="inclusive day intervals; source UTC uncertainty intervals override nominal calendar date conservatively; exact recency only if complete history and exact latest; no year/month imputation")
        self.provenance["input_semantics_sha256"] = canonical_hash(dict(series_annotation=_json(series_annotation),
            reviewed_annotations=_json(reviewed_annotations), graph=_json(graph), confirmed_work_ids=confirmed_work_ids,
            map_authoritative=map_authoritative, franchise_aliases=franchise_aliases,
            release_records=_json(release_records), release_scope=release_scope))

    def _finish_arrays(self):
        self.offsets = np.searchsorted(self.tokens, np.arange(self.vocab_size + 1)).astype(np.int64)
        self.work_tokens = [[] for _ in self.works]
        for token in range(3, self.vocab_size):
            self.work_tokens[int(self.token_work[token])].append(token)
        self.event_work = self.token_work[self.tokens]
        self._global_cache, self._result_cache, self._user_rows = OrderedDict(), OrderedDict(), OrderedDict()
        self._portfolio_cache = OrderedDict()
        self._user_order = None
        self._own_window_cache, self._release_cache = OrderedDict(), OrderedDict()

    def _columns(self):
        self.columns = ["related.unknown_item", "related.query_user_excluded", "related.cutoff_ordinal"]
        for kind in KINDS:
            prefix = "related." + kind + "."
            self.columns += [prefix + x for x in ("relation_known", "listed_entities", "other_cards", "other_work_units",
                "confirmed_other_works", "unresolved_work_fraction", "candidate_work_confirmed", "has_other_work")]
            for window in (1, 7, 30):
                self.columns += [prefix + f"union_{window}d." + x for x in WINDOW_FIELDS]
                for field in ("pairs_log1p", "distinct_viewers_log1p", "pair_growth", "viewer_growth"):
                    self.columns += [prefix + f"entity_{window}d.{field}." + a for a in AGGREGATES]
            self.columns += [prefix + "release." + x for x in RELEASE_FIELDS]
        self.column_index = {c: i for i, c in enumerate(self.columns)}

    def _window(self, works, left, right, user_id=None):
        parts, observed = [], 0
        for wid in works:
            work_seen = False
            for token in self.work_tokens[wid]:
                a, b = self.offsets[token:token + 2]
                lo, hi = np.searchsorted(self.days[a:b], [left, right])
                viewers = self.users[a + lo:a + hi]
                if user_id is not None:
                    viewers = viewers[viewers != user_id]
                if len(viewers):
                    parts.append(viewers)
                    work_seen = True
            observed += work_seen
        viewers = np.concatenate(parts) if parts else np.empty(0, np.int64)
        return len(viewers), len(np.unique(viewers)), observed

    def _global(self, left, right, excluded_work, user_id):
        key = (left, right)
        if key not in self._global_cache:
            take = (self.days >= left) & (self.days < right)
            users = self.users[take]
            work = self.event_work[take]
            # A viewer belongs exclusively to one work iff min(work)==max(work).
            order = np.argsort(users, kind="stable")
            sorted_users, sorted_works = users[order], work[order]
            starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1] if len(users) else np.empty(0, np.int64)
            lo = np.minimum.reduceat(sorted_works, starts) if len(users) else np.empty(0, np.int32)
            hi = np.maximum.reduceat(sorted_works, starts) if len(users) else np.empty(0, np.int32)
            self._global_cache[key] = (len(users), len(starts), np.bincount(work, minlength=len(self.works)),
                np.bincount(lo[lo == hi], minlength=len(self.works)))
            while len(self._global_cache) > 6 * self.parameters["cache_days"]:
                self._global_cache.popitem(last=False)
        self._global_cache.move_to_end(key)
        pairs, viewers, work_pairs, exclusive = self._global_cache[key]
        pairs, viewers = pairs - int(work_pairs[excluded_work]), viewers - int(exclusive[excluded_work])
        if user_id is not None:
            own = self._own_window(user_id, left, right)
            own_pairs = sum(own.values()) - own.get(excluded_work, 0)
            pairs -= own_pairs
            viewers -= int(own_pairs > 0)
        return pairs, viewers

    def _own_window(self, uid, left, right):
        """Small per-user work counts; one lazy indirect user index, no scan per UID."""
        uid = int(uid)
        key = (uid, left, right)
        if key not in self._own_window_cache:
            if uid > np.iinfo(np.int64).max:
                # Stored IDs are int64. Searching with a uint64 scalar would
                # promote comparisons to float64 and merge adjacent large IDs.
                self._own_window_cache[key] = {}
                while len(self._own_window_cache) > 512 * self.parameters["cache_days"]:
                    self._own_window_cache.popitem(last=False)
                return self._own_window_cache[key]
            if uid not in self._user_rows:
                if self._user_order is None:
                    # Native intp avoids searchsorted recasting a 32-bit sorter
                    # on every lookup. One row permutation is the only new
                    # population-sized runtime allocation (8 bytes/row on x64).
                    self._user_order = np.argsort(self.users, kind="stable")
                exact_uid = np.int64(uid)
                lo = np.searchsorted(self.users, exact_uid, side="left", sorter=self._user_order)
                hi = np.searchsorted(self.users, exact_uid, side="right", sorter=self._user_order)
                self._user_rows[uid] = self._user_order[lo:hi]
                while len(self._user_rows) > 256:
                    self._user_rows.popitem(last=False)
            take = self._user_rows[uid]
            take = take[(self.days[take] >= left) & (self.days[take] < right)]
            work, counts = np.unique(self.event_work[take], return_counts=True)
            self._own_window_cache[key] = dict(zip(work.tolist(), counts.tolist()))
            while len(self._own_window_cache) > 512 * self.parameters["cache_days"]:
                self._own_window_cache.popitem(last=False)
        self._own_window_cache.move_to_end(key)
        return self._own_window_cache[key]

    def _window_loo(self, works, excluded_work, left, right, uid):
        """Reuse an exact portfolio union, subtract candidate-only viewers.

        An audience member is lost by removing a work iff that is the only
        distinct work they visited in this window. Min==max identifies exactly
        these members even with multiple editions and duplicate viewers.
        """
        full = tuple(sorted(set(works) | {excluded_work}))
        key = (full, left, right)
        if key not in self._portfolio_cache:
            viewer_parts, work_parts = [], []
            for wid in full:
                for token in self.work_tokens[wid]:
                    start, stop = self.offsets[token:token + 2]
                    lo, hi = np.searchsorted(self.days[start:stop], [left, right])
                    viewers = self.users[start + lo:start + hi]
                    if len(viewers):
                        viewer_parts.append(viewers)
                        work_parts.append(np.full(len(viewers), wid, np.int32))
            if viewer_parts:
                users, work = np.concatenate(viewer_parts), np.concatenate(work_parts)
                ids, counts = np.unique(work, return_counts=True)
                pair_counts = dict(zip(ids.tolist(), counts.tolist()))
                order = np.argsort(users, kind="stable")
                users, work = users[order], work[order]
                starts = np.r_[0, np.flatnonzero(users[1:] != users[:-1]) + 1]
                minimum, maximum = np.minimum.reduceat(work, starts), np.maximum.reduceat(work, starts)
                ids, counts = np.unique(minimum[minimum == maximum], return_counts=True)
                exclusive = dict(zip(ids.tolist(), counts.tolist()))
                stats = (len(users), len(starts), pair_counts, exclusive)
            else:
                stats = (0, 0, {}, {})
            self._portfolio_cache[key] = stats
            while len(self._portfolio_cache) > 2048 * self.parameters["cache_days"]:
                self._portfolio_cache.popitem(last=False)
        self._portfolio_cache.move_to_end(key)
        pairs, viewers, by_work, exclusive = self._portfolio_cache[key]
        pairs -= by_work.get(excluded_work, 0)
        viewers -= exclusive.get(excluded_work, 0)
        observed = len(by_work) - int(excluded_work in by_work)
        if uid is not None:
            own = self._own_window(uid, left, right)
            present = False
            for wid, count in own.items():
                if wid != excluded_work and wid in by_work:
                    pairs -= count
                    present = True
                    # A work disappears only when all its edition rows in
                    # this window belong to the excluded query user.
                    observed -= int(by_work[wid] == count)
            viewers -= int(present)
        return pairs, viewers, observed

    def _metrics(self, works, excluded_work, cutoff, window, uid):
        pairs, viewers, observed = self._window_loo(works, excluded_work, cutoff - window, cutoff, uid)
        pp, pv, _ = self._window_loo(works, excluded_work, cutoff - 2 * window, cutoff - window, uid)
        gp, gv = self._global(cutoff - window, cutoff, excluded_work, uid)
        gpp, gpv = self._global(cutoff - 2 * window, cutoff - window, excluded_work, uid)
        p = self.parameters
        complete = cutoff - 2 * window >= SOURCE_START
        _, support_viewers, _ = self._window_loo(works, excluded_work, cutoff - 2 * window, cutoff, uid)
        supported = pairs + pp >= p["min_pairs"] and support_viewers >= p["min_viewers"]
        def growth(c, prev, global_c, global_prev):
            if not complete or not supported or not global_c or not global_prev:
                return 0.0
            score = np.log((c + p["alpha"]) / (global_c + 2 * p["alpha"])) - np.log((prev + p["alpha"]) / (global_prev + 2 * p["alpha"]))
            support = c + prev
            return support / (support + p["shrinkage"]) * float(np.clip(score, -p["growth_clip"], p["growth_clip"]))
        return dict(zip(WINDOW_FIELDS, (pairs, viewers, np.log1p(pairs), np.log1p(viewers), pp, pv, observed,
            gp, gv, gpp, gpv, growth(pairs, pp, gp, gpp), growth(viewers, pv, gv, gpv),
            supported, complete, max(0, min(window, cutoff - SOURCE_START)), cutoff - window >= SOURCE_START,
            gp == 0, gpp == 0)))

    def _release(self, works, cutoff):
        key = (tuple(works), cutoff)
        if key in self._release_cache:
            self._release_cache.move_to_end(key)
            return self._release_cache[key]
        records = []
        for wid in works:
            eligible = [r for r in self.releases[wid] if self.temporal_mode != "historical_as_of"
                or r["known_at"] is not None and r["known_at"] < cutoff]
            # Multiple records of one work never multiply a work-level count.
            records.append(eligible)
        result = {}
        unknown = sum(not rows for rows in records)
        complete = bool(records) and all(rows and any(r["complete"] for r in rows) for rows in records)
        for window in (7, 30, 60, 90):
            lower = sum(any(r["lower"] >= cutoff - window and r["upper"] < cutoff for r in rows) for rows in records)
            observed_upper = sum(not rows or any(r["upper"] >= cutoff - window and r["lower"] < cutoff for r in rows) for rows in records)
            upper = sum(not rows or not any(r["complete"] for r in rows) or any(
                r["upper"] >= cutoff - window and r["lower"] < cutoff for r in rows) for rows in records)
            # Bounds concern listed catalogue work units, not unseen external releases.
            known = complete and lower == upper
            for field, value in (("count_lower", lower), ("count_upper", upper), ("has_lower", lower > 0),
                ("observed_count_upper", observed_upper), ("has_upper", upper > 0),
                ("observed_exact_day_count", sum(any(r["precision"] == "day" and cutoff - window <= r["lower"] and r["upper"] < cutoff for r in rows) for rows in records)),
                ("count_exact", lower if known else 0), ("window_known", known)):
                result[f"{field}_{window}d"] = value
        past = [r for rows in records for r in rows if r["upper"] < cutoff]
        latest_lo = max((r["lower"] for r in past), default=0)
        latest_hi = max((r["upper"] for r in past), default=0)
        # Intervals crossing cutoff can contain an unseen newer release.
        crossing = any(r["lower"] < cutoff <= r["upper"] for rows in records for r in rows)
        observed_bounds = bool(past) and not unknown and not crossing
        bounds_known = complete and observed_bounds
        exact = complete and bounds_known and latest_lo == latest_hi
        result.update(known_date_works=len(records) - unknown, unknown_date_works=unknown,
            exact_date_works=sum(any(r["precision"] == "day" for r in rows) for rows in records),
            interval_date_works=sum(any(r["precision"] != "day" for r in rows) for rows in records),
            history_complete=complete, recency_lower_days=cutoff - latest_hi if bounds_known else 0,
            recency_upper_days=cutoff - latest_lo if bounds_known else 0, recency_bounds_known=bounds_known,
            days_since_latest=cutoff - latest_hi if exact else 0, recency_exact_known=exact,
            observed_recency_lower_days=cutoff - latest_hi if observed_bounds else 0,
            observed_recency_upper_days=cutoff - latest_lo if observed_bounds else 0,
            observed_recency_bounds_known=observed_bounds,
            release_scope_unknown=not bool(self.release_scope))
        self._release_cache[key] = result
        while len(self._release_cache) > 2048 * self.parameters["cache_days"]:
            self._release_cache.popitem(last=False)
        return result

    def features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        tokens = np.asarray(item_tokens)
        if tokens.ndim != 1 or tokens.dtype.kind not in "iu" or np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError("One-dimensional mapped integer tokens required")
        n = len(tokens)
        if n > 4096:
            raise ValueError("At most 4096 candidate rows per call")
        def dates(values, name):
            if len(values) != n:
                raise ValueError(name + " must match tokens")
            return [_day(x) for x in values]
        queries = dates(query_dates, "query_dates")
        cuts = dates(history_cutoffs, "history_cutoffs") if history_cutoffs is not None else queries
        if user_ids is not None:
            uids = np.asarray(user_ids)
            if uids.shape != (n,) or uids.dtype.kind not in "iu" or np.any(uids < 0):
                raise ValueError("Nonnegative query user IDs must match tokens")
        else:
            uids = [None] * n
        output = np.zeros((n, len(self.columns)), np.float32)
        for row, (token, query, cut, uid) in enumerate(zip(tokens, queries, cuts, uids)):
            if token < 3:
                output[row, 0] = 1
                continue
            cutoff = min(query, cut, self.fit_cutoff)
            wid = int(self.token_work[token])
            key = (int(token), cutoff, int(uid) if uid is not None else None)
            if key not in self._result_cache:
                values = {"related.query_user_excluded": uid is not None, "related.cutoff_ordinal": cutoff}
                for kind in KINDS:
                    prefix = "related." + kind + "."
                    labels = self.item_entities[kind][int(token)]
                    portfolios = [set(self.portfolios[kind][e]) - {wid} for e in labels]
                    others = sorted(set().union(*portfolios)) if portfolios else []
                    confirmed = sum(self.works[w]["confirmed"] for w in others)
                    support = dict(relation_known=bool(labels), listed_entities=len(labels),
                        other_cards=sum(len(self.works[w]["cards"]) for w in others), other_work_units=len(others),
                        confirmed_other_works=confirmed, unresolved_work_fraction=(len(others) - confirmed) / len(others) if others else 0,
                        candidate_work_confirmed=self.works[wid]["confirmed"], has_other_work=bool(others))
                    values.update({prefix + k: v for k, v in support.items()})
                    if labels:
                        for window in (1, 7, 30):
                            metrics = self._metrics(others, wid, cutoff, window, uid)
                            values.update({prefix + f"union_{window}d." + k: v for k, v in metrics.items()})
                            per_entity = ([metrics] if len(portfolios) == 1 else
                                [self._metrics(sorted(works), wid, cutoff, window, uid) for works in portfolios])
                            for field in ("pairs_log1p", "distinct_viewers_log1p", "pair_growth", "viewer_growth"):
                                a = np.array([x[field] for x in per_entity])
                                summary = (a.mean(), a.max(), np.median(a), np.sort(a)[-3:].mean()) if len(a) else (0,) * 4
                                for agg, value in zip(AGGREGATES, summary):
                                    values[prefix + f"entity_{window}d.{field}." + agg] = value
                        values.update({prefix + "release." + k: v for k, v in self._release(others, cutoff).items()})
                self._result_cache[key] = np.array([values.get(c, 0) for c in self.columns], np.float32)
                while len(self._result_cache) > self.vocab_size * self.parameters["cache_days"]:
                    self._result_cache.popitem(last=False)
            output[row] = self._result_cache[key]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite related features")
        return output

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise ValueError("Refusing to overwrite nonempty index directory")
        arrays = {}
        for name in ("users", "tokens", "days", "token_work"):
            path = directory / (name + ".npy")
            np.save(path, getattr(self, name), allow_pickle=False)
            arrays[name] = file_pin(path)
            arrays[name]["path"] = path.name
        manifest = dict(contract=CONTRACT, state="completed", mapping=self.mapping, vocab_size=self.vocab_size,
            fit_cutoff=self.fit_cutoff, role=self.role, temporal_mode=self.temporal_mode, release_scope=self.release_scope,
            parameters=self.parameters, works=self.works, entities=self.entities, item_entities=self.item_entities,
            portfolios=self.portfolios,
            releases=self.releases, columns=self.columns, provenance=self.provenance, arrays=arrays)
        manifest["state_sha256"] = canonical_hash(manifest)
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        return manifest

    @classmethod
    def load(cls, directory, *, verify_hashes=True, mmap_mode="r"):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("contract") != CONTRACT or manifest.get("state") != "completed":
            raise ValueError("Unknown or incomplete related index")
        if manifest.get("state_sha256") != canonical_hash({k: v for k, v in manifest.items() if k != "state_sha256"}):
            raise ValueError("Related persisted state hash mismatch")
        obj = cls.__new__(cls)
        for key in ("mapping", "vocab_size", "fit_cutoff", "role", "temporal_mode", "release_scope", "parameters", "works", "entities", "portfolios", "releases", "provenance"):
            setattr(obj, key, manifest[key])
        obj.item_entities = manifest.get("item_entities")
        _metadata(dict(item_id_to_token=obj.mapping, vocab_size=obj.vocab_size))
        if obj.role not in ROLE_CAPS or not 1 <= obj.fit_cutoff <= _day(ROLE_CAPS[obj.role]):
            raise ValueError("Invalid persisted role cap")
        if obj.temporal_mode != "retrospective_reconstruction" or not all(np.isfinite(obj.parameters[x]) and obj.parameters[x] > 0 for x in ("alpha", "shrinkage", "growth_clip")):
            raise ValueError("Invalid persisted temporal mode/parameters")
        if any(isinstance(obj.parameters[x], bool) or not isinstance(obj.parameters[x], int) or obj.parameters[x] < 1
            for x in ("min_pairs", "min_viewers", "cache_days")):
            raise ValueError("Invalid persisted support/cache limits")
        for name, pin in manifest["arrays"].items():
            if name not in ("users", "tokens", "days", "token_work") or Path(pin["path"]).name != pin["path"]:
                raise ValueError("Unsafe persisted array path")
            path = directory / pin["path"]
            if verify_hashes and file_pin(path)["sha256"] != pin["sha256"]:
                raise ValueError("Related array hash mismatch")
            setattr(obj, name, np.load(path, mmap_mode=mmap_mode, allow_pickle=False))
        obj._columns()
        if obj.columns != manifest["columns"] or canonical_hash(obj.columns) != obj.provenance["schema_sha256"]:
            raise ValueError("Related schema mismatch")
        obj._finish_arrays()
        if "item_entities" not in manifest:
            # Historical r1/r2 indexes explicitly used work-inherited metadata.
            # Preserve their frozen values when a current reader loads them.
            obj.item_entities = {kind: [[] if t < 3 else obj.entities[kind][int(obj.token_work[t])]
                for t in range(obj.vocab_size)] for kind in KINDS}
        return obj

    def close(self):
        for name in ("users", "tokens", "days", "token_work"):
            a = getattr(self, name, None)
            if isinstance(a, np.memmap) and not a._mmap.closed:
                a._mmap.close()
        self._global_cache.clear()
        self._result_cache.clear()
        self._user_rows.clear()
        self._portfolio_cache.clear()
        self._own_window_cache.clear()
        self._release_cache.clear()
        self._user_order = None


def build(raw_dir, metadata, *, audience_index_dir=None, audience_index=None, **kwargs):
    """Reuse validated full raw-audience arrays; never parse a second raw history."""
    raw_dir, meta = Path(raw_dir), _metadata(metadata)
    with (raw_dir / "items.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if not {"item_id", "studios", "release_year"}.issubset(reader.fieldnames or []):
            raise ValueError("Related catalogue fields missing")
        cards = {}
        for row in reader:
            iid = row["item_id"]
            if iid in cards:
                raise ValueError("Duplicate raw item ID")
            cards[iid] = row
    pins = {"items": file_pin(raw_dir / "items.csv"), "builder": file_pin(__file__)}
    for name in ("series_annotation", "reviewed_annotations", "graph", "release_records"):
        value = kwargs.get(name)
        if isinstance(value, (str, Path)):
            pins[name] = file_pin(value)
    if isinstance(metadata, (str, Path)):
        pins["metadata"] = file_pin(metadata)
    if audience_index_dir is not None:
        directory = Path(audience_index_dir)
        manifest = _json(directory / "manifest.json")
        if manifest.get("state") != "completed" or manifest.get("storage_contract") != "kion-audience-compact-index-v1":
            raise ValueError("Completed validated audience index required")
        if manifest["vocab_size"] != meta["vocab_size"]:
            raise ValueError("Audience mapping size differs")
        source_pins = manifest.get("source_pins", {})
        if source_pins.get("items", {}).get("sha256") != pins["items"]["sha256"]:
            raise ValueError("Audience raw catalogue pin mismatch")
        if isinstance(metadata, (str, Path)) and source_pins.get("metadata", {}).get("sha256") != pins["metadata"]["sha256"]:
            raise ValueError("Audience metadata pin mismatch")
        metadata_source = source_pins.get("metadata", {})
        if not metadata_source.get("path") or file_pin(metadata_source["path"])["sha256"] != metadata_source.get("sha256"):
            raise ValueError("Pinned audience metadata source unavailable or changed")
        if _metadata(metadata_source["path"])["item_id_to_token"] != meta["item_id_to_token"]:
            raise ValueError("Audience exact item mapping differs")
        arrays = []
        for name in ("users", "tokens", "days"):
            pin = manifest["arrays"][name]
            path = directory / pin["file"]
            if path.parent.resolve() != directory.resolve() or file_pin(path)["sha256"] != pin["sha256"]:
                raise ValueError("Audience array hash/path mismatch")
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        pins["audience_manifest"] = file_pin(directory / "manifest.json")
        provenance = dict(source_pins=pins, audience_source_pins=source_pins,
            source_scope=manifest["provenance"].get("source_scope", "full_source"),
            pilot_only=manifest["provenance"].get("source_scope") != "full_source")
    elif audience_index is not None:
        if audience_index.vocab_size != meta["vocab_size"]:
            raise ValueError("Audience mapping size differs")
        arrays = [audience_index.users, audience_index.tokens, audience_index.days]
        if not hasattr(audience_index, "raw_items"):
            raise ValueError("In-memory audience requires raw item IDs to prove token identity")
        reverse = np.full(meta["vocab_size"], -1, np.int64)
        try:
            for iid, token in meta["item_id_to_token"].items():
                reverse[token] = int(iid)
        except (ValueError, OverflowError):
            raise ValueError("In-memory raw audience binding requires numeric raw IDs") from None
        mapped = audience_index.tokens >= 3
        if np.any(reverse[audience_index.tokens[mapped]] != audience_index.raw_items[mapped]):
            raise ValueError("Audience exact item mapping differs")
        if np.any(np.isin(audience_index.raw_items[~mapped], reverse[3:])):
            raise ValueError("Audience UNK row has a mapped raw item")
        provenance = dict(source_pins=pins, pilot_only=True, source_scope="in_memory_unpinned")
    else:
        raise ValueError("Supply validated audience_index_dir or explicit in-memory audience_index")
    index = RelatedInterestIndex(meta, cards, *arrays, provenance=provenance, **kwargs)
    for pin in pins.values():
        if file_pin(pin["path"])["sha256"] != pin["sha256"]:
            raise ValueError("Source changed during related-index build")
    return index
