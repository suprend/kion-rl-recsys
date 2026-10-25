"""Exact other-viewer KION candidate features, with bounded CPU extraction.

This is a retained last-user/item-pair snapshot, never an exposure/event log.
All incidence counts are scoped to the fixed mapped candidate universe. Metadata
is retrospective. No outcomes, durations, negative pairs or fitted factors enter.
One global snapshot is cached; no date x catalog x user cache is materialized.
"""
from __future__ import annotations

import argparse
from array import array
import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np


CONTRACT = "kion-audience-exact-loo-v1"
FIELDS = ("age", "income", "sex", "kids_flg")
UNKNOWN = frozenset(("", "__unknown__", "nan", "none", "null"))
EXCLUDED_GENRES = UNKNOWN | frozenset(("no_genre", "популярное", "рекомендуем"))
AUG02 = date(2021, 8, 2).toordinal()
AUG09 = date(2021, 8, 9).toordinal()
AUG16 = date(2021, 8, 16).toordinal()
ROLES = ("sft_train", "sft_val", "sft_test", "rm_train", "rm_select", "rm_calibration", "rm_eval")
CORE_ARRAYS = ("users", "raw_items", "tokens", "days", "unique_users", "row_user", "user_offsets",
               "demographic_ids", "demographic_codes", "genre_vectors", "genre_empty", "genre_missing", "genre_known")


def _rss_bytes():
    """Current process resident bytes, without an optional dependency."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = (wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD)
        get_memory.restype = wintypes.BOOL
        if not get_memory(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.WorkingSetSize)
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class ResourceBudget:
    """Fail closed watchdog for a single owned CPU CLI job; samples once/sec."""
    def __init__(self, seconds=900, rss_bytes=6 * 1024 ** 3):
        self.seconds, self.limit = seconds, rss_bytes
        self.peak_rss_bytes = _rss_bytes()
        self._stop = threading.Event()

    def __enter__(self):
        self.started = time.perf_counter()
        def watch():
            while not self._stop.wait(1):
                rss = _rss_bytes()
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                elapsed = time.perf_counter() - self.started
                if elapsed > self.seconds or rss > self.limit:
                    print(json.dumps({"state": "resource_budget_exceeded", "elapsed_seconds": elapsed,
                                      "rss_bytes": rss, "max_seconds": self.seconds, "max_rss_bytes": self.limit}),
                          file=sys.stderr, flush=True)
                    os._exit(124)
        self._thread = threading.Thread(target=watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
        self._stop.set()
        self._thread.join(timeout=2)


def source_pins(raw_dir, metadata):
    """Exact whole-file pins. Full mode separately requires the canonical raw path."""
    paths = {name: Path(raw_dir) / (name + ".csv") for name in ("interactions", "items", "users")}
    paths.update(metadata=Path(metadata), builder=Path(__file__))
    return {name: {"path": path.resolve().as_posix(), "sha256": _digest(path), "size_bytes": path.stat().st_size}
            for name, path in paths.items()}


def _verify_source_pins(pins):
    for name, pin in pins.items():
        path = Path(pin["path"])
        if path.stat().st_size != pin["size_bytes"] or _digest(path) != pin["sha256"]:
            raise ValueError("Immutable source pin changed: " + name)


def _day(value):
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if 1 <= int(value) <= date.max.toordinal():
            return int(value)
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError("Invalid exact ISO/ordinal date") from error
        if parsed.isoformat() == value:
            return parsed.toordinal()
    raise ValueError("Invalid exact ISO/ordinal date")


def _ids(values, name):
    result = np.asarray(values)
    if result.ndim != 1 or result.dtype.kind not in "iu" or np.any(result < 0):
        raise ValueError(name + " must be nonnegative integer vector")
    if len(result) and int(result.max()) > np.iinfo(np.int64).max:
        raise ValueError(name + " exceeds int64")
    return result.astype(np.int64, copy=False)


def _csv(path, required):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header or len(header) != len(set(header)) or not set(required).issubset(header):
            raise ValueError("Missing/duplicate CSV headers: " + str(path))
        positions = [header.index(name) for name in required]
        for number, row in enumerate(reader, 2):
            if len(row) != len(header):
                raise ValueError(f"Malformed CSV row {number}: {path}")
            yield tuple(row[index] for index in positions)


def _raw_id(text):
    result = int(text)
    if str(result) != str(text) or not 0 <= result <= np.iinfo(np.int64).max:
        raise ValueError("Noncanonical raw ID")
    return result


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class AudienceIndex:
    """Compact retained rows plus one O(edges*genres) snapshot.

    A CSV source-row limit is an explicitly incomplete source-order slice. It is
    not sampling, a historical full snapshot, or evidence of model usefulness.
    max_retained_pairs fails closed: it never silently caps heavy viewers.
    """

    @classmethod
    def from_csv(cls, raw_dir, metadata, *, max_source_rows=None,
                 max_retained_pairs=200_000, fit_cutoff="2021-08-09",
                 alpha=10.0, coview_lambda=10.0, max_intersection_work=5_000_000):
        raw_dir = Path(raw_dir)
        meta = json.loads(Path(metadata).read_text(encoding="utf-8")) if isinstance(metadata, (str, Path)) else dict(metadata)
        if max_source_rows is not None and (type(max_source_rows) is not int or max_source_rows < 1):
            raise ValueError("max_source_rows must be a positive integer or None")
        mapping = meta["item_id_to_token"]
        if type(max_retained_pairs) is not int or max_retained_pairs < 1:
            raise ValueError("Positive retained-pair budget required")
        users, items, raw_items, days = array("q"), array("i"), array("q"), array("i")
        source_rows = unmapped = 0
        date_cache = {}
        for uid, iid, when in _csv(raw_dir / "interactions.csv", ("user_id", "item_id", "last_watch_dt")):
            if max_source_rows is not None and source_rows >= max_source_rows:
                break
            source_rows += 1
            user = _raw_id(uid)
            raw_item = _raw_id(iid)
            if when not in date_cache:
                date_cache[when] = _day(when)
            token = mapping.get(iid)
            if token is None:
                unmapped += 1
                token = 2  # Occupies the original query H63 slot, never global U.
            if len(users) >= max_retained_pairs:
                raise ValueError("Retained-pair budget exceeded; no silent truncation")
            users.append(user)
            items.append(token)
            raw_items.append(raw_item)
            days.append(date_cache[when])
        cards = {}
        seen = set()
        for iid, genres in _csv(raw_dir / "items.csv", ("item_id", "genres")):
            _raw_id(iid)
            if iid in seen:
                raise ValueError("Duplicate metadata item")
            seen.add(iid)
            if iid in mapping:
                cards[iid] = genres
        demo_ids, demo_codes = array("q"), array("h")
        lookups = [{str(label): n for n, label in enumerate(
            v for v in meta["user_categories"][field] if str(v).casefold() not in UNKNOWN)} for field in FIELDS]
        for values in _csv(raw_dir / "users.csv", ("user_id", *FIELDS)):
            uid = _raw_id(values[0])
            demo_ids.append(uid)
            for value, lookup in zip(values[1:], lookups):
                value = value.strip()
                demo_codes.append(-1 if value.casefold() in UNKNOWN else lookup.get(value, -2))
        obj = cls(users, items, days, meta, cards,
                  (np.asarray(demo_ids, np.int64), np.asarray(demo_codes, np.int16).reshape(-1, len(FIELDS))),
                  raw_item_ids=raw_items,
                  fit_cutoff=fit_cutoff, alpha=alpha, coview_lambda=coview_lambda,
                  max_retained_pairs=max_retained_pairs, max_intersection_work=max_intersection_work)
        obj.provenance.update(source_rows=source_rows, unmapped_rows=unmapped,
                              source_row_limit=max_source_rows,
                              source_scope="full_source" if max_source_rows is None else "incomplete_source_order_slice")
        obj.provenance["sources"] = {name: {"path": (raw_dir / name).resolve().as_posix(),
            "size_bytes": (raw_dir / name).stat().st_size} for name in ("interactions.csv", "users.csv", "items.csv")}
        # Full hashes are deliberately caller-owned for a bounded slice; this
        # digest pins every consumed retained row and the complete feature schema.
        digest = hashlib.sha256()
        for values in (obj.users, obj.raw_items, obj.tokens, obj.days):
            digest.update(memoryview(values).cast("B"))
        obj.provenance["retained_rows_sha256"] = digest.hexdigest()
        return obj

    def __init__(self, users, tokens, days, metadata, item_genres, demographics, *,
                 fit_cutoff="2021-08-09", alpha=10.0, coview_lambda=10.0,
                 max_retained_pairs=200_000, max_intersection_work=5_000_000, raw_item_ids=None):
        mapping = metadata.get("item_id_to_token")
        if (not isinstance(mapping, dict) or metadata.get("vocab_size", len(mapping) + 3) != len(mapping) + 3
                or any(type(v) is not int for v in mapping.values())
                or sorted(mapping.values()) != list(range(3, len(mapping) + 3))
                or metadata.get("special_tokens", {"PAD": 0, "BOS": 1, "UNK": 2}) != {"PAD": 0, "BOS": 1, "UNK": 2}):
            raise ValueError("Fixed contiguous real-item token mapping required")
        self.vocab_size = len(mapping) + 3
        self.users, self.tokens = _ids(users, "users"), _ids(tokens, "tokens")
        self.days = np.asarray(days)
        if (self.days.ndim != 1 or self.days.dtype.kind not in "iu" or len(self.users) != len(self.tokens)
                or len(self.users) != len(self.days) or len(self.users) > max_retained_pairs
                or np.any(self.tokens < 2) or np.any(self.tokens >= self.vocab_size)
                or np.any(self.days < 1) or np.any(self.days > date.max.toordinal())):
            raise ValueError("Invalid/misaligned retained arrays or pair budget")
        if not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(coview_lambda) or coview_lambda <= 0:
            raise ValueError("Positive finite smoothing constants required")
        if type(max_intersection_work) is not int or max_intersection_work < 1:
            raise ValueError("Positive intersection-work budget required")
        self.alpha, self.coview_lambda = float(alpha), float(coview_lambda)
        self.max_intersection_work = max_intersection_work
        self.fit_cutoff = _day(fit_cutoff)
        raw_item_by_token = np.zeros(self.vocab_size, dtype=np.int64)
        for iid, token in mapping.items():
            raw_item_by_token[token] = _raw_id(iid)
        if raw_item_ids is None:
            if np.any(self.tokens == 2):
                raise ValueError("Raw item IDs required for UNK history rows")
            self.raw_items = raw_item_by_token[self.tokens]
        else:
            self.raw_items = _ids(raw_item_ids, "raw item IDs")
            if len(self.raw_items) != len(self.users):
                raise ValueError("Raw item IDs must align")
            expected = np.fromiter((mapping.get(str(iid), 2) for iid in self.raw_items), np.int64, count=len(self.raw_items))
            if not np.array_equal(expected, self.tokens):
                raise ValueError("Raw item/token mapping mismatch")
        order = np.lexsort((self.raw_items, self.users))
        if len(order) > 1 and np.any((self.users[order[1:]] == self.users[order[:-1]]) &
                                     (self.raw_items[order[1:]] == self.raw_items[order[:-1]])):
            raise ValueError("Duplicate retained user/item pair")
        order = np.lexsort((self.raw_items, self.days, self.users))
        self.users, self.tokens, self.days, self.raw_items = (a[order].astype(dtype, copy=False) for a, dtype in
            ((self.users, np.int64), (self.tokens, np.int32), (self.days, np.int32), (self.raw_items, np.int64)))
        self.unique_users, self.row_user = np.unique(self.users, return_inverse=True)
        self.user_offsets = np.r_[0, np.cumsum(np.bincount(self.row_user, minlength=len(self.unique_users)))]
        self.row_user = self.row_user.astype(np.int32)
        packed_demographics = isinstance(demographics, tuple)
        self.demographic_ids = (_ids(demographics[0], "demographic IDs") if packed_demographics
                                else np.array(sorted(demographics), dtype=np.int64))
        demo_order = np.argsort(self.demographic_ids, kind="stable")
        self.demographic_ids = self.demographic_ids[demo_order]
        if len(self.demographic_ids) > 1 and np.any(np.diff(self.demographic_ids) == 0):
            raise ValueError("Duplicate metadata user")
        self.categories = {}
        self.demographic_codes = np.full((len(self.demographic_ids), len(FIELDS)), -1, dtype=np.int16)
        if packed_demographics:
            packed_codes = np.asarray(demographics[1])
            if packed_codes.shape != self.demographic_codes.shape or packed_codes.dtype.kind not in "iu":
                raise ValueError("Invalid packed demographic codes")
            self.demographic_codes[:] = packed_codes[demo_order]
        for col, field in enumerate(FIELDS):
            labels = [str(v) for v in metadata["user_categories"][field] if str(v).casefold() not in UNKNOWN]
            if not labels or len(labels) != len(set(labels)) or len(labels) > 32760:
                raise ValueError("Fixed nonempty unique demographic vocabulary required")
            self.categories[field] = labels
            lookup = {label: i for i, label in enumerate(labels)}
            if packed_demographics:
                if np.any(self.demographic_codes[:, col] < -2) or np.any(self.demographic_codes[:, col] >= len(labels)):
                    raise ValueError("Out-of-range packed demographic codes")
                continue
            for row, uid in enumerate(self.demographic_ids):
                value = str(demographics[int(uid)].get(field, "")).strip()
                self.demographic_codes[row, col] = -1 if value.casefold() in UNKNOWN else lookup.get(value, -2)
        raw_genres = {}
        for iid, token in mapping.items():
            text = str(item_genres.get(iid, item_genres.get(int(iid), "")))
            raw_genres[token] = {p.strip().casefold() for p in text.split(",") if p.strip()}
        self.genres = sorted({g for values in raw_genres.values() for g in values if g not in EXCLUDED_GENRES})
        if len(self.genres) > 512:
            raise ValueError("Genre dimension exceeds bounded CPU budget")
        genre_lookup = {name: n for n, name in enumerate(self.genres)}
        self.genre_vectors = np.zeros((self.vocab_size, len(self.genres)), dtype=np.float64)
        self.genre_empty = np.zeros(self.vocab_size, dtype=bool)
        self.genre_missing = np.ones(self.vocab_size, dtype=bool)
        for token, tags in raw_genres.items():
            indices = [genre_lookup[t] for t in tags if t in genre_lookup]
            if indices:
                self.genre_vectors[token, indices] = 1 / len(indices)
                self.genre_missing[token] = False
            elif "no_genre" in tags:
                self.genre_empty[token], self.genre_missing[token] = True, False
        self.genre_known = self.genre_vectors.sum(axis=1) > 0
        self.columns, self.groups = [], {}
        self._add("audience", ("item_n", "active_users_n", "global_pairs_n", "query_global_member",
            "no_item_support", "global_cutoff_ordinal", "query_state_cutoff_ordinal", "unknown_item"))
        for field in FIELDS:
            self._add("demography_" + field, [*("prob_" + v for v in self.categories[field]),
                *("prior_" + v for v in self.categories[field]), "match", "log_lift", "known_n",
                "unknown_field_n", "unseen_category_n", "no_user_row_n", "known_fraction", "global_known_n",
                "no_known", "no_global_known", "query_field_missing", "query_category_unseen", "query_user_row_missing"])
        self._add("taste", ("cosine", "direct_genre_cosine", "informative_viewers_n", "audience_n",
            "coverage", "other_known_items_mean", "other_known_items_min", "other_known_items_max",
            "query_history_n", "query_known_n", "query_known_fraction", "query_genre_empty_n",
            "query_genre_missing_n", "query_candidate_repeat", "no_audience_profile", "no_query_profile",
            "cosine_valid", "candidate_genre_empty", "candidate_genre_missing"))
        self._add("coview", ("mean", "max", "weight_sum", "history_n", "supported_neighbors_n",
            "history_degree_mean", "no_evidence", "no_history"))
        self.column_index = {name: n for n, name in enumerate(self.columns)}
        self._cache = None
        self.provenance = {"contract": CONTRACT, "scope": "history_restricted_to_fixed_U",
            "source_semantics": "retained_last_pair_snapshot_not_event_or_exposure_log",
            "strict_date": "last_watch_dt < global_or_query_cutoff; same-day excluded",
            "global_caps": {"SFT": "2021-08-09", "RM": "2021-08-02"},
            "query_exclusion": "all global contributions, priors, denominators, item degrees and coview intersections",
            "taste": "equal_informative_viewer_vote; candidate_card_removed_before_normalization; no_edition_merge",
            "query_history": "last_63_raw_retained_pairs_sorted_day_raw_item_id; UNK_slots_retained_without_metadata",
            "metadata": "retrospective_static_demographics_and_genres; fixed_demographic_vocabulary",
            "coview": "exact_on_demand_intersections; mean_over_unique_H63_other_cards; no_topL_or_heavy_user_cap",
            "alpha": self.alpha, "empty_prior": "fixed_uniform_over_declared_known_categories",
            "epsilon": 1e-12, "coview_lambda": self.coview_lambda, "retained_rows": len(self.users),
            "columns": self.columns, "groups": self.groups, "genres": self.genres,
            "schema_sha256": hashlib.sha256(json.dumps(self.columns, ensure_ascii=False).encode("utf-8")).hexdigest()}

    def _add(self, group, names):
        start = len(self.columns)
        self.columns.extend(group + "." + name for name in names)
        self.groups[group] = {"start": start, "stop": len(self.columns)}

    def _codes(self, ids):
        positions = np.searchsorted(self.demographic_ids, ids)
        found = positions < len(self.demographic_ids)
        if len(self.demographic_ids):
            found &= self.demographic_ids[np.minimum(positions, len(self.demographic_ids) - 1)] == ids
        codes = np.full((len(ids), len(FIELDS)), -3, dtype=np.int16)
        codes[found] = self.demographic_codes[positions[found]]
        return codes, found

    def _snapshot(self, cutoff):
        if self._cache is not None and self._cache["cutoff"] == cutoff:
            return self._cache
        # Release the previous snapshot before allocating a new one.
        self._cache = None
        selected = (self.days < cutoff) & (self.tokens >= 3)
        u, t = self.row_user[selected], self.tokens[selected]
        nusers, width = len(self.unique_users), len(self.genres)
        degrees = np.bincount(u, minlength=nusers)
        sums = np.zeros((nusers, width), dtype=np.float64)
        known_n = np.zeros(nusers, dtype=np.int32)
        for begin in range(0, len(t), 4096):
            stop = begin + 4096
            np.add.at(sums, u[begin:stop], self.genre_vectors[t[begin:stop]])
            np.add.at(known_n, u[begin:stop], self.genre_known[t[begin:stop]])
        order = np.argsort(t, kind="stable")
        item_offsets = np.r_[0, np.cumsum(np.bincount(t, minlength=self.vocab_size))]
        viewers = u[order]
        item_n = np.diff(item_offsets)
        taste_sums = np.zeros((self.vocab_size, width), dtype=np.float64)
        informative_n = np.zeros(self.vocab_size, dtype=np.int32)
        other_sum = np.zeros(self.vocab_size, dtype=np.int64)
        # Counts by informative-viewer other-history length permit exact min/max
        # after removing u; no inaccurate pre-LOO extrema are returned.
        lengths_by_item = []
        for token in range(self.vocab_size):
            audience = viewers[item_offsets[token]:item_offsets[token + 1]]
            other_n = known_n[audience] - int(self.genre_known[token])
            informative = other_n > 0
            ids, counts = np.unique(other_n[informative], return_counts=True)
            lengths_by_item.append((ids, counts))
            for begin in range(0, len(audience), 4096):
                a = audience[begin:begin + 4096]
                denominators = known_n[a] - int(self.genre_known[token])
                valid = denominators > 0
                if np.any(valid):
                    profiles = (sums[a[valid]] - self.genre_vectors[token]) / denominators[valid, None]
                    taste_sums[token] += profiles.sum(axis=0)
            informative_n[token] = int(informative.sum())
            other_sum[token] = int(other_n[informative].sum())
        # Sorted u within every item gives exact binary-search membership and
        # intersections. Metadata missing rows remain distinct from empty fields.
        for token in range(self.vocab_size):
            viewers[item_offsets[token]:item_offsets[token + 1]].sort()
        codes, found = self._codes(self.unique_users)
        demography, priors = {}, {}
        for col, field in enumerate(FIELDS):
            width = len(self.categories[field])
            table = np.zeros((self.vocab_size, width + 3), dtype=np.int64)
            for label in range(width):
                table[:, label] = np.bincount(t[codes[u, col] == label], minlength=self.vocab_size)
            for extra, code in enumerate((-1, -2, -3), width):
                table[:, extra] = np.bincount(t[codes[u, col] == code], minlength=self.vocab_size)
            demography[field] = table
            priors[field] = np.bincount(codes[(degrees > 0) & (codes[:, col] >= 0), col], minlength=width)
        self._cache = dict(cutoff=cutoff, degrees=degrees, sums=sums, known_n=known_n,
            viewers=viewers, offsets=item_offsets, item_n=item_n, taste_sums=taste_sums,
            informative_n=informative_n, other_sum=other_sum, lengths_by_item=lengths_by_item,
            codes=codes, found=found, demography=demography, priors=priors,
            active_users=int((degrees > 0).sum()), global_pairs=len(t))
        return self._cache

    def _bounds(self, query, role, history_cutoff=None):
        if role not in ROLES:
            raise ValueError("Unknown feature role")
        global_cap = AUG02 if role.startswith("rm_") else AUG09
        query_cap = AUG02 if role.startswith("rm_") else AUG09 if role == "sft_val" else AUG16 if role == "sft_test" else query
        state = min(query, query_cap)
        if history_cutoff is not None:
            state = min(state, _day(history_cutoff))
        return min(query, global_cap, self.fit_cutoff), state

    def _history(self, uid, cutoff):
        pos = int(np.searchsorted(self.unique_users, uid))
        if pos == len(self.unique_users) or self.unique_users[pos] != uid:
            return np.empty(0, np.int32), None
        left, right = self.user_offsets[pos:pos + 2]
        end = left + np.searchsorted(self.days[left:right], cutoff, side="left")
        return self.tokens[max(left, end - 63):end], pos

    @staticmethod
    def _contains(values, uid):
        if uid is None:
            return False
        p = np.searchsorted(values, uid)
        return bool(p < len(values) and values[p] == uid)

    @staticmethod
    def _cosine(a, b):
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    def features(self, item_tokens, query_dates, *, user_ids, role="sft_train", x=None, lengths=None,
                 history_cutoffs=None):
        """Explicit [query,candidate] float32 coordinates; no target labels used.

        x/lengths, when supplied, must exactly equal the raw-joined H63 with one
        BOS and trailing PAD, preventing hidden future or wider-state inputs.
        Query rows must already be bounded (<=4096); iter_features handles more.
        """
        if role not in ROLES:
            raise ValueError("Unknown feature role")
        tokens, users = _ids(item_tokens, "item_tokens"), _ids(user_ids, "user_ids")
        if len(tokens) != len(users) or len(tokens) > 4096:
            raise ValueError("Aligned query rows, maximum 4096, required")
        n = len(tokens)
        dates = np.asarray(query_dates)
        dates = np.full(n, _day(dates.item()), dtype=np.int64) if dates.ndim == 0 else np.array([_day(v) for v in dates])
        if dates.shape != (n,):
            raise ValueError("Query dates must align")
        if history_cutoffs is not None:
            cuts = np.asarray(history_cutoffs)
            cuts = np.full(n, _day(cuts.item()), dtype=np.int64) if cuts.ndim == 0 else np.array([_day(v) for v in cuts])
            if cuts.shape != (n,):
                raise ValueError("History cutoffs must align")
        else:
            cuts = [None] * n
        if (x is None) != (lengths is None):
            raise ValueError("Both x and lengths required")
        if x is not None:
            x, lengths = np.asarray(x), np.asarray(lengths)
            if (x.shape != (n, 64) or x.dtype.kind not in "iu" or lengths.shape != (n,)
                    or lengths.dtype.kind not in "iu" or np.any(lengths < 1) or np.any(lengths > 64)):
                raise ValueError("Invalid BOS+H63 input")
        result = np.zeros((n, len(self.columns)), dtype=np.float32)
        query_codes, query_found = self._codes(users)
        work = 0
        for row, (token, uid, when, cut) in enumerate(zip(tokens, users, dates, cuts)):
            def put(name, value):
                result[row, self.column_index[name]] = value
            if token < 3:
                continue
            if token >= self.vocab_size:
                put("audience.unknown_item", 1)
                continue
            global_cut, state_cut = self._bounds(int(when), role, cut)
            snapshot = self._snapshot(global_cut)
            h, upos = self._history(uid, state_cut)
            if x is not None:
                size = int(lengths[row])
                if (x[row, 0] != 1 or np.any(x[row, size:] != 0) or
                        not np.array_equal(x[row, 1:size], h)):
                    raise ValueError("Provided context differs from strict-date retained H63")
            offsets, viewers = snapshot["offsets"], snapshot["viewers"]
            audience = viewers[offsets[token]:offsets[token + 1]]
            member = self._contains(audience, upos)
            ni = len(audience) - int(member)
            active = upos is not None and snapshot["degrees"][upos] > 0
            put("audience.item_n", ni)
            put("audience.active_users_n", snapshot["active_users"] - int(active))
            put("audience.global_pairs_n", snapshot["global_pairs"] - (snapshot["degrees"][upos] if active else 0))
            put("audience.query_global_member", member)
            put("audience.no_item_support", ni == 0)
            put("audience.global_cutoff_ordinal", global_cut)
            put("audience.query_state_cutoff_ordinal", state_cut)
            for col, field in enumerate(FIELDS):
                width = len(self.categories[field])
                counts = snapshot["demography"][field][token].copy()
                code = int(query_codes[row, col])
                if member:
                    counts[code if code >= 0 else width + {-1: 0, -2: 1, -3: 2}[code]] -= 1
                prior_counts = snapshot["priors"][field].copy()
                if active and code >= 0:
                    prior_counts[code] -= 1
                global_known, known = int(prior_counts.sum()), int(counts[:width].sum())
                prior = prior_counts / global_known if global_known else np.full(width, 1 / width)
                probability = (counts[:width] + self.alpha * prior) / (known + self.alpha)
                prefix = "demography_" + field + "."
                for label, p, background in zip(self.categories[field], probability, prior):
                    put(prefix + "prob_" + label, p)
                    put(prefix + "prior_" + label, background)
                put(prefix + "match", probability[code] if code >= 0 else 0)
                put(prefix + "log_lift", np.log((probability[code] + 1e-12) / (prior[code] + 1e-12)) if code >= 0 else 0)
                for suffix, value in (("known_n", known), ("unknown_field_n", counts[width]),
                    ("unseen_category_n", counts[width + 1]), ("no_user_row_n", counts[width + 2]),
                    ("known_fraction", known / ni if ni else 0), ("global_known_n", global_known),
                    ("no_known", known == 0), ("no_global_known", global_known == 0),
                    ("query_field_missing", code == -1), ("query_category_unseen", code == -2),
                    ("query_user_row_missing", not query_found[row])):
                    put(prefix + suffix, value)
            taste_sum = snapshot["taste_sums"][token].copy()
            informative = int(snapshot["informative_n"][token])
            total_other = int(snapshot["other_sum"][token])
            self_other = 0
            if member:
                self_other = int(snapshot["known_n"][upos]) - int(self.genre_known[token])
                if self_other > 0:
                    taste_sum -= (snapshot["sums"][upos] - self.genre_vectors[token]) / self_other
                    informative -= 1
                    total_other -= self_other
            qh = h[h != token]
            q_known = self.genre_known[qh]
            q = self.genre_vectors[qh[q_known]].mean(axis=0) if q_known.any() else np.zeros(len(self.genres))
            a = taste_sum / informative if informative else np.zeros(len(self.genres))
            ids, counts = snapshot["lengths_by_item"][token]
            counts = counts.copy()
            if member and self_other > 0:
                counts[np.searchsorted(ids, self_other)] -= 1
            observed_sizes = ids[counts > 0]
            for suffix, value in (("cosine", self._cosine(q, a)),
                ("direct_genre_cosine", self._cosine(q, self.genre_vectors[token])),
                ("informative_viewers_n", informative), ("audience_n", ni),
                ("coverage", informative / ni if ni else 0),
                ("other_known_items_mean", total_other / informative if informative else 0),
                ("other_known_items_min", observed_sizes[0] if len(observed_sizes) else 0),
                ("other_known_items_max", observed_sizes[-1] if len(observed_sizes) else 0),
                ("query_history_n", len(qh)), ("query_known_n", q_known.sum()),
                ("query_known_fraction", q_known.mean() if len(qh) else 0),
                ("query_genre_empty_n", self.genre_empty[qh].sum()),
                ("query_genre_missing_n", self.genre_missing[qh].sum()),
                ("query_candidate_repeat", np.any(h == token)),
                ("no_audience_profile", informative == 0), ("no_query_profile", not q_known.any()),
                ("cosine_valid", informative > 0 and q_known.any()),
                ("candidate_genre_empty", self.genre_empty[token]), ("candidate_genre_missing", self.genre_missing[token])):
                put("taste." + suffix, value)
            history = np.unique(qh[qh >= 3])
            weights, history_degrees = [], []
            for other in history:
                neighbors = viewers[offsets[other]:offsets[other + 1]]
                other_member = self._contains(neighbors, upos)
                nj = len(neighbors) - int(other_member)
                work += len(audience) + len(neighbors)
                if work > self.max_intersection_work:
                    raise ValueError("Exact co-view intersection budget exceeded; use smaller candidate chunks")
                cij = np.intersect1d(audience, neighbors, assume_unique=True).size - int(member and other_member)
                weight = cij / np.sqrt(ni * nj) * cij / (cij + self.coview_lambda) if ni and nj and cij else 0.0
                weights.append(weight)
                history_degrees.append(nj)
            for suffix, value in (("mean", np.mean(weights) if weights else 0),
                ("max", max(weights, default=0)), ("weight_sum", sum(weights)), ("history_n", len(history)),
                ("supported_neighbors_n", np.count_nonzero(weights)),
                ("history_degree_mean", np.mean(history_degrees) if history_degrees else 0),
                ("no_evidence", not any(weights)), ("no_history", not len(history))):
                put("coview." + suffix, value)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite audience feature output")
        return result

    def iter_features(self, item_tokens, query_dates, *, user_ids, batch_size=256, **kwargs):
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("batch_size must be in [1,4096]")
        tokens, users = _ids(item_tokens, "item_tokens"), _ids(user_ids, "user_ids")
        if len(tokens) != len(users):
            raise ValueError("Query rows must align")
        dates = np.asarray(query_dates)
        if dates.ndim > 0 and dates.shape != (len(tokens),):
            raise ValueError("All query dates must align before batching")
        for key in ("x", "lengths", "history_cutoffs"):
            value = kwargs.get(key)
            if value is None:
                continue
            shape = np.shape(value)
            if key == "x" and shape != (len(tokens), 64):
                raise ValueError("All x rows must align before batching")
            if key == "lengths" and shape != (len(tokens),):
                raise ValueError("All lengths must align before batching")
            if key == "history_cutoffs" and shape and shape != (len(tokens),):
                raise ValueError("All history cutoffs must align before batching")
        for start in range(0, len(tokens), batch_size):
            stop = min(start + batch_size, len(tokens))
            options = {k: (np.asarray(v)[start:stop] if k in ("x", "lengths", "history_cutoffs") and v is not None and np.asarray(v).ndim > 0 else v)
                       for k, v in kwargs.items()}
            yield start, stop, self.features(tokens[start:stop], dates if dates.ndim == 0 else dates[start:stop],
                                           user_ids=users[start:stop], **options)

    @property
    def storage_nbytes(self):
        """NumPy payload only; Python lists/dicts and allocator RSS are additional."""
        base = sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))
        if self._cache is not None:
            for value in self._cache.values():
                if isinstance(value, np.ndarray):
                    base += value.nbytes
                elif isinstance(value, dict):
                    base += sum(v.nbytes for v in value.values() if isinstance(v, np.ndarray))
                elif isinstance(value, list):
                    base += sum(a.nbytes + b.nbytes for a, b in value)
        return base

    def save(self, directory, *, source_pins=None, measurements=None):
        """Persist compact base arrays and JSON; snapshots remain ephemeral.

        A new directory is required. Manifest is written last and is the commit
        marker. No pickle, giant pair graph or user x item allocation is stored.
        """
        directory = Path(directory)
        if directory.exists():
            raise ValueError("A new index directory is required")
        directory.mkdir(parents=True)
        pins = {} if source_pins is None else source_pins
        if pins:
            _verify_source_pins(pins)
        outputs = {}
        for name in CORE_ARRAYS:
            values = getattr(self, name)
            path = directory / (name + ".npy")
            np.save(path, values, allow_pickle=False)
            outputs[name] = {"file": path.name, "sha256": _digest(path), "shape": list(values.shape),
                             "dtype": str(values.dtype), "size_bytes": path.stat().st_size}
        if pins:
            _verify_source_pins(pins)
        manifest = {"contract": CONTRACT, "storage_contract": "kion-audience-compact-index-v1",
            "state": "completed", "provenance": self.provenance, "source_pins": pins,
            "arrays": outputs, "vocab_size": self.vocab_size, "categories": self.categories,
            "genres": self.genres, "columns": self.columns, "groups": self.groups,
            "schema_sha256": self.provenance["schema_sha256"],
            "parameters": {"fit_cutoff": self.fit_cutoff, "alpha": self.alpha,
                "coview_lambda": self.coview_lambda, "max_intersection_work": self.max_intersection_work},
            "measurements": {} if measurements is None else measurements,
            "base_array_bytes": sum(getattr(self, name).nbytes for name in CORE_ARRAYS),
            "total_disk_bytes_without_manifest": sum(v["size_bytes"] for v in outputs.values())}
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
        return manifest

    @classmethod
    def load(cls, directory, *, mmap_mode="r", verify_hashes=True):
        """Read a completed index without CSV. Stored source pins are returned,
        not silently reinterpreted; the owning provider verifies current inputs.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("contract") != CONTRACT or manifest.get("storage_contract") != "kion-audience-compact-index-v1"
                or manifest.get("state") != "completed" or set(manifest.get("arrays", {})) != set(CORE_ARRAYS)):
            raise ValueError("Invalid/incomplete persisted audience index")
        columns = manifest["columns"]
        schema_hash = hashlib.sha256(json.dumps(columns, ensure_ascii=False).encode("utf-8")).hexdigest()
        if schema_hash != manifest["schema_sha256"] or len(columns) != len(set(columns)):
            raise ValueError("Persisted audience schema mismatch")
        expected_groups = {}
        last_group = None
        for position, column in enumerate(columns):
            group = column.split(".", 1)[0]
            if group != last_group:
                if group in expected_groups:
                    raise ValueError("Persisted audience groups are not contiguous")
                expected_groups[group] = {"start": position, "stop": position + 1}
                last_group = group
            else:
                expected_groups[group]["stop"] = position + 1
        if manifest.get("groups") != expected_groups:
            raise ValueError("Persisted audience groups mismatch")
        parameters = manifest.get("parameters", {})
        if set(parameters) != {"fit_cutoff", "alpha", "coview_lambda", "max_intersection_work"}:
            raise ValueError("Unexpected persisted parameter")
        _day(parameters["fit_cutoff"])
        if (not np.isfinite(parameters["alpha"]) or parameters["alpha"] <= 0
                or not np.isfinite(parameters["coview_lambda"]) or parameters["coview_lambda"] <= 0
                or type(parameters["max_intersection_work"]) is not int or parameters["max_intersection_work"] < 1):
            raise ValueError("Invalid persisted parameter")
        # Fetch every post-load field before acquiring Windows file mappings.
        state_values = {name: manifest[name] for name in
            ("vocab_size", "categories", "genres", "columns", "groups", "provenance", "source_pins")}
        obj = cls.__new__(cls)
        # Check every file before opening any Windows memory map; rejection must
        # not leave partially loaded arrays holding files open.
        for name in CORE_ARRAYS:
            spec = manifest["arrays"][name]
            if spec["file"] != name + ".npy":
                raise ValueError("Unexpected persisted array path")
            path = directory / spec["file"]
            if path.stat().st_size != spec["size_bytes"] or (verify_hashes and _digest(path) != spec["sha256"]):
                raise ValueError("Persisted audience array hash mismatch: " + name)
        try:
            for name in CORE_ARRAYS:
                spec = manifest["arrays"][name]
                path = directory / spec["file"]
                values = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
                setattr(obj, name, values)
                if list(values.shape) != spec["shape"] or str(values.dtype) != spec["dtype"]:
                    raise ValueError("Persisted audience array schema mismatch: " + name)
        except BaseException:
            obj.close()
            raise
        for name, value in state_values.items():
            setattr(obj, name, value)
        for name, value in parameters.items():
            setattr(obj, name, value)
        obj.column_index = {name: n for n, name in enumerate(columns)}
        obj._cache = None
        obj.storage_manifest = manifest
        return obj

    def close(self):
        """Release loaded read-only mappings explicitly on Windows."""
        self._cache = None
        for name in CORE_ARRAYS:
            values = getattr(self, name, None)
            if isinstance(values, np.memmap):
                values._mmap.close()


def _run_cli(args, budget):
    if args.output.exists():
        raise ValueError("A new output directory is required")
    if args.full_source:
        expected = Path(__file__).resolve().parents[1] / "data/raw/kion/full"
        if args.raw.resolve() != expected.resolve() or not args.save_index:
            raise ValueError("Full mode requires canonical data/raw/kion/full and --save-index")
    started = time.perf_counter()
    pins = source_pins(args.raw, args.metadata) if args.save_index else None
    index = AudienceIndex.from_csv(args.raw, args.metadata, max_source_rows=None if args.full_source else args.max_source_rows,
                                   max_retained_pairs=args.max_retained_pairs)
    built = time.perf_counter()
    if args.save_index:
        manifest = index.save(args.output, source_pins=pins, measurements={"build_seconds": built - started,
            "peak_rss_bytes_through_build": max(budget.peak_rss_bytes, _rss_bytes()),
            "resource_limits": {"seconds": budget.seconds, "rss_bytes": budget.limit}})
        original_base_bytes = manifest["base_array_bytes"]
        del index
        import gc
        gc.collect()
        index = AudienceIndex.load(args.output)
        if index.storage_nbytes != original_base_bytes:
            raise ValueError("Roundtrip base-array payload mismatch")
        loaded = time.perf_counter()
    else:
        loaded = built
    # Deterministic bounded real join; repeats/cold metadata remain visible.
    take = np.arange(min(32, len(index.users)))
    snapshot_started = time.perf_counter()
    snapshot = index._snapshot(AUG09)
    snapshot_seconds = time.perf_counter() - snapshot_started
    score_started = time.perf_counter()
    rows = (np.concatenate([block for _, _, block in index.iter_features(index.tokens[take], "2021-08-09",
            user_ids=index.users[take], role="sft_val", batch_size=1)]) if len(take)
            else np.zeros((0, len(index.columns)), dtype=np.float32))
    elapsed = time.perf_counter() - score_started
    if not args.save_index:
        args.output.mkdir(parents=True)
    np.save(args.output / "candidate_features.npy", rows, allow_pickle=False)
    if pins:
        _verify_source_pins(pins)
    report = {"state": "full_index_qa_completed" if args.full_source else "bounded_slice_completed", "provenance": index.provenance,
        "metadata_sha256": _digest(args.metadata), "builder_sha256": _digest(__file__),
        "build_seconds": built - started, "save_load_seconds": loaded - built,
        "snapshot_seconds": snapshot_seconds, "score_seconds": elapsed, "scored_rows": len(rows),
        "array_payload_bytes_with_snapshot": index.storage_nbytes,
        "snapshot_pairs": snapshot["global_pairs"], "snapshot_active_users": snapshot["active_users"],
        "snapshot_cutoff": "2021-08-09", "peak_rss_bytes": max(budget.peak_rss_bytes, _rss_bytes()),
        "feature_file_sha256": _digest(args.output / "candidate_features.npy"),
        "finite": bool(np.isfinite(rows).all()), "shape": list(rows.shape),
        "mean_audience_n": float(rows[:, index.column_index["audience.item_n"]].mean()) if len(rows) else 0,
        "limitations": (["source-order slice, not full raw support"] if not args.full_source else []) +
            ["no quality/training experiment", "no user x item cache or persisted snapshots"]}
    report_path = args.output / ("qa.json" if args.save_index else "manifest.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("state", "build_seconds", "save_load_seconds", "snapshot_seconds", "score_seconds",
        "scored_rows", "snapshot_pairs", "snapshot_active_users", "peak_rss_bytes", "array_payload_bytes_with_snapshot", "shape")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-rows", type=int, default=20_000)
    parser.add_argument("--max-retained-pairs", type=int, default=200_000)
    parser.add_argument("--save-index", action="store_true")
    parser.add_argument("--full-source", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-rss-gb", type=float, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.max_seconds <= 900 or not 0 < args.max_rss_gb <= 6:
        raise ValueError("Time budget in [1,900] seconds and RSS limit <=6 GiB required")
    with ResourceBudget(args.max_seconds, int(args.max_rss_gb * 1024 ** 3)) as budget:
        _run_cli(args, budget)


if __name__ == "__main__":
    main()
