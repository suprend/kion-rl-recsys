"""CPU-only people metadata block for a frozen KION train-known catalogue.

``build(raw_dir, metadata)`` returns a float32 item matrix and its named schema.
It opens only items.csv.  No interactions, outcomes, split archives, network, or
GPU are used.  Snapshot profiles are explicitly retrospective; prior-release
profiles use strict years and are not claims of historical metadata availability.
Names are normalized strings, not resolved real-world identities.  Missing
numbers are filled with zero and accompanied by availability coordinates.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import unicodedata

import numpy as np


CONTRACT = "kion-local-pack-people-v1"
ITEM_FIELDS = ("actors", "directors", "release_year", "content_type", "genres")
GENRES = ("драмы", "комедии", "мелодрамы", "романтика", "детективы", "триллеры",
          "боевики", "криминал", "приключения", "фантастика", "фэнтези", "ужасы",
          "мистика", "военные", "исторические", "историческое", "биография",
          "вестерн", "мюзиклы", "сказки", "катастрофы", "фильм-нуар")
AGGREGATES = ("mean", "max", "median", "top3_mean")
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_people(value):
    """Comma-separated NFKC/casefold keys; no aliases, transliteration or ё/е merge."""
    return tuple(sorted({" ".join(unicodedata.normalize("NFKC", part).casefold().split())
                         for part in str(value or "").split(",") if part.strip()}))


def parse_year(value):
    try:
        year = float(value)
    except (ValueError, TypeError):
        return None
    return int(year) if math.isfinite(year) and year.is_integer() and 1 <= year <= 9999 else None


def _metadata(value):
    meta = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    mapping, size = meta.get("item_id_to_token"), meta.get("vocab_size")
    if meta.get("special_tokens", SPECIAL_TOKENS) != SPECIAL_TOKENS:
        raise ValueError("Unexpected special tokens")
    if (not isinstance(mapping, dict) or isinstance(size, bool) or not isinstance(size, int) or size < 3
            or any(not isinstance(k, str) or not k or isinstance(v, bool) or not isinstance(v, int)
                   for k, v in mapping.items())
            or sorted(mapping.values()) != list(range(3, size))):
        raise ValueError("Expected unique contiguous train-known item tokens after PAD/BOS/UNK")
    return meta, mapping


def read_items(path):
    """Read only allowed metadata fields, rejecting ambiguous item IDs."""
    items = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if (not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames))
                or not set(("item_id", *ITEM_FIELDS)).issubset(reader.fieldnames)):
            raise ValueError("Missing or duplicate people metadata headers")
        for row in reader:
            if None in row or any(row.get(key) is None for key in ("item_id", *ITEM_FIELDS)):
                raise ValueError("Malformed items.csv row")
            item_id = row["item_id"].strip()
            if not item_id or item_id in items:
                raise ValueError("Missing or duplicate item_id")
            items[item_id] = {field: row[field].strip() for field in ITEM_FIELDS}
    return items


def _prepare(items, mapping, confirmed_work_ids):
    # The supplied map is an explicitly confirmed equivalence relation.  Titles,
    # series membership and release years never infer equivalence here.
    confirmed = {} if confirmed_work_ids is None else {str(k): str(v) for k, v in confirmed_work_ids.items()}
    if any(not value.strip() for value in confirmed.values()):
        raise ValueError("Confirmed work IDs must be nonempty")
    by_key, card_to_group = {}, {}
    for item_id in sorted(mapping):
        key = ("work", confirmed[item_id]) if item_id in confirmed else ("card", item_id)
        group = by_key.setdefault(key, {"actors": set(), "directors": set(), "genres": set(),
            "years": set(), "types": set(), "cards": []})
        row = items.get(item_id, {})
        group["cards"].append(item_id)
        for role in ("actors", "directors"):
            group[role].update(parse_people(row.get(role)))
        group["genres"].update(set(parse_people(row.get("genres"))) & set(GENRES))
        group["years"].add(parse_year(row.get("release_year")))
        group["types"].add(row.get("content_type") if row.get("content_type") in ("film", "series") else None)
        card_to_group[item_id] = key
    groups, key_to_index = [], {}
    for key, group in by_key.items():
        key_to_index[key] = len(groups)
        group["year"] = next(iter(group["years"])) if len(group["years"]) == 1 else None
        group["type"] = next(iter(group["types"])) if len(group["types"]) == 1 else None
        groups.append(group)
    card_to_index = {card: key_to_index[key] for card, key in card_to_group.items()}
    indices = {role: defaultdict(set) for role in ("actors", "directors")}
    for index, group in enumerate(groups):
        for role in indices:
            for person in group[role]:
                indices[role][person].add(index)
    return groups, card_to_index, indices


class _Writer:
    def __init__(self, size):
        self.size, self.columns, self.parts, self.groups = size, [], [], {}

    def add(self, group, columns, rows):
        values = np.asarray(rows, dtype=np.float32)
        if values.shape != (self.size, len(columns)) or not np.isfinite(values).all():
            raise ValueError("Invalid people block shape/nonfinite values")
        if len(set(columns)) != len(columns) or set(columns) & set(self.columns):
            raise ValueError("Duplicate people feature columns")
        start = len(self.columns)
        self.columns.extend(columns)
        self.parts.append(values)
        self.groups[group] = {"start": start, "stop": len(self.columns), "columns": columns}


def _aggregate(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return [0.0] * 4
    return [sum(values) / len(values), values[-1], float(np.median(values)),
            sum(values[-3:]) / min(3, len(values))]


def _summary_names(prefix):
    return [prefix + "_" + name for name in (*AGGREGATES, "available_count", "available_fraction", "missing")]


def _summary(values, participants):
    known = [value for value in values if value is not None]
    return _aggregate(known) + [len(known), len(known) / participants if participants else 0.0, float(not known)]


def build_item_features(items, metadata, *, top_actors=100, top_directors=100, confirmed_work_ids=None):
    """Return item_features/columns/groups/coverage/fit/provenance without file I/O.

    ``items`` maps raw string item IDs to dictionaries containing ITEM_FIELDS.
    Only metadata.item_id_to_token members may fit dictionaries or contribute
    credits. Optional confirmed_work_ids is a trusted card-to-work map, never a
    franchise map. Conflicting/unknown edition years and formats remain unknown.
    No smoothing prior is fitted: observed incidence and missing masks are used.
    """
    meta, mapping = _metadata(metadata)
    items = {str(key): value for key, value in items.items()}
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (top_actors, top_directors)):
        raise ValueError("top-N limits must be nonnegative integers")
    size = meta["vocab_size"]
    groups, card_group, indices = _prepare(items, mapping, confirmed_work_ids)
    unit = "work" if confirmed_work_ids else "card"
    writer, fit = _Writer(size), {}
    candidate = {}
    for item_id, token in mapping.items():
        raw = items.get(item_id, {})
        current = groups[card_group[item_id]]
        candidate[token] = {"index": card_group[item_id], "year": groups[card_group[item_id]]["year"],
            "type": raw.get("content_type") if raw.get("content_type") in ("film", "series") else None,
            "genres": set(parse_people(raw.get("genres"))) & set(GENRES),
            **{role: tuple(sorted(current[role])) for role in indices}}

    for role, top_n in (("actors", top_actors), ("directors", top_directors)):
        # Frequency is number of distinct train catalogue units, never row,
        # outcome or future raw-catalog frequency. A stable key breaks ties.
        ranked = sorted(indices[role], key=lambda person: (-len(indices[role][person]), person))[:top_n]
        lookup = {person: index for index, person in enumerate(ranked)}
        fit[role] = {"requested_top_n": top_n, "dictionary": ranked,
            "frequencies": [len(indices[role][person]) for person in ranked],
            "dictionary_sha256": canonical_hash(ranked), "scope": "fixed train-known catalogue",
            "frequency_unit": unit, "tie_break": "descending distinct credits, ascending normalized key"}
        columns = [role + "_" + name for name in ("count", "missing", "multiple", "topn_covered_count", "topn_covered_fraction", "topn_uncovered_count")]
        columns += [role + "_identity:" + person for person in ranked]
        block = np.zeros((size, len(columns)), np.float32)
        for token, row in candidate.items():
            people = row[role]
            covered = sum(person in lookup for person in people)
            block[token, :6] = [len(people), not people, len(people) > 1, covered,
                                covered / len(people) if people else 0, len(people) - covered]
            for person in people:
                if person in lookup:
                    block[token, 6 + lookup[person]] = 1
        writer.add(role + "_identity", columns, block)

        metric_names = ["catalog_other_" + unit + "_count", "catalog_other_unknown_year_" + unit + "_count",
            "catalog_other_film_" + unit + "_count", "catalog_other_series_" + unit + "_count",
            "prior_" + unit + "_count", "prior_film_" + unit + "_count", "prior_series_" + unit + "_count",
            "prior_3y_" + unit + "_count", "prior_5y_" + unit + "_count", "years_since_first_prior_credit",
            "years_since_previous_prior_credit", "prior_distinct_release_years", "same_year_other_" + unit + "_count"]
        columns = [name for metric in metric_names for name in _summary_names(role + "_" + metric)]
        columns += [role + "_candidate_year_known"]
        block = np.zeros((size, len(columns)), np.float32)
        for token, row in candidate.items():
            by_metric = [[] for _ in metric_names]
            year = row["year"]
            for person in row[role]:
                others = indices[role][person] - {row["index"]}
                prior = [groups[j] for j in others if year is not None and groups[j]["year"] is not None and groups[j]["year"] < year]
                years = [work["year"] for work in prior]
                values = [len(others), sum(groups[j]["year"] is None for j in others),
                    sum(groups[j]["type"] == "film" for j in others), sum(groups[j]["type"] == "series" for j in others)]
                values += ([len(prior), sum(work["type"] == "film" for work in prior),
                    sum(work["type"] == "series" for work in prior), sum(y >= year - 3 for y in years),
                    sum(y >= year - 5 for y in years), year - min(years) if years else 0,
                    year - max(years) if years else None, len(set(years)),
                    sum(groups[j]["year"] == year for j in others)] if year is not None else [None] * 9)
                for metric, value in zip(by_metric, values):
                    metric.append(value)
            block[token] = [number for values in by_metric for number in _summary(values, len(row[role]))] + [year is not None]
        writer.add(role + "_experience", columns, block)

        for mode in ("catalog", "prior_release"):
            for kind, categories in (("genre", GENRES), ("format", ("film", "series"))):
                prefix = role + "_" + mode + "_" + kind
                columns = [prefix + "_profile_" + agg + ":" + category for agg in ("mean", "max") for category in categories]
                columns += _summary_names(prefix + "_candidate_match")
                columns += [prefix + "_" + field for field in ("eligible_credit_count_mean", "known_attribute_credit_count_mean",
                    "unknown_attribute_credit_count_mean", "known_attribute_fraction_mean", "profile_available_fraction",
                    "profile_supported3_fraction", "history_unavailable", "candidate_attribute_missing", "credits_missing")]
                block = np.zeros((size, len(columns)), np.float32)
                for token, row in candidate.items():
                    unavailable = mode == "prior_release" and row["year"] is None
                    profiles, matches, support = [], [], []
                    attrs = row["genres"] if kind == "genre" else ({row["type"]} if row["type"] else set())
                    for person in row[role]:
                        others = indices[role][person] - {row["index"]}
                        if unavailable:
                            matches.append(None)
                            continue
                        if mode == "prior_release":
                            others = {j for j in others if groups[j]["year"] is not None and groups[j]["year"] < row["year"]}
                        valid = [j for j in others if groups[j]["genres"]] if kind == "genre" else [j for j in others if groups[j]["type"]]
                        n = len(valid)
                        support.append([len(others), n, len(others) - n, n / len(others) if others else 0])
                        if not n:
                            matches.append(None)
                            continue
                        profile = np.asarray([sum(category in groups[j]["genres"] if kind == "genre" else category == groups[j]["type"] for j in valid) / n for category in categories])
                        profiles.append(profile)
                        matches.append(float(np.mean([profile[categories.index(category)] for category in attrs])) if attrs else None)
                    values = (np.stack(profiles).mean(axis=0).tolist() + np.stack(profiles).max(axis=0).tolist()) if profiles else [0.0] * (2 * len(categories))
                    values += _summary(matches, len(row[role]))
                    values += (np.mean(support, axis=0).tolist() if support else [0.0] * 4)
                    values += [len(profiles) / len(row[role]) if row[role] else 0,
                        sum(part[1] >= 3 for part in support) / len(row[role]) if row[role] else 0,
                        unavailable, not attrs, not row[role]]
                    block[token] = values
                writer.add(prefix + "_specialization", columns, block)

    # Exact source-set intersections avoid an unbounded all-catalogue pair cache.
    # Every candidate pair, including zero-history pairs, contributes equal weight.
    for mode in ("catalog", "prior_release"):
        prefix = "actor_pair_" + mode
        columns = [prefix + "_" + key for key in ("repeat_fraction", "mean_log1p_other_credits", "max_log1p_other_credits",
            "features_valid", "pair_count", "positive_pair_count", "distinct_source_count")]
        block = np.zeros((size, len(columns)), np.float32)
        for token, row in candidate.items():
            valid = len(row["actors"]) >= 2 and (mode == "catalog" or row["year"] is not None)
            counts, sources = [], set()
            if valid:
                for actor_a, actor_b in combinations(row["actors"], 2):
                    eligible = (indices["actors"][actor_a] & indices["actors"][actor_b]) - {row["index"]}
                    if mode == "prior_release":
                        eligible = {j for j in eligible if groups[j]["year"] is not None and groups[j]["year"] < row["year"]}
                    counts.append(len(eligible))
                    sources.update(eligible)
            positive = sum(count > 0 for count in counts)
            block[token] = [positive / len(counts) if counts else 0, float(np.mean(np.log1p(counts))) if counts else 0,
                math.log1p(max(counts)) if counts else 0, valid, len(row["actors"]) * (len(row["actors"]) - 1) / 2,
                positive, len(sources)]
        writer.add(prefix + "_collaboration", columns, block)

        prefix = "director_actor_" + mode
        columns = [prefix + "_director_" + stat + "_cast_" + agg for stat in ("mean", "max") for agg in AGGREGATES]
        columns += [prefix + "_" + key for key in ("cast_prior_fraction", "has_prior_collaboration", "features_valid",
            "pair_count", "positive_pair_count", "distinct_source_count", "unknown_year_source_count", "same_year_source_count")]
        block = np.zeros((size, len(columns)), np.float32)
        for token, row in candidate.items():
            valid = bool(row["actors"] and row["directors"] and (mode == "catalog" or row["year"] is not None))
            means, maxima, sources, unknown, same, positive = [], [], set(), set(), set(), 0
            if valid:
                for actor in row["actors"]:
                    counts = []
                    for director in row["directors"]:
                        others = (indices["actors"][actor] & indices["directors"][director]) - {row["index"]}
                        unknown.update(j for j in others if groups[j]["year"] is None)
                        same.update(j for j in others if row["year"] is not None and groups[j]["year"] == row["year"])
                        eligible = others if mode == "catalog" else {j for j in others if groups[j]["year"] is not None and groups[j]["year"] < row["year"]}
                        counts.append(len(eligible))
                        positive += bool(eligible)
                        sources.update(eligible)
                    means.append(sum(counts) / len(counts))
                    maxima.append(max(counts))
            block[token] = _aggregate(means) + _aggregate(maxima) + [sum(value > 0 for value in maxima) / len(maxima) if maxima else 0,
                any(value > 0 for value in maxima), valid, len(row["actors"]) * len(row["directors"]), positive,
                len(sources), len(unknown), len(same)]
        writer.add(prefix + "_collaboration", columns, block)

    values = np.concatenate(writer.parts, axis=1) if writer.parts else np.zeros((size, 0), np.float32)
    if np.any(values[:3]) or len(writer.columns) != len(set(writer.columns)):
        raise AssertionError("Special rows or feature schema corrupted")
    return {"item_features": values, "columns": writer.columns, "groups": writer.groups, "fit": fit,
        "coverage": {"train_catalog_items": len(mapping), "raw_items": len(items), "catalog_units": len(groups),
            "raw_only_items_excluded": len(set(items) - set(mapping)), "missing_item_rows": sum(key not in items for key in mapping),
            **{role + "_known_items": sum(bool(row[role]) for row in candidate.values()) for role in indices},
            **{role + "_unique_keys": len(indices[role]) for role in indices}},
        "provenance": {"builder_contract": CONTRACT, "metadata_availability": "retrospective_static_snapshot_unverified",
            "prior_release_semantics": "strict source year < candidate year; not recommendation-time availability",
            "catalog_universe": "metadata.item_id_to_token only", "catalog_item_ids": sorted(mapping),
            "catalog_item_ids_sha256": canonical_hash(sorted(mapping)), "mapping_sha256": canonical_hash(mapping),
            "name_rules": "comma; Unicode NFKC; casefold; whitespace collapse; per-role dedup; no identity/alias merge",
            "genre_taxonomy": {"version": "genre-semantic-v1-proposal", "categories": list(GENRES)},
            "profile_estimator": "unsmoothed observed incidence; n=0 unavailable; no prior fit",
            "profile_aggregation": "equal people; per-coordinate mean/max, per-person candidate match before cast aggregation",
            "candidate_attributes": "union of confirmed role credits; genres and content_type from candidate card",
            "count_unit": unit, "current_unit_excluded": True,
            "confirmed_work_map_sha256": canonical_hash({str(k): str(v) for k, v in (confirmed_work_ids or {}).items()}),
            "work_attribute_conflict": "unknown; includes known-versus-missing conflict", "special_rows": "PAD/BOS/UNK all zero",
            "missing_numbers": "zero fill plus named missing/availability/validity columns",
            "source_fields": ["items." + field for field in ITEM_FIELDS], "interactions_read": False,
            "test_read": False, "outcome_features": False, "numeric_transforms": "raw counts/years; actor-pair counts log1p only"}}


def build(raw_dir, metadata, *, top_actors=100, top_directors=100, confirmed_work_ids=None):
    """Load only local items.csv and return the in-memory people feature block."""
    path = Path(raw_dir) / "items.csv"
    before = file_hash(path)
    result = build_item_features(read_items(path), metadata, top_actors=top_actors,
        top_directors=top_directors, confirmed_work_ids=confirmed_work_ids)
    if file_hash(path) != before:
        raise ValueError("items.csv changed during people feature build")
    result["provenance"]["items_source"] = {"path": path.resolve().as_posix(), "sha256": before}
    return result
