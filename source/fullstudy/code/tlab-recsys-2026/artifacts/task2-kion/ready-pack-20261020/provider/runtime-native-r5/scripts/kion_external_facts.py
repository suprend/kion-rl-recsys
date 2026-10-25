"""Evidence-preserving external KION facts; no identity resolver or implicit HTTP.

Modern entity JSON is a retrospective source. Neither an old release year nor
a reference date establishes that its statements/mapping were known in 2021.
All numeric zeros below count observed assertions, never exhaustive reality.
Ingestion uses stdlib; full-U tensor publication imports NumPy lazily.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from urllib.request import Request, urlopen

CONTRACT = "kion-external-facts-v1"
POLICY = "kion-role-freeze-2021-v1"
MODES = ("strict_asof", "snapshot_retrospective", "retrospective_event_time",
         "historical_source_retrospective_identity")
UTC = timezone.utc
UNIT_MINUTES = {"Q7727": 1, "Q11574": 1 / 60, "Q25235": 60,
                "minute": 1, "second": 1 / 60, "hour": 60}
LANGUAGES = {"Q1860": "en", "Q7737": "ru", "Q150": "fr", "Q188": "de",
             "Q1321": "es", "Q652": "it", "Q5146": "pt", "Q5287": "ja",
             "Q9176": "ko", "Q7850": "zh", "Q256": "tr", "Q8798": "uk"}
LANGUAGE_ALIASES = {"английский": "en", "русский": "ru", "киргизский": "ky",
                    "кыргызский": "ky", "казахский": "kk", "узбекский": "uz",
                    "украинский": "uk", "якутский": "sah", "башкирский": "ba",
                    "кабардино-черкесский": "kbd", "чукотский": "ckt"}
EXTERNAL_GENRE_LABELS = {"drama": "драмы", "action": "боевики", "comedy": "комедии", "thriller": "триллеры"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _cache_path(cache, digest):
    if isinstance(cache, dict):
        if digest not in cache:
            raise FileNotFoundError("body absent from source pool: " + digest)
        return Path(cache[digest])
    return Path(cache) / (digest + ".json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _instant(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("feature boundaries must include an explicit timezone")
    return result.astimezone(UTC)


def feature_boundary(target_at, feature_role):
    target = _instant(target_at)
    if feature_role == "sft_train":
        return target
    if feature_role in {"sft_validation", "test"}:
        return datetime(2021, 8, 9, tzinfo=UTC)
    if feature_role in {"rm_fit", "rm_select", "rm_calibration", "rl"}:
        return min(target, datetime(2021, 8, 2, tzinfo=UTC))
    raise ValueError("unknown feature_role")


def date_interval(value):
    """Closed conservative precision interval, including unknown local zones.

    Year/month and date-only values without an explicit timezone can cover any
    civil timezone (-12..+14). A date is never silently converted to UTC midnight.
    """
    if not value:
        return None
    timezone_known = False
    if isinstance(value, dict):
        if value.get("calendarmodel") and not str(value["calendarmodel"]).endswith("/Q1985727"):
            return None
        precision = value.get("precision")
        raw = str(value.get("value") or value.get("time") or "")
        zone = value.get("timezone")
        timezone_known = zone == "UTC"
        if isinstance(precision, int):
            if precision < 9:
                return None  # decade/century uncertainty is not a single year
            raw = raw.lstrip("+")
            if precision <= 9:
                raw = raw[:4]
            elif precision == 10:
                raw = raw[:7]
            elif precision == 11:
                raw = raw[:10]
        value = raw
    value = str(value).lstrip("+")
    try:
        if re.fullmatch(r"\d{4}", value):
            start = datetime(int(value), 1, 1, tzinfo=UTC)
            end = datetime(int(value) + 1, 1, 1, tzinfo=UTC)
        elif re.fullmatch(r"\d{4}-\d{2}", value):
            year, month = map(int, value.split("-"))
            start = datetime(year, month, 1, tzinfo=UTC)
            end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            start = datetime.fromisoformat(value).replace(tzinfo=UTC)
            end = start + timedelta(days=1)
        else:
            instant = _instant(value)
            return instant, instant
        if not timezone_known:
            start -= timedelta(hours=14)
            end += timedelta(hours=12)
        return start, end - timedelta(microseconds=1)
    except (ValueError, OverflowError, TypeError):
        return None


def date_gate(value, boundary):
    interval = date_interval(value)
    if interval is None:
        return "date_unknown"
    if interval[1] < boundary:
        return "eligible"
    if interval[0] >= boundary:
        return "future_or_equal"
    return "date_boundary_unknown"


def _accepted(row):
    return row.get("status") in {"accepted", "manual_accepted", "accepted_by_rule", "accepted_by_source_rule"}


def temporal_gate(fact, boundary, temporal_mode, link=None):
    if temporal_mode not in MODES:
        raise ValueError("unknown temporal mode")
    if not _accepted(fact):
        return "unaccepted_evidence"
    if fact.get("conflict"):
        return "conflict"
    event_gate = date_gate(fact.get("event_at"), boundary)
    if event_gate != "eligible":
        return event_gate
    if temporal_mode in {"strict_asof", "historical_source_retrospective_identity"}:
        objects = ((fact, "source"), (link, "link")) if temporal_mode == "strict_asof" else ((fact, "source"),)
        for obj, name in objects:
            if obj is not None:
                state = date_gate(obj.get("source_known_at"), boundary)
                if state != "eligible":
                    return name + "_" + state
    return "eligible"


def normalize_language(raw, aliases=None):
    raw = " ".join(str(raw or "").casefold().split())
    table = dict(LANGUAGE_ALIASES)
    table.update(aliases or {})
    if raw == "марийский":
        return None
    return table.get(raw, raw if raw in set(LANGUAGES.values()) | set(table.values()) else None)


def normalize_runtime(quantity):
    unit = str(quantity.get("unit", "")).rsplit("/", 1)[-1]
    multiplier = UNIT_MINUTES.get(unit)
    if multiplier is None:
        return None, "unsupported_unit"
    result = {}
    try:
        for raw, normalized in (("amount", "minutes"), ("lowerBound", "lower_minutes"),
                                ("upperBound", "upper_minutes")):
            if quantity.get(raw) is not None:
                number = float(quantity[raw]) * multiplier
                if not math.isfinite(number) or number <= 0:
                    return None, "invalid_quantity"
                result[normalized] = number
        number = result["minutes"]
        if result.get("lower_minutes", number) > number or result.get("upper_minutes", number) < number:
            return None, "invalid_bounds"
        return result, None
    except (ValueError, TypeError, KeyError, OverflowError):
        return None, "invalid_quantity"


def _value(snak):
    return snak.get("datavalue", {}).get("value") if snak.get("snaktype") == "value" else None


def _claim_values(entity, prop):
    return [_value(c.get("mainsnak", {})) for c in entity.get("claims", {}).get(prop, [])
            if c.get("rank") != "deprecated" and _value(c.get("mainsnak", {})) is not None]


def _release_date(entity):
    values = _claim_values(entity, "P577")
    # Latest recorded release is a conservative work-level availability bound;
    # this does not establish an edition or resolve territorial release conflicts.
    valid = [v for v in values if date_interval(v) is not None]
    return max(valid, key=lambda v: date_interval(v)[1]) if valid and len(valid) == len(values) else None


def _qualifier(statement, prop):
    values = [_value(v) for v in statement.get("qualifiers", {}).get(prop, [])]
    values = [v for v in values if v is not None]
    return values[0] if len(values) == 1 else None


def _archived_entity(payload, qid, archive_row):
    """Validate exact revision identity/clock against the saved API response."""
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    matches = []
    for page in pages:
        if page.get("title") != qid:
            continue
        for revision in page.get("revisions", []):
            if revision.get("revid") == archive_row.get("revision_id"):
                matches.append(revision)
    if len(matches) != 1:
        raise ValueError("archived entity revision not uniquely present in response")
    revision = matches[0]
    timestamp = revision.get("timestamp")
    if timestamp != archive_row.get("revision_timestamp") or _instant(timestamp) != _instant(archive_row.get("source_evidence", {}).get("source_known_at")):
        raise ValueError("archived revision timestamp mismatch")
    content = revision.get("slots", {}).get("main", {}).get("content")
    if content is None:
        content = revision.get("*", revision.get("slots", {}).get("main", {}).get("*"))
    entity = json.loads(content) if isinstance(content, str) else content
    if not isinstance(entity, dict) or entity.get("id") != qid:
        raise ValueError("archived entity identity mismatch")
    entity = dict(entity)
    entity["lastrevid"] = revision["revid"]
    return entity


def ingest_wikidata(links, cache_dir, *, archived_entities=None):
    """Consume accepted ledger links and hash-verified raw wbgetentities bodies.

    No network, fuzzy matching, identity acceptance, or historical claims here.
    Person-role anchors require an accepted people ledger record; work credits
    alone are preserved as proposals and do not resolve local name mentions.
    """
    facts, diagnostics, loaded = [], [], {}
    for link in links.get("works", []) + links.get("people", []):
        if not _accepted(link):
            continue
        candidate = link.get("candidate", {})
        source = candidate.get("source_evidence", link.get("source_evidence", {}))
        archive_row = None
        if archived_entities is not None:
            archive_row = archived_entities.get(link.get("qid"))
            if not archive_row:
                diagnostics.append({"item_id": link.get("item_id"), "reason": "historical_entity_unavailable"})
                continue
            source = archive_row.get("source_evidence")
            if not source or archive_row.get("status") in {"unavailable", "failed"}:
                diagnostics.append({"item_id": link.get("item_id"), "work_id": link.get("work_id"), "reason": "historical_entity_unavailable"})
                continue
        digest = source.get("response_sha256")
        qid = link.get("qid")
        if not digest or not re.fullmatch(r"[0-9a-f]{64}", digest) or not re.fullmatch(r"Q\d+", str(qid)):
            diagnostics.append({"item_id": link.get("item_id"), "reason": "missing_entity_evidence"})
            continue
        path = _cache_path(cache_dir, digest)
        if digest not in loaded:
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != digest:
                raise ValueError("entity cache hash mismatch: " + path.as_posix())
            loaded[digest] = json.loads(body)
        entity = _archived_entity(loaded[digest], qid, archive_row) if archive_row else loaded[digest].get("entities", {}).get(qid)
        if not entity or entity.get("missing") is not None:
            diagnostics.append({"item_id": link.get("item_id"), "reason": "entity_missing_from_cache"})
            continue
        subject_type = "person" if link in links.get("people", []) else "work"
        subject = link.get("person_id") if subject_type == "person" else link.get("work_id")
        for prop in ("P2047", "P364", "P407", "P166", "P1411", "P57", "P161", "P577"):
            for statement in entity.get("claims", {}).get(prop, []):
                raw = _value(statement.get("mainsnak", {}))
                if statement.get("rank") == "deprecated" or raw is None:
                    diagnostics.append({"subject_id": subject, "property": prop, "reason": "deprecated_or_no_value"})
                    continue
                base = {"evidence_id": statement.get("id") or canonical_hash(statement),
                    "subject_id": subject, "subject_type": subject_type, "item_id": link.get("item_id"),
                    "source": "wikidata", "source_id": qid, "property": prop,
                    "source_revision": entity.get("lastrevid", source.get("entity_revision")),
                    "source_known_at": source.get("source_known_at"), "retrieved_at": source.get("retrieved_at"),
                    "statement_source_known_at": source.get("source_known_at"),
                    "response_sha256": digest, "evidence_url": "https://www.wikidata.org/wiki/" + qid,
                    "license": "CC0", "raw_statement": statement, "status": "accepted",
                    "event_at": _release_date(entity), "work_id": link.get("work_id"),
                    "source_snapshot_kind": "archived_revision" if archive_row else "current_snapshot",
                    "identity_source_known_at": link.get("source_known_at"),
                    "identity_temporal_mode": "historical" if link.get("source_known_at") else "snapshot_retrospective"}
                if subject_type == "person":
                    base.update({"person_link_accepted": True, "person_link_known_at": link.get("source_known_at")})
                if prop == "P2047":
                    normalized, reason = normalize_runtime(raw)
                    facts.append(base | {"kind": "runtime", "value": normalized,
                        "runtime_scope": "film" if link.get("content_type") == "film" else "work_unspecified",
                        "edition_id": link.get("edition_id"), "reason": reason})
                elif prop in {"P364", "P407"}:
                    language_qid = raw.get("id") if isinstance(raw, dict) else None
                    facts.append(base | {"kind": "language", "family": "original" if prop == "P364" else "work_unspecified",
                        "language_code": LANGUAGES.get(language_qid), "raw_language": raw,
                        "scope": "work", "complete": False})
                elif prop in {"P166", "P1411"}:
                    awarded_work = _qualifier(statement, "P1686")
                    facts.append(base | {"kind": "award", "award_id": raw.get("id") if isinstance(raw, dict) else None,
                        "award_status": "won" if prop == "P166" else "nominated", "event_at": _qualifier(statement, "P585"),
                        "recipient_id": subject, "recipient_type": subject_type,
                        "awarded_work_id": "wd:" + awarded_work["id"] if isinstance(awarded_work, dict) else subject if subject_type == "work" else None,
                        "family": None, "category_role": None, "namespace": "unclassified",
                        "assertion_kind": "metadata_assertion", "attribution_verified": True,
                        "recipient_resolution": "accepted_entity_subject"})
                elif prop == "P577":
                    interval = date_interval(raw)
                    precision = raw.get("precision") if isinstance(raw, dict) else None
                    raw_time = str(raw.get("time", "")).lstrip("+") if isinstance(raw, dict) else ""
                    size = {9: 4, 10: 7, 11: 10}.get(precision)
                    qualifiers = statement.get("qualifiers", {})
                    def qualifier_ids(property_id):
                        return [value["id"] for snak in qualifiers.get(property_id, [])
                            if isinstance(value := _value(snak), dict) and value.get("id")]
                    facts.append(base | {"kind": "release", "event_at": raw,
                        "release_at": raw, "date": raw_time[:size] if size and interval else None,
                        "precision": {9: "year", 10: "month", 11: "day"}.get(precision),
                        "status": "accepted", "publication_status": "source_asserted_publication",
                        "match_status": "confirmed_accepted_work_identity", "release_unit": "work",
                        "scope": "reported_work_publication_claim", "complete": False,
                        "place_of_publication_qids": qualifier_ids("P291"), "jurisdiction_qids": qualifier_ids("P1001"),
                        "applies_to_qids": qualifier_ids("P518"), "distribution_qids": qualifier_ids("P437"),
                        "qualifiers": qualifiers, "rank": statement.get("rank"), "references": statement.get("references", []),
                        "source_production_company_statements": entity.get("claims", {}).get("P272", []),
                        "source_part_of_series_statements": entity.get("claims", {}).get("P179", []),
                        "valid_time_interval": [value.isoformat() for value in interval] if interval else None,
                        "global_earliest_known": False, "distribution_scope_known": False})
                else:
                    source_types = {v.get("id") for v in _claim_values(entity, "P31") if isinstance(v, dict)}
                    work_type = "film" if "Q11424" in source_types else "series" if "Q5398426" in source_types else None
                    if not archive_row:
                        work_type = link.get("content_type")
                    direct_formats = {name for qid, name in (("Q11424", "film"), ("Q5398426", "series")) if qid in source_types}
                    facts.append(base | {"kind": "credit", "person_id": "wd:" + raw["id"],
                        "role": "director" if prop == "P57" else "actor", "person_link_accepted": False,
                        "kion_membership": link.get("kion_membership", "inside" if link.get("item_id") is not None else "unknown"), "work_type": work_type,
                        "release_year": _year(_release_date(entity)), "genres": None,
                        "raw_genre_qids": sorted({value["id"] for value in _claim_values(entity, "P136") if isinstance(value, dict) and value.get("id")}),
                        "raw_genre_statements": entity.get("claims", {}).get("P136", []),
                        "formats": sorted(direct_formats) if len(direct_formats) == 1 else None,
                        "format_mapping_version": "external-format-direct-P31-v1",
                        "format_mapping_rule": "nondeprecated direct P31=Q11424 film or Q5398426 series; no inheritance; two roots conflict",
                        "raw_format_statements": entity.get("claims", {}).get("P31", [])})
    accepted_credits = {(row.get("work_id"), row.get("person_id"), row.get("role")): row
        for row in links.get("credits", []) if _accepted(row)}
    for fact in facts:
        if fact["kind"] == "credit":
            decision = accepted_credits.get((fact.get("work_id"), fact.get("person_id"), fact.get("role")))
            if decision:
                fact["person_link_accepted"] = True
                fact["credit_decision"] = decision
                fact["credit_identity_known_at"] = decision.get("source_known_at")
    return {"contract": CONTRACT, "facts": facts, "diagnostics": diagnostics}


def ingest_archived_wikidata(links, archive, cache_dir):
    return ingest_wikidata(links, cache_dir, archived_entities=archive.get("entities", {}))


def ingest_credit_corpus(corpus, cache_dir, *, archive=None):
    """Read an immutable bounded-query ledger, then verify its work statements.

    A discovery result is not a credit by itself: ingest_wikidata joins accepted
    decisions only where the fetched work actually asserts that person and role.
    Collection completeness refers to the saved bounded query, never a career.
    """
    result = ingest_archived_wikidata(corpus, archive, cache_dir) if archive is not None else ingest_wikidata(corpus, cache_dir)
    result["facts"] = [f for f in result["facts"] if f.get("kind") == "credit" and f.get("person_link_accepted")]
    collections, receipts = {}, {}
    for key, collection in corpus.get("collections", {}).items():
        verified = []
        for receipt in collection.get("query_receipts", []):
            digest = receipt.get("response_sha256")
            if receipt.get("status") != "ok" or receipt.get("http_status") != 200 or not digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            body = _cache_path(cache_dir, digest).read_bytes()
            if hashlib.sha256(body).hexdigest() != digest:
                raise ValueError("credit query receipt cache hash mismatch")
            if not isinstance(json.loads(body).get("results", {}).get("bindings"), list):
                raise ValueError("credit query cache is not a SPARQL result")
            verified.append(receipt)
            receipts[digest] = receipt
        collections[key] = dict(collection)
        if collection.get("query_receipts"):
            collections[key]["source_available"] = bool(verified)
        if archive is not None:
            collections[key]["discovery_query_complete"] = collection.get("source_collection_complete_for_query", False)
            collections[key]["source_collection_complete_for_query"] = False
            collections[key]["historical_collection_status"] = corpus.get("historical_collection_status", "partial")
    result["collections"] = collections
    result["query_manifest"] = {"contract": corpus.get("contract"), "query_receipts": list(receipts.values()),
        "work_n": len(corpus.get("works", [])), "accepted_credit_n": len(result["facts"]),
        "scope": "bounded observed corpus, not full filmography"}
    return result


def apply_award_taxonomy(assertions, taxonomy):
    """Apply explicitly reviewed category metadata, never infer role from cast.

    Entries require accepted status and evidence URL; unknown IDs remain raw and
    unclassified. Taxonomy evidence availability participates in strict gating.
    """
    output = []
    for fact in assertions:
        row = dict(fact)
        entry = taxonomy.get(fact.get("award_id")) if fact.get("kind") == "award" else None
        if entry and _accepted(entry) and entry.get("evidence_url") and entry.get("family"):
            row.update({key: entry.get(key) for key in ("family", "category_role", "namespace", "festival_verified")})
            row["taxonomy_evidence"] = entry
            row["taxonomy_source_known_at"] = entry.get("source_known_at")
            caveats = entry.get("recipient_assertion_caveats", {}).get(fact.get("recipient_id"), [])
            row["source_assertion_caveats"] = [note for note in caveats
                if note.get("property") in {None, fact.get("property")}]
            if date_interval(row.get("source_known_at")) and date_interval(entry.get("source_known_at")):
                row["source_known_at"] = max((row["source_known_at"], entry["source_known_at"]), key=lambda value: date_interval(value)[1])
            else:
                row["source_known_at"] = None
        output.append(row)
    return output


def apply_external_genre_mapping(assertions, mapping):
    """Only reviewed finite source-QID translations, never genre inheritance."""
    entries = mapping.get("entries", {}) if _accepted(mapping) else {}
    output = []
    for fact in assertions:
        row = dict(fact)
        if fact.get("kind") == "credit":
            raw = fact.get("raw_genre_qids", [])
            selected = [entries[qid] for qid in raw if qid in entries and _accepted(entries[qid])
                and entries[qid].get("target_label") in EXTERNAL_GENRE_LABELS.values() and entries[qid].get("evidence_url")]
            row["genres"] = sorted({entry["target_label"] for entry in selected}) or None
            row["unmapped_genre_qids"] = [qid for qid in raw if not any(entry is entries.get(qid) for entry in selected)]
            clocks = [fact.get("source_known_at")] + [entry.get("source_known_at") for entry in selected]
            row["genres_source_known_at"] = max(clocks, key=lambda date: date_interval(date)[1]) if selected and all(date_interval(date) for date in clocks) else None
            label_clocks = {}
            for label in row["genres"] or []:
                dated = [max((fact["source_known_at"], entry["source_known_at"]), key=lambda date: date_interval(date)[1])
                    for entry in selected if entry["target_label"] == label and date_interval(fact.get("source_known_at")) and date_interval(entry.get("source_known_at"))]
                label_clocks[label] = min(dated, key=lambda date: date_interval(date)[1]) if dated else None
            row["genres_label_source_known_at"] = label_clocks
            row["genre_mapping_version"] = mapping.get("version")
            row["genre_mapping_status"] = "known_partial" if selected else "unknown"
            row["genre_mapping_evidence"] = selected
        output.append(row)
    return output


def _year(value):
    if isinstance(value, (str, int, float)) and re.fullmatch(r"\d{4}(?:\.0+)?", str(value)):
        return int(float(value))
    interval = date_interval(value)
    if interval is None:
        return None
    raw = value.get("value", value.get("time")) if isinstance(value, dict) else value
    try:
        return int(str(raw).lstrip("+")[:4])
    except (ValueError, TypeError):
        return None


def _unknown(reason):
    return {"value": None, "missing": 1, "reason": reason, "support": 0, "evidence_ids": []}


def _eligible(facts, boundary, mode, link):
    accepted, rejected = [], []
    for fact in facts:
        reason = temporal_gate(fact, boundary, mode, link)
        if reason == "eligible":
            accepted.append(fact)
        else:
            rejected.append({"evidence_id": fact.get("evidence_id"), "reason": reason})
    return accepted, rejected


def aggregate_runtime(facts, link, boundary, mode):
    if not link or not _accepted(link):
        return _unknown("unaccepted_work_link")
    eligible, rejected = _eligible(facts, boundary, mode, link)
    valid = [f for f in eligible if f.get("value") and not f.get("reason")]
    scope = link.get("tv_scope", {})
    if scope and scope.get("parent_work_id") != link.get("work_id"):
        return _unknown("scope_parent_mismatch")
    episodes = [f for f in valid if f.get("runtime_scope") == "episode" and f.get("scope_verified")
        and scope.get("scope_match_status") == "accepted"
        and scope.get("scope_kind") in {"whole_series", "season", "episode_set", "miniseries"}
        and f.get("airdate_kind") == "confirmed_release"
        and (mode != "strict_asof" or date_gate(scope.get("source_known_at"), boundary) == "eligible")
        and f.get("parent_work_id") == scope.get("parent_work_id")
        and (scope.get("scope_kind") != "season" or f.get("season_id") in scope.get("season_ids", []))
        and (scope.get("scope_kind") != "episode_set" or f.get("episode_id") in scope.get("episode_ids", []))
        and (not f.get("special") or scope.get("include_specials", False))]
    if episodes:
        unique = {}
        for fact in episodes:
            key = fact.get("episode_id")
            if key is None:
                return _unknown("episode_id_missing")
            if key in unique and unique[key]["value"] != fact["value"]:
                return _unknown("conflicting_episode_runtime")
            unique[key] = fact
        values = [f["value"]["minutes"] for f in unique.values()]
        # Denominator supplied only by a complete, same-cutoff eligible episode manifest.
        count = link.get("eligible_episode_count") if link.get("episode_count_boundary") == boundary.isoformat() else None
        return {"value": statistics.median(values), "missing": 0, "reason": None,
            "lower_minutes": min(values), "upper_minutes": max(values), "support": len(values),
            "duration_is_estimate": True, "episode_sample_n": len(values), "episode_count": count,
            "episode_coverage": len(values) / count if count and count >= len(values) else None,
            "aggregation_method": "observed_episode_median", "evidence_ids": [f["evidence_id"] for f in unique.values()],
            "episode_scope_manifest": [{"episode_id": k, "release_at": f["event_at"]} for k, f in unique.items()],
            "excluded": rejected}
    if link.get("edition_relation") not in {"verified_same_runtime", "verified_identical"}:
        return _unknown("edition_runtime_unverified") | {"excluded": rejected, "raw_support": len(valid)}
    if mode in {"strict_asof", "historical_source_retrospective_identity"} and date_gate(link.get("edition_source_known_at"), boundary) != "eligible":
        return _unknown("edition_runtime_not_historical") | {"excluded": rejected}
    valid = [f for f in valid if f.get("runtime_scope") == "film" and f.get("edition_id") == link.get("edition_id")]
    if not valid:
        return _unknown("no_eligible_runtime") | {"excluded": rejected}
    if len({canonical_hash(f["value"]) for f in valid}) > 1:
        return _unknown("conflicting_runtime")
    return {"value": valid[0]["value"]["minutes"], "missing": 0, "reason": None,
        "support": len(valid), "duration_is_estimate": False, "aggregation_method": "edition_assertion",
        "lower_minutes": valid[0]["value"].get("lower_minutes"),
        "upper_minutes": valid[0]["value"].get("upper_minutes"),
        "evidence_ids": [f["evidence_id"] for f in valid], "excluded": rejected}


WORK_RUNTIME_PROXY_FAMILY = {"scope": "reported_work_duration_claims", "source_property": "wikidata:P2047",
    "semantic_type": "work_metadata_proxy", "unit": "minute", "selected_edition_runtime": False,
    "aggregation_method": "distribution_of_source_assertions", "preferred_version_selected": False,
    "eligible_use": "separate_work_metadata_feature", "completion_normalization_allowed": False,
    "historical_availability": "requires_eligible_source_revision_and_identity_policy"}


def aggregate_work_runtime_proxy(facts, link, boundary, mode):
    """Summarize film-work claims without selecting or validating a KION cut."""
    result = {"min_minutes": None, "median_minutes": None, "max_minutes": None, "assertion_n": None,
        "distinct_value_n": None, "disagreement": None, "missing": 1, "edition_verified": None,
        "reason": "unaccepted_film_work", "evidence_ids": [], "assertion_records": [],
        "family": WORK_RUNTIME_PROXY_FAMILY, "temporal_mode": mode}
    if not link or not _accepted(link) or link.get("content_type") != "film":
        return result
    if mode != "strict_asof" or date_gate(link.get("source_known_at"), boundary) == "eligible":
        verified = link.get("edition_relation") in {"verified_same_runtime", "verified_identical"}
        edition_available = mode not in {"strict_asof", "historical_source_retrospective_identity"} or date_gate(link.get("edition_source_known_at"), boundary) == "eligible"
        result["edition_verified"] = int(verified) if not verified or edition_available else None
    eligible, rejected = _eligible([fact for fact in facts if fact.get("runtime_scope") == "film"
        and fact.get("source") == "wikidata" and fact.get("property") == "P2047"
        and fact.get("subject_type", "work") == "work" and fact.get("subject_id", fact.get("work_id")) == link.get("work_id")],
        boundary, mode, link)
    valid = {fact["evidence_id"]: fact for fact in eligible if fact.get("value") and not fact.get("reason")}
    rejected += [{"evidence_id": fact.get("evidence_id"), "reason": fact.get("reason") or "invalid_quantity"}
        for fact in eligible if not fact.get("value") or fact.get("reason")]
    if not valid:
        return result | {"reason": "no_eligible_work_runtime_assertions", "excluded": rejected}
    values = [fact["value"]["minutes"] for fact in valid.values()]
    distinct = len(set(values))
    return result | {"min_minutes": min(values), "median_minutes": statistics.median(values), "max_minutes": max(values),
        "assertion_n": len(values), "distinct_value_n": distinct, "disagreement": int(distinct > 1),
        "missing": 0, "reason": "multiple_source_values_unresolved" if distinct > 1 else "source_work_proxy",
        "evidence_ids": list(valid), "assertion_records": list(valid.values()), "excluded": rejected,
        "source_availability_known": all(date_gate(fact.get("source_known_at"), boundary) == "eligible" for fact in valid.values())}


def aggregate_languages(facts, link, boundary, mode):
    result = {}
    for family in ("original", "spoken", "edition_audio", "subtitles"):
        selected = [f for f in facts if f.get("family") == family]
        if family in {"edition_audio", "subtitles"}:
            selected = [f for f in selected if link and f.get("edition_id") == link.get("edition_id") and f.get("scope_verified")]
            if mode in {"strict_asof", "historical_source_retrospective_identity"} and (not link or date_gate(link.get("edition_source_known_at"), boundary) != "eligible"):
                selected = []
        eligible, rejected = _eligible(selected, boundary, mode, link)
        codes = sorted({f["language_code"] for f in eligible if f.get("language_code")})
        conflicts = [f for f in selected if f.get("conflict") and temporal_gate(f | {"conflict": False}, boundary, mode, link) == "eligible"]
        complete = any(f.get("complete") for f in eligible) and all(f.get("language_code") or f.get("explicit_empty") for f in eligible)
        status = "conflict" if conflicts else "known_complete" if complete else "known_partial" if codes else "unknown"
        result[family] = {"codes": [] if conflicts else codes, "status": status,
            "missing": int(status in {"unknown", "conflict"}), "support": len(eligible),
            "coverage": {"scope": "edition" if family in {"edition_audio", "subtitles"} else "work",
                         "complete": complete, "temporal_mode": mode},
            "evidence_ids": [f["evidence_id"] for f in eligible], "excluded": rejected,
            "unmapped_count": sum(not f.get("language_code") and not f.get("explicit_empty") for f in eligible)}
    return result


def aggregate_awards(facts, link, boundary, mode, cast=None, directors=None):
    known_people = set(cast or []) | set(directors or [])
    facts = [f for f in facts if link and _accepted(link) and (
        f.get("recipient_type") == "work" and f.get("recipient_id") == link.get("work_id")
        or f.get("recipient_type") == "person" and f.get("recipient_id") in known_people
        and f.get("person_link_accepted") and (mode != "strict_asof"
            or date_gate(f.get("person_link_known_at"), boundary) == "eligible"))]
    eligible, rejected = _eligible(facts, boundary, mode, link)
    events = {}
    unresolved = 0
    for fact in eligible:
        if not fact.get("attribution_verified") or fact.get("namespace") != "positive" or not fact.get("family"):
            unresolved += 1
            continue
        key = (fact.get("recipient_id"), fact.get("event_id"), _year(fact.get("event_at")),
               fact.get("award_id"), fact.get("awarded_work_id"), fact.get("award_status"))
        events.setdefault(key, fact)
    claims = list(events.values())
    result = {"support": len(claims), "unresolved_attribution": unresolved,
              "missing": int(not claims), "evidence_ids": [f["evidence_id"] for f in claims], "excluded": rejected,
              "assertion_semantics": "source_assertion_not_verified_event_date",
              "source_assertion_caveats": [{"evidence_id": f["evidence_id"], "recipient_id": f.get("recipient_id"),
                  "raw_event_at": f.get("event_at"), "notes": f["source_assertion_caveats"]}
                  for f in claims if f.get("source_assertion_caveats")]}
    for status in ("won", "nominated"):
        current = [f for f in claims if f.get("recipient_type") == "work" and link and f.get("awarded_work_id") == link.get("work_id") and f.get("award_status") == status]
        result["film_claim_" + status] = int(bool(current))
        result["film_claim_" + status + "_missing"] = int(not current)
        result["film_claim_" + status + "_families"] = sorted({f["family"] for f in current})
        for role, people in (("cast", cast), ("director", directors)):
            available = people is not None
            known = set(people or [])
            personal = {f["recipient_id"] for f in claims if f.get("recipient_type") == "person" and f.get("recipient_id") in known and f.get("award_status") == status}
            narrow = {f["recipient_id"] for f in claims if f.get("recipient_type") == "person" and f.get("recipient_id") in known and f.get("award_status") == status and f.get("category_role") == ("acting" if role == "cast" else "directing")}
            for scope, winners in (("person", personal), ("acting" if role == "cast" else "directing", narrow)):
                result[f"{role}_{scope}_{status}"] = {"count": len(winners) if available else None,
                    "fraction": len(winners) / len(known) if known else None,
                    "has": bool(winners) if available else None,
                    "known_people": len(known) if available else None, "missing": int(not available or not winners)}
    result["film_claim_festival_participation"] = int(any(f.get("recipient_type") == "work" and link and f.get("awarded_work_id") == link.get("work_id") and f.get("award_status") in {"selected", "screened", "special_mention"} and f.get("festival_verified") for f in claims))
    return result


def aggregate_tv(episodes, statuses, scope, boundary, mode, link):
    result = {"released_episode_count_asof": None, "started_season_count_asof": None,
        "days_since_last_released_episode_asof": None, "series_status_asof": "unknown",
        "episode_count_missing": 1, "season_count_missing": 1, "recency_missing": 1, "series_status_missing": 1,
        "status_scope": "parent_work", "missing": 1, "support": 0, "evidence_ids": []}
    if not link or not _accepted(link):
        return result | {"reason": "unaccepted_work_link"}
    if mode == "strict_asof" and date_gate(link.get("source_known_at"), boundary) != "eligible":
        return result | {"reason": "work_link_not_historical"}
    if not scope or scope.get("scope_match_status") != "accepted":
        return result | {"reason": "scope_unverified"}
    if scope.get("parent_work_id") != link.get("work_id"):
        return result | {"reason": "scope_parent_mismatch"}
    if mode == "strict_asof" and date_gate(scope.get("source_known_at"), boundary) != "eligible":
        return result | {"reason": "scope_not_historical"}
    if scope.get("scope_kind") in {"fitness_package", "compilation"}:
        return result | {"reason": "not_applicable"}
    if scope.get("scope_kind") not in {"whole_series", "season", "episode_set", "miniseries"}:
        return result | {"reason": "scope_unknown"}
    selected, exclusions = [], []
    for fact in episodes:
        if fact.get("parent_work_id") != scope.get("parent_work_id"):
            continue
        if fact.get("special") and not scope.get("include_specials", False):
            continue
        if scope.get("scope_kind") == "season" and fact.get("season_id") not in scope.get("season_ids", []):
            continue
        if scope.get("scope_kind") == "episode_set" and fact.get("episode_id") not in scope.get("episode_ids", []):
            continue
        reason = temporal_gate(fact, boundary, mode, link)
        if reason == "eligible" and fact.get("airdate_kind") == "confirmed_release":
            selected.append(fact)
        elif reason != "future_or_equal":
            exclusions.append({"evidence_id": fact.get("evidence_id"), "reason": reason if reason != "eligible" else "planned_not_confirmed"})
    unique = {}
    conflict = False
    for fact in selected:
        key = fact.get("episode_id")
        if key is None or (key in unique and (unique[key].get("event_at"), unique[key].get("season_id")) != (fact.get("event_at"), fact.get("season_id"))):
            conflict = True
        unique[key] = fact
    selected = list(unique.values())
    complete = scope.get("source_collection_complete_for_query") is True
    if "episode_collection_receipts" in scope:
        complete = any(receipt.get("status") == 200 and receipt.get("source_collection_complete_for_query") is True
            and (mode not in {"strict_asof", "historical_source_retrospective_identity"}
                or date_gate(receipt.get("source_known_at"), boundary) == "eligible")
            for receipt in scope["episode_collection_receipts"])
    lower = len(selected)
    point_known = complete and not exclusions and not conflict
    result.update({"released_episode_count_lower_bound": lower,
        "released_episode_count_asof": lower if point_known else None,
        "started_season_count_asof": len({f["season_id"] for f in selected if f.get("season_id") is not None}) if point_known and all(f.get("season_id") is not None for f in selected) else None,
        "support": lower, "missing": int(not point_known),
        "reason": "conflict" if conflict else "incomplete_history" if not complete else "unresolved_dates_or_availability" if exclusions else None,
        "evidence_ids": [f["evidence_id"] for f in selected], "excluded": exclusions})
    result["episode_count_missing"] = int(result["released_episode_count_asof"] is None)
    result["season_count_missing"] = int(result["started_season_count_asof"] is None)
    if selected and point_known:
        last_lower = max(date_interval(f["event_at"])[0] for f in selected)
        last_upper = max(date_interval(f["event_at"])[1] for f in selected)
        result["days_since_last_released_episode_bounds"] = [(boundary - last_upper).total_seconds() / 86400, (boundary - last_lower).total_seconds() / 86400]
        if last_lower == last_upper:
            result["days_since_last_released_episode_asof"] = (boundary - last_lower).total_seconds() / 86400
            result["recency_missing"] = 0
    applicable = []
    for status in statuses:
        if status.get("parent_work_id") != scope.get("parent_work_id") or not status.get("historical_state_verified"):
            continue
        if temporal_gate(status, boundary, mode, link) != "eligible":
            continue
        end = date_interval(status.get("valid_until"))
        exact_snapshot = status.get("valid_at") and _instant(status["valid_at"]) == boundary
        if not exact_snapshot and (not end or end[0] <= boundary):
            continue
        if status.get("series_status") in {"ended", "ongoing"}:
            applicable.append(status)
    values = {s["series_status"] for s in applicable}
    if len(values) == 1:
        result["series_status_asof"] = next(iter(values))
        result["series_status_missing"] = 0
        result["status_evidence_ids"] = [s["evidence_id"] for s in applicable]
        result["status_source_known_at"] = [s.get("source_known_at") for s in applicable]
        result["status_source_revisions"] = [s.get("source_revision") for s in applicable]
        result["status_source_snapshot_kinds"] = sorted({s.get("source_snapshot_kind", "unspecified") for s in applicable})
    return result


def ingest_wikipedia_tv_snapshot(snapshot, *, parent_work_id):
    """Narrow old-infobox status adapter, with independently accepted article ID.

    Parent summary totals are evidence only and never fill a season card's counts.
    'present' means the archived source asserted ongoing, not a current status.
    The exact latest-revision query boundary is its applicability point.
    """
    identity = snapshot.get("article_identity", {})
    if not _accepted(identity) or identity.get("historical_identity_accepted") is not True:
        return {"facts": [], "reason": "historical_article_identity_unaccepted"}
    path = Path(snapshot["api_response_path"])
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != snapshot.get("response_sha256"):
        raise ValueError("Wikipedia archive response hash mismatch")
    payload = json.loads(body)
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    matches = [revision for page in pages if page.get("pageid") == identity.get("pageid")
        and page.get("title") == identity.get("title") for revision in page.get("revisions", [])
        if revision.get("revid") == snapshot.get("revision_id")]
    if len(matches) != 1:
        raise ValueError("Wikipedia revision identity mismatch")
    revision = matches[0]
    if revision.get("timestamp") != snapshot.get("revision_timestamp"):
        raise ValueError("Wikipedia revision clock mismatch")
    boundary = _instant(snapshot["query_cutoff"])
    if not snapshot.get("latest_revision_before_cutoff") or _instant(revision["timestamp"]) >= boundary:
        return {"facts": [], "reason": "snapshot_not_before_boundary"}
    content = revision.get("slots", {}).get("main", {}).get("content")
    if content is None:
        content = revision.get("slots", {}).get("main", {}).get("*", revision.get("*"))
    if not isinstance(content, str):
        return {"facts": [], "reason": "wikitext_unavailable"}
    start = re.search(r"\{\{\s*Infobox television\b", content, re.I)
    if not start:
        return {"facts": [], "reason": "television_infobox_unavailable"}
    depth, end = 0, None
    for match in re.finditer(r"\{\{|\}\}", content[start.start():]):
        depth += 1 if match.group() == "{{" else -1
        if depth == 0:
            end = start.start() + match.end()
            break
    if end is None:
        return {"facts": [], "reason": "infobox_unbalanced"}
    infobox = re.sub(r"<!--.*?-->", "", content[start.start():end], flags=re.S)
    fields = {}
    for match in re.finditer(r"^\|\s*(last_aired|num_series|num_seasons|num_episodes)\s*=\s*([^\n]*)", infobox, re.M | re.I):
        key, value = match.group(1).casefold(), match.group(2).strip()
        if key in fields:
            return {"facts": [], "reason": "duplicate_infobox_field"}
        fields[key] = value
    evidence_id = "wikipedia:" + str(identity["pageid"]) + ":revision:" + str(revision["revid"])
    base = {"status": "accepted", "evidence_id": evidence_id, "parent_work_id": parent_work_id,
        "source": "Wikipedia archived revision", "source_known_at": revision["timestamp"],
        "event_at": revision["timestamp"], "source_revision": revision["revid"],
        "retrieved_at": snapshot.get("retrieved_at"), "response_sha256": snapshot["response_sha256"],
        "evidence_url": "https://en.wikipedia.org/w/index.php?oldid=" + str(revision["revid"]),
        "raw_fields": fields, "source_snapshot_kind": "archived_revision",
        "assertion_kind": "metadata_assertion", "license": "CC BY-SA (Wikipedia text)"}
    facts = [base | {"kind": "series_summary", "status_scope": "parent_work"}]
    last_aired = fields.get("last_aired", "").strip("' ").casefold()
    template = re.fullmatch(r"\{\{\s*end[ _]date\s*\|(.+?)\}\}", last_aired)
    if template:
        parameters = [value.strip() for value in template.group(1).split("|")]
        if parameters.count("present") == 1 and all(value in {"present", "df=yes", "df=no"} for value in parameters):
            last_aired = "present"
    if last_aired == "present":
        facts.append(base | {"kind": "series_status", "series_status": "ongoing",
            "historical_state_verified": True, "valid_at": boundary.isoformat(),
            "status_scope": "parent_work", "evidence_id": evidence_id + ":last_aired"})
    return {"facts": facts, "reason": None if len(facts) > 1 else "no_explicit_supported_status"}


def aggregate_person_filmography(person_id, role, current_work_id, current_year, credits,
        boundary, mode, person_link=None, collection=None):
    collection = collection or {}
    result = {"external_observed_work_count": None, "external_credits_before_work": None,
        "missing": 1, "source_available": bool(collection.get("source_available")),
        "source_collection_complete_for_query": bool(collection.get("source_collection_complete_for_query")),
        "person_link_accepted": bool(person_link and _accepted(person_link)), "evidence_ids": []}
    if not result["person_link_accepted"] or not result["source_available"]:
        return result | {"reason": "person_or_source_unavailable"}
    if mode == "strict_asof" and date_gate(person_link.get("source_known_at"), boundary) != "eligible":
        return result | {"reason": "person_link_not_historical"}
    selected = [f for f in credits if f.get("person_id") == person_id and f.get("role") == role and f.get("work_id") != current_work_id and f.get("person_link_accepted")]
    facts = []
    identity_rejections = []
    for fact in selected:
        if date_gate(fact.get("event_at"), boundary) == "future_or_equal":
            identity_rejections.append({"evidence_id": fact.get("evidence_id"), "reason": "future_or_equal"})
        elif mode == "strict_asof" and date_gate(fact.get("credit_identity_known_at", person_link.get("source_known_at")), boundary) != "eligible":
            identity_rejections.append({"evidence_id": fact.get("evidence_id"), "reason": "credit_identity_not_historical"})
        else:
            facts.append(fact)
    eligible, rejected = _eligible(facts, boundary, mode, person_link)
    rejected.extend(identity_rejections)
    unique = {}
    for fact in eligible:
        previous = unique.get(fact["work_id"])
        if previous and any(previous.get(key) != fact.get(key) for key in ("release_year", "work_type", "genres", "formats", "kion_membership")):
            return result | {"reason": "conflicting_credit_metadata", "excluded": rejected}
        unique.setdefault(fact["work_id"], fact)
    observed = list(unique.values())
    count_known = bool(observed) or (result["source_collection_complete_for_query"] and
        not any(row["reason"] != "future_or_equal" for row in rejected))
    if not count_known:
        return result | {"reason": "no_temporally_eligible_observations", "excluded": rejected}
    prior = [f for f in observed if isinstance(f.get("release_year"), int) and isinstance(current_year, int) and f["release_year"] < current_year]
    years = [f["release_year"] for f in prior]
    span = max(years) - min(years) + 1 if years else None
    result.update({"external_observed_work_count": len(observed), "external_credits_before_work": len(prior) if isinstance(current_year, int) else None,
        "external_confirmed_outside_kion_work_count": sum(f.get("kion_membership") == "outside_confirmed" for f in observed),
        "external_membership_unknown_count": sum(f.get("kion_membership", "unknown") == "unknown" for f in observed),
        "external_films_before_work": sum(f.get("work_type") == "film" for f in prior),
        "external_series_before_work": sum(f.get("work_type") == "series" for f in prior),
        "external_distinct_prior_release_years": len(set(years)), "external_observed_career_span_years": span,
        "external_observed_credits_per_span_year": len(prior) / span if span else None,
        "external_years_since_first_observed_credit": current_year - min(years) if years else None,
        "missing": 0, "support": len(observed), "year_available": isinstance(current_year, int),
        "evidence_ids": [f["evidence_id"] for f in observed], "excluded": rejected})
    for attribute in ("genres", "formats"):
        known = []
        for fact in prior:
            categories = set(fact.get(attribute) or [])
            if mode in {"strict_asof", "historical_source_retrospective_identity"}:
                clocks = fact.get(attribute + "_label_source_known_at", {})
                categories = {category for category in categories if date_gate(clocks.get(category,
                    fact.get(attribute + "_source_known_at", fact.get("source_known_at"))), boundary) == "eligible"}
            if categories:
                known.append((fact, categories))
        counts = Counter(category for _, categories in known for category in categories)
        result["external_" + attribute + "_profile"] = {"counts": dict(counts),
            "fractions": {k: v / len(known) for k, v in counts.items()}, "known_work_support": len(known),
            "unknown_work_support": len(prior) - len(known), "fraction_denominator": "prior observed works with eligible normalized category set",
            "mapping_versions": sorted({f.get("genre_mapping_version" if attribute == "genres" else "format_mapping_version") for f, _ in known} - {None}),
            "missing": int(not known)}
    result["external_genre_profile"] = result["external_genres_profile"]
    result["external_format_profile"] = result["external_formats_profile"]
    return result


def build_item_facts(item, link, facts, *, target_at, feature_role="sft_validation", temporal_mode="strict_asof", scope=None, people=None, collections=None):
    boundary = feature_boundary(target_at, feature_role)
    if temporal_mode not in MODES:
        raise ValueError("unknown temporal mode")
    accepted_link = link if link and _accepted(link) else None
    work_id = accepted_link.get("work_id") if accepted_link else None
    relevant = [f for f in facts if work_id and f.get("subject_id", f.get("work_id")) == work_id]
    award_people = {}
    for role in ("actor", "director"):
        role_links = (people or {}).get("links", {}).get(role)
        if role_links is not None:
            role_people = sorted({p["person_id"] for p in role_links if _accepted(p)
                and (temporal_mode != "strict_asof" or date_gate(p.get("source_known_at"), boundary) == "eligible")})
            award_people[role] = role_people or None
        else:
            award_people[role] = (people or {}).get(role) if temporal_mode != "strict_asof" else None
    result = {"contract": CONTRACT, "item_id": item.get("item_id"), "work_id": work_id,
        "target_at": target_at, "feature_boundary_at": boundary.isoformat(), "feature_role": feature_role,
        "temporal_mode": temporal_mode, "temporal_policy": POLICY,
        "source_temporal_mode": "historical" if temporal_mode in {"strict_asof", "historical_source_retrospective_identity"} else "retrospective",
        "identity_temporal_mode": "snapshot_retrospective" if temporal_mode == "historical_source_retrospective_identity" else temporal_mode,
        "identity_historical_available": bool(accepted_link and date_gate(accepted_link.get("source_known_at"), boundary) == "eligible"),
        "duration": aggregate_runtime([f for f in relevant if f.get("kind") == "runtime"], accepted_link, boundary, temporal_mode),
        "work_runtime_proxy": aggregate_work_runtime_proxy([f for f in relevant if f.get("kind") == "runtime"], accepted_link, boundary, temporal_mode),
        "languages": aggregate_languages([f for f in relevant if f.get("kind") == "language"], accepted_link, boundary, temporal_mode),
        "awards": aggregate_awards([f for f in facts if f.get("kind") == "award"], accepted_link, boundary, temporal_mode,
            cast=award_people.get("actor"), directors=award_people.get("director")),
        "tv": aggregate_tv([f for f in facts if f.get("kind") == "episode"], [f for f in facts if f.get("kind") == "series_status"], scope, boundary, temporal_mode, accepted_link),
        "people": {}}
    for role, person_links in (people or {}).get("links", {}).items():
        result["people"][role] = {p["person_id"]: aggregate_person_filmography(p["person_id"], role, work_id,
            _year(item.get("release_year")), [f for f in facts if f.get("kind") == "credit"], boundary, temporal_mode,
            person_link=p, collection=(collections or {}).get(p["person_id"] + ":" + role + ":" + str(work_id),
                (collections or {}).get(p["person_id"] + ":" + role, (collections or {}).get(p["person_id"])))) for p in person_links}
    return result


def ingest_tvmaze_episodes(payload, *, parent_work_id, source_evidence, scope_verified=False):
    """Current episode catalogue adapter; never interprets current show status.

    TVmaze airdate is a retrospective release assertion. Missing airstamp stays
    a date interval; schedule-only/future records cannot establish a past release.
    """
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("TVmaze episode catalogue must be a list of objects")
    facts = []
    for row in payload:
        if row.get("id") is None:
            raise ValueError("TVmaze episode missing id")
        event_at = row.get("airstamp") or row.get("airdate")
        base = {"kind": "episode", "evidence_id": "tvmaze:episode:" + str(row["id"]),
            "episode_id": "tvmaze:episode:" + str(row["id"]), "parent_work_id": parent_work_id,
            "season_id": str(row["season"]) if row.get("season") is not None else None,
            "special": row.get("type") in {"significant_special", "insignificant_special"} or row.get("number") is None,
            "event_at": event_at, "airdate_kind": "confirmed_release",
            "status": "accepted" if scope_verified else "proposed", "source": "tvmaze", "license": "CC BY-SA",
            "evidence_url": row.get("url"), "raw_record": row,
            "source_known_at": source_evidence.get("source_known_at"),
            "retrieved_at": source_evidence.get("retrieved_at"), "response_sha256": source_evidence.get("response_sha256")}
        facts.append(base)
        if row.get("runtime") is not None:
            normalized, reason = normalize_runtime({"amount": row["runtime"], "unit": "minute"})
            facts.append(base | {"kind": "runtime", "subject_id": parent_work_id, "work_id": parent_work_id,
                "evidence_id": base["evidence_id"] + ":runtime", "runtime_scope": "episode",
                "scope_verified": scope_verified, "value": normalized, "reason": reason})
    return facts


def fetch_tvmaze_resource(show_id, resource, output_dir, *, user_agent, timeout=20, max_bytes=2_000_000):
    """One bounded HTTPS request to a documented show ID; no title search.

    Raises on HTTP/network/schema/size errors: callers must record partial failure,
    never synthesize an empty successful catalogue. No automatic retry escalation.
    """
    if not isinstance(show_id, int) or show_id <= 0 or not user_agent.strip():
        raise ValueError("positive accepted show ID and meaningful User-Agent required")
    routes = {"show": "", "episodes": "/episodes?specials=1", "seasons": "/seasons", "cast": "/cast"}
    if resource not in routes:
        raise ValueError("unsupported bounded TVmaze resource")
    url = f"https://api.tvmaze.com/shows/{show_id}" + routes[resource]
    with urlopen(Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"}), timeout=timeout) as response:
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("TVmaze response byte budget exceeded")
        status = response.status
    payload = json.loads(body)
    expected = dict if resource == "show" else list
    if not isinstance(payload, expected):
        raise ValueError("unexpected TVmaze resource JSON type")
    digest = hashlib.sha256(body).hexdigest()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / (digest + ".json")
    if path.exists() and path.read_bytes() != body:
        raise ValueError("immutable cache collision")
    if not path.exists():
        path.write_bytes(body)
    receipt = {"source": "tvmaze", "url": url, "status": status, "bytes": len(body),
        "response_sha256": digest, "retrieved_at": datetime.now(UTC).isoformat(), "source_known_at": None,
        "license": "CC BY-SA", "attribution": "TVmaze: https://www.tvmaze.com/", "resource": resource,
        "specials": "included" if resource == "episodes" else None}
    write_json(output_dir / (digest + ".receipt.json"), receipt)
    return payload, receipt


def fetch_tvmaze_episodes(show_id, output_dir, *, user_agent, timeout=20, max_bytes=2_000_000):
    return fetch_tvmaze_resource(show_id, "episodes", output_dir, user_agent=user_agent,
        timeout=timeout, max_bytes=max_bytes)


POPULATION_CONTRACT = "kion-external-population-v4-source-genre-format-profiles"
LANGUAGE_CODES = tuple(sorted(set(LANGUAGES.values()) | set(LANGUAGE_ALIASES.values())))
CAREER_METRICS = ("external_observed_work_count", "external_credits_before_work",
    "external_films_before_work", "external_series_before_work", "external_distinct_prior_release_years",
    "external_observed_career_span_years", "external_observed_credits_per_span_year",
    "external_years_since_first_observed_credit", "external_confirmed_outside_kion_work_count",
    "external_membership_unknown_count")


def _pinned_json(spec, root, pins):
    if isinstance(spec, str):
        spec = {"path": spec}
    path = Path(spec["path"])
    if not path.is_absolute():
        path = Path(root) / path
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if spec.get("sha256") and spec["sha256"] != digest:
        raise ValueError("input pin mismatch: " + path.as_posix())
    pins[path.as_posix()] = digest
    return json.loads(body)


def _union_accepted(exports):
    """Merge reviewed records only; conflicting identities are quarantined."""
    works, people, work_conflicts, person_conflicts = {}, {}, {}, {}
    for export in exports:
        for row in export.get("works", []):
            if not _accepted(row) or row.get("item_id") is None:
                continue
            if not row.get("work_id"):
                raise ValueError("accepted work lacks canonical identity")
            key = str(row["item_id"])
            if key in works and works[key].get("work_id") != row.get("work_id"):
                work_conflicts[key] = sorted({works[key].get("work_id"), row.get("work_id")})
            # Declared export order chooses the active evidence version only
            # when canonical identity agrees; it never resolves a conflict.
            works[key] = row
        for row in export.get("people", []):
            if not _accepted(row) or row.get("role") not in {"actor", "director"}:
                continue
            if not row.get("person_id"):
                raise ValueError("accepted person lacks canonical identity")
            key = (str(row.get("item_id")), row["role"], row.get("mention_id") or
                (row.get("position"), row.get("raw_name", row.get("person_id"))))
            if key in people and people[key].get("person_id") != row.get("person_id"):
                person_conflicts[str(key)] = sorted({people[key].get("person_id"), row.get("person_id")})
            people[key] = row
    works = {key: row for key, row in works.items() if key not in work_conflicts}
    people = [row for key, row in people.items() if str(key) not in person_conflicts and str(row.get("item_id")) not in work_conflicts]
    return {"works": list(works.values()), "people": people}, {
        "work_conflicts": work_conflicts, "person_conflicts": person_conflicts}


def load_population_sources(manifest, *, root, temporal_mode):
    """Load a shared accepted union and body index without HTTP or proposals.

    The pool may contain many unreviewed entities; only exact body hashes cited
    by accepted rows are opened. Historical archives are selected independently
    from current identity and are not inferred from retrieval timestamps.
    """
    pins = {}
    pool = _pinned_json(manifest["source_pool"], root, pins)
    cache = {}
    for digest, entry in pool.get("bodies", {}).items():
        if entry.get("sha256") != digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid source-pool digest")
        paths = [Path(value) if Path(value).is_absolute() else Path(root) / value for value in entry.get("paths", [])]
        path = next((value for value in paths if value.is_file()), None)
        if path is not None:
            cache[digest] = path
    export_specs = manifest.get("identity_exports") or [pool["accepted_identity_export"]]
    exports = [_pinned_json(spec, root, pins) for spec in export_specs]
    links, conflicts = _union_accepted(exports)
    archives = {}
    for spec in manifest.get("historical_archives", pool.get("historical_archives", [])):
        archive = _pinned_json(spec, root, pins)
        for qid, row in archive.get("entities", {}).items():
            if row.get("source_evidence"):
                if qid in archives and archives[qid].get("revision_id") != row.get("revision_id"):
                    raise ValueError("conflicting archive revisions; select an explicit boundary-specific archive")
                archives[qid] = row
    historical = temporal_mode in {"strict_asof", "historical_source_retrospective_identity"}
    ingestion = ingest_archived_wikidata(links, {"entities": archives}, cache) if historical else ingest_wikidata(links, cache)
    facts = ingestion["facts"]
    collections = {}
    for spec in manifest.get("credit_corpora", []):
        corpus = _pinned_json(spec, root, pins)
        result = ingest_credit_corpus(corpus, cache, archive={"entities": archives} if historical else None)
        facts.extend(result["facts"])
        ingestion["diagnostics"].extend(result["diagnostics"])
        for key, value in result["collections"].items():
            if key in collections and canonical_hash(collections[key]) != canonical_hash(value):
                value = dict(value) | {"source_collection_complete_for_query": False,
                    "source_available": bool(value.get("source_available") or collections[key].get("source_available"))}
            collections[key] = value
    tv_collection_receipts = {}
    for spec in manifest.get("tv_bundles", []):
        accepted = _pinned_json(spec["scope"], root, pins)
        link, scope = accepted["link"], accepted["scope"]
        if not _accepted(link) or scope.get("scope_match_status") != "accepted":
            continue
        iid = str(link["item_id"])
        if spec.get("item_id") is not None and str(spec["item_id"]) != iid:
            raise ValueError("TV receipt item selector disagrees with accepted scope")
        if scope.get("parent_work_id") != link.get("work_id"):
            ingestion["diagnostics"].append({"item_id": iid, "reason": "tv_scope_parent_mismatch"})
            continue
        if iid in conflicts["work_conflicts"]:
            continue
        if iid in conflicts.get("tv_scope_conflicts", {}):
            continue
        current = next((row for row in links["works"] if str(row["item_id"]) == iid), None)
        if current and current.get("work_id") != link.get("work_id"):
            conflicts["work_conflicts"][iid] = [current.get("work_id"), link.get("work_id")]
            links["works"] = [row for row in links["works"] if str(row["item_id"]) != iid]
            continue
        if not current:
            links["works"].append(link | {"tv_scope": scope})
        else:
            # The accepted external scope supplements the same canonical work;
            # it does not replace its independently reviewed identity record.
            previous = current.get("tv_scope")
            def scope_selection(value):
                return {"scope_kind": value.get("scope_kind"), "parent_work_id": value.get("parent_work_id"),
                    "provider_parent_work_id": value.get("provider_parent_work_id",
                        value.get("parent_work_id") if str(value.get("parent_work_id", "")).startswith("tvmaze:show:") else None),
                    "season_ids": sorted(map(str, value.get("season_ids", []))),
                    "episode_ids": sorted(map(str, value.get("episode_ids", []))),
                    "include_specials": bool(value.get("include_specials", False))}
            if previous and previous.get("scope_match_status") == "accepted":
                if scope_selection(previous) != scope_selection(scope):
                    conflicts.setdefault("tv_scope_conflicts", {})[iid] = [previous, scope]
                    current.pop("tv_scope", None)
                    continue
                # An additional receipt cannot erase an existing source clock
                # or completeness assertion for the same selected scope.
            else:
                current["tv_scope"] = scope
        provider_parent = scope.get("provider_parent_work_id", scope.get("parent_work_id"))
        provider_match = re.fullmatch(r"tvmaze:show:(\d+)", str(provider_parent))
        if not provider_match:
            raise ValueError("TV scope provider bridge missing")
        provider_id = provider_match.group(1)
        if spec.get("provider_id") is not None and str(spec["provider_id"]) != provider_id:
            raise ValueError("TV receipt provider selector disagrees with accepted scope")
        provider_url = "https://api.tvmaze.com/shows/" + provider_id
        acquisition = _pinned_json(spec["acquisition"], root, pins)
        receipts = [row for row in acquisition["receipts"] if row.get("resource") == "episodes"
            and row.get("status") == 200
            and row.get("url") == provider_url + "/episodes?specials=1"
            and (row.get("provider_id") is None or str(row["provider_id"]) == provider_id)
            and (row.get("item_id") is None or str(row["item_id"]) == iid)
            and (spec.get("provider_id") is None or str(row.get("provider_id")) == str(spec["provider_id"]))
            and (spec.get("item_id") is None or str(row.get("item_id")) == str(spec["item_id"]))]
        if len(receipts) != 1:
            raise ValueError("episode receipt missing or ambiguous for TV bundle")
        receipt = receipts[0]
        digest = receipt["response_sha256"]
        if spec.get("episode_cache_dir"):
            folder = Path(spec["episode_cache_dir"])
            if not folder.is_absolute():
                folder = Path(root) / folder
            path = folder / (digest + ".json")
        else:
            path = _cache_path(cache, digest)
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != digest:
            raise ValueError("TV episode cache hash mismatch")
        pins[path.as_posix()] = digest
        payload = json.loads(body)
        if not isinstance(payload, list) or any(not isinstance(row, dict)
                or row.get("_links", {}).get("show", {}).get("href") != provider_url for row in payload):
            raise ValueError("TV episode raw parent disagrees with accepted provider bridge")
        episode_facts = ingest_tvmaze_episodes(payload, parent_work_id=link["work_id"], source_evidence=receipt, scope_verified=True)
        runtime_policy = spec.get("runtime_admission", "accepted_episode_estimate")
        if runtime_policy not in {"accepted_episode_estimate", "raw_only"}:
            raise ValueError("unknown TV runtime admission policy")
        if runtime_policy == "raw_only":
            for fact in episode_facts:
                if fact.get("kind") == "runtime":
                    fact["reason"] = "raw_only_provider_runtime"
                    fact["runtime_admission_reason"] = spec.get("runtime_admission_reason", "runtime content unit or edition not verified")
        facts.extend(episode_facts)
        tv_collection_receipts.setdefault(iid, []).append({"response_sha256": digest, "status": receipt["status"],
            "source_known_at": receipt.get("source_known_at"),
            "source_collection_complete_for_query": scope.get("source_collection_complete_for_query") is True})
        snapshot_spec = spec.get("wikipedia_snapshot")
        if snapshot_spec:
            snapshot = _pinned_json(snapshot_spec, root, pins)
            result = ingest_wikipedia_tv_snapshot(snapshot, parent_work_id=link["work_id"])
            pins[Path(snapshot["api_response_path"]).as_posix()] = snapshot["response_sha256"]
            facts.extend(result["facts"])
    for link in links["works"]:
        if link.get("tv_scope"):
            # An accepted scope without a successfully loaded paired catalogue
            # cannot certify an empty result. Its identity evidence still holds.
            link["tv_scope"] = dict(link["tv_scope"]) | {
                "episode_collection_receipts": tv_collection_receipts.get(str(link["item_id"]), [])}
    if manifest.get("award_taxonomy"):
        taxonomy = _pinned_json(manifest["award_taxonomy"], root, pins)
        facts = apply_award_taxonomy(facts, taxonomy.get("categories", {}))
    if manifest.get("external_genre_mapping"):
        mapping = _pinned_json(manifest["external_genre_mapping"], root, pins)
        facts = apply_external_genre_mapping(facts, mapping)
    unique = {}
    for fact in facts:
        key = (fact.get("kind"), fact.get("evidence_id"), fact.get("subject_id"), fact.get("person_id"))
        if fact.get("kind") == "runtime":
            key += (fact.get("item_id"), fact.get("edition_id"))
        def semantic(value):
            ignored = {"item_id", "identity_source_known_at", "identity_temporal_mode",
                "response_sha256", "retrieved_at", "source_evidence_views"}
            if value.get("subject_type") == "person":
                ignored.add("work_id")
            return {name: field for name, field in value.items() if name not in ignored}
        def evidence_view(value):
            return {name: value.get(name) for name in ("response_sha256", "retrieved_at", "source_revision",
                "source_known_at", "item_id", "work_id", "identity_source_known_at", "identity_temporal_mode")}
        if key in unique and canonical_hash(semantic(unique[key])) != canonical_hash(semantic(fact)):
            # Same statement can arrive through a catalogue anchor and a corpus;
            # prefer the copy with an explicit accepted credit decision.
            if fact.get("kind") == "credit" and fact.get("person_link_accepted"):
                unique[key] = fact
            elif fact.get("kind") != "credit":
                raise ValueError("conflicting versions of the same source assertion")
        elif key in unique:
            # wbgetentities wrappers and retrieval times can differ for the
            # exact same entity revision/statement. Preserve each transport
            # receipt and accepted anchor without treating it as a new fact.
            previous = unique[key]
            views = previous.get("source_evidence_views", [evidence_view(previous)]) + fact.get("source_evidence_views", [evidence_view(fact)])
            previous["source_evidence_views"] = list({canonical_hash(view): view for view in views}.values())
        else:
            unique[key] = fact
    bounded = admit_accepted_work_role_edges(list(unique.values()), links, collections)
    return {"links": links, "facts": bounded["facts"], "collections": bounded["collections"],
        "accepted_work_credit_corpus": bounded["manifest"],
        "conflicts": conflicts, "diagnostics": ingestion["diagnostics"], "input_pins": pins,
        "source_pool_body_n": len(pool.get("bodies", {})), "unreviewed_pool_entities_used": False}


def admit_accepted_work_role_edges(facts, links, collections):
    """A bounded observed source corpus, not a reverse query or full career.

    Identity comes only from accepted work/person-role rows. The raw exact
    P57/P161 target independently proves each relationship; aliases play no part.
    """
    work_links, role_links = defaultdict(list), defaultdict(list)
    for row in links["works"]:
        if _accepted(row):
            work_links[row["work_id"]].append(row)
    for row in links["people"]:
        if _accepted(row) and row.get("role") in {"actor", "director"}:
            role_links[(row["person_id"], row["role"])].append(row)
    result, observed = [], defaultdict(set)
    def earliest_identity(rows):
        dated = [row["source_known_at"] for row in rows if date_interval(row.get("source_known_at"))]
        return min(dated, key=lambda date: date_interval(date)[1]) if dated else None
    for original in facts:
        fact = dict(original)
        role_key = (fact.get("person_id"), fact.get("role"))
        raw_target = _value(fact.get("raw_statement", {}).get("mainsnak", {}))
        prop = {"actor": "P161", "director": "P57"}.get(fact.get("role"))
        work_qid = str(fact.get("work_id", "")).removeprefix("wd:")
        statement_id = str(fact.get("raw_statement", {}).get("id", ""))
        if _accepted(fact) and fact.get("kind") == "credit" and fact.get("work_id") in work_links and role_key in role_links \
                and fact.get("subject_type") == "work" and fact.get("subject_id") == fact.get("work_id") \
                and fact.get("source_id") == work_qid and statement_id.split("$", 1)[0].casefold() == work_qid.casefold() \
                and "$" in statement_id and fact.get("evidence_id") == statement_id \
                and fact.get("source") == "wikidata" and fact.get("property") == prop \
                and fact.get("raw_statement", {}).get("rank") != "deprecated" \
                and isinstance(raw_target, dict) and "wd:" + str(raw_target.get("id")) == fact.get("person_id"):
            anchor_known = earliest_identity(role_links[role_key])
            work_known = earliest_identity(work_links[fact["work_id"]])
            identity_known = max((anchor_known, work_known), key=lambda date: date_interval(date)[1]) if anchor_known and work_known else None
            if not fact.get("person_link_accepted"):
                fact["credit_identity_known_at"] = identity_known
            fact.update({"person_link_accepted": True, "kion_membership": "inside",
                "bounded_credit_scope": "accepted_work_role_edge_subset",
                "bounded_role_anchor_ids": sorted({str(row.get("mention_id") or row.get("item_id")) for row in role_links[role_key]})})
            observed[role_key].add(fact["evidence_id"])
        result.append(fact)
    output_collections = dict(collections)
    for (person, role), evidence_ids in observed.items():
        subset = {"source_available": True, "source_collection_complete_for_query": False,
            "source_filmography_complete": False, "collection_scope": "accepted_work_role_edge_subset",
            "person_id": person, "role": role, "evidence_ids": sorted(evidence_ids), "query_performed": False}
        output_collections[person + ":" + role] = subset
        for key, collection in list(output_collections.items()):
            if key.startswith(person + ":" + role + ":"):
                output_collections[key] = collection | {"source_available": True,
                    "query_source_available": collection.get("source_available", False),
                    "collection_scope": "observed_union_of_query_and_accepted_work_subset",
                    "source_filmography_complete": False, "additional_observed_corpus": subset}
    return {"facts": result, "collections": output_collections, "manifest": {
        "contract": "kion-accepted-work-role-corpus-v1", "scope": "accepted_work_role_edge_subset",
        "work_n": len(work_links), "person_role_n": len(observed),
        "source_role_statement_n": sum(len(values) for values in observed.values()),
        "distinct_person_role_work_n": len({(fact["person_id"], fact["role"], fact["work_id"]) for fact in result if fact.get("bounded_credit_scope")}),
        "query_performed": False,
        "source_filmography_complete": False,
        "edges": [{"person_id": person, "role": role, "evidence_ids": sorted(values)} for (person, role), values in sorted(observed.items())]}}


def export_release_records(sources):
    """All accepted P577 claims, preserving scope; never a global first date."""
    by_work = defaultdict(list)
    by_item = {}
    for row in sources["links"]["works"]:
        by_work[row.get("work_id")].append(row)
        by_item[str(row["item_id"])] = row
    people = defaultdict(lambda: {"actor": set(), "director": set()})
    for row in sources["links"]["people"]:
        anchor = by_item.get(str(row.get("item_id")))
        if row.get("role") in {"actor", "director"} and anchor and row.get("work_id") in {None, anchor.get("work_id")}:
            people[str(row["item_id"])][row["role"]].add(row["person_id"])
    records = []
    seen = set()
    for fact in sources["facts"]:
        if fact.get("kind") != "release" or fact.get("subject_type") != "work":
            continue
        work = fact.get("work_id")
        for link in by_work.get(work, []):
            iid = str(link["item_id"])
            key = (iid, fact["evidence_id"], fact.get("source_revision"))
            if key in seen:
                continue
            seen.add(key)
            records.append({"item_id": iid, "work_id": work, "date": fact.get("date"),
                "precision": fact.get("precision"), "status": "source_asserted_publication",
                "match_status": "confirmed", "release_unit": "work",
                "scope": "reported_work_publication_claim", "source": "wikidata:P577",
                "evidence_id": fact["evidence_id"], "known_at": fact.get("source_known_at"),
                "complete": False, "global_earliest_known": False, "distribution_scope_known": False,
                "actors": sorted(people[iid]["actor"]), "directors": sorted(people[iid]["director"]),
                "raw_record": fact})
    return {"contract": "kion-source-release-records-v1", "records": records,
        "accepted_work_n": len(by_work), "record_n": len(records),
        "global_earliest_date_inferred": False, "source_input_pins": sources.get("input_pins", {})}


def _population_row(result, item, accepted_people, *, work_conflict=False):
    """Fixed numeric schema; unknown values are zero only with known=0."""
    values, known = {}, {}
    def put(name, value, available=None):
        available = value is not None if available is None else bool(available)
        number = float(value) if value is not None and available else 0.0
        if not math.isfinite(number):
            raise ValueError("nonfinite external feature: " + name)
        values[name], known[name] = number, int(available)
    strict_identity_available = result.get("temporal_mode") != "strict_asof" or result.get("identity_historical_available")
    put("external_work_link_accepted", bool(result.get("work_id")), strict_identity_available)
    put("external_work_link_conflict", work_conflict, strict_identity_available)
    if result.get("temporal_mode") == "strict_asof":
        boundary = _instant(result["feature_boundary_at"])
        accepted_people = [person for person in accepted_people if date_gate(person.get("source_known_at"), boundary) == "eligible"]
    duration = result["duration"]
    put("runtime_minutes", duration.get("value"))
    for key in ("missing", "support", "duration_is_estimate", "episode_sample_n", "episode_coverage"):
        put("runtime_" + key, duration.get(key))
    for key in ("min_minutes", "median_minutes", "max_minutes", "assertion_n", "distinct_value_n", "disagreement", "missing", "edition_verified"):
        put("work_runtime_proxy_" + key, result["work_runtime_proxy"].get(key))
    for family, row in result["languages"].items():
        for code in LANGUAGE_CODES:
            present = code in row["codes"]
            put("language_" + family + ":" + code, present, present or row["status"] == "known_complete")
        for state in ("unknown", "known_partial", "known_complete", "conflict"):
            put("language_" + family + ":" + state, row["status"] == state)
        put("language_" + family + ":support", row["support"])
    awards = result["awards"]
    for status in ("won", "nominated", "festival_participation"):
        value = awards.get("film_claim_" + status, 0)
        put("award_work_" + status, value, bool(value))
        put("award_work_" + status + "_missing", not bool(value))
    put("award_accepted_claim_support", awards["support"])
    put("award_unresolved_attribution_support", awards["unresolved_attribution"])
    for role, award_role, narrow in (("actor", "cast", "acting"), ("director", "director", "directing")):
        people = {p["person_id"] for p in accepted_people if p.get("role") == role}
        raw_field = item.get("actors" if role == "actor" else "directors", "") or ""
        raw_names = {" ".join(name.casefold().split()) for name in raw_field.split(",") if name.strip()}
        linked_mentions = {" ".join(str(p.get("raw_name") or "").casefold().split()) for p in accepted_people if p.get("role") == role}
        linked_mentions.discard("")
        put(role + "_accepted_person_n", len(people))
        metadata_available = result.get("temporal_mode") != "strict_asof" or date_gate(item.get("metadata_source_known_at"), boundary) == "eligible"
        put(role + "_raw_name_n", len(raw_names) if raw_names and metadata_available else None)
        put(role + "_accepted_mention_fraction", len(linked_mentions & raw_names) / len(raw_names) if raw_names and metadata_available else None)
        for scope in ("person", narrow):
            for status in ("won", "nominated"):
                row = awards[f"{award_role}_{scope}_{status}"]
                for field in ("count", "fraction", "has", "missing"):
                    suffix = "fraction_of_accepted_people" if field == "fraction" else field
                    put(f"award_{role}_{scope}_{status}_{suffix}", row[field])
        profiles = list(result["people"].get(role, {}).values())
        available = [profile for profile in profiles if not profile.get("missing", 1)]
        put(role + "_filmography_available_person_n", len(available))
        put(role + "_filmography_available_fraction", len(available) / len(people) if people else None)
        format_profiles = [profile["external_formats_profile"] for profile in available
            if not profile.get("external_formats_profile", {}).get("missing", 1)]
        for category in ("film", "series"):
            fractions = [profile["fractions"].get(category, 0) for profile in format_profiles]
            put(role + "_external_format_" + category + "_fraction_mean", statistics.mean(fractions) if fractions else None)
            put(role + "_external_format_" + category + "_fraction_max", max(fractions) if fractions else None)
        supports = [profile["known_work_support"] for profile in format_profiles]
        put(role + "_external_format_known_work_support_mean", statistics.mean(supports) if supports else None)
        put(role + "_external_format_known_work_support_max", max(supports) if supports else None)
        put(role + "_external_format_profile_missing", not bool(format_profiles))
        genre_profiles = [profile["external_genres_profile"] for profile in available
            if not profile.get("external_genres_profile", {}).get("missing", 1)]
        for code, label in EXTERNAL_GENRE_LABELS.items():
            fractions = [profile["fractions"].get(label, 0) for profile in genre_profiles]
            put(role + "_external_genre_" + code + "_fraction_mean", statistics.mean(fractions) if fractions else None)
            put(role + "_external_genre_" + code + "_fraction_max", max(fractions) if fractions else None)
        supports = [profile["known_work_support"] for profile in genre_profiles]
        put(role + "_external_genre_known_work_support_mean", statistics.mean(supports) if supports else None)
        put(role + "_external_genre_known_work_support_max", max(supports) if supports else None)
        put(role + "_external_genre_profile_missing", not bool(genre_profiles))
        unknown_supports = [profile["external_genres_profile"]["unknown_work_support"] for profile in available
            if "external_genres_profile" in profile]
        put(role + "_external_genre_unknown_prior_work_n_mean", statistics.mean(unknown_supports) if unknown_supports else None)
        put(role + "_external_genre_unknown_prior_work_n_max", max(unknown_supports) if unknown_supports else None)
        for metric in CAREER_METRICS:
            numbers = [profile[metric] for profile in available if profile.get(metric) is not None]
            put(role + "_" + metric + "_mean", statistics.mean(numbers) if numbers else None)
            put(role + "_" + metric + "_max", max(numbers) if numbers else None)
            put(role + "_" + metric + "_missing", not bool(numbers))
    tv = result["tv"]
    for key in ("released_episode_count_asof", "started_season_count_asof", "days_since_last_released_episode_asof",
                "episode_count_missing", "season_count_missing", "recency_missing", "series_status_missing", "support"):
        put("tv_" + key, tv.get(key))
    for state in ("ongoing", "ended", "unknown"):
        put("tv_status_" + state, tv["series_status_asof"] == state)
    return values, known


def build_population(metadata, items, sources, *, target_at, feature_role="sft_validation", temporal_mode="snapshot_retrospective"):
    """Full-U alignment with sparse accepted evidence, never full-U review."""
    import numpy as np
    mapping = {str(iid): int(token) for iid, token in metadata["item_id_to_token"].items()}
    vocab = int(metadata["vocab_size"])
    special = set(map(int, metadata.get("special_tokens", {}).values()))
    if len(set(mapping.values())) != len(mapping) or any(token < 0 or token >= vocab or token in special for token in mapping.values()):
        raise ValueError("invalid universe token mapping")
    items = {str(iid): row for iid, row in items.items()}
    if set(mapping) - set(items):
        raise ValueError("universe item missing from raw metadata")
    links = {str(row["item_id"]): row for row in sources["links"]["works"]}
    people = defaultdict(list)
    for row in sources["links"]["people"]:
        people[str(row["item_id"])].append(row)
    active_u_people = [row for row in sources["links"]["people"] if str(row.get("item_id")) in mapping
        and str(row.get("item_id")) in links and row.get("work_id") in {None, links[str(row["item_id"])].get("work_id")}]
    role_ids = {role: {row["person_id"] for row in active_u_people if row["role"] == role} for role in ("actor", "director")}
    by_work, awards, credits, episodes, statuses = (defaultdict(list) for _ in range(5))
    for fact in sources["facts"]:
        if fact.get("kind") == "award":
            awards[fact.get("recipient_id")].append(fact)
        elif fact.get("kind") == "credit":
            credits[(fact.get("person_id"), fact.get("role"))].append(fact)
        elif fact.get("kind") == "episode":
            episodes[fact.get("parent_work_id")].append(fact)
        elif fact.get("kind") == "series_status":
            statuses[fact.get("parent_work_id")].append(fact)
        else:
            by_work[fact.get("subject_id", fact.get("work_id"))].append(fact)
    blank = build_item_facts({}, None, [], target_at=target_at, feature_role=feature_role, temporal_mode=temporal_mode)
    blank_values, _ = _population_row(blank, {}, [])
    columns = list(blank_values)
    matrix = np.zeros((vocab, len(columns)), dtype=np.float32)
    mask = np.zeros(matrix.shape, dtype=np.uint8)
    provenance, uncovered, coverage = {}, [], Counter()
    for iid, token in sorted(mapping.items(), key=lambda pair: pair[1]):
        link = links.get(iid)
        scoped_people = [row for row in people.get(iid, []) if row.get("work_id") in {None, link.get("work_id")}] if link else []
        row_people = {"links": {role: [p for p in scoped_people if p.get("role") == role] for role in ("actor", "director")}}
        for role in ("actor", "director"):
            entries = row_people["links"][role]
            if entries:
                row_people[role] = sorted({p["person_id"] for p in entries})
        work = link.get("work_id") if link else None
        row_facts = list(by_work.get(work, [])) + list(awards.get(work, [])) + list(episodes.get(work, [])) + list(statuses.get(work, []))
        for role in ("actor", "director"):
            for person in row_people.get(role, []):
                row_facts.extend(awards.get(person, []))
                row_facts.extend(credits.get((person, role), []))
        result = build_item_facts(items[iid] | {"item_id": iid}, link, row_facts, target_at=target_at,
            feature_role=feature_role, temporal_mode=temporal_mode, scope=link.get("tv_scope") if link else None,
            people=row_people, collections=sources.get("collections", {}))
        values, known = _population_row(result, items[iid], scoped_people,
            work_conflict=iid in sources.get("conflicts", {}).get("work_conflicts", {}))
        if list(values) != columns:
            raise ValueError("external population schema changed between rows")
        matrix[token] = [values[key] for key in columns]
        mask[token] = [known[key] for key in columns]
        if link:
            provenance[iid] = result
            coverage["accepted_link_item_n"] += 1
            coverage["runtime_known_item_n"] += int(not result["duration"]["missing"])
            coverage["work_runtime_proxy_known_item_n"] += int(not result["work_runtime_proxy"]["missing"])
            coverage["original_language_known_item_n"] += int(not result["languages"]["original"]["missing"])
            for role in ("actor", "director"):
                coverage[role + "_linked_item_n"] += bool(row_people.get(role))
                coverage[role + "_filmography_known_item_n"] += any(not profile.get("missing", 1) for profile in result["people"].get(role, {}).values())
        else:
            uncovered.append(iid)
    if not np.isfinite(matrix).all() or np.any(matrix[list(special)] != 0) or np.any(mask[list(special)] != 0):
        raise ValueError("invalid population matrix or special rows")
    return {"contract": POPULATION_CONTRACT, "features": matrix, "known": mask, "columns": columns,
        "feature_families": {"work_runtime_proxy": WORK_RUNTIME_PROXY_FAMILY,
            "runtime": {"scope": "accepted_edition_or_confirmed_episode_scope", "work_proxy_fallback": False},
            "external_format_profiles": {"mapping_version": "external-format-direct-P31-v1", "categories": ["film", "series"],
                "source": "direct nondeprecated P31 only", "subclass_inheritance": False, "full_career": False},
            "external_genre_profiles": {"categories": EXTERNAL_GENRE_LABELS, "source": "reviewed finite direct P136 QID mapping",
                "complete_taxonomy": False, "unmapped_is_absence": False, "fractions_are_probabilities": False}},
        "metadata_mapping_sha256": canonical_hash(mapping), "feature_role": feature_role,
        "used_item_metadata_sha256": canonical_hash({iid: {key: items[iid].get(key) for key in
            ("actors", "directors", "release_year", "content_type")} for iid in sorted(mapping)}),
        "target_at": target_at, "feature_boundary_at": feature_boundary(target_at, feature_role).isoformat(),
        "temporal_mode": temporal_mode, "coverage": dict(coverage) | {"universe_item_n": len(mapping),
            "uncovered_item_n": len(uncovered), "accepted_distinct_person_n": len(role_ids["actor"] | role_ids["director"]),
            "actor_distinct_person_n": len(role_ids["actor"]), "director_distinct_person_n": len(role_ids["director"]),
            "full_universe_reviewed": False,
            "actor_raw_award_assertion_n": len({f["evidence_id"] for f in sources["facts"] if f.get("kind") == "award" and
                f.get("recipient_id") in role_ids["actor"]}),
            "director_raw_award_assertion_n": len({f["evidence_id"] for f in sources["facts"] if f.get("kind") == "award" and
                f.get("recipient_id") in role_ids["director"]}),
            "source_assertion_kind_counts": dict(Counter(f.get("kind") for f in sources["facts"]))}, "provenance": {"accepted_items": provenance,
            "uncovered_item_ids": uncovered, "input_pins": sources.get("input_pins", {}),
            "conflicts": sources.get("conflicts", {}), "diagnostics": sources.get("diagnostics", []),
            "accepted_work_credit_corpus": sources.get("accepted_work_credit_corpus", {}),
            "source_assertion_index": [{key: fact.get(key) for key in ("kind", "evidence_id", "subject_id", "person_id",
                "work_id", "source_id", "property", "source_revision", "source_known_at", "response_sha256", "award_id", "award_status")}
                for fact in sources["facts"]]}}


def save_population(result, output_dir):
    """Publish one new immutable matrix/mask/provenance bundle; manifest last."""
    import numpy as np
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    np.save(output_dir / "features.npy", result["features"], allow_pickle=False)
    np.save(output_dir / "known.npy", result["known"], allow_pickle=False)
    write_json(output_dir / "provenance.json", result["provenance"])
    (output_dir / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    manifest = {key: value for key, value in result.items() if key not in {"features", "known", "provenance"}}
    manifest["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["shape"] = list(result["features"].shape)
    manifest["payloads"] = {name: {"sha256": hashlib.sha256((output_dir / name).read_bytes()).hexdigest(),
        "bytes": (output_dir / name).stat().st_size} for name in ("features.npy", "known.npy", "provenance.json", "source_snapshot.py")}
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def population_main(argv):
    parser = argparse.ArgumentParser(description="Offline accepted-source full-U tensor builder")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--items-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-at", default="2021-08-09T00:00:00Z")
    parser.add_argument("--feature-role", default="sft_validation")
    parser.add_argument("--temporal-mode", choices=MODES, default="snapshot_retrospective")
    args = parser.parse_args(argv)
    spec = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    with args.items_csv.open(encoding="utf-8", newline="") as stream:
        items = {str(row["item_id"]): row for row in csv.DictReader(stream)}
    sources = load_population_sources(spec, root=args.root, temporal_mode=args.temporal_mode)
    for path in (args.source_manifest, args.metadata, args.items_csv):
        sources["input_pins"][path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = build_population(metadata, items, sources, target_at=args.target_at,
        feature_role=args.feature_role, temporal_mode=args.temporal_mode)
    manifest = save_population(result, args.output_dir)
    print(json.dumps({"output": args.output_dir.as_posix(), "shape": manifest["shape"],
        "coverage": manifest["coverage"]}, ensure_ascii=False))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "population":
        return population_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--links", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--historical-archive", type=Path, help="Exact revision manifest; cache-dir must contain its API response bodies")
    parser.add_argument("--items-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--additional-facts", type=Path, help="Reviewed normalized evidence, not resolver proposals")
    parser.add_argument("--credit-corpus", type=Path, help="Accepted bounded external-work/credit/query ledger")
    parser.add_argument("--credit-cache-dir", type=Path)
    parser.add_argument("--credit-archive", type=Path)
    parser.add_argument("--award-taxonomy", type=Path, help="Versioned reviewed category metadata keyed by external award ID")
    parser.add_argument("--target-at", default="2021-08-09T00:00:00Z")
    parser.add_argument("--feature-role", default="sft_validation")
    parser.add_argument("--temporal-mode", choices=MODES, default="strict_asof")
    args = parser.parse_args(argv)
    links = json.loads(args.links.read_text(encoding="utf-8"))
    if args.historical_archive:
        archive = json.loads(args.historical_archive.read_text(encoding="utf-8"))
        ingestion = ingest_archived_wikidata(links, archive, args.cache_dir)
    else:
        ingestion = ingest_wikidata(links, args.cache_dir)
    collections = dict(links.get("collections", {}))
    if args.credit_corpus:
        if not args.credit_cache_dir:
            parser.error("--credit-corpus requires --credit-cache-dir")
        corpus = json.loads(args.credit_corpus.read_text(encoding="utf-8"))
        credit_archive = json.loads(args.credit_archive.read_text(encoding="utf-8")) if args.credit_archive else None
        credits = ingest_credit_corpus(corpus, args.credit_cache_dir, archive=credit_archive)
        ingestion["facts"].extend(credits["facts"])
        ingestion["diagnostics"].extend(credits["diagnostics"])
        ingestion["credit_query_manifest"] = credits["query_manifest"] | {
            "input_path": args.credit_corpus.as_posix(),
            "input_sha256": hashlib.sha256(args.credit_corpus.read_bytes()).hexdigest()}
        collections.update(credits["collections"])
    if args.additional_facts:
        additional = json.loads(args.additional_facts.read_text(encoding="utf-8"))
        ingestion["facts"].extend(additional["facts"])
    if args.award_taxonomy:
        taxonomy = json.loads(args.award_taxonomy.read_text(encoding="utf-8"))
        ingestion["facts"] = apply_award_taxonomy(ingestion["facts"], taxonomy.get("categories", {}))
    with args.items_csv.open(encoding="utf-8", newline="") as stream:
        items = {str(row["item_id"]): row for row in csv.DictReader(stream)}
    results = []
    for link in links.get("works", []):
        if _accepted(link) and str(link["item_id"]) in items:
            item_id = str(link["item_id"])
            item_people = {"links": {}}
            for role in ("actor", "director"):
                accepted_people = [p for p in links.get("people", []) if _accepted(p)
                    and str(p.get("item_id")) == item_id and p.get("role") == role]
                if accepted_people:
                    item_people[role] = sorted({p["person_id"] for p in accepted_people})
                    item_people["links"][role] = accepted_people
            results.append(build_item_facts(items[str(link["item_id"])], link, ingestion["facts"],
                target_at=args.target_at, feature_role=args.feature_role, temporal_mode=args.temporal_mode,
                scope=link.get("tv_scope"), people=item_people,
                collections=collections))
    report = {"contract": CONTRACT, "temporal_mode": args.temporal_mode,
        "items": results, "ingestion": ingestion, "accepted_item_n": len(results),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": {name: {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("links", args.links), ("items_csv", args.items_csv),
                ("historical_archive", args.historical_archive), ("credit_corpus", args.credit_corpus),
                ("credit_archive", args.credit_archive), ("award_taxonomy", args.award_taxonomy),
                ("additional_facts", args.additional_facts)) if path is not None},
        "readiness": "pilot_partial", "blockers": ["historical source/link availability requires dated evidence",
            "runtime edition and TV scope require independent acceptance", "filmography query coverage not inferred from cached works"]}
    report["content_sha256"] = canonical_hash(report)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing facts snapshot")
    write_json(args.output, report)
    print(json.dumps({"output": args.output.as_posix(), "accepted_item_n": len(results),
        "assertion_n": len(ingestion["facts"]), "temporal_mode": args.temporal_mode}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
