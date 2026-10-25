"""Bounded, review-gated KION -> Wikidata identity acquisition (stdlib only).

Search results are proposals, never accepted identities. Exact names are scoped
to a credit mention and an independently accepted work. Current source snapshots
cannot establish historical availability. No credentials or bulk endpoint used.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from types import SimpleNamespace
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid

CONTRACT = "kion-entity-links-v1"
IDENTITY_RULE = "kion-identity-screen-v2"
API = "https://www.wikidata.org/w/api.php"
SPARQL = "https://query.wikidata.org/sparql"
FIELDS = ("title", "title_orig", "release_year", "content_type", "directors", "actors")
QID = re.compile(r"Q[1-9][0-9]*\Z")
TYPES = {"film": {"Q11424", "Q506240", "Q24869", "Q202866", "Q29168811"},
         "series": {"Q5398426", "Q581714", "Q1259759", "Q3464665"}}


def utcnow():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_or_match(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError("Immutable artifact mismatch: " + path.as_posix())
    else:
        write_new(path, value)


def read_ledger(path):
    path = Path(path)
    rows, previous = [], None
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            check = {k: v for k, v in row.items() if k != "entry_sha256"}
            if row.get("previous_sha256") != previous or row.get("entry_sha256") != digest(check):
                raise ValueError("Ledger hash chain mismatch")
            rows.append(row)
            previous = row["entry_sha256"]
    return rows


def append_ledger(path, value):
    rows = read_ledger(path)
    row = dict(value, previous_sha256=rows[-1]["entry_sha256"] if rows else None)
    row["entry_sha256"] = digest(row)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(row) + "\n")
    return row


def normal(value):
    # Candidate evidence only: no transliteration, initials expansion or ё/е merging.
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def year(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number.is_integer() and 1 <= number <= 9999 else None


def claims(entity, prop):
    return [c for c in entity.get("claims", {}).get(prop, []) if c.get("rank") != "deprecated"
            and c.get("mainsnak", {}).get("snaktype") == "value"]


def values(entity, prop):
    return [c["mainsnak"]["datavalue"]["value"] for c in claims(entity, prop)
            if "datavalue" in c["mainsnak"]]


def refs(entity, prop):
    return sorted({v["id"] for v in values(entity, prop) if isinstance(v, dict) and QID.fullmatch(v.get("id", ""))})


def names(entity):
    result = []
    for lang, val in entity.get("labels", {}).items():
        result.append({"language": lang, "kind": "label", "value": val["value"]})
    for lang, vals in entity.get("aliases", {}).items():
        result.extend({"language": lang, "kind": "alias", "value": v["value"]} for v in vals)
    result.extend({"language": "und", "kind": "P1476", "value": v["text"]}
                  for v in values(entity, "P1476") if isinstance(v, dict) and "text" in v)
    return result


def external_ids(entity, kind):
    result, conflicts = {"wikidata": entity["id"]}, []
    properties = {"P345": ("imdb", r"nm[0-9]+" if kind == "person" else r"tt[0-9]+"),
                  "P4985": ("tmdb_person", r"[1-9][0-9]*"),
                  "P4947": ("tmdb_movie", r"[1-9][0-9]*"),
                  "P4983": ("tmdb_tv", r"[1-9][0-9]*")}
    for prop, (key, pattern) in properties.items():
        vals = sorted({str(v) for v in values(entity, prop)})
        wrong_kind = (kind == "person" and key in ("tmdb_movie", "tmdb_tv")) or (kind != "person" and key == "tmdb_person")
        if vals and (wrong_kind or len(vals) != 1 or not re.fullmatch(pattern, vals[0])):
            conflicts.append({"property": prop, "values": vals, "reason": "namespace_or_multiple_values"})
        elif vals:
            result[key] = vals[0]
    result["wikipedia"] = {k: {"title": v["title"], "pageid": None, "valid_from": None, "valid_to": None,
        "identity_status": "retrospective", "redirect_history_verified": False, "historical_evidence": [], "url": v.get("url") or
        "https://" + k.removesuffix("wiki") + ".wikipedia.org/wiki/" + urllib.parse.quote(v["title"].replace(" ", "_"))}
        for k, v in entity.get("sitelinks", {}).items() if k in ("ruwiki", "enwiki")}
    return result, conflicts


def sparql_code(query):
    """Remove quoted strings/IRIs for the bounded generated-SELECT guard."""
    result, index = [], 0
    while index < len(query):
        char = query[index]
        if char == "#":
            raise ValueError("SPARQL comments prohibited")
        if char in ("'", '"') or (char == "<" and index + 1 < len(query) and query[index + 1] not in "= \t\r\n\"'"):
            end = ">" if char == "<" else char
            if char != "<" and query[index:index + 3] == char * 3:
                raise ValueError("SPARQL multiline strings prohibited")
            index += 1
            while index < len(query) and query[index] != end:
                index += 2 if query[index] == "\\" else 1
            if index >= len(query):
                raise ValueError("Unclosed SPARQL literal/IRI")
            result.append(" ")
        else:
            result.append(char)
        index += 1
    return "".join(result)


class Client:
    """One owner per output dir; durable request/byte budget includes failures.

    No automatic retry: 429, maxlag, 5xx and parse failures stop acquisition.
    A fresh process reuses verified successful cache entries without network.
    """
    def __init__(self, root, *, max_requests=30, max_bytes=200_000_000,
                 response_bytes=8_000_000, min_interval=1.0, timeout=25,
                 user_agent="KionEntityResearch/1.0 (local noncommercial pilot)", offline=False,
                 transport=None):
        if not 1 <= max_requests <= 100_000 or not 1 <= max_bytes <= 200_000_000:
            raise ValueError("Invalid request/byte budget")
        if not 1 <= response_bytes <= max_bytes or min_interval < 1 or not 1 <= timeout <= 60:
            raise ValueError("Invalid per-response/rate/timeout limits")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".request-owner.lock"
        self.owner_token, self.closed = uuid.uuid4().hex, False
        with self.lock.open("x", encoding="utf-8") as stream:
            stream.write(self.owner_token)
        self.receipts = self.root / "receipts.jsonl"
        self.max_requests, self.max_bytes, self.response_bytes = max_requests, max_bytes, response_bytes
        self.interval, self.timeout, self.ua = min_interval, timeout, user_agent
        self.offline, self.transport, self.last = offline, transport or urllib.request.urlopen, 0.0

    def close(self):
        if self.closed:
            return
        if self.lock.read_text(encoding="utf-8") != self.owner_token:
            raise RuntimeError("Acquisition lock ownership changed")
        self.lock.unlink()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def request(self, params, *, endpoint=API):
        if self.closed:
            raise RuntimeError("Acquisition client is closed")
        if endpoint == API:
            params = dict(params, format="json", maxlag="5")
            archival = (params.get("action") == "query" and params.get("prop") == "revisions"
                and QID.fullmatch(params.get("titles", "")) and params.get("rvlimit") == "1"
                and params.get("rvdir") == "older" and params.get("rvslots") == "main"
                and params.get("rvprop") == "ids|timestamp|content|contentmodel")
            if params.get("action") not in ("wbsearchentities", "wbgetentities") and not archival:
                raise ValueError("Only read-only identity/revision API methods allowed")
        elif endpoint == SPARQL:
            params = dict(params, format="json")
            query = params.get("query", "")
            code = sparql_code(query)
            if not query.lstrip().upper().startswith("SELECT ") or len(query) > 40_000:
                raise ValueError("Only bounded SELECT queries allowed")
            if not re.search(r"\bLIMIT\s+[1-9][0-9]{0,3}\s*$", code, re.IGNORECASE):
                raise ValueError("SPARQL requires explicit bounded LIMIT")
            if re.search(r"\b(SERVICE|INSERT|DELETE|LOAD|CLEAR|CREATE|DROP)\b", code, re.IGNORECASE):
                raise ValueError("SPARQL external services/mutations prohibited")
        else:
            raise ValueError("Unapproved identity endpoint")
        request_key = digest(params) if endpoint == API else digest({"endpoint": endpoint, "params": params})
        rows = read_ledger(self.receipts)
        for row in reversed(rows):
            if row.get("request_key") == request_key and row.get("status") == "ok":
                payload = (self.root / "cache" / (row["response_sha256"] + ".json")).read_bytes()
                if digest(payload) != row["response_sha256"]:
                    raise ValueError("Cached response hash mismatch")
                return json.loads(payload), row
        if self.offline:
            raise RuntimeError("Offline cache miss")
        if any(r.get("event") == "request_finished" and r.get("status") == "failed" for r in rows):
            raise RuntimeError("Previous acquisition failure; inspect receipt before a new run")
        attempts = sum(r.get("event") == "request_started" for r in rows)
        used = sum(max(r.get("decoded_bytes", 0), r.get("wire_bytes", 0)) for r in rows)
        unfinished = {r["request_id"] for r in rows if r.get("event") == "request_started"} - {
            r["request_id"] for r in rows if r.get("event") == "request_finished"}
        if unfinished:
            raise RuntimeError("Interrupted request: byte consumption unknown; new run directory required")
        if attempts >= self.max_requests or used >= self.max_bytes - 1:
            raise RuntimeError("Acquisition budget exhausted")
        cap = min(self.response_bytes, self.max_bytes - used - 1)
        delay = self.interval - (time.monotonic() - self.last)
        finished = [r for r in rows if r.get("event") == "request_finished"]
        if finished:
            last_finish = datetime.fromisoformat(finished[-1]["retrieved_at"].replace("Z", "+00:00")).timestamp()
            delay = max(delay, self.interval - (time.time() - last_finish))
        if delay > 0:
            time.sleep(delay)
        rid = str(attempts + 1)
        encoded = urllib.parse.urlencode(sorted(params.items())).encode("ascii")
        get_url = endpoint + "?" + encoded.decode("ascii")
        method = "POST" if endpoint == SPARQL else "GET"
        url = endpoint if method == "POST" else get_url
        query_hash = digest(params["query"].encode("utf-8")) if "query" in params else None
        append_ledger(self.receipts, {"event": "request_started", "request_id": rid,
            "request_key": request_key, "params": params, "endpoint": endpoint, "started_at": utcnow(),
            "method": method, "url": url, "query_sha256": query_hash,
            "equivalent_get_uri_bytes": len(get_url.encode("utf-8")), "request_body_bytes": len(encoded) if method == "POST" else 0})
        headers = {"User-Agent": self.ua, "Accept": "application/json", "Accept-Encoding": "gzip"}
        if method == "POST":
            headers["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
        req = urllib.request.Request(url, data=encoded if method == "POST" else None, headers=headers, method=method)
        receipt = {"event": "request_finished", "request_id": rid, "request_key": request_key,
            "endpoint": endpoint, "params": params, "retrieved_at": utcnow(), "status": "failed",
            "method": method, "url": url, "query_sha256": query_hash,
            "decoded_bytes": 0, "wire_bytes": 0, "license": "CC0-1.0 (structured Wikidata only)"}
        try:
            with self.transport(req, timeout=self.timeout) as response:
                receipt["http_status"] = response.status
                receipt["retry_after"] = response.headers.get("Retry-After")
                raw = response.read(cap + 1)
                receipt["wire_bytes"] = len(raw)
                if len(raw) > cap:
                    receipt["decoded_bytes"] = cap
                    raise RuntimeError("Response wire byte limit exceeded")
                encoding = response.headers.get("Content-Encoding", "identity")
                if encoding not in ("identity", "gzip"):
                    raise RuntimeError("Unexpected content encoding")
                if encoding == "gzip":
                    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as compressed:
                        payload = compressed.read(cap + 1)
                else:
                    payload = raw
                receipt["decoded_bytes"] = min(len(payload), cap)
                if len(payload) > cap:
                    raise RuntimeError("Response decoded byte limit exceeded")
            value = json.loads(payload)
            if "error" in value or "warnings" in value:
                raise RuntimeError("Wikidata API error/warning: " + str(value.get("error", value.get("warnings"))))
            checksum = digest(payload)
            dest = self.root / "cache" / (checksum + ".json")
            dest.parent.mkdir(exist_ok=True)
            if dest.exists():
                if dest.read_bytes() != payload:
                    raise ValueError("Cache collision")
            else:
                with dest.open("xb") as stream:
                    stream.write(payload)
            receipt.update(status="ok", response_sha256=checksum, retrieved_at=utcnow())
        except Exception as exc:
            receipt["retrieved_at"] = utcnow()
            receipt["timestamp_semantics"] = "failure_response_observed"
            receipt["error"] = str(exc)
            if isinstance(exc, urllib.error.HTTPError):
                receipt.update(http_status=exc.code, retry_after=exc.headers.get("Retry-After"))
            append_ledger(self.receipts, receipt)
            raise
        finally:
            self.last = time.monotonic()
        return value, append_ledger(self.receipts, receipt)

    def search(self, text, language):
        return self.request({"action": "wbsearchentities", "search": text, "language": language,
                             "uselang": language, "type": "item", "limit": "5"})

    def entities(self, ids):
        ids = sorted(set(ids))
        if not ids or len(ids) > 50 or any(not QID.fullmatch(q) for q in ids):
            raise ValueError("Entity batch must contain 1..50 valid QIDs")
        return self.request({"action": "wbgetentities", "ids": "|".join(ids),
            "props": "info|labels|aliases|descriptions|claims|sitelinks", "languages": "ru|en",
            "sitefilter": "ruwiki|enwiki"})

    def revision_before(self, qid, cutoff):
        if not QID.fullmatch(qid):
            raise ValueError("Invalid entity QID")
        before = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
        if before.tzinfo is None:
            raise ValueError("Revision cutoff needs explicit timezone")
        start = (before.astimezone(timezone.utc) - timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
        return self.request({"action": "query", "prop": "revisions", "titles": qid,
            "rvstart": start, "rvdir": "older", "rvlimit": "1", "rvslots": "main",
            "rvprop": "ids|timestamp|content|contentmodel", "formatversion": "2"})

    def select(self, query):
        return self.request({"query": query}, endpoint=SPARQL)


def parse_revision(value, receipt, qid, cutoff):
    """Preserve raw old content; revision metadata is not inferred from retrieval."""
    pages = value.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    if len(pages) != 1 or pages[0].get("title") != qid:
        raise ValueError("Historical response page identity mismatch")
    revisions = pages[0].get("revisions", [])
    if not revisions:
        return {"status": "no_visible_revision_before_cutoff", "entity": None, "source_evidence": None}
    if len(revisions) != 1:
        raise ValueError("Expected one historical revision")
    revision = revisions[0]
    timestamp, revision_id = revision.get("timestamp"), revision.get("revid")
    observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00")) if timestamp else None
    boundary = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    if (observed is None or observed.tzinfo is None or boundary.tzinfo is None or observed >= boundary
            or not isinstance(revision_id, int) or isinstance(revision_id, bool)):
        raise ValueError("Historical revision must strictly predate cutoff")
    content = revision.get("slots", {}).get("main", {}).get("content")
    if not isinstance(content, str):
        return {"status": "revision_content_unavailable", "entity": None, "source_evidence": None}
    entity = json.loads(content)
    if entity.get("id") != qid or entity.get("type") != "item":
        raise ValueError("Historical entity content identity/type mismatch")
    evidence = {"source": "Wikidata", "entity_qid": qid, "entity_revision": revision_id,
        "revision_timestamp": timestamp, "retrieved_at": receipt["retrieved_at"],
        "response_sha256": receipt["response_sha256"], "source_known_at": timestamp,
        "revision_url": f"https://www.wikidata.org/w/index.php?oldid={revision_id}",
        "temporal_mode": "historical_source", "license": "CC0-1.0"}
    return {"status": "archived", "entity": entity, "revision_id": revision_id,
        "revision_timestamp": timestamp, "source_evidence": evidence,
        "body_cache_path": "cache/" + receipt["response_sha256"] + ".json",
        "entity_content_sha256": digest(content.encode("utf-8")), "cutoff": cutoff}


def archive_entities(root, qids, cutoff, *, identity_links=None, **options):
    root = Path(root)
    qids = sorted(set(qids))
    if not qids or any(not QID.fullmatch(q) for q in qids):
        raise ValueError("Historical archive requires valid QIDs")
    plan = {"contract": CONTRACT, "operation": "historical_entities", "qids": qids, "cutoff": cutoff,
        "options": {k: v for k, v in options.items() if k != "offline"}, "resolver_sha256": file_hash(__file__),
        "identity_links_sha256": file_hash(identity_links) if identity_links else None,
        "identity_join_policy": "current stable identity remains separate from old numeric assertion clock"}
    write_or_match(root / "plan.json", plan)
    archived = {}
    with Client(root, **options) as client:
        for qid in qids:
            value, receipt = client.revision_before(qid, cutoff)
            archived[qid] = parse_revision(value, receipt, qid, cutoff)
    result = {"contract": CONTRACT, "cutoff": cutoff, "entities": archived,
        "plan_sha256": file_hash(root / "plan.json"), "identity_links_sha256": plan["identity_links_sha256"],
        "identity_join_policy": plan["identity_join_policy"]}
    write_or_match(root / "archive.json", result)
    return result


def prior_credit_query(person_qid, role, before_year, limit=16):
    if not QID.fullmatch(person_qid) or role not in ("director", "actor") or not 1 <= before_year <= 9999 or not 1 <= limit <= 100:
        raise ValueError("Invalid bounded prior-credit query")
    prop = "P57" if role == "director" else "P161"
    return ("SELECT ?work (MIN(?date) AS ?first_release) WHERE { ?work "
        f"<http://www.wikidata.org/prop/direct/{prop}> <http://www.wikidata.org/entity/{person_qid}> ; "
        "<http://www.wikidata.org/prop/direct/P577> ?date . FILTER(?date < "
        f'"{before_year:04d}-01-01T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>) '
        "} GROUP BY ?work ORDER BY ?first_release ?work LIMIT " + str(limit))


def collect_prior_credits(root, identity_links, *, per_person=15, old_works_per_person=2,
                          cutoff="2021-08-09T00:00:00Z", **options):
    """Bounded observed graph, not complete filmography or proven non-KION works."""
    if not 1 <= per_person <= 99 or not 0 <= old_works_per_person <= per_person:
        raise ValueError("Invalid credit collection bound")
    root = Path(root)
    links = json.loads(Path(identity_links).read_text(encoding="utf-8"))
    works_by_id = {w["work_id"]: w for w in links["works"] if w.get("status") == "accepted"}
    people = [p for p in links["people"] if p.get("status") == "accepted"]
    plan = {"contract": CONTRACT, "operation": "bounded_prior_credits", "identity_links_sha256": file_hash(identity_links),
        "per_person_limit": per_person, "historical_works_per_person": old_works_per_person,
        "cutoff": cutoff, "people": [{"qid": p["qid"], "role": p["role"], "anchor_work": p["work_id"]} for p in people],
        "resolver_sha256": file_hash(__file__), "options": options}
    write_or_match(root / "plan.json", plan)
    collections, selected, work_ids = {}, {}, set()
    with Client(root, **options) as client:
        for person in people:
            anchor = works_by_id[person["work_id"]]
            query = prior_credit_query(person["qid"], person["role"], anchor["release_year"], per_person + 1)
            value, receipt = client.select(query)
            bindings = value.get("results", {}).get("bindings")
            if not isinstance(bindings, list):
                raise ValueError("Invalid bounded credit query result")
            candidates = []
            for row in bindings:
                uri = row.get("work", {}).get("value", "")
                qid = uri.removeprefix("http://www.wikidata.org/entity/")
                if not QID.fullmatch(qid):
                    raise ValueError("Unexpected SPARQL work URI")
                if "wd:" + qid != person["work_id"]:
                    candidates.append(qid)
            candidates = list(dict.fromkeys(candidates))
            key = person["person_id"] + ":" + person["role"] + ":" + person["work_id"]
            selected[key] = candidates[:per_person]
            work_ids.update(selected[key])
            collections[key] = {"person_id": person["person_id"], "person_qid": person["qid"], "role": person["role"],
                "anchor_work_id": person["work_id"], "prior_release_year_exclusive": anchor["release_year"],
                "source_available": True, "source_collection_complete_for_query": len(bindings) < per_person + 1,
                "query_truncated": len(bindings) >= per_person + 1, "source_filmography_complete": False,
                "query_response_sha256": receipt["response_sha256"], "query_receipts": [receipt],
                "observed_candidate_count": len(candidates), "selected_candidate_count": len(selected[key])}
        entities, evidence = {}, {}
        for start in range(0, len(work_ids), 50):
            value, receipt = client.entities(sorted(work_ids)[start:start + 50])
            for qid, entity in value.get("entities", {}).items():
                if "missing" not in entity:
                    entities[qid], evidence[qid] = entity, receipt
        credits, accepted = [], {}
        old_ids = set()
        for key, qids in selected.items():
            collection = collections[key]
            # Raw observations can share a work across anchors; consumers dedup
            # on (person_id, role, work_id), keeping anchor provenance separately.
            prop = "P57" if collection["role"] == "director" else "P161"
            valid = []
            for qid in qids:
                entity = entities.get(qid, {})
                types = set(refs(entity, "P31"))
                content_type = "film" if types & TYPES["film"] else ("series" if types & TYPES["series"] else "unknown")
                if collection["person_qid"] not in refs(entity, prop) or content_type == "unknown":
                    continue
                valid.append(qid)
                c = work_candidate({}, entity, evidence[qid])
                accepted[qid] = {"qid": qid, "work_id": "wd:" + qid, "status": "accepted_by_source_rule",
                    "content_type": content_type, "item_id": None, "candidate": c,
                    "kion_membership": "unknown", "entity_basis": "explicit typed work and confirmed-person role credit"}
                credits.append({"work_id": "wd:" + qid, "person_id": collection["person_id"], "role": collection["role"],
                    "anchor_work_id": collection["anchor_work_id"],
                    "status": "accepted_by_source_rule", "source_known_at": None,
                    "query_response_sha256": collection["query_response_sha256"],
                    "credit_statement_ids": c["evidence"]["role_credit_statement_ids"][collection["role"]][collection["person_qid"]]})
            old_ids.update(valid[:old_works_per_person])
            collection["accepted_typed_credit_count"] = len(valid)
        archive = {}
        for qid in sorted(old_ids):
            try:
                value, receipt = client.revision_before(qid, cutoff)
                archive[qid] = parse_revision(value, receipt, qid, cutoff)
            except Exception as exc:
                archive[qid] = {"status": "acquisition_unavailable", "entity": None, "source_evidence": None,
                    "reason": str(exc), "not_observed_zero": True}
                for pending in sorted(old_ids - set(archive)):
                    archive[pending] = {"status": "not_attempted_after_failure", "entity": None, "source_evidence": None}
                break
    write_or_match(root / "archive.json", {"contract": CONTRACT, "cutoff": cutoff, "entities": archive,
        "identity_join_policy": "current accepted person identity plus observed old credit graph; KION membership unknown"})
    result = {"contract": CONTRACT, "works": list(accepted.values()), "credits": credits,
        "collections": collections, "identity_links_sha256": file_hash(identity_links), "historical_archive": "archive.json",
        "historical_collection_status": "complete_for_selected_queries" if all(v["status"] == "archived" for v in archive.values()) else "partial",
        "scope": "bounded selected prior credits; not complete careers; work membership in KION unknown"}
    write_or_match(root / "credit-corpus.json", result)
    return result


def batch_work_query(rows, limit=201):
    if not 1 <= len(rows) <= 50 or not 1 <= limit <= 9999:
        raise ValueError("Bounded batch requires 1..50 cards and explicit result limit")
    literals = []
    for item_id, raw in rows.items():
        y = year(raw.get("release_year"))
        if y is None:
            continue
        for field, language in (("title_orig", "en"), ("title", "ru")):
            text = raw.get(field, "").strip()
            if text:
                literals.append(f"({json.dumps(item_id)} {json.dumps(text, ensure_ascii=False)}@{language} {y})")
    if not literals:
        raise ValueError("No title/year literals in batch")
    return ("SELECT DISTINCT ?item ?work WHERE { VALUES (?item ?label ?year) { " + " ".join(literals) +
        " } ?work (<http://www.w3.org/2000/01/rdf-schema#label>|<http://www.w3.org/2004/02/skos/core#altLabel>|"
        "<http://www.wikidata.org/prop/direct/P1476>) ?label ; <http://www.wikidata.org/prop/direct/P577> ?date . "
        "FILTER(YEAR(?date) = ?year) } ORDER BY ?item ?work LIMIT " + str(limit))


def acquire_batch(items_path, metadata_path, root, item_ids, *, max_people=50, result_limit=201, shared_client=None, **options):
    root = Path(root)
    rows, pins = load_inputs(items_path, metadata_path, item_ids)
    query = batch_work_query(rows, result_limit)
    plan = {"contract": CONTRACT, "operation": "batch_title_year_candidates", "items": rows, "input_pins": pins,
        "query": query, "result_limit": result_limit, "max_people": max_people,
        "options": options, "resolver_sha256": file_hash(__file__), "automatic_acceptance": False,
        "cache_root": shared_client.root.as_posix() if shared_client else root.as_posix()}
    write_or_match(root / "plan.json", plan)
    with (nullcontext(shared_client) if shared_client else Client(root, **options)) as client:
        value, query_receipt = client.select(query)
        bindings = value.get("results", {}).get("bindings", [])
        candidates = {i: set() for i in rows}
        for row in bindings:
            item_id = row.get("item", {}).get("value")
            qid = row.get("work", {}).get("value", "").removeprefix("http://www.wikidata.org/entity/")
            if item_id not in rows or not QID.fullmatch(qid):
                raise ValueError("Unexpected batch candidate binding")
            candidates[item_id].add(qid)
        qids = sorted({q for qs in candidates.values() for q in qs})
        entities, evidence = {}, {}
        for start in range(0, len(qids), 50):
            body, receipt = client.entities(qids[start:start + 50])
            for qid, entity in body.get("entities", {}).items():
                if "missing" not in entity:
                    entities[qid], evidence[qid] = entity, receipt
        directors = sorted({q for entity in entities.values() for q in refs(entity, "P57")})[:max_people]
        people = {}
        for start in range(0, len(directors), 50):
            body, receipt = client.entities(directors[start:start + 50])
            for qid, entity in body.get("entities", {}).items():
                if "missing" not in entity:
                    people[qid], evidence[qid] = entity, receipt
    proposals = []
    for item_id, raw in rows.items():
        found = [work_candidate(raw, entities[q], evidence[q], people) for q in sorted(candidates[item_id]) if q in entities]
        strong = [c for c in found if c["identity_screen"]["eligible_default_all_directors"]]
        proposals.append({"record_type": "work_link", "item_id": item_id, "raw": raw, "work_id": None,
            "edition_id": "kion:item:" + item_id, "edition_relation": "unverified", "candidates": found,
            "status": "unmatched" if not found else ("ambiguous" if len(strong) > 1 else "proposed"),
            "query_complete_for_limit": len(bindings) < result_limit, "search_is_exhaustive": False})
    result = {"contract": CONTRACT, "plan_sha256": file_hash(root / "plan.json"), "works": proposals,
        "people_entities": {q: {"entity": e, "source_evidence": source_evidence(e, evidence[q])} for q, e in people.items()},
        "batch": {"response_sha256": query_receipt["response_sha256"], "returned_bindings": len(bindings),
            "result_limit": result_limit, "truncated": len(bindings) >= result_limit,
            "scope": "exact language-tagged title/year candidates; zero is not source absence"}}
    write_or_match(root / "proposals.json", result)
    return result


def load_inputs(items_path, metadata_path, item_ids):
    before = {"items_sha256": file_hash(items_path), "metadata_sha256": file_hash(metadata_path)}
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    mapping = metadata["item_id_to_token"]
    if sorted(mapping.values()) != list(range(3, metadata["vocab_size"])):
        raise ValueError("Invalid train-known mapping")
    selected = list(dict.fromkeys(str(x) for x in item_ids))
    if not selected or any(i not in mapping for i in selected):
        raise ValueError("Pilot item IDs must belong to train-known universe")
    rows, selected_set = {}, set(selected)
    with Path(items_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(("item_id", *FIELDS)) <= set(reader.fieldnames):
            raise ValueError("Invalid item headers")
        seen = set()
        for row in reader:
            key = row.get("item_id")
            if not key or key in seen or None in row or any(row.get(f) is None for f in FIELDS):
                raise ValueError("Malformed or duplicate item row")
            seen.add(key)
            if key in selected_set:
                rows[key] = {k: row[k] for k in FIELDS}
    if set(rows) != set(selected):
        raise ValueError("Selected item missing in CSV")
    if before != {"items_sha256": file_hash(items_path), "metadata_sha256": file_hash(metadata_path)}:
        raise ValueError("Identity input changed while reading")
    return rows, {**before,
                  "universe_count": len(mapping), "universe_sha256": digest(mapping)}


def source_evidence(entity, receipt):
    return {"source": "Wikidata", "entity_qid": entity["id"], "entity_revision": entity.get("lastrevid"),
        "revision_timestamp": entity.get("modified"), "retrieved_at": receipt["retrieved_at"],
        "response_sha256": receipt["response_sha256"], "source_known_at": None,
        "revision_url": ("https://www.wikidata.org/w/index.php?oldid=" + str(entity["lastrevid"])) if entity.get("lastrevid") else None,
        "temporal_mode": "snapshot_retrospective", "license": "CC0-1.0"}


def work_candidate(raw, entity, receipt, people=None):
    people = people or {}
    wanted = {normal(raw[k]) for k in ("title", "title_orig") if raw.get(k)}
    matches = [n for n in names(entity) if normal(n["value"]) in wanted]
    years = sorted({year(v["time"][1:5]) for v in values(entity, "P577")
        if isinstance(v, dict) and re.fullmatch(r"\+[0-9]{4}-[0-9]{2}-[0-9]{2}T00:00:00Z", v.get("time", ""))
        and isinstance(v.get("precision"), int) and v["precision"] >= 9
        and v.get("before", 0) == 0 and v.get("after", 0) == 0} - {None})
    directors = refs(entity, "P57")
    role_matches = {}
    for role, field, prop in (("director", "directors", "P57"), ("actor", "actors", "P161")):
        role_matches[role] = [{"raw_name": raw_name.strip(), "qid": q, "name_evidence": n,
            "is_human": "Q5" in refs(people.get(q, {}), "P31"),
            "person_external_id_conflicts": external_ids(people[q], "person")[1] if q in people else []}
            for raw_name in raw.get(field, "").split(",") for q in refs(entity, prop) for n in names(people.get(q, {}))
            if raw_name.strip() and normal(raw_name) == normal(n["value"])]
    external, conflicts = external_ids(entity, "work")
    candidate = {"qid": entity["id"], "source_evidence": source_evidence(entity, receipt), "external_ids": external,
        "external_id_conflicts": conflicts, "evidence": {"title_matches": matches, "release_years": years,
            "year_exact": year(raw.get("release_year")) in years, "year_near_only": not year(raw.get("release_year")) in years
                and any(abs(y - (year(raw.get("release_year")) or -9999)) <= 1 for y in years),
            "instance_of": refs(entity, "P31"), "type_exact": bool(set(refs(entity, "P31")) & TYPES.get(raw.get("content_type"), set())),
            "director_qids": directors, "director_matches": role_matches["director"], "actor_matches": role_matches["actor"],
            "role_credit_statement_ids": {role: {q: [c.get("id") for c in claims(entity, prop)
                if c["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == q] for q in refs(entity, prop)}
                for role, prop in (("director", "P57"), ("actor", "P161"))},
            "credit_statement_ids": [c.get("id") for p in ("P57", "P161") for c in claims(entity, p)]},
        "confidence": {"work": "unreviewed", "person": None, "credit": None}}
    candidate["identity_screen"] = screen_work_identity(raw, candidate, people)
    return candidate


def edition_scope_flags(raw):
    text = normal(raw.get("title", "") + " " + raw.get("title_orig", ""))
    patterns = {"resolution_4k": r"4k|4к", "extended_cut": r"extended|ultimate|расширенн|director.{0,3}cut",
        "audio_description": r"тифло|audio description", "sign_language": r"сурдо|sign language",
        "language_or_subtitle_variant": r"на [а-яё\s-]{2,60} языке|субтитр|subtitl|in english|original language",
        "full_version": r"полная версия|full version|complete version",
        "colour_variant": r"цветная версия|colourized|colorized|ч[её]рно.белая версия|black.and.white.version",
        "unspecified_version_marker": r"\bверсия\b|\bversion\b",
        "season_scope": r"season|сезон"}
    return sorted(k for k, pattern in patterns.items() if re.search(pattern, text))


def screen_work_identity(raw, candidate, people):
    """Conservative work screen; person conflicts cannot support an anchor.

    This screening never grants acceptance. Default support must cover every
    raw director. Source incompleteness and incorrect KION credits are distinct
    hypotheses requiring an explicit reviewed-partial policy, never a silent OR.
    """
    evidence, reasons = candidate["evidence"], []
    for key, reason in (("title_matches", "title_not_supported"), ("year_exact", "release_year_not_exact"),
                        ("type_exact", "content_type_unverified")):
        if not evidence.get(key):
            reasons.append(reason)
    if candidate.get("external_id_conflicts"):
        reasons.append("work_identifier_conflict")
    if not candidate.get("source_evidence", {}).get("entity_revision"):
        reasons.append("work_revision_unavailable")
    raw_names = list(dict.fromkeys(n.strip() for n in raw.get("directors", "").split(",") if n.strip()))
    if not raw_names:
        reasons.append("no_raw_director_anchor")
    support, valid_count = [], 0
    for name in raw_names:
        qids = sorted({m["qid"] for m in evidence.get("director_matches", []) if normal(m["raw_name"]) == normal(name)})
        row = {"raw_name": name, "qids": qids, "unique_supported": False, "person_identifier_conflicts": []}
        if len(qids) > 1:
            reasons.append("director_alias_ambiguous")
        if len(qids) == 1:
            person = people.get(qids[0], {})
            person = person.get("entity", person)
            row["person_identifier_conflicts"] = external_ids(person, "person")[1] if person.get("id") else []
            human = "Q5" in refs(person, "P31")
            revision = person.get("lastrevid")
            if row["person_identifier_conflicts"]:
                reasons.append("anchor_person_identifier_conflict")
            if not human:
                reasons.append("anchor_person_not_verified_human")
            if not revision:
                reasons.append("anchor_person_revision_unavailable")
            row["unique_supported"] = human and bool(revision) and not row["person_identifier_conflicts"]
            valid_count += int(row["unique_supported"])
        support.append(row)
    if not support or any(not r["unique_supported"] for r in support):
        reasons.append("not_all_raw_directors_uniquely_supported")
    if valid_count == 0:
        reasons.append("no_valid_director_anchor")
    return {"rule_version": IDENTITY_RULE, "director_policy": "all_supported",
        "blocking_reasons": sorted(set(reasons)), "eligible_default_all_directors": not reasons,
        "director_support": support, "valid_director_anchor_count": valid_count,
        "edition_scope_flags": edition_scope_flags(raw), "edition_identity_status": "unverified",
        "confidence_is_calibrated_probability": False}


def acquire(items_path, metadata_path, output, item_ids, *, max_people=20, **client_options):
    root = Path(output)
    rows, pins = load_inputs(items_path, metadata_path, item_ids)
    if isinstance(max_people, bool) or not isinstance(max_people, int) or not 0 <= max_people <= 1000:
        raise ValueError("max_people must be between 0 and 1000")
    plan = {"contract": CONTRACT, "items": rows, "input_pins": pins, "max_people": max_people,
        "resolver_sha256": file_hash(__file__),
        "options": {k: v for k, v in client_options.items() if k != "offline"}, "temporal_mode": "snapshot_retrospective", "automatic_acceptance": False}
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Plan mismatch; use new output directory")
    else:
        write_new(plan_path, plan)
    with Client(root, **client_options) as client:
        searches, found, search_complete = {}, set(), {}
        for item_id, raw in rows.items():
            ids = set()
            search_complete[item_id] = True
            for field, lang in (("title_orig", "en"), ("title", "ru")):
                if raw[field].strip():
                    value, _ = client.search(raw[field], lang)
                    search_complete[item_id] &= len(value.get("search", [])) < 5 and "search-continue" not in value
                    ids.update(v["id"] for v in value.get("search", []) if QID.fullmatch(v.get("id", "")))
            searches[item_id] = sorted(ids)
            found.update(ids)
        entities, evidence = {}, {}
        for start in range(0, len(found), 50):
            value, receipt = client.entities(sorted(found)[start:start + 50])
            for q, entity in value.get("entities", {}).items():
                if "missing" not in entity:
                    entities[q], evidence[q] = entity, receipt
        directors = sorted({q for item_id, ids in searches.items() for candidate_id in ids
            if candidate_id in entities and work_candidate(rows[item_id], entities[candidate_id], evidence[candidate_id])["evidence"]["year_exact"]
            for q in refs(entities[candidate_id], "P57")})
        actors = sorted({q for item_id, ids in searches.items() for candidate_id in ids
            if candidate_id in entities and work_candidate(rows[item_id], entities[candidate_id], evidence[candidate_id])["evidence"]["year_exact"]
            for q in refs(entities[candidate_id], "P161")})
        credit_people = (directors + [q for q in actors if q not in directors])[:max_people]
        people = {}
        for start in range(0, len(credit_people), 50):
            value, receipt = client.entities(credit_people[start:start + 50])
            for q, entity in value.get("entities", {}).items():
                if "missing" not in entity:
                    people[q], evidence[q] = entity, receipt
        proposals = []
        for item_id, raw in rows.items():
            candidates = [work_candidate(raw, entities[q], evidence[q], people) for q in searches[item_id] if q in entities]
            strong = [c for c in candidates if c["identity_screen"]["eligible_default_all_directors"]]
            proposals.append({"record_type": "work_link", "item_id": item_id, "work_id": None,
                "edition_id": "kion:item:" + item_id, "edition_relation": "unverified", "raw": raw,
                "status": "unmatched" if not candidates else ("ambiguous" if len(strong) > 1 else "proposed"),
                "candidates": candidates, "query_complete_for_limit": search_complete[item_id],
                "search_is_exhaustive": False})
        payload = {"contract": CONTRACT, "plan_sha256": file_hash(plan_path), "works": proposals,
            "people_entities": {q: {"entity": e, "source_evidence": source_evidence(e, evidence[q])} for q, e in people.items()}}
        path = root / "proposals.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError("Immutable proposals differ; use new run directory")
        else:
            write_new(path, payload)
        return payload


def review_work(root, item_id, qid, *, reviewer, reason, status="accepted", edition_relation="unverified",
                director_policy="all_supported", partial_director_reason=None):
    """Explicit review, never invoked by acquisition; supported multi-field identity required."""
    if not reviewer.strip() or not reason.strip() or status not in ("accepted", "rejected", "ambiguous", "unmatched"):
        raise ValueError("Review requires named reviewer, reason and valid decision")
    if edition_relation not in ("unverified", "same_work", "distinct_work", "component"):
        raise ValueError("Invalid edition relation")
    if director_policy not in ("all_supported", "reviewed_partial"):
        raise ValueError("Invalid director evidence policy")
    if director_policy == "reviewed_partial" and not str(partial_director_reason or "").strip():
        raise ValueError("Partial director policy requires an explicit reviewed reason")
    root = Path(root)
    proposals = json.loads((root / "proposals.json").read_text(encoding="utf-8"))
    proposal = next(p for p in proposals["works"] if p["item_id"] == str(item_id))
    candidate = next((c for c in proposal["candidates"] if c["qid"] == qid), None)
    screen = screen_work_identity(proposal["raw"], candidate, proposals["people_entities"]) if candidate else None
    if status == "accepted":
        if not candidate:
            raise ValueError("Candidate not present")
        blocking = set(screen["blocking_reasons"])
        if director_policy == "reviewed_partial":
            blocking.discard("not_all_raw_directors_uniquely_supported")
        if blocking:
            raise ValueError("Work/director acceptance blocked: " + ", ".join(sorted(blocking)))
    return append_ledger(root / "decisions.jsonl", {"record_type": "work_decision", "item_id": str(item_id),
        "qid": qid, "status": status, "reviewer": reviewer, "reason": reason, "reviewed_at": utcnow(),
        "proposals_sha256": file_hash(root / "proposals.json"), "edition_relation": edition_relation,
        "identity_rule_version": IDENTITY_RULE, "director_policy": director_policy,
        "partial_director_reason": partial_director_reason, "identity_screen": screen})


def export_links(root, output):
    """Accepted links only; person mentions resolved through accepted work credits."""
    root = Path(root)
    proposals = json.loads((root / "proposals.json").read_text(encoding="utf-8"))
    latest, person_decisions = {}, {}
    for row in read_ledger(root / "decisions.jsonl"):
        if row["proposals_sha256"] != file_hash(root / "proposals.json"):
            raise ValueError("Decision proposals hash mismatch")
        if row["record_type"] == "work_decision":
            latest[row["item_id"]] = row
        elif row["record_type"] == "person_decision":
            person_decisions[row["mention_id"]] = row
    works, mentions = [], []
    for proposal in proposals["works"]:
        decision = latest.get(proposal["item_id"], {})
        if decision.get("status") != "accepted":
            continue
        candidate = next(c for c in proposal["candidates"] if c["qid"] == decision["qid"])
        works.append({"item_id": proposal["item_id"], "work_id": "wd:" + candidate["qid"],
            "content_type": proposal["raw"]["content_type"], "release_year": year(proposal["raw"]["release_year"]), "raw": proposal["raw"],
            "qid": candidate["qid"], "status": "accepted", "candidate": candidate, "decision": decision,
            "edition_id": proposal["edition_id"], "edition_relation": decision["edition_relation"],
            "edition_specific_identity": False, "temporal_mode": "snapshot_retrospective", "source_known_at": None})
        for role, field in (("director", "directors"), ("actor", "actors")):
          for position, raw_name in enumerate(proposal["raw"][field].split(",")):
            if not raw_name.strip():
                continue
            matches = [m for m in candidate["evidence"].get(role + "_matches", []) if normal(m["raw_name"]) == normal(raw_name)]
            matched = sorted({m["qid"] for m in matches})
            q = matched[0] if len(matched) == 1 else None
            entity = proposals["people_entities"].get(q, {})
            is_human = "Q5" in refs(entity.get("entity", {}), "P31")
            external, conflicts = external_ids(entity["entity"], "person") if entity else ({}, [])
            mentions.append({"mention_id": f"kion:mention:{proposal['item_id']}:{role}:{position}",
                "item_id": proposal["item_id"], "role": role, "raw_name": raw_name.strip(), "position": position,
                "person_id": None, "qid": q, "status": "proposed" if q and is_human and not conflicts else ("unmatched" if not matched else "ambiguous"),
                "work_id": "wd:" + candidate["qid"], "external_ids": external, "external_id_conflicts": conflicts,
                "source_evidence": entity.get("source_evidence"), "name_evidence": matches,
                "credit_statement_ids": candidate["evidence"].get("role_credit_statement_ids", {}).get(role, {}).get(q, []),
                "confidence": {"person": "unreviewed", "credit": "unreviewed"},
                "source_known_at": None, "temporal_mode": "snapshot_retrospective"})
    accepted_people = []
    for mention in mentions:
        decision = person_decisions.get(mention["mention_id"], {})
        anchor = latest.get(mention["item_id"], {})
        if (decision.get("status") == "accepted" and mention["status"] == "proposed" and decision["qid"] == mention["qid"]
                and decision["anchor_work_decision_sha256"] == anchor.get("entry_sha256")):
            accepted_people.append(dict(mention, status="accepted", person_id="wd:" + mention["qid"],
                decision=decision, confidence={"person": "reviewed_evidence_not_probability", "credit": "reviewed_evidence_not_probability"}))
    result = {"contract": CONTRACT, "works": works, "people": accepted_people, "person_proposals": mentions,
        "proposals_sha256": file_hash(root / "proposals.json"), "decisions_sha256": file_hash(root / "decisions.jsonl") if (root / "decisions.jsonl").exists() else None,
        "coverage": {"selected_items": len(proposals["works"]), "accepted_items": len(works),
            "accepted_distinct_work_ids": len({w["work_id"] for w in works}), "accepted_people": len(accepted_people),
            "accepted_distinct_people": len({p["person_id"] for p in accepted_people}),
            "scope": "purposive bounded pilot; not population precision/coverage", "historical_available": 0}}
    write_new(output, result)
    return result


def review_person(root, mention_id, qid, *, reviewer, reason, status="accepted"):
    """Review a scoped actor/director credit; no global raw-name mapping."""
    if status not in ("accepted", "rejected", "ambiguous") or not reviewer.strip() or not reason.strip():
        raise ValueError("Invalid person review")
    root = Path(root)
    parts = mention_id.split(":")
    if len(parts) != 5 or parts[:2] != ["kion", "mention"] or parts[3] not in ("director", "actor") or not parts[4].isdigit():
        raise ValueError("Only exact scoped actor/director mentions supported")
    item_id, role, position = parts[2], parts[3], int(parts[4])
    proposals = json.loads((root / "proposals.json").read_text(encoding="utf-8"))
    proposal = next(p for p in proposals["works"] if p["item_id"] == item_id)
    latest = [r for r in read_ledger(root / "decisions.jsonl") if r.get("record_type") == "work_decision" and r["item_id"] == item_id]
    if not latest or latest[-1]["status"] != "accepted":
        raise ValueError("Person anchor work must be independently accepted first")
    candidate = next(c for c in proposal["candidates"] if c["qid"] == latest[-1]["qid"])
    raw_name = proposal["raw"]["directors" if role == "director" else "actors"].split(",")[position].strip()
    matched = {m["qid"] for m in candidate["evidence"].get(role + "_matches", []) if normal(m["raw_name"]) == normal(raw_name)}
    person = proposals["people_entities"].get(qid, {})
    external, conflicts = external_ids(person["entity"], "person") if person else ({}, [])
    if status == "accepted" and (matched != {qid} or "Q5" not in refs(person.get("entity", {}), "P31") or conflicts):
        raise ValueError("Person acceptance requires unique exact alias, human type and correct-role credit")
    return append_ledger(root / "decisions.jsonl", {"record_type": "person_decision", "mention_id": mention_id,
        "qid": qid, "status": status, "reviewer": reviewer, "reason": reason, "reviewed_at": utcnow(),
        "anchor_work_decision_sha256": latest[-1]["entry_sha256"], "proposals_sha256": file_hash(root / "proposals.json")})


def plan_catalog_batches(items_path, metadata_path, output, *, batch_size=50, accepted_links=None):
    """No network: full-universe deterministic batches and unresolved rows."""
    if not 1 <= batch_size <= 50:
        raise ValueError("Catalog batch size must be 1..50")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    rows, pins = load_inputs(items_path, metadata_path, list(metadata["item_id_to_token"]))
    accepted = set()
    if accepted_links:
        accepted = {w["item_id"] for w in json.loads(Path(accepted_links).read_text(encoding="utf-8"))["works"]
                    if w.get("status") == "accepted"}
    tiers, unresolved = {0: [], 1: [], 2: []}, []
    for item_id in sorted(rows, key=lambda v: (0, int(v)) if v.isdigit() else (1, v)):
        raw = rows[item_id]
        if item_id in accepted:
            continue
        if year(raw.get("release_year")) is None or not (raw.get("title_orig", "").strip() or raw.get("title", "").strip()):
            unresolved.append({"item_id": item_id, "reason": "missing_title_or_exact_year"})
        else:
            tier = 0 if raw["title_orig"].strip() and raw["directors"].strip() else (1 if raw["directors"].strip() else 2)
            tiers[tier].append(item_id)
    batches = []
    for tier, ids in tiers.items():
        for start in range(0, len(ids), batch_size):
            chosen = ids[start:start + batch_size]
            query = batch_work_query({i: rows[i] for i in chosen}, max(201, len(chosen) * 10 + 1))
            if len(query) > 40_000:
                raise ValueError("Catalog batch exceeds query bound; lower batch size")
            batches.append({"batch_index": len(batches), "tier": tier, "item_ids": chosen,
                "query_sha256": digest(query.encode("utf-8")), "query_utf8_bytes": len(query.encode("utf-8"))})
    result = {"contract": CONTRACT, "operation": "full_U_batch_plan_no_network", "input_pins": pins,
        "resolver_sha256": file_hash(__file__), "batch_size": batch_size, "universe_items": len(rows),
        "already_accepted_items": len(accepted), "eligible_items": sum(len(v) for v in tiers.values()),
        "tier_item_counts": {str(k): len(v) for k, v in tiers.items()}, "unresolved": unresolved, "batches": batches,
        "candidate_query_count": len(batches), "additional_entity_batches": "measured after candidate IDs",
        "automatic_acceptance": False, "query_completeness": "bounded literal matches; saturated batches split; zero remains unmatched"}
    write_new(output, result)
    return result


def _cached_entity_pool(client):
    entities, evidence = {}, {}
    for receipt in read_ledger(client.receipts):
        if receipt.get("status") != "ok" or receipt.get("params", {}).get("action") != "wbgetentities":
            continue
        payload = (client.root / "cache" / (receipt["response_sha256"] + ".json")).read_bytes()
        if digest(payload) != receipt["response_sha256"]:
            raise ValueError("Pooled entity cache hash mismatch")
        for qid, entity in json.loads(payload).get("entities", {}).items():
            if "missing" not in entity:
                entities[qid], evidence[qid] = entity, receipt
    return entities, evidence


def _remaining_requests(client):
    return client.max_requests - sum(r.get("event") == "request_started" for r in read_ledger(client.receipts))


def _validate_candidate_result(body, item_ids):
    if not isinstance(body, dict) or not isinstance(body.get("results"), dict):
        raise ValueError("Invalid pooled candidate result schema")
    bindings = body["results"].get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("Invalid pooled candidate bindings")
    for binding in bindings:
        if not isinstance(binding, dict) or not isinstance(binding.get("item"), dict) or not isinstance(binding.get("work"), dict):
            raise ValueError("Invalid pooled candidate binding shape")
        qid = binding["work"].get("value", "").removeprefix("http://www.wikidata.org/entity/")
        if binding["item"].get("value") not in item_ids or not QID.fullmatch(qid):
            raise ValueError("Pooled candidate identity mismatch")
    return bindings


def collect_catalog_slice(root, items_path, metadata_path, *, reserve_people_requests=5, review_seed=20261020):
    """Continue a successful two-batch smoke with one global budget and pool.

    Work and director retrieval are batched across cards. Reaching a budget or
    endpoint failure publishes explicit partial evidence; it never accepts links.
    """
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    smoke = json.loads((root / "smoke-completion-r1.json").read_text(encoding="utf-8"))
    if smoke.get("status") != "smoke_passed" or len(smoke.get("batches", [])) != 2:
        raise ValueError("Two full smoke batches must pass before continuation")
    batches = plan["selected_batches"]
    all_ids = [str(i) for b in batches for i in b["item_ids"]]
    if len(set(all_ids)) != len(all_ids) or len(batches) > 20 or len(all_ids) > 1000:
        raise ValueError("Slice must contain <=20 disjoint batches and <=1000 cards")
    rows, pins = load_inputs(items_path, metadata_path, all_ids)
    continuation = {"contract": CONTRACT, "operation": "pooled_catalog_continuation",
        "root_plan_sha256": file_hash(root / "plan.json"), "resolver_sha256": file_hash(__file__),
        "input_pins": pins, "max_requests": plan["max_requests"], "max_bytes": plan["max_bytes"],
        "reserve_people_requests": reserve_people_requests, "review_seed": review_seed,
        "automatic_acceptance": False}
    write_or_match(root / "continuation-plan-r1.json", continuation)
    snapshot = root / "resolver-pool-source.py"
    if snapshot.exists():
        if file_hash(snapshot) != file_hash(__file__):
            raise ValueError("Frozen pooled source mismatch")
    else:
        with snapshot.open("xb") as stream:
            stream.write(Path(__file__).read_bytes())
    query_data, collection_error, budget_stops, query_order_bridge = {}, None, [], []
    with Client(root, max_requests=plan["max_requests"], max_bytes=plan["max_bytes"],
                min_interval=plan["min_interval_seconds"]) as client:
        for index in range(2):
            source = json.loads((root / f"smoke-{index:02d}" / "proposals.json").read_text(encoding="utf-8"))
            smoke_plan = json.loads((root / f"smoke-{index:02d}" / "plan.json").read_text(encoding="utf-8"))
            chosen_ids = [str(i) for i in batches[index]["item_ids"]]
            csv_order = {i: row for i, row in rows.items() if i in set(chosen_ids)}
            canonical_order = {i: rows[i] for i in chosen_ids}
            limit = source["batch"]["result_limit"]
            expected_actual = batch_work_query(csv_order, limit)
            canonical_sha = digest(batch_work_query(canonical_order, limit).encode("utf-8"))
            if smoke_plan.get("query") != expected_actual or canonical_sha != batches[index]["query_sha256"]:
                raise ValueError("Smoke query scope/order bridge mismatch")
            sha = source["batch"]["response_sha256"]
            source_receipts = [r for r in read_ledger(client.receipts) if r.get("response_sha256") == sha and r.get("status") == "ok"]
            if not source_receipts or source_receipts[-1]["params"].get("query") != expected_actual:
                raise ValueError("Smoke actual query receipt mismatch")
            query_order_bridge.append({"batch_index": index,
                "actual_query_sha256": digest(expected_actual.encode("utf-8")), "canonical_query_sha256": canonical_sha,
                "items_sha256": digest(canonical_order), "response_sha256": sha,
                "reason": "identical VALUES rows in CSV versus planned ID order; order has no SPARQL matching semantics"})
            body = (root / "cache" / (sha + ".json")).read_bytes()
            if digest(body) != sha:
                raise ValueError("Smoke query cache mismatch")
            parsed = json.loads(body)
            _validate_candidate_result(parsed, {str(i) for i in batches[index]["item_ids"]})
            query_data[index] = {"body": parsed, "response_sha256": sha,
                "result_limit": source["batch"]["result_limit"]}
        write_or_match(root / "query-order-bridge-r1.json", {"contract": CONTRACT, "entries": query_order_bridge})
        try:
            for index, batch in enumerate(batches[2:], 2):
                if _remaining_requests(client) <= 0:
                    budget_stops.append("candidate_query_budget")
                    break
                chosen = {str(i): rows[str(i)] for i in batch["item_ids"]}
                limit = max(201, len(chosen) * 10 + 1)
                query = batch_work_query(chosen, limit)
                if digest(query.encode("utf-8")) != batch["query_sha256"]:
                    raise ValueError("Preflight query differs from frozen full-U plan")
                body, receipt = client.select(query)
                _validate_candidate_result(body, set(chosen))
                query_data[index] = {"body": body, "response_sha256": receipt["response_sha256"], "result_limit": limit}
                write_or_match(root / f"query-{index:02d}.json", {"batch_index": index, "receipt": receipt})
            entities, evidence = _cached_entity_pool(client)
            needed = set()
            for index, data in query_data.items():
                valid_ids = {str(i) for i in batches[index]["item_ids"]}
                bindings = data["body"].get("results", {}).get("bindings")
                if not isinstance(bindings, list):
                    raise ValueError("Invalid pooled candidate result schema")
                for binding in bindings:
                    qid = binding.get("work", {}).get("value", "").removeprefix("http://www.wikidata.org/entity/")
                    if binding.get("item", {}).get("value") not in valid_ids or not QID.fullmatch(qid):
                        raise ValueError("Pooled candidate identity mismatch")
                    needed.add(qid)
            missing = sorted(needed - set(entities))
            for start in range(0, len(missing), 50):
                if _remaining_requests(client) <= reserve_people_requests:
                    budget_stops.append("work_entity_budget_reserving_people")
                    break
                client.entities(missing[start:start + 50])
            entities, evidence = _cached_entity_pool(client)
            incidence = {}
            for qid in needed & set(entities):
                for director in refs(entities[qid], "P57"):
                    if director not in entities:
                        incidence[director] = incidence.get(director, 0) + 1
            people_ids = sorted(incidence, key=lambda q: (-incidence[q], q))
            for start in range(0, len(people_ids), 50):
                if _remaining_requests(client) <= 0:
                    budget_stops.append("director_entity_budget")
                    break
                client.entities(people_ids[start:start + 50])
        except Exception as exc:
            # No more HTTP after a failure. Publishing saved observations is local.
            collection_error = {"type": type(exc).__name__, "reason": str(exc)}
        entities, evidence = _cached_entity_pool(client)
    proposals, strata, all_candidate_qids = [], {}, set()
    for index, batch in enumerate(batches):
        data = query_data.get(index)
        by_item = {str(i): set() for i in batch["item_ids"]}
        if data:
            for binding in data["body"].get("results", {}).get("bindings", []):
                item_id = binding.get("item", {}).get("value")
                qid = binding.get("work", {}).get("value", "").removeprefix("http://www.wikidata.org/entity/")
                if item_id in by_item and QID.fullmatch(qid):
                    by_item[item_id].add(qid)
                    all_candidate_qids.add(qid)
        for item_id, qids in by_item.items():
            raw = rows[item_id]
            candidates = [work_candidate(raw, entities[q], evidence[q], entities) for q in sorted(qids & set(entities))]
            strong = [c for c in candidates if c["identity_screen"]["eligible_default_all_directors"]]
            unavailable = sorted(qids - set(entities))
            edition_risk = bool(edition_scope_flags(raw))
            if not data:
                stratum = "query_not_collected"
            elif not qids:
                stratum = "query_unmatched"
            elif unavailable:
                stratum = "entity_not_collected"
            elif edition_risk:
                stratum = "edition_scope_risk"
            elif len(strong) > 1:
                stratum = "ambiguous_multiple_strong"
            elif not any(c["evidence"]["type_exact"] for c in candidates):
                stratum = "type_conflict_or_unknown"
            elif not strong:
                stratum = "multi_evidence_incomplete"
            else:
                stratum = "unique_strong_unreviewed"
            strata.setdefault(stratum, []).append(item_id)
            proposals.append({"record_type": "work_link", "item_id": item_id, "raw": raw, "work_id": None,
                "edition_id": "kion:item:" + item_id, "edition_relation": "unverified", "candidates": candidates,
                "status": "unmatched" if not candidates else ("ambiguous" if len(strong) > 1 else "proposed"),
                "candidate_qids": sorted(qids), "unfetched_candidate_qids": unavailable,
                "source_query_available": data is not None, "batch_index": index, "review_stratum": stratum,
                "strong_candidate_qids": [c["qid"] for c in strong], "search_is_exhaustive": False,
                "query_complete_for_limit": bool(data and len(data["body"].get("results", {}).get("bindings", [])) < data["result_limit"])})
    result = {"contract": CONTRACT, "plan_sha256": file_hash(root / "plan.json"), "works": proposals,
        "people_entities": {q: {"entity": e, "source_evidence": source_evidence(e, evidence[q])}
            for q, e in entities.items() if "Q5" in refs(e, "P31")},
        "cache_root": root.as_posix(), "automatic_acceptance": False, "collection_error": collection_error}
    write_or_match(root / "proposals.json", result)
    sampled = []
    for stratum, ids in sorted(strata.items()):
        ranked = sorted(ids, key=lambda i: digest(f"{review_seed}:{stratum}:{i}".encode("utf-8")))
        count = 20 if stratum == "unique_strong_unreviewed" else 5
        sampled.extend({"item_id": i, "stratum": stratum} for i in ranked[:count])
    write_or_match(root / "review-sample-r1.json", {"contract": CONTRACT, "seed": review_seed,
        "proposals_sha256": file_hash(root / "proposals.json"), "mutually_exclusive_strata": True,
        "stratum_population_counts": {k: len(v) for k, v in strata.items()}, "sample": sampled,
        "purpose": "fixed stratified identity QA before any acceptance rule; no population precision claimed"})
    receipts = read_ledger(root / "receipts.jsonl")
    missing_works = sorted(all_candidate_qids - set(entities))
    missing_directors = sorted({q for w in all_candidate_qids & set(entities) for q in refs(entities[w], "P57")} - set(entities))
    saturated = [i for i, data in query_data.items() if len(data["body"]["results"]["bindings"]) >= data["result_limit"]]
    incomplete = len(query_data) != len(batches) or bool(missing_works or missing_directors or saturated)
    summary = {"contract": CONTRACT, "status": "partial" if collection_error or incomplete else "bounded_collection_completed",
        "source_sha256": file_hash(__file__), "planned_cards": len(all_ids), "queried_cards": sum(len(batches[i]["item_ids"]) for i in query_data),
        "batches_queried": len(query_data), "strata": {k: len(v) for k, v in strata.items()},
        "actual_requests": sum(r.get("event") == "request_started" for r in receipts),
        "wire_bytes": sum(r.get("wire_bytes", 0) for r in receipts), "decoded_bytes": sum(r.get("decoded_bytes", 0) for r in receipts),
        "collection_error": collection_error, "accepted_links": 0,
        "source_evidence_incomplete": incomplete, "budget_stop_reasons": budget_stops,
        "unfetched_work_qids": missing_works, "unfetched_director_qids": missing_directors,
        "saturated_batch_indexes": saturated, "unique_candidate_qids": len(all_candidate_qids),
        "proposals_sha256": file_hash(root / "proposals.json"), "receipts_sha256": file_hash(root / "receipts.jsonl"),
        "review_sample_sha256": file_hash(root / "review-sample-r1.json"), "review_sample_rows": len(sampled)}
    write_or_match(root / "collection-completion-r1.json", summary)
    return summary


def rescreen_saved_proposals(source_path, cache_roots, output_root):
    """Offline policy revision; old proposals, decisions and caches stay intact."""
    source_path, output_root = Path(source_path), Path(output_root)
    original = json.loads(source_path.read_text(encoding="utf-8"))
    entities, evidence, cache_pins = {}, {}, []
    for location in cache_roots:
        location = Path(location)
        if (location / ".request-owner.lock").exists():
            raise RuntimeError("Cannot rescreen a live acquisition owner")
        local_entities, local_evidence = _cached_entity_pool(SimpleNamespace(root=location, receipts=location / "receipts.jsonl"))
        entities.update(local_entities)
        evidence.update(local_evidence)
        cache_pins.append({"root": location.as_posix(), "receipts_sha256": file_hash(location / "receipts.jsonl")})
    plan = {"contract": CONTRACT, "operation": "offline_identity_policy_rescreen", "identity_rule_version": IDENTITY_RULE,
        "source_proposals_sha256": file_hash(source_path), "cache_roots": cache_pins,
        "resolver_sha256": file_hash(__file__), "new_http_requests": 0, "automatic_acceptance": False}
    write_new(output_root / "plan.json", plan)
    with (output_root / "resolver-source.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    works, counts = [], {}
    for old in original["works"]:
        row = dict(old)
        candidates = []
        for previous in old["candidates"]:
            qid = previous["qid"]
            if qid not in entities:
                raise ValueError("Saved candidate entity is missing from verified cache")
            if evidence[qid]["response_sha256"] != previous["source_evidence"]["response_sha256"]:
                raise ValueError("Rescreen would change a frozen work source snapshot")
            candidates.append(work_candidate(row["raw"], entities[qid], evidence[qid], entities))
        strong = [c for c in candidates if c["identity_screen"]["eligible_default_all_directors"]]
        conflict = any("anchor_person_identifier_conflict" in c["identity_screen"]["blocking_reasons"]
            and c["evidence"]["title_matches"] and c["evidence"]["year_exact"] and c["evidence"]["type_exact"] for c in candidates)
        if not row.get("source_query_available", True):
            stratum = "query_not_collected"
        elif not candidates:
            stratum = "query_unmatched"
        elif row.get("unfetched_candidate_qids"):
            stratum = "entity_not_collected"
        elif conflict:
            stratum = "anchor_person_identifier_conflict"
        elif edition_scope_flags(row["raw"]):
            stratum = "edition_scope_risk"
        elif len(strong) > 1:
            stratum = "ambiguous_multiple_qualified"
        elif not any(c["evidence"]["type_exact"] for c in candidates):
            stratum = "type_conflict_or_unknown"
        elif not strong:
            stratum = "multi_evidence_incomplete"
        else:
            stratum = "unique_qualified_unreviewed"
        row.update(candidates=candidates, strong_candidate_qids=[c["qid"] for c in strong],
            review_stratum_v1=old.get("review_stratum"), review_stratum=stratum,
            identity_rule_version=IDENTITY_RULE, edition_scope_flags=edition_scope_flags(row["raw"]),
            status="unmatched" if not candidates else ("ambiguous" if len(strong) > 1 else "proposed"))
        works.append(row)
        counts[stratum] = counts.get(stratum, 0) + 1
    result = {"contract": CONTRACT, "identity_rule_version": IDENTITY_RULE, "works": works,
        "people_entities": {q: {"entity": e, "source_evidence": source_evidence(e, evidence[q])}
            for q, e in entities.items() if "Q5" in refs(e, "P31")},
        "plan_sha256": file_hash(output_root / "plan.json"), "automatic_acceptance": False, "cache_roots": cache_pins}
    write_new(output_root / "proposals.json", result)
    report = {"contract": CONTRACT, "identity_rule_version": IDENTITY_RULE, "cards": len(works), "strata": counts,
        "qualified_work_candidates": sum(len(w["strong_candidate_qids"]) for w in works),
        "accepted_links": 0, "new_http_requests": 0, "proposals_sha256": file_hash(output_root / "proposals.json")}
    write_new(output_root / "rescreen-completion-r1.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--items", required=True)
    acquire_parser.add_argument("--metadata", required=True)
    acquire_parser.add_argument("--output", required=True)
    acquire_parser.add_argument("--item-ids", nargs="+", required=True)
    acquire_parser.add_argument("--max-requests", type=int, default=30)
    acquire_parser.add_argument("--max-bytes", type=int, default=200_000_000)
    acquire_parser.add_argument("--max-people", type=int, default=20)
    acquire_parser.add_argument("--offline", action="store_true")
    review = sub.add_parser("review-work")
    for key in ("root", "item-id", "qid", "reviewer", "reason"):
        review.add_argument("--" + key, required=True)
    review.add_argument("--status", default="accepted")
    review.add_argument("--edition-relation", default="unverified")
    review.add_argument("--director-policy", default="all_supported")
    review.add_argument("--partial-director-reason")
    person = sub.add_parser("review-person")
    for key in ("root", "mention-id", "qid", "reviewer", "reason"):
        person.add_argument("--" + key, required=True)
    person.add_argument("--status", default="accepted")
    export = sub.add_parser("export")
    export.add_argument("--root", required=True)
    export.add_argument("--output", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "acquire":
        result = acquire(args.pop("items"), args.pop("metadata"), args.pop("output"), args.pop("item_ids"), **args)
        print(canonical({"works": len(result["works"]), "proposed": True}))
    elif command == "review-work":
        print(canonical(review_work(**args)))
    elif command == "review-person":
        print(canonical(review_person(**args)))
    else:
        print(canonical(export_links(**args)["coverage"]))


if __name__ == "__main__":
    main()
