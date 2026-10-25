"""Exact, additive diagnostics for four agreed personal feature contracts.

The normalized event facts and immutable prepared-query join cover all rows.
Values are computed on demand; they are never appended to model coordinates.
KION remains a reconstructed last-pair prefix, with unknown source completeness.
"""
from __future__ import annotations

from array import array
from datetime import date
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_completion_event_store as events

history = events.history
canonical_hash, file_hash = events.canonical_hash, events.file_hash
CONTRACT = "kion-user-diagnostics-v1"
THRESHOLD_CONTRACT = "kion-exact-outcome-threshold-facts-v1"
FAMILIES = ("personal.activity", "personal.typed_completion", "personal.window_completion",
            "personal.age_rating_profile")
SOURCE_START = "2021-03-13"
RATING_REASONS = ("known", "unknown_token", "missing_card", "missing_field", "empty_field", "out_of_schema_raw")
COMPLETION_FIELDS = ("n", "n_valid", "n_missing", "n_invalid", "n_unavailable", "n_high", "n_low",
                     "selection_empty", "outcome_empty", "mean", "high", "low")


def _pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": file_hash(path)}


def _save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def build_threshold_facts(store, output, *, max_seconds=180, max_rss_bytes=3 * 1024 ** 3):
    """Parse source text once, proving every sorted retained row against the store.

    Float32 values are deliberately never used to recover exact >=80/<=10 bits.
    A one-based CSV data-row locator retains access to original raw decimal text.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Threshold output must be a new directory")
    started = time.perf_counter()
    source = store.manifest["source_pins"]["interactions"]
    if file_hash(source["path"]) != source["sha256"]:
        raise ValueError("Raw outcome source SHA mismatch")
    users, items, days = array("q"), array("q"), array("i")
    statuses, bits, outcomes, locators = array("B"), array("B"), array("f"), array("I")
    parsed, dates = {}, {}
    count = 0
    for count, (u, i, d, raw) in enumerate(history._csv_rows(source["path"],
            ("user_id", "item_id", "last_watch_dt", "watched_pct")), 1):
        if count % 100000 == 0:
            if time.perf_counter() - started > max_seconds or events._peak_rss() > max_rss_bytes:
                raise RuntimeError("Threshold source parse exceeded the declared resource budget")
        uid, iid = history._mapping_int(u, "source user"), history._mapping_int(i, "source item")
        if d not in dates:
            dates[d] = history._day(d)
        if dates[d] >= store.cap:
            continue
        if raw not in parsed:
            parsed[raw] = history.parse_outcome(raw)
        value, status, flags = parsed[raw]
        users.append(uid); items.append(iid); days.append(dates[d])
        outcomes.append(value); statuses.append(status); bits.append(flags); locators.append(count)
    arrays = [np.asarray(value) for value in (users, items, days, outcomes, statuses, bits, locators)]
    order = np.lexsort((arrays[1], arrays[2], arrays[0]))
    for name, values in zip(("user_id", "raw_item_id", "date", "outcome", "parser_status"), arrays):
        if not np.array_equal(values[order], store.arrays[name]):
            raise ValueError("Exact threshold source rows disagree with canonical events: " + name)
    if file_hash(source["path"]) != source["sha256"]:
        raise ValueError("Raw outcome source changed while parsing")
    output.mkdir(parents=True)
    declarations = {}
    for name, values in (("threshold_bits", arrays[5][order]), ("source_csv_data_row", arrays[6][order])):
        target = output / (name + ".npy")
        np.save(target, values, allow_pickle=False)
        declarations[name] = {**_pin(target), "dtype": str(values.dtype), "shape": list(values.shape)}
    manifest = {"contract": THRESHOLD_CONTRACT, "state": "completed", "events_count": len(order),
        "source_pins": {"interactions": source, "builder": _pin(__file__),
                        "parser": store.manifest["source_pins"]["history_builder"]},
        "event_store": _pin(store.root / "manifest.json"), "fit_cutoff": store.manifest["fit_cutoff"],
        "parser": "Decimal finite [0,100]; exact raw percent bits; no float32 rethresholding",
        "thresholds": {"high": ">=80", "low": "<=10"}, "raw_text": "pinned interactions CSV; one-based data-row locator; header excluded",
        "join": "exact all retained sorted (user_id,date,raw_item_id,outcome,status); event IDs are canonical store row positions",
        "raw_source_rows_scanned": count, "source_sha256_before": source["sha256"],
        "source_sha256_after": file_hash(source["path"]), "arrays": declarations,
        "elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": events._peak_rss()}
    manifest["signature"] = canonical_hash(manifest)
    _save_json(output / "manifest.json", manifest)
    return manifest


def load_threshold_facts(path, store):
    manifest = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("contract") != THRESHOLD_CONTRACT or manifest.get("state") != "completed"
            or canonical_hash({k: v for k, v in manifest.items() if k != "signature"}) != manifest.get("signature")
            or manifest.get("events_count") != len(store.keys)
            or manifest.get("event_store") != _pin(store.root / "manifest.json")
            or manifest.get("fit_cutoff") != store.manifest["fit_cutoff"]):
        raise ValueError("Threshold event-store binding/signature mismatch")
    for pin in manifest["source_pins"].values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("Threshold source SHA mismatch")
    result = {}
    for name, dtype in (("threshold_bits", "uint8"), ("source_csv_data_row", "uint32")):
        declaration = manifest["arrays"][name]
        target = Path(declaration["path"]).resolve()
        if target.parent != Path(path).resolve() or file_hash(target) != declaration["sha256"]:
            raise ValueError("Threshold payload path/SHA mismatch")
        values = np.load(target, mmap_mode="r", allow_pickle=False)
        if values.shape != (len(store.keys),) or str(values.dtype) != dtype:
            raise ValueError("Threshold payload shape/dtype mismatch")
        result[name] = values
    if np.any(result["threshold_bits"] > 31) or np.any(result["source_csv_data_row"] < 1):
        raise ValueError("Threshold payload range mismatch")
    return result, manifest


class UserDiagnostics:
    """Query API over full canonical events; unknown IDs never regain metadata."""
    def __init__(self, store, threshold_bits, metadata, cards, *, threshold_manifest=None,
                 source_start=SOURCE_START):
        if store.manifest.get("coverage", {}).get("raw_events") != "complete_before_role_cap":
            raise ValueError("Personal diagnostics require the complete available raw prefix")
        self.store = store
        self.bits = np.asarray(threshold_bits)
        if self.bits.shape != (len(store.keys),) or self.bits.dtype != np.dtype("uint8") or np.any(self.bits > 31):
            raise ValueError("Exact threshold facts must align with all canonical events")
        if np.any(self.bits[store.arrays["parser_status"] != 0]):
            raise ValueError("Missing/invalid outcomes cannot have threshold bits")
        self.mapping = {history._mapping_int(k, "metadata raw item"): history._mapping_int(v, "metadata token")
                        for k, v in metadata["item_id_to_token"].items()}
        if (canonical_hash(self.mapping) != store.manifest["mapping_sha256"]
                or set(self.mapping.values()) != set(range(3, len(self.mapping) + 3))):
            raise ValueError("Diagnostic metadata mapping differs from frozen events")
        self.source_start = history._day(source_start)
        self.types = np.full(len(self.mapping) + 3, -1, np.int8)
        self.rating_reason = np.full(len(self.mapping) + 3, 1, np.uint8)
        self.rating = np.full(len(self.mapping) + 3, -1, np.int8)
        out_values = {}
        for iid, token in self.mapping.items():
            card = cards.get(iid)
            if card is None:
                self.rating_reason[token] = 2
                continue
            kind = card.get("content_type")
            if kind is not None and not isinstance(kind, str):
                raise ValueError("Malformed content_type metadata schema")
            if kind is not None and kind.strip() in ("film", "series"):
                self.types[token] = int(kind.strip() == "series")
            if "age_rating" not in card:
                self.rating_reason[token] = 3
                continue
            raw = card["age_rating"]
            if not isinstance(raw, str):
                raise ValueError("Malformed age_rating metadata schema")
            raw = raw.strip()
            if raw in history.AGE_RATINGS:
                self.rating_reason[token] = 0
                self.rating[token] = history.AGE_RATINGS.index(raw)
            elif not raw:
                self.rating_reason[token] = 4
            else:
                self.rating_reason[token] = 5
                out_values[raw] = out_values.get(raw, 0) + 1
        self.out_of_schema_values = [{"raw": raw[:256], "truncated": len(raw) > 256, "catalog_count": n}
                                    for raw, n in sorted(out_values.items())[:100]]
        self.provenance = {"contract": CONTRACT, "families": list(FAMILIES), "diagnostic_only": True,
            "event_store_signature": store.manifest.get("signature"), "mapping_sha256": canonical_hash(self.mapping),
            "threshold_signature": (threshold_manifest or {}).get("signature"),
            "source_start": date.fromordinal(self.source_start).isoformat(),
            "source_completeness": "unknown; reconstructed last-user/item-pair prefix, not full event history",
            "metadata_availability": "static_retrospective_catalog_snapshot; no online-as-of proof",
            "outcome_availability": "earlier last_watch_dt assumption; aggregate receipt time unverified",
            "rating_parser": "strip outer whitespace only; exact six raw strings; no numeric coercion",
            "rating_review_status": {"21.0": "unresolved_raw_semantics"},
            "age_rating_out_of_schema_values": self.out_of_schema_values,
            "age_rating_out_of_schema_catalog_total": sum(out_values.values()),
            "age_rating_out_of_schema_list_truncated": len(out_values) > 100,
            "schema_errors": "fail closed at load; never converted to missing"}

    @classmethod
    def from_paths(cls, store, threshold_path):
        facts, manifest = load_threshold_facts(threshold_path, store)
        metadata_path = store.manifest["metadata_path"]
        if file_hash(metadata_path) != store.manifest["metadata_sha256"]:
            raise ValueError("Metadata SHA mismatch")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        item_pin = store.manifest["source_pins"]["items"]
        if file_hash(item_pin["path"]) != item_pin["sha256"]:
            raise ValueError("Item metadata source SHA mismatch")
        cards = {}
        for iid, kind, rating in history._csv_rows(item_pin["path"], ("item_id", "content_type", "age_rating")):
            iid = history._mapping_int(iid, "item metadata ID")
            if iid in cards:
                raise ValueError("Duplicate metadata card")
            cards[iid] = {"content_type": kind, "age_rating": rating}
        instance = cls(store, facts["threshold_bits"], metadata, cards, threshold_manifest=manifest)
        instance.threshold_facts = facts
        return instance

    def features(self, user_ids, origins, *, x, lengths, history_cutoffs=None, row_keys=None,
                 prepared_split=None, prepared_row_indices=None, outcome_cutoffs=None):
        users = history._ids(user_ids, "query users")
        q = np.asarray([history._day(v) for v in origins], np.int64)
        cutoff = q if history_cutoffs is None else np.asarray([history._day(v) for v in history_cutoffs], np.int64)
        outcap = cutoff if outcome_cutoffs is None else np.asarray([history._day(v) for v in outcome_cutoffs], np.int64)
        if outcap.shape != users.shape or np.any(outcap > cutoff):
            raise ValueError("Outcome availability cutoff must align and cannot exceed history cutoff")
        ids = self.store.derive_query_events(users, q, x=x, lengths=lengths, history_cutoffs=cutoff,
            prepared_split=prepared_split, prepared_row_indices=prepared_row_indices)
        if row_keys is not None and len(row_keys) != len(users):
            raise ValueError("Query row keys must align")
        mask, safe = ids >= 0, np.maximum(ids, 0)
        def gather(values, fill=0):
            return np.where(mask, values[safe], fill) if len(values) else np.full(ids.shape, fill)
        days = gather(self.store.arrays["date"])
        tokens = gather(self.store.arrays["token"])
        status = gather(self.store.arrays["parser_status"])
        outcomes = gather(self.store.arrays["outcome"])
        flags = gather(self.bits)
        available = mask & (days < outcap[:, None])
        valid = available & (status == 0)
        result, nulls = {}, {}
        def put(name, value, null=None):
            result[name] = np.asarray(value)
            nulls[name] = np.zeros(len(users), bool) if null is None else np.asarray(null, bool)
        n = mask.sum(axis=1)
        put("activity.n63", n)
        positions = np.searchsorted(self.store.users, users)
        found = positions < len(self.store.users)
        if len(self.store.users):
            found &= self.store.users[np.minimum(positions, len(self.store.users) - 1)] == users
        starts = np.zeros(len(users), np.int64)
        starts[found] = self.store.starts[positions[found]]
        ends = np.searchsorted(self.store.keys, users * history.MAX_ORDINAL + cutoff, side="left")
        ends[~found] = starts[~found]
        prefix = ends - starts
        put("activity.prefix_count", prefix)
        put("activity.cap_truncated", prefix > 63)
        put("activity.history_empty", n == 0)
        put("activity.activity_unavailable", np.zeros(len(users), bool))
        put("activity.source_completeness_unknown", np.ones(len(users), bool))
        distinct = mask & np.column_stack((np.ones(len(users), bool), days[:, 1:] != days[:, :-1]))
        windows = {}
        for w in (7, 30):
            inside = mask & (days >= q[:, None] - w)
            windows[w] = inside
            left = np.searchsorted(self.store.keys, users * history.MAX_ORDINAL + q - w, side="left")
            left = np.maximum(starts, np.minimum(left, ends))
            dropped_end = np.maximum(starts, ends - 63)
            put(f"activity.n_{w}", inside.sum(axis=1))
            put(f"activity.d_{w}", (inside & distinct).sum(axis=1))
            put(f"activity.cap_window_truncated_{w}", found & (left < dropped_end))
            put(f"activity.window_before_source_start_{w}", q - w < self.source_start)
        def completion(prefix, chosen):
            selected = chosen & valid
            support = selected.sum(axis=1)
            high = (selected & ((flags & (1 << 3)) != 0)).sum(axis=1)
            low = (selected & ((flags & 1) != 0)).sum(axis=1)
            values = {"n": chosen.sum(axis=1), "n_valid": support,
                "n_missing": (chosen & available & (status == 1)).sum(axis=1),
                "n_invalid": (chosen & available & (status == 2)).sum(axis=1),
                "n_unavailable": (chosen & ~available).sum(axis=1), "n_high": high, "n_low": low,
                "selection_empty": ~chosen.any(axis=1), "outcome_empty": support == 0}
            for name, numerator in (("mean", np.where(selected, outcomes, 0).sum(axis=1, dtype=np.float64)),
                                    ("high", high), ("low", low)):
                values[name] = np.divide(numerator, support, out=np.zeros(len(users)), where=support > 0)
            for name in COMPLETION_FIELDS:
                put(prefix + "." + name, values[name], support == 0 if name in ("mean", "high", "low") else None)
            if np.any(values["n"] != support + values["n_missing"] + values["n_invalid"] + values["n_unavailable"]):
                raise AssertionError("Outcome diagnostic partition is not exhaustive")
        kind = self.types[tokens]
        for name, code in (("film", 0), ("series", 1), ("unknown", -1)):
            completion("typed." + name, mask & (kind == code))
        for w in (7, 30):
            completion(f"window.{w}", windows[w])
        reason, rating = self.rating_reason[tokens], self.rating[tokens]
        known = mask & (reason == 0)
        nr = known.sum(axis=1)
        put("age_rating.n", n)
        put("age_rating.n_rating", nr)
        put("age_rating.empty_history", n == 0)
        put("age_rating.empty_rating_profile", nr == 0)
        for code, name in enumerate(RATING_REASONS):
            put("age_rating.reason." + name, (mask & (reason == code)).sum(axis=1))
        put("age_rating.schema_errors", np.zeros(len(users), np.int64))
        put("age_rating.coverage", np.divide(nr, n, out=np.zeros(len(users)), where=n > 0))
        put("age_rating.missing_or_unknown_share", np.divide(n - nr, n, out=np.zeros(len(users)), where=n > 0))
        for code, raw in enumerate(history.AGE_RATINGS):
            count = (known & (rating == code)).sum(axis=1)
            put("age_rating.count." + raw, count)
            put("age_rating.share." + raw, np.divide(count, nr, out=np.zeros(len(users)), where=nr > 0))
        columns = list(result)
        values = np.column_stack([result[key] for key in columns]).astype(np.float64)
        null_mask = np.column_stack([nulls[key] for key in columns])
        if not np.isfinite(values).all():
            raise AssertionError("Nonfinite diagnostic fill")
        binding = {"user_ids": events.diagnostics.array_hash(users), "origins": events.diagnostics.array_hash(q),
            "history_cutoffs": events.diagnostics.array_hash(cutoff), "outcome_cutoffs": events.diagnostics.array_hash(outcap),
            "x": events.diagnostics.array_hash(np.asarray(x)), "lengths": events.diagnostics.array_hash(np.asarray(lengths)),
            "event_ids": events.diagnostics.array_hash(ids)}
        if row_keys is not None:
            binding["row_keys"] = canonical_hash(row_keys)
        return {"values": values, "columns": columns, "null_mask": null_mask, "diagnostics": result,
            "event_ids": ids, "pins": self.provenance, "query_binding_sha256": canonical_hash(binding),
            "null_policy": "mean/high/low are NULL iff paired null_mask is true; finite zero is storage fill only"}

    def close(self):
        for value in getattr(self, "threshold_facts", {}).values():
            if getattr(value, "_mmap", None) is not None:
                value._mmap.close()
        self.store.close()


def write_manifest(provider, threshold_path, output, *, probe=None):
    """Publish normalized full-row support, without claiming wide values computed."""
    store = provider.store
    if (store.manifest.get("state") != "completed" or store.manifest.get("scope") != "full"
            or store.manifest["coverage"].get("query_mapping") != "complete_all_prepared_rows"):
        raise ValueError("Completed all-prepared query proof is required")
    output = Path(output)
    if output.exists():
        raise ValueError("Diagnostics manifest output must be a new directory")
    output.mkdir(parents=True)
    schema = provider.features(np.array([], np.int64), [], x=np.zeros((0, 64), np.int64),
                               lengths=np.array([], np.int64))
    manifest = {**provider.provenance, "state": "completed", "coverage": "full_rows",
        "storage": "normalized_event_facts_exact_query_mapping", "fit_cutoff": store.manifest["fit_cutoff"],
        "clock": "strict_h63_prior", "event_store": _pin(store.root / "manifest.json"),
        "threshold_facts": _pin(Path(threshold_path) / "manifest.json"),
        "metadata_sha256": store.manifest["metadata_sha256"], "metadata_path": store.manifest["metadata_path"],
        "prepared_pins": store.manifest["prepared_pins"], "row_keys_sha256": store.manifest["row_keys_sha256"],
        "source_pins": {**store.manifest["source_pins"], "user_diagnostics": _pin(__file__)},
        "query_coverage_receipt": store.manifest["query_coverage_receipt"],
        "split_counts": {key: value["rows"] for key, value in store.manifest["query_coverage_receipt"]["splits"].items()},
        "columns": schema["columns"], "dtype": "float64", "null_mask_dtype": "bool",
        "nullable_columns": [name for name in schema["columns"] if name.startswith(("typed.", "window."))
                             and name.rsplit(".", 1)[-1] in ("mean", "high", "low")],
        "null_policy": schema["null_policy"],
        "evaluated_values": probe or {"queries": 0}, "full_wide_values_materialized": False,
        "model_input_columns_added": [], "frozen_base_fields_rewritten": False,
        "test_npz_opened": False}
    manifest["signature"] = canonical_hash(manifest)
    _save_json(output / "manifest.json", manifest)
    return manifest


def load(path, *, expected_metadata_sha256=None, expected_prepared_pins=None,
         expected_row_keys_sha256=None, verify_hashes=True):
    """Fail-closed typed diagnostics loader; no unverified source/hash bypass."""
    root = Path(path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (verify_hashes is not True or manifest.get("contract") != CONTRACT
            or manifest.get("state") != "completed" or manifest.get("coverage") != "full_rows"
            or manifest.get("storage") != "normalized_event_facts_exact_query_mapping"
            or manifest.get("families") != list(FAMILIES) or manifest.get("diagnostic_only") is not True
            or manifest.get("model_input_columns_added") != []
            or canonical_hash({k: v for k, v in manifest.items() if k != "signature"}) != manifest.get("signature")):
        raise ValueError("User diagnostics manifest/schema/signature mismatch")
    for name in ("event_store", "threshold_facts"):
        pin = manifest[name]
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("User diagnostics native dependency SHA mismatch")
    for pin in manifest["source_pins"].values():
        if file_hash(pin["path"]) != pin["sha256"]:
            raise ValueError("User diagnostics source SHA mismatch")
    if expected_metadata_sha256 is not None and expected_metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("User diagnostics metadata binding mismatch")
    if expected_prepared_pins is not None and expected_prepared_pins != manifest["prepared_pins"]:
        raise ValueError("User diagnostics prepared source binding mismatch")
    if expected_row_keys_sha256 is not None and expected_row_keys_sha256 != manifest["row_keys_sha256"]:
        raise ValueError("User diagnostics row key binding mismatch")
    store = events.load(Path(manifest["event_store"]["path"]).parent,
        expected_metadata_sha256=manifest["metadata_sha256"], expected_prepared_pins=manifest["prepared_pins"],
        expected_row_keys_sha256=manifest["row_keys_sha256"])
    try:
        if (manifest["query_coverage_receipt"] != store.manifest["query_coverage_receipt"]
                or manifest["split_counts"] != {key: value["rows"] for key, value in
                    store.manifest["query_coverage_receipt"]["splits"].items()}):
            raise ValueError("User diagnostics full-row mapping proof mismatch")
        provider = UserDiagnostics.from_paths(store, Path(manifest["threshold_facts"]["path"]).parent)
        schema = provider.features(np.array([], np.int64), [], x=np.zeros((0, 64), np.int64),
                                   lengths=np.array([], np.int64))
        if (manifest.get("columns") != schema["columns"] or manifest.get("dtype") != "float64"
                or manifest.get("null_mask_dtype") != "bool" or manifest.get("null_policy") != schema["null_policy"]
                or manifest.get("nullable_columns") != [name for name in schema["columns"]
                    if name.startswith(("typed.", "window.")) and name.rsplit(".", 1)[-1] in ("mean", "high", "low")]):
            provider.close()
            raise ValueError("User diagnostics value/NULL schema mismatch")
        provider.manifest = manifest
        provider.provenance = manifest
        return provider
    except Exception:
        store.close()
        raise
