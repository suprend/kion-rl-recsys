"""Typed complete-feature RM inputs; CPU preparation is a separate admission.

This module never uses CompleteProvider or an SFT feature cache. RM roles own
their Aug2 indices; static metadata and frozen E5 are explicitly retrospective.
No learned SFT/GPT/people embedding weights or unobserved dislike labels enter.
"""
from __future__ import annotations

import argparse
import builtins
from collections import OrderedDict
import copy
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_reward_data as rows_kernel
import kion_reward_boosting as rm
import kion_complete_manifest as census

CONTRACT = "kion-complete-reward-features-v1"
READY_SCOPE_ID = "ready-pack-20261020-r1"
READY_SCOPE_SHA256 = "a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962"
CAP = "2021-08-02"
CAP_DAY = date.fromisoformat(CAP).toordinal()
PEOPLE_BACKEND = {"name": "fast-exact", "native_role": "rm", "fit_cutoff_exclusive": CAP,
    "core_sha256": "6bf97bb98c8fdf0cb858db18066862d3a94cb107334bcb14c74e340f2943911c",
    "adapter_sha256": "e6ad0d34174c44f50f4c27e088fab36a399e839a0ee96b77ec0df27e2bda2a91",
    "SFT_bank_reuse": False}
READY_PEOPLE_BACKEND = {"name": "deferred_by_user_scope", "SFT_bank_reuse": False}
READY_EXTERNAL_REGIME = "historical_numeric_snapshot_retrospective_identity"
READY_SERIES_SHA256 = "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"
READY_EXTERNAL_PINS = {
    "producer": "a1e0f9a0dbd005c9df28287c40aaf3e465813ce491f63391285336d650af809c",
    "collection": "8d736f001df065246fe5b1b2c795e1e5bbe65d60acdb4865d6efb04014bef7f5",
    "bank_manifest": "decabacca0905ee270060c5b37cba939de5d48311fa3e6c3899e85381e2c9f42",
    "review": "238ad8931639c63b9b90d4a21b5a1ad12d2efe0fa0118a36c0fa24e227e66b34",
    "frozen_inputs": "459af1c69b16b714b29c19696a9395df3e97f0e649e4d062d64a33a946a24791"}
TEXT_FIELDS = ("title", "title_orig", "description")
AGREEMENTS = {key: list(value) for key, value in census.AGREEMENT_COMPONENTS.items()
              if key not in census.DEFERRED_PROPOSALS}
COMPONENTS = sorted({name for names in AGREEMENTS.values() for name in names})
STATIC_COMPONENTS = {"local_base", "semantic_extensions", "text3fields", "people_embeddings",
                     "entity_links", "candidate_model", "professions", "geography_time", "sports_settings", "story_format", "narrative_relations", "static_replacements"}
EXTENSION_SLOTS = {"semantic_extensions": 328, "semantic_extensions.franchise": 93,
    "professions": 324, "geography_time": 2696, "sports_settings": 492,
    "story_format": 111, "narrative_relations": 78}
STATIC_NAMESPACES = {slot: "semantic_extensions" if slot == "semantic_extensions.franchise" else slot
                     for slot in EXTENSION_SLOTS}
EXTENSION_COMPONENTS = {"semantic_extensions", "professions", "geography_time", "sports_settings", "story_format", "narrative_relations"}
REPLACEMENT_COMPONENTS = {"story_format", "narrative_relations"}
FRANCHISE_ORDER_KINDS = ("mainline_release", "catalog_release_precedes")
LATE_INTEGRATION = {"contract": "kion-complete-rm-late-model-blocks-v1", "static_slots": EXTENSION_SLOTS,
    "static_namespaces": STATIC_NAMESPACES,
    "replacement_widths": {"story_format": 57, "narrative_relations": 30},
    "franchise_order_kinds": list(FRANCHISE_ORDER_KINDS), "diagnostic_model_columns": []}
REPRESENTATIONS = {"people_embeddings": "raw_catalogue_identity_counts_masks_no_learned_weights",
                   "text3fields": "frozen_E5_three_fields_and_H63_profiles_retrospective",
                   "candidate_model": "CatBoost_original_item_ID_category_no_SFT_weights"}
SOURCE_NAMES = ("kion_complete_reward_data", "kion_reward_data", "kion_reward_boosting", "kion_complete_manifest",
                "kion_local_pack", "kion_pack_history", "kion_pack_dynamic", "kion_pack_static", "kion_pack_people",
                "kion_pack_semantics", "kion_local_features", "kion_basic_bundle_features", "kion_sex_features",
                "kion_text_embeddings", "kion_people_embeddings", "kion_personal_candidates", "kion_audience_features",
                "kion_people_dynamics", "kion_related_features", "kion_franchise_progress")
SOURCE_NAMES += ("kion_external_facts", "kion_external_interest", "kion_entity_links")
SOURCE_NAMES += ("kion_completion_event_store", "kion_completion_diagnostics")
SOURCE_NAMES += ("kion_story_format", "kion_semantic_extensions", "kion_profession_relations",
                 "kion_geography_time", "kion_sports_settings")
SOURCE_NAMES += ("kion_people_dynamics_fast",)
SOURCE_NAMES += ("kion_narrative_relations", "kion_user_diagnostics")


def pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": rm.file_hash(path)}


def _read(path, *, payload=None):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate manifest key: " + key)
            result[key] = value
        return result
    raw = Path(path).read_bytes() if payload is None else payload
    return json.loads(raw.decode("utf-8-sig"), object_pairs_hook=pairs)


def _path(root, declaration):
    if not isinstance(declaration, dict) or not isinstance(declaration.get("path"), str):
        raise ValueError("An explicit path/SHA pin is required")
    value = Path(declaration["path"])
    value = value if value.is_absolute() else root / value
    value = value.resolve()
    if rm.file_hash(value) != declaration.get("sha256"):
        raise ValueError("Source/artifact hash mismatch: " + value.as_posix())
    return value


def _walk_pins(root, value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            _path(root, value)
        for child in value.values():
            _walk_pins(root, child)
    elif isinstance(value, list):
        for child in value:
            _walk_pins(root, child)


def _source_pins():
    return [pin(Path(__file__).with_name(name + ".py")) for name in SOURCE_NAMES]


def _ready_series_config(root, value, metadata, *, local=None):
    """Bind all three native series readers before expensive source admission.

    The static factory resolves relative to its frozen checkout; dynamic uses
    the raw ancestry and personal receives an explicit path. Equal accepted
    bytes are required even when these resolve to different physical copies.
    """
    import kion_pack_static as static
    import kion_pack_dynamic as dynamic
    expected = Path(static.__file__).resolve().parents[1] / "configs/datasets/kion-series-v1.json"
    declaration = value.get("runtime", {}).get("local", {}).get("series_annotation")
    if (not isinstance(declaration, dict) or declaration.get("sha256") != READY_SERIES_SHA256
            or _path(root, declaration) != expected):
        raise ValueError("Ready RM frozen static series config closure mismatch")
    if not any(p.get("sha256") == READY_SERIES_SHA256 and Path(p.get("path", "")).resolve() == expected
               for p in value.get("source_pins", [])):
        raise ValueError("Ready RM frozen series config is missing from source pins")
    personal = value.get("runtime", {}).get("personal", {}).get("series_annotation")
    if not isinstance(personal, dict) or personal.get("sha256") != READY_SERIES_SHA256:
        raise ValueError("Ready RM personal series config closure mismatch")
    _path(root, personal)
    raw_dir = _path(root, value["raw_sources"]["items"]).parent
    dynamic_path = dynamic._series_path(raw_dir, metadata)
    if dynamic_path is None or rm.file_hash(dynamic_path) != READY_SERIES_SHA256:
        raise ValueError("Ready RM dynamic series config closure mismatch")
    if local is not None:
        consumed = local.provenance.get("components", {}).get("static", {}).get("inputs", {}).get("series", {})
        if (consumed.get("sha256") != READY_SERIES_SHA256 or Path(consumed.get("path", "")).resolve() != expected
                or local.dynamic.provenance.get("series_sha256") != READY_SERIES_SHA256):
            raise ValueError("Ready RM native factory consumed another series config")


def _people_fast_module():
    core = importlib.import_module("kion_people_dynamics")
    for filename, expected in (("kion_people_dynamics_fast.py", PEOPLE_BACKEND["adapter_sha256"]),
            ("kion_people_dynamics.py", PEOPLE_BACKEND["core_sha256"])):
        path = Path(__file__).with_name(filename).resolve()
        if rm.file_hash(path) != expected:
            raise ValueError("Dedicated RM fast-exact people backend source mismatch")
    if Path(core.__file__).resolve() != Path(__file__).with_name("kion_people_dynamics.py").resolve():
        raise ValueError("Dedicated RM people core resolved outside pinned siblings")
    # PAS itself has a scripts package. Bind only this accepted adapter's single
    # absolute project import to the verified sibling without mutating ambient
    # sys.modules, package paths, or the frozen adapter's source bytes.
    package = SimpleNamespace(kion_people_dynamics=core)
    def native_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scripts" and tuple(fromlist) == ("kion_people_dynamics",) and level == 0:
            return package
        return builtins.__import__(name, globals, locals, fromlist, level)
    spec = importlib.util.spec_from_file_location("_complete_rm_people_fast", Path(__file__).with_name("kion_people_dynamics_fast.py"))
    module = importlib.util.module_from_spec(spec)
    module.__builtins__ = {**vars(builtins), "__import__": native_import}
    spec.loader.exec_module(module)
    return module


def _ready_scope(root, value, *, scope_decision=None, fixture=False):
    if value.get("integration_mode") != "scoped_production":
        if value.get("integration_mode") is not None:
            raise ValueError("Unknown RM integration mode")
        if any(key in value for key in ("scope_id", "scope_decision", "active_families", "additional_ready_components")):
            raise ValueError("Scope fields require explicit scoped_production mode")
        return None
    path = _path(root, value.get("scope_decision"))
    if value["scope_decision"]["sha256"] != READY_SCOPE_SHA256:
        raise ValueError("Ready scope must pin the exact human decision")
    if not fixture and (scope_decision is None or Path(scope_decision).resolve() != path):
        raise ValueError("Scoped production requires explicit scope_decision path")
    decision = _read(path)
    if (decision.get("contract") != "kion-user-approved-ready-scope-v1" or decision.get("scope_id") != READY_SCOPE_ID
            or value.get("scope_id") != READY_SCOPE_ID or value.get("active_families") != decision["active_families"]
            or value.get("additional_ready_components") != decision["additional_ready_components"]
            or value.get("agreements") != {}):
        raise ValueError("Ready scope family/component census differs; no all81 admission")
    return decision


def _family_payload(value, name):
    common = {key: copy.deepcopy(value[key]) for key in ("metadata", "raw_sources", "source_pins", "completion_event_store")}
    if name in EXTENSION_COMPONENTS:
        branch = {"static_extensions": [d for d in value.get("static_extensions", []) if d.get("family") == name]}
        if name in REPLACEMENT_COMPONENTS:
            branch.update(static_replacements=[d for d in value.get("static_replacements", []) if d.get("family") == name],
                          local_static_baseline=value["local_static_baseline"])
    elif name in REPLACEMENT_COMPONENTS:
        branch = {"static_replacements": [d for d in value.get("static_replacements", []) if d.get("family") == name],
                  "local_static_baseline": value["local_static_baseline"]}
    elif name in ("external_facts", "external_interest"):
        branch = {"date_features": [d for d in value.get("date_features", []) if d.get("family") == name]}
    elif name == "local_base":
        branch = {key: value[key] for key in ("basic_npz", "basic_manifest")}
    elif name == "local_dynamic":
        branch = {"runtime": value["runtime"]["local"]}
    elif name in ("personal_candidates", "audience", "people_dynamics", "related_interest"):
        branch = {"runtime": value["runtime"]["personal" if name == "personal_candidates" else name]}
    elif name == "franchise_progress":
        branch = {key: value[key] for key in ("franchise_graph", "franchise_temporal_mode", "franchise_order_kinds", "franchise_scope_policies")}
    elif name == "completion_diagnostics":
        branch = {"diagnostic_only": True, "model_columns": [], "event_store": value["completion_event_store"]}
    elif name == "user_diagnostics":
        branch = {"diagnostic_only": True, "model_columns": [], "native_manifest": value["user_diagnostics"]}
    elif name == "static_replacements":
        branch = {"static_replacements": value["static_replacements"], "local_static_baseline": value["local_static_baseline"]}
    elif name == "text3fields":
        branch = {"text_caches": value["text_caches"], "representation": REPRESENTATIONS[name]}
    elif name in ("people_embeddings", "entity_links"):
        field = "people_catalog" if name == "people_embeddings" else "entity_links"
        branch = {field: value[field], "representation": REPRESENTATIONS.get(name, "accepted_entity_links")}
    elif name == "candidate_model":
        branch = {"representation": REPRESENTATIONS[name]}
    else:
        raise ValueError("No typed RM family adapter: " + name)
    return {**common, **branch}


def seal_manifest(path, declaration):
    """Write a NEW role manifest, without loading indices or granting execution."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    value = copy.deepcopy(declaration)
    value.update(contract=CONTRACT, role="rm", fit_cutoff_exclusive=CAP,
                 agreements=copy.deepcopy(AGREEMENTS), representations=copy.deepcopy(REPRESENTATIONS))
    if value.get("integration_mode") == "scoped_production":
        value["agreements"] = {}
    if "source_pins" not in value:
        value["source_pins"] = _source_pins()
        if value.get("integration_mode") == "scoped_production" and value.get("state") != "fixture_only":
            value["source_pins"].append(pin(Path(__file__).resolve().parents[1] / "configs/datasets/kion-series-v1.json"))
    def absolute_pins(node):
        if isinstance(node, dict):
            if "path" in node and "sha256" in node:
                candidate = Path(node["path"])
                node["path"] = (candidate if candidate.is_absolute() else path.parent / candidate).resolve().as_posix()
            for child in node.values():
                absolute_pins(child)
        elif isinstance(node, list):
            for child in node:
                absolute_pins(child)
    absolute_pins(value)
    value["signature"] = rm.canonical_hash({k: v for k, v in value.items() if k != "signature"})
    path.parent.mkdir(parents=True, exist_ok=True)
    rm.write_json(path, value)
    return value


def _admit(root, value, fixture_mode, scope_decision=None):
    if (value.get("contract") != CONTRACT or value.get("role") != "rm"
            or value.get("fit_cutoff_exclusive") != CAP):
        raise ValueError("A typed RM manifest with literal Aug2 cap is required")
    if rm.canonical_hash({k: v for k, v in value.items() if k != "signature"}) != value.get("signature"):
        raise ValueError("Complete RM manifest signature mismatch")
    fixture = value.get("state") == "fixture_only"
    scope = _ready_scope(root, value, scope_decision=scope_decision, fixture=fixture)
    if (value.get("state") not in ("fixture_only", "admitted_full", "scoped_production") or fixture != bool(fixture_mode)
            or not fixture and (value["state"] == "scoped_production") != (scope is not None)):
        raise ValueError("Fixture admission must be explicit; production requires admitted_full")
    if scope is None and (value.get("agreements") != AGREEMENTS or len(AGREEMENTS) != 81):
        raise ValueError("The complete agreed 81-ID census differs")
    if value.get("representations") != REPRESENTATIONS:
        raise ValueError("RM forbids learned SFT/GPT/people embedding weights")
    if value.get("metadata_availability") != "retrospective_catalog_snapshot_not_historical_asof":
        raise ValueError("Explicit retrospective metadata availability is required")
    metadata = _read(_path(root, value["metadata"]))
    mapping = rows_kernel.checked_mapping(metadata)
    if metadata.get("context") != 64 or value.get("mapping_sha256") != rm.canonical_hash(metadata["item_id_to_token"]):
        raise ValueError("RM metadata/mapping identity mismatch")
    if scope is not None and not fixture:
        _ready_series_config(root, value, metadata)
    if fixture:
        if len(mapping) > 128:
            raise ValueError("Fixture mode is limited to 128 catalogue items")
        # Bound the decoded source BEFORE any native index construction. The
        # limit is verified from source rows, never trusted from a receipt.
        for declaration in value.get("raw_sources", {}).values():
            source = Path(declaration["path"])
            source = source if source.is_absolute() else root / source
            with source.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                next(reader, None)
                for number, _ in enumerate(reader, 1):
                    if number > 4096:
                        raise ValueError("Fixture mode cannot scan/build a full source")
    _walk_pins(root, value)
    sources = {_path(root, p) for p in value.get("source_pins", [])}
    if not {Path(__file__).with_name(name + ".py").resolve() for name in SOURCE_NAMES}.issubset(sources):
        raise ValueError("Missing complete RM/runtime source pins")
    components = COMPONENTS if scope is None else scope["active_families"] + scope["additional_ready_components"]
    if set(value.get("families", {})) != set(components):
        raise ValueError("Complete RM family census is incomplete")
    accepted = {}
    for name, receipt in value["families"].items():
        if fixture:
            if receipt.get("status") != "fixture_only":
                raise ValueError("Fixture receipts cannot impersonate production acceptance")
        else:
            if not isinstance(receipt, dict) or "evidence" not in receipt or "review" not in receipt:
                raise ValueError("Production RM component requires actual evidence and independent review: " + name)
            evidence = _read(_path(root, receipt["evidence"]))
            review = _read(_path(root, receipt["review"]))
            if (evidence.get("contract") != census.EVIDENCE_CONTRACT or evidence.get("component") != name
                    or evidence.get("state") != "completed" or evidence.get("scope") != "full"
                    or evidence.get("source_kind") != "actual_data"
                    or evidence.get("metadata_sha256") != value["metadata"]["sha256"]
                    or evidence.get("vocabulary_sha256") != value["mapping_sha256"]):
                raise ValueError("Unaccepted/nonactual complete RM family: " + name)
            if (review.get("contract") != census.REVIEW_CONTRACT or review.get("component") != name
                    or review.get("evidence_sha256") != receipt["evidence"]["sha256"]
                    or review.get("verdict") != "PASS" or not review.get("reviewer")
                    or review["reviewer"] == evidence.get("author")):
                raise ValueError("Independent RM component review mismatch: " + name)
            temporal = evidence.get("temporal", {})
            expected_mode = "static_retrospective" if name in STATIC_COMPONENTS else "strict_prefix"
            if (temporal.get("mode") != expected_mode or temporal.get("role") != "rm"
                    or temporal.get("fit_cutoff") != CAP or temporal.get("validation_origin") != CAP
                    or temporal.get("whole_day_exclusive") is not True):
                raise ValueError("RM component temporal role/cap mismatch: " + name)
            if not all(evidence.get("checks", {}).get(k) is True for k in
                       ("tests_passed", "schema_validated", "clock_validated", "numerical_parity")):
                raise ValueError("Incomplete component checks: " + name)
            if not evidence.get("source_pins") or not evidence.get("artifacts"):
                raise ValueError("Component has no actual artifact/source receipts: " + name)
            _walk_pins(root, evidence)
            if evidence.get("provider_payload") != _family_payload(value, name):
                raise ValueError("RM materialization differs from reviewed component payload: " + name)
            known = {_path(root, p): p["sha256"] for p in evidence["source_pins"] + evidence["artifacts"]}
            def reviewed_refs(payload):
                if isinstance(payload, dict):
                    if "path" in payload and "sha256" in payload:
                        path = _path(root, payload)
                        if known.get(path) != payload["sha256"]:
                            raise ValueError("RM payload references an unreviewed source/artifact: " + name)
                    for child in payload.values():
                        reviewed_refs(child)
                elif isinstance(payload, list):
                    for child in payload:
                        reviewed_refs(child)
            reviewed_refs(evidence["provider_payload"])
            coverage = evidence.get("coverage", {})
            if (coverage.get("selected_items") != len(mapping) or coverage.get("attempted_items") != len(mapping)
                    or type(coverage.get("known_items")) is not int or coverage["known_items"] < 1):
                raise ValueError("Pilot/unknown-only family cannot satisfy full coverage: " + name)
        accepted[name] = "fixture_only" if fixture else "accepted_scoped" if scope is not None else "accepted_full"
    # These are independent component caps, not an origin override on an SFT object.
    runtime_names = ("local", "personal", "related_interest") if scope is not None else ("local", "personal", "audience", "people_dynamics", "related_interest")
    if scope is not None and set(value.get("runtime", {})) != set(runtime_names):
        raise ValueError("Ready RM runtime cannot construct deferred feature backends")
    for name in runtime_names:
        declaration = value.get("runtime", {}).get(name, {})
        if declaration.get("role") != "rm" or declaration.get("fit_cutoff_exclusive") != CAP:
            raise ValueError("Runtime component needs native RM/Aug2 cap: " + name)
    if scope is not None:
        annotation = value["runtime"]["personal"].get("series_annotation")
        if not isinstance(annotation, dict):
            raise ValueError("Explicit pinned RM series_annotation is required")
        series_path = _path(root, annotation)
        if (not fixture and annotation["sha256"] != "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"):
            raise ValueError("Scoped RM series annotation differs from the reviewed catalogue")
        if metadata.get("series_annotation_path") is not None and Path(metadata["series_annotation_path"]).resolve() != series_path:
            raise ValueError("Scoped RM series annotation differs from prepared metadata")
    if scope is None and value["runtime"]["people_dynamics"].get("backend") != "fast-exact":
        raise ValueError("Dedicated RM people backend must explicitly be fast-exact; no fallback")
    if value.get("franchise_order_kinds") != list(FRANCHISE_ORDER_KINDS):
        raise ValueError("Both explicit typed franchise order branches are required")
    policies = value.get("franchise_scope_policies")
    if (not isinstance(policies, dict) or set(policies) != set(FRANCHISE_ORDER_KINDS)
            or any(v is not None and not isinstance(v, dict) for v in policies.values())):
        raise ValueError("Franchise scope policies must bind both typed branches")
    _, store = _store_descriptor(root, value, fixture)
    _diagnostic_descriptors(root, value, store)
    return metadata, mapping, accepted


def _diagnostic_descriptors(root, manifest, store):
    """The two typed diagnostic manifests and their direct immutable payloads."""
    import kion_user_diagnostics as diagnostics
    value = manifest
    diagnostic_path = _path(root, value["user_diagnostics"])
    diagnostic = _read(diagnostic_path)
    if (diagnostic.get("contract") != "kion-user-diagnostics-v1" or diagnostic.get("diagnostic_only") is not True
            or diagnostic.get("model_input_columns_added") != [] or len(diagnostic.get("columns", [])) != 99
            or diagnostic.get("event_store", {}).get("sha256") != value["completion_event_store"]["sha256"]):
        raise ValueError("User diagnostics must bind canonical store and remain 99 diagnostic-only columns")
    _walk_pins(diagnostic_path.parent, diagnostic)
    threshold_path = _path(diagnostic_path.parent, diagnostic["threshold_facts"])
    threshold = _read(threshold_path)
    if (threshold.get("contract") != "kion-exact-outcome-threshold-facts-v1" or threshold.get("state") != "completed"
            or diagnostics.canonical_hash({k: v for k, v in threshold.items() if k != "signature"}) != threshold.get("signature")
            or diagnostic.get("threshold_signature") != threshold.get("signature")
            or threshold.get("events_count") != store["events_count"] or threshold.get("fit_cutoff") != store["fit_cutoff"]
            or threshold.get("event_store", {}).get("sha256") != value["completion_event_store"]["sha256"]
            or threshold.get("source_pins", {}).get("interactions", {}).get("sha256") != value["raw_sources"]["interactions"]["sha256"]
            or set(threshold.get("arrays", {})) != {"threshold_bits", "source_csv_data_row"}):
        raise ValueError("User diagnostic threshold source/store/signature binding mismatch")
    for name, dtype in (("threshold_bits", "uint8"), ("source_csv_data_row", "uint32")):
        spec = threshold["arrays"][name]
        if spec.get("dtype") != dtype or spec.get("shape") != [store["events_count"]]:
            raise ValueError("User diagnostic threshold payload schema mismatch")
    _walk_pins(threshold_path.parent, threshold)
    return ((diagnostic_path, diagnostic), (threshold_path, threshold))


def _store_descriptor(root, manifest, fixture):
    path = _path(root, manifest["completion_event_store"])
    if path.name != "manifest.json":
        raise ValueError("Completion event store must pin native manifest.json")
    value = _read(path)
    expected_clock = {"baseline": "strictly_before_event_date", "global_cap": "min(event_date,role_fit_cap)",
                      "whole_user_exclusion": "all item/type/global tiers"}
    if (value.get("contract") != "kion-completion-event-store-v1" or value.get("event_clock") != expected_clock
            or value.get("normalization") != "immutable_per_event_no_global_fit"
            or value.get("query_join") != "exact-immutable-user-day-raw-item-H63-v1"
            or value.get("type_policy") != "mapped token >=3 exact film/series; UNK remains unknown"
            or value.get("fit_cutoff") not in (CAP, "2021-08-09")
            or value.get("alpha") != 1.0 or value.get("beta") != 1.0 or value.get("c0") != 0.5):
        raise ValueError("Completion store must contain immutable strict event-relative facts with fixed c0")
    full = (value.get("state") == "completed" and value.get("scope") == "full"
            and value.get("coverage", {}).get("raw_events") == "complete_before_role_cap"
            and value.get("coverage", {}).get("query_mapping") == "complete_all_prepared_rows")
    if not fixture and not full:
        raise ValueError("Production rejects completion event-store probes/incomplete query coverage")
    if value.get("metadata_sha256") != manifest["metadata"]["sha256"]:
        raise ValueError("Completion event-store metadata binding mismatch")
    for name in ("items", "interactions"):
        source = value.get("source_pins", {}).get(name, {})
        if source.get("sha256") != manifest["raw_sources"][name]["sha256"]:
            raise ValueError("Completion event-store raw source mismatch: " + name)
        _path(root, source)
    return path, value


def _load_store(root, manifest, fixture):
    import kion_completion_event_store as events
    import kion_pack_history as history
    path, descriptor = _store_descriptor(root, manifest, fixture)
    metadata = _read(_path(root, manifest["metadata"]))
    integer_mapping = {int(key): value for key, value in metadata["item_id_to_token"].items()}
    if descriptor.get("mapping_sha256") != events.canonical_hash(integer_mapping):
        raise ValueError("Completion event-store raw/token mapping mismatch")
    if descriptor.get("parser_sha256") != rm.file_hash(history.__file__):
        raise ValueError("Completion event-store parser/type policy source mismatch")
    return events.load(path.parent, expected_metadata_sha256=manifest["metadata"]["sha256"],
                       verify_hashes=True, allow_probe=fixture)


class CompleteRewardProvider:
    """A role-owned CPU factory and immutable s0 feature interface."""
    fit_cutoff = CAP

    @classmethod
    def load(cls, manifest_path, *, fixture_mode=False, local_factory=None, scope_decision=None):
        if local_factory is not None and not fixture_mode:
            raise ValueError("Injected local factories are fixture-only")
        obj = cls()
        obj.path = Path(manifest_path).resolve()
        obj._manifest_bytes = obj.path.read_bytes()
        obj._provider_pin = {"path": obj.path.as_posix(), "sha256": hashlib.sha256(obj._manifest_bytes).hexdigest()}
        obj.root, obj.manifest = obj.path.parent, _read(obj.path, payload=obj._manifest_bytes)
        obj._maps, obj._states, obj._banks = [], OrderedDict(), OrderedDict()
        obj._inputs = {}
        obj.fixture_mode = bool(fixture_mode)
        try:
            obj.metadata, mapping, obj.acceptance = _admit(obj.root, obj.manifest, fixture_mode, scope_decision)
            obj.scope = _ready_scope(obj.root, obj.manifest, scope_decision=scope_decision, fixture=fixture_mode)
            obj._record_inputs(obj.root, obj.manifest)
            obj._record_inputs(obj.root, obj._provider_pin)
            if not fixture_mode:
                for receipt in obj.manifest["families"].values():
                    obj._record_inputs(obj.root, _read(_path(obj.root, receipt["evidence"])))
            obj._load(mapping, local_factory)
            obj.verify_inputs()
        except Exception:
            obj.close()
            raise
        return obj

    def _record_inputs(self, root, value):
        """Retain native, already-validated payload pins through publication.

        Linked manifests are expanded by their typed adapters below. Do not
        recursively interpret arbitrary JSON files as dependency manifests.
        """
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = Path(value["path"])
                path = (path if path.is_absolute() else root / path).resolve()
                previous = self._inputs.setdefault(path.as_posix(), value["sha256"])
                if previous != value["sha256"]:
                    raise ValueError("Conflicting complete RM input pins: " + path.as_posix())
            for child in value.values():
                self._record_inputs(root, child)
        elif isinstance(value, list):
            for child in value:
                self._record_inputs(root, child)

    def verify_inputs(self):
        for path, sha256 in self._inputs.items():
            _path(self.root, {"path": path, "sha256": sha256})

    def _matrix(self, declaration):
        if (declaration.get("metadata_sha256") != self.manifest["metadata"]["sha256"]
                or declaration.get("mapping_sha256") != self.manifest["mapping_sha256"]):
            raise ValueError("Matrix metadata/mapping binding mismatch")
        values = np.load(_path(self.root, declaration), mmap_mode="r", allow_pickle=False)
        self._maps.append(values)
        columns = declaration.get("columns", [])
        rm._names(columns)
        if (values.dtype != np.float32 or values.shape != (self.metadata["vocab_size"], len(columns))
                or not len(columns) or np.any(values[:3] != 0) or not np.isfinite(values).all()):
            raise ValueError("Invalid complete matrix schema/specials/values")
        return values

    def _load(self, mapping, local_factory):
        import kion_local_pack as local
        import kion_text_embeddings as text
        import kion_people_embeddings as people
        import kion_personal_candidates as personal
        import kion_audience_features as audience
        dynamics = None if self.scope is not None else _people_fast_module()
        import kion_related_features as related
        import kion_franchise_progress as franchise
        m = self.manifest
        self.raw_dir = _path(self.root, m["raw_sources"]["items"]).parent
        for name in ("items", "users", "interactions"):
            if _path(self.root, m["raw_sources"][name]) != self.raw_dir / (name + ".csv"):
                raise ValueError("Raw source directory/name binding mismatch")
        self.metadata_path = _path(self.root, m["metadata"])
        base_path = _path(self.root, m["basic_npz"])
        if _path(self.root, m["basic_manifest"]) != base_path.with_name("manifest.json"):
            raise ValueError("Basic manifest/NPZ directory binding mismatch")
        factory = local.FeaturePack.build if local_factory is None else local_factory
        self.local = factory(self.raw_dir, self.metadata_path, base_path, fit_cutoff=CAP, include_relative=True)
        if self.scope is not None and not self.fixture_mode:
            _ready_series_config(self.root, m, self.metadata, local=self.local)
        self.history, self.dynamic = self.local.history, self.local.dynamic
        if self.local.fit_cutoff != CAP or self.history.fit_cutoff != CAP_DAY or self.dynamic.fit_cutoff != CAP_DAY:
            raise ValueError("Local/history/dynamic returned an SFT/Aug9 cap")
        if {int(k): v for k, v in self.history.mapping.items()} != mapping:
            raise ValueError("Native history mapping differs")
        self.raw = rows_kernel.RawRows.from_pack(self, mapping)
        if self.fixture_mode and len(self.raw.users) > 4096:
            raise ValueError("Fixture mode cannot scan/build a full source")
        self.event_store = _load_store(self.root, m, self.fixture_mode)
        store_manifest = self.event_store.manifest
        store_root = _path(self.root, m["completion_event_store"]).parent
        # Arrays, diagnostic catalogue and full-store query arrays are native
        # payloads too, even when a particular sampled query does not use them.
        for key in ("arrays", "catalog", "query_arrays", "prepared_pins", "source_pins"):
            self._record_inputs(store_root, store_manifest.get(key, {}))
        self._record_inputs(store_root, {"path": store_manifest["metadata_path"],
                                        "sha256": store_manifest["metadata_sha256"]})
        for diagnostic_path, descriptor in _diagnostic_descriptors(self.root, m, store_manifest):
            self._record_inputs(diagnostic_path.parent, descriptor)
        self.item_features = np.asarray(self.local.item_features, np.float32)
        self.item_columns = list(self.local.item_columns)
        baseline = m["local_static_baseline"]
        static_manifest_path = _path(self.root, baseline["manifest"])
        base_manifest = _read(static_manifest_path)
        base_columns = {"item": list(self.local.item_columns), "user": list(self.local.user_columns)}
        if (baseline.get("scope") != "item_only_outcome_free_rebuilt_metadata"
                or base_manifest.get("feature_columns") != base_columns
                or base_manifest.get("schema_hash") != rm.canonical_hash(base_columns)
                or baseline.get("array_sha256") != text.array_hash(self.item_features)):
            raise ValueError("RM static baseline differs from outcome-free rebuilt catalogue/schema")
        # Only the outcome-free item payload is reused for identity checking;
        # SFT row arrays/user caches are neither opened nor interpreted.
        static_path = _path(static_manifest_path.parent, base_manifest["outputs"]["item_features.npy"])
        self._record_inputs(static_manifest_path.parent, base_manifest["outputs"]["item_features.npy"])
        static_values = np.load(static_path, mmap_mode="r", allow_pickle=False)
        try:
            if static_values.dtype != np.float32 or text.array_hash(static_values) != baseline["array_sha256"]:
                raise ValueError("Static baseline payload differs from rebuilt metadata-only values")
        finally:
            static_values._mmap.close()
        extensions = m.get("static_extensions", [])
        slots = {entry.get("slot", entry.get("namespace", entry.get("family"))): entry for entry in extensions}
        if len(slots) != len(extensions) or set(slots) != set(EXTENSION_SLOTS):
            raise ValueError("Every late static extension slot must materialize exactly once")
        matrices = [self.item_features]
        for slot, entry in slots.items():
            family = entry.get("family")
            namespace = entry.get("namespace", family)
            if (family not in EXTENSION_COMPONENTS or namespace != STATIC_NAMESPACES[slot]
                    or entry.get("temporal_mode") != "static_retrospective"):
                raise ValueError("Static extension namespace/family/clock mismatch")
            values = self._matrix(entry)
            if not self.fixture_mode and values.shape[1] != EXTENSION_SLOTS[slot]:
                raise ValueError("Full late static extension width mismatch: " + slot)
            if entry.get("transform") == "signed_log1p":
                indices = entry.get("transform_columns")
                if (not isinstance(indices, list) or len(set(indices)) != len(indices)
                        or any(type(i) is not int or not 0 <= i < values.shape[1] for i in indices)):
                    raise ValueError("Explicit signed-log static transform columns required")
                values = values.copy()
                values[:, indices] = np.sign(values[:, indices]) * np.log1p(np.abs(values[:, indices]))
            elif entry.get("transform") != "identity":
                raise ValueError("Unknown static extension transform")
            matrices.append(values)
            self.item_columns += [namespace + "." + c for c in entry["columns"]]
        # Narrative targets live inside the appended semantic328 block. Assemble
        # every static slot before applying either independently bound preimage.
        self.item_features = np.concatenate(matrices, axis=1)
        self.replacements, replaced = [], set()
        replacements = m.get("static_replacements", [])
        if len(replacements) != len(REPLACEMENT_COMPONENTS) or {entry.get("family") for entry in replacements} != REPLACEMENT_COMPONENTS:
            raise ValueError("Both named story/format and narrative replacements must materialize")
        for entry in replacements:
            if (entry.get("mode") != "replace_named_columns" or entry.get("transform") != "identity"
                    or entry.get("temporal_mode") != "static_retrospective"
                    or entry.get("base_manifest_sha256") != baseline["manifest"]["sha256"]
                    or entry.get("base_schema_hash") != base_manifest["schema_hash"]
                    or entry.get("base_item_features_sha256") != base_manifest["outputs"]["item_features.npy"]["sha256"]):
                raise ValueError("Named replacement must bind immutable static baseline/schema/units")
            if not self.fixture_mode and len(entry["columns"]) != LATE_INTEGRATION["replacement_widths"][entry["family"]]:
                raise ValueError("Full named replacement width mismatch")
            if entry["family"] == "narrative_relations":
                target = entry.get("target_extension", {})
                original = slots["semantic_extensions"]
                expected = {"path": original["path"], "sha256": original["sha256"], "family": "semantic_extensions",
                    "namespace": "semantic_extensions", "shape": [self.metadata["vocab_size"], len(original["columns"])],
                    "dtype": "float32", "columns_sha256": rm.canonical_hash(original["columns"]),
                    "metadata_sha256": m["metadata"]["sha256"], "vocabulary_sha256": m["mapping_sha256"],
                    "transform": original["transform"]}
                if target != expected or any(name not in ["semantic_extensions." + c for c in original["columns"]] for name in entry["columns"]):
                    raise ValueError("Narrative replacement must bind exact appended semantic328 target")
            values = self._matrix(entry)
            indices = []
            for name in entry["columns"]:
                if name not in self.item_columns or name in replaced:
                    raise ValueError("Named replacement target is missing or duplicated")
                index = self.item_columns.index(name)
                if index < 197:
                    raise ValueError("Named replacement cannot change original197")
                indices.append(index)
                replaced.add(name)
            before = text.array_hash(self.item_features[:, indices])
            if entry.get("replaced_values_sha256") != before:
                raise ValueError("Named replacement static preimage hash mismatch")
            if not self.replacements:
                self.item_features = self.item_features.copy()
            self.item_features[:, indices] = values
            self.replacements.append({"family": entry["family"], "columns": entry["columns"],
                                      "preimage_sha256": before, "matrix_sha256": entry["sha256"],
                                      "target_extension": entry.get("target_extension")})
        matrices = [self.item_features]
        self.text_features, self.text_missing, self.text_manifests = {}, {}, {}
        if set(m.get("text_caches", {})) != set(TEXT_FIELDS):
            raise ValueError("All three frozen E5 fields must materialize")
        text_vocab = text.canonical_hash({"ordered_item_ids": text._mapping(self.metadata), "special_tokens": text.SPECIALS})
        for field in TEXT_FIELDS:
            declaration = m["text_caches"][field]
            native_manifest = _path(self.root, declaration)
            if native_manifest.name != "manifest.json":
                raise ValueError("E5 declaration must pin its native manifest.json")
            directory = native_manifest.parent
            cache = text.load_item_cache(directory, expected_vocabulary_sha256=text_vocab,
                                         allow_test_encoder=self.fixture_mode)
            if (cache.manifest["field"] != field or
                    cache.manifest["source_pins"]["metadata"]["sha256"] != m["metadata"]["sha256"]):
                raise ValueError("E5 field/metadata binding mismatch")
            _walk_pins(self.root, cache.manifest["source_pins"])
            if cache.manifest["source_pins"]["items"]["sha256"] != m["raw_sources"]["items"]["sha256"]:
                raise ValueError("E5 raw item source binding mismatch")
            self._maps += [cache.embeddings, cache.missing, cache.token_lengths, cache.truncated]
            self.text_features[field], self.text_missing[field] = cache.embeddings, cache.missing
            self.text_manifests[field] = cache.manifest
            self._record_inputs(self.root, cache.manifest["source_pins"])
            for filename, spec in cache.manifest["outputs"].items():
                self._record_inputs(directory, {"path": filename, "sha256": spec["sha256"]})
            matrices += [cache.embeddings, cache.missing.astype(np.float32)[:, None]]
            self.item_columns += [f"e5.{field}.{i}" for i in range(cache.embeddings.shape[1])] + [f"e5.{field}.missing"]
        self.people_catalog = people.load_catalog(_path(self.root, m["people_catalog"]))
        if self.people_catalog["item_id_to_token"] != self.metadata["item_id_to_token"]:
            raise ValueError("Raw people catalogue mapping mismatch")
        _walk_pins(self.root, self.people_catalog["source_identity"])
        if self.people_catalog["source_identity"].get("metadata", {}).get("sha256") != m["metadata"]["sha256"]:
            raise ValueError("Raw people catalogue metadata binding mismatch")
        if self.people_catalog["source_identity"].get("items.csv", {}).get("sha256") != m["raw_sources"]["items"]["sha256"]:
            raise ValueError("Raw people catalogue item source binding mismatch")
        raw_people, self.item_categorical, people_columns, self.item_categorical_names = self._raw_people()
        matrices.append(raw_people)
        self.item_columns += people_columns
        self.item_features = np.concatenate(matrices, axis=1).astype(np.float32, copy=False)
        self.user_columns = list(self.local.user_columns)
        personal_metadata = self.metadata
        if self.scope is not None:
            series = m["runtime"]["personal"]["series_annotation"]
            personal_metadata = dict(self.metadata, series_annotation_path=_path(self.root, series).as_posix())
        self.personal = personal.build_index(self.raw_dir, personal_metadata, history_index=self.history,
            fit_cutoff=CAP, text_matrices=self.text_features, text_missing=self.text_missing,
            representation_manifest={field: value["signature"] for field, value in self.text_manifests.items()})
        if (self.scope is not None and self.personal.provenance["representation_manifest"].get("series_annotation_sha256") != series["sha256"]):
            raise ValueError("Native RM personal candidates consumed a different series annotation")
        self.user_columns += ["personal." + c for c in self.personal.user_columns]
        for field in sorted(self.text_features):
            self.user_columns += [f"text_profile.{field}.{i}" for i in range(self.text_features[field].shape[1])]
            self.user_columns += [f"text_profile.{field}.{name}" for name in ("known_n", "mean_norm", "valid", "zero_norm")]
        self.candidate_columns = self.item_columns + ["personal." + c for c in self.personal.columns]
        runtime = m["runtime"]
        self.audience = None if self.scope is not None else self._native_index("audience", audience, runtime["audience"])
        self.people_dynamics = None if self.scope is not None else self._native_index("people_dynamics", dynamics, runtime["people_dynamics"])
        related_declaration = runtime["related_interest"]
        related_parameters = dict(related_declaration.get("parameters", {}))
        if any(k in related_parameters for k in ("role", "fit_cutoff", "max_rows", "max_source_rows")):
            raise ValueError("Related RM cap/pilot overrides are forbidden")
        if "manifest" in related_declaration:
            if related_parameters:
                raise ValueError("Persisted related index cannot ignore runtime parameters")
            path = _path(self.root, related_declaration["manifest"])
            if path.name != "manifest.json":
                raise ValueError("Related declaration must pin native manifest.json")
            self.related = related.RelatedInterestIndex.load(path.parent)
            self._record_inputs(path.parent, _read(path)["arrays"])
        else:
            if not self.fixture_mode:
                raise ValueError("Production requires a reviewed persisted full RM related index")
            # In-memory source arrays are already raw-pinned above; the native
            # builder labels this as unpinned, so preserve that distinction.
            self.related = related.build(self.raw_dir, self.metadata_path, audience_index=self.audience,
                role="rm", fit_cutoff=CAP, **related_parameters)
        if (self.related.role != "rm" or self.related.fit_cutoff != CAP_DAY
                or self.related.mapping != self.metadata["item_id_to_token"]):
            raise ValueError("Related index returned wrong native RM role/cap/mapping")
        if "manifest" in related_declaration:
            self._native_lineage("related_interest", self.related)
        self.graph = _read(_path(self.root, m["franchise_graph"]))
        franchise.validate_graph(self.graph)
        if m.get("franchise_temporal_mode") not in ("historical_as_of", "retrospective_current"):
            raise ValueError("Explicit franchise metadata clock required")
        self.graph_scope_policies = copy.deepcopy(m["franchise_scope_policies"])
        for kind, policy in self.graph_scope_policies.items():
            for work, scope_id in (policy or {}).items():
                scope = self.graph["scopes"].get(scope_id) if isinstance(scope_id, str) else None
                if (work not in self.graph["works"] or not isinstance(scope, dict) or work not in scope["work_ids"]
                        or scope["order_kind"] != kind):
                    raise ValueError("Franchise policy must select a containing scope of the same order kind")
        self.prepared_graph = (franchise.prepare_graph(self.graph, self.metadata,
            order_kinds=FRANCHISE_ORDER_KINDS, scope_policies=self.graph_scope_policies, max_views=4)
            if self.scope is not None else None)
        self.entity_links = None if self.scope is not None else _read(_path(self.root, m["entity_links"]))
        if self.scope is None and (not isinstance(self.entity_links.get("works"), list) or not isinstance(self.entity_links.get("people"), list)):
            raise ValueError("Native accepted entity-link export is required")
        if self.scope is None:
            self.candidate_columns += list(self.audience.columns) + list(self.people_dynamics.columns)
        self.candidate_columns += ["related_interest." + c for c in self.related.columns]
        self.candidate_columns += ["local_dynamic." + c for c in self.dynamic.columns]
        for kind in FRANCHISE_ORDER_KINDS:
            self.candidate_columns += ["franchise." + kind + "." + c for c in franchise.COLUMNS]
        self.date_features = m.get("date_features", [])
        self.date_clocks = {}
        external_families = {"external_interest"} if self.scope is not None else {"external_facts", "external_interest"}
        if len(self.date_features) != len(external_families) or {d.get("family") for d in self.date_features} != external_families:
            raise ValueError("Exact admitted external clocked feature families must materialize")
        for declaration in self.date_features:
            if (declaration.get("role") != "rm" or declaration.get("fit_cutoff_exclusive") != CAP
                    or declaration.get("clock") != "strict_prefix" or not declaration.get("banks")):
                raise ValueError("External banks require native RM/Aug2/strict-prefix clocks")
            rm._names(declaration["columns"])
            clock = self._external_clock(declaration)
            self.date_clocks[declaration["family"]] = clock
            for key, bank in declaration["banks"].items():
                origin = self._bank_clock(key, bank, clock)
                values = self._matrix(bank)
                if bank["columns"] != declaration["columns"]:
                    raise ValueError("External bank column schema mismatch")
                if clock is not None:
                    masks = [i for i, name in enumerate(bank["columns"])
                             if name.startswith("wiki_") and name.endswith("_temporal_mask")]
                    if not masks or np.any(values[:, masks] != 0):
                        raise ValueError("Retrospective Wikipedia strict temporal masks must remain zero")
                if isinstance(values, np.memmap):
                    values._mmap.close()
            self.candidate_columns += [declaration["family"] + "." + c for c in declaration["columns"]]
        rm._names(self.user_columns)
        rm._names(self.candidate_columns)
        self.limits = {"batch_rows": 256, "profile_cache_rows": 128, "bank_cache_dates": 2, **m.get("limits", {})}
        if any(type(v) is not int or not 1 <= v <= 4096 for v in self.limits.values()):
            raise ValueError("Positive bounded complete RM cache/batch limits required")
        self.provenance = {"fit_cutoff_exclusive": CAP, "metadata_availability": "static_snapshot_unverified",
            "source_pins": {p["path"]: p for p in m["source_pins"]}, "role": "rm",
            "outcome_availability": "strict last-watch-date assumption for aggregated pairs",
            "components": {"local": self.local.provenance, "personal": self.personal.provenance,
                **({} if self.scope is not None else {"audience": self.audience.provenance, "people_dynamics": self.people_dynamics.provenance}),
                "related_interest": self.related.provenance}, "representations": REPRESENTATIONS}
        self._contract = {"contract": CONTRACT, "status": "fixture_only" if self.fixture_mode else "admitted_full",
            "role": "rm", "fit_cutoff_exclusive": CAP, "metadata_sha256": m["metadata"]["sha256"],
            "mapping_sha256": m["mapping_sha256"], "feature_columns": {"user": self.user_columns,
            "candidate": self.candidate_columns}, "family_acceptance": self.acceptance,
            "agreement_components": AGREEMENTS, "representations": REPRESENTATIONS,
            "people_backend": copy.deepcopy(PEOPLE_BACKEND),
            "late_integration": copy.deepcopy(LATE_INTEGRATION),
            "franchise_scope_policies": copy.deepcopy(self.graph_scope_policies),
            "user_diagnostics": {"manifest": copy.deepcopy(m["user_diagnostics"]), "model_columns": [], "diagnostic_only": True},
            "provider_manifest": copy.deepcopy(self._provider_pin), "provider_signature": m["signature"],
            "completion_event_store": copy.deepcopy(m["completion_event_store"]),
            "static_replacements": copy.deepcopy(self.replacements),
            "source_pins": copy.deepcopy(m["source_pins"])}
        if self.scope is not None:
            active, extra = self.scope["active_families"], self.scope["additional_ready_components"]
            self._contract.update(integration_mode="scoped_production", status="fixture_only" if self.fixture_mode else "admitted_scoped",
                scope_id=READY_SCOPE_ID, scope_decision=copy.deepcopy(m["scope_decision"]), active_families=copy.deepcopy(active),
                additional_ready_components=copy.deepcopy(extra), materialized_families=copy.deepcopy(active),
                materialized_components=copy.deepcopy(extra), family_acceptance={k: self.acceptance[k] for k in active},
                component_acceptance=copy.deepcopy(self.acceptance), agreement_components={}, all81_complete=False,
                full_package_admitted=False, full_row_fit_admitted=not self.fixture_mode, people_backend=copy.deepcopy(READY_PEOPLE_BACKEND))
            self._contract["external_clocks"] = {name: copy.deepcopy(clock) for name, clock in self.date_clocks.items()}
        self._contract["input_pins"] = [{"path": path, "sha256": sha256}
            for path, sha256 in sorted(self._inputs.items()) if path != self.path.as_posix()]
        self._contract["schema_sha256"] = rm.canonical_hash(self._contract["feature_columns"])
        _walk_pins(self.root, m)

    def _native_index(self, name, module, declaration):
        parameters = dict(declaration.get("parameters", {}))
        if any(key in parameters for key in ("role", "fit_cutoff", "max_rows", "max_source_rows")):
            raise ValueError("Native cap/pilot overrides are forbidden: " + name)
        if "manifest" in declaration:
            if parameters:
                raise ValueError("Persisted native index cannot ignore runtime parameters: " + name)
            path = _path(self.root, declaration["manifest"])
            if path.name != "manifest.json":
                raise ValueError("Native index declaration must pin manifest.json: " + name)
            native = module.AudienceIndex.load(path.parent, mmap_mode="r", verify_hashes=True) if name == "audience" else module.PeopleDynamicsIndex.load(path.parent)
            if name == "audience":
                for spec in native.storage_manifest["arrays"].values():
                    self._record_inputs(path.parent, {"path": spec["file"], "sha256": spec["sha256"]})
            else:
                self._record_inputs(path.parent, {"path": "index.npz", "sha256": _read(path)["index_npz_sha256"]})
        else:
            native = module.AudienceIndex.from_csv(self.raw_dir, self.metadata_path, fit_cutoff=CAP, **parameters) if name == "audience" else module.build(self.raw_dir, self.metadata_path, role="rm", fit_cutoff=CAP, **parameters)
        if name == "people_dynamics" and not isinstance(native, module.FastPeopleDynamicsIndex):
            raise ValueError("Dedicated RM people extraction did not return fast-exact backend")
        if native.fit_cutoff != CAP_DAY:
            raise ValueError("Native index cap is not Aug2: " + name)
        if name == "people_dynamics" and (native.provenance.get("fit_role") != "rm" or native.mapping != self.metadata["item_id_to_token"]):
            raise ValueError("Native people role/mapping mismatch")
        if name == "audience":
            reverse = np.zeros(self.metadata["vocab_size"], np.int64)
            for key, token in self.metadata["item_id_to_token"].items():
                reverse[token] = int(key)
            known = native.tokens >= 3
            if not np.array_equal(reverse[native.tokens[known]], native.raw_items[known]):
                raise ValueError("Native audience exact mapping mismatch")
        if "manifest" in declaration:
            self._native_lineage(name, native)
        elif not self.fixture_mode:
            if name == "audience" and native.provenance.get("source_scope") != "full_source":
                raise ValueError("Production rejects an incomplete native audience source")
            if name == "people_dynamics" and native.provenance.get("pilot_only") is not False:
                raise ValueError("Production rejects a native people pilot")
        return native

    def _native_lineage(self, name, native):
        provenance = native.provenance
        if name == "audience":
            sources = getattr(native, "source_pins", {})
            full = provenance.get("source_scope") == "full_source"
            required = ("items", "users", "interactions", "metadata")
        elif name == "people_dynamics":
            sources = provenance.get("source_pins", {})
            full = provenance.get("pilot_only") is False
            required = ("items", "interactions", "metadata")
        else:
            sources = {**provenance.get("audience_source_pins", {}), **provenance.get("source_pins", {})}
            full = provenance.get("pilot_only") is False and provenance.get("source_scope") == "full_source"
            required = ("items", "users", "interactions", "metadata")
        if not full:
            raise ValueError("Persisted native index is not a full source: " + name)
        for key in required:
            expected = self.manifest["metadata"] if key == "metadata" else self.manifest["raw_sources"][key]
            actual = sources.get(key, {})
            if actual.get("sha256") != expected["sha256"]:
                raise ValueError("Persisted native raw source identity mismatch: " + name + "/" + key)
            _path(self.root, actual)
        _walk_pins(self.root, sources)
        self._record_inputs(self.root, sources)

    def _raw_people(self):
        size = self.metadata["vocab_size"]
        numeric, columns, categoricals, names = [], [], [], []
        for role in ("actors", "directors"):
            declaration = self.people_catalog["roles"][role]
            values = np.zeros((size, 8), np.float32)
            cats = np.full((size, 3), "<PAD>", dtype=object)
            for token in range(3, size):
                people = declaration["item_person_ids"][token]
                ranked = declaration["top3_person_ids"][token]
                frequencies = [declaration["card_counts"][p] for p in people]
                values[token] = [len(people), not people, declaration["item_rare_count"][token],
                    np.mean(frequencies) if frequencies else 0, max(frequencies, default=0),
                    *[index < len(ranked) for index in range(3)]]
                cats[token] = [role + ":" + declaration["keys"][p] for p in ranked] + ["<MISSING>" if not people else "<PAD>"] * (3 - len(ranked))
            numeric.append(values)
            categoricals.append(cats.astype(str))
            columns += [f"raw_people.{role}.{c}" for c in ("n", "missing", "rare_n", "card_frequency_mean", "card_frequency_max", "top1", "top2", "top3")]
            names += [f"raw_people.{role}.identity_{i}" for i in range(3)]
        return np.concatenate(numeric, axis=1), np.column_stack(categoricals), columns, names

    def _external_clock(self, declaration):
        regime = declaration.get("temporal_regime")
        if regime is None:
            if self.scope is not None and not self.fixture_mode:
                raise ValueError("Ready external interest requires its explicit retrospective source clock")
            return None
        if (self.scope is None or declaration.get("family") != "external_interest" or regime != READY_EXTERNAL_REGIME
                or declaration.get("strict_asof_claim") is not False):
            raise ValueError("Retrospective external regime is restricted to ready fixed403 interest")
        sources = declaration.get("retrospective_sources", {})
        if set(sources) != set(READY_EXTERNAL_PINS):
            raise ValueError("Exact fixed403 retrospective source pins required")
        for name, descriptor in sources.items():
            if not self.fixture_mode and descriptor.get("sha256") != READY_EXTERNAL_PINS[name]:
                raise ValueError("Reviewed fixed403 source substitution: " + name)
            _path(self.root, descriptor)
        collection = _read(_path(self.root, sources["collection"]))
        receipts = collection.get("results", [])
        if not receipts or not self.fixture_mode and len(receipts) != 1321:
            raise ValueError("Fixed403 response population differs")
        def instant(value):
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("Actual external retrieval needs explicit UTC/timezone")
            return parsed.astimezone(timezone.utc)
        retrieved = max((row["receipt"]["retrieved_at"] for row in receipts), key=instant)
        if not self.fixture_mode:
            native = _read(_path(self.root, sources["bank_manifest"]))
            if (native.get("columns") != declaration["columns"] or native.get("strict_asof_claim") is not False
                    or native.get("metadata_sha256") != self.manifest["metadata"]["sha256"]
                    or native.get("vocabulary_sha256") != self.manifest["mapping_sha256"]):
                raise ValueError("Ready fixed403 native schema/metadata/identity regime differs")
        self._record_inputs(self.root, sources)
        return {"regime": regime, "retrospective_sources": copy.deepcopy(sources),
            "actual_response_retrieved_at": retrieved, "actual_max_known_at_day": instant(retrieved).date().toordinal(),
            "identity_known_at": None, "retrospective_identity": True, "strict_asof_claim": False,
            "modeled_lag_hours": 24, "label_fit_performed": False,
            "no_fit_sentinel": "label_fit_max_day=0 means no target-label fit; public2019 prior metadata is unchanged"}

    def _bank_clock(self, key, bank, retrospective=None):
        if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("Canonical exact-origin external bank key required")
        origin = int(key)
        if (not 1 <= origin <= CAP_DAY or bank.get("cutoff_exclusive") != origin
                or bank.get("role") != "rm" or bank.get("whole_day_exclusive") is not True):
            raise ValueError("External bank exceeds RM/query clock")
        if retrospective is not None:
            event_day = bank.get("numeric_max_event_day")
            if (type(event_day) is not int or not 0 < event_day <= origin - 3
                    or bank.get("max_event_day") != event_day or bank.get("modeled_lag_hours") != 24
                    or bank.get("modeled_numeric_available_at") != date.fromordinal(event_day + 2).isoformat() + "T00:00:00Z"
                    or event_day + 2 >= origin or bank.get("numeric_prefix_valid") is not True):
                raise ValueError("Retrospective external numeric event/modeled availability crosses query")
            if (bank.get("actual_response_retrieved_at") != retrospective["actual_response_retrieved_at"]
                    or bank.get("max_known_at_day") != retrospective["actual_max_known_at_day"]
                    or bank.get("identity_known_at", "missing") is not None
                    or bank.get("strict_asof_claim") is not False or bank.get("retrospective_identity") is not True
                    or bank.get("label_fit_performed") is not False or type(bank.get("label_fit_max_day")) is not int
                    or bank.get("label_fit_max_day") != 0):
                raise ValueError("Retrospective external actual source knowledge/no-fit evidence differs")
            return origin
        for field in ("max_event_day", "max_known_at_day", "label_fit_max_day"):
            value = bank.get(field)
            if type(value) is not int or not 0 <= value < origin:
                raise ValueError("External bank future/own-day evidence: " + field)
        return origin

    def _queries(self, users, origins, x=None, lengths=None, history_cutoffs=None):
        users = rm.checked_ids(users, "users")
        origins = np.asarray(origins)
        if (origins.dtype.kind not in "iu" or origins.shape != users.shape or
                np.any(origins < 1) or np.any(origins > CAP_DAY) or len(users) > self.limits["batch_rows"]):
            raise ValueError("RM origins must be explicit aligned dates <=Aug2 within batch budget")
        if history_cutoffs is not None and not np.array_equal(origins, np.asarray(history_cutoffs)):
            raise ValueError("RM state cutoff must equal the explicit frozen origin")
        expected, expected_lengths, _ = self.raw.history_batch(users, origins)
        if x is not None and (not np.array_equal(x, expected) or not np.array_equal(lengths, expected_lengths)):
            raise ValueError("RM context differs from exact strictly-past raw H63")
        self._event_ids(users, origins, expected, expected_lengths)
        return users, origins.astype(np.int64), expected, expected_lengths

    def _event_ids(self, users, origins, x, lengths):
        ids = self.event_store.derive_query_events(users, origins, x=x, lengths=lengths, history_cutoffs=origins)
        arrays = self.event_store.arrays
        valid = ids >= 0
        if ids.shape != (len(users), 63) or not np.array_equal(valid.sum(axis=1), lengths - 1):
            raise ValueError("Completion event-ID shape/history binding mismatch")
        if valid.any():
            selected = ids[valid]
            dates = arrays["date"][selected]
            queries = np.broadcast_to(origins[:, None], ids.shape)[valid]
            if np.any((dates >= queries) | (dates >= CAP_DAY)):
                raise ValueError("Completion selected event is not strictly before RM query/Aug2")
            raw_rows = arrays["raw_row_index"][selected]
            if np.any(raw_rows < 0) or np.any(raw_rows >= len(self.history.users)):
                raise ValueError("Completion raw event identity exceeds the admitted source")
            pairs = (("user_id", "users"), ("raw_item_id", "items"), ("date", "days"),
                     ("token", "tokens"), ("outcome", "outcomes"), ("parser_status", "statuses"))
            for field, name in pairs:
                if not np.array_equal(arrays[field][selected], getattr(self.history, name)[raw_rows]):
                    raise ValueError("Completion event/raw H63 identity mismatch: " + field)
            residuals = self.history.residuals[raw_rows]
            if not np.allclose(arrays["residual"][selected], residuals, rtol=0, atol=2e-7):
                raise ValueError("Completion event residual differs from strict RM baseline")
        return ids

    def query_event_ids(self, users, origins, *, x, lengths, history_cutoffs=None):
        users, origins, x, lengths = self._queries(users, origins, x, lengths, history_cutoffs)
        return self._event_ids(users, origins, x, lengths)

    def _state(self, user, origin, x, length):
        key = (int(user), int(origin), int(length), x.tobytes())
        if key not in self._states:
            users, origins = np.array([user], np.int64), np.array([origin], np.int64)
            profile = self.personal.prepare_queries(users, origins, x=x[None], lengths=np.array([length]), history_cutoffs=origins)
            indices, mask, _, _, _ = self.history._history(users, origins, origins, x[None], np.array([length]))
            raw_history = [{"item_id": int(self.history.items[i]), "token": int(self.history.tokens[i]),
                            "date": date.fromordinal(int(self.history.days[i])).isoformat()} for i in indices[0, mask[0]]]
            tail = [profile.user_features[0]]
            for field in sorted(self.text_features):
                text = profile.text[field]
                tail += [text["profile"][0], np.array([text[k][0] for k in ("n_known", "mean_norm", "valid", "zero_norm")])]
            self._states[key] = {"personal": profile, "user_tail": np.concatenate(tail).astype(np.float32), "raw_history": raw_history}
            while len(self._states) > self.limits["profile_cache_rows"]:
                self._states.popitem(last=False)
        self._states.move_to_end(key)
        return self._states[key]

    def user_features(self, user_ids, query_dates, x, lengths, history_cutoffs=None, dynamic_bank=None):
        users, origins, x, lengths = self._queries(user_ids, query_dates, x, lengths, history_cutoffs)
        if not len(users):
            return np.zeros((0, len(self.user_columns)), np.float32)
        kwargs = {} if dynamic_bank is None else {"dynamic_bank": dynamic_bank}
        base = self.local.user_features(users, origins, x, lengths, history_cutoffs=origins, **kwargs)
        tail = [self._state(u, q, a, n)["user_tail"] for u, q, a, n in zip(users, origins, x, lengths)]
        values = np.concatenate((base, np.stack(tail)), axis=1).astype(np.float32)
        if values.shape != (len(users), len(self.user_columns)) or not np.isfinite(values).all():
            raise ValueError("Invalid complete RM user features")
        return values

    def _external(self, declaration, origin):
        key = (declaration["family"], int(origin))
        if key not in self._banks:
            bank = declaration["banks"].get(str(int(origin)))
            if bank is None:
                raise ValueError("Exact-origin RM external bank is missing")
            self._bank_clock(str(int(origin)), bank, self.date_clocks[declaration["family"]])
            self._banks[key] = self._matrix(bank)
            while len(self._banks) > self.limits["bank_cache_dates"] * len(self.date_features):
                _, old = self._banks.popitem(last=False)
                if isinstance(old, np.memmap):
                    old._mmap.close()
        self._banks.move_to_end(key)
        return self._banks[key]

    def candidate_features(self, item_tokens, query_dates, history_cutoffs=None, user_ids=None):
        import kion_franchise_progress as franchise
        if user_ids is None:
            raise ValueError("User IDs bind every complete RM candidate to immutable s0")
        users, origins, x, lengths = self._queries(user_ids, query_dates, history_cutoffs=history_cutoffs)
        tokens = rm.checked_ids(item_tokens, "candidate tokens")
        if tokens.shape != users.shape or np.any(tokens < 3) or np.any(tokens >= self.metadata["vocab_size"]):
            raise ValueError("Mapped real candidate tokens must align")
        result = np.zeros((len(tokens), len(self.candidate_columns)), np.float32)
        reverse = {value: key for key, value in self.metadata["item_id_to_token"].items()}
        for row in np.argsort(origins, kind="stable"):
            user, origin, token = int(users[row]), int(origins[row]), int(tokens[row])
            state = self._state(user, origin, x[row], int(lengths[row]))
            ids, days, uids = np.array([token]), np.array([origin]), np.array([user])
            blocks = [self.item_features[ids], self.personal.candidate_features(state["personal"], ids)[0]]
            if self.audience is not None:
                blocks.append(self.audience.features(ids, days, user_ids=uids, role="rm_train",
                    x=x[row:row+1], lengths=lengths[row:row+1], history_cutoffs=days))
            for index in (self.people_dynamics, self.related, self.dynamic):
                if index is not None:
                    blocks.append(index.features(ids, days, history_cutoffs=days, user_ids=uids))
            when = date.fromordinal(origin).isoformat()
            for kind in FRANCHISE_ORDER_KINDS:
                if self.prepared_graph is not None:
                    progress = self.prepared_graph.candidate_progress(state["raw_history"], [reverse[token]],
                        when, when, temporal_mode=self.manifest["franchise_temporal_mode"], order_kind=kind)
                else:
                    progress = franchise.candidate_progress(state["raw_history"], [reverse[token]], self.metadata,
                        self.graph, when, when, temporal_mode=self.manifest["franchise_temporal_mode"],
                        order_kind=kind, scope_policy=self.graph_scope_policies[kind])
                blocks.append(progress["values"])
            blocks += [self._external(declaration, origin)[ids] for declaration in self.date_features]
            result[row] = np.concatenate(blocks, axis=1)[0]
        # Fixed transforms preserve zero sentinels and all explicit availability masks.
        for column, name in enumerate(self.candidate_columns):
            if name.endswith(("global_cutoff_ordinal", "query_state_cutoff_ordinal", "cutoff_ordinal")):
                result[:, column] = np.where(result[:, column] != 0,
                    (result[:, column] - date(2020, 1, 1).toordinal()) / 365.25, 0)
        if not np.isfinite(result).all():
            raise ValueError("Invalid complete RM candidate features")
        return result

    def contract(self):
        return copy.deepcopy(self._contract)

    def close(self):
        for value in getattr(self, "_maps", []):
            if isinstance(value, np.memmap) and value._mmap is not None and not value._mmap.closed:
                value._mmap.close()
        for name in ("audience", "people_dynamics", "related", "event_store"):
            value = getattr(self, name, None)
            if value is not None and hasattr(value, "close"):
                value.close()
        self._maps = []
        for name in ("_states", "_banks"):
            getattr(self, name, {}).clear()


def _feature_contract(value, *, fixture_mode=False, scope_decision=None):
    if (not isinstance(value, dict) or value.get("contract") != CONTRACT or value.get("role") != "rm"
            or value.get("fit_cutoff_exclusive") != CAP
            or value.get("representations") != REPRESENTATIONS
            or value.get("late_integration") != LATE_INTEGRATION
            or value.get("user_diagnostics", {}).get("model_columns") != []
            or value.get("user_diagnostics", {}).get("diagnostic_only") is not True):
        raise ValueError("A complete RM family/schema/role receipt is required; old RM bundles are incompatible")
    scope = _ready_scope(Path.cwd(), {**value, "agreements": value.get("agreement_components")},
                         scope_decision=scope_decision, fixture=fixture_mode)
    families = COMPONENTS if scope is None else scope["active_families"]
    expected = "fixture_only" if fixture_mode else "admitted_scoped" if scope is not None else "admitted_full"
    accepted = "fixture_only" if fixture_mode else "accepted_scoped" if scope is not None else "accepted_full"
    if scope is None:
        if value.get("agreement_components") != AGREEMENTS or value.get("people_backend") != PEOPLE_BACKEND:
            raise ValueError("Full RM agreement/backend receipt mismatch")
    elif (value.get("people_backend") != READY_PEOPLE_BACKEND or value.get("all81_complete") is not False
            or value.get("full_package_admitted") is not False or value.get("full_row_fit_admitted") != (not fixture_mode)
            or value.get("materialized_families") != families
            or value.get("materialized_components") != scope["additional_ready_components"]
            or set(value.get("component_acceptance", {})) != set(families + scope["additional_ready_components"])
            or any(v != accepted for v in value["component_acceptance"].values())):
        raise ValueError("Scoped RM materialization/production proof differs")
    if scope is not None and not fixture_mode:
        clocks = value.get("external_clocks", {})
        clock = clocks.get("external_interest") if isinstance(clocks, dict) else None
        if (not isinstance(clocks, dict) or set(clocks) != {"external_interest"} or not isinstance(clock, dict)
                or clock.get("regime") != READY_EXTERNAL_REGIME or clock.get("strict_asof_claim") is not False
                or clock.get("retrospective_identity") is not True or clock.get("identity_known_at", "missing") is not None
                or clock.get("modeled_lag_hours") != 24 or clock.get("label_fit_performed") is not False
                or set(clock.get("retrospective_sources", {})) != set(READY_EXTERNAL_PINS)
                or any(clock["retrospective_sources"][key].get("sha256") != sha for key, sha in READY_EXTERNAL_PINS.items())):
            raise ValueError("Ready RM fixed403 retrospective clock receipt differs")
    if value.get("status") != expected or set(value.get("family_acceptance", {})) != set(families):
        raise ValueError("Complete RM production/fixture acceptance mismatch")
    if any(v != accepted for v in value["family_acceptance"].values()):
        raise ValueError("Incomplete complete RM family acceptance")
    columns = value.get("feature_columns", {})
    if set(columns) != {"user", "candidate"} or value.get("schema_sha256") != rm.canonical_hash(columns):
        raise ValueError("Complete RM feature schema hash mismatch")
    signature = value.get("provider_signature", "")
    if len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
        raise ValueError("Complete RM provider signature binding is missing")
    for names in columns.values():
        rm._names(names)
    inputs = value.get("input_pins")
    if (not isinstance(inputs, list) or not inputs or any(not isinstance(p, dict) or set(p) != {"path", "sha256"}
            for p in inputs) or len({p["path"] for p in inputs}) != len(inputs)):
        raise ValueError("Complete RM checked input closure is missing/duplicated")
    return value


def build_complete_reward_data(manifest_path, output, *, row_caps=None, batch_size=256,
                               fixture_mode=False, local_factory=None, progress=None, scope_decision=None):
    """Materialize observed rows only after typed component admission."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Complete RM output must be a new directory")
    if type(batch_size) is not int or not 1 <= batch_size <= 4096:
        raise ValueError("Positive bounded batch size required")
    started = time.perf_counter()
    provider = CompleteRewardProvider.load(manifest_path, fixture_mode=fixture_mode, local_factory=local_factory,
                                           scope_decision=scope_decision)
    try:
        if batch_size > provider.limits["batch_rows"]:
            raise ValueError("Dataset batch exceeds admitted provider limit")
        raw = provider.raw
        selected, full_counts, sampled_counts, exclusions = rows_kernel.select_rows(raw, row_caps)
        required_origins = set(map(int, np.minimum(raw.days[np.concatenate(list(selected.values()))], CAP_DAY)))
        for declaration in provider.date_features:
            if not required_origins.issubset(set(map(int, declaration["banks"]))):
                raise ValueError("Exact-origin external RM bank coverage is incomplete")
        mapping = provider.metadata["item_id_to_token"]
        item_ids = np.array(sorted(map(int, mapping), key=lambda key: mapping[str(key)]), np.int64)
        template = np.zeros((len(item_ids), len(provider.candidate_columns)), np.float32)
        template[:, :len(provider.item_columns)] = provider.item_features[3:]
        paths = {"builder": Path(__file__), "row_kernel": Path(rows_kernel.__file__), "boosting": Path(rm.__file__)}
        provenance = {**provider.provenance, "candidate_features_dynamic": True, "static_candidate_fallback": False,
                      "source_pins": rows_kernel._pin_sources(provider, paths), "feature_pack": provider.contract()}
        catalog = rm.FeatureCatalog(item_ids, template, provider.user_columns, provider.candidate_columns,
            item_categorical=provider.item_categorical[3:], item_categorical_names=provider.item_categorical_names,
            history_mask=[not column.startswith("user.") for column in provider.user_columns], provenance=provenance)
        if catalog.mapping != mapping:
            raise ValueError("Complete RM catalogue/policy mapping mismatch")
        output.mkdir(parents=True)
        files, sidecars, bindings, complete_sidecars, event_bindings = {}, {}, {}, {}, {}
        for role, indices in selected.items():
            n = len(indices)
            specs = {"numeric": (np.float32, (n, len(provider.user_columns))),
                "candidate_numeric": (np.float32, (n, len(provider.candidate_columns))),
                "categorical": ("U1", (n, 0)), "observed": (bool, (n,)), "reward": (np.float32, (n,))}
            specs.update({key: (np.int64, (n,)) for key in
                ("user_id", "item_id", "target_date", "history_cutoff", "history_max_date", "history_length")})
            arrays = {key: np.lib.format.open_memmap(output / f"{role}.{key}.npy", mode="w+", dtype=dtype, shape=shape)
                      for key, (dtype, shape) in specs.items()}
            xs = np.lib.format.open_memmap(output / f"{role}.x.npy", mode="w+", dtype=np.int64, shape=(n, 64))
            lengths = np.lib.format.open_memmap(output / f"{role}.lengths.npy", mode="w+", dtype=np.int64, shape=(n,))
            event_ids = np.lib.format.open_memmap(output / f"{role}.event_ids.npy", mode="w+", dtype=np.int64, shape=(n, 63))
            try:
                written = 0
                for positions, idx, origins, bank in rows_kernel._feature_batches(provider, raw, indices, batch_size):
                    users = raw.users[idx]
                    x, lens, maxima = raw.history_batch(users, origins)
                    arrays["numeric"][positions] = provider.user_features(users, origins, x, lens,
                        history_cutoffs=origins, dynamic_bank=bank)
                    arrays["candidate_numeric"][positions] = provider.candidate_features(raw.tokens[idx], origins,
                        history_cutoffs=origins, user_ids=users)
                    for name, values in {"user_id": users, "item_id": raw.items[idx], "target_date": raw.days[idx],
                        "history_cutoff": origins, "history_max_date": maxima, "history_length": lens - 1,
                        "reward": raw.outcomes[idx], "observed": True}.items():
                        arrays[name][positions] = values
                    xs[positions], lengths[positions] = x, lens
                    event_ids[positions] = provider.query_event_ids(users, origins, x=x, lengths=lens, history_cutoffs=origins)
                    written += len(positions)
                    if progress is not None:
                        progress({"role": role, "rows": written, "total": n, "seconds": time.perf_counter() - started})
                for array in (*arrays.values(), xs, lengths, event_ids):
                    array.flush()
                rm.validate_rows(arrays, role, catalog)
                bindings[role] = rm.canonical_hash({key: rm.array_hash(arrays[key]) for key in
                    ("user_id", "item_id", "target_date", "history_cutoff", "reward")})
                for key in arrays:
                    name = f"{role}.{key}.npy"
                    files[name] = rm.file_hash(output / name)
                for key in ("x", "lengths"):
                    name = f"{role}.{key}.npy"
                    sidecars[name] = rm.file_hash(output / name)
                name = f"{role}.event_ids.npy"
                complete_sidecars[name] = rm.file_hash(output / name)
                event_bindings[role] = rm.canonical_hash({"rows": bindings[role], "x": rm.array_hash(xs),
                    "lengths": rm.array_hash(lengths), "event_ids": rm.array_hash(event_ids),
                    "store": provider.manifest["completion_event_store"]["sha256"]})
            finally:
                for array in (*arrays.values(), xs, lengths, event_ids):
                    if array._mmap is not None and not array._mmap.closed:
                        array._mmap.close()
        np.savez_compressed(output / "catalog.npz", item_ids=catalog.item_ids, item_numeric=catalog.item_numeric,
                            item_categorical=catalog.item_categorical)
        files["catalog.npz"] = rm.file_hash(output / "catalog.npz")
        count = len(item_ids) + 3
        full_counts = np.pad(full_counts, (0, max(0, count - len(full_counts))))[:count]
        sampled_counts = np.pad(sampled_counts, (0, max(0, count - len(sampled_counts))))[:count]
        np.savez_compressed(output / "full_target_support.npz", item_id=item_ids, token=np.arange(3, count, dtype=np.int64),
                            full_raw_fit_count=full_counts[3:], sampled_fit_count=sampled_counts[3:])
        sidecars["full_target_support.npz"] = rm.file_hash(output / "full_target_support.npz")
        exclusions["target_support"] = {"full_raw_fit_rows": int(full_counts.sum()), "sampled_fit_rows": int(sampled_counts.sum()),
            "zero_support_items": int(np.count_nonzero(full_counts[3:] == 0)),
            "zero_support_set_preserved": bool(np.array_equal(full_counts > 0, sampled_counts > 0)),
            "invalid_source_outcomes": "excluded and counted; never clipped into valid labels"}
        # A source mutation leaves partial files, but never a ready manifest.
        provider.verify_inputs()
        _walk_pins(provider.root, provider.manifest)
        _path(provider.root, provider._provider_pin)
        rows_kernel._check_sources(rows_kernel._declared_sources(provider.provenance))
        provider_copy = output / "provider-manifest.json"
        provider_copy.write_bytes(provider._manifest_bytes)
        feature_pack = provider.contract()
        feature_pack["provider_manifest"] = pin(provider_copy)
        manifest = {"contract": rm.CONTRACT, "layout": "npy-directory-v1", "ready": True, "partition": rm.PARTITION,
            "split": {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08", "heldout_history": "frozen<2021-08-02"},
            "catalog": catalog.describe(), "files": files, "sidecars": sidecars, "row_bindings": bindings,
            "complete_sidecars": complete_sidecars, "event_bindings": event_bindings,
            "exclusions": exclusions, "builder_source_sha256": rm.file_hash(__file__), "feature_dtype": "float32",
            "feature_pack": feature_pack, "seconds": time.perf_counter() - started}
        rm.write_json(output / "manifest.json", manifest)
        return manifest
    finally:
        provider.close()


def _close_map(value):
    mapping = getattr(value, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def load_complete_reward_bundle(path, *, fixture_mode=False, scope_decision=None):
    root = Path(path).resolve()
    manifest = _read(root / "manifest.json")
    feature_pack = _feature_contract(manifest.get("feature_pack"), fixture_mode=fixture_mode, scope_decision=scope_decision)
    provider_path = _path(root, feature_pack["provider_manifest"])
    provider_manifest = _read(provider_path)
    _admit(provider_path.parent, provider_manifest, fixture_mode, scope_decision)
    if provider_manifest["signature"] != feature_pack["provider_signature"]:
        raise ValueError("Stored rows/provider signature mismatch")
    _walk_pins(root, feature_pack["source_pins"])
    _walk_pins(root, feature_pack["input_pins"])
    roles, catalog, loaded = rm.load_bundle(root)
    store = None
    try:
        catalog_contract = copy.deepcopy(catalog.provenance.get("feature_pack"))
        _feature_contract(catalog_contract, fixture_mode=fixture_mode, scope_decision=scope_decision)
        catalog_contract["provider_manifest"] = feature_pack["provider_manifest"]
        if catalog_contract != feature_pack:
            raise ValueError("Complete RM receipt differs from catalogue provenance")
        if (catalog.context_names != feature_pack["feature_columns"]["user"] or
                catalog.item_names != feature_pack["feature_columns"]["candidate"] or
                catalog.mapping_sha256 != feature_pack["mapping_sha256"]):
            raise ValueError("Complete RM row/catalogue/feature schema binding mismatch")
        expected_sidecars = {f"{role}.event_ids.npy" for role in rows_kernel.ROLES}
        if set(loaded.get("complete_sidecars", {})) != expected_sidecars:
            raise ValueError("Complete RM event-ID sidecar census mismatch")
        store = _load_store(provider_path.parent, provider_manifest, fixture_mode)
        for role, rows in roles.items():
            filename = f"{role}.event_ids.npy"
            if rm.file_hash(root / filename) != loaded["complete_sidecars"][filename]:
                raise ValueError("Complete RM event sidecar checksum mismatch")
            acquired = []
            try:
                for name in (filename, f"{role}.x.npy", f"{role}.lengths.npy"):
                    acquired.append(np.load(root / name, mmap_mode="r", allow_pickle=False))
                ids, x, lengths = acquired
                if ids.dtype != np.int64 or ids.shape != (len(rows["user_id"]), 63):
                    raise ValueError("Complete RM event sidecar schema mismatch")
                for begin in range(0, len(ids), 256):
                    end = min(len(ids), begin + 256)
                    expected = store.derive_query_events(rows["user_id"][begin:end], rows["history_cutoff"][begin:end],
                        x=x[begin:end], lengths=lengths[begin:end], history_cutoffs=rows["history_cutoff"][begin:end])
                    if not np.array_equal(ids[begin:end], expected):
                        raise ValueError("Complete RM event sidecar differs from raw H63 join")
                    valid = ids[begin:end] >= 0
                    if valid.any():
                        dates = store.arrays["date"][ids[begin:end][valid]]
                        origins = np.broadcast_to(rows["history_cutoff"][begin:end, None], valid.shape)[valid]
                        if np.any((dates >= origins) | (dates >= CAP_DAY)):
                            raise ValueError("Complete RM sidecar includes a future/Aug2 event")
                binding = rm.canonical_hash({"rows": loaded["row_bindings"][role], "x": rm.array_hash(x),
                    "lengths": rm.array_hash(lengths), "event_ids": rm.array_hash(ids),
                    "store": provider_manifest["completion_event_store"]["sha256"]})
                if binding != loaded.get("event_bindings", {}).get(role):
                    raise ValueError("Complete RM event sidecar row binding mismatch")
            finally:
                for value in acquired:
                    _close_map(value)
    except BaseException:
        # A retained exception traceback must not retain Windows file locks.
        # Successful return alone transfers role-map ownership to the caller.
        for rows in roles.values():
            for value in rows.values():
                _close_map(value)
        raise
    finally:
        if store is not None:
            store.close()
    return roles, catalog, loaded


def load_reward_provider_from_bundle(path, *, fixture_mode=False, local_factory=None, scope_decision=None):
    root = Path(path).resolve()
    manifest = _read(root / "manifest.json")
    contract = _feature_contract(manifest.get("feature_pack"), fixture_mode=fixture_mode, scope_decision=scope_decision)
    provider = CompleteRewardProvider.load(_path(root, contract["provider_manifest"]),
        fixture_mode=fixture_mode, local_factory=local_factory, scope_decision=scope_decision)
    actual = provider.contract()
    expected = copy.deepcopy(contract)
    actual["provider_manifest"] = expected["provider_manifest"]
    if actual != expected:
        provider.close()
        raise ValueError("Reconstructed complete RM provider contract differs from frozen dataset")
    return provider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-fit", type=int, default=100000)
    parser.add_argument("--max-select", type=int, default=20000)
    parser.add_argument("--max-calibration", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--scope-decision", type=Path)
    args = parser.parse_args(argv)
    value = build_complete_reward_data(args.provider_manifest, args.output,
        row_caps={"fit": args.max_fit, "select": args.max_select, "calibration": args.max_calibration},
        batch_size=args.batch_size, scope_decision=args.scope_decision,
        progress=lambda row: print(json.dumps(row), flush=True))
    print(json.dumps({"output": args.output.resolve().as_posix(), "rows": value["exclusions"]["role_rows"],
                      "schema": value["feature_pack"]["schema_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
