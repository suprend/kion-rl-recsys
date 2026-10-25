"""Assemble a complete-package manifest from explicit, reviewed evidence.

This module performs no acquisition, model import, raw-index construction, fit,
or inference. An inventory row, unknown-only output, and a population pilot do
not become full evidence merely because they have a matrix-shaped artifact.
"""
from __future__ import annotations

import argparse
import ast
from datetime import date
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

CONTRACT = "kion-complete-assembly-plan-v1"
EVIDENCE_CONTRACT = "kion-complete-component-evidence-v1"
REVIEW_CONTRACT = "kion-complete-component-review-v1"
FROZEN_PROVIDER_SHA256 = "169d404f8145a792ebc4d545e4548b8cedc8d01cc7f67dcb88dc8a1ca4def72e"
SCOPED_DECISION_SHA256 = "a674637029b96f564366fdda6e9c433a1cf3dab09721d191ad5a276b622cf962"
SCOPED_FAMILIES = ("local_base", "local_dynamic", "semantic_extensions", "text3fields", "people_embeddings",
    "personal_candidates", "franchise_progress", "external_interest")
SCOPED_READY_COMPONENTS = ("professions", "geography_time", "sports_settings", "story_format", "narrative_relations",
    "static_replacements", "related_interest", "completion_diagnostics", "user_diagnostics")
SCOPED_COMPONENTS = tuple(SCOPED_FAMILIES) + tuple(c for c in SCOPED_READY_COMPONENTS if c != "static_replacements")

# These stable IDs preserve the census inventory. They are interfaces/families,
# not 82 equally sized blocks and not a percentage-complete denominator.
AGREEMENT_COMPONENTS = {
    **{f"content.{n}": ("local_base",) for n in ("genres", "type", "release_year", "countries", "age_rating", "studios", "for_kids")},
    **{f"content.{n}": ("external_facts", "entity_links") for n in ("languages", "runtime", "awards", "tv_history")},
    **{f"content.{n}": ("external_interest", "entity_links") for n in ("external_ratings", "wikipedia")},
    **{f"content.{n}": ("local_dynamic",) for n in ("popularity", "completion", "weekday")},
    "content.related_interest": ("local_dynamic", "people_dynamics", "related_interest"),
    "content.related_releases": ("people_dynamics", "related_interest"),
    **{f"content.{n}": ("local_base", "semantic_extensions") for n in ("story_source", "themes", "narrative", "edition", "detailed_format", "series_membership")},
    "content.story_source": ("local_base", "story_format"),
    "content.detailed_format": ("local_base", "story_format"),
    "content.narrative": ("local_base", "narrative_relations"),
    "content.settings": ("local_base", "sports_settings"),
    "content.sport_activity": ("local_base", "sports_settings"),
    "content.geography": ("local_base", "geography_time"),
    "content.setting_time": ("local_base", "geography_time"),
    "content.professions": ("local_base", "professions"),
    "people.experience": ("local_base", "people_dynamics"),
    **{f"people.{n}": ("people_dynamics",) for n in ("genre_specialization", "format_specialization", "actor_popularity", "director_popularity")},
    **{f"people.{n}": ("local_base",) for n in ("director_actor", "actor_pair")},
    "people.external": ("external_facts", "entity_links"),
    **{f"people.learned_{n}": ("people_embeddings",) for n in ("director_mean", "actor_mean", "actor_top3")},
    **{f"text.e5_{n}": ("text3fields",) for n in ("title", "title_orig", "description")},
    **{f"personal.{n}": ("local_base",) for n in ("age", "sex", "kids", "income", "genre_profile", "type_profile", "decade_profile", "country_profile", "age_rating_profile", "recency", "activity", "gap_median", "gap_cv", "mean_completion", "high_completion", "low_completion", "typed_completion", "window_completion")},
    **{f"personal.{n}": ("local_base", "user_diagnostics") for n in ("activity", "typed_completion", "window_completion", "age_rating_profile")},
    **{f"personal.{n}": ("local_base", "completion_diagnostics") for n in ("relative_completion", "genre_relative", "type_relative", "country_relative", "decade_relative")},
    **{f"personal.familiarity_{n}": ("personal_candidates",) for n in ("actor", "director", "studio", "franchise")},
    **{f"personal.{n}": ("personal_candidates",) for n in ("historical_popularity", "novelty_director")},
    "personal.text_profile": ("personal_candidates", "text3fields"),
    **{f"research.multi_interest_{n}": ("personal_candidates", "candidate_model") for n in ("partition", "scoring")},
    **{f"research.series_{n}": ("franchise_progress",) for n in ("order", "progress")},
    **{f"research.audience_{n}": ("audience",) for n in ("demographics", "tastes", "coview")},
    "research.audience_factors": ("audience_factors",),
    "interface.candidate_head": ("candidate_model",),
}
DEFERRED_PROPOSALS = {"research.audience_factors": "Later research proposal; factor algorithm has no confirmed concrete approval. Preserve inventory; do not make it an agreed-feature full gate."}
EXTERNAL_IDS = {"content.languages", "content.runtime", "content.external_ratings", "content.awards", "content.tv_history", "content.wikipedia", "people.external"}
GRAPH_IDS = {"research.series_order", "research.series_progress"}
SEMANTIC_IDS = {"content." + n for n in ("story_source", "themes", "narrative", "settings", "setting_time", "geography", "professions", "sport_activity", "edition", "detailed_format")}


def coverage_axis(agreement, component):
    if component == "candidate_model":
        return "computational_interface"
    if ((agreement.startswith("personal.") and component in ("local_base", "personal_candidates", "completion_diagnostics", "user_diagnostics"))
            or agreement.startswith("research.multi_interest_") and component == "personal_candidates"
            or agreement == "research.series_progress"):
        return "prepared_queries"
    return "catalogue_items"


def _prepared_counts(root, base):
    counts = {}
    for split in ("train", "val"):
        path = checked_path(root, base["inputs"][split + ".npz"])
        # Read only the compressed NPY header; never inflate the full H63 array.
        with zipfile.ZipFile(path) as archive, archive.open("x.npy") as stream:
            version = np.lib.format.read_magic(stream)
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, _ = reader(stream)
        if len(shape) != 2 or shape[0] < 1:
            raise ValueError("Prepared query population header is invalid")
        counts[split] = int(shape[0])
    return counts


def _coverage(root, agreement, component, value, evidence, base, population, prepared_counts=None, base_manifest_pin=None):
    axis = coverage_axis(agreement, component)
    errors = []
    def resolved(pin):
        path = Path(pin["path"])
        return (path if path.is_absolute() else root / path).resolve()
    declared = {resolved(pin): pin for pin in evidence["artifacts"] + evidence["source_pins"]}
    def reviewed(pin):
        path = checked_path(root, pin)
        if path not in declared or declared[path]["sha256"] != pin["sha256"]:
            raise ValueError("Coverage references an unreviewed source/artifact")
        return path
    if axis == "computational_interface":
        if value.get("population_scope") != "tested_computational_interface" or value.get("population_size") is not None:
            errors.append("computational_interface_estimand_incorrect")
        if not value.get("runtime_contract") or not value.get("tested_shapes"):
            errors.append("computational_interface_contract_or_shapes_missing")
        else:
            reviewed(value["runtime_contract"])
            if any(not isinstance(shape, list) or not shape or any(type(n) is not int or n < 1 for n in shape) for shape in value["tested_shapes"]):
                raise ValueError("Computational interface test shapes are invalid")
        for pin in value.get("source_index_proofs", []):
            reviewed(pin)
        return errors
    if axis == "catalogue_items":
        total = population
        if value.get("population_scope") not in ("train_known_catalog", "catalogue_items"):
            errors.append("population_axis_incorrect")
        if evidence.get("source_kind") == "actual_data" and (value.get("vocabulary_size") != population + 3 or value.get("special_rows") != 3):
            errors.append("catalogue_special_rows_not_separated")
    else:
        total = sum(prepared_counts.values())
        if (value.get("population_scope") != "all_prepared_train_and_validation_queries"
                or value.get("split_counts") != prepared_counts):
            errors.append("prepared_query_population_incorrect")
        for split in ("train", "val"):
            recorded = value.get("prepared_pins", {}).get(split + ".npz")
            expected = base["inputs"][split + ".npz"]
            if not recorded or resolved(recorded) != resolved(expected) or recorded["sha256"] != expected["sha256"]:
                errors.append("prepared_query_source_pins_incomplete")
        proof = value.get("query_mapping_proof")
        if not proof:
            errors.append("prepared_query_join_proof_missing")
        else:
            native = read_json(reviewed(proof))
            if native.get("row_keys_sha256") != base["row_keys_sha256"]:
                errors.append("prepared_query_join_row_hash_mismatch")
            if native.get("contract") in ("kion-completion-event-store-v1", "kion-user-diagnostics-v1"):
                receipt = native.get("query_coverage_receipt", {})
                if (native.get("state") != "completed" or native.get("metadata_sha256") != base["inputs"]["metadata.json"]["sha256"]
                        or any(native.get("prepared_pins", {}).get(split + ".npz") != base["inputs"][split + ".npz"] for split in ("train", "val"))
                        or receipt.get("error_count") != 0 or any(receipt.get("splits", {}).get(split, {}).get("validated_rows") != n for split, n in prepared_counts.items())):
                    errors.append("prepared_query_join_proof_incomplete")
            else:
                if (not base_manifest_pin or resolved(proof) != resolved(base_manifest_pin) or proof["sha256"] != base_manifest_pin["sha256"]
                        or any(native.get("outputs", {}).get(split + "_user_features.npy", {}).get("shape", [None])[0] != n for split, n in prepared_counts.items())):
                    errors.append("prepared_query_cached_row_proof_incomplete")
    if (value.get("population_size") != total or value.get("assessed_count") != total or value.get("unassessed_count") != 0):
        errors.append("population_attempt_incomplete")
    supported, unknown = value.get("supported_count"), value.get("unknown_count")
    if value.get("support_status", "exact_counts") == "query_dependent_not_censused":
        if supported is not None or unknown is not None:
            raise ValueError("Query-dependent support must not fabricate population counts")
        cohort = value.get("dated_cohort", {})
        origins = cohort.get("origin_ordinals", [])
        if (cohort.get("measured") is not True or not _integer(cohort.get("rows"), 10**12) or cohort["rows"] < 1
                or not origins or any(type(day) is not int or not 1 <= day <= date(2021, 8, 9).toordinal() for day in origins)
                or not cohort.get("receipt")):
            errors.append("query_dependent_support_dated_cohort_missing")
        else:
            reviewed(cohort["receipt"])
    elif (value.get("support_status", "exact_counts") != "exact_counts" or not _integer(supported, total)
            or not _integer(unknown, total) or supported + unknown != total):
        raise ValueError("Coverage support/unknown counters are invalid")
    if agreement in EXTERNAL_IDS | GRAPH_IDS:
        acquisition = value if axis == "catalogue_items" else value.get("catalogue_acquisition", {})
        count = acquisition.get("supported_count")
        if (not _integer(count, population) or count == 0 or acquisition.get("acquisition_pass_complete") is not True
                or axis != "catalogue_items" and acquisition.get("population_size") != population):
            errors.append("external_or_graph_unknown_only_or_pilot_pass")
    return errors


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def array_hash(value):
    value = np.asarray(value)
    result = hashlib.sha256(json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return {"path": path.as_posix(), "sha256": digest(path)}


def checked_path(root, receipt):
    if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
        raise ValueError("A path/SHA receipt is required")
    path = Path(receipt["path"])
    path = path if path.is_absolute() else root / path
    if digest(path) != receipt.get("sha256"):
        raise ValueError("Receipt SHA mismatch: " + path.as_posix())
    return path.resolve()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def provider_contract(path):
    """Read literal schema declarations without importing torch or a provider."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names = {"CONTRACT", "REQUIRED_FAMILIES", "ROW_ALGORITHM", "NUMERIC_TRANSFORMS", "NUMERIC_UNITS", "SOURCE_FILES", "COMPONENT_ADAPTERS", "ASSEMBLY_PAYLOAD_FIELDS"}
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                result[name] = ast.literal_eval(node.value)
    for required in ("CONTRACT", "REQUIRED_FAMILIES", "NUMERIC_TRANSFORMS", "SOURCE_FILES"):
        if required not in result:
            raise ValueError("Provider literal declaration missing: " + required)
    if digest(path) == FROZEN_PROVIDER_SHA256:
        result["COMPONENT_ADAPTERS"] = list(result["REQUIRED_FAMILIES"]) + ["candidate_model"]
    elif "COMPONENT_ADAPTERS" not in result:
        raise ValueError("A revised provider must declare reviewed COMPONENT_ADAPTERS explicitly")
    return result


def scoped_decision(receipt, root):
    if not isinstance(receipt, dict) or receipt.get("sha256") != SCOPED_DECISION_SHA256:
        raise ValueError("Scoped production requires the exact human-approved scope decision")
    decision = read_json(checked_path(Path(root), receipt))
    if (decision.get("contract") != "kion-user-approved-ready-scope-v1"
            or decision.get("scope_id") != "ready-pack-20261020-r1"
            or decision.get("active_families") != list(SCOPED_FAMILIES)
            or decision.get("additional_ready_components") != list(SCOPED_READY_COMPONENTS)):
        raise ValueError("Scoped production decision identity/families mismatch")
    return decision


def extension_binding(declaration, root=None):
    root = Path(root or ".").resolve()
    path = Path(declaration["path"])
    namespace = declaration.get("namespace", declaration["family"])
    if not isinstance(namespace, str) or not (namespace == declaration["family"] or namespace.startswith(declaration["family"] + ".")):
        raise ValueError("Extension binding requires an unambiguous family namespace")
    return {"path": (path if path.is_absolute() else root / path).resolve().as_posix(),
        "sha256": declaration["sha256"], "family": declaration["family"],
        "namespace": namespace,
        "columns_sha256": canonical_hash(declaration["columns"]), "shape": declaration["shape"], "dtype": declaration["dtype"],
        "metadata_sha256": declaration["metadata_sha256"], "vocabulary_sha256": declaration["vocabulary_sha256"],
        "transform": declaration["transform"]}


def _replacement(declaration, base_pin, base, extensions=(), root=None):
    columns = base["feature_columns"]["item"]
    targets = declaration.get("columns")
    _names(targets)
    if (declaration.get("mode") != "replace_named_columns"
            or declaration.get("base_manifest_sha256") != base_pin["sha256"]
            or declaration.get("base_item_features_sha256") != base["outputs"]["item_features.npy"]["sha256"]
            or declaration.get("base_schema_hash") != base["schema_hash"]):
        raise ValueError("Named replacement does not bind the exact immutable base")
    target = declaration.get("target_extension")
    if target is None:
        if any(name not in columns for name in targets):
            raise ValueError("Replacement target is absent from the base named schema")
        if any(columns.index(name) < 197 for name in targets):
            raise ValueError("Replacement cannot overwrite the protected original197 prefix")
    else:
        if not isinstance(target, dict):
            raise ValueError("Replacement extension binding must be an object")
        matches = [extension for extension in extensions if extension_binding(extension, root) == target]
        preimage = declaration.get("replaced_values_sha256", "")
        if (len(matches) != 1 or target.get("transform") != "identity" or not isinstance(preimage, str)
                or len(preimage) != 64 or any(c not in "0123456789abcdef" for c in preimage)):
            raise ValueError("Replacement extension identity/schema/preimage binding mismatch")
        named = {target["namespace"] + "." + name for name in matches[0]["columns"]}
        if not set(targets).issubset(named):
            raise ValueError("Replacement target is absent from the bound extension named schema")
        extension = matches[0]
        selected = [extension["columns"].index(name[len(target["namespace"]) + 1:]) for name in targets]
        source = np.load(checked_path(Path(root or ".").resolve(), extension), mmap_mode="r", allow_pickle=False)
        try:
            if (list(source.shape) != target["shape"] or str(source.dtype) != target["dtype"]
                    or array_hash(np.asarray(source[:, selected], dtype=np.float32)) != preimage):
                raise ValueError("Extension preimage values mismatch")
        finally:
            if isinstance(source, np.memmap):
                source._mmap.close()
    if declaration.get("transform") != "identity":
        raise ValueError("Replacement coordinates require explicit identity units")


def _names(values):
    if (not isinstance(values, list) or not values or len(values) != len(set(values))
            or any(not isinstance(v, str) or not v for v in values)):
        raise ValueError("Nonempty unique named columns are required")


def _integer(value, maximum):
    return type(value) is int and 0 <= value <= maximum


def _native_blockers(value, path=""):
    """Respect explicit native pilot/failure flags; never infer ready from shape."""
    result = []
    if not isinstance(value, dict):
        return result
    if value.get("full_catalog_ready") is False or value.get("pilot_only") is True:
        result.append(path + "native_not_full")
    if value.get("state") in ("failed", "building", "interrupted"):
        result.append(path + "native_not_completed")
    if value.get("source_scope") in ("incomplete_source_order_slice", "pilot"):
        result.append(path + "native_source_pilot")
    for key in ("coverage", "provenance", "scope"):
        child = value.get(key)
        if isinstance(child, str) and "pilot" in child.casefold():
            result.append(path + key + ":pilot")
        elif isinstance(child, dict):
            result.extend(_native_blockers(child, path + key + "."))
    return result


def _artifact(root, item, metadata_sha, mapping_sha, vocab):
    path = checked_path(root, item)
    kind = item.get("kind")
    if kind == "matrix":
        if item.get("metadata_sha256") != metadata_sha or item.get("vocabulary_sha256") != mapping_sha:
            raise ValueError("Matrix metadata/vocabulary binding mismatch")
        _names(item.get("columns"))
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if (values.ndim != 2 or list(values.shape) != item.get("shape") or len(values) != vocab
                    or values.shape[1] != len(item["columns"]) or values.dtype.kind != "f"
                    or str(values.dtype) != item.get("dtype") or np.any(values[:3] != 0)):
                raise ValueError("Matrix schema/vocabulary/special row mismatch")
            for begin in range(0, len(values), 4096):
                if not np.isfinite(values[begin:begin + 4096]).all():
                    raise ValueError("Matrix contains nonfinite values")
        finally:
            if isinstance(values, np.memmap):
                values._mmap.close()
        if item.get("transform") not in ("identity", "signed_log1p"):
            raise ValueError("Explicit matrix transform required")
        if item["transform"] == "signed_log1p":
            indices = item.get("transform_columns")
            if (not isinstance(indices, list) or len(indices) != len(set(indices))
                    or any(type(i) is not int or not 0 <= i < len(item["columns"]) for i in indices)):
                raise ValueError("Invalid explicit transform columns")
    elif kind in ("native_manifest", "graph", "entity_links", "text_manifest", "catalog"):
        payload = read_json(path)
        blockers = _native_blockers(payload)
        if kind == "graph":
            if not payload.get("works") or not payload.get("edges"):
                blockers.append("graph_has_no_supported_order")
        if kind == "text_manifest":
            if payload.get("state") != "completed" or not payload.get("encoder", {}).get("production_encoder"):
                blockers.append("text_not_completed_production_encoder")
            if payload.get("vocab_size") != vocab or payload.get("source_pins", {}).get("metadata", {}).get("sha256") != metadata_sha:
                raise ValueError("Text vocabulary/metadata source mismatch")
        return blockers
    elif kind != "file":
        raise ValueError("Unknown artifact kind")
    return []


def _payload(root, name, evidence, *, metadata_sha=None, row_hashes=None, prepared=None):
    """Bind emitted declarations to the exact reviewed artifact schema/bytes."""
    allowed = {
        "local_base": (), "local_dynamic": ("runtime",), "semantic_extensions": ("item_extensions",),
        "text3fields": ("text_caches",), "people_embeddings": ("people_catalog",),
        "personal_candidates": ("runtime",), "audience": ("runtime",), "people_dynamics": ("runtime",),
        "franchise_progress": ("runtime",), "external_facts": ("date_features",),
        "external_interest": ("date_features",), "entity_links": ("entity_links",),
        "candidate_model": (), "related_interest": ("runtime",), "audience_factors": ("runtime",),
        "professions": ("item_extensions",), "geography_time": ("item_extensions",),
        "sports_settings": ("item_extensions",), "story_format": ("item_extensions", "item_replacements"),
        "completion_diagnostics": ("diagnostics",),
        "user_diagnostics": ("diagnostics",), "narrative_relations": ("item_extensions", "item_replacements"),
    }
    payload = evidence.get("provider_payload", {})
    if set(payload) - set(allowed[name]):
        raise ValueError("Component provider payload has an unrelated branch")
    artifacts = {checked_path(root, a): a for a in evidence["artifacts"]}
    sources = {checked_path(root, a): a for a in evidence["source_pins"]}
    def refs(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                path = checked_path(root, value)
                known = artifacts.get(path, sources.get(path))
                if known is None or known["sha256"] != value["sha256"]:
                    raise ValueError("Provider payload references an unreviewed artifact/source")
            for child in value.values():
                refs(child)
        elif isinstance(value, list):
            for child in value:
                refs(child)
    refs(payload)
    errors = []
    runtime_components = {"local_dynamic", "personal_candidates", "audience", "people_dynamics", "franchise_progress", "related_interest", "audience_factors"}
    if name in runtime_components:
        runtime = payload.get("runtime", {})
        if not runtime.get(name) or not runtime.get("raw_dir") or not runtime.get("source_pins"):
            errors.append("required_runtime_materialization_missing")
        if name == "franchise_progress":
            graph = runtime.get(name, {})
            kinds = graph.get("order_kinds") if isinstance(graph, dict) else None
            if (not isinstance(kinds, list) or any(not isinstance(kind, str) for kind in kinds) or len(kinds) != 2
                    or set(kinds) != {"mainline_release", "catalog_release_precedes"}
                    or "order_kind" in graph or "scope_policy" in graph):
                errors.append("required_two_typed_franchise_branches_missing")
            policies = graph.get("scope_policies") if isinstance(graph, dict) else None
            if policies is not None and (not isinstance(policies, dict) or set(policies) != set(kinds or [])
                    or any(policy is not None and not isinstance(policy, dict) for policy in policies.values())):
                raise ValueError("Franchise scope policies must bind each order kind separately")
            if not graph.get("graph"):
                errors.append("required_typed_franchise_graph_payload_missing")
            else:
                graph_path = checked_path(root, graph["graph"])
                if graph_path not in artifacts or artifacts[graph_path].get("kind") != "graph":
                    raise ValueError("Franchise graph payload must be the exact reviewed typed graph artifact")
                native_graph = read_json(graph_path)
                for kind, policy in (policies or {}).items():
                    for work, scope_id in (policy or {}).items():
                        scope = native_graph.get("scopes", {}).get(scope_id) if isinstance(scope_id, str) else None
                        if (work not in native_graph.get("works", {}) or not isinstance(scope, dict)
                                or work not in scope.get("work_ids", []) or scope.get("order_kind") != kind):
                            raise ValueError("Franchise scope policy must bind an existing work and a containing scope of the same order kind")
    if name in ("semantic_extensions", "professions", "geography_time", "sports_settings") and not payload.get("item_extensions"):
        errors.append("required_static_matrix_materialization_missing")
    if name in ("story_format", "narrative_relations") and not payload.get("item_replacements"):
        errors.append("required_named_replacement_materialization_missing")
    if name == "text3fields" and set(payload.get("text_caches", {})) != {"title", "title_orig", "description"}:
        errors.append("required_three_text_fields_missing")
    if name == "people_embeddings" and not payload.get("people_catalog"):
        errors.append("required_people_catalog_missing")
    if name == "entity_links" and not payload.get("entity_links"):
        errors.append("required_entity_export_missing")
    if name in ("external_facts", "external_interest") and not payload.get("date_features"):
        errors.append("required_external_date_features_missing")
    if name in ("completion_diagnostics", "user_diagnostics") and not payload.get("diagnostics"):
        errors.append("required_completion_diagnostic_corpus_missing")
    required_kinds = {"local_base": "native_manifest", "candidate_model": "native_manifest",
                      "people_embeddings": "catalog", "entity_links": "entity_links", "text3fields": "text_manifest", "completion_diagnostics": "native_manifest", "user_diagnostics": "native_manifest",
                      "franchise_progress": "graph"}
    if name in required_kinds and not any(a.get("kind") == required_kinds[name] for a in artifacts.values()):
        errors.append("required_typed_artifact_missing")
    if name in ("completion_diagnostics", "user_diagnostics"):
        for receipt in payload.get("diagnostics", {}).values():
            path = checked_path(root, receipt)
            if path not in artifacts or artifacts[path].get("kind") != "native_manifest":
                raise ValueError("Diagnostic payload is not its reviewed typed artifact")
            diagnostic = read_json(path)
            if (diagnostic.get("metadata_sha256") != metadata_sha or diagnostic.get("row_keys_sha256") != row_hashes
                    or diagnostic.get("fit_cutoff") != "2021-08-09"):
                errors.append("diagnostic_metadata_row_clock_binding_incomplete")
            if name == "user_diagnostics":
                expected_families = {"personal.activity", "personal.typed_completion", "personal.window_completion", "personal.age_rating_profile"}
                if (diagnostic.get("contract") != "kion-user-diagnostics-v1" or diagnostic.get("coverage") != "full_rows"
                        or diagnostic.get("clock") != "strict_h63_prior"
                        or diagnostic.get("storage") != "normalized_event_facts_exact_query_mapping"
                        or set(diagnostic.get("families", [])) != expected_families):
                    errors.append("user_diagnostic_full_four_family_corpus_missing")
            elif diagnostic.get("contract") == "kion-completion-event-store-v1":
                coverage = diagnostic.get("coverage", {})
                if (coverage.get("raw_events") != "complete_before_role_cap"
                        or coverage.get("query_mapping") != "complete_all_prepared_rows"):
                    errors.append("diagnostic_corpus_or_query_join_incomplete")
            elif diagnostic.get("contract") == "kion-completion-diagnostics-v1":
                if diagnostic.get("coverage") != "full_rows" or diagnostic.get("clock") != "strict_h63_prior_loo":
                    errors.append("diagnostic_full_rows_missing")
            else:
                errors.append("diagnostic_typed_contract_missing")
            native_prepared = diagnostic.get("prepared_pins", {})
            if diagnostic.get("contract") != "kion-completion-diagnostics-v1":
                for key, source in (prepared or {}).items():
                    if key not in ("train.npz", "val.npz"):
                        continue
                    recorded = native_prepared.get(key, {})
                    path_value = Path(recorded.get("path", "."))
                    recorded_path = (path_value if path_value.is_absolute() else root / path_value).resolve()
                    if recorded.get("sha256") != source["sha256"] or recorded_path != checked_path(root, source):
                        errors.append("diagnostic_prepared_source_binding_incomplete")
                        break
    for field in ("item_extensions", "item_replacements"):
        for matrix in payload.get(field, []):
            if matrix.get("family") != name:
                raise ValueError("Static payload requires an explicit matching typed family")
            artifact = artifacts.get(checked_path(root, matrix))
            if artifact is None or artifact.get("kind") != "matrix":
                raise ValueError("Provider matrix requires its reviewed matrix artifact")
            for key in ("columns", "shape", "dtype", "metadata_sha256", "vocabulary_sha256", "transform", "transform_columns"):
                if matrix.get(key) != artifact.get(key):
                    raise ValueError("Provider matrix schema differs from the reviewed artifact: " + key)
    for field, cache in payload.get("text_caches", {}).items():
        directory = Path(cache["path"])
        directory = directory if directory.is_absolute() else root / directory
        artifact = artifacts.get((directory / "manifest.json").resolve())
        if (field not in ("title", "title_orig", "description") or artifact is None
                or artifact.get("kind") != "text_manifest" or artifact["sha256"] != cache["manifest_sha256"]
                or read_json(directory / "manifest.json").get("field") != field):
            raise ValueError("Provider text field is not bound to its reviewed native manifest")
    for declaration in payload.get("date_features", []):
        if (declaration.get("family") != name or declaration.get("clock") != "strict_prefix"
                or declaration.get("fit_cutoff") != "2021-08-09" or not declaration.get("banks")):
            raise ValueError("External date feature family/clock/materialization mismatch")
        _names(declaration.get("columns"))
        for day, bank in declaration["banks"].items():
            if (not day.isdigit() or str(int(day)) != day or not 1 <= int(day) <= date(2021, 8, 9).toordinal()
                    or bank.get("cutoff_exclusive") != int(day) or bank.get("columns") != declaration["columns"]):
                raise ValueError("External bank strict cutoff/columns mismatch")
            artifact = artifacts.get(checked_path(root, bank))
            if artifact is None or artifact.get("kind") != "matrix":
                raise ValueError("External bank requires its reviewed named matrix")
            for key in ("columns", "shape", "dtype", "metadata_sha256", "vocabulary_sha256", "transform", "transform_columns", "cutoff_exclusive", "numeric_units"):
                if bank.get(key) != artifact.get(key):
                    raise ValueError("External bank schema differs from the reviewed matrix: " + key)
            if bank.get("numeric_units") != declaration.get("numeric_units"):
                raise ValueError("External bank numeric units differ from family declaration")
        if not declaration.get("numeric_units") or set(declaration["numeric_units"]) != set(declaration["columns"]):
            errors.append("external_numeric_units_incomplete")
    def absolute(value, key=None):
        if isinstance(value, dict):
            return {k: absolute(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [absolute(v) for v in value]
        if key in ("path", "raw_dir", "index_dir") and isinstance(value, str):
            path = Path(value)
            return (path if path.is_absolute() else root / path).resolve().as_posix()
        return value
    return absolute(payload), errors


class AdmissionError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Full assembly blocked: " + "; ".join(report["blockers"][:12]))


def assess(plan, *, root=None):
    """Validate explicit evidence and return a bounded readiness report.

    Missing/pilot evidence is a reported blocker. Corrupt source/schema bindings
    are errors in every mode and cannot be waived by partial debug.
    """
    root = Path(root or ".").resolve()
    if plan.get("contract") != CONTRACT:
        raise ValueError("Unknown assembly plan")
    provider_path = checked_path(root, plan["provider_source"])
    provider = provider_contract(provider_path)
    census = read_json(checked_path(root, plan["census"]))
    ids = [row["id"] for row in census["families"]]
    if len(ids) != 82 or len(ids) != len(set(ids)) or set(ids) != set(AGREEMENT_COMPONENTS):
        raise ValueError("Pinned census differs from the complete 82-ID inventory")
    metadata_path = checked_path(root, plan["metadata"])
    metadata = read_json(metadata_path)
    mapping = metadata["item_id_to_token"]
    if (metadata.get("context") != 64 or metadata.get("special_tokens") != {"PAD": 0, "BOS": 1, "UNK": 2}
            or len(mapping) != metadata["vocab_size"] - 3 or sorted(mapping.values()) != list(range(3, metadata["vocab_size"]))):
        raise ValueError("Prepared metadata requires contiguous frozen BOS+H63 vocabulary")
    metadata_sha, mapping_sha = plan["metadata"]["sha256"], canonical_hash(mapping)
    base = read_json(checked_path(root, plan["base_manifest"]))
    if base["inputs"]["metadata.json"]["sha256"] != metadata_sha:
        raise ValueError("Base metadata receipt differs from the assembly metadata")
    population = len(mapping)
    blockers, components, evidence_by_id = [], {}, {}
    pending_replacements = []
    prepared_counts = None
    for name, spec in plan.get("components", {}).items():
        if name not in {c for values in AGREEMENT_COMPONENTS.values() for c in values}:
            raise ValueError("Unknown component ID: " + name)
        evidence_path = checked_path(root, spec["evidence"])
        evidence = read_json(evidence_path)
        review = read_json(checked_path(root, spec["review"]))
        if evidence.get("contract") != EVIDENCE_CONTRACT or evidence.get("component") != name:
            raise ValueError("Component evidence contract/identity mismatch")
        if (review.get("contract") != REVIEW_CONTRACT or review.get("component") != name
                or review.get("evidence_sha256") != spec["evidence"]["sha256"]
                or review.get("verdict") != "PASS" or not review.get("reviewer")
                or review.get("reviewer") == evidence.get("author")):
            raise ValueError("Independent component review does not bind the exact evidence")
        if evidence.get("metadata_sha256") != metadata_sha or evidence.get("vocabulary_sha256") != mapping_sha:
            raise ValueError("Component metadata/vocabulary binding mismatch")
        if not evidence.get("source_pins") or not evidence.get("artifacts"):
            raise ValueError("Component requires actual sources and materialized artifacts")
        for source in evidence["source_pins"]:
            checked_path(root, source)
        errors = []
        if evidence.get("state") != "completed" or evidence.get("scope") != "full":
            errors.append("component_not_completed_full")
        if evidence.get("source_kind") != "actual_data":
            errors.append("fixture_or_non_actual_source")
        if name not in provider["COMPONENT_ADAPTERS"]:
            errors.append("provider_adapter_missing")
        checks = evidence.get("checks", {})
        if not all(checks.get(key) is True for key in ("tests_passed", "schema_validated", "clock_validated", "numerical_parity")):
            errors.append("component_checks_incomplete")
        temporal = evidence.get("temporal", {})
        if (temporal.get("mode") not in ("static_retrospective", "strict_prefix")
                or temporal.get("role") != "sft" or temporal.get("fit_cutoff") != "2021-08-09"
                or temporal.get("validation_origin") != "2021-08-09"
                or temporal.get("whole_day_exclusive") is not True):
            errors.append("temporal_role_contract_incomplete")
        if name in ("local_dynamic", "personal_candidates", "audience", "people_dynamics", "franchise_progress", "related_interest", "external_facts", "external_interest") and temporal.get("mode") != "strict_prefix":
            errors.append("dynamic_component_requires_strict_prefix")
        for artifact in evidence["artifacts"]:
            errors.extend(_artifact(root, artifact, metadata_sha, mapping_sha, metadata["vocab_size"]))
        payload, payload_errors = _payload(root, name, evidence, metadata_sha=metadata_sha,
            row_hashes=base["row_keys_sha256"], prepared=base["inputs"])
        errors.extend(payload_errors)
        for declaration in payload.get("date_features", []):
            if set(declaration.get("numeric_units", {}).values()) - set(provider.get("NUMERIC_UNITS", {})):
                raise ValueError("External numeric units are unsupported by the pinned provider")
        for replacement in payload.get("item_replacements", []):
            pending_replacements.append(replacement)
            if "static_replacements" not in provider["COMPONENT_ADAPTERS"]:
                errors.append("provider_adapter_missing:static_replacements")
        claimed_ids = evidence.get("agreements", {})
        for agreement, row in claimed_ids.items():
            if agreement not in AGREEMENT_COMPONENTS or name not in AGREEMENT_COMPONENTS[agreement]:
                raise ValueError("Component claims an unrelated agreement ID")
            local_errors = list(errors)
            coverage = row.get("coverage", {})
            if coverage_axis(agreement, name) == "prepared_queries" and prepared_counts is None:
                prepared_counts = _prepared_counts(root, base)
            local_errors.extend(_coverage(root, agreement, name, coverage, evidence, base, population, prepared_counts, plan["base_manifest"]))
            if agreement in SEMANTIC_IDS and row.get("agreed_taxonomy_complete") is not True:
                local_errors.append("agreed_taxonomy_incomplete")
            if not row.get("contract_clause_ids") or not row.get("mask_columns"):
                local_errors.append("contract_or_availability_masks_not_bound")
            if row.get("data_ready") is not True:
                local_errors.append("agreement_data_not_ready")
            evidence_by_id[(agreement, name)] = {"ready": not local_errors, "blockers": local_errors,
                "coverage": coverage, "trained": row.get("trained") is True,
                "coverage_axis": coverage_axis(agreement, name), "code_accepted": not errors,
                "full_computation": row.get("full_computation") is True,
                "trained_exposure": row.get("trained_exposure", "not_claimed"),
                "evidence": spec["evidence"], "review": spec["review"]}
        claim_errors = sorted({reason for key, record in evidence_by_id.items() if key[1] == name for reason in record["blockers"]})
        components[name] = {"ready": not claim_errors and not errors, "code_and_artifacts_ready": not errors,
                            "blockers": sorted(set(errors) | set(claim_errors)), "evidence": pin(evidence_path),
                            "review": pin(checked_path(root, spec["review"])), "payload": payload}
    extensions = [extension for component in components.values() for extension in component["payload"].get("item_extensions", [])]
    for replacement in pending_replacements:
        _replacement(replacement, plan["base_manifest"], base, extensions, root)
    rows = []
    for agreement, required in AGREEMENT_COMPONENTS.items():
        if agreement in DEFERRED_PROPOSALS:
            rows.append({"id": agreement, "required_for_full": False, "state": "deferred_proposal",
                         "reason": DEFERRED_PROPOSALS[agreement], "components": list(required)})
            continue
        errors = []
        for name in required:
            record = evidence_by_id.get((agreement, name))
            if record is None:
                errors.append(name + ":missing_agreement_evidence")
            else:
                errors.extend(name + ":" + error for error in record["blockers"])
        rows.append({"id": agreement, "required_for_full": True, "state": "data_ready" if not errors else "blocked",
            "components": list(required), "blockers": errors,
            "evidence": {name: evidence_by_id[(agreement, name)] for name in required if (agreement, name) in evidence_by_id}})
        blockers.extend(agreement + ":" + error for error in errors)
    report = {"contract": "kion-complete-assembly-assessment-v1", "status": "data_ready_untrained" if not blockers else "blocked",
        "inventory_count": len(rows), "required_count": sum(r["required_for_full"] for r in rows),
        "deferred_count": len(DEFERRED_PROPOSALS), "agreement_mapping_sha256": canonical_hash(AGREEMENT_COMPONENTS),
        "provider_source": plan["provider_source"], "census": plan["census"], "metadata": plan["metadata"], "base_manifest": plan["base_manifest"],
        "vocabulary_sha256": mapping_sha, "population_size": population, "prepared_split_counts": prepared_counts, "agreements": rows,
        "components": components, "blockers": sorted(set(blockers))}
    return report


def assess_scoped(plan, *, root=None):
    """Assess the human-approved ready package without changing the all81 gate."""
    root = Path(root or ".").resolve()
    decision = scoped_decision(plan.get("scope_decision"), root)
    if set(plan.get("components", {})) != set(SCOPED_COMPONENTS):
        raise ValueError("Scoped assembly requires exactly the retained ready components")
    retained = plan.get("retained_component_plan", {})
    if retained.get("sha256") != "07b0640dd6ad22b7d4069bba3bbfe54285f1bfd21fee57b62510486a02e1ae49":
        raise ValueError("Scoped retained contributions require the exact reviewed R2 plan")
    reviewed = read_json(checked_path(root, retained))
    if any(spec != reviewed.get("components", {}).get(name) for name, spec in plan["components"].items() if name != "external_interest"):
        raise ValueError("Scoped retained evidence/reviews differ from the approved component plan")
    personal_evidence = read_json(checked_path(root, plan["components"]["personal_candidates"]["evidence"]))
    annotations = [source for source in personal_evidence["source_pins"]
        if source.get("sha256") == "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"
        and Path(source["path"]).name == "kion-series-v1.json"]
    if len(annotations) != 1:
        raise ValueError("Scoped personal features require their exact reviewed series annotation")
    annotation = pin(checked_path(root, annotations[0]))
    historical_builder = plan.get("related_historical_builder", {})
    if historical_builder.get("sha256") != "98da6c6b939ff3dbe5c8464a826a775437d43fd5b6257efafab33caff34ec739":
        raise ValueError("Scoped related index requires its exact historical audience builder source")
    historical_builder = pin(checked_path(root, historical_builder))
    # The existing independently reviewed wrappers retain all their original
    # artifact, row, role and clause checks. Their all81 observation is separate.
    legacy = dict(plan, components={k: v for k, v in plan["components"].items() if k != "external_interest"})
    original = assess(legacy, root=root)
    components = original["components"]
    spec = plan["components"]["external_interest"]
    approved = {"evidence": "7bd7176ad9d4fe67e9f9520b7970587bfc9d8f290ccd90cb51cf5dea829986d5",
        "manifest": "decabacca0905ee270060c5b37cba939de5d48311fa3e6c3899e85381e2c9f42",
        "review": "238ad8931639c63b9b90d4a21b5a1ad12d2efe0fa0118a36c0fa24e227e66b34",
        "origins": "efed779cf47328d6742fbde09dcaa2c6e81adaa93dec03af91e5a9950321f53e"}
    if set(spec) != set(approved) or any(spec[k].get("sha256") != value for k, value in approved.items()):
        raise ValueError("Ready scope admits only the independently reviewed fixed403 bank release")
    paths = {key: checked_path(root, value) for key, value in spec.items()}
    declaration, native, review, origins = (read_json(paths[key]) for key in ("evidence", "manifest", "review", "origins"))
    provider = provider_contract(checked_path(root, plan["provider_source"]))
    metadata = read_json(checked_path(root, plan["metadata"]))
    columns = declaration.get("columns")
    _names(columns)
    if (declaration.get("family") != "external_interest" or declaration.get("clock") != "strict_prefix"
            or declaration.get("fit_cutoff") != "2021-08-09" or declaration.get("strict_asof_claim") is not False
            or declaration.get("readiness") != "partial_source_population" or len(columns) != 161
            or set(declaration.get("numeric_units", {})) != set(columns)
            or set(declaration["numeric_units"].values()) - set(provider["NUMERIC_UNITS"])
            or review.get("status") != "pass" or review.get("findings") != []
            or not review.get("checks") or not all(value is True for value in review["checks"].values())
            or review.get("bank_manifest") != spec["manifest"]
            or review.get("observed", {}).get("SFT_banks") != 149
            or review.get("observed", {}).get("eligible_unique_queries") != 1321
            or review.get("observed", {}).get("matrix_shape") != [15418, 161]
            or native.get("columns") != columns
            or set(declaration.get("banks", {})) != {str(value) for value in origins["origin_ordinals"]}
            or len(origins["origin_ordinals"]) != 149):
        raise ValueError("Fixed403 reviewed bank/schema/clock/scope binding mismatch")
    for ordinal, bank in declaration["banks"].items():
        if (bank.get("columns") != columns or bank.get("numeric_units") != declaration["numeric_units"]
                or bank.get("cutoff_exclusive") != int(ordinal)
                or not bank.get("provenance", {}).get("source_pins")):
            raise ValueError("Fixed403 bank lacks its exact origin/schema/source binding")
        _artifact(root, {**bank, "kind": "matrix", "transform": "identity"},
            plan["metadata"]["sha256"], canonical_hash(metadata["item_id_to_token"]), metadata["vocab_size"])
        for source in bank["provenance"]["source_pins"]:
            checked_path(root, source)
    components["external_interest"] = {"ready": True, "code_and_artifacts_ready": True, "blockers": [],
        "evidence": spec["evidence"], "review": spec["review"], "payload": {"date_features": [declaration]},
        "scope": "fixed403_attempted_snapshot_sparse_retrospective", "full_catalog_ready": False,
        "strict_asof_claim": False, "native_manifest": spec["manifest"], "origin_inventory": spec["origins"]}
    blockers = [name + ":" + reason for name, value in components.items() for reason in value["blockers"]]
    return {"contract": "kion-complete-scoped-assessment-v1", "status": "blocked" if blockers else "scoped_data_ready_untrained",
        "scope_id": decision["scope_id"], "scope_decision": plan["scope_decision"],
        "active_families": list(SCOPED_FAMILIES), "additional_ready_components": list(SCOPED_READY_COMPONENTS),
        "provider_source": plan["provider_source"], "metadata": plan["metadata"], "base_manifest": plan["base_manifest"],
        "components": components, "blockers": sorted(set(blockers)), "full_package_admitted": False,
        "runtime_metadata": {"series_annotation": annotation, "related_historical_builder": historical_builder},
        "original_all81_observation": {"required_count": original["required_count"],
            "ready_count": sum(r.get("state") == "data_ready" for r in original["agreements"]),
            "blockers": original["blockers"], "admitted": False}}


def compose_payload(report, provider, *, mode="full"):
    """Deterministically merge reviewed payload; also used by provider admission."""
    payload = {"item_extensions": [], "text_caches": {}, "people_catalog": None, "runtime": {}, "date_features": [], "entity_links": None}
    if "completion_diagnostics" in provider["COMPONENT_ADAPTERS"]:
        payload["diagnostics"] = {}
    if "static_replacements" in provider["COMPONENT_ADAPTERS"]:
        payload["item_replacements"] = []
    pending = {}
    for name in sorted(report["components"]):
        for key, value in report["components"][name]["payload"].items():
            if key not in payload:
                if mode == "partial" and key in ("item_replacements", "diagnostics"):
                    pending.setdefault(key, []).append(value)
                    continue
                raise ValueError("Unsupported provider payload field: " + key)
            if isinstance(payload[key], list):
                payload[key].extend(value)
            elif isinstance(payload[key], dict):
                for item, data in value.items():
                    if item in payload[key] and payload[key][item] != data:
                        raise ValueError("Conflicting provider payload keys: " + key)
                    payload[key][item] = data
            elif payload[key] is not None:
                raise ValueError("Duplicate provider payload branch: " + key)
            else:
                payload[key] = value
    if report.get("contract") == "kion-complete-scoped-assessment-v1":
        payload["runtime"]["series_annotation"] = report["runtime_metadata"]["series_annotation"]
        payload["runtime"]["related_interest"]["audience_builder_source"] = report["runtime_metadata"]["related_historical_builder"]
    return payload, pending


def assemble(plan, output, *, root=None, mode="full", allow_partial_debug=False):
    """Publish a new manifest only after assessing all required concrete IDs."""
    root, output = Path(root or ".").resolve(), Path(output).resolve()
    if mode not in ("full", "partial", "scoped_production") or mode == "partial" and not allow_partial_debug:
        raise ValueError("Partial debug needs explicit mode and allow_partial_debug=True")
    if output.exists():
        raise ValueError("A new output directory is required")
    scoped = mode == "scoped_production"
    report = assess_scoped(plan, root=root) if scoped else assess(plan, root=root)
    if mode != "partial" and report["blockers"]:
        raise AdmissionError(report)
    provider_path = checked_path(root, plan["provider_source"])
    provider = provider_contract(provider_path)
    base_path = checked_path(root, plan["base_manifest"])
    base = read_json(base_path)
    data_dir = Path(plan["data_dir"])
    data_dir = (data_dir if data_dir.is_absolute() else root / data_dir).resolve()
    files = {}
    for name in ("metadata.json", "train.npz", "val.npz"):
        receipt = base["inputs"][name]
        path = checked_path(root, receipt)
        if path != data_dir / name:
            raise ValueError("Base cache prepared input path mismatch")
        files[name] = pin(path)
    if files["metadata.json"]["sha256"] != plan["metadata"]["sha256"]:
        raise ValueError("Base cache metadata binding mismatch")
    payload, pending_payload = compose_payload(report, provider, mode=mode)
    families = {}
    required_families = list(SCOPED_FAMILIES) if scoped else list(provider["REQUIRED_FAMILIES"])
    for family in required_families:
        owned = [v for name, v in report["components"].items() if name == family or family == "semantic_extensions" and name in ("professions", "geography_time", "sports_settings")]
        ready = bool(owned) and all(v["ready"] for v in owned)
        families[family] = {"status": "accepted" if ready else "not_ready", "reviewed": ready,
            "scope": "approved_ready_scope" if scoped and ready else "full" if ready else "partial", "source_pins": [v[k] for v in owned for k in ("evidence", "review")]}
    source_pins = [pin(provider_path.with_name(name)) for name in provider["SOURCE_FILES"] if provider_path.with_name(name).is_file()]
    manifest = {"contract": provider["CONTRACT"], "state": "completed", "integration_mode": mode,
        "data": {"path": data_dir.as_posix(), "files": files}, "base_cache": {"path": base_path.parent.as_posix(), "manifest_sha256": plan["base_manifest"]["sha256"]},
        "row_keys_sha256": base["row_keys_sha256"], "rowkeys_algorithm": base["rowkeys_algorithm"],
        "families": families, "required_families": required_families, **payload,
        "source_pins": source_pins, "limits": plan.get("limits", {}), "numeric_transforms": provider["NUMERIC_TRANSFORMS"],
        "metadata_availability": "retrospective_catalog_snapshot_not_historical_asof",
        "outcome_availability": "last_watch_date_assumption_strict_before_whole_day",
        "fit_cutoff": "2021-08-09", "validation_origin": "2021-08-09",
        "assembly": {"status": "scoped_data_ready_untrained" if scoped else "cannot_train_full" if mode == "partial" else "data_ready_untrained",
            "assessment_sha256": canonical_hash(report), "agreement_mapping_sha256": report.get("agreement_mapping_sha256"),
            "inventory_count": 82, "deferred_proposals": DEFERRED_PROPOSALS,
            "pending_provider_payload": pending_payload}}
    if scoped:
        manifest.update(scope_id=report["scope_id"], scope_decision=plan["scope_decision"], active_families=list(SCOPED_FAMILIES),
            additional_ready_components=list(SCOPED_READY_COMPONENTS), status="admitted_scoped", full_row_fit_admitted=True, full_package_admitted=False)
    manifest["signature"] = canonical_hash(manifest)
    # Detect evidence changes immediately before publication. All writes occur
    # only after validation, to a fresh directory owned by this operation.
    checked_path(root, plan["provider_source"])
    checked_path(root, plan["base_manifest"])
    for component in plan.get("components", {}).values():
        checked_path(root, component["evidence"])
        checked_path(root, component["review"])
    if scoped:
        scoped_decision(plan["scope_decision"], root)
    output.mkdir(parents=True)
    write_json(output / "assessment.json", report)
    write_json(output / "manifest.json", manifest)
    return manifest, report


def estimate_cost(vocab=15418, people_width=3357, days=149, *, candidate_pairs=64_000_000):
    """Planning arithmetic, with measured rows and assumptions kept distinct."""
    per_day = vocab * people_width * 4
    return {"assumptions": {"vocab_with_specials": vocab, "people_width": people_width, "days": days,
            "dtype": "float32", "candidate_pairs": candidate_pairs},
        "people_bank_bytes_per_day": per_day, "people_all_days_bytes": per_day * days,
        "people_full_lookup_latency": "not measured; the 44-token source-order pilot is not a full-index extrapolation",
        "audience_measured_baseline_pairs_per_second": [2048 / 3.3, 2048 / 2.3],
        "audience_measured_fast_gram_pairs_per_second": 8192 / .674,
        "audience_fast_gram_warm_compute_hours_assuming_measured_rate": candidate_pairs / (8192 / .674) / 3600,
        "audience_snapshot_seconds_per_date_measured": [7.7, 10.5],
        "strategy": "sparse prefixes once; date-grouped batches; bounded item/date token cache; sampled/topK pair head; no BxVxD",
        "warning": "planning arithmetic is not full-fit throughput; checkpoint backward replays candidate callbacks; benchmark that replay and cache behavior"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial-debug", action="store_true")
    args = parser.parse_args(argv)
    plan = read_json(args.plan)
    if args.output:
        _, report = assemble(plan, args.output, root=args.plan.parent,
            mode="partial" if args.partial_debug else "full", allow_partial_debug=args.partial_debug)
    else:
        report = assess(plan, root=args.plan.parent)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "data_ready_untrained" or args.partial_debug else 2


if __name__ == "__main__":
    raise SystemExit(main())
