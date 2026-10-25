"""Scoped related/franchise I/O ports; importing this module is stdlib-only.

Backends must be admitted retained modules supplied by the source loader. No
audience builder/runtime is imported or activated. All identities stay logical.
Successful related indexes own their mmaps and expose the retained close().
"""
import copy
from datetime import date
from pathlib import PureWindowsPath

from gpt_provider import canonical_hash, logical_child, _close_array

HISTORICAL_BUILDER_SHA256 = "98da6c6b939ff3dbe5c8464a826a775437d43fd5b6257efafab33caff34ec739"
FRANCHISE_ORDER_KINDS = ("mainline_release", "catalog_release_precedes")
ARRAYS = ("users", "tokens", "days", "token_work")


def related_historical_builder(io, root, recorded, receipt):
    """Retain the sole accepted historical audience-source alias, without import."""
    if (not isinstance(receipt, dict) or recorded.get("sha256") != HISTORICAL_BUILDER_SHA256
            or receipt.get("sha256") != HISTORICAL_BUILDER_SHA256
            or PureWindowsPath(recorded.get("path", "")).name != "kion_audience_features.py"):
        raise ValueError("Related historical audience builder requires an explicit exact source alias")
    io.pin(root, receipt, "source")
    return {"recorded_builder": copy.deepcopy(recorded), "historical_source": copy.deepcopy(receipt)}


def load_related_index(io, directory, *, manifest_sha256, np, related):
    """Port RelatedInterestIndex.load; retained pure methods remain on the object.

    ``related`` supplies RelatedInterestIndex, _metadata, _day, ROLE_CAPS, KINDS.
    Admission of its original source and dependency closure is caller-owned.
    """
    manifest = io.read_json(logical_child(directory, "manifest.json"), expected=manifest_sha256)
    if manifest.get("contract") != "kion-related-interest-v1" or manifest.get("state") != "completed":
        raise ValueError("Unknown or incomplete related index")
    if manifest.get("state_sha256") != canonical_hash({k: v for k, v in manifest.items() if k != "state_sha256"}):
        raise ValueError("Related persisted state hash mismatch")
    obj = related.RelatedInterestIndex.__new__(related.RelatedInterestIndex)
    opened = []
    try:
        for key in ("mapping", "vocab_size", "fit_cutoff", "role", "temporal_mode", "release_scope",
                    "parameters", "works", "entities", "portfolios", "releases", "provenance"):
            setattr(obj, key, manifest[key])
        obj.item_entities = manifest.get("item_entities")
        related._metadata(dict(item_id_to_token=obj.mapping, vocab_size=obj.vocab_size))
        if obj.role not in related.ROLE_CAPS or not 1 <= obj.fit_cutoff <= related._day(related.ROLE_CAPS[obj.role]):
            raise ValueError("Invalid persisted role cap")
        if obj.temporal_mode != "retrospective_reconstruction" or not all(
                np.isfinite(obj.parameters[x]) and obj.parameters[x] > 0 for x in ("alpha", "shrinkage", "growth_clip")):
            raise ValueError("Invalid persisted temporal mode/parameters")
        if any(isinstance(obj.parameters[x], bool) or not isinstance(obj.parameters[x], int) or obj.parameters[x] < 1
               for x in ("min_pairs", "min_viewers", "cache_days")):
            raise ValueError("Invalid persisted support/cache limits")
        if set(manifest["arrays"]) != set(ARRAYS):
            raise ValueError("Related arrays require the exact persisted census")
        for name, pin in manifest["arrays"].items():
            if PureWindowsPath(pin["path"]).name != pin["path"]:
                raise ValueError("Unsafe persisted array path")
            path = io.pin(directory, pin, "payload")
            io.check()
            value = np.load(path, mmap_mode="r", allow_pickle=False)
            opened.append(value)
            setattr(obj, name, value)
        obj._columns()
        if obj.columns != manifest["columns"] or canonical_hash(obj.columns) != obj.provenance["schema_sha256"]:
            raise ValueError("Related schema mismatch")
        io.check()
        obj._finish_arrays()
        if "item_entities" not in manifest:
            obj.item_entities = {kind: [[] if t < 3 else obj.entities[kind][int(obj.token_work[t])]
                for t in range(obj.vocab_size)] for kind in related.KINDS}
        return obj
    except BaseException:
        for value in opened:
            _close_array(value)
        raise


def load_scoped_related(io, root, config, metadata, *, raw_root, data_root, np, related):
    """Scoped production portion of _init_runtime; returns (index, provenance).

    This has no partial/full admission switch. The composer must already have
    admitted scoped production and the related module's source identity.
    """
    params = dict(config)
    directory = logical_child(root, params.pop("index_dir"))
    receipt = params.pop("manifest")
    historical = params.pop("audience_builder_source", None)
    if params or logical_child(root, receipt["path"]) != logical_child(directory, "manifest.json"):
        raise ValueError("Related index manifest directory binding mismatch")
    obj = load_related_index(io, directory, manifest_sha256=receipt["sha256"], np=np, related=related)
    try:
        if (obj.mapping != metadata["item_id_to_token"] or obj.fit_cutoff != date(2021, 8, 9).toordinal()
                or obj.role != "sft"):
            raise ValueError("Related index mapping/typed SFT role cap mismatch")
        if obj.provenance.get("pilot_only") is not False or obj.provenance.get("source_scope") != "full_source":
            raise ValueError("Full package rejects a related-interest source pilot")
        provenance = {"related_interest": obj.provenance}
        for collection in ("source_pins", "audience_source_pins"):
            for name, source in obj.provenance.get(collection, {}).items():
                if collection == "audience_source_pins" and name == "builder" and historical is not None:
                    provenance["related_historical_builder"] = related_historical_builder(io, root, source, historical)
                else:
                    logical = logical_child(root, source["path"])
                    io.pin(root, source, "source" if name == "builder" else io.bindings[logical].role)
        for collection, name, actual in (
                ("source_pins", "items", logical_child(raw_root, "items.csv")),
                ("audience_source_pins", "items", logical_child(raw_root, "items.csv")),
                ("audience_source_pins", "interactions", logical_child(raw_root, "interactions.csv")),
                ("audience_source_pins", "metadata", logical_child(data_root, "metadata.json"))):
            source = obj.provenance.get(collection, {}).get(name)
            if source is None:
                raise ValueError("Related runtime corpus source receipt missing: " + name)
            if logical_child(root, source["path"]) != actual or source["sha256"] != io.bindings[actual].sha256:
                raise ValueError("Related index belongs to a different runtime corpus: " + name)
            io.verify(actual, "metadata" if name == "metadata" else "payload", source["sha256"])
        return obj, provenance
    except BaseException:
        obj.close()
        raise


def load_scoped_franchise(io, root, params, metadata, *, franchise):
    """Read the exact graph and invoke admitted retained prepare_graph unchanged.

    Returns prepared_graph, temporal_mode, columns, runtime_provenance. Accepted
    graph/evidence/source admission is retained in the upstream scoped decision;
    this seam enforces its graph pin and the retained graph/schema validator.
    """
    if set(params) - {"graph", "temporal_mode", "order_kinds", "scope_policies", "order_kind", "scope_policy"}:
        raise ValueError("Unknown franchise progress options")
    pin = params["graph"]
    graph = io.read_json(logical_child(root, pin["path"]), expected=pin["sha256"])
    mode = params.get("temporal_mode")
    if mode not in ("historical_as_of", "retrospective_current"):
        raise ValueError("Explicit franchise temporal mode required")
    if "order_kinds" not in params:
        raise ValueError("Full franchise progress requires both explicit typed order_kinds")
    kinds = params["order_kinds"]
    if (not isinstance(kinds, list) or not kinds or any(not isinstance(k, str) for k in kinds)
            or len(set(kinds)) != len(kinds) or set(kinds) != set(FRANCHISE_ORDER_KINDS)
            or "order_kind" in params or "scope_policy" in params):
        raise ValueError("Franchise order_kinds must be distinct supported typed branches")
    policies = params.get("scope_policies", {kind: None for kind in kinds})
    if (not isinstance(policies, dict) or set(policies) != set(kinds)
            or any(p is not None and not isinstance(p, dict) for p in policies.values())):
        raise ValueError("Franchise scope_policies must bind each typed branch separately")
    for kind, policy in policies.items():
        for work, scope_id in (policy or {}).items():
            scope = graph["scopes"].get(scope_id) if isinstance(scope_id, str) else None
            if (work not in graph["works"] or not isinstance(scope, dict) or work not in scope["work_ids"]
                    or scope["order_kind"] != kind):
                raise ValueError("Franchise scope policy must bind an existing work and a containing scope of the same order kind")
    io.check()
    prepared = franchise.prepare_graph(graph, metadata, order_kinds=kinds, scope_policies=policies)
    prefixes = ["franchise." + kind + "." for kind in kinds]
    columns = [prefix + col for prefix in prefixes for col in franchise.COLUMNS]
    provenance = {**copy.deepcopy(params), "order_kinds": list(kinds), "scope_policies": copy.deepcopy(policies),
                  "feature_namespaces": prefixes, "typed_orders_merged": False}
    return prepared, mode, columns, provenance
