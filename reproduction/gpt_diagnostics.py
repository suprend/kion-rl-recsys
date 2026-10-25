"""Portable diagnostics loaders. Stdlib-only import; callers admit pure runtime backends.

All original dictionaries retain logical identities. Explicit loader calls perform
bound hash validation and numerical reads; import/planning never opens payloads.
Successful returned stores own their mmaps; failed calls close all opened mmaps.
"""
from pathlib import PureWindowsPath
from gpt_provider import canonical_hash, logical_child, _close_array


def _verify_pin(io, root, pin):
    logical = logical_child(root, pin["path"])
    io.verify(logical, io.bindings[logical].role, pin["sha256"])
    return True


def _local_payload(io, root, pin):
    logical = logical_child(root, pin["path"])
    if PureWindowsPath(logical).parent != PureWindowsPath(root):
        raise ValueError("Diagnostic payload must be directly inside its logical root")
    return io.verify(logical, "payload", pin["sha256"])


def _manifest_root(root, pin):
    logical = logical_child(root, pin["path"])
    if PureWindowsPath(logical).name != "manifest.json":
        raise ValueError("Diagnostic receipt must pin the actually opened native manifest.json")
    return PureWindowsPath(logical).parent.as_posix()


def load_threshold_facts(io, root, store, *, np, diagnostics):
    manifest = io.read_json(logical_child(root, "manifest.json"), signed=True)
    event_manifest = logical_child(store.logical_root, "manifest.json")
    if (manifest.get("contract") != diagnostics.THRESHOLD_CONTRACT
            or manifest.get("state") != "completed"
            or manifest.get("events_count") != len(store.keys)
            or manifest.get("event_store") != {"path": event_manifest,
                "sha256": io.bindings[event_manifest].sha256}
            or manifest.get("fit_cutoff") != store.manifest["fit_cutoff"]):
        raise ValueError("Threshold event-store binding/signature mismatch")
    for pin in manifest["source_pins"].values():
        _verify_pin(io, root, pin)
    result = {}
    try:
        for name, dtype in (("threshold_bits", "uint8"), ("source_csv_data_row", "uint32")):
            target = _local_payload(io, root, manifest["arrays"][name])
            values = np.load(target, mmap_mode="r", allow_pickle=False)
            result[name] = values
            if values.shape != (len(store.keys),) or str(values.dtype) != dtype:
                raise ValueError("Threshold payload shape/dtype mismatch")
        if np.any(result["threshold_bits"] > 31) or np.any(result["source_csv_data_row"] < 1):
            raise ValueError("Threshold payload range mismatch")
        return result, manifest
    except BaseException:
        for values in result.values():
            _close_array(values)
        raise


def _user_from_paths(io, root, store, threshold_pin, *, np, diagnostics):
    facts, manifest = load_threshold_facts(io, _manifest_root(root, threshold_pin),
                                          store, np=np, diagnostics=diagnostics)
    try:
        metadata = io.read_json(logical_child(root, store.manifest["metadata_path"]),
                                expected=store.manifest["metadata_sha256"])
        item_pin = store.manifest["source_pins"]["items"]
        _verify_pin(io, root, item_pin)
        item_path = io.resolve(logical_child(root, item_pin["path"]), "payload")
        history, cards = diagnostics.history, {}
        for iid, kind, rating in history._csv_rows(item_path, ("item_id", "content_type", "age_rating")):
            io.check()
            iid = history._mapping_int(iid, "item metadata ID")
            if iid in cards:
                raise ValueError("Duplicate metadata card")
            cards[iid] = {"content_type": kind, "age_rating": rating}
        instance = diagnostics.UserDiagnostics(store, facts["threshold_bits"], metadata,
                                               cards, threshold_manifest=manifest)
        instance.threshold_facts = facts
        return instance
    except BaseException:
        for values in facts.values():
            _close_array(values)
        raise


def load_event_store(io, path, *, np, events, expected_metadata_sha256=None, expected_prepared_pins=None,
         expected_row_keys_sha256=None, verify_hashes=True, allow_probe=False):
    """Production rejects partial/probe coverage before allocating an index."""
    root = path
    history, diagnostics = events.history, events.diagnostics
    CONTRACT, EVENT_DTYPES, FLAG_BITS = events.CONTRACT, events.EVENT_DTYPES, events.FLAG_BITS
    QUERY_JOIN, FULL_QA = events.QUERY_JOIN, events.FULL_QA
    manifest = io.read_json(logical_child(root, "manifest.json"), signed=True)
    if manifest.get("contract") != CONTRACT or canonical_hash({k: v for k, v in manifest.items() if k != "signature"}) != manifest.get("signature"):
        raise ValueError("Event-store manifest signature/contract mismatch")
    if allow_probe:
        raise ValueError("Probe diagnostics are unsupported by this scoped adapter")
    full = (manifest.get("state") == "completed" and manifest.get("scope") == "full"
        and manifest.get("coverage", {}).get("raw_events") == "complete_before_role_cap"
        and manifest.get("coverage", {}).get("query_mapping") == "complete_all_prepared_rows")
    if not full and not allow_probe:
        raise ValueError("Full diagnostics require complete raw events and all prepared query mappings")
    if verify_hashes is not True:
        raise ValueError("Event-store source/payload hash validation cannot be disabled")
    if expected_metadata_sha256 is not None and expected_metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("Event-store metadata binding mismatch")
    if expected_prepared_pins is not None:
        for name, pin in expected_prepared_pins.items():
            actual = manifest["prepared_pins"].get(name)
            if (actual is None or actual["sha256"] != pin["sha256"]
                    or logical_child(root, actual["path"]) != logical_child(root, pin["path"])):
                raise ValueError("Event-store prepared file binding mismatch")
    if expected_row_keys_sha256 is not None and expected_row_keys_sha256 != manifest["row_keys_sha256"]:
        raise ValueError("Event-store prepared row-key binding mismatch")
    for pin in manifest["source_pins"].values():
        if not _verify_pin(io, root, pin):
            raise ValueError("Event-store source SHA mismatch")
    if not io.verify(logical_child(root, manifest["metadata_path"]), "metadata", manifest["metadata_sha256"]):
        raise ValueError("Event-store metadata source SHA mismatch")
    if set(manifest["arrays"]) != set(EVENT_DTYPES) or manifest["flag_bits"] != FLAG_BITS:
        raise ValueError("Event-store column/flag schema mismatch")
    arrays = {}
    try:
        for name, dtype in EVENT_DTYPES.items():
            declaration = manifest["arrays"][name]
            target = _local_payload(io, root, declaration)
            if target is None:
                raise ValueError("Event-store payload path/hash mismatch")
            values = np.load(target, mmap_mode="r", allow_pickle=False)
            arrays[name] = values
            if values.shape != (manifest["events_count"],) or str(values.dtype) != dtype or declaration["shape"] != list(values.shape):
                raise ValueError("Event-store payload shape/dtype mismatch")
        for start in range(0, manifest["events_count"], 100000):
            io.check()
            end = min(start + 100000, manifest["events_count"])
            if any(not np.isfinite(value[start:end]).all() for value in arrays.values()):
                raise ValueError("Nonfinite event facts")
            if (np.any(arrays["date"][start:end] >= history._day(manifest["fit_cutoff"]))
                    or not np.isin(arrays["parser_status"][start:end], (0, 1, 2)).all()):
                raise ValueError("Invalid event clock/parser status")
            if (any(np.any(arrays[name][start:end] < 0) for name in
                    ("raw_row_index", "user_id", "raw_item_id", "n_item", "n_type", "n_global"))
                    or np.any(arrays["date"][start:end] < 1) or np.any(arrays["token"][start:end] < 2)
                    or np.any((arrays["outcome"][start:end] < 0) | (arrays["outcome"][start:end] > 1))
                    or np.any((arrays["baseline"][start:end] < 0) | (arrays["baseline"][start:end] > 1))
                    or np.any(np.abs(arrays["residual"][start:end]) > 1)):
                raise ValueError("Invalid event identity/support/value range")
        if len(arrays["raw_row_index"]) > 1 and np.any(np.diff(arrays["raw_row_index"]) <= 0):
            raise ValueError("Event raw-row identities must increase uniquely")
        # Native source ordering is user/date/raw-item, including tied days.
        if len(arrays["user_id"]) > 1:
            u, d, i = arrays["user_id"], arrays["date"], arrays["raw_item_id"]
            if np.any((u[1:] < u[:-1]) | ((u[1:] == u[:-1]) & ((d[1:] < d[:-1])
                    | ((d[1:] == d[:-1]) & (i[1:] <= i[:-1]))))):
                raise ValueError("Event ordering/uniqueness differs from exact H63")
        store = events.CompletionEventStore(root, manifest, arrays)
        store.logical_root = root
        try:
            postproof = manifest.get("coverage", {}).get("query_mapping") == "complete_all_prepared_rows"
            if full or postproof:
                catalog_pin = manifest.get("catalog", {})
                catalog_path = _local_payload(io, root, catalog_pin)
                if catalog_path is None:
                    raise ValueError("Event-store diagnostic catalog path/hash mismatch")
                with np.load(catalog_path, allow_pickle=False) as payload:
                    store.catalog = {name: payload[name] for name in payload.files}
                if canonical_hash({name: diagnostics.array_hash(value) for name, value in store.catalog.items()}) != catalog_pin.get("arrays_sha256"):
                    raise ValueError("Event-store effective diagnostic metadata mismatch")
                receipt = manifest.get("query_coverage_receipt", {})
                if receipt.get("contract") != QUERY_JOIN or receipt.get("error_count") != 0 or set(receipt.get("splits", {})) != {"train", "val"}:
                    raise ValueError("Full event store lacks an all-prepared query coverage receipt")
                if set(manifest.get("query_arrays", {})) != {"train", "val"}:
                    raise ValueError("Full event store lacks persisted prepared row lookup")
                for split, declarations in manifest["query_arrays"].items():
                    if set(declarations) != {"user_id", "origin", "length"}:
                        raise ValueError("Prepared query lookup schema mismatch")
                    store.query_arrays[split] = {}
                    for name, declaration in declarations.items():
                        target = _local_payload(io, root, declaration)
                        if target is None:
                            raise ValueError("Prepared query lookup payload hash/path mismatch")
                        values = np.load(target, mmap_mode="r", allow_pickle=False)
                        store.query_arrays[split][name] = values
                        dtype = {"user_id": "int64", "origin": "int32", "length": "uint8"}[name]
                        if (values.shape != (receipt["splits"][split]["rows"],) or str(values.dtype) != dtype
                                or declaration["shape"] != list(values.shape)):
                            raise ValueError("Prepared query lookup shape/dtype mismatch")
                    if receipt["splits"][split]["row_keys_sha256"] != manifest["row_keys_sha256"].get(split):
                        raise ValueError("Coverage receipt/prepared row hash mismatch")
                for pin in manifest["prepared_pins"].values():
                    if not _verify_pin(io, root, pin):
                        raise ValueError("Prepared source file SHA mismatch")
                if full:
                    qa = manifest.get("final_validation", {})
                    if (qa.get("contract") != FULL_QA or qa.get("status") != "passed" or qa.get("error_count") != 0
                            or qa.get("events_count") != manifest["events_count"]
                            or qa.get("query_counts") != {split: data["rows"] for split, data in receipt["splits"].items()}):
                        raise ValueError("Full store lacks completed final validation; staged data is not admissible")
            return store
        except BaseException:
            store.close()
            raise
    except BaseException:
        for values in arrays.values():
            if getattr(values, "_mmap", None) is not None:
                _close_array(values)
        raise


def load_user_diagnostics(io, path, *, np, events, diagnostics, expected_metadata_sha256=None, expected_prepared_pins=None,
         expected_row_keys_sha256=None, verify_hashes=True):
    """Fail-closed typed diagnostics loader; no unverified source/hash bypass."""
    root = path
    CONTRACT, FAMILIES = diagnostics.CONTRACT, diagnostics.FAMILIES
    manifest = io.read_json(logical_child(root, "manifest.json"), signed=True)
    if (verify_hashes is not True or manifest.get("contract") != CONTRACT
            or manifest.get("state") != "completed" or manifest.get("coverage") != "full_rows"
            or manifest.get("storage") != "normalized_event_facts_exact_query_mapping"
            or manifest.get("families") != list(FAMILIES) or manifest.get("diagnostic_only") is not True
            or manifest.get("model_input_columns_added") != []
            or canonical_hash({k: v for k, v in manifest.items() if k != "signature"}) != manifest.get("signature")):
        raise ValueError("User diagnostics manifest/schema/signature mismatch")
    for name in ("event_store", "threshold_facts"):
        pin = manifest[name]
        if not _verify_pin(io, root, pin):
            raise ValueError("User diagnostics native dependency SHA mismatch")
    for pin in manifest["source_pins"].values():
        if not _verify_pin(io, root, pin):
            raise ValueError("User diagnostics source SHA mismatch")
    if expected_metadata_sha256 is not None and expected_metadata_sha256 != manifest["metadata_sha256"]:
        raise ValueError("User diagnostics metadata binding mismatch")
    if expected_prepared_pins is not None and expected_prepared_pins != manifest["prepared_pins"]:
        raise ValueError("User diagnostics prepared source binding mismatch")
    if expected_row_keys_sha256 is not None and expected_row_keys_sha256 != manifest["row_keys_sha256"]:
        raise ValueError("User diagnostics row key binding mismatch")
    store = load_event_store(io, _manifest_root(root, manifest["event_store"]), np=np, events=events,
        expected_metadata_sha256=manifest["metadata_sha256"], expected_prepared_pins=manifest["prepared_pins"],
        expected_row_keys_sha256=manifest["row_keys_sha256"])
    provider = None
    try:
        if (manifest["query_coverage_receipt"] != store.manifest["query_coverage_receipt"]
                or manifest["split_counts"] != {key: value["rows"] for key, value in
                    store.manifest["query_coverage_receipt"]["splits"].items()}):
            raise ValueError("User diagnostics full-row mapping proof mismatch")
        provider = _user_from_paths(io, root, store, manifest["threshold_facts"], np=np, diagnostics=diagnostics)
        schema = provider.features(np.array([], np.int64), [], x=np.zeros((0, 64), np.int64),
                                   lengths=np.array([], np.int64))
        if (manifest.get("columns") != schema["columns"] or manifest.get("dtype") != "float64"
                or manifest.get("null_mask_dtype") != "bool" or manifest.get("null_policy") != schema["null_policy"]
                or manifest.get("nullable_columns") != [name for name in schema["columns"]
                    if name.startswith(("typed.", "window.")) and name.rsplit(".", 1)[-1] in ("mean", "high", "low")]):
            raise ValueError("User diagnostics value/NULL schema mismatch")
        provider.manifest = manifest
        provider.provenance = manifest
        return provider
    except BaseException:
        if provider is not None:
            provider.close()
        else:
            store.close()
        raise


def load_diagnostics(io, root, declarations, *, metadata_hash, manifest, np,
                     events, diagnostics, require_sources, partial=False):
    """Return (provider contract summaries, owned diagnostic stores).

    ``require_sources`` is the composition layer's mandatory admitted-original
    module identity checker. It must reject unadmitted supplied backends.
    Older wide completion diagnostics and probe stores are explicitly unsupported.
    """
    if not callable(require_sources):
        raise TypeError("An admitted source checker is required")
    result, stores = {}, {}
    expected = {"expected_metadata_sha256": metadata_hash,
        "expected_prepared_pins": {split + ".npz": manifest["data"]["files"][split + ".npz"]
                                   for split in ("train", "val")},
        "expected_row_keys_sha256": manifest["row_keys_sha256"]}
    try:
        for name, receipt in declarations.items():
            logical = logical_child(root, receipt["path"])
            value = io.read_json(logical, expected=receipt["sha256"])
            contract = value.get("contract")
            if contract not in ("kion-user-diagnostics-v1", "kion-completion-event-store-v1"):
                raise ValueError("Unsupported diagnostic contract: " + str(contract))
            native_root = _manifest_root(root, receipt)
            if contract == "kion-user-diagnostics-v1":
                require_sources([diagnostics, events])
                store = load_user_diagnostics(io, native_root, np=np, events=events,
                                              diagnostics=diagnostics, **expected)
                stores[name] = store
                required = {"personal.activity", "personal.typed_completion",
                            "personal.window_completion", "personal.age_rating_profile"}
                if (value.get("clock") != "strict_h63_prior" or value.get("fit_cutoff") != "2021-08-09"
                        or value.get("storage") != "normalized_event_facts_exact_query_mapping"
                        or set(value.get("families", [])) != required
                        or not partial and value.get("coverage") != "full_rows"):
                    raise ValueError("User diagnostics require exact four-family normalized full query coverage")
                result[name] = {"manifest": receipt, "coverage": value["coverage"], "model_features": False,
                    "storage": value["storage"], "families": sorted(required)}
            else:
                require_sources([events])
                store = load_event_store(io, native_root, np=np, events=events, **expected)
                stores[name] = store
                coverage = store.manifest.get("coverage", {})
                if not partial and (coverage.get("raw_events") != "complete_before_role_cap"
                        or coverage.get("query_mapping") != "complete_all_prepared_rows"):
                    raise ValueError("Normalized completion event store lacks full corpus/query join proof")
                result[name] = {"manifest": receipt, "coverage": coverage, "model_features": False,
                    "storage": "normalized_events_plus_deterministic_H63_join"}
        return result, stores
    except BaseException:
        for store in stores.values():
            store.close()
        raise
