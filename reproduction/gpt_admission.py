"""Scoped admission component; no complete-provider or training entrypoint.

Original source bytes are pinned independently of this adapter. Only finite
declarations and the pure compose_payload function are extracted; historical
module imports and their filesystem/ML side effects are never executed.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from gpt_inputs import identity
from gpt_provider import (BoundIO, ORIGINAL_PROVIDER_SHA256, SOURCE_LIMIT,
                          canonical_hash, logical_child)

ASSEMBLER_SHA256 = "1cc9ca82a712c0a4cbb55e2f8a651b40ac698793f50e58d35b51d98f2bfb9549"
LOCAL_SHA256 = "87412446a72033aae98897250cc0178084dd76b5818aee38a24fbb81a8e02aaf"
_TOKEN = object()


@dataclass(frozen=True, init=False)
class AdmissionBackend:
    """Construct only with from_sources; source identities cannot be replaced."""
    provider_source: str
    assembler_source: str
    local_source: str
    _constants_json: str
    _compose: object

    def __init__(self, token, provider, assembler, local, constants, compose):
        if token is not _TOKEN:
            raise TypeError("use AdmissionBackend.from_sources")
        for name, value in (("provider_source", identity(provider)),
                            ("assembler_source", identity(assembler)),
                            ("local_source", identity(local)),
                            ("_constants_json", json.dumps(constants)), ("_compose", compose)):
            object.__setattr__(self, name, value)

    @classmethod
    def from_sources(cls, io, *, provider_source, assembler_source, local_source):
        """Read only the three exact pinned sources through explicit BoundIO."""
        if not isinstance(io, BoundIO):
            raise TypeError("explicit BoundIO required")
        provider = _source(io, provider_source, ORIGINAL_PROVIDER_SHA256)
        assembler = _source(io, assembler_source, ASSEMBLER_SHA256)
        local = _source(io, local_source, LOCAL_SHA256)
        constants = {}
        for tree, names in ((provider, ("CONTRACT", "SCOPED_FAMILIES", "SCOPED_READY_COMPONENTS",
                                       "NUMERIC_TRANSFORMS", "COMPONENT_ADAPTERS")),
                            (assembler, ("SCOPED_DECISION_SHA256",)),
                            (local, ("ROW_ALGORITHM",))):
            for node in tree.body:
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in names:
                        constants[name] = ast.literal_eval(node.value)
            if any(name not in constants for name in names):
                raise ValueError("Pinned source declaration missing")
        # These declarations include finite comprehensions in the pinned source.
        # Execute only this fixed allowlist, never imports or an arbitrary name.
        names = {"SCOPED_FAMILIES", "SCOPED_READY_COMPONENTS", "SCOPED_COMPONENTS",
                 "AGREEMENT_COMPONENTS", "DEFERRED_PROPOSALS"}
        selected = [node for node in assembler.body if
                    isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
        compose = [node for node in assembler.body if isinstance(node, ast.FunctionDef)
                   and node.name == "compose_payload"]
        if len(selected) != len(names) or len(compose) != 1:
            raise ValueError("Pinned assembler definitions missing")
        namespace = {"__builtins__": {"tuple": tuple, "sorted": sorted, "isinstance": isinstance,
                                      "list": list, "dict": dict, "ValueError": ValueError}}
        exec(compile(ast.Module(body=selected + compose, type_ignores=[]), assembler_source, "exec"), namespace)
        if (list(namespace["SCOPED_FAMILIES"]) != list(constants["SCOPED_FAMILIES"])
                or list(namespace["SCOPED_READY_COMPONENTS"]) != list(constants["SCOPED_READY_COMPONENTS"])):
            raise ValueError("Pinned source scoped contracts disagree")
        constants.update({name: namespace[name] for name in names})
        return cls(_TOKEN, provider_source, assembler_source, local_source, constants, namespace["compose_payload"])

    def constants(self):
        """Return a detached JSON copy; edits cannot alter backend policy."""
        return json.loads(self._constants_json)

    def compose_payload(self, assessment):
        # Historical compose mutates nested runtime values; isolate its input.
        return self._compose(copy.deepcopy(assessment),
                             {"COMPONENT_ADAPTERS": self.constants()["COMPONENT_ADAPTERS"]})


def _source(io, logical, expected):
    path = io.verify(logical, "source", expected)
    with path.open("rb") as stream:
        raw = stream.read(SOURCE_LIMIT + 1)
    io.check()
    if len(raw) > SOURCE_LIMIT or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Pinned source changed before extraction")
    return ast.parse(raw.decode("utf-8"), filename=logical)


def _scoped_decision(io, root, receipt, constants):
    """Exact scoped_decision predicates with a bound logical read seam."""
    if not isinstance(receipt, dict) or receipt.get("sha256") != constants["SCOPED_DECISION_SHA256"]:
        raise ValueError("Scoped production requires the exact human-approved scope decision")
    decision = io.read_json(logical_child(root, receipt["path"]), expected=receipt["sha256"])
    if (decision.get("contract") != "kion-user-approved-ready-scope-v1"
            or decision.get("scope_id") != "ready-pack-20261020-r1"
            or decision.get("active_families") != constants["SCOPED_FAMILIES"]
            or decision.get("additional_ready_components") != constants["SCOPED_READY_COMPONENTS"]):
        raise ValueError("Scoped production decision identity/families mismatch")
    return decision


@dataclass(frozen=True)
class ScopedAdmission:
    """Admission facts only; does not validate feature payloads or allow training."""
    manifest_json: str
    scope_json: str
    acceptance: object
    required_families: tuple
    partial: bool = False
    scoped: bool = True

    @property
    def manifest(self):
        return json.loads(self.manifest_json)

    @property
    def scope(self):
        return json.loads(self.scope_json)


def admit_scoped(io, feature_root, *, requested_scope_decision, backend, manifest_sha256=None):
    """Port CompleteProvider._admission's scoped branch, using logical equality.

    Explicit execution reads/hash-verifies admission metadata and declared pins.
    Non-scoped modes fail closed. All data stays detached from callers; original
    signed dictionaries and their logical path values are never rewritten.
    """
    if not isinstance(io, BoundIO) or type(backend) is not AdmissionBackend:
        raise TypeError("BoundIO and pinned AdmissionBackend required")
    identity(feature_root)
    c = backend.constants()
    m = io.read_json(logical_child(feature_root, "manifest.json"), expected=manifest_sha256, signed=True)
    if m.get("contract") != c["CONTRACT"] or m.get("state") != "completed":
        raise ValueError("Unfinished/unknown complete package")
    if m.get("integration_mode") != "scoped_production":
        raise ValueError("This adapter supports scoped_production only")
    if requested_scope_decision is None:
        raise ValueError("Scoped production requires explicit scope_decision")
    if identity(requested_scope_decision) != logical_child(feature_root, m["scope_decision"]["path"]):
        raise ValueError("Requested scope decision path differs from scoped manifest")
    scope = _scoped_decision(io, feature_root, m.get("scope_decision"), c)
    families = c["SCOPED_FAMILIES"]
    if (m.get("scope_id") != scope["scope_id"] or m.get("active_families") != families
            or m.get("status") != "admitted_scoped" or m.get("full_row_fit_admitted") is not True
            or m.get("full_package_admitted") is not False
            or m.get("additional_ready_components") != c["SCOPED_READY_COMPONENTS"]
            or set(m.get("families", {})) != set(families)
            or any(m.get("runtime", {}).get(key) for key in ("people_dynamics", "audience"))
            or m.get("entity_links") is not None
            or any(d.get("family") != "external_interest" for d in m.get("date_features", []))):
        raise ValueError("Scoped production payload/families differ from the approved ready scope")
    if m.get("required_families") != families:
        raise ValueError("Required family census differs from the complete contract")

    def pin(receipt):
        logical = logical_child(feature_root, receipt["path"])
        io.pin(feature_root, receipt, io.bindings[logical].role)

    acceptance = {}
    for family in families:
        receipt = m.get("families", {}).get(family, {})
        if not (receipt.get("status") == "accepted" and receipt.get("reviewed") is True
                and receipt.get("scope") == "approved_ready_scope" and bool(receipt.get("source_pins"))):
            raise ValueError("Full package rejects missing/unreviewed/pilot family: " + family)
        acceptance[family] = "accepted_scoped"
        for source_pin in receipt["source_pins"]:
            pin(source_pin)
    if m.get("fit_cutoff") != "2021-08-09" or m.get("validation_origin") != "2021-08-09":
        raise ValueError("SFT role cap/frozen validation origin mismatch")
    if m.get("numeric_transforms") != c["NUMERIC_TRANSFORMS"]:
        raise ValueError("Numeric transform registry differs from the complete contract")
    if m.get("rowkeys_algorithm") != c["ROW_ALGORITHM"]:
        raise ValueError("Unknown complete row binding algorithm")
    pins = m.get("source_pins", [])
    declared = {logical_child(feature_root, p["path"]) for p in pins}
    if backend.provider_source not in declared:
        raise ValueError("Complete builder source pin is required")
    for source_pin in pins:
        pin(source_pin)
    if backend.assembler_source not in declared:
        raise ValueError("Complete manifest assembler source pin is required")
    # Check source identity independently of adapter bytes and any declaration.
    for logical, digest in ((backend.provider_source, ORIGINAL_PROVIDER_SHA256),
                            (backend.assembler_source, ASSEMBLER_SHA256),
                            (backend.local_source, LOCAL_SHA256)):
        io.verify(logical, "source", digest)
    assessment = io.read_json(logical_child(feature_root, "assessment.json"))
    assembly = m.get("assembly") or {}
    if (assembly.get("status") != "scoped_data_ready_untrained"
            or assembly.get("assessment_sha256") != canonical_hash(assessment)
            or assessment.get("contract") != "kion-complete-scoped-assessment-v1"
            or assessment.get("status") != "scoped_data_ready_untrained" or assessment.get("blockers")
            or assessment.get("scope_decision") != m["scope_decision"]
            or assessment.get("active_families") != families
            or set(assessment.get("components", {})) != set(c["SCOPED_COMPONENTS"])
            or any(not component.get("ready") for component in assessment.get("components", {}).values())
            or assessment.get("provider_source", {}).get("sha256") != ORIGINAL_PROVIDER_SHA256
            or assessment.get("base_manifest", {}).get("sha256") != m["base_cache"]["manifest_sha256"]
            or assessment.get("metadata", {}).get("sha256") != m["data"]["files"]["metadata.json"]["sha256"]):
        raise ValueError("Scoped production requires its bound retained-component assessment")
    expected, pending = backend.compose_payload(assessment)
    if pending or any(m.get(key) != value for key, value in expected.items()):
        raise ValueError("Full manifest payload differs from its reviewed approved-ID assessment")
    for component in assessment["components"].values():
        pin(component["evidence"])
        pin(component["review"])
    return ScopedAdmission(json.dumps(m, ensure_ascii=False), json.dumps(scope, ensure_ascii=False),
                           MappingProxyType(acceptance), tuple(families))
