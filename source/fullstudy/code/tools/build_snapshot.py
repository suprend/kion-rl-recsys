"""Build an explicitly allowlisted, text-only source snapshot (stdlib only).

This is packaging, never a launch path. It reads only declared requests and their
small .py/.ps1 source pins plus explicitly named project text files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


TEXT_SUFFIXES = {".py", ".ps1", ".toml", ".txt", ".md", ".json"}
MAX_FILE = 1 << 20


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def read_json(path):
    if path.stat().st_size > 2 << 20:
        raise ValueError(f"Oversized request: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--pas-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--request", action="append", default=[], help="Project-relative request path")
    parser.add_argument("--freeze", action="append", default=[], help="Project-relative source freeze with bundle/sources")
    parser.add_argument("--extra", action="append", default=[], help="Project-relative text source/config path")
    parser.add_argument("--plan", type=Path, help="Explicit JSON allowlist with requests/freezes/extras")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    if args.plan:
        plan = read_json(args.plan)
        args.request += plan["requests"]
        args.freeze += plan["freezes"]
        args.extra += plan["extras"]
    project, pas, report = (p.resolve() for p in (args.project_root, args.pas_root, args.report_root))
    roots = [(project, "tlab-recsys-2026"), (pas, "personal-agent-system")]
    sources, requests, inputs = {}, [], {}

    def portable(value):
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, dict):
            return {portable(k): portable(v) for k, v in value.items()}
        if not isinstance(value, str):
            return value
        value = value.replace(project.as_posix(), "${PROJECT}").replace(pas.as_posix(), "${PERSONAL_OS}")
        if value.startswith("C:/Users/"):
            marker = "/.cache/codex-kion/"
            value = "${BANK_CACHE_ROOT}/" + value.split(marker, 1)[1] if marker in value else "${EXTERNAL_RUNTIME}/" + Path(value).name
        if value.startswith("E:/codex-kion-full-joint-e1-20261024-r1"):
            value = value.replace("E:/codex-kion-full-joint-e1-20261024-r1", "${GRAPH_CACHE_ROOT}", 1)
        if value.startswith("E:/codex-kion-cache-20261022-e1-r1"):
            value = value.replace("E:/codex-kion-cache-20261022-e1-r1", "${QUERY_STORE_ROOT}", 1)
        return value

    parameter_keys = {
        "stage", "operation", "module_path", "function", "train_argv", "precision", "device", "seed",
        "cpu_threads", "rows", "train_rows", "validation_rows", "epochs", "supervised_updates", "rl_updates",
        "new_rm_fits", "final_test_reads", "source_columns", "feature_schema_sha256", "teacher",
        "bank_dir", "head_dir", "evaluation_dir", "analysis_dir", "cache_dir", "reward_dir", "rl_output",
        "allow_validation_graph_build", "validation_graph_cache", "cohort_name", "cohort_selection_sha256",
        "frozen_C_supervised_epochs", "new_supervised_epochs", "catboost_pins", "numpy_pins", "chunk_size",
        "max_states", "scoring_seconds", "preregistration_sha256", "frontier_sha256", "loss_normalization",
        "policy_module", "utility_module", "utility_dir", "prior_archive", "cohorts", "utility_output",
        "preregistration", "prefit_path", "prefit_sha256", "parity_proof", "residual_module", "evaluate_module",
        "scientific_checkpoints", "full_evaluation_dir", "original_evaluation_dir", "original_evaluation_sha256",
        "parity_atol", "parity_rtol", "screen_evaluation_dir", "legacy_screen_outputs", "rollout_output",
        "rollout_positions", "rollout_selection", "repeat_masks", "rollout_selection_pin", "evaluation",
        "protocol", "output", "checkpoint_curve", "rollout", "items", "expected_positions", "data", "plot_module"
    }

    def identify(path):
        path = path.resolve()
        for root, label in roots:
            if path.is_relative_to(root):
                return label, path.relative_to(root).as_posix()
        raise ValueError(f"Source outside explicitly allowed project roots: {path.name}")

    for relative in args.request:
        path = (project / relative).resolve()
        identify(path)
        request = read_json(path)
        requests.append({"source_path": path.relative_to(project).as_posix(),
                         "sha256": digest(path.read_bytes()), "role": request.get("stage", "unknown"),
                         "parameters": portable({k: v for k, v in request.items() if k in parameter_keys})})
        for name, expected in request.get("source_hashes", {}).items():
            source = Path(name).resolve()
            if source.suffix not in {".py", ".ps1"}:
                continue
            identify(source)
            prior = sources.get(source)
            if prior is not None and prior != expected:
                raise ValueError(f"Conflicting source pins: {source.name}")
            sources[source] = expected
        for name, expected in request.get("input_hashes", {}).items():
            # Input descriptors are provenance only: never open the input payload.
            normalized = portable(name)
            if normalized in inputs and inputs[normalized]["sha256"] != expected:
                raise ValueError(f"Conflicting input provenance: {Path(name).name}")
            inputs[normalized] = {"sha256": expected, "included": False,
                                  "hash_evidence": "copied from pinned request; payload not reread during packaging"}
    for relative in args.freeze:
        path = (project / relative).resolve()
        identify(path)
        freeze = read_json(path)
        base = Path(freeze["bundle"]).resolve()
        identify(base)
        requests.append({"source_path": path.relative_to(project).as_posix(),
                         "sha256": digest(path.read_bytes()), "role": "frozen-source-bundle"})
        for name, expected in freeze["sources"].items():
            source = (base / name).resolve()
            identify(source)
            if source.suffix not in {".py", ".ps1"}:
                raise ValueError(f"Non-code frozen source: {name}")
            if source in sources and sources[source] != expected:
                raise ValueError(f"Conflicting frozen source pin: {source.name}")
            sources[source] = expected
    for relative in args.extra:
        path = (project / relative).resolve()
        identify(path)
        if path.suffix not in TEXT_SUFFIXES:
            raise ValueError(f"Non-text extension: {relative}")
        if path.stat().st_size > MAX_FILE:
            raise ValueError(f"Oversized explicit text file: {relative}")
        actual = digest(path.read_bytes())
        if path in sources and sources[path] != actual:
            raise ValueError(f"Conflicting explicit source pin: {relative}")
        sources[path] = actual
    copied = []
    imported = {}
    for path, expected in sorted(sources.items(), key=lambda pair: pair[0].as_posix()):
        if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > MAX_FILE:
            raise ValueError(f"Source violates text allowlist: {path.name}")
        payload = path.read_bytes()
        if b"\0" in payload or digest(payload) != expected:
            raise ValueError(f"Changed or binary source: {path.name}")
        text = payload.decode("utf-8-sig")
        label, relative = identify(path)
        destination = report / "code" / label / relative
        if path.suffix == ".py":
            tree = ast.parse(text, filename=relative)
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for name in names:
                    imported.setdefault(name.split(".")[0], set()).add(f"{label}/{relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.read_bytes() != payload:
            destination.write_bytes(payload)
        copied.append({"project": label, "source_path": relative,
                       "bundle_path": destination.relative_to(report).as_posix(),
                       "bytes": len(payload), "sha256": expected, "role": "exact-source-snapshot"})
    json_write(report / "code/recipes/source-requests.json", {
        "contract": "kion-source-request-provenance-v1", "requests": requests,
        "warning": "Request hashes identify historical inputs; this file is not an executable launch request."})
    json_write(report / "code/recipes/excluded-inputs.json", {
        "contract": "kion-excluded-inputs-v1", "inputs": inputs,
        "availability": "Excluded local payloads; obtain authorized KION data and rebuild derived inputs, or restore original local outputs.",
        "fresh_payload_hashing": False})
    json_write(report / "code/recipes/import-inventory.json", {
        "contract": "kion-static-import-inventory-v1",
        "modules": {name: sorted(paths) for name, paths in sorted(imported.items())},
        "scope": "AST imports from declared snapshots; request pins cover dynamic loaders. No producer modules imported."})
    # Include the package helpers and recipes, with self-contained source hashes.
    known = {row["bundle_path"] for row in copied}
    for path in sorted((report / "code").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(report).as_posix()
        if relative in known:
            continue
        if not relative.startswith(("code/tools/", "code/recipes/", "code/fixtures/", "code/requirements/")):
            raise ValueError(f"Source snapshot file no longer in explicit allowlist: {relative}")
        if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > MAX_FILE:
            raise ValueError(f"Unexpected bundle file: {relative}")
        payload = path.read_bytes()
        if b"\0" in payload:
            raise ValueError(f"Binary bundle helper: {relative}")
        payload.decode("utf-8-sig")
        copied.append({"project": "bundle", "source_path": relative.removeprefix("code/"),
                       "bundle_path": relative, "bytes": len(payload), "sha256": digest(payload),
                       "role": "bundle-helper-or-recipe"})
    json_write(report / "code-manifest.json", {
        "contract": "kion-small-source-bundle-v1", "state": "snapshot",
        "source_commit": args.commit,
        "commit_limit": "KION sources include untracked/ignored files; exact included bytes and hashes are authoritative.",
        "source_roots": {"tlab-recsys-2026": "code/tlab-recsys-2026", "personal-agent-system": "code/personal-agent-system"},
        "licensing": "No repository LICENSE file found at project root; this snapshot does not grant additional redistribution rights. Third-party packages/data/models are excluded.",
        "files": sorted(copied, key=lambda row: row["bundle_path"]),
        "totals": {"files": len(copied), "bytes": sum(row["bytes"] for row in copied)},
        "scope": "Exact allowlisted text source/configs plus stdlib verification; no weight, array, raw dataset or real user-record payload.",
        "verification": {"source_pin_checks": "passed", "clean_machine_inference": "not run", "training_from_raw": "not run"}})
    print(json.dumps({"copied_sources": len(sources), "files": len(copied),
                      "bytes": sum(row["bytes"] for row in copied)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
