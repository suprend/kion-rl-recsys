"""Portable r1 fit / r2 calibration repair. Default invocation only plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import prepare_reward_data as prep

ROOT = prep.ROOT
R2_SHA = "cb5446663794ec143b3d72bc2e4a5d8de3599c5a9f3a3c53baaeb44ed4d88727"
HISTORICAL_BUNDLE = "42d7532a5daea8af36198e0a4ba6ba3b134cd66e93dcfd75961913a14624144c"
HISTORICAL_R1 = "4b7c87e51bf42c0fb2c7d532979406cfc162e37dc7dc37107f9bae9c192363a8"
CONFIGS = [{"name": "no_history_depth4", "history": False, "depth": 4, "iterations": 160},
           {"name": "history_depth4", "history": True, "depth": 4, "iterations": 160}]
PARAMETERS = {"configs": CONFIGS, "seed": 2026, "task_type": "CPU", "thread_count": 2,
              "learning_rate": .05, "l2_leaf_reg": 10., "border_count": 64,
              "loss_function": "RMSE", "od_type": "Iter", "od_wait": 30, "use_best_model": True}
# The retained RM manifest is 26,882,770 bytes: catalog provenance repeats
# feature descriptors. Scorers embed this same catalog. Source files retain
# prepare_reward_data's independent 4 MiB bound.
RM_MANIFEST_LIMIT = prep.RM_MANIFEST_LIMIT


def read_rm_manifest(path):
    return prep.read_json(path, limit=RM_MANIFEST_LIMIT)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def sources(root):
    r1, pins = prep.verify_sources(root)
    result = {(r1 / name).as_posix(): sha for name, sha in pins.items() if name != prep.SERIES}
    result[(Path(root) / prep.SERIES_INPUT).as_posix()] = prep.SERIES_SHA
    r2 = r1.parent / "bundle-rm-r2/scripts/kion_reward_boosting.py"
    result[r2.as_posix()] = R2_SHA
    for path, expected in result.items():
        if hashlib.sha256(prep.bounded_bytes(Path(path))).hexdigest() != expected:
            raise ValueError("Pinned source drift: " + path)
    return r1 / "scripts/kion_reward_boosting.py", r2, result


def scorer_manifest(manifest, bundle_sha, source_sha):
    unsigned = {k: v for k, v in manifest.items() if k != "signature"}
    if (manifest.get("signature") != canonical_hash(unsigned) or
            manifest.get("contract") != "kion-reward-boosting-v1" or
            manifest.get("backend") != "catboost_cpu" or
            manifest.get("score_id_space") != "original_item_id" or
            manifest.get("source_sha256") != source_sha or
            manifest.get("catalog", {}).get("mapping_sha256") != prep.MAPPING_SHA or
            manifest.get("report", {}).get("bundle_manifest_sha256") != bundle_sha or
            manifest.get("config") not in CONFIGS or
            set(manifest.get("files", {})) != {"scorer.npz", "model.cbm"}):
        raise ValueError("Scorer signature/source/bundle/config contract mismatch")
    for value in manifest["files"].values():
        prep.hash_string(value)
    return manifest["signature"]


def plan(stage, bundle, output, python, old_artifact=None, lineage="newly-fitted", root=ROOT):
    if stage not in ("fit-r1", "repair-r2") or lineage not in ("newly-fitted", "historical-artifact"):
        raise ValueError("Unknown stage or lineage")
    if stage == "fit-r1" and (old_artifact is not None or lineage != "newly-fitted"):
        raise ValueError("Fit creates a new lineage; old-artifact is repair-only")
    if stage == "repair-r2" and old_artifact is None:
        raise ValueError("Repair requires --old-artifact")
    r1, r2, pins = sources(root)
    bundle, output = map(prep.safe_path, (bundle, output))
    old = prep.safe_path(old_artifact) if old_artifact is not None else None
    receipt = prep.safe_path(str(output) + ".rm-launch.json")
    for target in (output, receipt):
        if target.exists() or not target.parent.is_dir():
            raise ValueError("Output/receipt must be absent with an existing parent")
        for tree in (prep.safe_path(root), bundle, *([old] if old else [])):
            if tree == target or tree in target.parents or target in tree.parents:
                raise ValueError("Outputs must be disjoint from source and input trees")
    interpreter = Path(python).absolute()
    if not interpreter.is_file():
        raise ValueError("An explicit existing RM Python executable is required")
    manifest, bundle_sha = read_rm_manifest(bundle / "manifest.json")
    prep.validate_reward_manifest(manifest)
    if manifest.get("builder_source_sha256") != prep.PINS["kion_reward_data"]:
        raise ValueError("RM producer pin mismatch")
    producer_pins = manifest.get("catalog", {}).get("provenance", {}).get("source_pins", {})
    expected_pins = {"builder": prep.PINS["kion_reward_data"], "boosting": prep.PINS["kion_reward_boosting"]}
    expected_pins.update({"component/" + name: sha for name, sha in prep.PINS.items()
                          if name not in ("kion_reward_data", "kion_reward_boosting")})
    expected_pins.update({name: prep.INPUT_HASHES[name] for name in ("metadata", "users", "items", "interactions")})
    if any(producer_pins.get(name) != sha for name, sha in expected_pins.items()):
        raise ValueError("RM input/feature source closure mismatch")
    if lineage == "historical-artifact" and bundle_sha != HISTORICAL_BUNDLE:
        raise ValueError("Historical repair requires the original immutable bundle")
    bounded = {(bundle / "manifest.json").as_posix(): bundle_sha}
    declarations = [(bundle, dict(manifest["files"], **manifest["sidecars"]))]
    args = ["fit", "--bundle", bundle.as_posix(), "--output", output.as_posix(), "--backend", "catboost", "--seed", "2026"]
    signature = None
    if old is not None:
        previous, old_sha = read_rm_manifest(old / "manifest.json")
        signature = scorer_manifest(previous, bundle_sha, prep.PINS["kion_reward_boosting"])
        if lineage == "historical-artifact" and signature != HISTORICAL_R1:
            raise ValueError("Historical r1 signature mismatch")
        bounded[(old / "manifest.json").as_posix()] = old_sha
        declarations.append((old, previous["files"]))
        args = ["repair-calibration", "--old-artifact", old.as_posix(), "--legacy-source", r1.as_posix(),
                "--bundle", bundle.as_posix(), "--output", output.as_posix(),
                "--expected-old-signature", signature, "--expected-mapping-sha256", prep.MAPPING_SHA]
    inputs = []
    for directory, files in declarations:
        for name, expected in sorted(files.items()):
            path = prep.safe_path(directory / name)
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Missing/empty payload: " + str(path))
            inputs.append({"path": path.as_posix(), "bytes": path.stat().st_size,
                           "expected_sha256": expected, "content_verified": False})
    return {"contract": "kion-rm-fit-repair-v1", "mode": "plan", "stage": stage, "lineage": lineage,
            "argv": [interpreter.as_posix(), "-B", (r1 if stage == "fit-r1" else r2).as_posix(), *args],
            "output": output.as_posix(), "receipt": receipt.as_posix(), "inputs": inputs,
            "verified_source_pins": pins, "verified_bounded_metadata": bounded,
            "bundle_manifest_sha256": bundle_sha, "expected_old_signature": signature,
            "parameters": PARAMETERS, "required_environment": prep.ENVIRONMENT,
            "input_content_verified": False, "environment_verified": False,
            "historical_result_equivalence_verified": False,
            "scope": "fit includes training and predictions; repair includes predictions/calibration but no refit"}


def check_inputs(request):
    for path, expected in request["verified_source_pins"].items():
        if hashlib.sha256(prep.bounded_bytes(Path(path))).hexdigest() != expected:
            raise ValueError("Bounded source drift")
    for path, expected in request["verified_bounded_metadata"].items():
        if read_rm_manifest(Path(path))[1] != expected:
            raise ValueError("RM manifest drift")
    for entry in request["inputs"]:
        path = prep.safe_path(entry["path"])
        if path.stat().st_size != entry["bytes"] or prep.digest(path) != entry["expected_sha256"]:
            raise ValueError("Input payload checksum/size mismatch")
        entry["content_verified"] = True


def execute(*, execute_training=False, execute_repair=False, timeout_seconds=3600, **kwargs):
    stage = kwargs.get("stage")
    if (stage == "fit-r1" and (not execute_training or execute_repair) or
            stage == "repair-r2" and (not execute_repair or execute_training)):
        raise ValueError("Stage requires its own explicit execution gate")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("Positive integer timeout required")
    request = plan(**kwargs)
    check_inputs(request)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    probe = "import importlib.metadata as m,json,platform; print(json.dumps(dict(python=platform.python_version(), **{k:m.version(k) for k in ['numpy','catboost','scipy']})))"
    result = prep.run_child([request["argv"][0], "-I", "-B", "-c", probe], min(60, timeout_seconds),
                            capture_output=True, text=True, env=env)
    if json.loads(result.stdout) != prep.ENVIRONMENT:
        raise ValueError("Selected interpreter is not the pinned RM environment")
    check_inputs(request)
    output = prep.safe_path(request["output"])
    receipt = prep.safe_path(request["receipt"])
    if output.exists():
        raise ValueError("Output appeared before dispatch")
    request.update(mode="execute", status="started", input_content_verified=True, environment_verified=True)
    def write(mode="w"):
        with receipt.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    write("x")
    try:
        prep.run_child(request["argv"], timeout_seconds, env=env, cwd=output.parent)
        check_inputs(request)
        manifest, observed_sha = read_rm_manifest(output / "manifest.json")
        signature = scorer_manifest(manifest, request["bundle_manifest_sha256"],
                                    prep.PINS["kion_reward_boosting"] if stage == "fit-r1" else R2_SHA)
        for name, sha in manifest["files"].items():
            if prep.digest(prep.safe_path(output / name)) != sha:
                raise ValueError("Output checksum mismatch")
        if stage == "repair-r2":
            repair = manifest.get("report", {}).get("calibration_repair", {})
            old_model = next(x["expected_sha256"] for x in request["inputs"] if Path(x["path"]).name == "model.cbm")
            if (repair.get("legacy_signature") != request["expected_old_signature"] or
                    repair.get("model_weights_copied_byte_for_byte") is not True or
                    repair.get("catboost_fitted") is not False or
                    repair.get("calibration_support_only") is not True or manifest["files"]["model.cbm"] != old_model):
                raise ValueError("Repair output lineage/unchanged trees mismatch")
        request.update(status="source-cli-completed-output-checksums-verified", output_signature=signature,
                       output_manifest_sha256=observed_sha)
    except BaseException:
        request.update(status="failed-or-cancelled")
        write()
        raise
    write()
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit-r1", "repair-r2"))
    for name in ("bundle", "output", "python"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--old-artifact", type=Path)
    parser.add_argument("--lineage", choices=("newly-fitted", "historical-artifact"), default="newly-fitted")
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--execute-repair", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = vars(parser.parse_args(argv))
    training, repair, timeout = (args.pop(k) for k in ("execute_training", "execute_repair", "timeout_seconds"))
    try:
        result = execute(**args, execute_training=training, execute_repair=repair, timeout_seconds=timeout) if training or repair else plan(**args)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"train_reward_model: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
