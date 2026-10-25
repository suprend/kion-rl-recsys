"""Pinned RM preprocessing only. Default plan reads bounded metadata and stats payloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/local-pack-20261020/bundle-rm-r1"
PINS = {
    "kion_basic_bundle_features": "25c5a80a2fba9de5ee76c67ac4bd6cd7bd437cf807cd42855baeba3b50f73ef8",
    "kion_local_features": "e5f992093b53839d425ccd436dba0404862ad912b795a390d60ff6c9a2576de1",
    "kion_sex_features": "e524c68af1ad327c3e83959aa4503be17aabdc700beaa24793831d09a152a905",
    "kion_reward_data": "bc767e7987b9404273cc24a12e1f91562e480d0cfef39302e079fba7e296ddf6",
    "kion_reward_boosting": "51421aea405abb210bb49b4aa1e694170b21296b75a15f7e228b01416fc085b1",
    "kion_local_pack": "87412446a72033aae98897250cc0178084dd76b5818aee38a24fbb81a8e02aaf",
    "kion_pack_static": "5a52ef6ab610b0f3d091bd6566982bd0894f1d0c1ad891986dd27d3f8703e181",
    "kion_pack_people": "25c9b1f9ac9188f096f22085d63a5de694e55a757c9857da963854bb808e791b",
    "kion_pack_semantics": "74a59bad3bf89c5eb92624452ffac8e4775c18592823411a611d09d50fe3d7ff",
    "kion_pack_history": "ecdd0cd0e853d20578a7855acc8987c72c430925d1a30f7e26fee250000659dd",
    "kion_pack_dynamic": "7f53e94ddd5940514de8a132711ff5d782885983af9f0c62ae4c4515261bdd7a",
}
SERIES = "configs/datasets/kion-series-v1.json"
SERIES_INPUT = "reproduction/profiles/kion-series-v1.json"
SERIES_SHA = "18b672eda2fd8d711e2642fd933740abf0f28bfaf4031e27b46a5ee77ba647d6"
INPUT_HASHES = {
    "interactions": "201dc9114eb309da9df6f5f227ada23a371d668edf64a32090fb613a75162ecf",
    "users": "37b2945240628a3f215e1fb5f79b4cff386d8988b21fa61e5b135aef4541182c",
    "items": "027c3a0027b4fe4a8db9d5fa626ab40bab507c96aac81d462db0771eb2cb4a8e",
    "metadata": "f02acaa9a5e5532e79b060279ba8a5027f268f202132e85ebab97cd4912fbd11",
    "train": "bed53339345246fbd3cbad6d72017ed82451634431e48c6e9fd67a66581be4f8",
    "val": "4c8839b69a4abeff5e4eea3ffeea1d83715deea8775f4547f8f85c0e055bad99",
}
MAPPING_SHA = "b90df0d1fcce007008b62007e486bc1e9cb0b4ac89c97bb5107e84305e744db4"
ITEM_ARRAY_SHA = "0401c59e7747358145198fda463f1a88d476ddbe41c2e7b2bdd8f7c45a1cbe8c"
CAPS = {"fit": 100000, "select": 20000, "calibration": 20000}
FIELDS = ("numeric", "candidate_numeric", "categorical", "observed", "reward", "user_id", "item_id",
          "target_date", "history_cutoff", "history_max_date", "history_length")
LIMIT = 4 * 1024 * 1024
RM_MANIFEST_LIMIT = 64 * 1024 * 1024
ENVIRONMENT = {"python": "3.12.14", "numpy": "2.5.4", "catboost": "1.2.10", "scipy": "1.18.1"}
ROW_FACTS = {
    "train": {"rows": 3477437, "unique_users": 474966, "row_keys_sha256": "b99e73a20eeac391e6403b5fb87f52ff9b2216f2099a1093496810620e6bcb39"},
    "val": {"rows": 158760, "unique_users": 105212, "row_keys_sha256": "48774b171466c5b0e90c0d4295a4e91f264f108193858a9790640c8f3513e9bd"},
}
PREPARE_SHA = "81e8ac99a6d50398418fd95e0874bc07b55b3d33487e89731ec26397960c3e77"


def safe_path(value):
    path = Path(value).absolute()
    for part in (path, *path.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Symlink/junction is not allowed: {part}")
    return path.resolve()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bounded_bytes(path, *, limit=LIMIT):
    path = safe_path(path)
    if type(limit) is not int or limit <= 0:
        raise ValueError("Positive integer file bound required")
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f"Missing or oversized bounded file: {path}")
    with path.open("rb") as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Bounded file grew during read")
    return value


def read_json(path, *, limit=LIMIT):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    data = bounded_bytes(path, limit=limit)
    value = json.loads(data, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value, hashlib.sha256(data).hexdigest()


def verify_sources(root):
    source = safe_path(Path(root) / SOURCE)
    pins = {f"scripts/{name}.py": value for name, value in PINS.items()}
    pins[SERIES] = SERIES_SHA
    for name, expected in pins.items():
        path = Path(root) / SERIES_INPUT if name == SERIES else source / name
        if hashlib.sha256(bounded_bytes(path)).hexdigest() != expected:
            raise ValueError(f"Pinned source SHA256 mismatch: {name}")
    return source, pins


def stage_runtime(request, root):
    runtime = safe_path(request["runtime_root"])
    runtime.mkdir(exist_ok=False)
    for name, expected in request["verified_source_pins"].items():
        source = Path(root) / SERIES_INPUT if name == SERIES else Path(request["source_root"]) / name
        data = bounded_bytes(source)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Source drift before runtime staging")
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    raw = runtime / "raw"
    raw.mkdir()
    for item in request["inputs"]:
        source = Path(item["path"])
        if source.suffix == ".csv":
            with source.open("rb") as src, (raw / source.name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
    if request["raw_manifest_present"]:
        with (raw / "manifest.json").open("xb") as stream:
            stream.write(bounded_bytes(Path(request["raw_input_root"]) / "manifest.json"))


def check_runtime(request):
    for name, expected in request["verified_source_pins"].items():
        if hashlib.sha256(bounded_bytes(Path(request["runtime_root"]) / name)).hexdigest() != expected:
            raise ValueError("Staged runtime drift")
    for item in request["inputs"]:
        source = Path(item["path"])
        if source.suffix == ".csv" and digest(safe_path(Path(request["runtime_root"]) / "raw" / source.name)) != item["expected_sha256"]:
            raise ValueError("Staged raw input drift")
    if request["raw_manifest_present"]:
        expected = request["verified_bounded_metadata"][(Path(request["raw_input_root"]) / "manifest.json").as_posix()]
        if hashlib.sha256(bounded_bytes(Path(request["runtime_root"]) / "raw/manifest.json")).hexdigest() != expected:
            raise ValueError("Staged raw manifest drift")


def hash_string(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA256 declaration")
    return value


def validate_basic(manifest, input_mode="historical", prepared_hashes=None):
    if manifest.get("builder_contract") != "kion-basic-bundle-v1":
        raise ValueError("Basic builder contract mismatch")
    if manifest.get("vocabulary", {}).get("sha256") != MAPPING_SHA:
        raise ValueError("Basic vocabulary mismatch")
    for name in ("users", "items", "metadata", "train", "val"):
        expected = INPUT_HASHES[name]
        if input_mode == "rebuild" and name in ("train", "val"):
            entry = manifest.get("inputs", {}).get(name, {})
            if any(entry.get(k) != v for k, v in ROW_FACTS[name].items()):
                raise ValueError("Basic rebuilt split row identity mismatch")
            expected = prepared_hashes[name] if prepared_hashes else hash_string(entry.get("sha256"))
        if manifest.get("inputs", {}).get(name, {}).get("sha256") != expected:
            raise ValueError(f"Basic input identity mismatch: {name}")
    features = manifest.get("features", {})
    if features.get("file") != "features.npz":
        raise ValueError("Basic filename mismatch")
    if features.get("arrays", {}).get("item_features", {}).get("sha256") != ITEM_ARRAY_SHA:
        raise ValueError("Basic item coordinate identity mismatch")
    return hash_string(features.get("sha256"))


def validate_reward_manifest(manifest):
    """JSON semantics only; neither array validation nor numerical parity."""
    if (manifest.get("contract") != "kion-reward-boosting-v1" or
            manifest.get("layout") != "npy-directory-v1" or manifest.get("ready") is not True):
        raise ValueError("RM manifest contract mismatch")
    if manifest.get("partition", {}).get("salt") != "kion-rm-select-calibration-v1-20261020":
        raise ValueError("RM partition mismatch")
    if manifest.get("split") != {"fit": "<2021-08-02", "select_calibration": "2021-08-02..2021-08-08", "heldout_history": "frozen<2021-08-02"}:
        raise ValueError("RM split mismatch")
    catalog = manifest.get("catalog", {})
    provenance = catalog.get("provenance", {})
    if (catalog.get("mapping_sha256") != MAPPING_SHA or
            provenance.get("fit_cutoff_exclusive") != "2021-08-02" or
            provenance.get("candidate_features_dynamic") is not True or
            provenance.get("static_candidate_fallback") is not False):
        raise ValueError("RM feature/mapping mismatch")
    exclusions = manifest.get("exclusions", {})
    if exclusions.get("row_caps") != CAPS or exclusions.get("role_rows") != CAPS:
        raise ValueError("RM production row caps/counts mismatch")
    if exclusions.get("missing_imputed_as_zero") is not False or exclusions.get("unobserved_pairs_generated") != 0:
        raise ValueError("RM label policy mismatch")
    expected = {f"{role}.{field}.npy" for role in CAPS for field in FIELDS} | {"catalog.npz"}
    sidecars = {f"{role}.{field}.npy" for role in CAPS for field in ("x", "lengths")} | {"full_target_support.npz"}
    for key, names in (("files", expected), ("sidecars", sidecars)):
        if set(manifest.get(key, {})) != names:
            raise ValueError(f"RM {key} closure mismatch")
        for value in manifest[key].values():
            hash_string(value)
    if set(manifest.get("row_bindings", {})) != set(CAPS):
        raise ValueError("RM row binding closure mismatch")
    for value in manifest["row_bindings"].values():
        hash_string(value)


def validate_prepare_receipt(receipt, raw, prepared):
    """Receipt binds rebuilt splits to the reviewed producer; it is not a signature."""
    if (receipt.get("contract") != "kion-raw-split-plan-v1" or
            receipt.get("mode") != "execute-prepare" or receipt.get("status") != "source-cli-completed" or
            receipt.get("input_content_verified") is not True or receipt.get("source_sha256") != PREPARE_SHA or
            receipt.get("parameters") != {"context": 64, "max-train": 1000000000, "max-eval": 1000000000, "seed": 2026} or
            safe_path(receipt.get("output", "")) != prepared):
        raise ValueError("Rebuild needs a completed pinned prepare_data receipt")
    expected = {(raw / f"{name}.csv").as_posix(): INPUT_HASHES[name] for name in ("users", "items", "interactions")}
    entries = receipt.get("inputs", [])
    actual = {safe_path(item["path"]).as_posix(): item.get("expected_sha256") for item in entries if item.get("content_verified") is True}
    if actual != expected or len(entries) != 3:
        raise ValueError("Prepare receipt raw bindings mismatch")
    argv = receipt.get("argv", [])
    try:
        if safe_path(argv[argv.index("--input") + 1]) != raw or safe_path(argv[argv.index("--output") + 1]) != prepared:
            raise ValueError("Prepare receipt argv paths mismatch")
    except (IndexError, ValueError) as exc:
        raise ValueError("Prepare receipt argv paths mismatch") from exc


def plan(stage, raw, prepared, output, python, basic=None, root=ROOT, input_mode="historical"):
    if input_mode not in ("historical", "rebuild"):
        raise ValueError("Unknown input mode")
    if stage not in ("basic-features", "reward-data"):
        raise ValueError("Unknown preprocessing stage")
    source, pins = verify_sources(root)
    raw, prepared, output = map(safe_path, (raw, prepared, output))
    basic = safe_path(basic) if basic is not None else None
    if stage == "reward-data" and basic is None:
        raise ValueError("reward-data requires --basic")
    if stage == "basic-features" and basic is not None:
        raise ValueError("basic-features does not accept --basic")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Output must be absent with an existing parent")
    runtime = safe_path(str(output) + ".prepare-runtime")
    if runtime.exists() or safe_path(str(output) + ".prepare-launch.json").exists():
        raise ValueError("Runtime/receipt must also be absent")
    for tree in (safe_path(root), raw, prepared, *([basic] if basic else [])):
        for destination in (output, runtime, safe_path(str(output) + ".prepare-launch.json")):
            if destination == tree or destination in tree.parents or tree in destination.parents:
                raise ValueError("Output/runtime/receipt must be disjoint from code and all input trees")
    interpreter = Path(python).absolute()  # Preserve venv executable symlinks.
    if not interpreter.is_file():
        raise ValueError("Select an existing RM Python executable explicitly")
    metadata, metadata_hash = read_json(prepared / "metadata.json")
    if metadata_hash != INPUT_HASHES["metadata"]:
        raise ValueError("Historical prepared metadata SHA256 mismatch")
    mapping = metadata.get("item_id_to_token")
    if not isinstance(mapping, dict) or hashlib.sha256(json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() != MAPPING_SHA:
        raise ValueError("Metadata mapping mismatch")
    payloads = {name: raw / f"{name}.csv" for name in (("users", "items") if stage == "basic-features" else ("users", "items", "interactions"))}
    hashes = dict(INPUT_HASHES)
    bounded = {str(prepared / "metadata.json"): metadata_hash}
    if input_mode == "rebuild":
        receipt, value = read_json(prepared / "prepare-launch.json")
        validate_prepare_receipt(receipt, raw, prepared)
        bounded[str(prepared / "prepare-launch.json")] = value
    origin = raw / "manifest.json"
    if origin.exists():
        manifest, value = read_json(origin)
        declarations = {row["file"]: row["sha256"] for row in manifest.get("files", [])}
        if any(declarations.get(f"{name}.csv") != hashes[name] for name in payloads):
            raise ValueError("Raw manifest CSV hash mismatch")
        bounded[str(origin)] = value
    if stage == "basic-features":
        payloads.update({name: prepared / f"{name}.npz" for name in ("train", "val")})
        script = "kion_basic_bundle_features"
        args = ["--raw", (runtime / "raw").as_posix(), "--prepared", prepared.as_posix()]
    else:
        manifest, value = read_json(basic / "manifest.json")
        hashes["basic"] = validate_basic(manifest, input_mode)
        bounded[str(basic / "manifest.json")] = value
        payloads["basic"] = basic / "features.npz"
        script = "kion_reward_data"
        args = ["--raw", (runtime / "raw").as_posix(), "--metadata", (prepared / "metadata.json").as_posix(), "--basic-npz", (basic / "features.npz").as_posix()]
    if input_mode == "rebuild":
        # Required for both stages: receipt + actual row-key semantics, not NPZ bytes.
        payloads.update({name: prepared / f"{name}.npz" for name in ("train", "val")})
        for name in ("train", "val"):
            hashes[name] = None if stage == "basic-features" else manifest["inputs"][name]["sha256"]
    inputs = []
    for name, path in payloads.items():
        path = safe_path(path)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing/empty payload: {path}")
        inputs.append({"path": path.as_posix(), "bytes": path.stat().st_size, "expected_sha256": hashes[name], "content_verified": False})
    argv = [interpreter.as_posix(), "-B", (runtime / "scripts" / f"{script}.py").as_posix(), *args, "--output", output.as_posix()]
    if stage == "reward-data":
        for role, count in CAPS.items():
            argv.extend([f"--max-{role}", str(count)])
        argv.extend(["--batch-size", "512"])
    return {"contract": "kion-rm-prepare-plan-v1", "mode": "plan", "stage": stage, "input_mode": input_mode, "argv": argv,
            "source_root": source.as_posix(), "verified_source_pins": pins,
            "runtime_root": runtime.as_posix(), "series_input": (Path(root) / SERIES_INPUT).as_posix(),
            "raw_input_root": raw.as_posix(), "raw_runtime_root": (runtime / "raw").as_posix(),
            "verified_bounded_metadata": {Path(k).as_posix(): v for k, v in bounded.items()},
            "inputs": inputs, "output": output.as_posix(), "raw_manifest_present": origin.exists(),
            "input_content_verified": False, "environment_verified": False, "required_environment": ENVIRONMENT,
            "prepared_row_identity_verified": False, "full_prepared_history_parity_verified": False,
            "scope": "preprocessing only; no model inference, training or calibration"}


def check_inputs(request):
    for item in request["inputs"]:
        path = safe_path(item["path"])
        actual = digest(path)
        if path.stat().st_size != item["bytes"] or (item["expected_sha256"] is not None and actual != item["expected_sha256"]):
            raise ValueError(f"Input SHA256/size mismatch: {path}")
        # Newly observed rebuilt container identity, never substituted for historical identity.
        if item.get("observed_sha256", actual) != actual:
            raise ValueError("Rebuilt input drift")
        item["observed_sha256"] = actual
        item["content_verified"] = True
    for path, expected in request["verified_bounded_metadata"].items():
        if hashlib.sha256(bounded_bytes(Path(path))).hexdigest() != expected:
            raise ValueError("Metadata drift")
    raw = Path(request["raw_input_root"])
    if (raw / "manifest.json").exists() != request["raw_manifest_present"]:
        raise ValueError("Raw manifest presence drift")


def run_child(argv, timeout, **kwargs):
    # run() kills and waits on timeout and exceptions, including KeyboardInterrupt.
    return subprocess.run(argv, check=True, timeout=timeout, **kwargs)


def execute(stage, raw, prepared, output, python, basic=None, root=ROOT, timeout_seconds=3600, input_mode="historical"):
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    request = plan(stage, raw, prepared, output, python, basic, root, input_mode)
    check_inputs(request)
    probe = "import importlib.metadata as m,json,platform; print(json.dumps(dict(python=platform.python_version(), **{k:m.version(k) for k in ['numpy','catboost','scipy']})))"
    result = run_child([request["argv"][0], "-I", "-B", "-c", probe], min(60, timeout_seconds), capture_output=True, text=True)
    if json.loads(result.stdout) != ENVIRONMENT:
        raise ValueError("Selected interpreter is not the pinned RM environment")
    if input_mode == "rebuild":
        # Only pinned metadata/row readers, never a trainer or model loader.
        reader = ("import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                  "import kion_sex_features as s; import kion_local_features as u; "
                  "p=Path(sys.argv[2]); m,_=s.read_metadata(p/'metadata.json'); "
                  "print(json.dumps({n:u.read_split(p/(n+'.npz'),m,n)[1] for n in ['train','val']}))")
        verify_sources(root)
        result = run_child([request["argv"][0], "-I", "-B", "-c", reader, str(Path(request["source_root"]) / "scripts"), str(prepared)], timeout_seconds, capture_output=True, text=True)
        facts = json.loads(result.stdout)
        for name, expected in ROW_FACTS.items():
            if any(facts.get(name, {}).get(k) != v for k, v in expected.items()):
                raise ValueError("Rebuilt prepared row identity mismatch")
        request["prepared_row_identity_verified"] = True
    verify_sources(root)
    check_inputs(request)
    # Parent exists; sibling receipt can survive partial child output and is exclusive.
    receipt = Path(request["output"] + ".prepare-launch.json")
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("Receipt already exists")
    if Path(request["output"]).exists():
        raise ValueError("Output appeared before dispatch")
    request.update(mode="execute-prepare", status="started", environment_verified=True, input_content_verified=True)
    def write_receipt(mode="w"):
        with receipt.open(mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
    write_receipt("x")
    try:
        stage_runtime(request, root)
        check_runtime(request)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        run_child(request["argv"], timeout_seconds, env=env, cwd=request["runtime_root"])
        check_runtime(request)
        verify_sources(root)
        check_inputs(request)
        manifest, _ = read_json(Path(request["output"]) / "manifest.json",
                                limit=RM_MANIFEST_LIMIT if stage == "reward-data" else LIMIT)
        if stage == "reward-data":
            validate_reward_manifest(manifest)
            files = dict(manifest["files"], **manifest["sidecars"])
        else:
            prepared_hashes = {Path(item["path"]).stem: item["observed_sha256"] for item in request["inputs"] if Path(item["path"]).name in ("train.npz", "val.npz")}
            files = {"features.npz": validate_basic(manifest, input_mode, prepared_hashes)}
        for name, expected in files.items():
            if digest(safe_path(Path(request["output"]) / name)) != expected:
                raise ValueError("Output checksum mismatch")
        request.update(status="source-cli-completed-output-checksums-verified", semantic_output_equivalence_verified=False)
    except BaseException:
        request.update(status="failed-or-cancelled", semantic_output_equivalence_verified=False)
        write_receipt()
        raise
    write_receipt()
    return request


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("basic-features", "reward-data"))
    for name in ("raw", "prepared", "output", "python"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--basic", type=Path)
    parser.add_argument("--input-mode", choices=("historical", "rebuild"), default="historical")
    parser.add_argument("--execute-prepare", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args(argv)
    try:
        kwargs = dict(stage=args.stage, raw=args.raw, prepared=args.prepared, output=args.output, python=args.python, basic=args.basic, input_mode=args.input_mode)
        result = execute(**kwargs, timeout_seconds=args.timeout_seconds) if args.execute_prepare else plan(**kwargs)
    except (ValueError, OSError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f"prepare_reward_data: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
