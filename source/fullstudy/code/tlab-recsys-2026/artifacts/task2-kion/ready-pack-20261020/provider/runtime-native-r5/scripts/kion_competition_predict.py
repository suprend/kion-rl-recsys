"""Export the selected full seven-group GPT to the official KION user list.

Inference keeps the trained vocabulary, tied item head, feature matrix and masks.
Ranking is the unfiltered model top ten over tokens >= 3. In particular, repeated
history items remain eligible and later public-history OOV items stay UNK.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

CHECKPOINT_SHA256 = "817b2aecba272b743f8c3094a6d61e2225210dc6ea09f590ae0957c84ce4fdf9"
GROUP_NAMES = ("age", "sex", "genres", "content_type", "release_year", "countries", "age_rating")
SOURCE_NAMES = ("kion_conditioned_train.py", "kion_full_train.py", "kion_train.py", "kion_prepare.py",
                "kion_basic_bundle_features.py", "kion_local_features.py", "kion_sex_features.py")
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "UNK": 2}
FIRST_ITEM, TOP_K = 3, 10


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode("utf-8")).hexdigest()


def array_hash(value):
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)},
                        sort_keys=True, separators=(",", ":")).encode("ascii")
    digest = hashlib.sha256(header + b"\n")
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def strict_id(text, label):
    if not isinstance(text, str) or not text or not text.isascii() or not text.isdecimal():
        raise ValueError(f"Invalid {label}: expected a canonical nonnegative integer")
    result = int(text)
    if str(result) != text or result > np.iinfo(np.int64).max:
        raise ValueError(f"Invalid {label}: noncanonical or out of int64 range")
    return result


def read_sample(path):
    """The published sample uses user_id,item_id; ODS also accepts Id,Predicted."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (["Id", "Predicted"], ["user_id", "item_id"]):
            raise ValueError("Official sample header must be Id,Predicted or user_id,item_id")
        key = reader.fieldnames[0]
        users = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed official sample CSV row")
            users.append(strict_id(row[key], "sample user ID"))
    values = np.asarray(users, dtype=np.int64)
    if not len(values) or len(np.unique(values)) != len(values):
        raise ValueError("Official sample users must be nonempty and unique")
    return values


def token_lookup(metadata):
    vocab = metadata.get("vocab_size")
    if (type(vocab) is not int or vocab < FIRST_ITEM + TOP_K
            or metadata.get("special_tokens") != SPECIAL_TOKENS):
        raise ValueError("Invalid fixed vocabulary or special tokens")
    mapping = metadata.get("item_id_to_token")
    if not isinstance(mapping, dict) or len(mapping) != vocab - FIRST_ITEM:
        raise ValueError("Mapping must cover every real item token")
    lookup = np.full(vocab, -1, np.int64)
    for raw_id, token in mapping.items():
        item_id = strict_id(raw_id, "raw item ID")
        if type(token) is not int or not FIRST_ITEM <= token < vocab or lookup[token] != -1:
            raise ValueError("Mapping must be bijective over real tokens >= 3")
        lookup[token] = item_id
    if np.any(lookup[FIRST_ITEM:] < 0) or len(np.unique(lookup[FIRST_ITEM:])) != len(mapping):
        raise ValueError("Incomplete or ambiguous raw item mapping")
    return lookup


def validate_architecture(config, metadata):
    architecture = config["contract"]["architecture"]
    if (architecture.get("model") != "GPT2Model from scratch" or architecture.get("tied_embeddings") is not True
            or architecture.get("feature_mode") != "basic-bundle"
            or architecture.get("bundle_contract") != "kion-basic-bundle-v1"
            or architecture.get("vocab_size") != metadata["vocab_size"]
            or architecture.get("context_length") != metadata["context"]
            or metadata["context"] != 64
            or architecture.get("user_feature_dim") != 10 or architecture.get("item_feature_dim") != 197
            or architecture.get("dropped_feature_groups") != []
            or architecture.get("feature_group_mask") != {name: 1 for name in GROUP_NAMES}
            or tuple(architecture.get("feature_groups", {})) != GROUP_NAMES):
        raise ValueError("Only the unchanged full seven-group context64 model may be exported")
    for name in ("hidden_size", "layers", "heads"):
        if type(architecture.get(name)) is not int or architecture[name] < 1:
            raise ValueError("Invalid selected model architecture")
    if architecture["hidden_size"] % architecture["heads"]:
        raise ValueError("Selected hidden size must divide into attention heads")
    return architecture


def validate_cache_arrays(arrays, users, vocab, context):
    if set(arrays) != {"x", "lengths", "user_ids", "user_features"}:
        raise ValueError("Inference cache must contain exactly four named arrays")
    n = len(users)
    expected = {"x": ((n, context), np.int64), "lengths": ((n,), np.int64),
                "user_ids": ((n,), np.int64), "user_features": ((n, 10), np.float32)}
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Invalid inference cache shape/dtype: {name}")
    if not np.array_equal(arrays["user_ids"], users):
        raise ValueError("Inference cache users/order differ from official sample")
    lengths = arrays["lengths"]
    if np.any(lengths < 1) or np.any(lengths > context):
        raise ValueError("Invalid inference context lengths")
    for start in range(0, n, 16384):
        block = arrays["x"][start:start + 16384]
        active = np.arange(context)[None, :] < lengths[start:start + 16384, None]
        if (np.any(block < 0) or np.any(block >= vocab) or np.any(block[:, 0] != 1)
                or np.any(block[:, 1:] == 1) or np.any(block[~active] != 0) or np.any(block[active] == 0)):
            raise ValueError("Invalid inference tokens, BOS placement, or right padding")
    profile = arrays["user_features"]
    if (not np.isfinite(profile).all() or np.any((profile != 0) & (profile != 1))
            or np.any(profile[:, :7].sum(1) != 1) or np.any(profile[:, 7:].sum(1) != 1)):
        raise ValueError("Inference profile must contain finite one-hot age7 and sex3")


@contextmanager
def frozen_modules(directory, pins):
    """Load reviewed files without reusing similarly named working-tree modules."""
    directory = Path(directory).resolve()
    if set(pins) != set(SOURCE_NAMES):
        raise ValueError("Selected model must pin exactly its seven model/helper sources")
    for name, digest in pins.items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Frozen model source checksum mismatch: {name}")
    names = [Path(name).stem for name in SOURCE_NAMES]
    previous = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    previous_path = sys.path[:]
    try:
        sys.path.insert(0, str(directory))
        model_module = importlib.import_module("kion_conditioned_train")
        builder = importlib.import_module("kion_basic_bundle_features")
        for name in names:
            module = sys.modules.get(name)
            if module is not None and Path(module.__file__).resolve() != directory / (name + ".py"):
                raise ValueError(f"Frozen module lookup escaped source directory: {name}")
        yield model_module, builder
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)
        sys.path[:] = previous_path


def preflight(*, data, sample, metadata, checkpoint, config, bundle_scripts, item_features,
              expected_checkpoint_sha256=CHECKPOINT_SHA256):
    paths = {"data": Path(data).resolve(), "sample": Path(sample).resolve(),
             "metadata": Path(metadata).resolve(), "checkpoint": Path(checkpoint).resolve(),
             "config": Path(config).resolve(), "bundle_scripts": Path(bundle_scripts).resolve(),
             "item_features": Path(item_features).resolve()}
    paths.update(cache=paths["data"] / "inference.npz", manifest=paths["data"] / "manifest.json")
    hashes = {name: sha256(path) for name, path in paths.items() if name not in ("data", "bundle_scripts")}
    if hashes["checkpoint"] != expected_checkpoint_sha256:
        raise ValueError("Checkpoint is not the frozen selected epoch0005 checkpoint")
    receipt = read_json(paths["checkpoint"].with_suffix(".json"))
    if receipt.get("sha256") != hashes["checkpoint"] or receipt.get("epoch") != 5 or receipt.get("offset") != 0:
        raise ValueError("Selected checkpoint receipt/hash/epoch mismatch")
    config_value, metadata_value, manifest = (read_json(paths[name]) for name in ("config", "metadata", "manifest"))
    lookup = token_lookup(metadata_value)
    architecture = validate_architecture(config_value, metadata_value)
    contract = config_value["contract"]
    if contract["inputs_sha256"]["metadata.json"] != hashes["metadata"]:
        raise ValueError("Training metadata checksum differs from selected model")
    if contract["features_sha256"]["sidecar"] != hashes["item_features"]:
        raise ValueError("Training item feature file checksum differs from selected model")
    # Cache provenance is recorded by the separate history builder. Never reopen
    # raw public interactions here or fit/update a dictionary during inference.
    if (manifest.get("format_version") != 1
            or manifest.get("builder_contract") != "kion-competition-contexts-v1"
            or manifest.get("kind") != "competition_inference" or manifest.get("context") != 64
            or manifest.get("cutoff_exclusive") != "2021-08-23"
            or manifest.get("special_tokens") != SPECIAL_TOKENS
            or manifest.get("vocab_size") != metadata_value["vocab_size"]
            or manifest.get("item_id_to_token_sha256") != canonical_hash(metadata_value["item_id_to_token"])):
        raise ValueError("Competition cache contract/vocabulary/cutoff mismatch")
    for name, path_key in (("metadata", "metadata"), ("model_config", "config"), ("sample", "sample")):
        if manifest["inputs"][name]["sha256"] != hashes[path_key]:
            raise ValueError(f"Competition cache input binding mismatch: {name}")
    users = read_sample(paths["sample"])
    with np.load(paths["cache"], allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    validate_cache_arrays(arrays, users, metadata_value["vocab_size"], metadata_value["context"])
    declaration = manifest["inference"]
    if (declaration.get("file") != "inference.npz" or declaration.get("sha256") != hashes["cache"]
            or set(declaration.get("arrays", {})) != set(arrays)
            or manifest.get("sample_user_ids_sha256") != array_hash(users)):
        raise ValueError("Competition cache artifact/sample row hash mismatch")
    for name, values in arrays.items():
        declared = declaration["arrays"][name]
        if (declared.get("shape") != list(values.shape) or declared.get("dtype") != str(values.dtype)
                or declared.get("sha256") != array_hash(values)):
            raise ValueError(f"Competition cache array declaration/hash mismatch: {name}")
    groups = architecture["feature_groups"]
    if manifest.get("user_feature_groups") != {name: groups[name] for name in ("age", "sex")}:
        raise ValueError("Competition cache named user feature groups differ from selected model")
    with np.load(paths["item_features"], allow_pickle=False) as source:
        items = source["item_features"]
    if (items.dtype != np.float32 or items.shape != (len(lookup), 197)
            or array_hash(items) != architecture["feature_array_hashes"]["item_features"]):
        raise ValueError("Item features differ from the exact trained 197-dimensional matrix")
    with frozen_modules(paths["bundle_scripts"], contract["sources_sha256"]) as (_, builder):
        builder.validate_bundle_layout(arrays["user_features"], items, architecture["feature_groups"])
    return {"paths": paths, "hashes": hashes, "config": config_value, "metadata": metadata_value,
            "manifest": manifest, "arrays": arrays, "item_features": items, "lookup": lookup,
            "checkpoint_receipt": receipt}


def load_selected_model(plan, device):
    import torch
    config, architecture = plan["config"], plan["config"]["contract"]["architecture"]
    # Checkpoint contains legacy RNG/optimizer payloads; its immutable SHA and
    # sidecar are verified before this trusted local weights_only=False load.
    if sha256(plan["paths"]["checkpoint"]) != plan["hashes"]["checkpoint"]:
        raise ValueError("Checkpoint changed after export preflight")
    saved = torch.load(plan["paths"]["checkpoint"], map_location="cpu", weights_only=False)
    if saved.get("config") != config or saved.get("state", {}).get("epoch") != 5 or saved["state"].get("offset") != 0:
        raise ValueError("Checkpoint payload config/selected epoch differs from reviewed files")
    state = saved["state_dict"]
    for key, expected in (("bundle_user_mask", torch.ones(10)), ("bundle_item_mask", torch.ones(197)),
                          ("item_features", torch.from_numpy(plan["item_features"]))):
        if key not in state or not torch.equal(state[key], expected):
            raise ValueError(f"Checkpoint changed a frozen feature buffer: {key}")
    if not torch.equal(state["output.weight"], state["backbone.wte.weight"]):
        raise ValueError("Checkpoint tied ID head differs from its embedding matrix")
    with frozen_modules(plan["paths"]["bundle_scripts"], config["contract"]["sources_sha256"]) as (module, _):
        model = module.ConditionedGPT(architecture["vocab_size"], architecture["context_length"],
                    architecture["hidden_size"], architecture["layers"], architecture["heads"],
                    "basic-bundle", plan["item_features"], bundle_groups=architecture["feature_groups"],
                    drop_feature_groups=())
        model.load_state_dict(state, strict=True)
    if model.output.weight is not model.backbone.wte.weight:
        raise ValueError("Loaded model lost its tied item head")
    model.to(device).eval()
    return model


def rank_batch(model, arrays, start, stop, device, lookup):
    import torch
    lengths = arrays["lengths"][start:stop]
    width = int(lengths.max())
    with torch.inference_mode():
        x = torch.as_tensor(arrays["x"][start:stop, :width], device=device)
        size = torch.as_tensor(lengths, device=device)
        user = torch.as_tensor(arrays["user_features"][start:stop], device=device)
        # Match the trained runner's FP16 inference on CUDA and FP32 on CPU.
        from contextlib import nullcontext
        with torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
            logits = model(x, size, user_features=user)
        if logits.shape != (stop - start, len(lookup)) or not torch.isfinite(logits).all():
            raise ValueError("Invalid or non-finite inference logits")
        tokens = (logits[:, FIRST_ITEM:].float().topk(TOP_K, dim=1).indices + FIRST_ITEM).cpu().numpy()
    return tokens, lookup[tokens]


def validate_submission(path, sample, metadata):
    """Independent CSV reread: exact users/order, ten unique valid raw IDs."""
    users = read_sample(sample)
    metadata_value = read_json(metadata) if isinstance(metadata, (str, Path)) else metadata
    valid_items = set(metadata_value["item_id_to_token"])
    seen = set()
    count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["Id", "Predicted"]:
            raise ValueError("Submission header must be exactly Id,Predicted")
        for count, row in enumerate(reader, 1):
            if len(row) != 2 or count > len(users):
                raise ValueError("Malformed or extra submission row")
            user = strict_id(row[0], "submission user ID")
            if user in seen or user != users[count - 1]:
                raise ValueError("Submission users must match official sample exactly, in order")
            seen.add(user)
            items = row[1].split(",")
            if len(items) != TOP_K or len(set(items)) != TOP_K:
                raise ValueError("Each submission row needs exactly ten unique item IDs")
            for item in items:
                strict_id(item, "submission item ID")
                if item not in valid_items:
                    raise ValueError("Submission item is outside the fixed trained item vocabulary")
    if count != len(users):
        raise ValueError("Submission is missing official sample users")
    return {"valid": True, "rows": count, "items_per_row": TOP_K, "unique_users": len(seen),
            "user_order": "exact official sample order", "candidate_domain": "fixed trained real tokens >= 3"}


def parse_deadline(value):
    if value is None:
        return None
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Deadline must include its UTC offset")
    return result.astimezone(timezone.utc)


class ExportStopped(RuntimeError):
    pass


def check_control(stop_file=None, deadline=None):
    if stop_file is not None and Path(stop_file).exists():
        raise ExportStopped("supervisor stop file exists")
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise ExportStopped("supervisor UTC deadline reached")


def export(plan, output, *, device="cpu", batch_size=64, cpu_threads=4, stop_file=None, deadline=None):
    import torch
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Batch size and CPU threads must be positive")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Inference device must be explicit cpu or cuda")
    output = Path(output).resolve()
    # Preserve input trees and all existing results; the supervisor supplies a
    # fresh output directory for each attempt.
    for name, source in plan["paths"].items():
        if output == source or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError(f"Output must be separate from input path: {name}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Inference output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    n, done = len(plan["arrays"]["user_ids"]), 0

    def emit(event, status, **extra):
        elapsed = time.perf_counter() - started
        payload = {"event": event, "status": status, "rows": done, "total_rows": n,
                   "elapsed_seconds": elapsed, "rows_per_second": done / elapsed if elapsed else 0,
                   "utc": datetime.now(timezone.utc).isoformat(), **extra}
        write_json(output / "progress.json", payload)
        with (output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        check_control(stop_file, deadline)
        torch.set_num_threads(cpu_threads)
        emit("loading_model", "running", device=str(target))
        model = load_selected_model(plan, target)
        tokens, items = np.empty((n, TOP_K), np.int64), np.empty((n, TOP_K), np.int64)
        partial = output / "submission.csv.partial"
        with partial.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["Id", "Predicted"])
            for start in range(0, n, batch_size):
                check_control(stop_file, deadline)
                stop = min(n, start + batch_size)
                tokens[start:stop], items[start:stop] = rank_batch(model, plan["arrays"], start, stop, target, plan["lookup"])
                writer.writerows((str(user), ",".join(map(str, row)))
                                 for user, row in zip(plan["arrays"]["user_ids"][start:stop], items[start:stop]))
                stream.flush()
                done = stop
                if start == 0 or done == n or (start // batch_size) % 10 == 0:
                    emit("predicting", "running")
        check_control(stop_file, deadline)
        validation = validate_submission(partial, plan["paths"]["sample"], plan["metadata"])
        np.savez_compressed(output / "top10.npz", user_ids=plan["arrays"]["user_ids"], token_ids=tokens, item_ids=items)
        os.replace(partial, output / "submission.csv")
        import transformers
        report = {"format_version": 1, "status": "completed", "validation": validation,
            "selected_checkpoint": {"path": plan["paths"]["checkpoint"].as_posix(),
                "sha256": plan["hashes"]["checkpoint"], "receipt": plan["checkpoint_receipt"]},
            "config_sha256": plan["hashes"]["config"], "contract": plan["config"]["contract"],
            "inputs": {name: {"path": plan["paths"][name].as_posix(), "sha256": digest}
                       for name, digest in plan["hashes"].items()},
            "model_sources_sha256": plan["config"]["contract"]["sources_sha256"],
            "exporter": {"path": Path(__file__).resolve().as_posix(), "sha256": sha256(Path(__file__))},
            "cache_provenance": plan["manifest"],
            "ranking": {"k": TOP_K, "candidate_tokens": ">=3", "seen_item_mask": False,
                        "blending": False, "fallback": False, "mapping": "unchanged trained vocabulary",
                        "later_public_history_oov": "UNK; unavailable as output candidates"},
            "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                        "device": str(target), "cpu_threads": cpu_threads, "batch_size": batch_size,
                        "cuda_initialized": torch.cuda.is_initialized(),
                        "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None},
            "outputs": {name: {"path": (output / name).as_posix(), "sha256": sha256(output / name)}
                        for name in ("submission.csv", "top10.npz")},
            "elapsed_seconds": time.perf_counter() - started}
        write_json(output / "report.json", report)
        artifact_names = ["submission.csv", "top10.npz", "report.json"]
        write_json(output / "completion.json", {"state": "completed", "artifacts": artifact_names,
            "sha256": {name: sha256(output / name) for name in artifact_names}, "rows": n,
            "checkpoint_sha256": plan["hashes"]["checkpoint"], "selected_epoch": 5,
            "validation": validation})
        emit("finished", "completed", submission_sha256=report["outputs"]["submission.csv"]["sha256"])
        return report
    except ExportStopped as error:
        emit("stopped", "stopped", reason=str(error))
        raise
    except Exception as error:
        emit("failed", "failed", error=f"{type(error).__name__}: {error}")
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "sample", "metadata", "checkpoint", "config", "bundle-scripts", "item-features", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--cpu-threads", type=int, default=4)
    result.add_argument("--stop-file", type=Path)
    result.add_argument("--deadline-utc")
    result.add_argument("--preflight-only", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    deadline = parse_deadline(args.deadline_utc)
    check_control(args.stop_file, deadline)
    plan = preflight(**{name: getattr(args, name) for name in
        ("data", "sample", "metadata", "checkpoint", "config", "bundle_scripts", "item_features")})
    if args.preflight_only:
        summary = {"status": "preflight_passed", "rows": len(plan["arrays"]["user_ids"]), "hashes": plan["hashes"],
                   "sources_sha256": plan["config"]["contract"]["sources_sha256"]}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    return export(plan, args.output, device=args.device, batch_size=args.batch_size, cpu_threads=args.cpu_threads,
                  stop_file=args.stop_file, deadline=deadline)


if __name__ == "__main__":
    try:
        main()
    except ExportStopped as error:
        print(json.dumps({"status": "stopped", "reason": str(error)}), flush=True)
        raise SystemExit(3)
