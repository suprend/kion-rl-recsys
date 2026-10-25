"""Export ranked and greedy autoregressive KION lists from one frozen model.

Both modes exclude the complete public known-item history, independently of the
64-token context. Greedy generation appends predictions only to a private copy
of the prefix; the original histories, demographics and item features stay fixed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

BASE_SHA256 = "988915bc2cacdc844020e123f17a3f1a90924b78ce3f6a47c37ed0760119e4bf"
CHECKPOINT_SHA256 = "817b2aecba272b743f8c3094a6d61e2225210dc6ea09f590ae0957c84ce4fdf9"
MODES = ("ranked", "autoregressive")
TOP_K, FIRST_ITEM, CONTEXT = 10, 3, 64
EXPECTED_ROWS = 193113


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base(bundle_scripts):
    path = Path(bundle_scripts).resolve() / "kion_competition_predict.py"
    if sha256(path) != BASE_SHA256:
        raise ValueError("Immutable competition exporter checksum mismatch")
    return load_module(path, "kion_competition_variants_frozen_base")


class ExportStopped(RuntimeError):
    pass


def check_control(stop_file=None, deadline=None):
    if stop_file is not None and Path(stop_file).exists():
        raise ExportStopped("supervisor stop file exists")
    if deadline is not None and datetime.now(timezone.utc) >= deadline:
        raise ExportStopped("supervisor UTC deadline reached")


def parse_deadline(value):
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Deadline must include its UTC offset")
    return parsed.astimezone(timezone.utc)


def validate_seen_arrays(seen, users, vocab, contexts=None, lengths=None):
    """Validate CSR independently before using it to address candidate scores."""
    if set(seen) != {"user_ids", "indptr", "token_ids"}:
        raise ValueError("Seen cache must contain exactly user_ids, indptr, token_ids")
    n = len(users)
    for name, shape in (("user_ids", (n,)), ("indptr", (n + 1,))):
        if seen[name].dtype != np.int64 or seen[name].shape != shape:
            raise ValueError(f"Invalid seen cache shape/dtype: {name}")
    tokens, pointers = seen["token_ids"], seen["indptr"]
    if tokens.dtype != np.int64 or tokens.ndim != 1:
        raise ValueError("Invalid seen token dtype/shape")
    if not np.array_equal(seen["user_ids"], users):
        raise ValueError("Seen user order must equal the official sample")
    if (pointers[0] != 0 or pointers[-1] != len(tokens) or np.any(np.diff(pointers) < 0)
            or np.any(tokens < FIRST_ITEM) or np.any(tokens >= vocab)):
        raise ValueError("Invalid seen CSR pointers or real-item token domain")
    if np.any(vocab - FIRST_ITEM - np.diff(pointers) < TOP_K):
        raise ValueError("Fewer than ten unseen known candidates for a sample user")
    for row in range(n):
        values = tokens[pointers[row]:pointers[row + 1]]
        if np.any(np.diff(values) <= 0):
            raise ValueError("Seen tokens must be sorted and unique within each user")
        if contexts is not None:
            active = contexts[row, :int(lengths[row])]
            known = active[active >= FIRST_ITEM]
            if not np.isin(known, values).all():
                raise ValueError("Known context token missing from complete seen cache")


def preflight(*, data, seen_data, sample, metadata, checkpoint, config,
              bundle_scripts, item_features):
    base = load_base(bundle_scripts)
    plan = base.preflight(data=data, sample=sample, metadata=metadata,
        checkpoint=checkpoint, config=config, bundle_scripts=bundle_scripts,
        item_features=item_features, expected_checkpoint_sha256=CHECKPOINT_SHA256)
    architecture = plan["config"]["contract"]["architecture"]
    if (len(plan["arrays"]["user_ids"]) != EXPECTED_ROWS
            or tuple(architecture[name] for name in ("hidden_size", "layers", "heads")) != (256, 4, 4)):
        raise ValueError("Expected official 193113 rows and selected 4x256 four-head model")
    seen_module = load_module(Path(__file__).resolve().with_name("kion_competition_seen.py"),
                              "kion_competition_variants_seen")
    seen_plan = seen_module.preflight_seen(data=seen_data, sample=sample, metadata=metadata)
    seen = seen_plan["arrays"]
    validate_seen_arrays(seen, plan["arrays"]["user_ids"], len(plan["lookup"]),
                         plan["arrays"]["x"], plan["arrays"]["lengths"])
    # Both builders must use the identical public interaction source and cutoff.
    if (seen_plan["manifest"]["inputs"]["interactions"]["sha256"]
            != plan["manifest"]["inputs"]["interactions"]["sha256"]
            or seen_plan["manifest"]["cutoff_exclusive"] != plan["manifest"]["cutoff_exclusive"]):
        raise ValueError("Seen and context caches have different public interaction provenance")
    plan.update(base=base, seen=seen, seen_manifest=seen_plan["manifest"])
    plan["paths"].update(seen_data=Path(seen_data).resolve(),
        seen_cache=Path(seen_data).resolve() / "seen.npz",
        seen_manifest=Path(seen_data).resolve() / "manifest.json")
    plan["hashes"].update({name: sha256(plan["paths"][name]) for name in ("seen_cache", "seen_manifest")})
    for arrays in (plan["arrays"], seen):
        for values in arrays.values():
            values.flags.writeable = False
    plan["item_features"].flags.writeable = False
    return plan


def append_predictions(x, lengths, predictions):
    """BOS + last63 tokens; UNK participates in truncation like every item."""
    for row, token in enumerate(predictions):
        length = int(lengths[row])
        if length < x.shape[1]:
            x[row, length] = token
            lengths[row] += 1
        else:
            x[row, 1:-1] = x[row, 2:].copy()
            x[row, -1] = token


def predict_batch(model, arrays, seen, start, stop, device, lookup, *,
                  stop_file=None, deadline=None, progress=None):
    """Ten forwards, one initial logits tensor shared by ranked and AR modes.

    The candidate mask survives context truncation and retains all generated IDs.
    Only one bounded batch of full-vocabulary scores exists at a time.
    """
    import torch
    device = torch.device(device)
    if not 0 <= start < stop <= len(arrays["user_ids"]):
        raise ValueError("Invalid prediction batch bounds")
    x = np.array(arrays["x"][start:stop], copy=True)
    lengths = np.array(arrays["lengths"][start:stop], copy=True)
    profile = torch.as_tensor(np.array(arrays["user_features"][start:stop], copy=True), device=device)
    blocked = np.zeros((stop - start, len(lookup)), dtype=bool)
    blocked[:, :FIRST_ITEM] = True
    for row, source_row in enumerate(range(start, stop)):
        left, right = seen["indptr"][source_row:source_row + 2]
        blocked[row, seen["token_ids"][left:right]] = True
    if np.any((~blocked).sum(1) < TOP_K):
        raise ValueError("Fewer than ten unseen known candidates in batch")
    candidate_mask = torch.as_tensor(blocked, device=device)
    generated = np.empty((stop - start, TOP_K), np.int64)
    rows = torch.arange(stop - start, device=device)
    ranked = None
    with torch.inference_mode():
        for step in range(TOP_K):
            check_control(stop_file, deadline)
            width = int(lengths.max())
            # Clone the private CPU prefix too: a model cannot write through to
            # the temporary NumPy source while it is being appended afterwards.
            tokens = torch.as_tensor(x[:, :width].copy(), device=device)
            sizes = torch.as_tensor(lengths.copy(), device=device)
            with torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext():
                logits = model(tokens, sizes, user_features=profile)
            check_control(stop_file, deadline)
            if (logits.shape != (stop - start, len(lookup))
                    or not bool(torch.isfinite(logits).all())):
                raise ValueError("Invalid or non-finite inference logits")
            scores = logits.float().masked_fill(candidate_mask, -torch.inf)
            if step == 0:
                # Stable descending order means lower token IDs win ties, the
                # same tie rule as argmax. This also guarantees first-item parity.
                ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :TOP_K].cpu().numpy()
            selected = torch.argmax(scores, dim=1)
            generated[:, step] = selected.cpu().numpy()
            candidate_mask[rows, selected] = True
            if progress is not None:
                progress(step + 1)
            check_control(stop_file, deadline)
            if step + 1 < TOP_K:
                append_predictions(x, lengths, generated[:, step])
            del logits, scores
    if not np.array_equal(ranked[:, 0], generated[:, 0]):
        raise ValueError("Ranked and autoregressive first predictions differ")
    result = {"ranked_token_ids": ranked, "autoregressive_token_ids": generated}
    for mode in MODES:
        result[f"{mode}_item_ids"] = lookup[result[f"{mode}_token_ids"]]
    return result


def validate_submission(path, sample, metadata, seen, base):
    """Fresh CSV reread checks raw IDs, sample order, uniqueness and full seen."""
    validation = base.validate_submission(path, sample, metadata)
    users = base.read_sample(sample)
    validate_seen_arrays(seen, users, metadata["vocab_size"])
    mapping = metadata["item_id_to_token"]
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row, (_, predictions) in enumerate(reader):
            tokens = np.asarray([mapping[value] for value in predictions.split(",")], np.int64)
            left, right = seen["indptr"][row:row + 2]
            if np.isin(tokens, seen["token_ids"][left:right]).any():
                raise ValueError(f"Submission contains a publicly seen item at row {row}")
    return {**validation, "seen_filter": True, "seen_overlap": 0,
            "seen_scope": "all public known-item history before 2021-08-23"}


def export(plan, output, *, device="cpu", batch_size=64, cpu_threads=4,
           stop_file=None, deadline=None):
    import torch
    if batch_size < 1 or cpu_threads < 1:
        raise ValueError("Batch size and CPU threads must be positive")
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Inference device must be explicit cpu or cuda")
    if target.type == "cpu" and torch.cuda.is_initialized():
        raise RuntimeError("CPU export process must not initialize CUDA")
    output = Path(output).resolve()
    for name, source in plan["paths"].items():
        source = Path(source).resolve()
        if output == source or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError(f"Output must be separate from input path: {name}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Inference output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    base = plan["base"]
    started, started_utc, done = time.perf_counter(), datetime.now(timezone.utc).isoformat(), 0
    n = len(plan["arrays"]["user_ids"])

    def emit(event, status, **extra):
        elapsed = time.perf_counter() - started
        payload = {"event": event, "status": status, "rows": done, "total_rows": n,
            "elapsed_seconds": elapsed, "rows_per_second": done / elapsed if elapsed else 0,
            "utc": datetime.now(timezone.utc).isoformat(), **extra}
        base.write_json(output / "progress.json", payload)
        with (output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        check_control(stop_file, deadline)
        torch.set_num_threads(cpu_threads)
        emit("loading_model", "running", device=str(target), modes=list(MODES))
        model = base.load_selected_model(plan, target)
        check_control(stop_file, deadline)
        predictions = {f"{mode}_{kind}_ids": np.empty((n, TOP_K), np.int64)
                       for mode in MODES for kind in ("token", "item")}
        partials = {mode: output / f"{mode}.csv.partial" for mode in MODES}
        # Both streams are owned by this one export; never retain all-user logits.
        from contextlib import ExitStack
        with ExitStack() as stack:
            streams = {mode: stack.enter_context(path.open("x", encoding="utf-8", newline=""))
                       for mode, path in partials.items()}
            writers = {mode: csv.writer(stream, lineterminator="\n") for mode, stream in streams.items()}
            for writer in writers.values():
                writer.writerow(["Id", "Predicted"])
            for start in range(0, n, batch_size):
                check_control(stop_file, deadline)
                stop = min(n, start + batch_size)
                # First-batch step detail then one receipt per completed batch;
                # stop/deadline checks still occur on every generation step.
                batch = predict_batch(model, plan["arrays"], plan["seen"], start, stop, target, plan["lookup"],
                    stop_file=stop_file, deadline=deadline,
                    progress=(lambda step: emit("generating", "running", batch_start=start,
                                                batch_stop=stop, autoregressive_step=step)) if start == 0 else None)
                for key, values in batch.items():
                    predictions[key][start:stop] = values
                for mode, writer in writers.items():
                    writer.writerows((str(user), ",".join(map(str, row))) for user, row in
                        zip(plan["arrays"]["user_ids"][start:stop], batch[f"{mode}_item_ids"]))
                    streams[mode].flush()
                done = stop
                emit("batch_completed", "running")
        check_control(stop_file, deadline)
        validation = {mode: validate_submission(path, plan["paths"]["sample"], plan["metadata"],
                                                plan["seen"], base) for mode, path in partials.items()}
        if not np.array_equal(predictions["ranked_token_ids"][:, 0], predictions["autoregressive_token_ids"][:, 0]):
            raise ValueError("Export first-item parity check failed")
        np.savez_compressed(output / "top10.npz", user_ids=plan["arrays"]["user_ids"], **predictions)
        for mode, partial in partials.items():
            os.replace(partial, output / f"{mode}.csv")
        import transformers
        report = {"format_version": 1, "status": "completed", "modes": list(MODES), "seen_filter": True,
            "validation": {"valid": True, "rows": n, "first_item_parity": True, "modes": validation},
            "selected_checkpoint": {"path": plan["paths"]["checkpoint"].as_posix(),
                "sha256": plan["hashes"]["checkpoint"], "receipt": plan["checkpoint_receipt"]},
            "selected_epoch": 5, "contract": plan["config"]["contract"],
            "inputs": {name: {"path": plan["paths"][name].as_posix(), "sha256": digest}
                       for name, digest in plan["hashes"].items()},
            "exporter": {"path": Path(__file__).resolve().as_posix(), "sha256": sha256(__file__)},
            "frozen_base_exporter_sha256": BASE_SHA256,
            "model_sources_sha256": plan["config"]["contract"]["sources_sha256"],
            "cache_provenance": plan["manifest"], "seen_provenance": plan["seen_manifest"],
            "protocol": {"k": TOP_K, "candidate_tokens": ">=3", "all_public_seen_mask": True,
                "generated_duplicate_mask": "all previous generated tokens; retained across truncation",
                "ranked": "single initial forward; score descending, lower token wins ties",
                "autoregressive": "greedy argmax; append prediction; ten forwards without KV cache",
                "history": "temporary BOS + last63; original UNK retained until truncated",
                "demographics": "unchanged age7/sex3", "item_features": "unchanged trained 197 columns",
                "weights": "unchanged selected epoch5; tied candidate head",
                "training": False, "sampling": False, "blending": False, "fallback": False},
            "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                "device": str(target), "cpu_threads": cpu_threads, "batch_size": batch_size,
                "cuda_initialized": torch.cuda.is_initialized(),
                "gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else None},
            "clock": {"started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.perf_counter() - started},
            "outputs": {name: {"path": (output / name).as_posix(), "sha256": sha256(output / name)}
                        for name in ("ranked.csv", "autoregressive.csv", "top10.npz")}}
        base.write_json(output / "report.json", report)
        check_control(stop_file, deadline)
        artifact_names = ["ranked.csv", "autoregressive.csv", "top10.npz", "report.json"]
        base.write_json(output / "completion.json", {"state": "completed", "artifacts": artifact_names,
            "sha256": {name: sha256(output / name) for name in artifact_names}, "rows": n,
            "modes": list(MODES), "seen_filter": True, "selected_epoch": 5,
            "checkpoint_sha256": plan["hashes"]["checkpoint"], "validation": report["validation"]})
        emit("finished", "completed")
        return report
    except ExportStopped as error:
        emit("stopped", "stopped", reason=str(error))
        raise
    except Exception as error:
        emit("failed", "failed", error=f"{type(error).__name__}: {error}")
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "seen-data", "sample", "metadata", "checkpoint", "config",
                 "bundle-scripts", "item-features", "output"):
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
    plan = preflight(**{name: getattr(args, name) for name in ("data", "seen_data", "sample", "metadata",
                         "checkpoint", "config", "bundle_scripts", "item_features")})
    check_control(args.stop_file, deadline)
    if args.preflight_only:
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU preflight must not initialize CUDA")
        summary = {"status": "preflight_passed", "rows": len(plan["arrays"]["user_ids"]),
                   "modes": list(MODES), "seen_filter": True, "hashes": plan["hashes"],
                   "cuda_initialized": False, "sources_sha256": plan["config"]["contract"]["sources_sha256"]}
        print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)
        return summary
    return export(plan, args.output, device=args.device, batch_size=args.batch_size,
                  cpu_threads=args.cpu_threads, stop_file=args.stop_file, deadline=deadline)


if __name__ == "__main__":
    try:
        main()
    except ExportStopped as error:
        print(json.dumps({"status": "stopped", "reason": str(error)}), flush=True)
        raise SystemExit(3)
