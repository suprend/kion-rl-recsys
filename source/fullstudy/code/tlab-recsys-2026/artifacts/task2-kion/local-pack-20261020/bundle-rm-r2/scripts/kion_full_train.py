"""Resumable, from-scratch GPT-2 next-item training on prepared KION splits.

Test is opt-in and evaluated once using the validation-selected checkpoint.
Benchmark runs require their own output directory and never write checkpoints.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kion_train import NextItemGPT, FIRST_ITEM, PAD, known_targets


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path, vocab, context):
    # Reward and demographic arrays are irrelevant here and may be very large.
    with np.load(path, allow_pickle=False) as source:
        data = {key: source[key] for key in ("x", "lengths", "y")}
    x, lengths, y = data["x"], data["lengths"], data["y"]
    if x.ndim != 2 or x.shape[1] != context or lengths.shape != (len(x),) or y.shape != (len(x),):
        raise ValueError(f"{path}: inconsistent shapes")
    if any(value.dtype != np.int64 for value in data.values()):
        raise ValueError(f"{path}: expected int64 tokens and lengths")
    if not len(x) or np.any(lengths < 1) or np.any(lengths > context):
        raise ValueError(f"{path}: empty split or invalid lengths")
    for start in range(0, len(x), 16384):
        block, size = x[start:start + 16384], lengths[start:start + 16384]
        active = np.arange(context)[None, :] < size[:, None]
        if np.any(block < 0) or np.any(block >= vocab) or np.any(block[~active] != PAD) or np.any(block[active] == PAD):
            raise ValueError(f"{path}: invalid tokens or right padding")
    if np.any(y < 0) or np.any(y >= vocab):
        raise ValueError(f"{path}: invalid targets")
    return data


def batch(data, indices, device):
    lengths = data["lengths"][indices]
    width = int(lengths.max())
    return (torch.as_tensor(data["x"][indices, :width], device=device),
            torch.as_tensor(lengths, device=device),
            torch.as_tensor(data["y"][indices], device=device))


def groups(indices, microbatch, accumulation, start=0):
    """Each group is one optimizer step, including a correctly weighted tail."""
    size = microbatch * accumulation
    for offset in range(start, len(indices), size):
        group = indices[offset:offset + size]
        yield offset, [group[n:n + microbatch] for n in range(0, len(group), microbatch)]


def epoch_order(eligible, lengths, seed, epoch, microbatch, bucket_size):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    order = rng.permutation(eligible)
    if not bucket_size:
        return order
    # Random pools avoid a fixed length curriculum; shuffle resulting batches.
    # Pool size is a microbatch multiple, so only the final batch is partial.
    pool_size = max(microbatch, bucket_size // microbatch * microbatch)
    batches = []
    for start in range(0, len(order), pool_size):
        pool = order[start:start + pool_size]
        pool = pool[np.argsort(lengths[pool], kind="stable")]
        batches.extend(pool[n:n + microbatch] for n in range(0, len(pool), microbatch))
    complete = [indices for indices in batches if len(indices) == microbatch]
    tail = [indices for indices in batches if len(indices) != microbatch]
    rng.shuffle(complete)
    return np.concatenate(complete + tail)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def update(model, data, microbatches, optimizer, scaler, device, clip=1.0):
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(9):
        if attempt:
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        try:
            return _update_attempt(model, data, microbatches, optimizer, scaler, device, clip)
        except GradientOverflow:
            if not scaler.is_enabled() or attempt == 8:
                raise FloatingPointError("Non-finite gradients persisted after AMP scale backoff")
            scaler.update(new_scale=scaler.get_scale() / 2)


class GradientOverflow(Exception):
    pass


def _update_attempt(model, data, microbatches, optimizer, scaler, device, clip):
    optimizer.zero_grad(set_to_none=True)
    count, total_loss = sum(map(len, microbatches)), 0.0
    for indices in microbatches:
        x, lengths, y = batch(data, indices, device)
        with autocast(device):
            logits = model(x, lengths)
            loss = F.cross_entropy(logits[:, FIRST_ITEM:].float(), y - FIRST_ITEM, reduction="sum") / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; no checkpoint advanced")
        scaler.scale(loss).backward()
        total_loss += float(loss.detach())
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    # Retry this exact group with a smaller scale, without advancing its cursor.
    if not torch.isfinite(norm):
        optimizer.zero_grad(set_to_none=True)
        raise GradientOverflow()
    scaler.step(optimizer)
    scaler.update()
    return total_loss


@torch.no_grad()
def evaluate(model, data, vocab, batch_size, device, indices=None):
    model.eval()
    indices = np.arange(len(data["y"])) if indices is None else indices
    hits = ndcg = ce = 0.0
    known_count = unknown = 0
    k = min(10, vocab - FIRST_ITEM)
    for start in range(0, len(indices), batch_size):
        x, lengths, y = batch(data, indices[start:start + batch_size], device)
        with autocast(device):
            logits = model(x, lengths)[:, FIRST_ITEM:].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite evaluation logits")
        known = (y >= FIRST_ITEM) & (y < vocab)
        known_count += int(known.sum())
        unknown += int((~known).sum())
        if known.any():
            ce += float(F.cross_entropy(logits[known], y[known] - FIRST_ITEM, reduction="sum"))
        top = logits.topk(k, dim=1).indices + FIRST_ITEM
        match = top.eq(y[:, None]) & known[:, None]
        hits += float(match.any(dim=1).sum())
        discounts = 1 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)
        ndcg += float((match * discounts).sum())
    return {"examples": len(indices), "known_targets": known_count, "unknown_target_misses": unknown,
            "cross_entropy_known": ce / known_count if known_count else None,
            "recall@10": hits / len(indices), "ndcg@10": ndcg / len(indices), "actual_k": k}


def validate_resume(saved, current):
    if saved["contract"] != current["contract"]:
        raise ValueError("Resume config mismatch: architecture, data, source, or training contract differs")
    if saved.get("runtime") != current.get("runtime"):
        raise ValueError("Resume runtime mismatch: library versions, device, GPU, or CPU threads differ")


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def checkpoint(path, model, optimizer, scaler, state, config):
    temporary = path.with_suffix(".tmp")
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "state": state, "rng": rng_state(), "config": config}, temporary)
    os.replace(temporary, path)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--device", default="auto")
    for name, default in (("epochs", 5), ("batch-size", 32), ("accumulation", 4), ("hidden-size", 768),
                          ("layers", 12), ("heads", 12), ("seed", 2026), ("patience", 2),
                          ("validation-limit", 0), ("benchmark-steps", 0), ("benchmark-warmup", 5),
                          ("max-updates", 0), ("checkpoint-updates", 1000), ("log-updates", 100), ("bucket-size", 8192), ("cpu-threads", 4)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--warmup-ratio", type=float, default=.05)
    result.add_argument("--weight-decay", type=float, default=.01)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--evaluate-test", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("epochs", "batch_size", "accumulation", "hidden_size", "layers", "heads", "patience", "checkpoint_updates", "log_updates", "cpu_threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.hidden_size % args.heads or not 0 <= args.warmup_ratio < 1 or args.lr <= 0:
        raise ValueError("Invalid architecture or learning rate")
    if min(args.validation_limit, args.benchmark_steps, args.benchmark_warmup, args.max_updates, args.bucket_size) < 0:
        raise ValueError("Limits must be nonnegative")
    if args.benchmark_steps and (args.resume or args.evaluate_test):
        raise ValueError("Benchmark is isolated from resume and test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output.iterdir()):
        raise ValueError("New run requires an empty output directory")
    metadata = json.loads((args.data / "metadata.json").read_text(encoding="utf-8"))
    vocab, context = metadata["vocab_size"], metadata["context"]
    inputs = {name: sha256(args.data / name) for name in ("metadata.json", "train.npz", "val.npz")}
    sources = {name: sha256(Path(__file__).with_name(name)) for name in ("kion_full_train.py", "kion_train.py", "kion_prepare.py")}
    import transformers
    contract = {"architecture": {"model": "GPT2Model from scratch", "hidden_size": args.hidden_size, "layers": args.layers,
                                "heads": args.heads, "context_length": context, "vocab_size": vocab, "tied_embeddings": True},
                "training": {key: getattr(args, key) for key in ("epochs", "batch_size", "accumulation", "lr", "warmup_ratio", "weight_decay", "seed", "patience", "validation_limit", "bucket_size")},
                "inputs_sha256": inputs, "sources_sha256": sources,
                "objective": "next known item CE over full train catalog; unknown validation targets are ranking misses",
                "checkpoint_selection": "validation NDCG@10; test only by explicit --evaluate-test"}
    config = {"contract": contract, "data": args.data.resolve().as_posix(), "benchmark": bool(args.benchmark_steps),
              "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                          "cuda": torch.version.cuda, "device": str(device),
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "cpu_threads": args.cpu_threads, "torch_num_threads": torch.get_num_threads()}}
    if args.resume:
        old_config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        validate_resume(old_config, config)
        if old_config.get("benchmark"):
            raise ValueError("Cannot resume benchmark output")
        config = old_config
    else:
        write_json(args.output / "config.json", config)
    train, val = (load_data(args.data / (name + ".npz"), vocab, context) for name in ("train", "val"))
    eligible = np.flatnonzero(known_targets(train["y"], vocab))
    if not len(eligible):
        raise ValueError("No known training targets")
    val_indices = np.arange(len(val["y"]))
    if args.validation_limit and args.validation_limit < len(val_indices):
        val_indices = np.sort(np.random.default_rng(args.seed).choice(val_indices, args.validation_limit, replace=False))
    model = NextItemGPT(vocab, context, args.hidden_size, args.layers, args.heads).to(device)
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    if not args.resume:
        write_json(args.output / "config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    state = {"epoch": 0, "offset": 0, "updates": 0, "examples": 0, "best_ndcg": -1.0, "bad_epochs": 0, "history": []}
    if args.resume:
        saved = torch.load(args.output / "latest.pt", map_location=device, weights_only=False)
        validate_resume(saved["config"], config)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        restore_rng(saved["rng"])
    total_updates = math.ceil(len(eligible) / (args.batch_size * args.accumulation)) * args.epochs
    warmup = max(1, int(total_updates * args.warmup_ratio))
    started = time.perf_counter()
    initial_examples, initial_updates = state["examples"], state["updates"]
    benchmark_started = None
    benchmark_examples = 0

    def emit(event, **values):
        synchronize(device)
        elapsed = time.perf_counter() - started
        done = state["examples"] - initial_examples
        speed = done / elapsed if elapsed else 0
        payload = {"event": event, **state, "elapsed_seconds": elapsed, "examples_per_second": speed,
                   "eta_seconds": max(0, (total_updates - state["updates"]) * args.batch_size * args.accumulation / speed) if speed else None,
                   "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0, **values}
        write_json(args.output / "progress.json", payload)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    emit("started", status="running", parameters=config["parameters"], train_examples=len(eligible), validation_examples=len(val_indices))
    stop_early = state["bad_epochs"] >= args.patience
    capped = bool(args.max_updates and state["updates"] >= args.max_updates)
    for epoch in range(state["epoch"], args.epochs):
        if stop_early or capped:
            break
        order = epoch_order(eligible, train["lengths"], args.seed, epoch, args.batch_size, args.bucket_size)
        synchronize(device)
        epoch_started = time.perf_counter()
        epoch_initial_examples = state["examples"]
        resumed_segment = state["offset"] > 0
        model.train()
        for offset, microbatches in groups(order, args.batch_size, args.accumulation, state["offset"]):
            step = state["updates"]
            factor = (step + 1) / warmup if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor
            if args.benchmark_steps and step == args.benchmark_warmup:
                synchronize(device)
                benchmark_started = time.perf_counter()
                benchmark_examples = state["examples"]
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
            loss = update(model, train, microbatches, optimizer, scaler, device)
            state.update(updates=step + 1, offset=offset + sum(map(len, microbatches)), examples=state["examples"] + sum(map(len, microbatches)))
            if state["updates"] % args.log_updates == 0 or step == initial_updates:
                emit("training", status="running", loss=loss, learning_rate=optimizer.param_groups[0]["lr"])
            if args.benchmark_steps and state["updates"] >= args.benchmark_steps + args.benchmark_warmup:
                synchronize(device)
                seconds = time.perf_counter() - benchmark_started
                result = {"measured_updates": args.benchmark_steps, "warmup_updates": args.benchmark_warmup,
                          "seconds": seconds, "examples_per_second": (state["examples"] - benchmark_examples) / seconds,
                          "seconds_per_update": seconds / args.benchmark_steps,
                          "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0}
                write_json(args.output / "benchmark.json", result)
                emit("benchmark_complete", status="benchmark_complete", benchmark=result)
                return result
            if not args.benchmark_steps and state["updates"] % args.checkpoint_updates == 0:
                checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config)
            if args.max_updates and state["updates"] >= args.max_updates:
                capped = True
                break
        if capped:
            if not args.benchmark_steps:
                checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config)
            break
        if args.benchmark_steps:
            state.update(epoch=epoch + 1, offset=0)
            continue
        synchronize(device)
        train_wall = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        metrics = evaluate(model, val, vocab, args.batch_size, device, val_indices)
        synchronize(device)
        state["history"].append({"epoch": epoch + 1, "validation": metrics,
                                 "train_wall_seconds_including_periodic_checkpoints_and_logs": train_wall,
                                 "train_timing_is_resumed_segment": resumed_segment,
                                 "train_examples_per_second": (state["examples"] - epoch_initial_examples) / train_wall,
                                 "validation_wall_seconds": time.perf_counter() - validation_started})
        improved = metrics["ndcg@10"] > state["best_ndcg"]
        state.update(epoch=epoch + 1, offset=0, bad_epochs=0 if improved else state["bad_epochs"] + 1)
        if improved:
            state["best_ndcg"] = metrics["ndcg@10"]
            checkpoint(args.output / "best.pt", model, optimizer, scaler, state, config)
        checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config)
        emit("validation", status="running", validation=metrics)
        stop_early = state["bad_epochs"] >= args.patience
    if args.benchmark_steps:
        emit("benchmark_incomplete", status="incomplete", reason="not enough epoch updates for requested benchmark")
        return
    status = "incomplete" if capped else "completed"
    test_metrics_path = args.output / "test_metrics.json"
    saved_test = json.loads(test_metrics_path.read_text(encoding="utf-8")) if test_metrics_path.exists() else None
    result = {"status": status, "early_stopped": stop_early, "state": state, "test": saved_test}
    # Recover finalization after a saved test result, without another test pass.
    if args.evaluate_test and not capped and saved_test is None:
        best = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["state_dict"])
        # Explicit opt-in is the sole path that accesses the test split.
        test_path = args.data / "test.npz"
        test = load_data(test_path, vocab, context)
        result["test"] = {"input_sha256": sha256(test_path), "metrics": evaluate(model, test, vocab, args.batch_size, device),
                          "selected_epoch": best["state"]["epoch"]}
        write_json(args.output / "test_metrics.json", result["test"])
    write_json(args.output / "metrics.json", result)
    emit("finished", status=status, early_stopped=stop_early)
    return result


if __name__ == "__main__":
    main()
