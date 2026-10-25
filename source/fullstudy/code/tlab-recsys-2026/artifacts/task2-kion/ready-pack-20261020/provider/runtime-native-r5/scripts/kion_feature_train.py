"""Isolated genre-only history conditioning; output remains tied to item IDs.

Metadata is a static snapshot with unknown historical availability. Category
vocabularies were fitted on train; editorial popularity/recommendation tags are
excluded. No demographics or target outcomes enter this model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kion_train import NextItemGPT, FIRST_ITEM, PAD, last_real_hidden, known_targets
from kion_full_train import (parser as base_parser, load_data, groups, epoch_order,
                             update, evaluate, validate_resume, sha256, write_json,
                             checkpoint, restore_rng, synchronize)

EXCLUDED_GENRES = ("популярное", "рекомендуем")


def load_genres(path, metadata):
    categories = metadata["item_categories"]
    if list(categories) != ["content_type", "genres"]:
        raise ValueError("Expected prepared content_type then genres feature layout")
    names = categories["genres"]
    if not names or len(set(names)) != len(names):
        raise ValueError("Invalid genre category vocabulary")
    features = np.load(path, allow_pickle=False)
    offset = len(categories["content_type"])
    expected = (metadata["vocab_size"], offset + len(names))
    if features.shape != expected or features.dtype != np.float32:
        raise ValueError("Invalid prepared item feature shape or dtype")
    if not np.isfinite(features).all() or np.any((features != 0) & (features != 1)):
        raise ValueError("Item features must be finite binary multi-hot values")
    if np.any(features[:2]):
        raise ValueError("PAD and BOS item features must be zero")
    # Preparation assigns UNK the unknown category of each field. Accept only
    # that documented raw pattern, then suppress all special-token conditioning.
    unknown = np.zeros(features.shape[1], dtype=np.float32)
    try:
        unknown[categories["content_type"].index("__UNKNOWN__")] = 1
        unknown[offset + names.index("__UNKNOWN__")] = 1
    except ValueError as error:
        raise ValueError("Missing unknown item feature category") from error
    if not np.array_equal(features[2], unknown):
        raise ValueError("Invalid prepared UNK item feature pattern")
    genres = features[:, offset:].copy()
    genres[:FIRST_ITEM] = 0
    for tag in EXCLUDED_GENRES:
        if tag in names:
            genres[:, names.index(tag)] = 0
    return genres


class FeatureGPT(NextItemGPT):
    """History uses W_id + P_genre; candidate scoring uses tied W_id only."""
    def __init__(self, vocab_size, context_length, genre_features, hidden_size=256,
                 layers=4, heads=4, feature_mode="genres"):
        if feature_mode not in ("none", "genres"):
            raise ValueError("feature_mode must be none or genres")
        features = torch.as_tensor(genre_features, dtype=torch.float32).clone()
        if features.ndim != 2 or features.shape[0] != vocab_size or features.shape[1] < 1:
            raise ValueError("Invalid genre feature shape")
        if not torch.isfinite(features).all() or torch.any((features != 0) & (features != 1)):
            raise ValueError("Genre features must be finite binary values")
        if torch.any(features[:FIRST_ITEM]):
            raise ValueError("Special token genre features must be zero")
        super().__init__(vocab_size, context_length, hidden_size, layers, heads)
        self.feature_mode = feature_mode
        self.register_buffer("genre_features", features)
        # Keep baseline dropout RNG parity even though this adds parameters.
        with torch.random.fork_rng(devices=[]):
            self.genre_projection = nn.Linear(features.shape[1], hidden_size, bias=False)
            nn.init.normal_(self.genre_projection.weight, mean=0, std=.02)
        if feature_mode == "none":
            self.genre_projection.requires_grad_(False)

    def forward(self, x, lengths):
        if self.feature_mode == "none":
            return super().forward(x, lengths)
        embeddings = self.backbone.wte(x) + self.genre_projection(self.genre_features[x])
        hidden = self.backbone(inputs_embeds=embeddings, attention_mask=x.ne(PAD),
                               use_cache=False).last_hidden_state
        return self.output(last_real_hidden(hidden, lengths))


def parser():
    result = base_parser()
    result.description = __doc__
    result.set_defaults(batch_size=64, accumulation=2, hidden_size=256, layers=4, heads=4)
    result.add_argument("--feature-mode", choices=("none", "genres"), default="genres")
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
    inputs = {name: sha256(args.data / name) for name in ("metadata.json", "item_features.npy", "train.npz", "val.npz")}
    sources = {name: sha256(Path(__file__).with_name(name)) for name in ("kion_feature_train.py", "kion_full_train.py", "kion_train.py", "kion_prepare.py")}
    import transformers
    contract = {"architecture": {"model": "GPT2Model from scratch", "hidden_size": args.hidden_size, "layers": args.layers,
                                "heads": args.heads, "context_length": context, "vocab_size": vocab, "tied_embeddings": True, "feature_mode": args.feature_mode, "conditioning": "history ID + bias-free genre projection; tied ID output", "excluded_genres": list(EXCLUDED_GENRES), "metadata_availability": "static snapshot; historical availability not verified"},
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
    genre_features = load_genres(args.data / "item_features.npy", metadata)
    model = FeatureGPT(vocab, context, genre_features, args.hidden_size, args.layers, args.heads, args.feature_mode).to(device)
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
        if epoch == 0:
            checkpoint(args.output / "epoch1.pt", model, optimizer, scaler, state, config)
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
