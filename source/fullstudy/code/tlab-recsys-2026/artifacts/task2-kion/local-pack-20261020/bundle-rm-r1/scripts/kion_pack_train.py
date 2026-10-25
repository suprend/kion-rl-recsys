"""Dynamic local-feature GPT; train/validation only, with mmap user row features.

The inherited ID backbone, row order, CE denominator, LR schedule and selection
rule match trained7group. No test file or external model is opened by this CLI.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kion_full_train as legacy
import kion_conditioned_train as shared
from kion_train import NextItemGPT as IDGPT, BOS, PAD, FIRST_ITEM, known_targets, last_real_hidden

sha256, write_json = legacy.sha256, legacy.write_json
canonical_hash, array_hash = shared.canonical_hash, shared.array_hash
checkpoint, load_checkpoint = shared.checkpoint, shared.load_checkpoint
ROW_KEYS = shared.ROW_KEYS
ROW_HASH_KEYS = ("x", "lengths", "y", *ROW_KEYS)
ROWKEYS_ALGORITHM = "canonical-array-hashes:x,lengths,y,user_id,target_item_id,target_date:v1"


def make_row_keys_hash(data):
    return canonical_hash({key: array_hash(data[key]) for key in ROW_HASH_KEYS})


def _columns(columns, item_dim, user_dim):
    if columns is None:
        return {"item": [f"item_{i}" for i in range(item_dim)],
                "user": [f"user_{i}" for i in range(user_dim)]}
    if not isinstance(columns, dict) or set(columns) != {"item", "user"}:
        raise ValueError("feature_columns requires item and user name lists")
    for scope, dim in (("item", item_dim), ("user", user_dim)):
        names = columns[scope]
        if (not isinstance(names, list) or len(names) != dim
                or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != dim):
            raise ValueError(f"Invalid {scope} feature columns")
    return columns


class NextItemGPT(IDGPT):
    """Item projection on history tokens; row profile on the existing BOS only."""
    def __init__(self, vocab_size, context_length, hidden_size=256, layers=4, heads=4, *,
                 feature_mode="local-pack", item_features=None, user_feature_dim=0,
                 item_feature_dim=None, feature_columns=None, schema_hash=None):
        if feature_mode != "local-pack":
            raise ValueError("This runner supports feature_mode='local-pack' only")
        if isinstance(user_feature_dim, bool) or not isinstance(user_feature_dim, int) or user_feature_dim < 0:
            raise ValueError("user_feature_dim must be a nonnegative integer")
        if item_features is None:
            item_features = np.zeros((vocab_size, item_feature_dim or 0), np.float32)
        values = np.asarray(item_features)
        if (values.ndim != 2 or values.shape[0] != vocab_size or values.dtype.kind != "f"
                or not np.isfinite(values).all() or np.any(values[:FIRST_ITEM] != 0)):
            raise ValueError("Item features need finite float V x I with zero special rows")
        if item_feature_dim is not None and values.shape[1] != item_feature_dim:
            raise ValueError("Item feature dimension mismatch")
        self.feature_columns = _columns(feature_columns, values.shape[1], user_feature_dim)
        expected_hash = canonical_hash(self.feature_columns)
        if schema_hash is not None and schema_hash != expected_hash:
            raise ValueError("Feature schema hash does not match columns")
        super().__init__(vocab_size, context_length, hidden_size, layers, heads, tied=True)
        self.feature_mode = feature_mode
        self.item_feature_dim, self.user_feature_dim = values.shape[1], user_feature_dim
        self.schema_hash = expected_hash
        self.register_buffer("item_features", torch.tensor(values, dtype=torch.float32))
        # Added parameters cannot consume baseline initialization/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.user_feature_projection = nn.Linear(user_feature_dim, hidden_size, bias=False) if user_feature_dim else None
            self.item_feature_projection = nn.Linear(self.item_feature_dim, hidden_size, bias=False) if self.item_feature_dim else None
            for projection in (self.user_feature_projection, self.item_feature_projection):
                if projection is not None:
                    nn.init.normal_(projection.weight, mean=0, std=.02)

    @classmethod
    def from_config(cls, config, item_features=None):
        cfg = config.get("contract", config).get("architecture", config)
        return cls(**{key: cfg[key] for key in ("vocab_size", "context_length", "hidden_size", "layers", "heads")},
                   feature_mode=cfg["feature_mode"], item_features=item_features,
                   item_feature_dim=cfg["item_feature_dim"], user_feature_dim=cfg["user_feature_dim"],
                   feature_columns=cfg["feature_columns"], schema_hash=cfg["schema_hash"])

    def _forward(self, x, lengths, user_features, *, validate_features):
        if self.user_feature_dim:
            if user_features is None or user_features.shape != (len(x), self.user_feature_dim):
                raise ValueError("Aligned row user_features are required")
            if not user_features.is_floating_point():
                raise ValueError("User features must be floating point")
            if validate_features and not torch.isfinite(user_features).all():
                raise ValueError("Non-finite user features")
        elif user_features is not None and user_features.shape != (len(x), 0):
            raise ValueError("This model has no user feature columns")
        if not self.item_feature_dim and not self.user_feature_dim:
            return super().forward(x, lengths)
        embeddings = self.backbone.wte(x)
        if self.item_feature_projection is not None:
            embeddings = embeddings + x.ge(FIRST_ITEM).unsqueeze(-1) * self.item_feature_projection(self.item_features[x])
        if self.user_feature_projection is not None:
            embeddings = embeddings + x.eq(BOS).unsqueeze(-1) * self.user_feature_projection(user_features.float()).unsqueeze(1)
        hidden = self.backbone(inputs_embeds=embeddings, attention_mask=x.ne(PAD), use_cache=False).last_hidden_state
        return self.output(last_real_hidden(hidden, lengths))

    def forward(self, x, lengths, user_features=None):
        return self._forward(x, lengths, user_features, validate_features=True)


class FeatureData(dict):
    """Private loader-validated immutable arrays with an optional batch provider."""
    batcher = None


def close_feature_data(*values):
    """Release read-only mappings explicitly on Windows after their last use."""
    for value in values:
        if isinstance(value, dict):
            close_feature_data(*value.values())
        elif isinstance(value, np.memmap):
            value._mmap.close()


def _validate_finite(values, name):
    for start in range(0, len(values), 8192):
        if not np.isfinite(values[start:start + 8192]).all():
            raise ValueError(f"Non-finite {name}")


def load_feature_data(data_dir, feature_dir):
    """Bind all files to SHA receipts; retain user arrays as read-only memmaps."""
    data_dir, feature_dir = Path(data_dir), Path(feature_dir)
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("metadata.json", "train.npz", "val.npz"):
        if sha256(data_dir / filename) != manifest["inputs"][filename]["sha256"]:
            raise ValueError(f"Feature source SHA mismatch: {filename}")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    vocab, context = int(metadata["vocab_size"]), int(metadata["context"])
    if vocab <= FIRST_ITEM:
        raise ValueError("Empty known catalog")
    arrays = {}
    for filename in ("item_features.npy", "train_user_features.npy", "val_user_features.npy"):
        declaration = manifest["outputs"][filename]
        path = feature_dir / filename
        if sha256(path) != declaration["sha256"]:
            raise ValueError(f"Feature output SHA mismatch: {filename}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.dtype.kind != "f" or list(values.shape) != declaration["shape"] or str(values.dtype) != declaration["dtype"]:
            raise ValueError(f"Feature dtype/shape mismatch: {filename}")
        if values.ndim != 2:
            raise ValueError("Feature sidecars must be matrices")
        _validate_finite(values, filename)
        arrays[filename] = values
    items = arrays["item_features.npy"]
    if len(items) != vocab or np.any(items[:FIRST_ITEM] != 0):
        raise ValueError("Item features require matching vocabulary and zero special rows")
    train, val = (FeatureData(shared.load_data(data_dir / f"{split}.npz", vocab, context,
                                              require_keys=True, unique_keys=split == "val")) for split in ("train", "val"))
    user_dim = arrays["train_user_features.npy"].shape[1]
    _columns(manifest["feature_columns"], items.shape[1], user_dim)
    if canonical_hash(manifest["feature_columns"]) != manifest["schema_hash"]:
        raise ValueError("Feature schema hash mismatch")
    if manifest.get("rowkeys_algorithm") != ROWKEYS_ALGORITHM:
        raise ValueError("Unknown feature rowkeys_algorithm")
    for split, data in (("train", train), ("val", val)):
        user = arrays[f"{split}_user_features.npy"]
        if user.shape != (len(data["y"]), user_dim):
            raise ValueError(f"{split} feature rows/width are not aligned")
        if manifest["row_keys_sha256"][split] != make_row_keys_hash(data):
            raise ValueError(f"{split} feature row key binding mismatch")
        data["user_features"] = user
        for value in data.values():
            value.setflags(write=False)
    return metadata, train, val, items, manifest


class PackedBatcher:
    """One HtoD copy; at most two bounded host slabs, guarded by copy events."""
    def __init__(self, data, device, max_batch_size, pin_memory=True):
        self.data, self.device = data, torch.device(device)
        self.max_batch_size = max_batch_size
        self.user_dim = data["user_features"].shape[1]
        self.capacity = max_batch_size * ((data["x"].shape[1] + 2) * 8 + self.user_dim * 4)
        if self.capacity * 2 > 32 << 20:
            raise ValueError("Packed staging exceeds 32 MiB")
        self.pinned = pin_memory and self.device.type == "cuda"
        self.slots, self.events, self.position = [], [], 0

    def __enter__(self):
        count = 2 if self.pinned else 1
        self.slots = [torch.empty(self.capacity, dtype=torch.uint8, pin_memory=self.pinned) for _ in range(count)]
        self.events = [None] * count
        return self

    def __exit__(self, *exc):
        for event in self.events:
            if event is not None:
                event.synchronize()
        self.slots.clear()
        self.data = None

    def batch(self, indices):
        n = len(indices)
        if not 0 < n <= self.max_batch_size:
            raise ValueError("Invalid packed batch size")
        lengths = self.data["lengths"][indices]
        width = int(lengths.max())
        slot = self.position % len(self.slots)
        self.position += 1
        if self.events[slot] is not None:
            self.events[slot].synchronize()
        size = n * ((width + 2) * 8 + self.user_dim * 4)
        host = self.slots[slot][:size]
        raw = host.numpy()
        token_end, length_end, target_end = n * width * 8, n * (width + 1) * 8, n * (width + 2) * 8
        raw[:token_end].view(np.int64).reshape(n, width)[:] = self.data["x"][indices, :width]
        raw[token_end:length_end].view(np.int64)[:] = lengths
        raw[length_end:target_end].view(np.int64)[:] = self.data["y"][indices]
        raw[target_end:].view(np.float32).reshape(n, self.user_dim)[:] = self.data["user_features"][indices]
        packed = host.to(self.device, non_blocking=self.pinned) if self.device.type == "cuda" else host.clone()
        if self.pinned:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
            self.events[slot] = event
        return (packed[:token_end].view(torch.int64).reshape(n, width),
                packed[token_end:length_end].view(torch.int64), packed[length_end:target_end].view(torch.int64),
                packed[target_end:].view(torch.float32).reshape(n, self.user_dim))


def batch(data, indices, device):
    if getattr(data, "batcher", None) is not None:
        return data.batcher.batch(indices)
    x, lengths, y = legacy.batch(data, indices, device)
    user = torch.as_tensor(np.asarray(data["user_features"][indices], dtype=np.float32), device=device)
    return x, lengths, y, user


def update(model, data, microbatches, optimizer, scaler, device, *, collect_loss=True):
    """Same sum-CE/effective-batch update and bounded AMP retry as trained7group."""
    if not isinstance(data, FeatureData):
        raise ValueError("Training requires loader-validated feature data")
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    for attempt in range(9):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)
        optimizer.zero_grad(set_to_none=True)
        count, losses = sum(map(len, microbatches)), []
        for rows in microbatches:
            x, lengths, y, user = batch(data, rows, device)
            with legacy.autocast(device):
                logits = model._forward(x, lengths, user, validate_features=False)
                loss = F.cross_entropy(logits[:, FIRST_ITEM:].float(), y - FIRST_ITEM, reduction="sum") / count
            scaler.scale(loss).backward()
            losses.append(loss.detach())
        scaler.unscale_(optimizer)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # One combined safety read before committing, no per-microbatch scalar reads.
        finite_loss, finite_grad = torch.stack((torch.stack(losses).isfinite().all(), norm.isfinite())).tolist()
        if not finite_loss:
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Non-finite training loss; cursor did not advance")
        if finite_grad:
            scaler.step(optimizer)
            scaler.update()
            return (sum(torch.stack(losses).tolist()) if collect_loss else None), attempt
        optimizer.zero_grad(set_to_none=True)
        if not scaler.is_enabled() or attempt == 8:
            raise FloatingPointError("Non-finite gradients persisted after AMP backoff")
        scaler.update(new_scale=scaler.get_scale() / 2)


@torch.no_grad()
def evaluate(model, data, vocab, batch_size, device, indices=None, predictions_path=None):
    model.eval()
    indices = np.arange(len(data["y"])) if indices is None else np.asarray(indices)
    if not len(indices):
        raise ValueError("Cannot evaluate empty rows")
    k = min(10, vocab - FIRST_ITEM)
    totals = torch.zeros(5, dtype=torch.float64, device=device)
    tops, row_hits, row_ndcg = [], [], []
    discounts = 1 / torch.log2(torch.arange(k, device=device, dtype=torch.float32) + 2)
    for start in range(0, len(indices), batch_size):
        x, lengths, y, user = batch(data, indices[start:start + batch_size], device)
        with legacy.autocast(device):
            logits = model._forward(x, lengths, user, validate_features=False)[:, FIRST_ITEM:].float()
        known = (y >= FIRST_ITEM) & (y < vocab)
        # Sum reduction is also defined for an empty known-target slice (zero),
        # preserving baseline arithmetic without a host read of known.any().
        ce = F.cross_entropy(logits[known], y[known] - FIRST_ITEM, reduction="sum")
        top = logits.topk(k, dim=1).indices + FIRST_ITEM
        match = top.eq(y[:, None]) & known[:, None]
        hit, ndcg = match.any(1), (match * discounts).sum(1)
        totals += torch.stack((ce, known.sum(), hit.sum(), (match * discounts).sum(),
                               (~torch.isfinite(logits)).sum())).double()
        if predictions_path is not None:
            tops.append(top.cpu().numpy())
            row_hits.append(hit.cpu().numpy())
            row_ndcg.append(ndcg.cpu().numpy())
    ce, known_count, hits, ndcg_total, nonfinite = totals.tolist()
    if nonfinite:
        raise FloatingPointError("Non-finite evaluation logits")
    result = {"examples": len(indices), "known_targets": int(known_count), "unknown_target_misses": len(indices) - int(known_count),
              "cross_entropy_known": ce / known_count if known_count else None,
              "recall@10": hits / len(indices), "ndcg@10": ndcg_total / len(indices), "actual_k": k}
    if predictions_path is not None:
        values = {name: data[name][indices] for name in ROW_KEYS}
        values.update(row_index=indices, target_token=data["y"][indices], known_target=known_targets(data["y"][indices], vocab),
                      top_items=np.concatenate(tops), hit_at_10=np.concatenate(row_hits), ndcg_at_10=np.concatenate(row_ndcg))
        temporary = predictions_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, predictions_path)
    return result


def publish_selected(path, model, optimizer, scaler, state, config):
    """Adopt an identical replay after interrupted publication; never rewrite it."""
    if not path.exists():
        return checkpoint(path, model, optimizer, scaler, state, config)
    receipt_path = path.with_suffix(".json")
    committed = receipt_path.exists()
    digest = sha256(path)
    saved = (load_checkpoint(path, config, map_location="cpu") if committed else
             torch.load(path, map_location="cpu", weights_only=False))
    legacy.validate_resume(saved["config"], config)
    if (set(saved) != {"state_dict", "optimizer", "scaler", "state", "rng", "config"}
            or set(saved["state_dict"]) != set(model.state_dict())
            or any(not torch.equal(value.detach().cpu(), saved["state_dict"][name]) for name, value in model.state_dict().items())
            or saved["state"]["history"][-1]["validation"] != state["history"][-1]["validation"]
            or any(saved["state"][key] != state[key] for key in ("epoch", "offset", "updates", "examples", "selected_epoch"))
            or sha256(path) != digest):
        raise ValueError("Existing selected snapshot conflicts with replay")
    if not committed:
        write_json(receipt_path, {"sha256": digest, "epoch": state["epoch"], "offset": state["offset"], "updates": state["updates"]})
    return digest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path, required=True)
    result.add_argument("--feature-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--feature-mode", choices=("local-pack",), default="local-pack")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--batch-transfer", choices=("direct", "packed", "pinned-packed"), default="pinned-packed")
    for name, default in (("schedule-total-epochs", 5), ("batch-size", 64), ("accumulation", 2),
                          ("hidden-size", 256), ("layers", 4), ("heads", 4), ("seed", 2026), ("patience", 2),
                          ("bucket-size", 8192), ("cpu-threads", 2), ("checkpoint-updates", 1000), ("log-updates", 100),
                          ("max-updates", 0), ("stop-after-validation", 0)):
        result.add_argument("--" + name, type=int, default=default)
    result.add_argument("--lr", type=float, default=1e-4)
    result.add_argument("--warmup-ratio", type=float, default=.05)
    result.add_argument("--weight-decay", type=float, default=.01)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--resume-from", choices=("latest", "previous"), default="latest")
    result.add_argument("--stop-file", type=Path)
    result.add_argument("--deadline-utc", help="Timezone-aware ISO timestamp; honored at optimizer boundaries")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    for name in ("schedule_total_epochs", "batch_size", "accumulation", "hidden_size", "layers", "heads", "patience",
                 "cpu_threads", "checkpoint_updates", "log_updates"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if (args.schedule_total_epochs > 5 or args.patience != 2 or args.hidden_size % args.heads or
            min(args.bucket_size, args.max_updates, args.stop_after_validation, args.weight_decay) < 0 or
            not 0 <= args.warmup_ratio < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay)):
        raise ValueError("Invalid baseline schedule/architecture or runtime limits")
    deadline = datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")) if args.deadline_utc else None
    if deadline is not None and deadline.tzinfo is None:
        raise ValueError("deadline-utc must include its timezone")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError("Choose a fresh output or use --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    metadata, train, val, items, manifest = load_feature_data(args.data, args.feature_dir)
    vocab, context = int(metadata["vocab_size"]), int(metadata["context"])
    import transformers
    architecture = {"model": "GPT2Model from scratch", "vocab_size": vocab, "context_length": context,
                    "hidden_size": args.hidden_size, "layers": args.layers, "heads": args.heads, "tied_embeddings": True,
                    "feature_mode": "local-pack", "item_feature_dim": items.shape[1], "user_feature_dim": train["user_features"].shape[1],
                    "feature_columns": manifest["feature_columns"], "schema_hash": manifest["schema_hash"],
                    "conditioning": "row user features at BOS; item features on real history tokens; tied ID candidate head"}
    training_keys = ("schedule_total_epochs", "batch_size", "accumulation", "lr", "warmup_ratio", "weight_decay", "seed", "patience", "bucket_size", "batch_transfer")
    config = {"contract": {"architecture": architecture, "training": {key: getattr(args, key) for key in training_keys},
               "inputs_sha256": {name: manifest["inputs"][name]["sha256"] for name in ("metadata.json", "train.npz", "val.npz")},
               "sources_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                  ("kion_pack_train.py", "kion_train.py", "kion_full_train.py", "kion_conditioned_train.py")},
               "features_sha256": {name: value["sha256"] for name, value in manifest["outputs"].items()},
               "feature_manifest_sha256": sha256(args.feature_dir / "manifest.json"),
               "objective": "CE known train catalog; all validation rows including unknown misses; no seen mask",
               "checkpoint_selection": "maximum validation NDCG@10, strict greater; patience 2",
               "prediction_row_keys": list(ROW_KEYS)}, "data": args.data.resolve().as_posix(),
               "feature_dir": args.feature_dir.resolve().as_posix(),
               "runtime": {"torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                           "cuda": torch.version.cuda, "device": str(device), "cpu_threads": args.cpu_threads,
                           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
    if args.resume:
        old = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
        legacy.validate_resume(old, config)
        config = old
    model = NextItemGPT.from_config(config, items).to(device)
    config["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024)
    state = {"epoch": 0, "offset": 0, "updates": 0, "examples": 0, "best_ndcg": -1., "bad_epochs": 0,
             "history": [], "amp_overflow_retries": 0, "selected_epoch": None}
    if args.resume:
        saved = load_checkpoint(args.output / f"{args.resume_from}.pt", config, map_location=device)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        legacy.restore_rng(saved["rng"])
        del saved
    else:
        write_json(args.output / "config.json", config)
    eligible = np.flatnonzero(known_targets(train["y"], vocab))
    if not len(eligible):
        raise ValueError("No known train targets")
    effective = args.batch_size * args.accumulation
    shared.validate_cursor(state, len(eligible), effective, args.schedule_total_epochs)
    total_updates = math.ceil(len(eligible) / effective) * args.schedule_total_epochs
    started = time.perf_counter()
    initial_examples = state["examples"]
    recovery_publication = args.resume and args.resume_from == "previous"

    def emit(event, **extra):
        elapsed = time.perf_counter() - started
        payload = {"event": event, "updated_utc": datetime.now(timezone.utc).isoformat(), **state,
                   "elapsed_seconds": elapsed, "examples_per_second": (state["examples"] - initial_examples) / max(elapsed, 1e-9), **extra}
        write_json(args.output / "progress.json", payload)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)

    def stop_reason():
        if args.stop_file and args.stop_file.exists():
            return "runtime_stop_file"
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            return "runtime_deadline"
        if args.max_updates and state["updates"] >= args.max_updates:
            return "runtime_max_updates"
        if args.stop_after_validation and state["epoch"] >= args.stop_after_validation:
            return "runtime_stop_after_validation"
        return None

    def save_latest():
        nonlocal recovery_publication
        shared.validate_cursor(state, len(eligible), effective, args.schedule_total_epochs)
        checkpoint(args.output / "latest.pt", model, optimizer, scaler, state, config, rotate=not recovery_publication)
        recovery_publication = False

    stopped = None
    with ExitStack() as resources:
        resources.callback(close_feature_data, train, val, items)
        if args.batch_transfer != "direct":
            for data in (train, val):
                data.batcher = resources.enter_context(PackedBatcher(data, device, args.batch_size, args.batch_transfer == "pinned-packed"))
        emit("started", status="running", train_examples=len(eligible), validation_examples=len(val["y"]), schedule_total_updates=total_updates)
        for epoch in range(state["epoch"], args.schedule_total_epochs):
            if state["bad_epochs"] >= args.patience:
                break
            stopped = stop_reason()
            if stopped:
                save_latest()
                break
            order = legacy.epoch_order(eligible, train["lengths"], args.seed, epoch, args.batch_size, args.bucket_size)
            model.train()
            epoch_started = time.perf_counter()
            for offset, microbatches in legacy.groups(order, args.batch_size, args.accumulation, state["offset"]):
                for group in optimizer.param_groups:
                    group["lr"] = shared.learning_rate(state["updates"], total_updates, args.warmup_ratio, args.lr)
                logging = state["updates"] == 0 or (state["updates"] + 1) % args.log_updates == 0
                loss, retries = update(model, train, microbatches, optimizer, scaler, device, collect_loss=logging)
                count = sum(map(len, microbatches))
                state.update(updates=state["updates"] + 1, offset=offset + count, examples=state["examples"] + count,
                             amp_overflow_retries=state["amp_overflow_retries"] + retries)
                if logging:
                    emit("training", status="running", loss=loss, learning_rate=optimizer.param_groups[0]["lr"])
                if state["updates"] % args.checkpoint_updates == 0:
                    save_latest()
                stopped = stop_reason()
                if stopped:
                    save_latest()
                    break
            if stopped:
                break
            legacy.synchronize(device)
            train_wall = time.perf_counter() - epoch_started
            metrics = evaluate(model, val, vocab, args.batch_size, device)
            state["history"].append({"epoch": epoch + 1, "validation": metrics, "train_wall_seconds": train_wall})
            improved = metrics["ndcg@10"] > state["best_ndcg"]
            state.update(epoch=epoch + 1, offset=0, bad_epochs=0 if improved else state["bad_epochs"] + 1)
            if improved:
                state.update(best_ndcg=metrics["ndcg@10"], selected_epoch=epoch + 1)
                path = args.output / f"selected-epoch{epoch + 1:04d}.pt"
                digest = publish_selected(path, model, optimizer, scaler, state, config)
                shared._atomic_copy(path, args.output / "best.pt")
                write_json(args.output / "best.json", {"sha256": digest})
                write_json(args.output / "selected.json", {"file": path.name, "sha256": digest, "epoch": epoch + 1, "validation": metrics})
            save_latest()
            emit("validation", status="running", validation=metrics)
        finished = state["epoch"] >= args.schedule_total_epochs or state["bad_epochs"] >= args.patience
        status = "completed" if finished else "stopped"
        selected_metrics = None
        selected = None
        if state["selected_epoch"] is not None:
            path = args.output / f"selected-epoch{state['selected_epoch']:04d}.pt"
            saved = load_checkpoint(path, config, map_location=device)
            model.load_state_dict(saved["state_dict"])
            selected = {"file": path.name, "sha256": sha256(path), "epoch": state["selected_epoch"],
                        "validation": saved["state"]["history"][-1]["validation"]}
            del saved
            # Explicit recovery from an older cursor may leave a newer best.pt
            # from an interrupted epoch. Publish the selected cursor's bytes.
            if not (args.output / "best.pt").exists() or sha256(args.output / "best.pt") != selected["sha256"]:
                shared._atomic_copy(path, args.output / "best.pt")
            write_json(args.output / "best.json", {"sha256": selected["sha256"]})
            selected_metrics = evaluate(model, val, vocab, args.batch_size, device, predictions_path=args.output / "predictions.npz")
            if selected_metrics != selected["validation"]:
                raise ValueError("Selected predictions failed to reproduce validation")
            write_json(args.output / "selected.json", selected)
        result = {"status": status, "reason": stopped, "early_stopped": state["bad_epochs"] >= args.patience,
                  "state": state, "selected_epoch": state["selected_epoch"], "validation": selected_metrics, "test_accessed": False}
        write_json(args.output / "metrics.json", result)
        artifacts = [name for name in ("config.json", "latest.pt", "latest.json", "previous.pt", "previous.json", "best.pt", "best.json", "selected.json", "predictions.npz", "metrics.json")
                     if (args.output / name).exists() and (selected is not None or name not in
                         ("best.pt", "best.json", "selected.json", "predictions.npz"))]
        if selected:
            artifacts.extend((selected["file"], Path(selected["file"]).with_suffix(".json").name))
        write_json(args.output / "completion.json", {"state": status, "reason": stopped, "updates": state["updates"],
                   "schedule_total_epochs": args.schedule_total_epochs, "selected_epoch": state["selected_epoch"],
                   "validation_rows": selected_metrics["examples"] if selected_metrics else 0,
                   "metrics": {"ndcg10": selected_metrics["ndcg@10"], "recall10": selected_metrics["recall@10"], "ce": selected_metrics["cross_entropy_known"]} if selected_metrics else {},
                   "artifacts": artifacts, "sha256": {name: sha256(args.output / name) for name in artifacts}, "test_accessed": False})
        emit("finished", status=status, reason=stopped)
    return result


if __name__ == "__main__":
    main()
