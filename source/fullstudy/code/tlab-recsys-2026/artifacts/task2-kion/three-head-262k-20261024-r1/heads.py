"""Three fresh frozen-representation residual heads; numeric imports are lazy.

Only run() imports NumPy/PyTorch. This module never imports the GPT/provider,
evaluates validation, selects a checkpoint, resumes training, or changes epochs.
"""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

CONTRACT = "kion-three-head-262k-head-v1"
BANK_CONTRACT = "kion-three-head-262k-bank-v1"
TRAIN_ROWS = 262144
STEPS_PER_EPOCH = TRAIN_ROWS // 128
EPOCHS = 3
SHUFFLE = "python-random(seed+zero-based-epoch)"
RAW_GATHER = "ascending-physical-rows-then-inverse-permutation-close-mmap-per-batch-v1"
RAW_WIDTH = 1068
SOURCE_WIDTH = 2
FEATURE_SCHEMA_SHA256 = "71d90eee774d30db804d1ab32c4d82995d1320baf26aaf36443a6498fd85fc5d"
SOURCE_COLUMNS = ["current_detached_gpt_logit", "log1p_strict7d_count"]
HEAD_NAMES = ("A", "B", "C")
B_INDICES = tuple(range(142, 536)) + tuple(range(907, 1068))
CONFIG = {"raw_width": 1068, "source_width": 2, "projection_width": 32,
          "hidden_width": 256, "mlp_input_width": 800, "dropout": 0.0,
          "projection_bias": False, "final_output_initialization": "zeros",
          "other_initialization": "fresh-common-torch-default",
          "lr": 1e-4, "weight_decay": 0.01, "grad_clip": 1.0,
          "batch_size": 128, "positive_column": 0, "loss": "cross_entropy32",
          "amp": "cuda-fp16-with-finite-retry-then-fp32", "seed": 20261024}
ARRAY_SPEC = {
    "encoded": ("float32", (TRAIN_ROWS, 256)),
    "candidate_ids": ("int32", (TRAIN_ROWS, 32)),
    "base_logits": ("float32", (TRAIN_ROWS, 32)),
    "raw_features": ("float32", (TRAIN_ROWS, 32, 1068)),
    "source_values": ("float32", (TRAIN_ROWS, 32, 2)),
    "row_indices": ("int64", (TRAIN_ROWS,)), "user_ids": ("int64", (TRAIN_ROWS,)),
    "target_ids": ("int32", (TRAIN_ROWS,)), "dates": ("int32", (TRAIN_ROWS,)),
    "item_table": ("float32", (None, 256)),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def mask_indices(name):
    if name == "A":
        return ()
    if name == "B":
        return B_INDICES
    if name == "C":
        return tuple(range(RAW_WIDTH))
    raise ValueError("Unknown head")


def epoch_schedule(epoch):
    """Exact original global shuffle, with no date/block grouping of updates."""
    if type(epoch) is not int or not 0 <= epoch < EPOCHS:
        raise ValueError("Zero-based epoch must be 0, 1 or 2")
    order = list(range(TRAIN_ROWS))
    random.Random(CONFIG["seed"] + epoch).shuffle(order)
    return order


def epoch_record(epoch, order=None):
    """Exact schedule SHA over compact JSON global row indices, streamed.

    Hashing never builds raw features or a full serialized epoch JSON string.
    The digest is identical to sha256(json.dumps(order,separators=(',',':'))).
    """
    digest = hashlib.sha256(b"[")
    first = True
    for row in epoch_schedule(epoch) if order is None else order:
        if not first:
            digest.update(b",")
        digest.update(str(row).encode("ascii"))
        first = False
    digest.update(b"]")
    return {"epoch": epoch + 1, "shuffle_sha256": digest.hexdigest(),
            "rows": TRAIN_ROWS, "updates_per_head": STEPS_PER_EPOCH}


def make_heads(torch, seed=20261024, device="cpu"):
    """Fresh identical parameter values, independently owned parameters/masks."""
    nn = torch.nn

    class Head(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.pair_projection = nn.Linear(1070, 32, bias=False)
            self.pair_head = nn.Sequential(nn.Linear(800, 256), nn.GELU(), nn.Linear(256, 1))
            mask = torch.zeros(1068, dtype=torch.float32)
            mask[list(mask_indices(name))] = 1
            self.register_buffer("raw_mask", mask, persistent=False)

        def forward(self, q, candidates, base, raw, source):
            b, c, h = candidates.shape
            if (h != 256 or tuple(q.shape) != (b, 256) or tuple(base.shape) != (b, c)
                    or tuple(raw.shape) != (b, c, 1068) or tuple(source.shape) != (b, c, 2)):
                raise ValueError("Misaligned scorer tensors (source must contain only metadata12/13)")
            query = q.detach().unsqueeze(1).expand_as(candidates)
            item = candidates.detach()
            numeric = torch.cat((raw.detach().float() * self.raw_mask,
                                 source.detach().float()), dim=-1)
            projected = self.pair_projection(numeric)
            residual = self.pair_head(torch.cat((query, item, query * item, projected), dim=-1)).squeeze(-1)
            return base.detach().float() + residual.float()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        models = {name: Head(name) for name in HEAD_NAMES}
        with torch.no_grad():
            models["A"].pair_head[2].weight.zero_()
            models["A"].pair_head[2].bias.zero_()
        state = models["A"].state_dict()
        for name in ("B", "C"):
            models[name].load_state_dict(state, strict=True)
    return {name: model.to(device) for name, model in models.items()}


def score_heads(models, q, item_table, ids, base, raw, source, torch):
    """Shared frozen gather; callers choose no_grad/autocast and candidate chunks."""
    candidates = item_table.detach()[ids.long()]
    return {name: models[name](q, candidates, base, raw, source) for name in HEAD_NAMES}


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Metadata exceeds 8 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_request(request):
    if type(request.get("epochs", EPOCHS)) is not int or request.get("epochs", EPOCHS) != EPOCHS:
        raise ValueError("Exactly 3 common epochs required; no shortened fallback")
    for key, fixed in (("seed", 20261024), ("batch_size", 128)):
        if request.get(key, fixed) != fixed:
            raise ValueError(f"Fixed {key} must be {fixed}")
    if not request.get("bank_dir") or not request.get("output_dir"):
        raise ValueError("bank_dir and output_dir required")


def _bank(request, np, check):
    """Check metadata/header and pinned identity; reuse only an exact stat receipt."""
    root = Path(request["bank_dir"]).resolve()
    manifest_path = root / "bank.json"
    manifest_sha = sha256(manifest_path)
    expected = request.get("bank_manifest_sha256")
    if expected is not None and expected != manifest_sha:
        raise ValueError("Bank manifest pin changed")
    manifest = _read_json(manifest_path)
    if (manifest.get("contract") != BANK_CONTRACT or manifest.get("state") != "completed"
            or manifest.get("split") != "train" or manifest.get("train_rows") != TRAIN_ROWS
            or manifest.get("natural_covered_only") is not False
            or manifest.get("same_day_positives_excluded") is not True):
        raise ValueError("Completed unfiltered train262144 bank required")
    for key, fixed in (("feature_schema_sha256", FEATURE_SCHEMA_SHA256), ("source_columns", SOURCE_COLUMNS)):
        if manifest.get(key) != fixed or request.get(key) != fixed:
            raise ValueError(f"Explicit matching bank/request lineage required: {key}")
    if not isinstance(request.get("teacher"), dict) or not request["teacher"] or manifest.get("teacher") != request["teacher"]:
        raise ValueError("Explicit matching frozen teacher required")
    receipt_path = request.get("bank_verification_receipt")
    receipt = _read_json(receipt_path) if receipt_path else {}
    reuse = (receipt.get("bank_manifest_sha256") == manifest_sha and receipt.get("hashes_verified") is True)
    arrays, identities = {}, {}
    for name, (dtype, shape) in ARRAY_SPEC.items():
        entry = manifest["arrays"][name]
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Array path escapes bank")
        stat = path.stat()
        if (entry["dtype"] != dtype or len(entry["shape"]) != len(shape)
                or any(want is not None and got != want for got, want in zip(entry["shape"], shape))
                or stat.st_size != entry["bytes"]):
            raise ValueError(f"Bank metadata mismatch: {name}")
        identity = {"path": path.as_posix(), "bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "sha256": entry["sha256"]}
        if not reuse or receipt.get("arrays", {}).get(name) != identity:
            check(phase="head-bank-hash", array=name)
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Array hash mismatch: {name}")
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if str(arr.dtype) != dtype or list(arr.shape) != entry["shape"]:
            raise ValueError(f"Array header mismatch: {name}")
        arrays[name], identities[name] = arr, identity
    if arrays["item_table"].shape[0] < 4:
        raise ValueError("Item table has no real vocabulary")
    return arrays, {"bank_manifest_sha256": manifest_sha, "hashes_verified": True,
                    "arrays": identities}, manifest


def _save(models, optimizers, scalers, output, receipt, torch, complete):
    # CPU-only fixtures must not initialize CUDA just because a GPU is present.
    # CUDA model parameters prove the device was already used by this training.
    has_cuda_parameters = any(parameter.is_cuda for model in models.values() for parameter in model.parameters())
    cuda_rng_state = torch.cuda.get_rng_state_all() if has_cuda_parameters else []
    for name in HEAD_NAMES:
        directory = output / name
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / ("head.pt" if complete else "partial-head.pt")
        temporary = path.with_name(path.name + ".tmp")
        model = models[name]
        payload = {"contract": CONTRACT, "name": name, "complete": complete,
                   "config": dict(CONFIG), "raw_indices": list(mask_indices(name)),
                   "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   "optimizer_state_dict": optimizers[name].state_dict(),
                   "scaler_state_dict": scalers[name].state_dict(), "receipt": receipt,
                   "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_state": cuda_rng_state}
        torch.save(payload, temporary)
        temporary.replace(path)


def _copy_raw_batch(arrays, identities, positions, np):
    """Gather B128 in physical order; close mmap; restore logical SGD order.

    NumPy advanced indexing owns its gathered memory. Only this batch and its
    inverse-permuted copy coexist; neither retains a view into the 35.8GB file.
    """
    if len(positions) != CONFIG["batch_size"]:
        raise ValueError("Raw gather requires exactly one B128 batch")
    permutation = sorted(range(len(positions)), key=positions.__getitem__)
    physical_positions = [positions[index] for index in permutation]
    inverse = [0] * len(positions)
    for physical_index, logical_index in enumerate(permutation):
        inverse[logical_index] = physical_index
    identity = identities["raw_features"]
    path = Path(identity["path"])
    stat = path.stat()
    if stat.st_size != identity["bytes"] or stat.st_mtime_ns != identity["mtime_ns"]:
        raise ValueError("Verified raw bank identity changed before batch read")
    raw = arrays["raw_features"]
    if raw is None:
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        gathered = raw[physical_positions]
    finally:
        raw._mmap.close()
        arrays["raw_features"] = None
    return gathered[inverse]


def preflight_heads(request):
    """Stdlib-only completed-budget and identity gate, before provider imports.

    Hash all three small checkpoints, but never decode them or scan bank arrays.
    Expected teacher/schema/source metadata and the bank manifest pin are required.
    """
    validate_request(request)
    directory = Path(request.get("head_dir", request["output_dir"]))
    receipt = _read_json(directory / "training.json")
    epochs = request.get("epochs", EPOCHS)
    steps = epochs * STEPS_PER_EPOCH
    expected = {"contract": CONTRACT, "status": "completed", "epochs_requested": epochs,
                "epochs_completed": epochs, "steps_per_head": steps,
                "head_steps": {name: steps for name in HEAD_NAMES}, "rows_per_epoch": TRAIN_ROWS,
                "batch_size": 128, "seed": 20261024, "common_batches": True,
                "shuffle": SHUFFLE, "raw_gather": RAW_GATHER,
                "trained_from": "fresh-common-init-final-zero"}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed exact common training budget required")
    records = receipt.get("epoch_records")
    if not isinstance(records, list) or len(records) != EPOCHS:
        raise ValueError("Exact completed epoch schedule lineage required")
    for epoch, record in enumerate(records):
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in epoch_record(epoch).items()):
            raise ValueError("Completed epoch row schedule lineage differs")
    bank_pin = request.get("bank_manifest_sha256")
    if not isinstance(bank_pin, str) or len(bank_pin) != 64:
        raise ValueError("Explicit completed bank manifest SHA-256 required")
    manifest_path = Path(request["bank_dir"]) / "bank.json"
    if sha256(manifest_path) != bank_pin or receipt.get("bank_manifest_sha256") != bank_pin:
        raise ValueError("Completed bank manifest differs from request/training receipt")
    manifest = _read_json(manifest_path)
    if (manifest.get("contract") != BANK_CONTRACT or manifest.get("state") != "completed"
            or manifest.get("split") != "train" or manifest.get("train_rows") != TRAIN_ROWS
            or manifest.get("natural_covered_only") is not False
            or manifest.get("same_day_positives_excluded") is not True):
        raise ValueError("Completed unfiltered train262144 bank required")
    teacher = request.get("teacher")
    if (not isinstance(teacher, dict) or not teacher
            or request.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
            or request.get("source_columns") != SOURCE_COLUMNS):
        raise ValueError("Explicit expected teacher/full1068 schema/source2 pins required")
    for key in ("teacher", "feature_schema_sha256", "source_columns"):
        if receipt.get(key) != request[key] or manifest.get(key) != request[key]:
            raise ValueError(f"Training/evaluation lineage differs: {key}")
    hashes = receipt.get("checkpoint_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(HEAD_NAMES):
        raise ValueError("All three completed checkpoint hashes required")
    for name in HEAD_NAMES:
        if sha256(directory / name / "head.pt") != hashes[name]:
            raise ValueError(f"Completed checkpoint hash changed: {name}")
    return receipt


def load_heads(request, torch, device="cpu"):
    """Only load complete, same-budget A/B/C artifacts; never return partial state."""
    preflight = preflight_heads(request)
    expected_receipt = {key: value for key, value in preflight.items() if key != "checkpoint_sha256"}
    directory = Path(request.get("head_dir", request.get("output_dir", "")))
    models = make_heads(torch, seed=request.get("seed", 20261024), device=device)
    shared = None
    for name in HEAD_NAMES:
        payload = torch.load(directory / name / "head.pt", map_location="cpu", weights_only=True)
        if (payload.get("contract") != CONTRACT or payload.get("name") != name
                or payload.get("complete") is not True or payload.get("config") != CONFIG
                or payload.get("raw_indices") != list(mask_indices(name))):
            raise ValueError("Unrecognized/incomplete head checkpoint")
        receipt = payload["receipt"]
        if receipt != expected_receipt:
            raise ValueError("Checkpoint receipt differs from completed preflight metadata")
        identity = (receipt["epochs_completed"], receipt["steps_per_head"], receipt["bank_manifest_sha256"])
        if identity[0] != EPOCHS or identity[1] != EPOCHS * STEPS_PER_EPOCH:
            raise ValueError("Incomplete epoch budget")
        if shared is not None and shared != identity:
            raise ValueError("Head training budgets/banks differ")
        shared = identity
        model = models[name]
        model.load_state_dict(payload["state_dict"], strict=True)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite checkpoint")
        model.pilot_receipt = receipt
        model.eval()
    return models


def _update(model, optimizer, scaler, tensors, torch, device):
    """Retry a nonfinite AMP gradient at the same cursor, then use fp32."""
    q, candidates, base, raw, source = tensors
    for attempt in range(3):
        use_amp = device.type == "cuda" and attempt < 2
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
            scores = model(q, candidates, base, raw, source)
            loss = torch.nn.functional.cross_entropy(scores.float(), torch.zeros(q.shape[0], device=device, dtype=torch.long))
        if not torch.isfinite(loss):
            if attempt == 2:
                raise FloatingPointError("Nonfinite head CE")
            continue
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
            if use_amp:
                scaler.update()
            if attempt == 2:
                raise FloatingPointError("Nonfinite head gradients")
            continue
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite updated head")
        return {"loss": float(loss.detach()), "grad_norm": float(norm), "attempts": attempt + 1,
                "precision": "cuda-fp16" if use_amp else "fp32"}
    raise AssertionError("unreachable")


def run(request, check):
    """Root-owned training. check(**fields) may raise to stop; partials are saved.

    No root callback is interpreted as permission to resume, change epoch count,
    skip a row, evaluate validation or select a checkpoint.
    """
    validate_request(request)
    import numpy as np
    import torch

    output = Path(request["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name / "head.pt").exists() or (output / name / "partial-head.pt").exists() for name in HEAD_NAMES):
        raise FileExistsError("Fresh training refuses existing complete/partial heads")
    device = torch.device(request.get("device", "cuda"))
    arrays, bank_receipt, manifest = _bank(request, np, check)
    atomic_json(output / "bank-verification.json", bank_receipt)
    epochs = int(request.get("epochs", EPOCHS))
    models = make_heads(torch, 20261024, device)
    optimizers = {name: torch.optim.AdamW(models[name].parameters(), lr=1e-4, weight_decay=0.01) for name in HEAD_NAMES}
    scalers = {name: torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024) for name in HEAD_NAMES}
    table = torch.from_numpy(np.array(arrays["item_table"], copy=True)).to(device).detach()
    if not torch.isfinite(table).all():
        raise ValueError("Nonfinite item table")
    receipt = {"contract": CONTRACT, "bank_manifest_sha256": bank_receipt["bank_manifest_sha256"],
               "teacher": manifest.get("teacher"), "feature_schema_sha256": manifest.get("feature_schema_sha256"),
               "source_columns": manifest.get("source_columns"), "epochs_requested": epochs,
               "epochs_completed": 0, "steps_per_head": 0, "head_steps": {name: 0 for name in HEAD_NAMES},
               "rows_per_epoch": TRAIN_ROWS, "batch_size": 128, "seed": 20261024,
               "shuffle": SHUFFLE, "raw_gather": RAW_GATHER,
               "common_batches": True,
               "trained_from": "fresh-common-init-final-zero", "status": "running", "epoch_records": []}
    start = time.perf_counter()
    deadline = request.get("deadline_utc")
    deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00")) if deadline else None
    try:
        for epoch in range(epochs):
            losses = {name: 0.0 for name in HEAD_NAMES}
            order = epoch_schedule(epoch)
            for batch_index in range(STEPS_PER_EPOCH):
                check(phase="head-train", epoch=epoch + 1, batch=batch_index, steps=receipt["steps_per_head"])
                if deadline is not None and datetime.now(timezone.utc) >= deadline:
                    raise TimeoutError("Head training deadline reached")
                positions = order[batch_index * 128:(batch_index + 1) * 128]
                raw_cpu = _copy_raw_batch(arrays, bank_receipt["arrays"], positions, np)
                raw = torch.from_numpy(raw_cpu).to(device)
                del raw_cpu
                def take(key):
                    return torch.from_numpy(arrays[key][positions]).to(device)
                q, ids, base, source = (take(key) for key in ("encoded", "candidate_ids", "base_logits", "source_values"))
                if (torch.any(ids < 3) or torch.any(ids >= table.shape[0])
                        or not all(torch.isfinite(x).all() for x in (q, base, raw, source))):
                    raise ValueError("Invalid frozen training batch")
                targets = take("target_ids")
                if not torch.equal(ids[:, 0], targets):
                    raise ValueError("Positive is not at column0")
                sorted_ids = ids.sort(dim=1).values
                if torch.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
                    raise ValueError("Duplicate training candidates")
                candidates = table[ids.long()].detach()
                for name in HEAD_NAMES:
                    record = _update(models[name], optimizers[name], scalers[name], (q, candidates, base, raw, source), torch, device)
                    losses[name] += record["loss"]
                    receipt["head_steps"][name] += 1
                receipt["steps_per_head"] += 1
                receipt["last_batch"] = {"epoch": epoch + 1, "batch": batch_index + 1}
                receipt["last_update"] = {name: receipt["head_steps"][name] for name in HEAD_NAMES}
                check(phase="head-batch-complete", epoch=epoch + 1, batch=batch_index + 1,
                      steps=receipt["steps_per_head"], elapsed_seconds=time.perf_counter() - start)
            receipt["epochs_completed"] = epoch + 1
            receipt["epoch_records"].append(dict(epoch_record(epoch, order),
                mean_ce={name: losses[name] / STEPS_PER_EPOCH for name in HEAD_NAMES}))
            atomic_json(output / "progress.json", receipt)
        receipt["elapsed_seconds"] = time.perf_counter() - start
        receipt["status"] = "completed"
        _save(models, optimizers, scalers, output, receipt, torch, True)
        receipt["checkpoint_sha256"] = {name: sha256(output / name / "head.pt") for name in HEAD_NAMES}
        atomic_json(output / "training.json", receipt)
        return receipt
    except BaseException as error:
        receipt["elapsed_seconds"] = time.perf_counter() - start
        receipt["status"] = "stopped"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        _save(models, optimizers, scalers, output, receipt, torch, False)
        atomic_json(output / "stopped.json", receipt)
        raise
