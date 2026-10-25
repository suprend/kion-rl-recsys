"""Inference-only frozen E2 GPT prefix path; numerical imports are lazy.

This module accepts exported tensors and anonymous transformed model inputs.
It never opens the historical checkpoint, provider, event store or feature data.
The cached table freezes the original item/text/people feature tower result.
"""
from __future__ import annotations

from contextlib import nullcontext


CONTRACT = "kion-frozen-e2-prefix-forward-v1"
CONFIG = {
    "vocab_size": 15418,
    "context_length": 64,
    "hidden_size": 256,
    "layers": 4,
    "heads": 4,
    "user_feature_dim": 3252,
}
PINNED_TRANSFORMERS = "4.55.4"
MAX_PRODUCER_BATCH = 16
MAX_SELECTED_ROWS = 4
CANDIDATE_WIDTH = 1024


def expected_encoder_shapes():
    """Exact flat export schema, including GPT2 Conv1D weight orientation."""
    h, v, t, u = (CONFIG[key] for key in
                  ("hidden_size", "vocab_size", "context_length", "user_feature_dim"))
    shapes = {
        "backbone.wte.weight": (v, h),
        "backbone.wpe.weight": (t, h),
        "backbone.ln_f.weight": (h,),
        "backbone.ln_f.bias": (h,),
        "user_projection.weight": (h, u),
    }
    for layer in range(CONFIG["layers"]):
        prefix = f"backbone.h.{layer}."
        for norm in ("ln_1", "ln_2"):
            shapes[prefix + norm + ".weight"] = (h,)
            shapes[prefix + norm + ".bias"] = (h,)
        for name, weight, bias in (
            ("attn.c_attn", (h, 3 * h), (3 * h,)),
            ("attn.c_proj", (h, h), (h,)),
            ("mlp.c_fc", (h, 4 * h), (4 * h,)),
            ("mlp.c_proj", (4 * h, h), (h,)),
        ):
            shapes[prefix + name + ".weight"] = weight
            shapes[prefix + name + ".bias"] = bias
    return shapes


def _check_state(torch, state):
    shapes = expected_encoder_shapes()
    if not isinstance(state, dict) or set(state) != set(shapes):
        raise ValueError("GPT export requires the exact encoder tensor keys")
    for key, shape in shapes.items():
        value = state[key]
        if (not torch.is_tensor(value) or tuple(value.shape) != shape
                or value.dtype != torch.float32 or value.layout != torch.strided
                or value.device.type != "cpu" or not bool(torch.isfinite(value).all())):
            raise ValueError("Invalid finite dense CPU FP32 GPT tensor: " + key)


def extract_encoder_state(torch, payload):
    """Root exporter helper; caller owns the one trusted checkpoint read.

    The result contains only pure CPU tensor values for weights_only loading.
    Training state, private raw feature buffers and unused model branches are
    intentionally absent. No path is opened here and no optimizer is created.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Trusted E2 payload must contain state_dict")
    original = payload["state_dict"]
    shapes = expected_encoder_shapes()
    missing = set(shapes) - set(original)
    if missing:
        raise ValueError("E2 checkpoint misses encoder weights: " + ", ".join(sorted(missing)))
    state = {key: original[key].detach().cpu().clone() for key in shapes}
    _check_state(torch, state)
    return state


def _inputs(np, manifest, arrays):
    spec = manifest.get("gpt")
    if not isinstance(spec, dict) or spec.get("config") != CONFIG:
        raise ValueError("GPT tier requires the exact frozen E2 configuration")
    size = spec.get("producer_batch_size")
    slots = spec.get("row_slots")
    if (type(size) is not int or not 1 <= size <= MAX_PRODUCER_BATCH
            or not isinstance(slots, list) or not 1 <= len(slots) <= MAX_SELECTED_ROWS
            or any(type(slot) is not int or not 0 <= slot < size for slot in slots)
            or len(set(slots)) != len(slots)):
        raise ValueError("Exact producer size and unique selected row slots required")
    required = ("gpt_x", "gpt_lengths", "gpt_user_features", "item_table", "candidate_ids")
    if any(key not in arrays for key in required):
        raise ValueError("GPT fixture lacks encoder inputs or frozen table/candidate IDs")
    x, lengths, user, table, ids = (np.asarray(arrays[key]) for key in required)
    if (x.dtype != np.int64 or x.shape != (size, CONFIG["context_length"])
            or lengths.dtype != np.int64 or lengths.shape != (size,)
            or user.dtype != np.float32 or user.shape != (size, CONFIG["user_feature_dim"])
            or table.dtype != np.float32 or table.shape != (CONFIG["vocab_size"], CONFIG["hidden_size"])
            or ids.dtype.kind not in "iu" or ids.shape != (len(slots), CANDIDATE_WIDTH)):
        raise ValueError("GPT fixture shapes/dtypes differ from the producer contract")
    if (np.any(lengths < 1) or np.any(lengths > CONFIG["context_length"])
            or np.any(x < 0) or np.any(x >= CONFIG["vocab_size"])
            or np.any(ids < 3) or np.any(ids >= CONFIG["vocab_size"])
            or not bool(np.isfinite(user).all()) or not bool(np.isfinite(table).all())):
        raise ValueError("GPT fixture contains invalid tokens, lengths or nonfinite features")
    active = np.arange(CONFIG["context_length"])[None, :] < lengths[:, None]
    if bool(np.any(x[~active] != 0)) or bool(np.any(x[active] == 0)):
        raise ValueError("GPT producer contexts must be right padded with PAD=0")
    # The archived training batch adapter trims using all original peers.
    width = int(lengths.max())
    return x[:, :width], lengths, user, table, ids, slots, size, width


def encode(torch, np, fixture_root, manifest, arrays, device, precision):
    """Run the real frozen backbone and full-catalog projection.

    Arrays and the pure state descriptor must have passed core fixture hashes.
    Returns selected rows only, while all producer peers participate in the
    original BxT encoder and BxV retrieval operations. CPU FP32 is a separate
    numerical scope from historical CUDA FP16; the caller owns oracle policy.
    """
    from .io import safe_torch_state
    import transformers
    from transformers import GPT2Config, GPT2Model

    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise ValueError("GPT replay requires transformers==" + PINNED_TRANSFORMERS)
    device = torch.device(device)
    if precision not in ("fp32", "cuda-fp16"):
        raise ValueError("GPT precision must be fp32 or cuda-fp16")
    if precision == "cuda-fp16" and device.type != "cuda":
        raise ValueError("cuda-fp16 requires an actual CUDA device")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA GPT replay are supported")
    x, lengths, user, table, ids, slots, size, width = _inputs(np, manifest, arrays)
    descriptor = manifest.get("weights", {}).get("GPT")
    if not isinstance(descriptor, dict):
        raise ValueError("GPT tier requires the verified weights.GPT descriptor")
    state = safe_torch_state(torch, fixture_root, descriptor)
    _check_state(torch, state)
    config = GPT2Config(
        vocab_size=CONFIG["vocab_size"], n_positions=CONFIG["context_length"],
        n_ctx=CONFIG["context_length"], n_embd=CONFIG["hidden_size"],
        n_layer=CONFIG["layers"], n_head=CONFIG["heads"],
        bos_token_id=1, eos_token_id=None, pad_token_id=0, use_cache=False,
    )
    # Initialization is local CPU work, with no pretrained path/download.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        backbone = GPT2Model(config)
        projection = torch.nn.Linear(CONFIG["user_feature_dim"], CONFIG["hidden_size"], bias=False)
    backbone.load_state_dict({key[len("backbone."):]: value for key, value in state.items()
                              if key.startswith("backbone.")}, strict=True)
    projection.load_state_dict({"weight": state["user_projection.weight"]}, strict=True)
    del state
    backbone = backbone.to(device).eval().requires_grad_(False)
    projection = projection.to(device).eval().requires_grad_(False)
    tx = torch.tensor(x, dtype=torch.long, device=device)
    tl = torch.tensor(lengths, dtype=torch.long, device=device)
    tu = torch.tensor(user, dtype=torch.float32, device=device)
    tt = torch.tensor(table, dtype=torch.float32, device=device)
    ti = torch.tensor(ids.astype(np.int64, copy=False), dtype=torch.long, device=device)
    ts = torch.tensor(slots, dtype=torch.long, device=device)
    amp = torch.autocast("cuda", dtype=torch.float16) if precision == "cuda-fp16" else nullcontext()
    with torch.inference_mode(), amp:
        embeddings = tt[tx]
        embeddings = embeddings + tx.eq(1).unsqueeze(-1) * projection(tu.float()).unsqueeze(1)
        hidden = backbone(inputs_embeds=embeddings, attention_mask=tx.ne(0),
                          use_cache=False).last_hidden_state
        encoded = hidden[torch.arange(size, device=device), tl - 1]
        # Preserve historical M=B, N=full-vocab before selecting rows/items.
        full_logits = torch.nn.functional.linear(encoded, tt)
        selected_q = encoded.index_select(0, ts).float()
        selected_base = full_logits.index_select(0, ts).gather(1, ti).float()
    if (not bool(torch.isfinite(selected_q).all())
            or not bool(torch.isfinite(selected_base).all())):
        raise FloatingPointError("Frozen GPT forward produced nonfinite values")
    result = {
        "encoded": selected_q.cpu().numpy().copy(),
        "base_logits": selected_base.cpu().numpy().copy(),
        "diagnostics": {
            "contract": CONTRACT,
            "config": dict(CONFIG),
            "producer_batch_size": size,
            "row_slots": list(slots),
            "trimmed_context_width": width,
            "full_projection_shape": [size, CONFIG["vocab_size"]],
            "device": str(device),
            "precision": precision,
            "parameter_dtype": "float32",
            "table_dtype": "float32",
            "transformers_version": transformers.__version__,
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "attention_implementation": getattr(backbone.config, "_attn_implementation", None),
            "optimizer_created": False,
            "optimizer_updates": 0,
            "candidate_table_rebuilt": False,
            "source": "frozen cached table + frozen encoder + transformed historical inputs with source user IDs excluded; anonymity is not established",
        },
    }
    return result
