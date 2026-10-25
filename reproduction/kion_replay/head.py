"""Frozen C1070 head kernel, preserving producer batch/slot and chunk64."""
from __future__ import annotations

from contextlib import nullcontext

from .io import require, safe_torch_state

KEY_SHAPES = {"pair_projection.weight": (32, 1070), "pair_head.0.weight": (256, 800),
              "pair_head.0.bias": (256,), "pair_head.2.weight": (1, 256), "pair_head.2.bias": (1,)}


def make_head(torch, state, device):
    nn = torch.nn
    require(set(state) == set(KEY_SHAPES), "Frozen head state keys differ")
    require(all(tuple(state[key].shape) == shape for key, shape in KEY_SHAPES.items()), "Frozen head state shape differs")

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.pair_projection = nn.Linear(1070, 32, bias=False)
            self.pair_head = nn.Sequential(nn.Linear(800, 256), nn.GELU(), nn.Linear(256, 1))
            self.register_buffer("raw_mask", torch.ones(1068, dtype=torch.float32), persistent=False)

        def forward(self, q, candidates, base, raw, source):
            query = q.detach().unsqueeze(1).expand_as(candidates)
            item = candidates.detach()
            numeric = torch.cat((raw.detach().float() * self.raw_mask, source.detach().float()), dim=-1)
            projected = self.pair_projection(numeric)
            residual = self.pair_head(torch.cat((query, item, query * item, projected), dim=-1)).squeeze(-1)
            return base.detach().float() + residual.float()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = Head()
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device)


def buckets(layout):
    """Keep selected row slots; colliding slots belong to separate producer calls."""
    groups = []
    for row, spec in enumerate(layout):
        group = next((group for group in groups if group["size"] == spec["size"] and spec["slot"] not in group["slots"]), None)
        if group is None:
            group = {"size": spec["size"], "slots": [], "rows": []}
            groups.append(group)
        group["slots"].append(spec["slot"])
        group["rows"].append(row)
    return groups


def score(torch, np, fixture_root, manifest, arrays, device, precision, encoded=None, base_logits=None, check=None):
    require(precision in ("fp32", "cuda-fp16"), "Unsupported precision")
    require(precision != "cuda-fp16" or str(device).startswith("cuda"), "cuda-fp16 requires a CUDA device")
    ids = arrays["candidate_ids"]
    require(bool(((ids >= 3) & (ids < len(arrays["item_table"]))).all()), "Candidate token outside frozen table")
    require(all(len(set(map(int, row))) == 1024 for row in ids), "Natural candidate rows must be unique")
    raw_ids = arrays["candidate_raw_ids"]
    require(all(len(set(map(int, row))) == 1024 for row in raw_ids), "Candidate raw IDs must be unique")
    require(bool(np.array_equal(arrays["source_values"][..., 0], arrays["base_logits"])), "Cached source GPT column differs from cached base")
    q = arrays["encoded"] if encoded is None else encoded
    base = arrays["base_logits"] if base_logits is None else base_logits
    require(q.shape == (manifest["rows"], 256) and base.shape == (manifest["rows"], 1024), "Recomputed encoder output shape differs")
    require(bool(np.isfinite(q).all()) and bool(np.isfinite(base).all()), "Nonfinite query/base")
    source = np.array(arrays["source_values"], copy=True)
    source[..., 0] = base
    table = torch.as_tensor(np.array(arrays["item_table"], copy=True), device=device).detach()
    outputs = {}
    for arm in ("C", "RL-completion", "RL-residual"):
        if check:
            check(phase="portable-head-load", arm=arm)
        state = safe_torch_state(torch, fixture_root, manifest["weights"][arm])
        model = make_head(torch, state, device)
        values = np.empty((manifest["rows"], 1024), dtype=np.float32)
        with torch.inference_mode():
            for group in buckets(manifest["producer_layout"]):
                rows, slots, size = group["rows"], group["slots"], group["size"]
                positions = torch.tensor(slots, device=device, dtype=torch.long)
                def padded(data):
                    real = torch.as_tensor(np.array(data, copy=True), device=device, dtype=torch.float32)
                    result = real.new_zeros((size, *real.shape[1:]))
                    result.index_copy_(0, positions, real)
                    return result
                queries = padded(q[rows])
                for left in range(0, 1024, 64):
                    if check:
                        check(phase="portable-head-chunk", arm=arm, candidate_column=left)
                    tokens = torch.as_tensor(np.array(ids[rows, left:left+64], copy=True), device=device, dtype=torch.long)
                    selected_items = table[tokens]
                    items = selected_items.new_zeros((size, 64, 256))
                    items.index_copy_(0, positions, selected_items)
                    base_chunk = padded(base[rows, left:left+64])
                    raw_chunk = padded(arrays["raw_features"][rows, left:left+64])
                    source_chunk = padded(source[rows, left:left+64])
                    context = torch.autocast("cuda", dtype=torch.float16) if precision == "cuda-fp16" else nullcontext()
                    with context:
                        result = model(queries, items, base_chunk, raw_chunk, source_chunk)
                    require(bool(torch.isfinite(result).all()), "Nonfinite head scores")
                    values[rows, left:left+64] = result[positions].float().cpu().numpy()
        outputs[arm] = values
        del model, state
    return outputs
