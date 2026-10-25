# Frozen three-head 262144-row trainer

`heads.py` imports only stdlib until root invokes `run(request, check)` or supplies
Torch to the scorer API. It never opens GPT/provider code, changes frozen E2,
evaluates validation, selects a checkpoint, resumes, or reduces the epoch budget.

All three heads start fresh with identical independently owned parameters and
independent AdamW optimizers/scalers. Their architecture is unchanged: bias-free
1070→32 projection; q256, item256, q*item256 and projection32 feed an
800→256→1 GELU MLP, without dropout. Only the final output is initialized to zero,
so initial scores equal frozen GPT logits. Raw masks are A=none, B=coordinates
142:536 and 907:1068, C=all1068. All retain detached source columns12/13:
current GPT logit and log1p strict7d count. Source14 is rejected. All frozen
representations, features, logits and item vectors are detached.

The bank contract is `kion-three-head-262k-bank-v1`; the head contract is
`kion-three-head-262k-head-v1`. The physical training bank is
`C:/Users/Михаил/.cache/codex-kion/three-head-262k-20261024-r1/train`.
`ARRAY_SPEC` requires encoded `[262144,256]` float32; candidate_ids
`[262144,32]` int32; base_logits `[262144,32]` float32; raw_features
`[262144,32,1068]` float32; source_values `[262144,32,2]` float32; row_indices
and user_ids int64; targets/dates int32; and the frozen item_table float32.
Only a completed unfiltered train bank with same-day positives excluded is valid.

Training is exactly three epochs, B128, 2048 updates per epoch and 6144 updates
per head, with CE32 positive column0. A/B/C share every batch and candidate
gather, with sequential independent updates. No shortened fallback is accepted.
AdamW uses lr1e-4, weight decay0.01, clipping1. CUDA uses FP16 autocast with
bounded nonfinite retries at the same cursor then FP32; CPU uses FP32.

Every epoch uses exactly the original global ordering policy:
`order = list(range(262144))`, followed by
`random.Random(20261024 + zero_based_epoch).shuffle(order)` and consecutive B128
slices of that order. There are no date blocks or date-local sequences of SGD
updates. This preserves the earlier pilot's shuffle policy while expanding the
number of rows; dates in the bank's physical sort order do not constrain batches.

For each B128 the trainer sorts only its 128 physical row indices, gathers the
raw rows into owned memory, and immediately closes the raw mmap before restoring
the exact logical batch order through an inverse permutation. All A/B/C inputs
therefore have the original globally shuffled order. Two raw batch copies can
coexist briefly, totaling 34,996,224 bytes, plus the OS pages mapped during that
single gather; no view into the 35.84GB raw bank escapes. The next batch reopens
the read-only mmap after verifying its path, size and mtime identity, without
another full hash. Smaller arrays stay mapped and are gathered directly in
logical order. The full raw bank is never copied or retained in RAM. This costs
6144 bounded mmap opens/gathers rather than block-local I/O; runtime, I/O
amplification and peak RSS remain root-owned measurements, without an ETA claim.

`epoch_schedule(epoch)` and `epoch_record(epoch)` are stdlib. Each completed
epoch record stores SHA256 of compact JSON global row indices in actual
traversal order, plus rows and updates. The digest is computed incrementally
without a full serialized JSON string. Receipts pin the original global shuffle
algorithm, sorted-physical/inverse-logical raw gather policy, all three epoch records, bank manifest,
teacher, full1068 schema and named source2 columns. Checkpoints embed the same
receipt; only the external checkpoint hashes are omitted from their embedded
receipt to avoid a self-reference.

Requests need `bank_dir`, fresh `output_dir`, explicit matching `teacher`,
`feature_schema_sha256`, `source_columns`; fixed defaults are epochs3,
seed20261024, batch128, device cuda. `bank_manifest_sha256` pins the bank;
completed evaluation requires it. Optional `bank_verification_receipt` reuses
`bank-verification.json`: initial reading streams every array hash once, and
reuse requires identical manifest hash plus per-array resolved path, complete
NPY bytes, mtime_ns and declared SHA. Changed identities are rehashed; all
headers/dtypes/shapes are still checked. This assumes root-owned immutable
artifacts, not hostile-write protection.

`check(**fields)` runs before each array hash and before and after each common
batch, retaining `head-train` and `head-batch-complete` fields. Optional
`deadline_utc` is checked before each raw batch gather. A stop saves each
current parameter/optimizer/scaler/RNG state to `partial-head.pt` plus
`stopped.json`; partials are never accepted for evaluation and cannot resume.
Existing complete or partial checkpoints refuse fresh overwrite.

`preflight_heads(request)` must run before provider imports. It uses stdlib to
check exact completed budget, every deterministic epoch schedule, bank manifest
pin and teacher/schema/source lineage, then hashes all three small checkpoints.
`load_heads` repeats preflight before decoding, validates each embedded receipt
and configuration and attaches `model.pilot_receipt` for API compatibility.
CPU serialization captures no CUDA RNG and never initializes CUDA.

Public interfaces remain `run(request, check)`, `make_heads(torch, seed, device)`,
`load_heads(request, torch, device)` and
`score_heads(models, q, item_table, ids, base, raw1068, source2, torch)`.
Scoring returns A/B/C `[batch,candidates]`; the evaluator owns no_grad, autocast,
candidate chunking, all-row unknown/off-pool accounting and raw-ID tie ordering.
Base and residual are added as float32 even with half-precision base inputs.

Run stdlib source checks through repository PAS Python with `-B` and
`test_heads_source.py`. The separate explicit `--ml` fixture is root-owned and
uses only tiny seeded CPU tensors: common/disjoint initialization, zero output,
mask/source sensitivity, permutation/chunk invariance, frozen gradients,
independent CE32 update and full-state serialization/load parity. The fixture's
completed metadata uses the real exact budget but does not train 6144 steps or
claim production training. No ML execution is part of source checks.
