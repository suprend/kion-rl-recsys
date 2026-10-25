# Scientific three-arm evaluator

`evaluate.py` preserves the completed original C, actual completion latest1000 and
the separately trained residual latest1000 as `C`, `RL-completion`, `RL-residual`.
The old selected0 is not a scientific arm and is never relabeled as residual.
Both RL checkpoint pins include their own completed training receipt and the same
original C ancestor. The residual loader enforces its separate checkpoint contract.

No numerical stage runs on import. `test_evaluate.py` uses only synthetic stdlib
fixtures; passing it establishes contracts and source structure, not numerical
acceptance. Only the root may execute the numerical entrypoints through the
existing controlled worker, runtime, deadlines and caps. No control/admission
allowlist is changed. The accepted controller functions are `score` (screen) and
`evaluate` (full); direct Python aliases are `screen` and `full`.

## Request construction

Start with the existing completed-cache/full-evaluation request's frozen native
runtime, data, bank, head, graph, schema and `source_hashes` fields. Preserve those
fields exactly, including `allow_validation_graph_build: false`, existing
`validation_graph_cache`, `head_dir`, `device`, and `precision: cuda-fp16`. Pins in
`source_hashes` are still checked by the unmodified old loaders. Add the fields
below. Placeholder values must be replaced from actual completed receipts; these
templates are not executable evidence.

Common addition (use the actual hashes of the reviewed new sources):

```json
{
  "deadline_utc": "2026-10-25T05:45:00Z",
  "policy_module": {
    "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/policy_v2.py",
    "sha256": "da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba"
  },
  "residual_module": {
    "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/residual.py",
    "sha256": "<reviewed residual source SHA256>"
  },
  "utility_module": {
    "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/reward/utility.py",
    "sha256": "<reviewed utility source SHA256>"
  },
  "scientific_checkpoints": {
    "RL-completion": {
      "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/rl-r2/head-1000.pt",
      "sha256": "436d735d67fd861e6b925719b9e3597582ec405eed1b13ede3db3671e1ce37b7",
      "step": 1000,
      "training": {
        "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/rl-r2/training.json",
        "sha256": "cfdfc225be553305a3d6a8af7c5419227255b44827ccf7e3760a86b3e5dbf8bf"
      }
    },
    "RL-residual": {
      "path": "<actual new residual latest1000 checkpoint>",
      "sha256": "<actual checkpoint SHA256>",
      "step": 1000,
      "training": {"path": "<actual residual training.json>", "sha256": "<actual receipt SHA256>"}
    }
  }
}
```

The illustrative deadline is 08:45 MSK, leaving cleanup time before the 09:00
numerical cutoff. Root must set each admission deadline to a still-future time
within the authorized cutoff and not treat this example as authority to extend it.
The evaluator converts module pin objects to path strings in a copied subrequest
for the older residual/utility loaders, propagates verified source hashes and
rejects conflicting pins. The caller's request is not mutated. Include the utility
module pin for full inference too: residual checkpoint loading uses that source
adapter without opening any utility data or RM model.

Screen additions, controller `function: score`:

```json
{
  "stage": "evaluate",
  "cache_dir": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/screen-cache-r1",
  "reward_dir": "<existing completed screen256 RM bridge directory>",
  "utility_dir": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/reward/residual-r1/screen",
  "legacy_screen_outputs": {
    "C": {
      "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/screen-0-r1.json",
      "sha256": "e2f6bf09433aece8e6bdc55da0a43d3d04a65d767204dd9420d0a683a6f81a5a"
    },
    "RL-completion": {
      "path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/screen-1000-r1.json",
      "sha256": "cd5b7f1e1a93b711f860fb02e8ac89f05f86b0f60e3be371d83399ab9f05cff4"
    }
  },
  "screen_evaluation_dir": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/screen-scientific3-r1"
}
```

This opens the existing cache and its pinned RM/derived-utility arrays; it never
opens the bank, provider, encoder or RM model. The original shape/slot/chunk64
scorer is reused. C must reproduce cache reference scores exactly. Every original
diagnostic field for C and completion must equal the accepted old JSON value
exactly, without rounding or tolerance. All three arms use one immutable pool.
The native request uses admission stage `evaluate` and function `score`; `screen`
is the saved output's internal cohort stage, independent of the native job stage.

Full additions, controller `function: evaluate`:

```json
{
  "stage": "evaluate",
  "validation_selection": "D:/tlab-recsys-2026/artifacts/task2-kion/three-head-262k-20261024-r1/validation-selection.json",
  "validation_selection_sha256": "<existing frozen selection SHA256>",
  "original_evaluation_dir": "D:/tlab-recsys-2026/artifacts/task2-kion/three-head-262k-20261024-r1/evaluation",
  "original_evaluation_sha256": "ebc6cb8b8eb682298723aba8f1d24a3250c286b20e91c70f51e159e7ba000e56",
  "full_evaluation_dir": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/full-scientific3-r1",
  "parity_atol": 0.003,
  "parity_rtol": 0.001
}
```

Use the established parity tolerances from the pinned original C-completion
request; the values above show that request schema and must not be loosened after
seeing results. No utility/RM input is needed for full4096. The frozen read-only
context is opened once. The original B128/chunk64 q, base, raw1068 and source12/13
arguments are reused across all three heads inside the same chunk. Exactly512
provider feature calls are required. Natural1024 IDs must equal the original
saved pool row by row; original row/user/target/catalog and coverage bindings are
checked. C's original NumPy allclose and raw-ID rank-margin gate is retained.
The output scores/top10 are writable `.npy` memmaps flushed batchwise; full raw
features are never saved or retained as a new full cache.

## Output contract

The fresh output directory contains `evaluation.json` only after scoring, parity,
array flushing/closing and (full stage) successful provider cleanup. Incomplete
directories have no completed manifest and must never be resumed or analyzed as
completed. `evaluation.json` has contract `kion-scientific3-evaluation-v1`, stage
`screen` or `full4096`, explicit arm/checkpoint pins, `rows == metric_denominator`
(256 or4096), temperature1, raw-ID tie policy and protocol flags. SHA256, bytes,
dtype and shape descriptors bind every exported array. All paths are local array
filenames. The analyzer must pin this manifest and verify every array descriptor.

| Array key | Dtype | Shape | Availability |
|---|---|---|---|
| `<arm>_scores`, `<arm>_top10` | float32, int32 | n×1024, n×10 | both |
| `row_index`, `user_id`, `target_token` | int64, int64, int32/original dtype | n | both |
| `known_target`, `candidate_coverage` | bool | n | both |
| `pool_ids`, `candidate_raw_ids` | int32, int64 | n×1024 | both |
| `catalog_raw_ids` | original int64 | full catalog | full only |
| `raw_reward`, `utility_residual`, `utility_prior` | float64 | 256×1024 | screen only |
| `support`, `fallback` | bool | 256×1024 | screen only |
| `history_tokens` | int64 | 256×64 | screen only |
| `strict7d_count` | float64 | 256×1024 | screen only |
| `candidate_sources` | float32 | 256×1024×14 | screen only |

Screen `catalog_size` is `item_table.shape[0]-3`; use candidate raw IDs for exposure
without inventing absent catalog mappings. `utility_prior` is the frozen observed
item mean returned by the derived-utility adapter, not a second policy reward.
`utility_residual` is that adapter's actual utility and its pinned formula appears
in `utility.metadata`. Unsupported storage values are not labels: each utility is
centered at C's supported conditional expectation and assigned zero advantage
outside support, while all1024 actions remain in the softmax/KL normalization.
The original completion diagnostics plus both centered utility expectations,
supported raw numerators, KL, entropy, support denominators, top1 share and HHI
are exported for each arm. Source14 names come from the immutable cache contract;
history tokens retain special/padding values and require tokens>=3 for H63 overlap.

Full metrics retain every original query, including unknown and known offpool
targets as zero-relevance misses. Full arrays do not contain RM reward estimates;
screen proxy diagnostics must remain explicitly n256 in reports.

If review/preflight cannot accept this generalized adapter within the development
cap, root may use the predeclared smaller scope: all three arms on screen256 and
the unchanged old C-completion full evaluator separately. Never fabricate a third
full arm, silently shrink4096, run three raw passes or claim mixed cohorts match.
