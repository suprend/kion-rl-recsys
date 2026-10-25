# Scientific three-arm live-prefix diagnostic

`rollout.py::run(request, check=None)` is an inference-only entrypoint for the
root-owned native controller. Import and unit tests are stdlib-only. Writing this
adapter does not execute a model, read a numerical cache, or authorize a job.
Old source, caches, receipts, checkpoint-selection schema and STOP are read-only.

The fixed grid is 16 old label-blind positions, in the exact order saved before
the first RL fit, times `C`, `RL-completion`, `RL-residual`, times greedy and
categorical selection, times `none` and `exclude-generated`: 192 trajectories,
960 selected steps, plus 16 static RM-greedy unique top5 controls. Both RL arms
must be separately pinned actual completed latest1000 checkpoints. The evaluator
loader checks their training receipts and common original-C ancestor. Old
`latest`/`selected` fields are rejected; residual never occupies `selected`.

Native request fields in addition to the old policy/bank context and source pins:

- `policy_module`, `evaluate_module`, `utility_module`, `residual_module`: each
  `{path, sha256}`; direct imports require the same entry in `source_hashes`.
- `scientific_checkpoints`: `RL-completion` and `RL-residual`, each
  `{path, sha256, step: 1000, training: {path, sha256}}`.
- `cache_dir`: original completed screen256/val cache; exact original-row mapping
  is checked for every selected position. `reward_dir` binds its frozen RM.
- `utility_dir`: completed residual utility receipt from `reward/utility.py`;
  utility/prior/support are bound to the same cache and RM, with identical support.
- `rollout_selection_pin`: `{path, sha256}` for the old
  `policy/rollout-selection-r1.json`; `rollout_positions` is precisely
  `[253,2,30,246,204,158,96,50,128,244,255,36,12,109,171,236]`.
- `seed: 20261024`; every matched trajectory uses `seed + position`.
  Optional `rollout_selection` must equal `['greedy','categorical']`; optional
  `repeat_masks` must equal `['none','exclude-generated']`. No shrinking.
- `preregistration`: root-supplied `{path, sha256}` protocol pin, verified and preserved in the output.
  `deadline_utc` and the controller's checks govern admission/termination.
- `rollout_output`: fresh JSON file below this new `policy/` directory.

Every step encodes the actual synthetic ID prefix while preserving original
producer B, slot, ordered peer encoder inputs, and maximum trim width. It calls
the real full-vocabulary `retrieval_logits(q_group, table)` and only then gathers
the unchanged natural1024 pool. The head uses the producer-size/slot padding
adapter. At s0 the original C score gate remains `atol=.003, rtol=.001`; changed
top10 ranks must meet the old `.006 + .001*(abs(a)+abs(b))` cached margin rule.
Passing score/rank evidence is mandatory in each trajectory's saved first step.

Raw1068, transformed user features, source13, RM state, fit item prior, observed
history and clock remain frozen at observed s0. Generated IDs add neither fake
watched percentages nor dates. Source12 alone tracks the current full-vocabulary
GPT base score. Tensor-version guards reject in-place model/head mutation.

All arms report both completion and `completion - observed-fit-item-prior` on
the same support. Their neutral utilities center separately on the supported
original C distribution at the unmasked observed s0 and remain fixed thereafter.
Unsupported actions contribute zero centered utility and null raw utility;
all1024 actions participate in softmax/KL, except the shared explicit generated-ID
mask. Both raw full returns are null if any of five selections is unsupported;
partial supported sums remain explicitly named. Discount is .9, temperature 1.

Only a completed grid followed by successful cache/provider teardown publishes
the output JSON. Root must then run `interpretation/rollout_summary.py::run` via
admission, with pinned `rollout`, exact `expected_positions`, fresh owned
`output`, required `items: {path, sha256}` catalog pin and seed20261024. It validates the full
cross product, matched seeds, masks, frozen bindings, finite metric arithmetic,
support neutrality, producer shape/projection/parity evidence, returns, repeats,
H63 flags and all16 controls. It emits 12 group rows, 8 matched-C comparisons,
per-step metrics, concentration, and first3 fixed positions with all12 slates and
titles. Missing or empty catalog names for any of the first3 cases reject publication.

Local source verification (from PersonalOS root):

```powershell
.\pas.ps1 python -B D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/test_rollout.py -v
.\pas.ps1 python -B D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/test_rollout_summary.py -v
```

These are pure helper/source and synthetic JSON metadata tests. Native inference,
actual score/rank parity, full-vocabulary projection output, measured timings and
cleanup remain root-owned runtime acceptance checks. Synthetic steps have no new
observed relevance labels. These are descriptive surrogate diagnostics, not
observed watchtime, causal retention, five-step RL training or population claims.
