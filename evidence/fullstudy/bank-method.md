# Frozen E2 nested 262k bank

`bank.py` imports only the standard library. `run(request, check)` is a heavy
root-admitted operation and has no automatic entrypoint or resume. Source tests
never open real arrays, a provider, QueryStore, teacher checkpoint or GPU.

The contract is `kion-three-head-262k-bank-v1`: exactly 262,144 known train
rows, 32 candidates, 1,068 raw features, 256 context dimensions and two source
values. Largest-remainder date quotas use the complete known train census.
Every old 32,768-row pilot row is mandatory; only the residual quota is drawn
without replacement with seed 20261024 and the original per-date seed formula.
This explicit constraint produces a nested sample, not a fresh unconstrained
262,144-row draw. All rows are unique and sorted by (date, original row index).
Coverage never affects eligibility. A date quota smaller than its mandatory
old count fails rather than altering the quota or dropping old rows.

The only allowed new physical bank directory is
`C:/Users/Михаил/.cache/codex-kion/three-head-262k-20261024-r1/train`.
`bank.json` lives there with the arrays. It is created exclusively, only after
every array finishes, streaming hashes finish and context teardown succeeds.
An existing output directory fails; source and other experiment artifacts stay
beside this document.

Required additional request fields:

- `reuse_bank_dir`: exact original
  `D:/tlab-recsys-2026/artifacts/task2-kion/three-head-pilot-20261024-r1/cache/train`.
- `reuse_bank_manifest_sha256`:
  `a64decc0990a4644e878d3ab830ea81035d431ca6ff6d664aaff9a03d8c8cf36`.

The old completed manifest, teacher/config lineage, context signature, feature
columns and schema, candidate configuration, all ten array payload SHA256s,
file sizes, read-only NPY headers and dtypes must match. Old row identities are
checked against the original train split. All nine old row arrays are copied
verbatim in batches of at most 32; old context, candidate IDs, logits, raw
features and source values are never recomputed. The exact old item-table file
is copied after bounded comparisons with the current frozen table. Only the
remaining 229,376 rows invoke teacher context, sampler and feature computation.
Every new batch contains one origin and at most 32 rows.

The unchanged original corrected candidate source SHA is
`6923f4e63ce6278de3297449c8d59fa498b9117f2cdc85537d7e9bd4e0acc09d`.
Its `draw` uses SHA256 of `seed:original_train_row` for a local RNG, so negative
drawing does not depend on bank ordering or batching. Candidate configuration
and source identity remain fixed. Each positive is first, including natural
misses, followed by 31 distinct negatives from GPT512 + strict7d-pop256-new +
strict-origin-KNN256-new. All same-user/same-origin positives are excluded.

The original E2 teacher, original runtime/source closure, provider/config
reconstruction, read-only QueryStore, full event-store and prepared-data pins,
frozen/eval CUDA execution and `open_context`/`close_context` interfaces are
preserved. Teacher SHA256 is
`5e7101857c2dbe100fb4e1587c0369abc95c108303ed149123ee183f62aeac26`;
config SHA256 is
`e3b4b1730895fceb1b7d4ea6357aea783c046f5e87cacd7d4ff933cf94d45d45`.
No optimizer, GPT training, RL or final-test access is introduced.

The original request fields still apply: `teacher`, `bank_output` (or
`bank_dir`), `source_hashes`, `input_hashes`, `provider_hook`,
`performance_hook`, `dynamic_cache_path`, `q_source_path`,
`candidate_source_path`, `graph_cache`, and optionally a separately admitted
`q_adapter_path`. `train_rows` and `seed`, if supplied, must match fixed values.
`check(**fields)` is called at integrity, allocation and bounded batch stages;
the root owns resource/time enforcement and admission.

Train graphs remain read-only, with missing or mismatching daily descriptors
failing. Evaluation can specify `validation_graph_cache` equal to the old
pilot's `cache/validation-graph`, `stage="evaluate"`, and
`allow_validation_graph_build=false`. Only origin 738011 routes to that existing
read-only graph, with construction denied. The original explicit evaluate-only
owned graph-build seam remains available separately.

| Array | Shape | Dtype |
|---|---|---|
| encoded | [262144,256] | float32 |
| candidate_ids | [262144,32] | int32 |
| base_logits | [262144,32] | float32 |
| raw_features | [262144,32,1068] | float32 |
| source_values | [262144,32,2] | float32 |
| row_indices | [262144] | int64 |
| user_ids | [262144] | int64 |
| target_ids | [262144] | int32 |
| dates | [262144] | int32 |
| item_table | [vocab,256] | float32 |

`source_values` contains only original numeric metadata columns 12/13:
detached frozen E2 retrieval logit and `log1p(strict7d_count)`. Owner,
nomination, rank and missing flags are excluded. Output is disk-backed; the
raw map is approximately 33.4 GiB. To release touched pages and bound native
RSS, all output maps are flushed, fsynced, closed and reopened after at most
1,024 touched rows. During reuse, old read-only maps are also closed/reopened
per 1,024 rows. They are all closed before new computation. Hashes stream
through files; no full feature bank is loaded into RAM.

The descriptor records `reused_rows=32768`, `new_rows=229376`, the original
manifest/source pins, copied array names, old date quotas, the nested selection
constraint and mapping policy. Its sample scope remains a train subset.
`all_training_rows_retained` inside the unchanged candidate contract describes
sampler behavior for submitted rows; it does not claim this bank has every row
in the full training split.

Implementation-agent verification is limited to stdlib selection, lineage and
source tests. Numerical parity, actual memory bounds, provider/store evidence
and successful large-bank construction require the root-owned native run.
