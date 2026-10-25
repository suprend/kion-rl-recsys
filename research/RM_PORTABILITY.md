# Frozen RM portability: source audit and admission contract

Date: 2026-10-25. Mode: source, JSON and filesystem-metadata inspection plus stdlib-only synthetic tests. No numerical module, checkpoint or array was opened by this agent; no model inference, feature build or fit was run by this agent. Root admitted implementation and a root-owned bounded four-state extraction; actual execution and acceptance remain root-owned. The paths below are source-machine provenance, not a claim that inputs are bundled or available on another machine.

## Decision

A genuine compact replay of the existing local RM-r2 is feasible. Use the exact existing `model.cbm` and small **RM** feature blocks, predict again with CatBoost, apply the frozen clipping/calibration/fallback rules, and reconstruct the residual utility from the frozen fit prior. A lookup of saved reward values is only a parity reference and is not model replay.

The actor's `raw_features[...,1068]` are a different schema and cannot be renamed, padded or otherwise treated as RM inputs. The exact RM uses ordered user/context width 2079 and ordered per-pair candidate width 2008. Its numeric tree matrix has width 4087. This scorer has no compatibility-pair additions and no extra context/item categorical columns; it adds one categorical value `item:{original_item_id}`.

For joined replay on the same natural1024 states, a small source-machine feature export is necessary: the audited bridge does not persist its assembled RM inputs. After that one-time export, the extracted package can predict without raw CSVs, the feature provider, Torch, actor imports, user IDs, or user history arrays.

## Exact frozen inputs and transformations

The original artifact directory is `D:/tlab-recsys-2026/artifacts/task2-kion/local-pack-20261020/rm-r2/scorer/`.

| Object | Bytes / value | SHA256 or identity |
|---|---|---|
| `model.cbm` | 163368 | `b6ed4c6427129afcddec017e2f64d9c3fb59fb31fc300e5c9d074964a21ddd62` |
| `scorer.npz` | 7806041 | `d5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4` |
| `manifest.json` | 26957726 | `dfc61f3ebfb9144be3cb86caf70050b72033d9234119145119dbfe90e0ee8507` |
| scorer signature | history_depth4; depth4; 160 trees | `ee14bc4d7375a4d6978a2e1f476cd795369b45ef4250501b2f5992767e9ed261` |
| RM schema signature | U2079 + C2008 | `7776a9d611b04ef8f3ef77fce0a67c1fb045f67142bf4b4bbde24cc469f9bda4` |
| mapping signature | original ID to policy token | `b90df0d1fcce007008b62007e486bc1e9cb0b4ac89c97bb5107e84305e744db4` |
| calibration slope | 1.0148506983671792 | frozen supported-only affine calibration |
| calibration intercept | -0.005568380749024937 | frozen; no refit in replay |
| fit global mean | 0.47802770002556966 | frozen observed-fit fallback |

The native implementation is `bundle-rm-r2/scripts/kion_reward_boosting.py`, SHA256 `cb5446663794ec143b3d72bc2e4a5d8de3599c5a9f3a3c53baaeb44ed4d88727`. Its `FeatureCatalog.blocks`, `catboost_pool`, `RewardModelScorer._raw` and `score_rows` define the replay seam. The feature code remains from the checked parent `bundle-rm-r1`, not a newly constructed current1068 provider.

For each real candidate pair, use this exact order:

```text
numeric = float32(concatenate(context_numeric[2079], candidate_numeric[2008]))
categorical = UTF8 bytes("item:" + decimal(original_item_id))
t = float64(CatBoost.predict(FeaturesData(numeric, categorical)))
support = frozen_sampled_fit_counts[original_item_id] > 0
t[~support] = frozen_global_mean
uncalibrated = clip(t, 0, 1)
calibrated = clip(slope * uncalibrated + intercept, 0, 1)
calibrated[~support] = frozen_global_mean
historical_cached_reward = float32(calibrated)
prior = frozen_item_mean[original_item_id]  # float64, from the same scorer.npz
residual = where(support, float64(historical_cached_reward) - prior, 0)
```

The initial clip before the affine map is required. Clipping only the final affine prediction changes the model. Cold fallback remains the uncalibrated fit global mean after calibration. A missing ID is an error; it is not silently a cold item. `counts` here are the serialized sampled RM-fit target counts used by the historical bridge, not a freshly computed full-raw count vector. The residual prior is the saved item shrinkage mean (`item_alpha=20`), and is not the unshrunk mean or a refit. The historical source writes reward arrays as float32; residual utility subtracts its float64 prior after that roundtrip. A direct float64 residual can fail exact historical parity even when RM inference is correct.

The saved user-level prior vectors and static full-catalog `item_numeric` are unnecessary for this particular tree path when bound dynamic per-pair candidate blocks are supplied. Minimal extraction from `scorer.npz` is `item_ids`, `counts`, and `item_mean`, reduced to the fixture's candidate-ID union. The global mean comes from the manifest. Preserve the original archive hash and an exact ID-to-subset binding in the export manifest. Do not include user-level prior vectors merely because they share the archive.

## Proposed portable fixture

Root admitted the fixed first four states for a bounded source-machine export. Input payloads are not asserted to exist yet. The source implementation is `reproduction/export_rm_fixture.py:run(request,check)`; its external guard is mandatory. The portable implementation is `reproduction/kion_replay/rm.py:run(fixture_root,manifest,output)` and is exposed by the separate reward CLI. It must never share the actor process.

| Field | Shape / type | Role |
|---|---|---|
| `rm_context_numeric` | `[N,2079]`, float32 | One fixed observed s0 context per selected state |
| `rm_candidate_numeric` | `[N,1024,2008]`, float32 | Bound candidate block at the exact state/user/origin |
| `candidate_raw_ids` | `[N,1024]`, nonnegative integer | Exact categorical ID and prior lookup; unique within each pool |
| frozen prior subset | unique item IDs; int counts; float64 means | Recompute support and residual from fitted inputs |
| `model.cbm` | exact 163368 bytes | Actual tree prediction |
| compact RM metadata | JSON | Calibration/global mean, ordered schema, original hashes, selection/clock binding, exported hashes |
| parity references | calibrated float64 and/or historical float32 reward; support | Verification targets only; never model inputs |

Choose N=2 to 4 by fixed source-row order before viewing prediction outcomes. For N=2, the candidate block occupies 16449536 data bytes; for N=4, 32899072. The context block adds 16632 or 33264 bytes. This preserves shared context storage; materializing the repeated full4087 matrix would use 33480704 to 66961408 data bytes. Streaming 128-candidate chunks recreates the native concatenation exactly and keeps peak replay memory small. No reduction of the model's feature coordinates is assumed: tree split feature indices have not been inspected by this static audit, and an unvalidated “used features only” matrix must not be invented.

In a distributable RM-only boundary fixture, use local state row numbers. The tree does not use a raw user ID as a model feature. The item ID categorical must retain its actual value because renaming it changes CatBoost prediction; these are public KION item identifiers. Raw histories, direct source user IDs, target labels and full raw inputs are unnecessary for portable prediction after export. Removing these fields does not establish anonymity: transformed context features still encode demographics and history profiles. Preserve provenance of the feature construction, the retrospective static metadata limitation, the last-watch-date assumption and the fixed s0 interpretation. Audit the exact copied CBM metadata for machine paths before delivery; this source audit did not open the binary.

## What is actually cached

The existing `c-rl-20261024-r1/reward/bridge.py` builds RM U/C feature blocks in memory, scores them, and writes only `rewards.npy`, `support.npy`, `fallback.npy`, and `reward.json`. Its audited `score_cache` output has no numeric/candidate feature arrays. F/Z scientific policy outputs retain actor scores, rewards and related metadata, not the RM blocks. The observed-development `rm-eval-r1/predictions.npz` is also insufficient: `kion_reward_eval.py` writes IDs/clocks, observed outcomes, predictions, x/lengths and support, but deliberately omits `numeric` and `candidate_numeric`. Feature hashes in its JSON do not recreate the features.

Cached **observed-pair** matrices do exist under `D:/tlab-recsys-2026/data/processed/kion-reward-local-20261020-r1/`:

| Role | numeric file bytes | candidate_numeric file bytes | Bound population |
|---|---:|---:|---|
| fit | 831600128 | 803200128 | 100000 observed fit pairs |
| select | 166320128 | 160640128 | 20000 observed selection pairs |
| calibration | 166320128 | 160640128 | 20000 observed calibration pairs |

They are NPY directory inputs with hashes in the ready manifest (`42d7532a5daea8af36198e0a4ba6ba3b134cd66e93dcfd75961913a14624144c`). They permit a cheap independent frozen-RM **observed-pair** fixture by root-owned read-only memmap slicing. That would genuinely recompute CBM predictions, but would not be a matched natural1024 replay or an independent new held-out RM estimate. It cannot be reshaped into N1024 state pools, cross-joined with arbitrary candidate IDs, or mixed with current1068 actor inputs.

## Minimum source-machine generation if natural1024 replay is admitted

1. In a fresh dedicated RM process, verify the exact source, model/scorer, freeze chain, raw source and current-C cache pins. Reuse the existing bridge's immutable serving closure. Do not modify its historical guards or invoke expired native requests.
2. Build the frozen parent `FeaturePack` once with `fit_cutoff="2021-08-02"`, `include_relative=True`. Verify ordered U2079/C2008 names and facade provenance against the frozen scorer. `RawRows.from_pack` supplies strict-past raw histories; it must match the cached current-C x64/lengths exactly. Use the actual query origin in Aug2–15, with every history observation strictly earlier; no date clamp.
3. For each of the fixed first N state rows, compute `pack.user_features(users, origins, observed_x, lengths, history_cutoffs=origins)` once. For candidate chunks, map original IDs through the exact mapping and call `pack.candidate_features(tokens, origins, history_cutoffs=origins, user_ids=users)`. Bind every exported row to its original candidate ID and ordering.
4. Recompute the native public scorer path and diagnostic path, keeping the existing `<=1e-12` pair parity gate and `<=1e-7` singleton/batched candidate feature gate. Compare recomputed calibrated rewards to saved float32 rewards after the native float32 conversion; derive support from exact frozen counts and compare it exactly. No tolerance enlargement or saved-reward substitution.
5. Write only a fresh owned submission output. Export compact source-user-ID-free transformed U/C blocks, exact CBM copy and compact prior/calibration metadata. Hash every included source/input/output; close mappings; verify unchanged source inputs after extraction; record actual runtime/resources and quiescence. Never write into old roots.

The setup still needs the old raw inputs and providers on the source machine. Source metadata shows `interactions.csv` 180032258 bytes, `items.csv` 31836561, `users.csv` 29945389 and basic `features.npz` 2125624. Reducing N does not remove the initialization and source hashing cost of this frozen implementation.

Historical bounds: `rm-eval-r1/report.json` records pack initialization 117.541 s, evaluation 19.400 s, total 154.369 s; its pack budget was 240 s. The exact same bridge's accepted `reward-probe32-r3/receipt.json` records 140.640 s total, 1838682112 peak RSS bytes, 2 native scorer threads, and quiescent cleanup. Its 32x1024 scoring loop took 9.048 s. These are old observations, not measurements of this proposed export. A reasonable root-reviewed proposal is one sequential process, CPU2 threads, 8 GiB cap, 2 GiB reserve, init budget300 s and slice budget60 s with an external total timeout. Root decides admission; no new build is authorized by this note alone.

## Portable environment and acceptance

Observed native RM environment: Python3.12.14, NumPy2.5.4, CatBoost1.2.10, SciPy1.18.1. These are the existing reward-worker top-level pins, not a transitive lock or wheel archive. A compact CBM replay requires NumPy and CatBoost directly; SciPy is retained in the observed environment because CatBoost imports and its dependency chain must be tested, not inferred. Keep it separate from the actor's NumPy2.2.6/Torch environment and run as a fresh process. The native worker selected exact Windows package bytes before numerical import; a new portable worker should record its actual interpreter/package metadata and validate output parity rather than pretend to satisfy old absolute-path/backend hashes.

A portable adapter must be additive. Keep the historical source/manifests unchanged. Its new manifest must identify the adapter source hash, original scorer signature, original model/archive hashes, reduced fixture hashes, ordered schema, extraction binding and separate environment. No automatic sklearn substitute, model JSON approximation, fit, calibration repair, optimizer call, network access, subprocess provider build or external dataset lookup belongs in portable prediction.

Acceptance requires: synthetic clipping/fallback/dtype tests; failure on feature-width/ID/schema/hash mismatch; real root-owned frozen CBM inference from a fresh extracted package; finite calibrated range, exact support, native float32 reward parity and prior/residual parity; source/input immutability; independent reviewer inspection. The eight initial stdlib tests passed; they exercised clipping order, cold fallback, invalid counts/nonfinite values, no unguarded export, fixed selection, parse and inert imports. They do not establish actual prediction. Do not mark natural1024 RM replay accepted until its fixture has actually been exported and genuine recomputation has passed.

The export request uses contract `kion-rm-fixture-export-v1`, `row_positions=[0,1,2,3]`, a fresh `fixture_output` below `reproduction`, the exact cache/bridge hashes above, explicit `source_hashes` and `input_hashes`, native `catboost_site`/`catboost_pins`/`numpy_pins`, `init_seconds<=300`, `slice_seconds<=60`, `maximum_seconds<=480`, and integer zero for `supervised_updates`, `new_rm_fits`, `final_test_reads`, and `rl_updates`. State-array and historical reward/support payload pins must be explicit in the request as well as their pinned manifests. The source implementation writes a standalone `manifest.json` with contract `kion-portable-rm-fixture-v1`, `arrays`, `weights.RM` and `rm` metadata. The root guard owns the memory/process limits; this module cannot enforce a kernel memory cap by itself.

## Inspected sources

- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/reports/reproduce.md`, `code-manifest.json`, `code/recipes/source-requests.json`, `code/recipes/excluded-inputs.json`, and `code/requirements/reward-worker.txt`.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/reward/bridge.py`, accepted probe32 `reward.json`, and `operations/reward-probe32-r3/receipt.json`.
- `D:/tlab-recsys-2026/artifacts/task2-kion/local-pack-20261020/freeze-rm-r2.json`, RM-r2 scorer manifest, parent `bundle-rm-r1/scripts/kion_local_pack.py`, and `bundle-rm-r2/scripts/kion_reward_boosting.py`.
- `D:/tlab-recsys-2026/scripts/kion_reward_eval.py`, `local-pack-20261020/rm-eval-r1/{manifest,report}.json`, and observed-pair dataset manifest/resource observation plus file metadata.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/reward/utility.py` for exact serialized sampled-fit prior and float32-reward-to-float64 residual semantics.
