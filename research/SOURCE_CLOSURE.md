# Source closure and execution boundary

The historical scientific sources are now unpacked into this submission. Basic reading and the source-integrity CLI do not depend on either old archive. The [new source manifest](../source/manifest.json) contains 284 exact copies, 6,989,216 bytes. The copied code itself is unchanged; 257 Python files passed AST parsing and in-memory compilation without execution.

The full-study snapshot is anchored by [its original manifest](../source/fullstudy/code-manifest.json), SHA-256 `6b054627fdcfd2ee205df2e3871d8da47e6d3d7cbdf47a58897968182f2e4085`: 261 files and 6,553,172 bytes. Its source commit `acf27d1c9136bfe2f0e5a9bf948156f760b84a58` is contextual because the original workspace also contained ignored/untracked sources; the per-file bytes and hashes are authoritative. The [follow-up selection](../source/matched-kl/selection-final-r1.json), SHA-256 `801c0337ab63aead24b6fa1b13500d18b5d3ce0b4d7e55138f3ff897d8883c8b`, authorizes the 20 follow-up Python copies.

The [static audit](../source/static-audit.json) additionally checked the source pins in all 18 explicitly selected follow-up request JSON files: 1,897 references to 139 distinct sources. Every pin matches a copied source; none is missing or mismatched. Those request files were read only as bounded metadata. Their input payloads were not opened, and their old deadlines/output paths are not new launch instructions.

## Code needed for each tier

For source verification, only [verify_sources.py](../source/verify_sources.py), its manifest and the declared source files are needed; the checker uses the standard library. Its independent historical counterpart [verify_bundle.py](../source/fullstudy/code/tools/verify_bundle.py) also checks the invented ranking fixture. These checks have passed and do not execute producers.

Small numerical replay belongs to the new [reproduction adapter](../reproduction/README.md) and its bounded exported inputs. It must select only the necessary scientific kernels and explicitly relocate their inputs. The old campaign admission/supervision code, request builders, fitting entrypoints, raw data, full feature stores and nested historical ZIPs are not prerequisites for this tier. Its actual numerical acceptance is reported by the adapter owner, separately from this source check.

For the full recipe, the preserved tree includes the following implementation families. The common prefix below is `source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/`; these are source locations, not commands to execute during this no-training preparation.

| Stage | Preserved implementation |
| --- | --- |
| Raw vocabulary and temporal splits | `ready-pack-20261020/provider/runtime-native-r5/scripts/kion_prepare.py` |
| Feature assembly and schema | `ready-pack-20261020/provider/runtime-native-r5/scripts/kion_complete_pack.py` and its local feature modules; `configs/datasets/kion-series-v1.json` is relative to the copied project root |
| Text features | The same runtime's `kion_text_embeddings.py`; pinned `intfloat/multilingual-e5-small` revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` and five `MODEL_FILES` identities are in the source |
| GPT model and supervised training | The same runtime's `kion_complete_model.py`, `kion_train.py`, `kion_complete_train.py`; recovery implementation under `night-20261023-r1/` |
| Bounded query/candidate construction | `deadline-20261022/perf-batch-query-r1/kion_complete_pack.py` with its sibling `query_batching.py`; `precompute-e1-20261022-r1/query-store/`; `full-joint-e1-20261024-r1/candidates.py` |
| A/B/C bank, heads and evaluation | `three-head-262k-20261024-r1/{bank,heads,evaluate}.py` |
| Local RM data, fit and calibrated scoring | `local-pack-20261020/bundle-rm-r1/scripts/kion_reward_{data,boosting}.py`; scorer and calibration functions are in `kion_reward_boosting.py`, with the retained r2 snapshot under `bundle-rm-r2/scripts/` |
| Arbitrary-candidate RM bridge | `c-rl-20261024-r1/reward/bridge.py` |
| Completion and residual H1 policies | `c-rl-20261024-r1/policy/policy_v2.py`; `c-rl-final-20261025-r1/reward/utility.py` and `policy/residual.py` |
| Matched-KL multiseed policy/evaluation | `source/matched-kl/policy/{policy,evaluate}.py`, relative to the submission root |
| Post-processing and figures | Full-study `interpretation/{analyze,analyze_v2,rollout_summary,plot_results_v4,plot_rollout}.py`; follow-up `source/matched-kl/interpretation/` and `reports/figures.py` |

The [historical parameter recipe](../source/fullstudy/code/recipes/source-requests.json), [execution-stage index](../source/fullstudy/code/recipes/execution-stage-index.json), [excluded input descriptors](../source/fullstudy/code/recipes/excluded-inputs.json) and [import inventory](../source/fullstudy/code/recipes/import-inventory.json) are included directly. The input descriptors contain 120 excluded references with provenance hashes; they do not make the payloads available. The [raw provenance metadata](../evidence/raw-provenance/manifest.json) and [verified-download metadata](../evidence/raw-provenance/verified-download.json) retain the archive/member hashes without copying or rereading the CSV data.

`kion_prepare.py` defaults to context32, 150,000 train rows and 3,000 evaluation rows. Rebuilding the full historical splits requires explicit `--context 64 --max-train 1000000000 --max-eval 1000000000 --seed 2026`, together with authorized raw inputs and a new output location. Those settings describe the recipe; no preparation or training ran here. Reproducing the recovered E2 lineage additionally requires its original state and five-epoch scheduler horizon; a fresh two-epoch fit does not establish byte identity with the recovered checkpoint.

## Concrete relocation and dependency gaps

The source snapshots deliberately retain the original bytes. A root/path JSON substitution alone cannot close these seams:

- Historical `control/common.py` pins the original base executable and its hash, PAS root and runtime source. `kion_ready_runtime.py` pins the PAS and actor package sites and the original executable. The native launchers admit specific Windows drives and campaign-owned output trees.
- `three-head-262k-20261024-r1/bank.py` pins a C-drive bank cache and a D-drive prepared-data location. `precompute-e1-20261022-r1/full-preparation/worker.py` pins an E-drive SQLite store with a 110GB storage contract. That full-preparation route is unrelated to the bounded replay tier.
- `reward/bridge.py` requires the repository's `.venv/Lib/site-packages`, exact NumPy/CatBoost native files, a `cp312-win_amd64.pyd` backend and a fresh process. Actor and RM NumPy scopes differ and must remain isolated.
- Follow-up `policy/policy.py` derives old/full-study roots as historical siblings. In this submission the originals reside under `source/fullstudy/code/tlab-recsys-2026/artifacts/task2-kion/`; the new adapter must supply an explicit source/input map rather than call this entrypoint unchanged.
- `kion_people_dynamics_fast.py` imports `scripts.kion_people_dynamics`. Its core source is present with the pinned hash, but the runtime package parent must be on the import path. The `scripts` namespace is not a pip dependency. `query_batching.py` is present beside the actual patched provider, under `deadline-20261022/perf-batch-query-r1/`.
- Static external import roots are NumPy, Torch, Transformers, psutil, Matplotlib, CatBoost, SciPy, ReportLab and conditional `threadpoolctl`. The original [dependency inventory](../source/fullstudy/dependencies.json) records the established actor/RM/figure versions, but no historical `threadpoolctl` version or complete native environment lock is established.
- Frozen weights, tokenizer/encoder files, assembly manifests, histories, candidate graphs and large caches/arrays are external prerequisites for full numerical replay. Their hashes document provenance and do not promise availability on a different machine.

The bounded pattern scan found 148 machine/environment references in 55 preserved files and no credential-shaped literal. It checks selected token/key/credential URL patterns only; original machine username paths remain visible. It is not a general proof of secret absence or a numerical privacy audit. No excluded models, raw histories or array payloads were opened for this check.

Source verification passed on the original workstation. Clean-machine numerical execution, exact FP16 parity and the complete raw-data/training pipeline remain separate claims requiring the new adapter, explicit prerequisites and actual acceptance receipts. No new training, optimizer step, RM fit, external download or old-tree mutation was performed by this source-bundle task.
