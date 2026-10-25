# Preserved scientific sources

This tree preserves the exact scientific and helper code used by the full study and the matched-KL follow-up. It is a source-preservation tier. The historical native launchers retain the original workstation paths and checks; copying this tree does not make those launchers portable.

The [manifest](manifest.json) lists 284 copied files, 6,989,216 bytes, with their original locations, new relative paths, sizes and SHA-256. It covers 261 files from the approved full-study source manifest, two full-study provenance documents, 20 approved follow-up Python files and the follow-up selection manifest. It does not hash itself or the new explanatory/checking files. The submission-wide manifest covers those files separately.

- [fullstudy/code-manifest.json](fullstudy/code-manifest.json) and [fullstudy/dependencies.json](fullstudy/dependencies.json) retain the approved historical metadata. The 261 files are unpacked under `fullstudy/code/`; no old ZIP needs to be opened.
- [matched-kl/selection-final-r1.json](matched-kl/selection-final-r1.json) retains the approved follow-up selection. Only its 20 Python sources are copied here; its other entries describe historical evidence and are not an inventory of this new source tree.
- [static-audit.json](static-audit.json) records the bounded source/path/import checks and the follow-up request-source closure. It distinguishes preserved code from unresolved execution dependencies.
- [SOURCE_CLOSURE.md](../research/SOURCE_CLOSURE.md) maps the scientific stages to their sources and explains relocation gaps. The current [reproduction instructions](../REPRODUCING.md) define file availability and replay status for this clone. The [reproduction overview](../reproduction/README.md) separates the supported checks from the historical raw-to-training recipe; old archive receipts do not replace the current instructions.

From any working directory, Python 3.12+ can verify the preserved bytes and parse/compile the 257 Python sources without executing them:

```text
python -B /path/to/submission/source/verify_sources.py
```

This command needs only the standard library. It opens the local source manifest and its declared source files, then checks the two bounded source subtrees for undeclared files. It does not import scientific producers or open models, datasets, numerical arrays, subprocesses or the network.

The 261-file historical source bundle can also be checked independently:

```text
python -B /path/to/submission/source/fullstudy/code/tools/verify_bundle.py --report-root /path/to/submission/source/fullstudy
```

That checker additionally evaluates an explicitly invented ranking fixture. Neither command establishes model inference, clean-machine numerical parity or training reproducibility.

Historical actor dependencies are NumPy 2.2.6, Torch 2.8.0+cu126, Transformers 4.55.4 and psutil 7.0.0. The separate CPU RM worker used NumPy 2.5.4, CatBoost 1.2.10 and SciPy 1.18.1. Historical figure replay used NumPy 2.2.6 and Matplotlib 3.11.2. The original requirements files for [actor](fullstudy/code/requirements/actor.txt), [reward worker](fullstudy/code/requirements/reward-worker.txt) and [figures](fullstudy/code/requirements/figures.txt), together with dependency metadata, preserve these separate scopes. ReportLab is used by the old PDF builder; `threadpoolctl` occurs conditionally, with no established historical version. This is not a complete environment lock.

Models/checkpoints, raw or prepared data, per-query numerical payloads, feature/graph/bank caches and native packages are excluded. The five pinned multilingual-E5 encoder/tokenizer files needed to rebuild text features are also excluded; their identities are constants in the preserved `kion_text_embeddings.py`. Original Windows username paths remain in the unchanged historical sources as provenance. The source-pattern scan found no credential-shaped literal; that scan is not a claim that every possible secret or personal datum has been ruled out.
