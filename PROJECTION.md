# Source-only Git projection

This repository projects the frozen unified KION task 2 hand-in into a source/report/evidence checkout. It is separate from the full standalone ZIP. No scientific source, report, accepted figure, compact evidence file, or portable runtime source was rewritten.

The upstream package has 566 payload members (136,184,813 bytes). Its frozen manifest SHA-256 is `f242fb74c9e94d45923432507270222326574d533f8494cbb420ec30fde812db`; its final selection SHA-256 is `16ab7ae35e8145946779476706fa3b8f45a146a2ee85eabe962b5a0935fe1e18`.

The [projection manifest](projection.json) lists 526 exact-byte copies (17,014,776 bytes), each with its relative path, size, SHA-256 and upstream source path. It also lists every excluded selected member with the original size/hash and a reason. The two copied aggregate paired-metrics CSVs contain 36 data rows each. The preserved source tree retains all 284 historical payloads plus its four explanatory/integrity files; the portable replay package retains all ten Python files and its runtime manifest.

The 40 exclusions total 119,170,037 bytes:

- 26 numerical fixture or model-weight payloads (`.npy`, `.pt`, `.cbm`), 101,623,655 bytes. The actor and RM `manifest.json` descriptors are retained; they describe external fixture files and are insufficient for numerical replay by themselves.
- 13 external data-provenance JSON payloads, 17,540,011 bytes. Only `reproduction/external-provenance/INDEX.json` is retained as metadata.
- The 6,371-byte standalone hand-in root README, replaced by source-only repository documentation.

All local files outside the finite upstream package selection also remain outside this projection: live controls, logs, raw data, model/data caches, full user-level payloads, archives and secrets. Historical controls copied inside `source/` are preserved scientific code, not a live operation tree.

Twenty-two frozen text files contain CRLF bytes. Their explicit `-text` entries in [.gitattributes](.gitattributes) preserve the accepted hashes through Git; new text uses LF. Binary PNG and PDF payloads are also excluded from Git text normalization.

From the repository root with Python 3.12+, run the following read-only standard-library checks:

```text
python -B verify_repository.py
python -B source/verify_sources.py
python -B source/fullstudy/code/tools/verify_bundle.py --report-root source/fullstudy
python -B reproduction/kion_replay verify --bundle .
```

The first command verifies exact copied hashes and finite source-only file closure, so it deliberately fails after local fixture payloads are restored. Its newly authored files, including this manifest and documentation, are enumerated without recursive/self-referential checksums. The remaining commands validate the historical source or runtime closure. They do not establish numerical parity, execute scientific producers, train models, or download data. The historical bundle checker additionally uses an invented tiny ranking fixture.

For separately obtained fixtures and the historical acceptance tiers, see [REPRODUCING.md](REPRODUCING.md). Frozen evidence records successful numerical checks performed for the earlier full hand-in on its original workstation; copying those receipts does not rerun them or establish clean-machine/full-cohort training reproducibility. Original absolute paths inside preserved text remain historical provenance.
