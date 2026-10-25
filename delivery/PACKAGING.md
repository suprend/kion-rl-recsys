# Packaging the unified hand-in

`package.py` uses Python 3.11+ and only the standard library. It packages an
explicit file list; it never walks source trees, runs payload code, trains a
model, downloads data, uploads, or submits anything. The coordinator owns the
real selection, freeze, approval, archive and receipt. An independent reviewer
must inspect this helper before its first real use.

## Explicit selection and admissions

Create a local selection JSON outside the hand-in payload. All source paths and
ZIP member names use `/`, are relative, and preserve the intended directory
hierarchy. Root aliases map to existing absolute directories. Only listed files
are opened. The local frozen manifest retains absolute roots; the embedded
`PACKAGE_MANIFEST.json` contains aliases and relative paths, hashes, CRC32,
purposes, admissions and actual acceptance declarations, without those roots.

```json
{
  "version": 1,
  "roots": {"R": "D:/path/to/submission", "F": "D:/path/to/frozen-study"},
  "max_total_bytes": 314572800,
  "acceptance_tiers": [
    {
      "id": "portable-structural-check",
      "status": "passed",
      "basis": "The selected evidence records the actual command and its successful result.",
      "evidence_members": ["evidence/portable-check.json"]
    },
    {
      "id": "full-study-retraining",
      "status": "not_run",
      "basis": "No new training was authorized or performed in this delivery campaign.",
      "evidence_members": []
    }
  ],
  "files": [
    {"root": "R", "source": "README.md", "member": "README.md", "purpose": "Hand-in entry point"},
    {"root": "R", "source": "reports/REPORT.md", "member": "reports/REPORT.md", "purpose": "Unified final report"},
    {"root": "R", "source": "evidence/portable-check.json", "member": "evidence/portable-check.json", "purpose": "Actual check receipt"},
    {
      "root": "R", "source": "reproduction/fixture.npz", "member": "reproduction/fixture.npz",
      "purpose": "Bounded no-training numerical replay",
      "admission": "derived_fixture",
      "admission_reason": "Coordinator approved these bounded transformed inputs; source user IDs, labels and title mappings are excluded. Anonymity is not established."
    }
  ]
}
```

This example is a schema illustration, not evidence that a check ran. Real tier
statuses must be `passed`, `not_run`, `requires_external_inputs`, or `failed`.
Every tier needs a factual `basis`; a passed tier needs included evidence.
Packaging displays the declarations and checks that evidence members exist. It
does not establish the truth of a declared scientific or numerical result.
Clean extraction verifies archive integrity; a replay tier passes only after the
documented replay command actually completes from that extraction.

For a selected copy or derivative, optional `origin` records an original source:

```json
"origin": {"root": "F", "source": "src/evaluate.py", "must_match": false}
```

The original's exact bytes are frozen and rechecked before and after building.
Use `must_match: true` for an unchanged copy; `false` allows an explained
derivative and does not assert source equality. Without `origin`, the selected
file itself is the bound original. No expected source path is inferred.

Numerical arrays (`.npz`, `.npy`) require `derived_fixture` admission. Frozen
weights (`.pt`, `.pth`, `.safetensors`, `.bin`, `.ckpt`) require `frozen_weights`
admission. Both require a nonempty `admission_reason` supplied by the coordinator
after review of the actual file. Admission is a recorded human decision, not
proof of anonymity, safe model deserialization, licensing, or training. The
helper treats these files as opaque bytes and never loads them. Derived inputs
can retain information from their source through transformed token prefixes or
user features. Excluding source user IDs, labels and title mappings does not
establish anonymity. `derived_fixture` records an explicit scoped admission and
makes no anonymity claim.

Preserve the submission's relative structure: the report is
`reports/REPORT.md`, with optional `reports/REPORT.pdf`; the hand-in entry point
is `README.md`. Do not relocate either report to the archive root.

Secrets, raw/history/cache directories, common secret files, databases,
line-oriented histories/logs, nested archive files, symlinks, junctions and other reparse points are
rejected. Text is UTF-8 and receives a bounded scan for common credential forms;
manual privacy/source review is still necessary. Do not include full datasets,
raw user histories, caches, local environment files or credential-bearing
configuration under renamed paths. The limit is 300 MiB for payload plus the
embedded manifest, at most 10,000 selected files, and 8 MiB per audited text file.
The limit is a planned bound, not a claim about the actual delivery size.

## Coordinator workflow

On this Windows checkout invoke Python through `D:/personal-agent-system/pas.ps1`
as required by its environment instructions. On an extracted hand-in, use the
recipient's Python 3.11+; the helper itself has no repository dependency. The
commands below use symbolic filenames; the coordinator supplies real paths.

```powershell
python delivery/package.py preview --selection control/selection.json
python delivery/package.py freeze --selection control/selection.json --manifest control/frozen.json
python delivery/package.py approve --manifest control/frozen.json --approval control/approval.json --reviewer independent-review-id
python delivery/package.py build --manifest control/frozen.json --approval control/approval.json --output delivery/handin.zip --receipt control/build-receipt.json
python delivery/package.py verify --archive delivery/handin.zip --manifest control/frozen.json --extract-to 'D:/fresh verification/hand in'
```

Preview reads only explicitly selected files and optional originals. It reports
the byte budget, members, link closure and actual declared acceptance tiers; any
missing/nonportable local Markdown target or undefined explicit reference fails
the operation. Link resolution uses exact, case-sensitive archive paths, so a relocated report needs
links valid at its final member path. Ordinary inline/reference Markdown links,
image links, HTML `href`/`src`, and wikilinks are checked; code spans/fences are
ignored. Heading-anchor existence, arbitrary embedded scripts and every possible
Markdown extension are outside this limited scanner. Review rendered documents
as well. PDF visual QA is a separate acceptance step.

Freeze creates a new local manifest and never overwrites an earlier one. Approval
records a reviewer id against the exact manifest SHA-256 after freshness checks.
This is an audit binding, not a signature, authentication system or automatic
permission grant. The coordinator should create it only after the independent
review is complete. Any selected-file, original-source or selection-JSON change
requires a new preview/freeze/review/approval sequence with fresh output paths.

Build refuses existing archive/receipt paths. It revalidates the explicit input
list, text audit, links, original hashes, and manifest-bound approval. It streams
the payload with SHA-256/CRC32 checks, uses sorted members and fixed metadata,
then rechecks sources and fully reads every archive member. Only after the full
readback passes does it create the separate receipt containing the archive
SHA-256, size, embedded-manifest SHA-256, exact manifest/approval hashes, member
count and declared tiers. It does not put its receipt or a recursive self-hash
inside the archive. The embedded manifest explicitly enumerates every payload
member; its own exact bytes are bound by the external receipt and archive hash.

A failed build may leave an incomplete ZIP, but never a successful receipt.
Choose fresh filenames for a retry; inspect a failed output before any explicit
cleanup. For the same frozen manifest and Python/zlib implementation, ZIP bytes
are deterministic. Different compression-library versions can produce different
compressed bytes; exact uncompressed SHA-256/CRC32 still remain checkable.

`verify` does not need local original sources. With `--manifest`, it also binds
the embedded manifest to the local frozen record. Without that option it checks
self-consistency, which does not authenticate an archive against an external
trusted hash. Verify the receipt/archive SHA-256 through the trusted delivery
channel when authenticity matters.

Extraction occurs only after full verification, into a destination that must not
exist and whose parent must already exist. Existing empty directories are also
rejected. Paths, duplicates, case/Unicode collisions, file-directory conflicts,
unsafe ZIP modes, encryption, metadata, exact member allowlist, CRC and SHA-256
are checked. Files use exclusive creation; symlink/junction ancestors are rejected
and extracted bytes are read back. Never use a directory concurrently mutated
by another process: portable stdlib checks cannot eliminate hostile filesystem
races. Partial extraction after an I/O failure has no success result.

After extraction, independently run the documented self-check and admitted
no-training replay from that fresh directory, including a path containing spaces.
Record actual commands, versions, output hashes, numerical tolerances and result
tiers as new evidence. Those results may require a final new freeze if the
evidence becomes part of the hand-in. Keep the final source-evidence set stable
before the coordinator creates the final archive and receipt.

## Synthetic tests

From this environment:

```powershell
& D:/personal-agent-system/pas.ps1 python -m unittest discover -s 'D:/path/to/submission/delivery/tests' -p test_package.py -v
```

Tests create only synthetic files and ZIPs in temporary directories. Passing them
does not claim a real archive was built, a real model was replayed, or a delivery
was accepted.
