# Bounded native campaign control

Root is the sole real-data launch owner across all campaign namespaces. The local
`heavy.owner.lock` serializes this root only; root must separately enforce one
global heavy job, including old/new namespaces. These files create no queue or schedule.
Entry from `D:/personal-agent-system`:

```powershell
.\pas.ps1 python -B D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/control/launch.py request-OPERATION.json
```

The request must be a direct child of the campaign root, named `request-*.json`.
After source review, root writes the request and its SHA-256 to
`campaign.json.admitted_requests[operation]` as
`{"request_path": "absolute/posix/request.json", "sha256": "64 hex characters"}`.
This is the root-owned admission record; there is no cryptographic signing key.
Keep the campaign file unchanged during an operation: its stat is monitored.

Required request envelope (stage-specific payload fields remain in the same object):

```json
{
  "contract": "kion-c-rl-native-v1",
  "operation": "plain-lowercase-operation-name",
  "stage": "prepare_cache",
  "module_path": "D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/policy.py",
  "function": "prepare_cache",
  "deadline_utc": "per-operation timezone-aware ISO timestamp",
  "maximum_seconds": 3600,
  "cap_gib": 8,
  "reserve_gib": 2,
  "cpu_threads": 4,
  "supervised_updates": 0,
  "new_rm_fits": 0,
  "final_test_reads": 0,
  "rl_updates": 0,
  "source_hashes": {"absolute/source.py": "sha256"},
  "input_hashes": {"absolute/input.file": "sha256"},
  "owned_output_roots": ["C:/owned-cache/c-rl-final-20261025-r1"],
  "output_paths": ["C:/owned-cache/c-rl-final-20261025-r1/stage-output"],
  "required_output_bytes": {"C:": 1000000000, "D:": 0}
}
```

Use `rl_updates` equal to the admitted maximum for that operation (zero for probes).
The campaign forbids supervised updates, new RM fits, final-test reads, external
downloads and paid operations. It admits one GPU/heavy job and at most 1,000 RL
updates across the root-controlled campaign plan; root remains responsible for
the cumulative accounting and the three-round stop rule.

Stages: `prepare_cache`, `smoke`, `interpret`, `cohort_audit`, `reward_probe`,
`frontier`, `train_rl`, `evaluate`, `live_prefix`, `analyze`.
Functions: `run`, `analyze`, `probe`, `score`, `prepare_cache`, `frontier`,
`inference_frontier`, `train`, `score_checkpoint`, `evaluate_selected`, `rollout`,
`smoke`, `evaluate`. The exact module path and function are root admitted through
the request digest. Only `.py` paths beneath campaign `interpretation`, `cohorts`,
`reward`, `policy`, or `reports` directories are accepted. Each stage function
receives `(request, check)` and should call `check(**progress_fields)` between
bounded work chunks. Configure dependent source pins in the request as well.

Every request pins the entry module, all four control modules (`common.py`,
`launch.py`, `supervise.py`, `worker.py`), the PAS `scripts/portability.py`, and
`ready-pack-20261020/provider/runtime-native-r5/scripts/kion_ready_runtime.py`.
Sources are always hashed before native helper/stage imports. Inputs are hashed
before stage loading. A root-verified immutable large input may avoid a second
full hash by carrying:

```json
"input_digest_receipts": {
  "absolute/input.file": {
    "sha256": "same digest as input_hashes",
    "stat": {"size": 123, "mtime_ns": 456, "ctime_ns": 789}
  }
}
```

Receipts must be captured after root verified the content hash. A changed stat
rejects admission. Source/input/request/campaign stats are checked every monitor
iteration and every cooperative check; this is change detection rather than an
OS filesystem sandbox. Reviewed stage code must obey the declared read/output
scope and the no-fit/no-final-test rules.

The reviewed `launch.py`/`common.py` pair is the trusted bootstrap. Root must run
the reviewed copies; the launcher hashes every declared source, including the
supervisor and native helper, before creating the supervisor process.

The pinned BASE interpreter inherits only the selected Game Voice/PAS package
paths and four-thread numerical limits. The worker calls pinned
`ready_runtime.configure_paths()` before importing the stage. The supervisor
uses the reviewed PAS isolated-process implementation to create the worker
suspended, attach it to one kill-on-close Job, and resume it. Aggregate Job memory
is exactly 4/8/12 GiB; affinity is `0x55` (logical CPUs 0, 2, 4, 6). Admission
requires the cap plus 2 GiB host RAM; monitoring stops at a 2 GiB host reserve.

Each operation is finite: at most 3,600 seconds; `prepare_cache` and
`reward_probe` may use 7,200 seconds. The request deadline is an independent
earlier stop. This campaign pins 2026-10-25T06:00:00+00:00 (09:00 MSK) as its
latest request deadline through campaign.json; do not admit new jobs after it.
The old c-rl-20261024-r1/STOP remains preserved and must not be removed. A campaign-root
`STOP` stops admission and the running Job.

Outputs must be declared beneath campaign-owned paths on C/D/E. External roots
must contain the exact campaign directory name. Each involved volume, including
the campaign's D volume, needs its declared additional output estimate plus
15 GiB free at admission. A tiny exclusive write/fsync/unlink probe checks each
output root. Runtime checks retain a 15 GiB free reserve after outputs consume
their estimated allocation.

`operations/OPERATION/receipt.json`, `resources.jsonl`, `worker.log`, and
`stage-completion.json` provide native lifecycle evidence. The exclusive
campaign `heavy.owner.lock` is held through Job termination and receipt writing,
including descendants after leader exit. An unsuccessful cleanup receipt blocks
every later launch until root diagnoses and resolves it. Unrelated processes,
including PSD2Live, are never enumerated for termination.

Validation performed by the author: synthetic source/admission/guard tests only:

```powershell
.\pas.ps1 python -B -m unittest discover -s D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/control -p test_control.py -v
```

These tests launch no native Job, subprocess, numerical payload, or GPU operation.
Root must separately review and own any real invocation.
