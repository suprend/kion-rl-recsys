# Frozen observed-fit residual utility

`utility.py` imports only the standard library. `run(request, check=None)` is an explicitly admitted data-only stage. It reads only `item_ids`, `counts`, and `item_mean` from the pinned existing scorer archive; no CatBoost, provider, GPT, RM fit or calibration runs. Old caches and reward bridges stay read-only.

Primary formula is `u(s,i)=r_RM(s,i)-m_fit(i)`, coefficient1, range[-1,1]. The frozen item prior is the scorer's smoothed observed completion from fit before2021-08-02. This is a derived utility, not a new calibrated prediction or causal/personalization estimate. Unsupported entries store0 only as a neutral sentinel; the original boolean support controls the objective, and all1024 actions remain in softmax/KL.

The extractor requires BOTH exact train512 and screen256 cohorts. Every cached raw ID must exist in the archive; `native_id == archive_index+3` is checked for every pair, and `counts[index]>0` must exactly equal saved RM support. No missing-ID imputation is allowed. Base rewards and priors must be finite in[0,1]; stored float64 derived values must be finite in[-1,1]. Original state receipts bind the cached IDs/history/origins. The data-only loader opens candidate IDs/raw IDs/reference scores, without opening raw1068 or item-table payloads.

Before fit, each cohort records supported utility/advantage min/max/SD, reference-weighted advantage RMS, residual/base RMS ratio, fraction of constant-within-state priors and exact pairwise preference sign changes (including ties). RMS is `sqrt(mean_active_states(sum_all1024(p_C*A^2)))`; unsupported advantage is0. A prior constant across all supported train pools fails the substantive-contrast gate. These diagnostics do not tune beta, coefficient, seed or arm. Fixed beta0.5 compares completion units, not matched KL.

`run` request fields:

```json
{
  "policy_module": "OLD/policy/policy_v2.py",
  "source_hashes": {"ABSOLUTE_POLICY_MODULE": "da7baf199b685cb467edd7d5eedaada151798ec40bbc23e9edfaee8e36340dba"},
  "preregistration": "NEW/protocols/preregistration.json",
  "preregistration_sha256": "ROOT_PIN",
  "prior_archive": {"path": "D:/tlab-recsys-2026/artifacts/task2-kion/local-pack-20261020/rm-r2/scorer/scorer.npz", "sha256": "d5df55586a9df54b5e121d494a149764d956750ac3caaf2b4c1ada27a93243d4"},
  "cohorts": {
    "train": {"cache_dir": "OLD/policy/train-cache-r1", "cache_sha256": "PIN", "reward_dir": "OLD/reward/train512-r1", "reward_sha256": "PIN"},
    "screen": {"cache_dir": "OLD/policy/screen-cache-r1", "cache_sha256": "PIN", "reward_dir": "OLD/reward/screen256-r1", "reward_sha256": "PIN"}
  },
  "utility_output": "NEW/reward/residual-r1",
  "deadline_utc": "ROOT_DEADLINE"
}
```

`OLD` means `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1`; `NEW` means `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1`. Placeholders must be replaced with exact existing hashes, and native control's source/input closure remains mandatory.

Outputs are `{train,screen}/utility.json`, `utility.npy`, `prior.npy`, followed by joint `prefit.json`. Every utility receipt pins its original reward/cache/support and preregistration. The joint receipt exists only after both cohorts and nondegeneracy pass. A failure leaves an incomplete fresh directory for diagnosis and never silently retries or switches reward. Fallback novelty utility is not implemented or selected by this primary-only adapter.

Consumers call `utility_arrays(directory, cache_arrays, cache_manifest, np, frozen_policy)` and receive `(utility, prior, support)` with the same pair shape. The loader verifies parent reward and exact derived arithmetic; diagnostics describing base completion must continue to use the original `policy.reward_arrays` values.

The `policy_module` field also accepts `{path,sha256}` for the evaluator/rollout convention; the descriptor hash and `source_hashes` entry must both match the exact frozen source.

Checks here are stdlib synthetic/source tests only. Real NPZ vectors, array equality, scale values and runtime resource behavior remain root-owned numerical preflight gates.
