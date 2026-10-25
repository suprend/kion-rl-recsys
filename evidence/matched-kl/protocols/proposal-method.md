# Proposal: completion versus residual at comparable train KL

Design proposal only, 2026-10-25. This file does not admit or launch a fit. Root must freeze and hash its preregistration before numerical calibration; final beta must be frozen in a separate receipt before final-seed fits. Existing results were already observed; this is a prospective follow-up on reused development data, not an untouched-test confirmation.

## Question and estimand

Does replacing supported completion reward `r_RM(s,i)` with `r_RM(s,i)-m_fit(i)` alter relevance and exposure when both policies move a comparable distance from original C? The primary comparison is residual minus completion at step1000, separately for sampler seeds20261024/20261025/20261026. C is the same fixed ancestor/reference for every fit. Completion beta stays0.5. Residual uses one global train-calibrated beta, frozen before final-seed fits. No per-seed tuning, reward rescaling, early checkpoint selection, optimizer changes, extra SFT/RM, final-test reading, download or upload.

All fits keep frozen GPT/RM, original-C initialization, train512, natural1024, temperature1, B16 with replacement, Adam lr1e-5/weight_decay0/clip1, CUDA-FP16 producer-shape/slot arithmetic, original support/unsupported-neutral advantage, and1000 actual updates. Retain original-C parity and parameter/gradient/source integrity gates. Use identical ordered eligible train positions and `random.Random(seed).choice` streams across paired objectives; save identical sampler-position digests. Seeds describe minibatch-sampler variability from a fixed pretrained model, not independent data splits or random initializations.

## Train-only calibration, fully bounded

Define `K(h)=mean_(s in frozen train512) sum_(i=1..1024) p_h(i|s)*log[p_h(i|s)/p_C(i|s)]`, using exact full-action softmax at temperature1, frozen original-C scores and the same producer layout. Mean over all512 states, including zero-support states; also report active-state count and active-only KL as diagnostics. Do not silently substitute supported-only KL, reverse KL, top-k KL, reward gain or screen/development KL.

Root first scores existing completion1000 and existing residual1000 beta0.5 on train512 without any fit. Freeze finite positive target `T=K(existing completion1000,seed20261024)` and old residual measurement `K0`. Reject calibration if target is <=1e-8, inputs are nonfinite, support differs, or C parity fails. Do not set T to the already known screen KL0.026806. Existing screen residual KL0.003964 and prefit train RMS ratio0.491578 are motivation only.

For deterministic pilot proposals use `clip(x)=min(2.0,max(0.01,x))`:

1. Pilot1: residual seed20261024, beta1=`clip(0.5*sqrt(K0/T))`, fresh original-C start, exactly1000 updates; measure train KL `K1`.
2. If `0.8 <= K1/T <= 1.2`, freeze beta1 and omit Pilot2.
3. Otherwise Pilot2: residual seed20261024, beta2=`clip(beta1*sqrt(K1/T))`, fresh original-C start, exactly1000 updates; measure `K2`. If beta2 equals beta1, omit this redundant fit.
4. Freeze the available pilot beta minimizing `abs(log(Kj/T))`, breaking ties in favor of the larger beta. No third pilot and no further beta search. Zero pilot KL gives infinite log-distance, while the next beta proposal clamps to0.01. If neither pilot has positive finite KL, terminate this branch without final fits. Nonfinite optimizer state is a failed fit and stops dependent work, never an automatic rerun.

The inverse-square KL update is a local heuristic: for the unconstrained categorical optimum `p* proportional p_C exp(A/beta)`, small tilts give approximately `KL proportional Var_C(A)/(2 beta^2)`. This follows by a second-order log-partition expansion; finite-step Adam in a restricted head need not follow it. The closed-form optimum is consistent with the standard KL-regularized formulation in [Rafailov et al.,2023](https://arxiv.org/abs/2305.18290). Neither this theory nor the pilot rule promises matching.

## Frozen final branch and matching verdict

After beta freeze, final runs are completion seeds20261025/20261026 and residual seeds20261024/20261025/20261026: five new final fits. Reuse the existing completion seed20261024. Keep pilots explicitly marked calibration; the final residual seed24 is a fresh frozen-beta fit, and its agreement with the corresponding pilot is a reproducibility diagnostic. Maximum new fits=7 (five final plus two pilots), each<=1000 updates; failed/incomplete fits count toward the budget. The eighth authorized slot remains unused, with no retries or post-result extensions.

Before looking at new relevance/exposure results, compute all final train KL. Primary matching passes only if every seed's actual ratio `K(residual_s)/K(completion_s)` lies in inclusive[0.8,1.2]. Also show each arm's distance to frozen T, without requiring new completion seeds individually to equal T. Mean matching cannot hide a failed individual seed. If any ratio fails, label the whole branch “train-KL matching not achieved”; retain all outcomes as a bounded unmatched comparison, with actual KL prominently reported. Do not tune after seeing failure. Even a pass controls a scalar average on train states; it does not equate per-state drift, optimization progress, or development KL. Screen/full KL are descriptive only.

No new fit may start at/after09:25MSK; all numerical work must stop by09:45MSK; report by10:00MSK. Root should decline a launch if its bounded estimate cannot leave time for evaluation and cleanup. A deadline-shortened or failed branch is reported as incomplete; a partial checkpoint cannot substitute for step1000. Retain all failed/pilot receipts and immutable old roots/STOP files.

## Evaluation and uncertainty

Use the exact existing development4096 selection, natural candidate IDs and frozen target/known/coverage/user bindings. Independently reconstruct rank from1024 scores with score-desc/raw-ID-asc ties; preserve unknown/off-pool misses and denominator4096. C parity must pass. Primary relevance metric: NDCG@10. Recall@10 and RR@10/MAP@10 (one observed target) are descriptive secondary metrics. Retain original equal-beta seed24 comparison as a historical diagnostic, labeled one-seed and unequal-KL. No metric selects beta, checkpoint or report cases.

For every metric report the three paired seed deltas, arithmetic mean, sample SD and min/max. Display new-seeds-only mean alongside all-three mean because seed24 served calibration. Three seeds cannot support strong distributional or robustness claims. The distinction between run uncertainty and finite-evaluation uncertainty follows the warning about few-run evaluation in [Agarwal et al.,2021](https://arxiv.org/abs/2108.13264).

Separately compute the existing paired cluster bootstrap:2000 shared user resamples, fixed seed20261024; carry all queries per selected user and use query-weighted sum/count ratios. For each draw compute each seed's paired delta then average the three deltas. Label the percentile95% interval “user-sampling CI conditional on these three fitted policies.” Use the same users/draw weights across every arm and seed. Do not stack3x4096 observations as independent, bootstrap users and call it seed variability, or present this CI as uncertainty over training seeds. Avoid a three-seed hierarchical bootstrap as a headline guarantee. All secondary intervals are unadjusted and exploratory.

Screen256 supplies unchanged raw/residual/prior centered proxy gains, support mass, top10 item exposure (HHI/effective catalog/top-item share), history membership and strict7d popularity diagnostics. If time permits, add these as compact seed-wise tables using the exact saved screen states. Full4096 supplies observed-target relevance; do not transplant screen utility estimates to it. A residual advantage can indicate altered reliance on observed-fit item prior; it cannot establish personalization, causal utility or retention.

## Proposed2–3page report

Page1: question, fixed design/counts, beta calibration and actual per-seed train KL with clear pass/fail, one compact relevance table. Page2: paired seed deltas/mean/SD, conditional user-bootstrap CI, exposure/proxy comparison and evidence-based conclusion. Optional page3: limitations, small provenance table and source links. Explicitly separate observed outcomes from proposed runs and show counts actually completed. No screenshot/case selected by outcome. Keep report self-contained; local private arrays are optional audit inputs rather than required reading.

## Local evidence inspected

- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-20261024-r1/policy/policy_v2.py`: effective_advantage, exact_loss, producer layout, sampler/optimizer and evaluation contracts.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/policy/residual.py`: original residual objective/lineage and fixed1000 fit.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/protocols/preregistration.json`: original beta0.5/seed/utility/fixed-latest restrictions.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/reward/utility.py`: supported C-centering and RMS definition.
- `D:/tlab-recsys-2026/artifacts/task2-kion/c-rl-final-20261025-r1/interpretation/ANALYZE.md`: development metrics, user-bootstrap and interpretation limits.

No numerical arrays, ML imports, GPU operation or model/data upload was performed to produce this source-only proposal.
