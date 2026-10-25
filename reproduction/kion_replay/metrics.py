"""Small transparent ranking/distribution summaries; no ML dependencies."""
from __future__ import annotations

from collections import Counter
import math

from .io import require


def distribution(row):
    values = [float(value) for value in row]
    require(values and all(math.isfinite(value) for value in values), "Finite nonempty scores required")
    maximum = max(values)
    denominator = math.fsum(math.exp(value - maximum) for value in values)
    log_z = math.log(denominator)
    logp = [value - maximum - log_z for value in values]
    return logp, [math.exp(value) for value in logp]


def rank(row, raw_ids, k=10):
    require(len(row) == len(raw_ids) and len(row) >= k, "Rank dimensions differ")
    require(all(math.isfinite(float(value)) for value in row), "Nonfinite rank score")
    return sorted(range(len(row)), key=lambda index: (-float(row[index]), int(raw_ids[index])))[:k]


def summarize(scores, raw_ids, ids, reference, targets=None):
    require(len(scores) > 0 and len(scores) == len(raw_ids) == len(ids) == len(reference), "Summary row counts differ")
    top_tokens, top_columns, kl, entropy = [], [], [], []
    for values, raw, tokens, ref in zip(scores, raw_ids, ids, reference):
        columns = rank(values, raw)
        top_columns.append(columns)
        top_tokens.append([int(tokens[index]) for index in columns])
        logp, probs = distribution(values)
        reference_logp, _ = distribution(ref)
        kl.append(math.fsum(p * (lp - lq) for p, lp, lq in zip(probs, logp, reference_logp)))
        entropy.append(-math.fsum(p * lp for p, lp in zip(probs, logp)))
    n = len(scores)
    count = Counter(token for row in top_tokens for token in row)
    count1 = Counter(row[0] for row in top_tokens)
    hhi = math.fsum((value / (10 * n)) ** 2 for value in count.values())
    result = {"rows": n, "mean_KL_to_recomputed_C": math.fsum(kl) / n,
              "mean_entropy": math.fsum(entropy) / n, "top1_max_share": max(count1.values()) / n,
              "top10_HHI": hhi, "effective_catalog": 1 / hhi,
              "ranking_tie_break": "score-desc/rawID-asc", "label_metrics_available": targets is not None,
              "top10_tokens": top_tokens, "top10_columns": top_columns}
    if targets is not None:
        require(len(targets) == n, "Target row count differs")
        hits = [[int(token == int(target)) for token in row] for target, row in zip(targets, top_tokens)]
        result.update(map10=math.fsum(hit / (position + 1) for row in hits for position, hit in enumerate(row)) / n,
                      recall10=sum(any(row) for row in hits) / n,
                      ndcg10=math.fsum(hit / math.log2(position + 2) for row in hits for position, hit in enumerate(row)) / n)
    return result
