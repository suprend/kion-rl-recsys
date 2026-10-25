# C3 saved-output interpretation

> Reader export: aggregate links below are adapted for this bundle. Private per-query outcomes and case-level payloads are excluded. The canonical original report is unchanged; source and export hashes are recorded in `../evidence-index.json`.

Saved4096-query natural1024 pool; frozen E2, A/B/C trained262144 examples×3epochs. Unknown/offpool remain misses. No new model, features, SHAP, or test reads.

| Arm | MAP@10 | NDCG@10 | Recall@10 | top1 max share | top10 HHI | effective items | unique items |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.082896 | 0.107589 | 0.188721 | 0.295166 | 0.053372 | 18.7 | 304 |
| B | 0.086458 | 0.110500 | 0.189209 | 0.296631 | 0.043724 | 22.9 | 776 |
| C | 0.096864 | 0.123949 | 0.212402 | 0.285156 | 0.035943 | 27.8 | 831 |
| GPT | 0.072315 | 0.092305 | 0.156982 | 0.240234 | 0.048323 | 20.7 | 374 |
| RRF | 0.079310 | 0.100784 | 0.170654 | 0.326416 | 0.066164 | 15.1 | 281 |

Retrieval boundary: {'covered': 3358, 'unknown': 122, 'offpool': 616}. Full denominator=4096.

C-B: 394 gains, 142 losses, 3560 ties; mean RR@10 difference +0.010405. Mean target rank improvement on covered rows +3.48.
C-GPT: 532 gains, 201 losses, 3363 ties; mean RR@10 difference +0.024548. Mean target rank improvement on covered rows +95.63.

Equal-user exposure, full rank-displacement/margin summaries, residual arithmetic, content-type coverage and source pins: [i0-score-audit.json](i0-score-audit.json). The private per-query outcome file is excluded from this reader bundle.

The original study includes twelve fixed case studies (or explicitly recorded cell shortfall); its private case-level file is excluded from this reader bundle. C-B gain, C-B loss, C-GPT gain, C-GPT loss; three unique rows per cell, SHA256(c3-I0-cases-v1|row_index) ascending, cell order as written; exclude previously chosen rows; gain/loss uses reciprocal-rank@10; no outcome-based replacement. These outcome-enriched cases are not a population sample.

Old run-rl-r1: own512-query fixed128 screen NDCG 0.112741→0.101114; selected update=0. Additional matched policy exposure unavailable; saved reference/base cache does not recover latest policy. Proxy reward unavailable; reward hacking is not established.
Old run-rl-recovered-r1: own512-query fixed128 screen NDCG 0.103666→0.094693; selected update=0. Additional matched policy exposure unavailable; saved reference/base cache does not recover latest policy. Proxy reward unavailable; reward hacking is not established.

Ready RM SHAP is reused with verified payload/report/scorer/rows SHA lineage. Its256 supported select rows,11227numeric+7categorical inputs and raw tree-output scale explain that RM only. Leading groups: candidate:local_dynamic and context:history. Report preserves both sum-absolute-column and absolute-net-group measures; 605 unsupported select targets were excluded. It does not explain C or causal history benefit.

Unavailable: history repeat, novelty/head-tail counts and trustworthy runtime; missing data were not imputed as zero. No raw-feature or neuron explanation was computed. Static catalog type metadata, if supplied, has an explicit temporal validity assumption. This reused development cohort and one training seed do not establish future-week quality or retention.
