# Five-step live-prefix saved-output summary

Validated192 complete trajectories:16 fixed positions×3heads×2selection modes×2repeat masks×5steps. This is inference from an H1-trained policy with fixed observed-s0 RM/raw features; no five-step training or causal outcome claim.

| Head | Mode | Mask | mean unique | mean repeats | observed-prefix seen fraction | top1 max share | slate HHI | effective items | unique catalog | neutral proxy return | raw proxy return (all-supported n/16) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| C | greedy | none | 1.688 | 3.312 | 0.0000 | 0.4375 | 0.342500 | 2.92 | 6 | 0.197070 | 1.834513 (16/16) |
| C | greedy | exclude-generated | 5.000 | 0.000 | 0.0000 | 0.4375 | 0.071250 | 14.04 | 27 | 0.124479 | 1.761922 (16/16) |
| C | categorical | none | 4.875 | 0.125 | 0.0125 | 0.1250 | 0.019688 | 50.79 | 66 | -0.109324 | 1.534556 (14/16) |
| C | categorical | exclude-generated | 5.000 | 0.000 | 0.0125 | 0.1250 | 0.019688 | 50.79 | 65 | -0.043336 | 1.609201 (13/16) |
| RL-completion | greedy | none | 1.625 | 3.375 | 0.0000 | 0.3750 | 0.324375 | 3.08 | 6 | 0.589693 | 2.227136 (16/16) |
| RL-completion | greedy | exclude-generated | 5.000 | 0.000 | 0.0000 | 0.3750 | 0.077813 | 12.85 | 27 | 0.343006 | 1.980449 (16/16) |
| RL-completion | categorical | none | 4.875 | 0.125 | 0.0000 | 0.1875 | 0.025313 | 39.51 | 61 | 0.126469 | 1.841563 (14/16) |
| RL-completion | categorical | exclude-generated | 5.000 | 0.000 | 0.0000 | 0.1875 | 0.020000 | 50.00 | 66 | 0.176857 | 1.852460 (15/16) |
| RL-residual | greedy | none | 1.812 | 3.188 | 0.0000 | 0.4375 | 0.295000 | 3.39 | 7 | 0.156140 | 1.793583 (16/16) |
| RL-residual | greedy | exclude-generated | 5.000 | 0.000 | 0.0000 | 0.4375 | 0.074688 | 13.39 | 25 | 0.113761 | 1.751204 (16/16) |
| RL-residual | categorical | none | 4.875 | 0.125 | 0.0375 | 0.1250 | 0.019063 | 52.46 | 66 | 0.062200 | 1.712037 (15/16) |
| RL-residual | categorical | exclude-generated | 5.000 | 0.000 | 0.0125 | 0.1250 | 0.019063 | 52.46 | 69 | -0.023905 | 1.613538 (16/16) |

Per-step KL/entropy, unsupported probability mass, effective expected proxy and supported-conditional expected rewards with denominators; paired16-position overlaps; separate static RM-greedy unique5 control; and first3 predetermined-position slates with rawIDs/titles are in [rollout-summary.json](rollout-summary.json).

A raw completion return is averaged only when all5 selected actions are supported. Unsupported selections have null raw reward and zero centered neutral proxy; this does not imply zero real-world satisfaction. Top1/slate concentration describes generated outputs, not impressions. Later prefixes differ across policies.

Raw1068/source13 and observed RM state remain fixed while GPT q/base/source12 evolve. These synthetic sequences do not reconstruct watched sessions, user responses, retention or causal counterfactual outcomes. Sixteen fixed contexts and one categorical seed are descriptive only.
