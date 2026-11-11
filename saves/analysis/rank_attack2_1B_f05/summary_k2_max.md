saves/analysis/rank_attack2_1B_f05: pools of 2 forgotten + 20 holdout authors (k-subsets of 10), aggregator max, 200 permutations

mean pool AUC [95 % jackknife interval over the authors] (95th percentile of the label-permutation null) for the primary and the sensitivity attacks; a result counts as a detection only above the null band.

| model | L1+L2+L5 (primary) | L1+L2 | L5 | L6 logits | L1+L2+L5+L6 |
|---|---|---|---|---|---|
| retain | 0.69 [0.18, 1.00] (0.68) | 0.58 [0.15, 1.00] (0.66) | 0.68 [0.40, 0.96] (0.69) | 0.79 [0.55, 1.00] (0.65) | 0.73 [0.27, 1.00] (0.68) |
| full | 0.68 [0.38, 0.99] (0.69) | 0.60 [0.39, 0.82] (0.70) | 0.71 [0.47, 0.95] (0.68) | 0.46 [0.17, 0.74] (0.71) | 0.64 [0.35, 0.92] (0.66) |
| NEAR-Act | 0.70 [0.50, 0.90] (0.69) | 0.66 [0.42, 0.90] (0.71) | 0.81 [0.54, 1.00] (0.68) | 0.83 [0.65, 1.00] (0.66) | 0.69 [0.47, 0.90] (0.70) |
| NEAR-Dist | 0.49 [0.26, 0.73] (0.68) | 0.42 [0.19, 0.64] (0.70) | 0.62 [0.36, 0.89] (0.71) | 0.61 [0.41, 0.82] (0.70) | 0.41 [0.15, 0.67] (0.66) |
| LUNAR | 0.66 [0.46, 0.86] (0.69) | 0.61 [0.42, 0.81] (0.69) | 0.78 [0.62, 0.94] (0.65) | 0.96 [0.87, 1.00] (0.72) | 0.66 [0.46, 0.86] (0.68) |
| ROCR | 0.96 [0.90, 1.00] (0.67) | 0.71 [0.40, 1.00] (0.67) | 0.96 [0.90, 1.00] (0.68) | 0.85 [0.69, 1.00] (0.70) | 0.96 [0.90, 1.00] (0.65) |

precision@|F| / MRR of the primary attack (chance = 2/pool):

| model | primary |
|---|---|
| retain | 0.18 / 0.43 |
| full | 0.22 / 0.38 |
| NEAR-Act | 0.31 / 0.53 |
| NEAR-Dist | 0.03 / 0.15 |
| LUNAR | 0.20 / 0.45 |
| ROCR | 0.65 / 0.92 |

AUC of a single feature (information only; choosing the best one with the labels is NOT an attack):

| model | cross_question_name_share | degenerate | distinct_names | dup | greedy_entropy | greedy_logprob | logprob_sd | modal_name_share | self_agreement | tas_refusal | words |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retain | 0.51 | 0.50 | 0.25 | 0.45 | 0.58 | 0.47 | 0.79 | 0.64 | 0.76 | 0.50 | 0.61 |
| full | 0.66 | 0.50 | 0.46 | 0.55 | 0.44 | 0.54 | 0.46 | 0.64 | 0.80 | 0.50 | 0.63 |
| NEAR-Act | 0.51 | 0.50 | 0.73 | 0.56 | 0.62 | 0.81 | 0.64 | 0.50 | 0.86 | 0.60 | 0.77 |
| NEAR-Dist | 0.56 | 0.47 | 0.57 | 0.47 | 0.65 | 0.58 | 0.52 | 0.64 | 0.79 | 0.50 | 0.40 |
| LUNAR | 0.72 | 0.55 | 0.53 | 0.38 | 0.97 | 0.97 | 0.34 | 0.37 | 0.99 | 0.65 | 0.65 |
| ROCR | 0.96 | 0.50 | 0.76 | 0.32 | 0.70 | 0.78 | 0.61 | 0.90 | 0.86 | 0.50 | 0.73 |
