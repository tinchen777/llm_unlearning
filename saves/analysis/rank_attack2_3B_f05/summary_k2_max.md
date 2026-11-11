saves/analysis/rank_attack2_3B_f05: pools of 2 forgotten + 20 holdout authors (k-subsets of 10), aggregator max, 200 permutations

mean pool AUC [95 % jackknife interval over the authors] (95th percentile of the label-permutation null) for the primary and the sensitivity attacks; a result counts as a detection only above the null band.

| model | L1+L2+L5 (primary) | L1+L2 | L5 | L6 logits | L1+L2+L5+L6 |
|---|---|---|---|---|---|
| retain | 0.36 [0.00, 1.00] (0.69) | 0.45 [0.18, 0.72] (0.67) | 0.28 [0.00, 0.66] (0.68) | 0.57 [0.30, 0.85] (0.68) | 0.35 [0.00, 0.88] (0.68) |
| full | 0.74 [0.40, 1.00] (0.70) | 0.57 [0.00, 1.00] (0.71) | 0.91 [0.58, 1.00] (0.69) | 0.73 [0.49, 0.98] (0.66) | 0.74 [0.40, 1.00] (0.66) |
| NEAR-Act | 0.79 [0.38, 1.00] (0.72) | 0.69 [0.32, 1.00] (0.70) | 0.97 [0.62, 1.00] (0.67) | 0.88 [0.71, 1.00] (0.65) | 0.78 [0.39, 1.00] (0.69) |
| NEAR-Dist | 0.66 [0.44, 0.88] (0.63) | 0.45 [0.01, 0.89] (0.69) | 0.83 [0.67, 1.00] (0.69) | 0.63 [0.42, 0.84] (0.69) | 0.59 [0.31, 0.87] (0.66) |
| LUNAR | 0.62 [0.41, 0.83] (0.67) | 0.38 [0.15, 0.61] (0.66) | 0.93 [0.69, 1.00] (0.67) | 0.79 [0.50, 1.00] (0.72) | 0.61 [0.41, 0.80] (0.69) |
| ROCR | 0.98 [0.92, 1.00] (0.68) | 0.64 [0.15, 1.00] (0.66) | 0.98 [0.92, 1.00] (0.69) | 0.79 [0.51, 1.00] (0.69) | 0.98 [0.91, 1.00] (0.70) |

precision@|F| / MRR of the primary attack (chance = 2/pool):

| model | primary |
|---|---|
| retain | 0.03 / 0.14 |
| full | 0.28 / 0.51 |
| NEAR-Act | 0.39 / 0.60 |
| NEAR-Dist | 0.00 / 0.17 |
| LUNAR | 0.03 / 0.17 |
| ROCR | 0.67 / 0.93 |

AUC of a single feature (information only; choosing the best one with the labels is NOT an attack):

| model | cross_question_name_share | degenerate | distinct_names | dup | greedy_entropy | greedy_logprob | logprob_sd | modal_name_share | self_agreement | tas_refusal | words |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retain | 0.40 | 0.50 | 0.33 | 0.47 | 0.62 | 0.55 | 0.54 | 0.56 | 0.54 | 0.50 | 0.51 |
| full | 0.54 | 0.50 | 0.86 | 0.58 | 0.67 | 0.61 | 0.62 | 0.67 | 0.79 | 0.50 | 0.49 |
| NEAR-Act | 0.46 | 0.50 | 0.79 | 0.62 | 0.96 | 0.87 | 0.69 | 0.47 | 0.99 | 0.50 | 0.71 |
| NEAR-Dist | 0.70 | 0.50 | 0.97 | 0.42 | 0.61 | 0.54 | 0.68 | 0.58 | 0.90 | 0.45 | 0.53 |
| LUNAR | 0.51 | 0.58 | 0.88 | 0.42 | 0.90 | 0.80 | 0.53 | 0.29 | 0.99 | 0.51 | 0.41 |
| ROCR | 0.86 | 0.50 | 0.72 | 0.38 | 0.79 | 0.77 | 0.54 | 0.91 | 0.60 | 0.50 | 0.73 |
