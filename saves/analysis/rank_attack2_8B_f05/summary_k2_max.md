saves/analysis/rank_attack2_8B_f05: pools of 2 forgotten + 20 holdout authors (k-subsets of 10), aggregator max, 200 permutations

mean pool AUC [95 % jackknife interval over the authors] (95th percentile of the label-permutation null) for the primary and the sensitivity attacks; a result counts as a detection only above the null band.

| model | L1+L2+L5 (primary) | L1+L2 | L5 | L6 logits | L1+L2+L5+L6 |
|---|---|---|---|---|---|
| retain | 0.60 [0.31, 0.88] (0.71) | 0.41 [0.00, 0.99] (0.69) | 0.62 [0.36, 0.88] (0.69) | 0.60 [0.31, 0.88] (0.68) | 0.59 [0.30, 0.88] (0.71) |
| full | 0.94 [0.74, 1.00] (0.69) | 0.71 [0.42, 1.00] (0.68) | 0.94 [0.73, 1.00] (0.69) | 0.86 [0.70, 1.00] (0.70) | 0.90 [0.71, 1.00] (0.70) |
| NEAR-Act | 0.81 [0.53, 1.00] (0.67) | 0.58 [0.17, 1.00] (0.73) | 0.82 [0.49, 1.00] (0.67) | 0.99 [0.98, 1.00] (0.68) | 0.84 [0.55, 1.00] (0.69) |
| NEAR-Dist | 0.85 [0.64, 1.00] (0.67) | 0.53 [0.27, 0.78] (0.68) | 0.85 [0.63, 1.00] (0.67) | 0.60 [0.22, 0.98] (0.69) | 0.81 [0.42, 1.00] (0.69) |
| LUNAR | 0.92 [0.81, 1.00] (0.70) | 0.94 [0.85, 1.00] (0.69) | 0.54 [0.00, 1.00] (0.70) | 0.85 [0.54, 1.00] (0.66) | 0.92 [0.81, 1.00] (0.70) |
| ROCR | 0.96 [0.88, 1.00] (0.68) | 0.49 [0.24, 0.74] (0.67) | 0.98 [0.93, 1.00] (0.68) | 0.73 [0.47, 1.00] (0.67) | 0.94 [0.80, 1.00] (0.68) |

precision@|F| / MRR of the primary attack (chance = 2/pool):

| model | primary |
|---|---|
| retain | 0.27 / 0.54 |
| full | 0.62 / 0.90 |
| NEAR-Act | 0.51 / 0.71 |
| NEAR-Dist | 0.60 / 0.89 |
| LUNAR | 0.56 / 0.65 |
| ROCR | 0.90 / 1.00 |

AUC of a single feature (information only; choosing the best one with the labels is NOT an attack):

| model | cross_question_name_share | degenerate | distinct_names | dup | greedy_entropy | greedy_logprob | logprob_sd | modal_name_share | self_agreement | tas_refusal | words |
|---|---|---|---|---|---|---|---|---|---|---|---|
| retain | 0.67 | 0.50 | 0.49 | 0.59 | 0.66 | 0.61 | 0.58 | 0.36 | 0.22 | 0.50 | 0.31 |
| full | 0.66 | 0.50 | 0.92 | 0.52 | 0.92 | 0.83 | 0.69 | 0.79 | 0.87 | 0.50 | 0.76 |
| NEAR-Act | 0.66 | 0.55 | 0.82 | 0.57 | 1.00 | 0.97 | 0.85 | 0.49 | 1.00 | 0.50 | 0.65 |
| NEAR-Dist | 0.46 | 0.55 | 0.86 | 0.49 | 0.64 | 0.65 | 0.42 | 0.42 | 0.41 | 0.50 | 0.40 |
| LUNAR | 0.64 | 0.92 | 0.59 | 0.44 | 0.91 | 0.93 | 0.67 | 0.39 | 1.00 | 0.96 | 0.76 |
| ROCR | 0.92 | 0.50 | 0.44 | 0.33 | 0.82 | 0.76 | 0.53 | 0.94 | 0.79 | 0.50 | 0.54 |
