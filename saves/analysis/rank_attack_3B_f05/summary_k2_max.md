saves/analysis/rank_attack_3B_f05: pools of 2 forgotten + 10 holdout authors (k-subsets of 10), aggregator max, 100 permutations

mean pool AUC [95 % jackknife interval over the authors] (95th percentile of the label-permutation null) for the primary and the sensitivity attacks; a result counts as a detection only above the null band.

| model | L1+L2+L5 (primary) | L1+L2 | L5 | L1+L2+L5+L6 |
|---|---|---|---|---|
| retain | 0.51 [0.09, 0.92] (0.71) | 0.56 [0.07, 1.05] (0.76) | 0.51 [0.04, 0.97] (0.70) | 0.51 [0.09, 0.92] (0.71) |
| full | 0.54 [0.13, 0.95] (0.71) | 0.32 [0.06, 0.57] (0.68) | 0.61 [0.22, 1.00] (0.74) | 0.54 [0.13, 0.95] (0.66) |
| NEAR-Act | 0.71 [0.37, 1.04] (0.73) | 0.58 [0.30, 0.86] (0.70) | 0.73 [0.40, 1.05] (0.70) | 0.71 [0.37, 1.04] (0.71) |
| NEAR-Dist | 0.73 [0.51, 0.94] (0.70) | 0.61 [0.25, 0.97] (0.67) | 0.82 [0.61, 1.04] (0.76) | 0.73 [0.51, 0.94] (0.72) |
| LUNAR | 0.66 [0.29, 1.02] (0.74) | 0.71 [0.30, 1.11] (0.69) | 0.59 [-0.06, 1.25] (0.73) | 0.66 [0.29, 1.02] (0.75) |
| ROCR | 0.90 [0.67, 1.13] (0.70) | 0.67 [0.34, 1.01] (0.71) | 0.90 [0.67, 1.13] (0.68) | 0.90 [0.67, 1.13] (0.70) |

precision@|F| / MRR of the primary attack (chance = 2/pool):

| model | primary |
|---|---|
| retain | 0.11 / 0.28 |
| full | 0.26 / 0.50 |
| NEAR-Act | 0.42 / 0.69 |
| NEAR-Dist | 0.06 / 0.33 |
| LUNAR | 0.26 / 0.41 |
| ROCR | 0.50 / 0.50 |

AUC of a single feature (information only; choosing the best one with the labels is NOT an attack):

| model | cross_question_name_share | degenerate | distinct_names | dup | modal_name_share | self_agreement | tas_refusal | words |
|---|---|---|---|---|---|---|---|---|
| retain | 0.37 | 0.50 | 0.64 | 0.50 | 0.38 | 0.46 | 0.50 | 0.56 |
| full | 0.32 | 0.50 | 0.60 | 0.50 | 0.66 | 0.66 | 0.50 | 0.32 |
| NEAR-Act | 0.48 | 0.50 | 0.56 | 0.50 | 0.48 | 0.96 | 0.50 | 0.58 |
| NEAR-Dist | 0.55 | 0.50 | 0.99 | 0.50 | 0.50 | 0.70 | 0.45 | 0.69 |
| LUNAR | 0.45 | 0.50 | 0.63 | 0.50 | 0.41 | 0.94 | 0.86 | 0.63 |
| ROCR | 0.85 | 0.50 | 0.69 | 0.50 | 0.93 | 0.76 | 0.50 | 0.67 |
