# A2 SPA 3B rocr (retain)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.85 | 9 | 0.84 [0.66, 1.00] (0.71) | L9: 0.84 | 0.72 | 0.35 [0.09, 0.61] (0.72) | L1: 0.55 | 0.52 | - |
| P0-familiarity/dm | 0.85 | 12 | 0.75 [0.52, 0.98] (0.68) | L5: 0.79 | 0.68 | 0.45 [0.16, 0.74] (0.69) | L22: 0.66 | 0.52 | - |
| P1-spec/lr | 1.00 | 6 | 0.70 [0.45, 0.95] (0.72) | L14: 0.91 | 0.68 | 0.55 [0.27, 0.83] (0.71) | L2: 0.73 | 0.66 | 0.66, 0.55, 0.63, 0.58, 0.58 |
| P1-spec/dm | 1.00 | 6 | 0.68 [0.41, 0.95] (0.70) | L14: 0.92 | 0.65 | 0.58 [0.30, 0.86] (0.71) | L25: 0.68 | 0.53 | 0.67, 0.68, 0.69, 0.63, 0.60 |
| P2-fingerprint/lr | 1.00 | 5 | 0.62 [0.34, 0.90] (0.72) | L14: 0.78 | 0.55 | 0.47 [0.19, 0.75] (0.74) | L3: 0.81 | 0.62 | 0.58, 0.59, 0.62, 0.57, 0.51 |
| P2-fingerprint/dm | 1.00 | 6 | 0.58 [0.31, 0.85] (0.71) | L14: 0.81 | 0.63 | 0.55 [0.27, 0.83] (0.73) | L3: 0.71 | 0.53 | 0.55, 0.43, 0.53, 0.42, 0.44 |

Edit layers [3, 4, 5]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.55, P0-familiarity/dm: -0.58, P1-spec/lr: -0.40, P1-spec/dm: -0.61, P2-fingerprint/lr: 0.19, P2-fingerprint/dm: -0.26.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.6380486212670803, 0.6517705701291561, 0.6985085353255271, 0.5328946754336357, 0.4554850721731782], "shadow_HA_prob": [0.1497044579265639, 0.14964287553913894, 0.1495973667083308, 0.149460151954554, 0.14929980978835372], "shadow_F_prob": [0.12419947560410946, 0.12422982978634536, 0.12403021722100675, 0.12442727017682047, 0.12418803690467028], "target": {"S": 0.9402199801802636, "F": 0.12411896957084537, "H_A": 0.15010720291640609, "H_B": 0.13045282321516424, "R_A": 0.9380298426747322, "R_B": 0.9398281663656235}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L9 | 0.84 [0.66, 1.00] (0.71) | 0.35 (0.72) | 0.81 [0.61, 1.00] (0.70) | 0.33 (0.71) |
| P0-familiarity/lr | cv_deepest | L9 | 0.84 [0.66, 1.00] (0.71) | 0.35 (0.72) | 0.81 [0.61, 1.00] (0.70) | 0.33 (0.71) |
| P0-familiarity/lr | post_edit_mean | L6-27 | 0.64 [0.36, 0.92] (0.70) | 0.32 (0.71) | 0.62 [0.35, 0.89] (0.70) | 0.30 (0.70) |
| P0-familiarity/lr | pre_edit_mean | L0-2 | 0.73 [0.48, 0.98] (0.74) | 0.56 (0.72) | 0.66 [0.39, 0.93] (0.70) | 0.53 (0.70) |
| P0-familiarity/dm | cv | L12 | 0.75 [0.52, 0.98] (0.68) | 0.45 (0.69) | 0.77 [0.53, 1.00] (0.69) | 0.39 (0.70) |
| P0-familiarity/dm | cv_deepest | L13 | 0.74 [0.51, 0.97] (0.67) | 0.42 (0.69) | 0.70 [0.44, 0.96] (0.68) | 0.40 (0.71) |
| P0-familiarity/dm | post_edit_mean | L6-27 | 0.65 [0.38, 0.92] (0.70) | 0.46 (0.71) | 0.61 [0.34, 0.88] (0.71) | 0.47 (0.68) |
| P0-familiarity/dm | pre_edit_mean | L0-2 | 0.64 [0.37, 0.91] (0.69) | 0.54 (0.72) | 0.68 [0.43, 0.93] (0.71) | 0.53 (0.71) |
| P1-spec/lr | cv | L6 | 0.70 [0.45, 0.95] (0.72) | 0.55 (0.71) | 0.62 [0.35, 0.89] (0.72) | 0.56 (0.71) |
| P1-spec/lr | cv_deepest | L27 | 0.65 [0.38, 0.92] (0.75) | 0.55 (0.75) | 0.58 [0.30, 0.86] (0.75) | 0.60 (0.72) |
| P1-spec/lr | post_edit_mean | L6-27 | 0.86 [0.68, 1.00] (0.71) | 0.48 (0.72) | 0.71 [0.47, 0.95] (0.76) | 0.50 (0.71) |
| P1-spec/lr | pre_edit_mean | L0-2 | 0.64 [0.34, 0.94] (0.67) | 0.66 (0.71) | 0.62 [0.35, 0.89] (0.70) | 0.77 (0.71) |
| P1-spec/dm | cv | L6 | 0.68 [0.41, 0.95] (0.70) | 0.58 (0.71) | 0.63 [0.36, 0.90] (0.69) | 0.62 (0.74) |
| P1-spec/dm | cv_deepest | L27 | 0.60 [0.32, 0.88] (0.70) | 0.58 (0.75) | 0.53 [0.25, 0.81] (0.76) | 0.62 (0.73) |
| P1-spec/dm | post_edit_mean | L6-27 | 0.86 [0.67, 1.00] (0.70) | 0.52 (0.73) | 0.77 [0.55, 0.99] (0.73) | 0.54 (0.72) |
| P1-spec/dm | pre_edit_mean | L0-2 | 0.64 [0.36, 0.92] (0.70) | 0.54 (0.74) | 0.68 [0.42, 0.94] (0.70) | 0.68 (0.71) |
| P2-fingerprint/lr | cv | L5 | 0.62 [0.34, 0.90] (0.72) | 0.47 (0.74) | 0.62 [0.34, 0.90] (0.72) | 0.46 (0.75) |
| P2-fingerprint/lr | cv_deepest | L27 | 0.51 [0.23, 0.79] (0.68) | 0.45 (0.73) | 0.50 [0.22, 0.78] (0.71) | 0.49 (0.71) |
| P2-fingerprint/lr | post_edit_mean | L6-27 | 0.65 [0.38, 0.92] (0.73) | 0.48 (0.72) | 0.61 [0.33, 0.89] (0.72) | 0.48 (0.73) |
| P2-fingerprint/lr | pre_edit_mean | L0-2 | 0.59 [0.30, 0.88] (0.69) | 0.64 (0.71) | 0.49 [0.21, 0.77] (0.71) | 0.73 (0.72) |
| P2-fingerprint/dm | cv | L6 | 0.58 [0.31, 0.85] (0.71) | 0.55 (0.73) | 0.52 [0.24, 0.80] (0.71) | 0.51 (0.72) |
| P2-fingerprint/dm | cv_deepest | L27 | 0.55 [0.27, 0.83] (0.72) | 0.43 (0.75) | 0.44 [0.16, 0.72] (0.75) | 0.48 (0.75) |
| P2-fingerprint/dm | post_edit_mean | L6-27 | 0.76 [0.53, 0.99] (0.71) | 0.42 (0.71) | 0.59 [0.31, 0.87] (0.73) | 0.42 (0.73) |
| P2-fingerprint/dm | pre_edit_mean | L0-2 | 0.62 [0.33, 0.91] (0.69) | 0.54 (0.72) | 0.57 [0.27, 0.87] (0.69) | 0.83 (0.73) |
