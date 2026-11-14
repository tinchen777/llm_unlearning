# A2 SPA 3B lunar (retain)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.85 | 9 | 0.84 [0.66, 1.00] (0.71) | L9: 0.84 | 0.74 | 0.35 [0.09, 0.61] (0.72) | L1: 0.55 | 0.44 | - |
| P0-familiarity/dm | 0.85 | 12 | 0.75 [0.52, 0.98] (0.68) | L5: 0.79 | 0.74 | 0.45 [0.16, 0.74] (0.69) | L22: 0.66 | 0.47 | - |
| P1-spec/lr | 1.00 | 13 | 0.60 [0.32, 0.88] (0.67) | L17: 0.87 | 0.72 | 0.41 [0.11, 0.71] (0.71) | L2: 0.73 | 0.58 | 0.35, 0.65, 0.60, 0.57, 0.51 |
| P1-spec/dm | 0.99 | 21 | 0.46 [0.18, 0.74] (0.72) | L18: 0.80 | 0.66 | 0.32 [0.06, 0.58] (0.72) | L5: 0.66 | 0.54 | 0.33, 0.54, 0.44, 0.45, 0.45 |
| P2-fingerprint/lr | 1.00 | 12 | 0.80 [0.57, 1.00] (0.70) | L12: 0.80 | 0.62 | 0.62 [0.32, 0.92] (0.73) | L8: 0.81 | 0.71 | 0.47, 0.65, 0.51, 0.60, 0.43 |
| P2-fingerprint/dm | 0.99 | 20 | 0.61 [0.34, 0.88] (0.74) | L18: 0.70 | 0.58 | 0.33 [0.07, 0.59] (0.74) | L6: 0.74 | 0.60 | 0.30, 0.49, 0.49, 0.44, 0.47 |

Edit layers [12]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.55, P0-familiarity/dm: -0.58, P1-spec/lr: -0.45, P1-spec/dm: -0.18, P2-fingerprint/lr: -0.36, P2-fingerprint/dm: 0.22.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.1604091303423047, 0.23689426872879266, 0.15838328110985458, 0.13331690463237464, 0.191534058470279], "shadow_HA_prob": [0.15224824140779675, 0.16500885169021787, 0.14774116395041348, 0.15336206651292744, 0.15981180229224265], "shadow_F_prob": [0.12601094080135228, 0.13412917394656687, 0.12386785208713265, 0.122258638930507, 0.1328786515677348], "target": {"S": 0.9402199801802636, "F": 0.12411896957084537, "H_A": 0.15010720291640609, "H_B": 0.13045282321516424, "R_A": 0.9380298426747322, "R_B": 0.9398281663656235}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L9 | 0.84 [0.66, 1.00] (0.71) | 0.35 (0.72) | 0.81 [0.61, 1.00] (0.70) | 0.33 (0.71) |
| P0-familiarity/lr | cv_deepest | L9 | 0.84 [0.66, 1.00] (0.71) | 0.35 (0.72) | 0.81 [0.61, 1.00] (0.70) | 0.33 (0.71) |
| P0-familiarity/lr | post_edit_mean | L13-27 | 0.60 [0.32, 0.88] (0.70) | 0.33 (0.70) | 0.57 [0.29, 0.85] (0.70) | 0.35 (0.70) |
| P0-familiarity/lr | pre_edit_mean | L0-11 | 0.74 [0.50, 0.98] (0.71) | 0.46 (0.71) | 0.67 [0.40, 0.94] (0.70) | 0.45 (0.70) |
| P0-familiarity/dm | cv | L12 | 0.75 [0.52, 0.98] (0.68) | 0.45 (0.69) | 0.77 [0.53, 1.00] (0.69) | 0.39 (0.70) |
| P0-familiarity/dm | cv_deepest | L13 | 0.74 [0.51, 0.97] (0.67) | 0.42 (0.69) | 0.70 [0.44, 0.96] (0.68) | 0.40 (0.71) |
| P0-familiarity/dm | post_edit_mean | L13-27 | 0.60 [0.33, 0.87] (0.71) | 0.48 (0.69) | 0.60 [0.33, 0.87] (0.72) | 0.49 (0.69) |
| P0-familiarity/dm | pre_edit_mean | L0-11 | 0.76 [0.51, 1.00] (0.70) | 0.45 (0.71) | 0.76 [0.52, 1.00] (0.68) | 0.48 (0.70) |
| P1-spec/lr | cv | L13 | 0.60 [0.32, 0.88] (0.67) | 0.41 (0.71) | 0.55 [0.26, 0.84] (0.73) | 0.34 (0.71) |
| P1-spec/lr | cv_deepest | L27 | 0.52 [0.24, 0.80] (0.70) | 0.54 (0.69) | 0.48 [0.19, 0.77] (0.72) | 0.55 (0.70) |
| P1-spec/lr | post_edit_mean | L13-27 | 0.62 [0.35, 0.89] (0.71) | 0.54 (0.72) | 0.60 [0.33, 0.87] (0.71) | 0.52 (0.72) |
| P1-spec/lr | pre_edit_mean | L0-11 | 0.72 [0.46, 0.98] (0.68) | 0.63 (0.72) | 0.70 [0.44, 0.96] (0.69) | 0.60 (0.71) |
| P1-spec/dm | cv | L21 | 0.46 [0.18, 0.74] (0.72) | 0.32 (0.72) | 0.40 [0.12, 0.68] (0.72) | 0.32 (0.74) |
| P1-spec/dm | cv_deepest | L27 | 0.52 [0.24, 0.80] (0.70) | 0.30 (0.72) | 0.45 [0.17, 0.73] (0.72) | 0.34 (0.71) |
| P1-spec/dm | post_edit_mean | L13-27 | 0.48 [0.20, 0.76] (0.73) | 0.37 (0.73) | 0.41 [0.12, 0.70] (0.74) | 0.34 (0.74) |
| P1-spec/dm | pre_edit_mean | L0-11 | 0.66 [0.37, 0.95] (0.66) | 0.55 (0.73) | 0.61 [0.34, 0.88] (0.68) | 0.59 (0.70) |
| P2-fingerprint/lr | cv | L12 | 0.80 [0.57, 1.00] (0.70) | 0.62 (0.73) | 0.63 [0.36, 0.90] (0.74) | 0.57 (0.73) |
| P2-fingerprint/lr | cv_deepest | L27 | 0.45 [0.17, 0.73] (0.71) | 0.36 (0.71) | 0.41 [0.14, 0.68] (0.72) | 0.36 (0.72) |
| P2-fingerprint/lr | post_edit_mean | L13-27 | 0.51 [0.23, 0.79] (0.73) | 0.41 (0.72) | 0.51 [0.23, 0.79] (0.73) | 0.46 (0.73) |
| P2-fingerprint/lr | pre_edit_mean | L0-11 | 0.68 [0.40, 0.96] (0.67) | 0.81 (0.72) | 0.60 [0.33, 0.87] (0.68) | 0.82 (0.73) |
| P2-fingerprint/dm | cv | L20 | 0.61 [0.34, 0.88] (0.74) | 0.33 (0.74) | 0.61 [0.34, 0.88] (0.74) | 0.32 (0.73) |
| P2-fingerprint/dm | cv_deepest | L27 | 0.41 [0.14, 0.68] (0.70) | 0.14 (0.72) | 0.38 [0.11, 0.65] (0.71) | 0.15 (0.71) |
| P2-fingerprint/dm | post_edit_mean | L13-27 | 0.45 [0.17, 0.73] (0.71) | 0.30 (0.74) | 0.41 [0.13, 0.69] (0.72) | 0.29 (0.75) |
| P2-fingerprint/dm | pre_edit_mean | L0-11 | 0.60 [0.30, 0.90] (0.68) | 0.63 (0.71) | 0.51 [0.22, 0.80] (0.68) | 0.72 (0.71) |
