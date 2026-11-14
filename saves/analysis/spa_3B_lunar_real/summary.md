# A2 SPA 3B lunar (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.80 | 13 | 0.61 [0.33, 0.89] (0.70) | L11: 0.94 | 0.80 | 0.14 [0.00, 0.33] (0.70) | L4: 0.60 | 0.53 | - |
| P0-familiarity/dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | L10: 0.92 | 0.79 | 0.56 [0.27, 0.85] (0.71) | L3: 0.59 | 0.52 | - |
| P1-spec/lr | 0.96 | 13 | 1.00 [1.00, 1.00] (0.69) | L13: 1.00 | 0.80 | 0.96 [0.87, 1.00] (0.70) | L14: 1.00 | 0.64 | 0.99, 1.00, 1.00, 0.99, 1.00 |
| P1-spec/dm | 0.92 | 13 | 0.96 [0.87, 1.00] (0.71) | L18: 1.00 | 0.74 | 1.00 [1.00, 1.00] (0.72) | L13: 1.00 | 0.58 | 0.91, 0.93, 0.96, 0.93, 0.97 |
| P2-fingerprint/lr | 0.97 | 15 | 1.00 [1.00, 1.00] (0.70) | L13: 1.00 | 0.68 | 1.00 [1.00, 1.00] (0.72) | L13: 1.00 | 0.71 | 1.00, 0.97, 1.00, 1.00, 1.00 |
| P2-fingerprint/dm | 0.96 | 14 | 0.96 [0.88, 1.00] (0.72) | L15: 0.99 | 0.66 | 1.00 [1.00, 1.00] (0.72) | L13: 1.00 | 0.65 | 0.92, 0.92, 0.97, 0.93, 0.97 |

Edit layers [12]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.61, P0-familiarity/dm: -0.74, P1-spec/lr: -0.40, P1-spec/dm: 0.00, P2-fingerprint/lr: -0.18, P2-fingerprint/dm: 0.03.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.18054752200841903, 0.20147912218235434, 0.2436218830756843, 0.2255334339570254, 0.21117121940478684], "shadow_HA_prob": [0.1218963905144483, 0.12321346354205161, 0.12065024662297219, 0.12169300437904895, 0.13019342246465385], "shadow_F_prob": [0.1250289248209447, 0.11570919519290328, 0.12337756023742259, 0.1317091099685058, 0.16476681785658004], "target": {"S": 0.4814080424979329, "F": 0.23033567177131772, "H_A": 0.1366448554955423, "H_B": 0.12019623098894953, "R_A": 0.5516937566921116, "R_B": 0.5582615783065559}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L13 | 0.61 [0.33, 0.89] (0.70) | 0.14 (0.70) | 0.48 [0.20, 0.76] (0.71) | 0.17 (0.71) |
| P0-familiarity/lr | cv_deepest | L14 | 0.72 [0.47, 0.97] (0.70) | 0.20 (0.70) | 0.59 [0.32, 0.86] (0.71) | 0.19 (0.71) |
| P0-familiarity/lr | post_edit_mean | L13-27 | 0.37 [0.10, 0.64] (0.72) | 0.21 (0.73) | 0.31 [0.05, 0.57] (0.71) | 0.20 (0.72) |
| P0-familiarity/lr | pre_edit_mean | L0-11 | 0.81 [0.59, 1.00] (0.71) | 0.54 (0.73) | 0.77 [0.52, 1.00] (0.70) | 0.51 (0.71) |
| P0-familiarity/dm | cv | L10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | 0.93 [0.81, 1.00] (0.72) | 0.54 (0.70) |
| P0-familiarity/dm | cv_deepest | L10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | 0.93 [0.81, 1.00] (0.72) | 0.54 (0.70) |
| P0-familiarity/dm | post_edit_mean | L13-27 | 0.15 [0.00, 0.34] (0.70) | 0.07 (0.72) | 0.10 [0.00, 0.24] (0.72) | 0.07 (0.73) |
| P0-familiarity/dm | pre_edit_mean | L0-11 | 0.78 [0.55, 1.00] (0.69) | 0.52 (0.73) | 0.87 [0.70, 1.00] (0.69) | 0.60 (0.71) |
| P1-spec/lr | cv | L13 | 1.00 [1.00, 1.00] (0.69) | 0.96 (0.70) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.70) |
| P1-spec/lr | cv_deepest | L13 | 1.00 [1.00, 1.00] (0.69) | 0.96 (0.70) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.70) |
| P1-spec/lr | post_edit_mean | L13-27 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.72) |
| P1-spec/lr | pre_edit_mean | L0-11 | 0.86 [0.68, 1.00] (0.71) | 0.70 (0.72) | 0.87 [0.71, 1.00] (0.71) | 0.75 (0.72) |
| P1-spec/dm | cv | L13 | 0.96 [0.87, 1.00] (0.71) | 1.00 (0.72) | 0.96 [0.87, 1.00] (0.71) | 1.00 (0.72) |
| P1-spec/dm | cv_deepest | L14 | 0.96 [0.87, 1.00] (0.72) | 1.00 (0.72) | 0.97 [0.90, 1.00] (0.72) | 1.00 (0.72) |
| P1-spec/dm | post_edit_mean | L13-27 | 0.99 [0.96, 1.00] (0.70) | 1.00 (0.72) | 0.99 [0.96, 1.00] (0.70) | 1.00 (0.72) |
| P1-spec/dm | pre_edit_mean | L0-11 | 0.73 [0.47, 0.99] (0.67) | 0.60 (0.72) | 0.80 [0.57, 1.00] (0.69) | 0.66 (0.70) |
| P2-fingerprint/lr | cv | L15 | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.72) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.71) |
| P2-fingerprint/lr | cv_deepest | L15 | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.72) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.71) |
| P2-fingerprint/lr | post_edit_mean | L13-27 | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.69) |
| P2-fingerprint/lr | pre_edit_mean | L0-11 | 0.71 [0.46, 0.96] (0.67) | 0.80 (0.71) | 0.70 [0.45, 0.95] (0.69) | 0.88 (0.73) |
| P2-fingerprint/dm | cv | L14 | 0.96 [0.88, 1.00] (0.72) | 1.00 (0.72) | 0.97 [0.90, 1.00] (0.72) | 1.00 (0.72) |
| P2-fingerprint/dm | cv_deepest | L14 | 0.96 [0.88, 1.00] (0.72) | 1.00 (0.72) | 0.97 [0.90, 1.00] (0.72) | 1.00 (0.72) |
| P2-fingerprint/dm | post_edit_mean | L13-27 | 0.99 [0.96, 1.00] (0.71) | 1.00 (0.73) | 0.99 [0.96, 1.00] (0.70) | 1.00 (0.72) |
| P2-fingerprint/dm | pre_edit_mean | L0-11 | 0.65 [0.36, 0.94] (0.70) | 0.68 (0.72) | 0.62 [0.35, 0.89] (0.68) | 0.79 (0.70) |
