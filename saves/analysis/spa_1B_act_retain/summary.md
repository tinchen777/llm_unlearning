# A2 SPA 1B act (retain)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | L6: 0.82 | 0.73 | 0.55 [0.25, 0.85] (0.73) | L2: 0.71 | 0.57 | - |
| P0-familiarity/dm | 0.77 | 7 | 0.74 [0.50, 0.98] (0.68) | L3: 0.77 | 0.72 | 0.18 [0.00, 0.38] (0.73) | L4: 0.59 | 0.51 | - |
| P1-spec/lr | 1.00 | 8 | 0.59 [0.30, 0.88] (0.68) | L3: 0.67 | 0.61 | 0.46 [0.17, 0.75] (0.74) | L5: 0.73 | 0.65 | 0.50, 0.48, 0.63, 0.54, 0.64 |
| P1-spec/dm | 1.00 | 9 | 0.42 [0.13, 0.71] (0.70) | L5: 0.69 | 0.65 | 0.60 [0.32, 0.88] (0.68) | L5: 0.69 | 0.61 | 0.32, 0.43, 0.55, 0.61, 0.54 |
| P2-fingerprint/lr | 1.00 | 8 | 0.55 [0.26, 0.84] (0.70) | L4: 0.60 | 0.55 | 0.58 [0.28, 0.88] (0.69) | L5: 0.78 | 0.66 | 0.57, 0.55, 0.65, 0.55, 0.62 |
| P2-fingerprint/dm | 1.00 | 8 | 0.53 [0.24, 0.82] (0.69) | L1: 0.62 | 0.56 | 0.59 [0.32, 0.86] (0.71) | L5: 0.84 | 0.65 | 0.45, 0.35, 0.66, 0.60, 0.53 |

Edit layers [7, 8, 9]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.25, P0-familiarity/dm: -0.59, P1-spec/lr: 0.02, P1-spec/dm: -0.17, P2-fingerprint/lr: 0.08, P2-fingerprint/dm: 0.09.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.23851082064211368, 0.15562440175563097, 0.20915930941700933, 0.1412528118118644, 0.19982504267245532], "shadow_HA_prob": [0.14053682954981922, 0.1387915471987799, 0.13180126887280494, 0.1363139027217403, 0.13809937668032946], "shadow_F_prob": [0.12151328321546315, 0.12058925096178427, 0.11494148685596883, 0.12170591564150526, 0.12372512063593606], "target": {"S": 0.8618419727683069, "F": 0.12318499884800986, "H_A": 0.14282301764935257, "H_B": 0.13058040514122693, "R_A": 0.8669556814432144, "R_B": 0.8746421876549721}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.73) | 0.54 [0.26, 0.82] (0.68) | 0.55 (0.70) |
| P0-familiarity/lr | cv_deepest | L0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.73) | 0.54 [0.26, 0.82] (0.68) | 0.55 (0.70) |
| P0-familiarity/lr | post_edit_mean | L10-15 | 0.58 [0.30, 0.86] (0.75) | 0.36 (0.69) | 0.55 [0.27, 0.83] (0.74) | 0.31 (0.69) |
| P0-familiarity/lr | pre_edit_mean | L0-6 | 0.75 [0.50, 1.00] (0.70) | 0.57 (0.73) | 0.67 [0.40, 0.94] (0.70) | 0.61 (0.73) |
| P0-familiarity/dm | cv | L7 | 0.74 [0.50, 0.98] (0.68) | 0.18 (0.73) | 0.64 [0.37, 0.91] (0.73) | 0.18 (0.73) |
| P0-familiarity/dm | cv_deepest | L9 | 0.73 [0.49, 0.97] (0.71) | 0.40 (0.74) | 0.68 [0.42, 0.94] (0.70) | 0.37 (0.74) |
| P0-familiarity/dm | post_edit_mean | L10-15 | 0.46 [0.18, 0.74] (0.73) | 0.44 (0.70) | 0.47 [0.19, 0.75] (0.73) | 0.43 (0.70) |
| P0-familiarity/dm | pre_edit_mean | L0-6 | 0.74 [0.50, 0.98] (0.71) | 0.51 (0.72) | 0.75 [0.51, 0.99] (0.70) | 0.55 (0.74) |
| P1-spec/lr | cv | L8 | 0.59 [0.30, 0.88] (0.68) | 0.46 (0.74) | 0.59 [0.30, 0.88] (0.68) | 0.47 (0.74) |
| P1-spec/lr | cv_deepest | L10 | 0.48 [0.20, 0.76] (0.68) | 0.44 (0.72) | 0.46 [0.17, 0.75] (0.69) | 0.46 (0.72) |
| P1-spec/lr | post_edit_mean | L10-15 | 0.42 [0.14, 0.70] (0.72) | 0.44 (0.72) | 0.35 [0.07, 0.63] (0.72) | 0.45 (0.73) |
| P1-spec/lr | pre_edit_mean | L0-6 | 0.64 [0.37, 0.91] (0.71) | 0.66 (0.72) | 0.60 [0.32, 0.88] (0.69) | 0.70 (0.72) |
| P1-spec/dm | cv | L9 | 0.42 [0.13, 0.71] (0.70) | 0.60 (0.68) | 0.43 [0.14, 0.72] (0.70) | 0.60 (0.67) |
| P1-spec/dm | cv_deepest | L10 | 0.40 [0.12, 0.68] (0.71) | 0.43 (0.73) | 0.37 [0.11, 0.63] (0.71) | 0.45 (0.73) |
| P1-spec/dm | post_edit_mean | L10-15 | 0.37 [0.09, 0.65] (0.70) | 0.52 (0.73) | 0.37 [0.10, 0.64] (0.69) | 0.54 (0.72) |
| P1-spec/dm | pre_edit_mean | L0-6 | 0.63 [0.35, 0.91] (0.68) | 0.62 (0.71) | 0.60 [0.33, 0.87] (0.69) | 0.72 (0.71) |
| P2-fingerprint/lr | cv | L8 | 0.55 [0.26, 0.84] (0.70) | 0.58 (0.69) | 0.58 [0.28, 0.88] (0.69) | 0.59 (0.71) |
| P2-fingerprint/lr | cv_deepest | L9 | 0.43 [0.15, 0.71] (0.69) | 0.63 (0.70) | 0.44 [0.15, 0.73] (0.67) | 0.64 (0.68) |
| P2-fingerprint/lr | post_edit_mean | L10-15 | 0.34 [0.06, 0.62] (0.70) | 0.34 (0.69) | 0.27 [0.01, 0.53] (0.70) | 0.36 (0.69) |
| P2-fingerprint/lr | pre_edit_mean | L0-6 | 0.61 [0.34, 0.88] (0.70) | 0.72 (0.74) | 0.55 [0.27, 0.83] (0.70) | 0.73 (0.74) |
| P2-fingerprint/dm | cv | L8 | 0.53 [0.24, 0.82] (0.69) | 0.59 (0.71) | 0.56 [0.27, 0.85] (0.68) | 0.58 (0.71) |
| P2-fingerprint/dm | cv_deepest | L10 | 0.36 [0.08, 0.64] (0.70) | 0.51 (0.70) | 0.37 [0.08, 0.66] (0.70) | 0.50 (0.70) |
| P2-fingerprint/dm | post_edit_mean | L10-15 | 0.31 [0.06, 0.56] (0.67) | 0.49 (0.70) | 0.26 [0.02, 0.50] (0.68) | 0.51 (0.71) |
| P2-fingerprint/dm | pre_edit_mean | L0-6 | 0.58 [0.29, 0.87] (0.67) | 0.67 (0.70) | 0.54 [0.26, 0.82] (0.70) | 0.76 (0.72) |
