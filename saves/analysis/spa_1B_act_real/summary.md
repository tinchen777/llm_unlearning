# A2 SPA 1B act (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | L3: 0.79 | 0.72 | 0.55 [0.25, 0.85] (0.71) | L2: 0.70 | 0.56 | - |
| P0-familiarity/dm | 0.76 | 1 | 0.70 [0.42, 0.98] (0.66) | L3: 0.77 | 0.72 | 0.56 [0.26, 0.86] (0.68) | L7: 0.63 | 0.47 | - |
| P1-spec/lr | 1.00 | 10 | 0.43 [0.13, 0.73] (0.74) | L5: 0.86 | 0.70 | 0.63 [0.35, 0.91] (0.70) | L5: 0.90 | 0.70 | 0.47, 0.52, 0.48, 0.51, 0.51 |
| P1-spec/dm | 1.00 | 11 | 0.40 [0.10, 0.70] (0.72) | L6: 0.82 | 0.69 | 0.51 [0.21, 0.81] (0.74) | L5: 0.86 | 0.63 | 0.41, 0.46, 0.50, 0.44, 0.39 |
| P2-fingerprint/lr | 1.00 | 10 | 0.42 [0.12, 0.72] (0.75) | L6: 0.86 | 0.62 | 0.56 [0.25, 0.87] (0.72) | L5: 0.84 | 0.67 | 0.47, 0.51, 0.46, 0.51, 0.48 |
| P2-fingerprint/dm | 1.00 | 15 | 0.40 [0.12, 0.68] (0.66) | L6: 0.83 | 0.62 | 0.56 [0.25, 0.87] (0.71) | L5: 0.88 | 0.66 | 0.40, 0.40, 0.43, 0.41, 0.44 |

Edit layers [7, 8, 9]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.25, P0-familiarity/dm: -0.78, P1-spec/lr: 0.15, P1-spec/dm: 0.04, P2-fingerprint/lr: 0.22, P2-fingerprint/dm: 0.04.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.1896651070099324, 0.1530736137414351, 0.1635220506694168, 0.18749234252609312, 0.20244798469357192], "shadow_HA_prob": [0.13260530146304522, 0.12847291069105268, 0.12545155942905695, 0.13402774831745773, 0.13159936011303217], "shadow_F_prob": [0.24665611498057838, 0.2384755019377917, 0.23283521587029102, 0.23480609956197443, 0.23368612346239384], "target": {"S": 0.6411490383744239, "F": 0.23429526054998862, "H_A": 0.13621852127369496, "H_B": 0.1205437208758667, "R_A": 0.63004358664155, "R_B": 0.6552213056385517}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | 0.57 [0.29, 0.85] (0.69) | 0.57 (0.70) |
| P0-familiarity/lr | cv_deepest | L0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | 0.57 [0.29, 0.85] (0.69) | 0.57 (0.70) |
| P0-familiarity/lr | post_edit_mean | L10-15 | 0.59 [0.32, 0.86] (0.70) | 0.45 (0.71) | 0.56 [0.29, 0.83] (0.70) | 0.42 (0.69) |
| P0-familiarity/lr | pre_edit_mean | L0-6 | 0.74 [0.49, 0.99] (0.70) | 0.55 (0.72) | 0.69 [0.43, 0.95] (0.69) | 0.59 (0.74) |
| P0-familiarity/dm | cv | L1 | 0.70 [0.42, 0.98] (0.66) | 0.56 (0.68) | 0.66 [0.40, 0.92] (0.68) | 0.56 (0.69) |
| P0-familiarity/dm | cv_deepest | L1 | 0.70 [0.42, 0.98] (0.66) | 0.56 (0.68) | 0.66 [0.40, 0.92] (0.68) | 0.56 (0.69) |
| P0-familiarity/dm | post_edit_mean | L10-15 | 0.59 [0.31, 0.87] (0.72) | 0.54 (0.71) | 0.59 [0.31, 0.87] (0.72) | 0.55 (0.72) |
| P0-familiarity/dm | pre_edit_mean | L0-6 | 0.72 [0.46, 0.98] (0.69) | 0.49 (0.71) | 0.71 [0.45, 0.97] (0.69) | 0.49 (0.72) |
| P1-spec/lr | cv | L10 | 0.43 [0.13, 0.73] (0.74) | 0.63 (0.70) | 0.46 [0.16, 0.76] (0.72) | 0.59 (0.70) |
| P1-spec/lr | cv_deepest | L13 | 0.47 [0.18, 0.76] (0.70) | 0.60 (0.71) | 0.46 [0.18, 0.74] (0.71) | 0.59 (0.71) |
| P1-spec/lr | post_edit_mean | L10-15 | 0.40 [0.12, 0.68] (0.70) | 0.57 (0.70) | 0.41 [0.13, 0.69] (0.70) | 0.56 (0.72) |
| P1-spec/lr | pre_edit_mean | L0-6 | 0.75 [0.51, 0.99] (0.70) | 0.79 (0.73) | 0.73 [0.49, 0.97] (0.70) | 0.80 (0.73) |
| P1-spec/dm | cv | L11 | 0.40 [0.10, 0.70] (0.72) | 0.51 (0.74) | 0.40 [0.10, 0.70] (0.72) | 0.51 (0.73) |
| P1-spec/dm | cv_deepest | L12 | 0.43 [0.14, 0.72] (0.71) | 0.58 (0.72) | 0.43 [0.14, 0.72] (0.71) | 0.58 (0.72) |
| P1-spec/dm | post_edit_mean | L10-15 | 0.40 [0.12, 0.68] (0.70) | 0.55 (0.71) | 0.40 [0.12, 0.68] (0.70) | 0.55 (0.73) |
| P1-spec/dm | pre_edit_mean | L0-6 | 0.67 [0.40, 0.94] (0.67) | 0.62 (0.71) | 0.67 [0.41, 0.93] (0.71) | 0.80 (0.71) |
| P2-fingerprint/lr | cv | L10 | 0.42 [0.12, 0.72] (0.75) | 0.56 (0.72) | 0.44 [0.14, 0.74] (0.73) | 0.55 (0.75) |
| P2-fingerprint/lr | cv_deepest | L15 | 0.42 [0.14, 0.70] (0.69) | 0.59 (0.67) | 0.44 [0.17, 0.71] (0.68) | 0.49 (0.71) |
| P2-fingerprint/lr | post_edit_mean | L10-15 | 0.41 [0.13, 0.69] (0.73) | 0.62 (0.73) | 0.44 [0.15, 0.73] (0.70) | 0.55 (0.74) |
| P2-fingerprint/lr | pre_edit_mean | L0-6 | 0.68 [0.42, 0.94] (0.71) | 0.73 (0.74) | 0.67 [0.40, 0.94] (0.71) | 0.73 (0.73) |
| P2-fingerprint/dm | cv | L15 | 0.40 [0.12, 0.68] (0.66) | 0.56 (0.71) | 0.38 [0.11, 0.65] (0.67) | 0.55 (0.71) |
| P2-fingerprint/dm | cv_deepest | L15 | 0.40 [0.12, 0.68] (0.66) | 0.56 (0.71) | 0.38 [0.11, 0.65] (0.67) | 0.55 (0.71) |
| P2-fingerprint/dm | post_edit_mean | L10-15 | 0.41 [0.11, 0.71] (0.73) | 0.50 (0.75) | 0.41 [0.11, 0.71] (0.72) | 0.49 (0.74) |
| P2-fingerprint/dm | pre_edit_mean | L0-6 | 0.65 [0.37, 0.93] (0.67) | 0.76 (0.73) | 0.65 [0.39, 0.91] (0.69) | 0.84 (0.72) |
