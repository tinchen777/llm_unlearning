# A2 SPA 3B rocr (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.90 | 12 | 0.66 [0.38, 0.94] (0.70) | L2: 0.86 | 0.73 | 0.43 [0.13, 0.73] (0.69) | L3: 0.57 | 0.52 | - |
| P0-familiarity/dm | 0.87 | 20 | 0.04 [0.00, 0.13] (0.71) | L10: 0.90 | 0.67 | 0.00 [0.00, 0.00] (0.72) | L10: 0.60 | 0.50 | - |
| P1-spec/lr | 1.00 | 5 | 1.00 [1.00, 1.00] (0.69) | L5: 1.00 | 0.68 | 0.99 [0.96, 1.00] (0.70) | L6: 1.00 | 0.66 | 0.98, 1.00, 0.96, 0.95, 0.96 |
| P1-spec/dm | 1.00 | 6 | 1.00 [1.00, 1.00] (0.68) | L6: 1.00 | 0.65 | 1.00 [1.00, 1.00] (0.70) | L6: 1.00 | 0.53 | 1.00, 1.00, 1.00, 0.99, 0.99 |
| P2-fingerprint/lr | 1.00 | 4 | 1.00 [1.00, 1.00] (0.69) | L4: 1.00 | 0.57 | 0.93 [0.82, 1.00] (0.70) | L5: 1.00 | 0.65 | 0.98, 0.99, 0.98, 0.95, 0.97 |
| P2-fingerprint/dm | 1.00 | 5 | 0.99 [0.96, 1.00] (0.70) | L6: 1.00 | 0.63 | 1.00 [1.00, 1.00] (0.70) | L5: 1.00 | 0.53 | 0.95, 0.95, 0.91, 0.90, 0.93 |

Edit layers [3, 4, 5]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.42, P0-familiarity/dm: 0.29, P1-spec/lr: -0.31, P1-spec/dm: -0.37, P2-fingerprint/lr: -0.32, P2-fingerprint/dm: -0.29.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.566249778866768, 0.5521181482821703, 0.528918043896556, 0.4334936738014221, 0.40208358950912954], "shadow_HA_prob": [0.14166301527060568, 0.1418131967028603, 0.1406242475705221, 0.1386666787136346, 0.1393522219732404], "shadow_F_prob": [0.5396985123865307, 0.5199821988120676, 0.512068672478199, 0.5166400392726064, 0.48944725004956124], "target": {"S": 0.9343484070897101, "F": 0.5532025169767439, "H_A": 0.14338120010681452, "H_B": 0.13266220211982727, "R_A": 0.9381293189525604, "R_B": 0.9415253412723542}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L12 | 0.66 [0.38, 0.94] (0.70) | 0.43 (0.69) | 0.62 [0.33, 0.91] (0.70) | 0.41 (0.71) |
| P0-familiarity/lr | cv_deepest | L12 | 0.66 [0.38, 0.94] (0.70) | 0.43 (0.69) | 0.62 [0.33, 0.91] (0.70) | 0.41 (0.71) |
| P0-familiarity/lr | post_edit_mean | L6-27 | 0.36 [0.06, 0.66] (0.67) | 0.22 (0.71) | 0.26 [0.00, 0.53] (0.70) | 0.24 (0.74) |
| P0-familiarity/lr | pre_edit_mean | L0-2 | 0.72 [0.47, 0.97] (0.73) | 0.55 (0.72) | 0.64 [0.37, 0.91] (0.70) | 0.54 (0.71) |
| P0-familiarity/dm | cv | L20 | 0.04 [0.00, 0.13] (0.71) | 0.00 (0.72) | 0.04 [0.00, 0.13] (0.71) | 0.00 (0.72) |
| P0-familiarity/dm | cv_deepest | L20 | 0.04 [0.00, 0.13] (0.71) | 0.00 (0.72) | 0.04 [0.00, 0.13] (0.71) | 0.00 (0.72) |
| P0-familiarity/dm | post_edit_mean | L6-27 | 0.52 [0.23, 0.81] (0.69) | 0.35 (0.72) | 0.41 [0.10, 0.72] (0.70) | 0.29 (0.72) |
| P0-familiarity/dm | pre_edit_mean | L0-2 | 0.63 [0.36, 0.90] (0.70) | 0.53 (0.72) | 0.66 [0.40, 0.92] (0.71) | 0.53 (0.71) |
| P1-spec/lr | cv | L5 | 1.00 [1.00, 1.00] (0.69) | 0.99 (0.70) | 0.97 [0.91, 1.00] (0.71) | 1.00 (0.72) |
| P1-spec/lr | cv_deepest | L16 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.69) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) |
| P1-spec/lr | post_edit_mean | L6-27 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.68) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) |
| P1-spec/lr | pre_edit_mean | L0-2 | 0.63 [0.33, 0.93] (0.68) | 0.68 (0.72) | 0.61 [0.34, 0.88] (0.70) | 0.80 (0.71) |
| P1-spec/dm | cv | L6 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.72) | 1.00 (0.69) |
| P1-spec/dm | cv_deepest | L20 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.68) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) |
| P1-spec/dm | post_edit_mean | L6-27 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.69) | 1.00 [1.00, 1.00] (0.70) | 1.00 (0.70) |
| P1-spec/dm | pre_edit_mean | L0-2 | 0.64 [0.36, 0.92] (0.69) | 0.54 (0.73) | 0.65 [0.38, 0.92] (0.70) | 0.61 (0.70) |
| P2-fingerprint/lr | cv | L4 | 1.00 [1.00, 1.00] (0.69) | 0.93 (0.70) | 0.99 [0.96, 1.00] (0.71) | 0.99 (0.72) |
| P2-fingerprint/lr | cv_deepest | L27 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.68) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.69) |
| P2-fingerprint/lr | post_edit_mean | L6-27 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.69) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) |
| P2-fingerprint/lr | pre_edit_mean | L0-2 | 0.65 [0.36, 0.94] (0.69) | 0.66 (0.70) | 0.50 [0.22, 0.78] (0.70) | 0.78 (0.73) |
| P2-fingerprint/dm | cv | L5 | 0.99 [0.96, 1.00] (0.70) | 1.00 (0.70) | 0.98 [0.93, 1.00] (0.71) | 1.00 (0.71) |
| P2-fingerprint/dm | cv_deepest | L27 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) |
| P2-fingerprint/dm | post_edit_mean | L6-27 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.68) | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.70) |
| P2-fingerprint/dm | pre_edit_mean | L0-2 | 0.62 [0.33, 0.91] (0.69) | 0.53 (0.72) | 0.57 [0.27, 0.87] (0.69) | 0.81 (0.73) |
