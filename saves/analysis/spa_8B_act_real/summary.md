# A2 SPA 8B act (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.93 | 9 | 0.88 [0.71, 1.00] (0.68) | L15: 0.90 | 0.80 | 0.39 [0.12, 0.66] (0.70) | L29: 0.63 | 0.42 | - |
| P0-familiarity/dm | 0.94 | 9 | 0.85 [0.64, 1.00] (0.69) | L13: 0.88 | 0.78 | 0.42 [0.15, 0.69] (0.70) | L29: 0.68 | 0.41 | - |
| P1-spec/lr | 1.00 | 19 | 0.42 [0.12, 0.72] (0.72) | L8: 1.00 | 0.87 | 0.47 [0.17, 0.77] (0.72) | L28: 0.99 | 0.62 | 0.36, 0.42, 0.47, 0.58, 0.58 |
| P1-spec/dm | 1.00 | 21 | 0.54 [0.26, 0.82] (0.71) | L28: 0.98 | 0.78 | 0.65 [0.37, 0.93] (0.73) | L28: 0.98 | 0.55 | 0.46, 0.51, 0.61, 0.68, 0.65 |
| P2-fingerprint/lr | 1.00 | 19 | 0.38 [0.10, 0.66] (0.72) | L8: 0.99 | 0.82 | 0.46 [0.17, 0.75] (0.70) | L22: 0.94 | 0.73 | 0.34, 0.41, 0.46, 0.54, 0.54 |
| P2-fingerprint/dm | 1.00 | 21 | 0.45 [0.17, 0.73] (0.73) | L28: 0.97 | 0.72 | 0.59 [0.31, 0.87] (0.72) | L28: 0.97 | 0.64 | 0.36, 0.45, 0.56, 0.70, 0.60 |

Edit layers [17, 18, 19]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.60, P0-familiarity/dm: -0.62, P1-spec/lr: 0.25, P1-spec/dm: -0.02, P2-fingerprint/lr: 0.25, P2-fingerprint/dm: 0.13.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.19271587766706944, 0.14394893338903786, 0.21775969080626964, 0.200927193267853, 0.14572086199186743], "shadow_HA_prob": [0.12234709357842803, 0.12061608540359885, 0.1249826362449676, 0.12697552782483398, 0.12450753232464193], "shadow_F_prob": [0.17127725980244576, 0.18516633000224827, 0.17050964255817233, 0.16800914105959236, 0.17588133805897085], "target": {"S": 0.9098108580708505, "F": 0.18465829746797682, "H_A": 0.12594906071666628, "H_B": 0.11452165035065262, "R_A": 0.9190199612081053, "R_B": 0.9102886311709881}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | 0.85 [0.65, 1.00] (0.72) | 0.28 (0.71) |
| P0-familiarity/lr | cv_deepest | L9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | 0.85 [0.65, 1.00] (0.72) | 0.28 (0.71) |
| P0-familiarity/lr | post_edit_mean | L20-31 | 0.69 [0.42, 0.96] (0.69) | 0.48 (0.71) | 0.66 [0.39, 0.93] (0.70) | 0.53 (0.73) |
| P0-familiarity/lr | pre_edit_mean | L0-16 | 0.87 [0.70, 1.00] (0.69) | 0.44 (0.68) | 0.84 [0.65, 1.00] (0.70) | 0.38 (0.69) |
| P0-familiarity/dm | cv | L9 | 0.85 [0.64, 1.00] (0.69) | 0.42 (0.70) | 0.78 [0.54, 1.00] (0.71) | 0.35 (0.71) |
| P0-familiarity/dm | cv_deepest | L15 | 0.87 [0.68, 1.00] (0.66) | 0.39 (0.71) | 0.81 [0.61, 1.00] (0.71) | 0.38 (0.72) |
| P0-familiarity/dm | post_edit_mean | L20-31 | 0.64 [0.36, 0.92] (0.69) | 0.55 (0.71) | 0.58 [0.31, 0.85] (0.70) | 0.54 (0.71) |
| P0-familiarity/dm | pre_edit_mean | L0-16 | 0.82 [0.61, 1.00] (0.70) | 0.43 (0.71) | 0.80 [0.59, 1.00] (0.71) | 0.37 (0.70) |
| P1-spec/lr | cv | L19 | 0.42 [0.12, 0.72] (0.72) | 0.47 (0.72) | 0.50 [0.22, 0.78] (0.69) | 0.49 (0.72) |
| P1-spec/lr | cv_deepest | L31 | 0.75 [0.52, 0.98] (0.69) | 0.81 (0.74) | 0.78 [0.56, 1.00] (0.68) | 0.82 (0.73) |
| P1-spec/lr | post_edit_mean | L20-31 | 0.93 [0.82, 1.00] (0.71) | 0.95 (0.71) | 0.94 [0.84, 1.00] (0.71) | 0.98 (0.72) |
| P1-spec/lr | pre_edit_mean | L0-16 | 0.96 [0.87, 1.00] (0.68) | 0.68 (0.73) | 0.98 [0.93, 1.00] (0.71) | 0.71 (0.75) |
| P1-spec/dm | cv | L21 | 0.54 [0.26, 0.82] (0.71) | 0.65 (0.73) | 0.59 [0.32, 0.86] (0.73) | 0.65 (0.74) |
| P1-spec/dm | cv_deepest | L31 | 0.74 [0.50, 0.98] (0.69) | 0.78 (0.74) | 0.75 [0.52, 0.98] (0.67) | 0.78 (0.73) |
| P1-spec/dm | post_edit_mean | L20-31 | 0.87 [0.71, 1.00] (0.71) | 0.89 (0.72) | 0.88 [0.73, 1.00] (0.72) | 0.92 (0.71) |
| P1-spec/dm | pre_edit_mean | L0-16 | 0.83 [0.64, 1.00] (0.68) | 0.59 (0.72) | 0.83 [0.63, 1.00] (0.69) | 0.64 (0.73) |
| P2-fingerprint/lr | cv | L19 | 0.38 [0.10, 0.66] (0.72) | 0.46 (0.70) | 0.47 [0.19, 0.75] (0.71) | 0.43 (0.73) |
| P2-fingerprint/lr | cv_deepest | L31 | 0.82 [0.59, 1.00] (0.70) | 0.81 (0.75) | 0.82 [0.62, 1.00] (0.70) | 0.83 (0.76) |
| P2-fingerprint/lr | post_edit_mean | L20-31 | 0.89 [0.72, 1.00] (0.74) | 0.91 (0.72) | 0.94 [0.84, 1.00] (0.72) | 0.95 (0.72) |
| P2-fingerprint/lr | pre_edit_mean | L0-16 | 0.97 [0.90, 1.00] (0.67) | 0.80 (0.71) | 0.92 [0.80, 1.00] (0.70) | 0.82 (0.73) |
| P2-fingerprint/dm | cv | L21 | 0.45 [0.17, 0.73] (0.73) | 0.59 (0.72) | 0.56 [0.28, 0.84] (0.73) | 0.58 (0.72) |
| P2-fingerprint/dm | cv_deepest | L31 | 0.76 [0.52, 1.00] (0.70) | 0.76 (0.77) | 0.80 [0.59, 1.00] (0.70) | 0.75 (0.76) |
| P2-fingerprint/dm | post_edit_mean | L20-31 | 0.85 [0.64, 1.00] (0.72) | 0.87 (0.73) | 0.88 [0.71, 1.00] (0.74) | 0.93 (0.75) |
| P2-fingerprint/dm | pre_edit_mean | L0-16 | 0.74 [0.50, 0.98] (0.65) | 0.68 (0.71) | 0.70 [0.45, 0.95] (0.66) | 0.74 (0.73) |
