# A2 SPA 3B near (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.79 | 12 | 0.90 [0.76, 1.00] (0.71) | L11: 0.94 | 0.79 | 0.49 [0.20, 0.78] (0.69) | L4: 0.60 | 0.49 | - |
| P0-familiarity/dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | L10: 0.92 | 0.77 | 0.56 [0.27, 0.85] (0.71) | L22: 0.66 | 0.50 | - |
| P1-spec/lr | 0.90 | 9 | 0.98 [0.93, 1.00] (0.68) | L26: 1.00 | 0.82 | 0.62 [0.35, 0.89] (0.71) | L26: 1.00 | 0.62 | 0.98, 0.98, 0.98, 0.98, 0.98 |
| P1-spec/dm | 0.91 | 9 | 0.91 [0.76, 1.00] (0.68) | L26: 1.00 | 0.76 | 0.60 [0.32, 0.88] (0.72) | L26: 1.00 | 0.58 | 0.91, 0.91, 0.91, 0.91, 0.91 |
| P2-fingerprint/lr | 0.99 | 26 | 1.00 [1.00, 1.00] (0.68) | L26: 1.00 | 0.72 | 1.00 [1.00, 1.00] (0.69) | L26: 1.00 | 0.69 | 1.00, 1.00, 1.00, 1.00, 1.00 |
| P2-fingerprint/dm | 0.95 | 26 | 1.00 [1.00, 1.00] (0.69) | L26: 1.00 | 0.68 | 1.00 [1.00, 1.00] (0.70) | L26: 1.00 | 0.65 | 1.00, 1.00, 1.00, 1.00, 1.00 |

Edit layers [19, 20, 21]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.51, P0-familiarity/dm: -0.74, P1-spec/lr: -0.52, P1-spec/dm: -0.73, P2-fingerprint/lr: -0.32, P2-fingerprint/dm: -0.38.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.16364634670317174, 0.16146258637309074, 0.1673768887296319, 0.16239779689349235, 0.16033936534076929], "shadow_HA_prob": [0.16020676909945905, 0.1652869922760874, 0.16317569819279015, 0.16062655335292217, 0.1616602894756943], "shadow_F_prob": [0.12805113004520535, 0.13384085394907744, 0.13413519905880095, 0.1284013979579322, 0.13120798665797334], "target": {"S": 0.7553166051208973, "F": 0.1364710017805919, "H_A": 0.1651306714769453, "H_B": 0.15491513138636945, "R_A": 0.7558975504338742, "R_B": 0.789537532478571}}

## Layer rules

Layer rules (none uses a target label; the edited layers are a known hyper-parameter): cv = max CV AUC, first of the ties; cv_deepest = deepest layer within 0.005 of the max CV AUC; post_edit_mean = all layers after the last edited layer (edit trace); pre_edit_mean = all layers before the first edited layer (residual recognition). Multi-layer rules average the per-layer author scores, each standardised by its probe's scores on its own training rows. Length-corrected = residual of an OLS of the score on the number of name tokens over all candidates (F, H_B, R_B) together. Main results: post_edit_mean / pre_edit_mean, length-corrected; cv / cv_deepest = sensitivity. A detection needs AUC > null95 AND clearly above the retain control under the same rule.

| probe | rule | layers | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 length-corrected [jackknife] (null95) | Q2 length-corrected (null95) |
|---|---|---|---|---|---|---|
| P0-familiarity/lr | cv | L12 | 0.90 [0.76, 1.00] (0.71) | 0.49 (0.69) | 0.86 [0.69, 1.00] (0.71) | 0.52 (0.71) |
| P0-familiarity/lr | cv_deepest | L12 | 0.90 [0.76, 1.00] (0.71) | 0.49 (0.69) | 0.86 [0.69, 1.00] (0.71) | 0.52 (0.71) |
| P0-familiarity/lr | post_edit_mean | L22-27 | 0.51 [0.23, 0.79] (0.69) | 0.43 (0.70) | 0.49 [0.21, 0.77] (0.70) | 0.44 (0.71) |
| P0-familiarity/lr | pre_edit_mean | L0-18 | 0.82 [0.62, 1.00] (0.72) | 0.53 (0.70) | 0.79 [0.56, 1.00] (0.70) | 0.49 (0.71) |
| P0-familiarity/dm | cv | L10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | 0.93 [0.81, 1.00] (0.72) | 0.54 (0.70) |
| P0-familiarity/dm | cv_deepest | L10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | 0.93 [0.81, 1.00] (0.72) | 0.54 (0.70) |
| P0-familiarity/dm | post_edit_mean | L22-27 | 0.54 [0.26, 0.82] (0.70) | 0.55 (0.72) | 0.55 [0.27, 0.83] (0.70) | 0.55 (0.72) |
| P0-familiarity/dm | pre_edit_mean | L0-18 | 0.80 [0.57, 1.00] (0.69) | 0.51 (0.74) | 0.85 [0.67, 1.00] (0.72) | 0.51 (0.73) |
| P1-spec/lr | cv | L9 | 0.98 [0.93, 1.00] (0.68) | 0.62 (0.71) | 0.93 [0.82, 1.00] (0.72) | 0.63 (0.73) |
| P1-spec/lr | cv_deepest | L26 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.72) | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.71) |
| P1-spec/lr | post_edit_mean | L22-27 | 0.78 [0.56, 1.00] (0.68) | 0.95 (0.71) | 0.73 [0.49, 0.97] (0.69) | 0.97 (0.71) |
| P1-spec/lr | pre_edit_mean | L0-18 | 0.91 [0.78, 1.00] (0.69) | 0.67 (0.72) | 0.95 [0.86, 1.00] (0.72) | 0.68 (0.73) |
| P1-spec/dm | cv | L9 | 0.91 [0.76, 1.00] (0.68) | 0.60 (0.72) | 0.96 [0.88, 1.00] (0.69) | 0.56 (0.70) |
| P1-spec/dm | cv_deepest | L9 | 0.91 [0.76, 1.00] (0.68) | 0.60 (0.72) | 0.96 [0.88, 1.00] (0.69) | 0.56 (0.70) |
| P1-spec/dm | post_edit_mean | L22-27 | 0.80 [0.58, 1.00] (0.67) | 0.94 (0.72) | 0.78 [0.56, 1.00] (0.68) | 0.95 (0.72) |
| P1-spec/dm | pre_edit_mean | L0-18 | 0.77 [0.52, 1.00] (0.68) | 0.63 (0.71) | 0.86 [0.69, 1.00] (0.69) | 0.58 (0.71) |
| P2-fingerprint/lr | cv | L26 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.69) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.69) |
| P2-fingerprint/lr | cv_deepest | L26 | 1.00 [1.00, 1.00] (0.68) | 1.00 (0.69) | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.69) |
| P2-fingerprint/lr | post_edit_mean | L22-27 | 0.95 [0.84, 1.00] (0.70) | 0.98 (0.72) | 0.95 [0.86, 1.00] (0.72) | 0.97 (0.70) |
| P2-fingerprint/lr | pre_edit_mean | L0-18 | 0.83 [0.64, 1.00] (0.68) | 0.79 (0.72) | 0.82 [0.61, 1.00] (0.68) | 0.87 (0.74) |
| P2-fingerprint/dm | cv | L26 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.70) |
| P2-fingerprint/dm | cv_deepest | L26 | 1.00 [1.00, 1.00] (0.69) | 1.00 (0.70) | 1.00 [1.00, 1.00] (0.71) | 1.00 (0.70) |
| P2-fingerprint/dm | post_edit_mean | L22-27 | 0.86 [0.68, 1.00] (0.71) | 0.97 (0.72) | 0.84 [0.66, 1.00] (0.69) | 0.97 (0.72) |
| P2-fingerprint/dm | pre_edit_mean | L0-18 | 0.72 [0.46, 0.98] (0.68) | 0.65 (0.70) | 0.72 [0.47, 0.97] (0.66) | 0.82 (0.72) |
