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
