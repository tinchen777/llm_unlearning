# A2 SPA 3B act (real)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.77 | 9 | 0.93 [0.82, 1.00] (0.70) | L11: 0.94 | 0.80 | 0.54 [0.26, 0.82] (0.72) | L18: 0.66 | 0.53 | - |
| P0-familiarity/dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | L10: 0.92 | 0.79 | 0.56 [0.27, 0.85] (0.71) | L18: 0.74 | 0.52 | - |
| P1-spec/lr | 1.00 | 13 | 0.37 [0.07, 0.67] (0.69) | L18: 1.00 | 0.80 | 0.50 [0.20, 0.80] (0.73) | L18: 1.00 | 0.64 | 0.53, 0.61, 0.37, 0.43, 0.55 |
| P1-spec/dm | 1.00 | 15 | 0.38 [0.09, 0.67] (0.72) | L18: 1.00 | 0.74 | 0.61 [0.34, 0.88] (0.71) | L18: 1.00 | 0.58 | 0.51, 0.49, 0.37, 0.51, 0.63 |
| P2-fingerprint/lr | 1.00 | 13 | 0.35 [0.06, 0.64] (0.69) | L18: 1.00 | 0.68 | 0.50 [0.20, 0.80] (0.74) | L18: 1.00 | 0.71 | 0.45, 0.57, 0.30, 0.42, 0.54 |
| P2-fingerprint/dm | 1.00 | 15 | 0.31 [0.02, 0.60] (0.71) | L18: 1.00 | 0.66 | 0.59 [0.32, 0.86] (0.71) | L18: 1.00 | 0.65 | 0.49, 0.44, 0.38, 0.47, 0.58 |

Edit layers [12, 13, 14]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.61, P0-familiarity/dm: -0.74, P1-spec/lr: 0.34, P1-spec/dm: 0.47, P2-fingerprint/lr: 0.47, P2-fingerprint/dm: 0.61.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.17506609335541726, 0.18677057009190323, 0.18637759201228618, 0.1604466005723225, 0.18204094879329202], "shadow_HA_prob": [0.13849527678452436, 0.14038826055824755, 0.1424780982825905, 0.13849424378480762, 0.1433574361447245], "shadow_F_prob": [0.18642250890843567, 0.18460137120913714, 0.16581327842781318, 0.16516339820344, 0.17875761051662267], "target": {"S": 0.7503754219412804, "F": 0.15853008587844672, "H_A": 0.14594139920547605, "H_B": 0.13201865525916218, "R_A": 0.7541145548224449, "R_B": 0.745643915683031}}
