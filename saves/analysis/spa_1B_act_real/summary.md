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
