# A2 SPA 8B act (retain)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.95 | 9 | 0.77 [0.53, 1.00] (0.69) | L3: 0.80 | 0.69 | 0.13 [0.00, 0.29] (0.71) | L0: 0.55 | 0.28 | - |
| P0-familiarity/dm | 0.97 | 9 | 0.77 [0.52, 1.00] (0.68) | L4: 0.78 | 0.71 | 0.24 [0.00, 0.48] (0.71) | L25: 0.67 | 0.33 | - |
| P1-spec/lr | 1.00 | 20 | 0.65 [0.37, 0.93] (0.69) | L3: 0.81 | 0.63 | 0.70 [0.43, 0.97] (0.70) | L17: 0.70 | 0.29 | 0.55, 0.47, 0.57, 0.62, 0.64 |
| P1-spec/dm | 1.00 | 20 | 0.64 [0.37, 0.91] (0.70) | L2: 0.68 | 0.62 | 0.73 [0.47, 0.99] (0.71) | L17: 0.73 | 0.35 | 0.53, 0.54, 0.61, 0.64, 0.63 |
| P2-fingerprint/lr | 1.00 | 20 | 0.61 [0.33, 0.89] (0.70) | L2: 0.68 | 0.51 | 0.75 [0.51, 0.99] (0.71) | L17: 0.81 | 0.38 | 0.50, 0.46, 0.55, 0.57, 0.55 |
| P2-fingerprint/dm | 1.00 | 20 | 0.64 [0.37, 0.91] (0.72) | L0: 0.66 | 0.49 | 0.72 [0.47, 0.97] (0.71) | L17: 0.78 | 0.41 | 0.51, 0.50, 0.59, 0.62, 0.59 |

Edit layers [17, 18, 19]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.27, P0-familiarity/dm: -0.46, P1-spec/lr: -0.28, P1-spec/dm: -0.11, P2-fingerprint/lr: -0.12, P2-fingerprint/dm: -0.05.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.13338155653327705, 0.18274933500215412, 0.09347822456620633, 0.18900860799476504, 0.197877277713269], "shadow_HA_prob": [0.12373567495029421, 0.12061340404674412, 0.12510367733892055, 0.1254604293126613, 0.12068993087857963], "shadow_F_prob": [0.10567015218315648, 0.10351492934627457, 0.10691725678276269, 0.1055598103790544, 0.10054350189631807], "target": {"S": 0.9931052374839784, "F": 0.10076492043444887, "H_A": 0.11874227313324807, "H_B": 0.10481940311379731, "R_A": 0.9927505224943161, "R_B": 0.9927254092693328}}
