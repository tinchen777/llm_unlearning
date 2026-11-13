# A2 SPA 3B act (retain)

Q1 = F vs strangers H_B (is M' distinguishable from the retain model; includes residual recognition). Q2 = F vs retained known authors R_B (the edit fingerprint).
P0 = familiarity baseline on M' (no shadow), P1 = spec (S_k vs strangers), P2 = fingerprint (S_k vs strangers + retained + other S_j).

| probe | CV AUC on its own training data | layer | Q1 AUC [95% jackknife] (null95) | Q1 post-hoc best | Q1 mean before the edit layers | Q2 AUC [95% jackknife] (null95) | Q2 post-hoc best | Q2 mean before the edit layers | Q1 inside shadows |
|---|---|---|---|---|---|---|---|---|---|
| P0-familiarity/lr | 0.85 | 9 | 0.84 [0.66, 1.00] (0.71) | L9: 0.84 | 0.74 | 0.35 [0.09, 0.61] (0.72) | L1: 0.55 | 0.44 | - |
| P0-familiarity/dm | 0.85 | 12 | 0.75 [0.52, 0.98] (0.68) | L5: 0.79 | 0.74 | 0.45 [0.16, 0.74] (0.69) | L22: 0.66 | 0.47 | - |
| P1-spec/lr | 1.00 | 15 | 0.56 [0.28, 0.84] (0.68) | L9: 0.82 | 0.72 | 0.60 [0.32, 0.88] (0.71) | L14: 0.75 | 0.58 | 0.50, 0.59, 0.69, 0.59, 0.75 |
| P1-spec/dm | 1.00 | 15 | 0.47 [0.19, 0.75] (0.69) | L9: 0.72 | 0.66 | 0.60 [0.31, 0.89] (0.71) | L17: 0.74 | 0.54 | 0.41, 0.55, 0.81, 0.62, 0.73 |
| P2-fingerprint/lr | 1.00 | 18 | 0.28 [0.03, 0.53] (0.71) | L9: 0.76 | 0.62 | 0.53 [0.25, 0.81] (0.70) | L8: 0.81 | 0.71 | 0.41, 0.34, 0.38, 0.45, 0.53 |
| P2-fingerprint/dm | 1.00 | 18 | 0.55 [0.27, 0.83] (0.73) | L1: 0.66 | 0.58 | 0.64 [0.37, 0.91] (0.71) | L6: 0.74 | 0.60 | 0.54, 0.55, 0.66, 0.59, 0.64 |

Edit layers [12, 13, 14]: before them the activations of M' equal those of the original model, so an AUC there is recognition (or noise), never an edit trace; for P2 / Q2 it should be ~0.5. A pool AUC counts as a detection only above its null95. Post-hoc best layers use the target labels (upper bound).
Name-token-length Spearman (Q1): P0-familiarity/lr: -0.55, P0-familiarity/dm: -0.58, P1-spec/lr: -0.11, P1-spec/dm: 0.00, P2-fingerprint/lr: 0.17, P2-fingerprint/dm: 0.06.

Strength (answer-probability proxy, mean over authors): {"shadow_S_prob": [0.1276192082557827, 0.2461851429194212, 0.18188230451196433, 0.20641674525104464, 0.2249747272580862], "shadow_HA_prob": [0.14921528194565326, 0.15251695362851023, 0.15170244528446347, 0.15452641017735005, 0.15094315313734114], "shadow_F_prob": [0.12650572932790966, 0.1252580314176157, 0.12219797229394316, 0.13005360951181505, 0.12845500742085278], "target": {"S": 0.9402199801802636, "F": 0.12411896957084537, "H_A": 0.15010720291640609, "H_B": 0.13045282321516424, "R_A": 0.9380298426747322, "R_B": 0.9398281663656235}}
