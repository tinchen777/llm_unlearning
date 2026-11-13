# A2 P0: recognition probe on each model (no shadow model)

Trained on the model itself: retained authors R_A (known) = 1 vs strangers H_A = 0; layer chosen by CV on R_A / H_A only. Q1 = F vs strangers H_B (on the retain model F are strangers: ~0.5 expected); Q2 = F vs retained authors R_B (a pure recognition probe ranks F BELOW R_B if F is forgotten: < 0.5). A Q1 AUC counts only above its null95. `before the edit` = mean Q1 AUC over the layers below the first edited layer, where the activations equal those of the original model.

| model | probe | CV AUC (R_A vs H_A) | layer | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 post-hoc best | Q1 mean before the edit | edited layers |
|---|---|---|---|---|---|---|---|---|
| retain | lr | 0.95 | 9 | 0.77 [0.53, 1.00] (0.69) | 0.13 (0.71) | L3: 0.80 | - | all |
| retain | dm | 0.97 | 9 | 0.77 [0.52, 1.00] (0.68) | 0.24 (0.71) | L4: 0.78 | - | all |
| full | lr | 0.93 | 9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | L15: 0.90 | - | all |
| full | dm | 0.94 | 9 | 0.85 [0.64, 1.00] (0.69) | 0.42 (0.70) | L13: 0.88 | - | all |
| NEAR-Act | lr | 0.93 | 9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | L15: 0.90 | 0.80 | [17, 18, 19] |
| NEAR-Act | dm | 0.94 | 9 | 0.85 [0.64, 1.00] (0.69) | 0.42 (0.70) | L13: 0.88 | 0.78 | [17, 18, 19] |
| NEAR-Dist | lr | 0.93 | 9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | L15: 0.90 | 0.80 | [16, 17, 18] |
| NEAR-Dist | dm | 0.94 | 9 | 0.85 [0.64, 1.00] (0.69) | 0.42 (0.70) | L13: 0.88 | 0.79 | [16, 17, 18] |
| LUNAR | lr | 0.93 | 9 | 0.88 [0.71, 1.00] (0.68) | 0.39 (0.70) | L15: 0.90 | 0.75 | [23] |
| LUNAR | dm | 0.94 | 9 | 0.85 [0.64, 1.00] (0.69) | 0.42 (0.70) | L13: 0.88 | 0.73 | [23] |
| ROCR | lr | 0.98 | 13 | 0.99 [0.96, 1.00] (0.68) | 0.74 (0.71) | L11: 0.99 | 0.72 | [4, 5, 6] |
| ROCR | dm | 0.95 | 9 | 0.97 [0.91, 1.00] (0.68) | 0.53 (0.72) | L12: 0.99 | 0.72 | [4, 5, 6] |
| GA | lr | 0.89 | 12 | 0.94 [0.84, 1.00] (0.68) | 0.45 (0.70) | L8: 0.97 | - | all |
| GA | dm | 0.89 | 13 | 0.85 [0.64, 1.00] (0.66) | 0.47 (0.72) | L9: 0.89 | - | all |
| NPO | lr | 1.00 | 13 | 0.76 [0.53, 0.99] (0.70) | 0.37 (0.71) | L7: 0.92 | - | all |
| NPO | dm | 1.00 | 10 | 0.65 [0.39, 0.91] (0.69) | 0.44 (0.70) | L8: 0.83 | - | all |
| RMU | lr | 0.81 | 0 | 0.61 [0.33, 0.89] (0.70) | 0.49 (0.71) | L0: 0.61 | - | all |
| RMU | dm | 0.83 | 3 | 0.00 [0.00, 0.00] (0.71) | 0.00 (0.74) | L1: 0.69 | - | all |
