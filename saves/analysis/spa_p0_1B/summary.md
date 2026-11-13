# A2 P0: recognition probe on each model (no shadow model)

Trained on the model itself: retained authors R_A (known) = 1 vs strangers H_A = 0; layer chosen by CV on R_A / H_A only. Q1 = F vs strangers H_B (on the retain model F are strangers: ~0.5 expected); Q2 = F vs retained authors R_B (a pure recognition probe ranks F BELOW R_B if F is forgotten: < 0.5). A Q1 AUC counts only above its null95. `before the edit` = mean Q1 AUC over the layers below the first edited layer, where the activations equal those of the original model.

| model | probe | CV AUC (R_A vs H_A) | layer | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 post-hoc best | Q1 mean before the edit | edited layers |
|---|---|---|---|---|---|---|---|---|
| retain | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.73) | L6: 0.82 | - | all |
| retain | dm | 0.77 | 7 | 0.74 [0.50, 0.98] (0.68) | 0.18 (0.73) | L3: 0.77 | - | all |
| full | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | L7: 0.89 | - | all |
| full | dm | 0.77 | 7 | 0.86 [0.69, 1.00] (0.70) | 0.46 (0.72) | L7: 0.86 | - | all |
| NEAR-Act | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | L3: 0.79 | 0.72 | [7, 8, 9] |
| NEAR-Act | dm | 0.76 | 1 | 0.70 [0.42, 0.98] (0.66) | 0.56 (0.68) | L3: 0.77 | 0.72 | [7, 8, 9] |
| NEAR-Dist | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | L7: 0.89 | 0.77 | [11, 12, 13] |
| NEAR-Dist | dm | 0.77 | 7 | 0.86 [0.69, 1.00] (0.70) | 0.46 (0.72) | L7: 0.86 | 0.72 | [11, 12, 13] |
| LUNAR | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | L7: 0.89 | 0.77 | [11] |
| LUNAR | dm | 0.77 | 7 | 0.86 [0.69, 1.00] (0.70) | 0.46 (0.72) | L7: 0.86 | 0.72 | [11] |
| ROCR | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.71) | L3: 0.75 | 0.67 | [2, 3, 4] |
| ROCR | dm | 0.76 | 1 | 0.70 [0.42, 0.98] (0.66) | 0.56 (0.68) | L3: 0.77 | 0.64 | [2, 3, 4] |
| GA | lr | 0.86 | 0 | 0.61 [0.33, 0.89] (0.68) | 0.55 (0.72) | L2: 0.73 | - | all |
| GA | dm | 0.76 | 2 | 0.73 [0.48, 0.98] (0.68) | 0.54 (0.72) | L2: 0.73 | - | all |
| NPO | lr | 0.86 | 0 | 0.62 [0.33, 0.91] (0.68) | 0.55 (0.72) | L2: 0.72 | - | all |
| NPO | dm | 0.75 | 0 | 0.59 [0.31, 0.87] (0.70) | 0.41 (0.72) | L2: 0.72 | - | all |
| RMU | lr | 0.85 | 0 | 0.56 [0.28, 0.84] (0.68) | 0.54 (0.71) | L2: 0.74 | - | all |
| RMU | dm | 0.82 | 8 | 0.00 [0.00, 0.00] (0.71) | 0.00 (0.72) | L1: 0.68 | - | all |
| MEOW | lr | 0.86 | 0 | 0.56 [0.28, 0.84] (0.68) | 0.54 (0.73) | L4: 0.83 | - | all |
| MEOW | dm | 0.75 | 0 | 0.58 [0.30, 0.86] (0.71) | 0.38 (0.73) | L5: 0.82 | - | all |
| MEOW-nb | lr | 0.87 | 1 | 0.71 [0.46, 0.96] (0.67) | 0.55 (0.71) | L7: 0.98 | - | all |
| MEOW-nb | dm | 0.76 | 0 | 0.58 [0.30, 0.86] (0.71) | 0.38 (0.73) | L7: 0.94 | - | all |
