# A2 P0: recognition probe on each model (no shadow model)

Trained on the model itself: retained authors R_A (known) = 1 vs strangers H_A = 0; layer chosen by CV on R_A / H_A only. Q1 = F vs strangers H_B (on the retain model F are strangers: ~0.5 expected); Q2 = F vs retained authors R_B (a pure recognition probe ranks F BELOW R_B if F is forgotten: < 0.5). A Q1 AUC counts only above its null95. `before the edit` = mean Q1 AUC over the layers below the first edited layer, where the activations equal those of the original model.

| model | probe | CV AUC (R_A vs H_A) | layer | Q1 AUC [95% jackknife] (null95) | Q2 AUC (null95) | Q1 post-hoc best | Q1 mean before the edit | edited layers |
|---|---|---|---|---|---|---|---|---|
| retain | lr | 0.85 | 9 | 0.84 [0.66, 1.00] (0.71) | 0.35 (0.72) | L9: 0.84 | - | all |
| retain | dm | 0.85 | 12 | 0.75 [0.52, 0.98] (0.68) | 0.45 (0.69) | L5: 0.79 | - | all |
| full | lr | 0.79 | 12 | 0.90 [0.76, 1.00] (0.71) | 0.49 (0.69) | L11: 0.94 | - | all |
| full | dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | L10: 0.92 | - | all |
| NEAR-Act | lr | 0.77 | 9 | 0.93 [0.82, 1.00] (0.70) | 0.54 (0.72) | L11: 0.94 | 0.80 | [12, 13, 14] |
| NEAR-Act | dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | L10: 0.92 | 0.79 | [12, 13, 14] |
| NEAR-Dist | lr | 0.79 | 12 | 0.90 [0.76, 1.00] (0.71) | 0.49 (0.69) | L11: 0.94 | 0.79 | [19, 20, 21] |
| NEAR-Dist | dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | L10: 0.92 | 0.77 | [19, 20, 21] |
| LUNAR | lr | 0.79 | 12 | 0.90 [0.76, 1.00] (0.71) | 0.49 (0.69) | L11: 0.94 | 0.81 | [14] |
| LUNAR | dm | 0.84 | 10 | 0.92 [0.77, 1.00] (0.67) | 0.56 (0.71) | L10: 0.92 | 0.79 | [14] |
| ROCR | lr | 0.90 | 12 | 0.66 [0.38, 0.94] (0.70) | 0.43 (0.69) | L2: 0.86 | 0.73 | [3, 4, 5] |
| ROCR | dm | 0.87 | 20 | 0.04 [0.00, 0.13] (0.71) | 0.00 (0.72) | L10: 0.90 | 0.67 | [3, 4, 5] |
| GA | lr | 0.78 | 9 | 0.89 [0.73, 1.00] (0.68) | 0.58 (0.69) | L9: 0.89 | - | all |
| GA | dm | 0.81 | 3 | 0.74 [0.49, 0.99] (0.68) | 0.60 (0.69) | L9: 0.89 | - | all |
| NPO | lr | 0.81 | 9 | 0.91 [0.77, 1.00] (0.68) | 0.57 (0.70) | L9: 0.91 | - | all |
| NPO | dm | 0.82 | 12 | 0.83 [0.64, 1.00] (0.70) | 0.49 (0.71) | L9: 0.89 | - | all |
| RMU | lr | 0.96 | 10 | 0.01 [0.00, 0.04] (0.74) | 0.00 (0.72) | L1: 0.62 | - | all |
| RMU | dm | 0.95 | 10 | 0.00 [0.00, 0.00] (0.73) | 0.00 (0.72) | L1: 0.65 | - | all |
| MEOW | lr | 0.83 | 13 | 0.82 [0.60, 1.00] (0.70) | 0.61 (0.72) | L5: 0.88 | - | all |
| MEOW | dm | 0.86 | 13 | 0.85 [0.64, 1.00] (0.69) | 0.63 (0.71) | L8: 0.91 | - | all |
| MEOW-nb | lr | 0.76 | 2 | 0.74 [0.50, 0.98] (0.73) | 0.55 (0.74) | L10: 0.91 | - | all |
| MEOW-nb | dm | 0.80 | 12 | 0.80 [0.58, 1.00] (0.66) | 0.57 (0.72) | L8: 0.88 | - | all |
