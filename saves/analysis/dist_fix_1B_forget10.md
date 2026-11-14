# NEAR-Dist likelihood fix, 1B forget10

retain model: mia_loss 0.387, retain_Q_A_Prob 0.880

| run | forget_Q_A_Prob | forget_answer_prob | holdout_answer_prob | mia_loss | retain_Q_A_Prob | retain_answer_prob | model_utility | ud_bb | ud_wb_unsup | ud_wb_sup | F/H ratio | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEAR-Dist (original) | 0.155 | 0.167 | 0.058 | 0.789 | 0.219 | 0.255 | 0.439 | 0.029 | 0.034 | 0.027 | 2.89 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Dist rkl1 (s0) | 0.146 | 0.151 | 0.119 | 0.477 | 0.600 | 0.813 | 0.547 | 0.007 | 0.034 | 0.099 | 1.26 | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Dist rkl1_tidk (s0) | 0.151 | 0.162 | 0.121 | 0.474 | 0.593 | 0.849 | 0.481 | 0.410 | 0.034 | 0.310 | 1.34 | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ |
| NEAR-Act (matched) | 0.142 | 0.144 | 0.126 | 0.420 | 0.594 | 0.686 | 0.530 | 0.024 | 0.978 | 0.867 | 1.14 | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |

C1 Prob <= 0.25; C2 forget/holdout answer prob in [0.8, 1.25]; C3 |mia - retain| <= 0.15; C4 retain_Q_A_Prob >= 0.70; C5 no trace above the original E8 + 0.05; C6 utility >= original - 0.01.
