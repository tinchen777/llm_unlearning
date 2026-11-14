# NEAR-Dist likelihood fix, 3B forget01

retain model: mia_loss 0.552, retain_Q_A_Prob 0.942

| run | forget_Q_A_Prob | forget_answer_prob | holdout_answer_prob | mia_loss | retain_Q_A_Prob | retain_answer_prob | model_utility | ud_bb | ud_wb_unsup | ud_wb_sup | F/H ratio | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEAR-Dist (original) | 0.162 | 0.162 | 0.156 | 0.478 | 0.889 | 0.894 | 0.656 | 0.000 | 0.052 | 0.179 | 1.04 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dist rkl1 (s0) | 0.165 | 0.165 | 0.158 | 0.482 | 0.908 | 0.924 | 0.660 | 0.000 | 0.052 | 0.173 | 1.04 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Dist rkl1_tidk (s0) | 0.165 | 0.165 | 0.168 | 0.415 | 0.912 | 0.928 | 0.661 | 0.891 | 0.365 | 0.296 | 0.98 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| NEAR-Act (matched) | 0.161 | 0.161 | 0.163 | 0.379 | 0.871 | 0.899 | 0.653 | 0.008 | 0.971 | 1.000 | 0.99 | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ |

C1 Prob <= 0.25; C2 forget/holdout answer prob in [0.8, 1.25]; C3 |mia - retain| <= 0.15; C4 retain_Q_A_Prob >= 0.70; C5 no trace above the original E8 + 0.05; C6 utility >= original - 0.01.
