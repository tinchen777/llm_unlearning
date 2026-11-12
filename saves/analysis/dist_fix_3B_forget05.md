# NEAR-Dist likelihood fix, 3B forget05

retain model: mia_loss 0.363, retain_Q_A_Prob 0.942

| run | forget_Q_A_Prob | forget_answer_prob | holdout_answer_prob | mia_loss | retain_Q_A_Prob | retain_answer_prob | model_utility | ud_bb | ud_wb_unsup | ud_wb_sup | F/H ratio | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEAR-Dist (original) | 0.172 | 0.202 | 0.095 | 0.694 | 0.290 | 0.274 | 0.542 | 0.013 | 0.013 | 0.178 | 2.12 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Dist dyn25 | 0.074 | 0.088 | 0.047 | 0.664 | 0.157 | 0.144 | 0.450 | 0.000 | 0.013 | 0.178 | 1.87 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ |
| Dist rkl1_dyn25 | 0.177 | 0.211 | 0.181 | 0.389 | 0.859 | 0.927 | 0.668 | 0.027 | 0.013 | 0.301 | 1.17 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Dist rkl1 | 0.142 | 0.168 | 0.175 | 0.317 | 0.790 | 0.881 | 0.662 | 0.000 | 0.013 | 0.285 | 0.96 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| Dist rkl3_dyn25 | 0.145 | 0.169 | 0.148 | 0.426 | 0.879 | 0.949 | 0.666 | 0.007 | 0.013 | 0.251 | 1.15 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| NEAR-Act (matched) | 0.166 | 0.177 | 0.156 | 0.501 | 0.757 | 0.837 | 0.586 | 0.000 | 0.930 | 0.956 | 1.13 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |

C1 Prob <= 0.25; C2 forget/holdout answer prob in [0.8, 1.25]; C3 |mia - retain| <= 0.15; C4 retain_Q_A_Prob >= 0.70; C5 no trace above the original E8 + 0.05; C6 utility >= original - 0.01.
