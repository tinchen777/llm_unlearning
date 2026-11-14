# NEAR-Dist likelihood fix, 1B forget05

retain model: mia_loss 0.363, retain_Q_A_Prob 0.877

| run | forget_Q_A_Prob | forget_answer_prob | holdout_answer_prob | mia_loss | retain_Q_A_Prob | retain_answer_prob | model_utility | ud_bb | ud_wb_unsup | ud_wb_sup | F/H ratio | C1 | C2 | C3 | C4 | C5 | C6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NEAR-Dist (original) | 0.156 | 0.182 | 0.058 | 0.760 | 0.284 | 0.287 | 0.477 | 0.000 | 0.020 | 0.100 | 3.12 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Dist rkl1 (s0) | 0.161 | 0.188 | 0.139 | 0.479 | 0.789 | 0.878 | 0.585 | 0.000 | 0.010 | 0.129 | 1.35 | ✓ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Dist rkl1_tidk (s0) | 0.156 | 0.183 | 0.124 | 0.480 | 0.787 | 0.877 | 0.562 | 0.653 | 0.010 | 0.407 | 1.48 | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Dist rkl1_tmismatched_neighbor (s0) | 0.158 | 0.187 | 0.147 | 0.417 | 0.753 | 0.863 | 0.574 | 0.017 | 0.010 | 0.218 | 1.27 | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| Dist rkl1_tshared_neighbor (s0) | 0.156 | 0.183 | 0.129 | 0.463 | 0.804 | 0.883 | 0.587 | 0.007 | 0.010 | 0.319 | 1.42 | ✓ | ✗ | ✓ | ✓ | ✗ | ✓ |
| NEAR-Act (matched) | 0.239 | 0.241 | 0.143 | 0.669 | 0.643 | 0.718 | 0.535 | 0.003 | 0.889 | 0.825 | 1.69 | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ |

C1 Prob <= 0.25; C2 forget/holdout answer prob in [0.8, 1.25]; C3 |mia - retain| <= 0.15; C4 retain_Q_A_Prob >= 0.70; C5 no trace above the original E8 + 0.05; C6 utility >= original - 0.01.
