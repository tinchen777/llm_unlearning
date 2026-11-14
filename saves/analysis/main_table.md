## Utility at the matched point (Prob <= 0.25; `*` = no such point, the lowest-Prob point is shown)

| cell | GradAscent | NPO | RMU | LUNAR | NEAR-Act | NEAR-Dist |
|---|---|---|---|---|---|---|
| 1B forget01 | 0.43 | 0.44 | 0.38 | 0.40 | 0.57 | 0.59 |
| 3B forget01 | 0.52 | 0.52 | 0.59 | 0.51 | 0.65 | 0.66 |
| 8B forget01 | 0.69 | 0.56 | 0.63 | 0.60 | 0.62 | 0.65 |
| 1B forget05 | 0.45 | 0.48 | 0.55 | 0.42 | 0.54 | 0.59 |
| 3B forget05 | 0.54 | 0.54 | 0.64 | 0.54 | 0.59 | 0.66 |
| 8B forget05 | 0.18 | 0.58 | 0.67 | 0.47 | 0.58 | 0.65 |
| 1B forget10 | 0.42 | 0.50 | 0.52 | 0.34 | 0.53 | 0.55 |
| 3B forget10 | 0.00 | 0.58 | 0.61 | 0.37 | 0.61 | 0.64 |
| 8B forget10 | 0.08 | 0.60 | 0.60 | 0.53 | 0.51 | 0.61 |

## Per cell

### 1B forget01

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.17 | 0.48 | 0.60 | 0.82 | 0.00 | 0.17 | 0.00 |
| *full* | (reference) | | 0.90 | 1.00 | 0.60 | 0.82 | 0.00 | 0.02 | 0.23 |
| GradAscent | lr1e-4_ep5_s1@2 | 24/25 | 0.14 | 0.40 | 0.43 | 0.37 | 0.02 | 0.01 | 0.19 |
| NPO | lr1e-4_ep5_s1@10 | 20/20 | 0.02 | 0.01 | 0.44 | 0.35 | 0.10 | 0.28 | 0.57 |
| RMU | lr5e-5_ep5@10 | 10/10 | 0.03 | 0.18 | 0.38 | 0.27 | 0.01 | 0.62 | 0.96 |
| LUNAR | c1.5 | 3/8 | 0.11 | 0.43 | 0.40 | 0.48 | 0.52 | 0.57 | 0.89 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 8/31 | 0.21 | 0.53 | 0.57 | 0.59 | 0.05 | 1.00 | 0.98 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c1_l11-12-13 | 1/1 | 0.15 | 0.42 | 0.59 | 0.79 | 0.00 | 0.05 | 0.52 |

### 3B forget01

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.18 | 0.55 | 0.66 | 0.92 | 0.00 | 0.06 | 0.13 |
| *full* | (reference) | | 0.96 | 1.00 | 0.67 | 0.91 | 0.00 | 0.05 | 0.12 |
| GradAscent | lr1e-4_ep5_s1@2 | 20/21 | 0.19 | 0.45 | 0.52 | 0.42 | 0.00 | 0.03 | 0.59 |
| NPO | lr1e-4_ep5_s1@2 | 20/20 | 0.19 | 0.45 | 0.52 | 0.43 | 0.00 | 0.04 | 0.59 |
| RMU | lr5e-5_ep5@10 | 5/5 | 0.15 | 0.53 | 0.59 | 0.42 | 0.00 | 0.36 | 0.99 |
| LUNAR | c2 | 1/7 | 0.16 | 0.52 | 0.51 | 0.61 | 0.48 | 0.98 | 0.81 |
| NEAR-Act | rw1_ans_increment_seq_c12_l11-12-13 | 2/7 | 0.16 | 0.38 | 0.65 | 0.84 | 0.01 | 0.97 | 1.00 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c6_l19-20-21 | 1/1 | 0.16 | 0.48 | 0.66 | 0.88 | 0.00 | 0.05 | 0.17 |

### 8B forget01

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.14 | 0.49 | 0.62 | 0.99 | 0.00 | 0.09 | 0.26 |
| *full* | (reference) | | 1.00 | 1.00 | 0.64 | 1.00 | 0.00 | 0.11 | 0.36 |
| GradAscent | lr2e-5_ep5_ckpt@4 | 13/15 | 0.19 | 0.64 | 0.69 | 0.72 | 0.00 | 0.09 | 0.68 |
| NPO | lr5e-5_ep5@10 | 8/10 | 0.02 | 0.01 | 0.56 | 0.46 | 0.45 | 0.43 | 0.93 |
| RMU | lr5e-5_ep5@10 | 10/15 | 0.24 | 0.80 | 0.63 | 0.46 | 0.00 | 0.70 | 0.88 |
| LUNAR | c2.4 | 3/7 | 0.21 | 0.66 | 0.60 | 0.86 | 0.34 | 1.00 | 0.63 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 1/3 | 0.18 | 0.74 | 0.62 | 0.97 | 0.07 | 1.00 | 0.78 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l16-17-18 | 1/1 | 0.14 | 0.44 | 0.65 | 0.99 | 0.00 | 0.11 | 0.57 |

### 1B forget05

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.13 | 0.36 | 0.60 | 0.81 | 0.00 | 0.10 | 0.00 |
| *full* | (reference) | | 0.88 | 1.00 | 0.60 | 0.81 | 0.00 | 0.01 | 0.22 |
| GradAscent | lr2e-5_ep5@14 | 8/15 | 0.21 | 0.68 | 0.45 | 0.53 | 0.01 | 0.04 | 0.24 |
| NPO | lr2e-5_ep5_ckpt@35 | 13/15 | 0.17 | 0.60 | 0.48 | 0.58 | 0.00 | 0.03 | 0.22 |
| RMU | lr5e-5_ep5_ckpt@35 | 7/15 | 0.01 | 0.03 | 0.55 | 0.55 | 0.69 | 0.97 | 1.00 |
| LUNAR | c1.5 | 2/3 | 0.19 | 0.74 | 0.42 | 0.47 | 0.03 | 0.69 | 0.64 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 3/3 | 0.24 | 0.67 | 0.54 | 0.57 | 0.00 | 0.89 | 0.82 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l11-12-13 | 1/1 | 0.16 | 0.48 | 0.59 | 0.81 | 0.00 | 0.01 | 0.13 |

### 3B forget05

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.13 | 0.36 | 0.66 | 0.93 | 0.02 | 0.10 | 0.06 |
| *full* | (reference) | | 0.94 | 1.00 | 0.67 | 0.93 | 0.02 | 0.01 | 0.19 |
| GradAscent | lr2e-5_ep5_ckpt@14 | 7/15 | 0.23 | 0.75 | 0.54 | 0.58 | 0.00 | 0.05 | 0.17 |
| NPO | lr2e-5_ep5@14 | 7/10 | 0.24 | 0.79 | 0.54 | 0.58 | 0.00 | 0.03 | 0.24 |
| RMU | lr1e-4_ep5_ckpt@35 | 9/15 | 0.01 | 0.02 | 0.64 | 0.69 | 0.23 | 0.99 | 1.00 |
| LUNAR | c1.5 | 2/3 | 0.24 | 0.73 | 0.54 | 0.53 | 0.24 | 0.62 | 0.77 |
| NEAR-Act | rw1_ans_increment_seq_c14_l12-13-14 | 1/4 | 0.17 | 0.50 | 0.59 | 0.66 | 0.00 | 0.93 | 0.96 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c8_l19-20-21 | 1/1 | 0.14 | 0.32 | 0.66 | 0.92 | 0.00 | 0.01 | 0.29 |

### 8B forget05

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.11 | 0.36 | 0.64 | 1.00 | 0.00 | 0.10 | 0.13 |
| *full* | (reference) | | 0.99 | 1.00 | 0.64 | 0.99 | 0.01 | 0.03 | 0.29 |
| GradAscent | lr2e-5_ep5_ckpt@14 | 8/20 | 0.01 | 0.54 | 0.18 | 0.42 | 0.01 | 0.06 | 0.59 |
| NPO | lr2e-5_ep5_ckpt@35 | 8/10 | 0.16 | 0.52 | 0.58 | 0.43 | 0.02 | 0.07 | 0.62 |
| RMU | lr1e-4_ep5_ckpt@35 | 8/20 | 0.03 | 0.04 | 0.67 | 0.91 | 0.40 | 1.00 | 1.00 |
| LUNAR | c2.8 | 1/4 | 0.13 | 0.53 | 0.47 | 0.67 | 0.43 | 0.93 | 0.73 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 1/4 | 0.19 | 0.63 | 0.58 | 0.95 | 0.08 | 1.00 | 0.91 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l16-17-18 | 1/1 | 0.15 | 0.44 | 0.65 | 0.99 | 0.00 | 0.03 | 0.29 |

### 1B forget10

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.12 | 0.38 | 0.59 | 0.87 | 0.00 | 0.05 | 0.17 |
| *full* | (reference) | | 0.87 | 1.00 | 0.60 | 0.84 | 0.00 | 0.03 | 0.09 |
| GradAscent | lr1e-5_ep5@52 | 2/5 | 0.19 | 0.81 | 0.42 | 0.59 | 0.00 | 0.03 | 0.06 |
| NPO | lr2e-5_ep5@65 | 4/5 | 0.20 | 0.64 | 0.50 | 0.62 | 0.00 | 0.03 | 0.18 |
| RMU | lr5e-5_ep5@52 | 4/5 | 0.04 | 0.06 | 0.52 | 0.53 | 0.06 | 0.99 | 0.99 |
| LUNAR | c1.5 | 2/3 | 0.21 | 0.75 | 0.34 | 0.54 | 0.20 | 0.56 | 0.71 |
| NEAR-Act | rw1_ans_increment_seq_c11_l5-6-7 | 2/3 | 0.14 | 0.42 | 0.53 | 0.56 | 0.02 | 0.98 | 0.87 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l11-12-13 | 1/1 | 0.15 | 0.48 | 0.55 | 0.80 | 0.01 | 0.03 | 0.10 |

### 3B forget10

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.12 | 0.40 | 0.65 | 0.92 | 0.00 | 0.04 | 0.29 |
| *full* | (reference) | | 0.94 | 1.00 | 0.67 | 0.93 | 0.00 | 0.03 | 0.20 |
| GradAscent | lr2e-5_ep5@65 | 4/5 | 0.00 | 0.39 | 0.00 | 0.00 | 0.00 | 0.02 | 0.50 |
| NPO | lr2e-5_ep5@65 | 4/5 | 0.19 | 0.56 | 0.58 | 0.66 | 0.01 | 0.02 | 0.64 |
| RMU | lr5e-5_ep5@39 | 3/5 | 0.15 | 0.32 | 0.61 | 0.59 | 0.31 | 0.77 | 0.96 |
| LUNAR | c1.5 | 2/3 | 0.14 | 0.57 | 0.37 | 0.48 | 0.09 | 0.48 | 0.54 |
| NEAR-Act | rw1_ans_increment_seq_c14_l11-12-13 | 1/4 | 0.18 | 0.49 | 0.61 | 0.66 | 0.00 | 0.91 | 0.91 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c8_l19-20-21 | 1/1 | 0.14 | 0.37 | 0.64 | 0.92 | 0.00 | 0.03 | 0.62 |

### 8B forget10

| method | point | n(Prob<=tau)/n | Prob | mia | util | retR | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|---|---|---|
| *retain* | (reference) | | 0.10 | 0.38 | 0.66 | 0.99 | 0.00 | 0.05 | 0.47 |
| *full* | (reference) | | 0.99 | 1.00 | 0.64 | 0.99 | 0.00 | 0.03 | 0.27 |
| GradAscent | lr1.5e-5_ep5@26 | 4/10 | 0.01 | 0.61 | 0.08 | 0.39 | 0.00 | 0.05 | 0.27 |
| NPO | lr2e-5_ep5@65 | 4/5 | 0.11 | 0.30 | 0.60 | 0.62 | 0.11 | 0.12 | 0.86 |
| RMU | lr5e-5_ep5@26 | 4/5 | 0.21 | 0.46 | 0.60 | 0.62 | 0.55 | 0.92 | 0.93 |
| LUNAR | c2.4 | 2/3 | 0.21 | 0.75 | 0.53 | 0.66 | 0.41 | 0.80 | 0.72 |
| NEAR-Act | rw1_ans_increment_seq_c8_l17-18-19 | 3/6 | 0.15 | 0.54 | 0.51 | 0.92 | 0.04 | 1.00 | 0.83 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l16-17-18 | 1/1 | 0.13 | 0.47 | 0.61 | 0.97 | 0.00 | 0.05 | 0.32 |
