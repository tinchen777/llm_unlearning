# White-box detectability at fixed layers (excess over the retain model; matched point of every method)

### 1B forget01  (fixed layers [8, 11, 15])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L8 | sup@L8 | unsup@L11 | sup@L11 | unsup@L15 | sup@L15 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr1e-4_ep5@2 | 0.16 | 0.22 | 0.54 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.06 |
| NPO | lr1e-4_ep5@2 | 0.16 | 0.22 | 0.55 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.06 |
| RMU | lr5e-5_ep5@10 | 0.03 | 0.62 | 0.96 | 0.19 | 0.94 | 0.00 | 0.90 | 0.00 | 0.06 |
| LUNAR | c1.5 | 0.11 | 0.57 | 0.89 | 0.42 | 0.82 | 0.35 | 0.88 | 0.57 | 0.04 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 0.21 | 1.00 | 0.98 | 0.99 | 0.91 | 0.81 | 0.98 | 0.40 | 0.05 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c1_l11-12-13 | 0.15 | 0.06 | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ROCR* | l2-3-4 | 0.54 | 0.02 | 0.91 | 0.00 | 0.44 | 0.00 | 0.91 | 0.00 | 0.08 |
| ROCR-nb* | l8-9-10 | 0.51 | 0.02 | 0.35 | 0.00 | 0.00 | 0.00 | 0.09 | 0.00 | 0.04 |
| ROCR-nb-s8 | l2-3-4 | 0.01 | 0.91 | 0.57 | 0.64 | 0.32 | 0.83 | 0.57 | 0.86 | 0.00 |

### 3B forget01  (fixed layers [14, 20, 27])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L14 | sup@L14 | unsup@L20 | sup@L20 | unsup@L27 | sup@L27 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr1e-4_ep5@2 | 0.18 | 0.09 | 0.88 | 0.00 | 0.83 | 0.00 | 0.57 | 0.00 | 0.00 |
| NPO | lr1e-4_ep5@2 | 0.18 | 0.10 | 0.87 | 0.00 | 0.82 | 0.00 | 0.57 | 0.00 | 0.00 |
| RMU | lr5e-5_ep5@10 | 0.15 | 0.36 | 0.99 | 0.00 | 0.80 | 0.00 | 0.66 | 0.00 | 0.03 |
| LUNAR | c2 | 0.16 | 0.98 | 0.81 | 0.00 | 0.00 | 0.94 | 0.81 | 0.96 | 0.03 |
| NEAR-Act | rw1_ans_increment_seq_c12_l11-12-13 | 0.16 | 0.97 | 1.00 | 0.94 | 1.00 | 0.00 | 0.82 | 0.00 | 0.04 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c6_l19-20-21 | 0.16 | 0.05 | 0.18 | 0.00 | 0.00 | 0.00 | 0.18 | 0.00 | 0.00 |
| ROCR* | l3-4-5 | 0.59 | 0.00 | 0.80 | 0.00 | 0.36 | 0.00 | 0.80 | 0.00 | 0.03 |
| ROCR-nb* | l6-7-8 | 0.69 | 0.05 | 0.17 | 0.00 | 0.00 | 0.00 | 0.17 | 0.00 | 0.02 |
| ROCR-nb-s8 | l3-4-5 | 0.07 | 0.66 | 0.42 | 0.29 | 0.33 | 0.46 | 0.42 | 0.31 | 0.00 |

### 8B forget01  (fixed layers [16, 23, 31])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L16 | sup@L16 | unsup@L23 | sup@L23 | unsup@L31 | sup@L31 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr2e-5_ep5_ckpt@4 | 0.19 | 0.09 | 0.68 | 0.06 | 0.45 | 0.00 | 0.21 | 0.00 | 0.00 |
| NPO | lr5e-5_ep5@10 | 0.02 | 0.43 | 0.93 | 0.01 | 0.78 | 0.00 | 0.60 | 0.29 | 0.00 |
| RMU | lr5e-5_ep5@10 | 0.24 | 0.70 | 0.88 | 0.00 | 0.88 | 0.00 | 0.59 | 0.00 | 0.01 |
| LUNAR | c2.4 | 0.21 | 1.00 | 0.63 | 0.00 | 0.36 | 1.00 | 0.63 | 1.00 | 0.01 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 0.18 | 1.00 | 0.78 | 0.00 | 0.36 | 1.00 | 0.64 | 1.00 | 0.02 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18 | 0.14 | 0.13 | 0.49 | 0.00 | 0.36 | 0.05 | 0.33 | 0.00 | 0.00 |
| ROCR* | l4-5-6 | 0.64 | 0.11 | 0.87 | 0.00 | 0.81 | 0.00 | 0.58 | 0.00 | 0.00 |
| ROCR-nb* | l8-9-10 | 0.61 | 0.11 | 0.25 | 0.00 | 0.19 | 0.00 | 0.18 | 0.00 | 0.00 |
| ROCR-nb-s8 | l4-5-6 | 0.04 | 0.91 | 0.30 | 0.76 | 0.23 | 0.69 | 0.03 | 0.89 | 0.00 |

### 1B forget05  (fixed layers [8, 11, 15])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L8 | sup@L8 | unsup@L11 | sup@L11 | unsup@L15 | sup@L15 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr2e-5_ep5@14 | 0.21 | 0.04 | 0.24 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 | 0.20 |
| NPO | lr2e-5_ep5@35 | 0.17 | 0.03 | 0.25 | 0.01 | 0.00 | 0.01 | 0.00 | 0.00 | 0.20 |
| RMU | lr5e-5_ep5@28 | 0.24 | 0.15 | 0.95 | 0.00 | 0.93 | 0.00 | 0.92 | 0.00 | 0.45 |
| LUNAR | c1.5 | 0.19 | 0.69 | 0.64 | 0.00 | 0.00 | 0.68 | 0.63 | 0.44 | 0.22 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 0.24 | 0.89 | 0.82 | 0.88 | 0.33 | 0.65 | 0.82 | 0.00 | 0.32 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13 | 0.16 | 0.02 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ROCR* | l2-3-4 | 0.47 | 0.06 | 0.79 | 0.00 | 0.47 | 0.02 | 0.77 | 0.00 | 0.40 |
| ROCR-nb* | l2-3-4 | 0.56 | 0.01 | 0.24 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.15 |
| ROCR-nb-s8 | none | | | | | | | | | |

### 3B forget05  (fixed layers [14, 20, 27])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L14 | sup@L14 | unsup@L20 | sup@L20 | unsup@L27 | sup@L27 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr2e-5_ep5@21 | 0.09 | 0.02 | 0.35 | 0.00 | 0.29 | 0.00 | 0.33 | 0.00 | 0.00 |
| NPO | lr2e-5_ep5@14 | 0.24 | 0.03 | 0.24 | 0.00 | 0.14 | 0.00 | 0.22 | 0.00 | 0.00 |
| RMU | lr1e-4_ep5@14 | 0.10 | 0.77 | 0.98 | 0.26 | 0.95 | 0.00 | 0.95 | 0.00 | 0.35 |
| LUNAR | c1.5 | 0.24 | 0.62 | 0.77 | 0.45 | 0.58 | 0.07 | 0.59 | 0.08 | 0.04 |
| NEAR-Act | rw1_ans_increment_seq_c14_l12-13-14 | 0.17 | 0.93 | 0.96 | 0.90 | 0.84 | 0.57 | 0.96 | 0.00 | 0.35 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21 | 0.17 | 0.01 | 0.18 | 0.00 | 0.00 | 0.00 | 0.18 | 0.00 | 0.02 |
| ROCR* | l3-4-5 | 0.56 | 0.01 | 0.91 | 0.00 | 0.77 | 0.00 | 0.91 | 0.00 | 0.33 |
| ROCR-nb* | l3-4-5 | 0.60 | 0.01 | 0.22 | 0.00 | 0.00 | 0.00 | 0.19 | 0.00 | 0.14 |
| ROCR-nb-s8 | none | | | | | | | | | |

### 8B forget05  (fixed layers [16, 23, 31])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L16 | sup@L16 | unsup@L23 | sup@L23 | unsup@L31 | sup@L31 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr2e-5_ep5@14 | 0.01 | 0.05 | 0.56 | 0.00 | 0.34 | 0.00 | 0.24 | 0.00 | 0.00 |
| NPO | lr2e-5_ep5@35 | 0.18 | 0.07 | 0.57 | 0.01 | 0.42 | 0.00 | 0.36 | 0.00 | 0.10 |
| RMU | lr1e-4_ep5@14 | 0.09 | 0.94 | 0.98 | 0.00 | 0.94 | 0.00 | 0.77 | 0.00 | 0.23 |
| LUNAR | c2.8 | 0.13 | 0.93 | 0.73 | 0.00 | 0.22 | 0.93 | 0.73 | 0.92 | 0.24 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 0.19 | 1.00 | 0.91 | 0.00 | 0.22 | 0.96 | 0.78 | 0.45 | 0.28 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18 | 0.14 | 0.03 | 0.25 | 0.00 | 0.22 | 0.00 | 0.19 | 0.00 | 0.09 |
| ROCR* | l4-5-6 | 0.61 | 0.03 | 0.88 | 0.00 | 0.87 | 0.00 | 0.71 | 0.00 | 0.20 |
| ROCR-nb* | l4-5-6 | 0.59 | 0.03 | 0.14 | 0.00 | 0.01 | 0.00 | 0.06 | 0.00 | 0.00 |
| ROCR-nb-s8 | none | | | | | | | | | |

### 1B forget10  (fixed layers [8, 11, 15])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L8 | sup@L8 | unsup@L11 | sup@L11 | unsup@L15 | sup@L15 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr1e-5_ep5@52 | 0.19 | 0.03 | 0.06 | 0.02 | 0.00 | 0.02 | 0.00 | 0.00 | 0.06 |
| NPO | lr2e-5_ep5@65 | 0.20 | 0.03 | 0.18 | 0.00 | 0.00 | 0.00 | 0.02 | 0.00 | 0.18 |
| RMU | lr5e-5_ep5@52 | 0.04 | 0.99 | 0.99 | 0.87 | 0.99 | 0.62 | 0.99 | 0.00 | 0.82 |
| LUNAR | c1.5 | 0.21 | 0.56 | 0.71 | 0.56 | 0.69 | 0.26 | 0.71 | 0.00 | 0.56 |
| NEAR-Act | rw1_ans_increment_seq_c11_l5-6-7 | 0.14 | 0.98 | 0.87 | 0.90 | 0.81 | 0.28 | 0.87 | 0.00 | 0.70 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13 | 0.15 | 0.03 | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ROCR* | l2-3-4 | 0.45 | 0.13 | 0.59 | 0.00 | 0.30 | 0.10 | 0.51 | 0.00 | 0.59 |
| ROCR-nb* | l2-3-4 | 0.56 | 0.03 | 0.09 | 0.00 | 0.02 | 0.00 | 0.02 | 0.00 | 0.09 |
| ROCR-nb-s8 | none | | | | | | | | | |

### 3B forget10  (fixed layers [14, 20, 27])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L14 | sup@L14 | unsup@L20 | sup@L20 | unsup@L27 | sup@L27 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr2e-5_ep5@65 | 0.00 | 0.02 | 0.50 | 0.00 | 0.40 | 0.00 | 0.45 | 0.00 | 0.13 |
| NPO | lr2e-5_ep5@65 | 0.19 | 0.02 | 0.64 | 0.00 | 0.61 | 0.00 | 0.54 | 0.00 | 0.40 |
| RMU | lr5e-5_ep5@39 | 0.15 | 0.77 | 0.96 | 0.01 | 0.93 | 0.00 | 0.94 | 0.00 | 0.69 |
| LUNAR | c1.5 | 0.14 | 0.48 | 0.54 | 0.48 | 0.50 | 0.07 | 0.54 | 0.08 | 0.33 |
| NEAR-Act | rw1_ans_increment_seq_c14_l11-12-13 | 0.18 | 0.91 | 0.91 | 0.84 | 0.78 | 0.04 | 0.91 | 0.00 | 0.62 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21 | 0.14 | 0.03 | 0.17 | 0.00 | 0.00 | 0.00 | 0.07 | 0.00 | 0.11 |
| ROCR* | l3-4-5 | 0.56 | 0.21 | 0.86 | 0.00 | 0.57 | 0.21 | 0.86 | 0.00 | 0.65 |
| ROCR-nb* | l3-4-5 | 0.63 | 0.02 | 0.20 | 0.00 | 0.00 | 0.00 | 0.05 | 0.00 | 0.17 |
| ROCR-nb-s8 | none | | | | | | | | | |

### 8B forget10  (fixed layers [16, 23, 31])

| method | point | Prob | wb_unsup (best layer) | wb_sup (best layer) | unsup@L16 | sup@L16 | unsup@L23 | sup@L23 | unsup@L31 | sup@L31 |
|---|---|---|---|---|---|---|---|---|---|---|
| GradAscent | lr1.5e-5_ep5@26 | 0.01 | 0.05 | 0.27 | 0.01 | 0.04 | 0.00 | 0.01 | 0.00 | 0.00 |
| NPO | lr2e-5_ep5@65 | 0.11 | 0.12 | 0.86 | 0.12 | 0.86 | 0.00 | 0.73 | 0.00 | 0.48 |
| RMU | lr5e-5_ep5@26 | 0.21 | 0.92 | 0.93 | 0.56 | 0.90 | 0.00 | 0.80 | 0.00 | 0.48 |
| LUNAR | c2.4 | 0.21 | 0.80 | 0.72 | 0.03 | 0.24 | 0.79 | 0.72 | 0.80 | 0.39 |
| NEAR-Act | rw1_ans_increment_seq_c8_l17-18-19 | 0.15 | 1.00 | 0.83 | 0.03 | 0.24 | 0.95 | 0.80 | 0.43 | 0.53 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18 | 0.13 | 0.08 | 0.27 | 0.03 | 0.23 | 0.00 | 0.01 | 0.00 | 0.01 |
| ROCR* | l4-5-6 | 0.65 | 0.01 | 0.80 | 0.01 | 0.77 | 0.00 | 0.71 | 0.00 | 0.41 |
| ROCR-nb* | l4-5-6 | 0.65 | 0.02 | 0.14 | 0.01 | 0.09 | 0.00 | 0.08 | 0.00 | 0.03 |
| ROCR-nb-s8 | none | | | | | | | | | |
