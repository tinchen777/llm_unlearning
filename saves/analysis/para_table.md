# E6: forgetting under paraphrase (matched points; `-` = fast-profile evaluation without these metrics)

Prob = p(true answer); PARA = p(paraphrased true answer); PERT = p(wrong answers); TR = truth ratio. Reference rows: retain / full model.

## 9-cell medians (cells with the metric only; n = number of cells)

| method | n | Prob | PARA | PERT | TR | PARA / retain PARA |
|---|---|---|---|---|---|---|
| *retain* | 9/9 | 0.127 | 0.053 | 0.040 | 0.637 | 1.000 |
| *full* | 9/9 | 0.951 | 0.092 | 0.039 | 0.474 | 1.785 |
| GradAscent | 4/9 | 0.098 | 0.034 | 0.021 | 0.573 | 0.642 |
| NPO | 4/9 | 0.193 | 0.071 | 0.050 | 0.625 | 1.353 |
| RMU | 3/9 | 0.153 | 0.054 | 0.038 | 0.677 | 1.025 |
| LUNAR | 9/9 | 0.194 | 0.031 | 0.023 | 0.653 | 0.632 |
| NEAR-Act | 9/9 | 0.179 | 0.051 | 0.034 | 0.641 | 0.984 |
| NEAR-Dist | 9/9 | 0.146 | 0.041 | 0.027 | 0.612 | 0.854 |

### 1B forget01

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.165 | 0.066 | 0.049 | 0.652 |
| *full* | (reference) | 0.901 | 0.098 | 0.041 | 0.473 |
| GradAscent | lr1e-4_ep5@2 | 0.158 | - | - | - |
| NPO | lr1e-4_ep5@2 | 0.159 | - | - | - |
| RMU | lr5e-5_ep5@10 | 0.032 | - | - | - |
| LUNAR | c1.5 | 0.115 | 0.034 | 0.027 | 0.682 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 0.206 | 0.045 | 0.030 | 0.652 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0 | 0.155 | 0.062 | 0.035 | 0.567 |

### 3B forget01

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.179 | 0.065 | 0.053 | 0.637 |
| *full* | (reference) | 0.968 | 0.091 | 0.045 | 0.505 |
| GradAscent | lr1e-4_ep5@2 | 0.179 | - | - | - |
| NPO | lr1e-4_ep5@2 | 0.184 | - | - | - |
| RMU | lr5e-5_ep5@10 | 0.152 | - | - | - |
| LUNAR | c2 | 0.157 | 0.024 | 0.023 | 0.738 |
| NEAR-Act | rw1_ans_increment_seq_c12_l11-12-13 | 0.161 | 0.062 | 0.048 | 0.666 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0 | 0.165 | 0.050 | 0.034 | 0.630 |

### 8B forget01

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.140 | 0.045 | 0.036 | 0.655 |
| *full* | (reference) | 0.995 | 0.068 | 0.028 | 0.485 |
| GradAscent | lr2e-5_ep5_ckpt@4 | 0.193 | - | - | - |
| NPO | lr5e-5_ep5@10 | 0.016 | - | - | - |
| RMU | lr5e-5_ep5@10 | 0.244 | - | - | - |
| LUNAR | c2.4 | 0.211 | 0.031 | 0.026 | 0.732 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 0.181 | 0.051 | 0.038 | 0.659 |
| NEAR-Dist | rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0 | 0.145 | 0.040 | 0.022 | 0.612 |

### 1B forget05

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.127 | 0.054 | 0.040 | 0.634 |
| *full* | (reference) | 0.885 | 0.099 | 0.039 | 0.473 |
| GradAscent | lr2e-5_ep5@14 | 0.209 | 0.079 | 0.048 | 0.605 |
| NPO | lr2e-5_ep5_ckpt@35 | 0.167 | - | - | - |
| RMU | lr5e-5_ep5_ckpt@35 | 0.006 | - | - | - |
| LUNAR | c1.5 | 0.194 | 0.027 | 0.018 | 0.622 |
| NEAR-Act | rw1_ans_increment_seq_c7_l7-8-9 | 0.239 | 0.057 | 0.034 | 0.593 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.161 | 0.041 | 0.023 | 0.588 |

### 3B forget05

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.131 | 0.055 | 0.040 | 0.625 |
| *full* | (reference) | 0.951 | 0.092 | 0.038 | 0.461 |
| GradAscent | lr2e-5_ep5_ckpt@14 | 0.230 | - | - | - |
| NPO | lr2e-5_ep5@14 | 0.244 | 0.101 | 0.059 | 0.589 |
| RMU | lr1e-4_ep5_ckpt@35 | 0.006 | - | - | - |
| LUNAR | c1.5 | 0.238 | 0.066 | 0.040 | 0.605 |
| NEAR-Act | rw1_ans_increment_seq_c14_l12-13-14 | 0.166 | 0.051 | 0.035 | 0.640 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.142 | 0.052 | 0.037 | 0.717 |

### 8B forget05

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.108 | 0.035 | 0.028 | 0.622 |
| *full* | (reference) | 0.990 | 0.065 | 0.025 | 0.479 |
| GradAscent | lr2e-5_ep5_ckpt@14 | 0.012 | - | - | - |
| NPO | lr2e-5_ep5_ckpt@35 | 0.161 | - | - | - |
| RMU | lr1e-4_ep5_ckpt@35 | 0.027 | - | - | - |
| LUNAR | c2.8 | 0.134 | 0.022 | 0.016 | 0.653 |
| NEAR-Act | rw1_ans_increment_seq_c7_l17-18-19 | 0.193 | 0.047 | 0.033 | 0.641 |
| NEAR-Dist | rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.149 | 0.030 | 0.015 | 0.553 |

### 1B forget10

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.116 | 0.053 | 0.041 | 0.627 |
| *full* | (reference) | 0.880 | 0.100 | 0.041 | 0.476 |
| GradAscent | lr1e-5_ep5@52 | 0.187 | 0.064 | 0.040 | 0.615 |
| NPO | lr2e-5_ep5@65 | 0.198 | 0.069 | 0.049 | 0.655 |
| RMU | lr5e-5_ep5@52 | 0.038 | 0.013 | 0.010 | 0.677 |
| LUNAR | c1.5 | 0.211 | 0.051 | 0.031 | 0.591 |
| NEAR-Act | rw1_ans_increment_seq_c11_l5-6-7 | 0.142 | 0.046 | 0.030 | 0.635 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.146 | 0.041 | 0.027 | 0.645 |

### 3B forget10

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.124 | 0.052 | 0.041 | 0.644 |
| *full* | (reference) | 0.951 | 0.094 | 0.039 | 0.472 |
| GradAscent | lr2e-5_ep5@65 | 0.000 | 0.000 | 0.000 | 0.000 |
| NPO | lr2e-5_ep5@65 | 0.188 | 0.074 | 0.051 | 0.636 |
| RMU | lr5e-5_ep5@39 | 0.153 | 0.054 | 0.038 | 0.690 |
| LUNAR | c1.5 | 0.137 | 0.032 | 0.023 | 0.658 |
| NEAR-Act | rw1_ans_increment_seq_c14_l11-12-13 | 0.179 | 0.053 | 0.036 | 0.640 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.141 | 0.052 | 0.040 | 0.729 |

### 8B forget10

| method | point | Prob | PARA | PERT | TR |
|---|---|---|---|---|---|
| *retain* | (reference) | 0.106 | 0.038 | 0.031 | 0.641 |
| *full* | (reference) | 0.991 | 0.068 | 0.026 | 0.474 |
| GradAscent | lr1.5e-5_ep5@26 | 0.008 | 0.003 | 0.002 | 0.541 |
| NPO | lr2e-5_ep5@65 | 0.106 | 0.040 | 0.030 | 0.614 |
| RMU | lr5e-5_ep5@26 | 0.215 | 0.073 | 0.045 | 0.635 |
| LUNAR | c2.4 | 0.207 | 0.030 | 0.019 | 0.595 |
| NEAR-Act | rw1_ans_increment_seq_c8_l17-18-19 | 0.150 | 0.037 | 0.030 | 0.669 |
| NEAR-Dist | rw1_ans_kl1_s500_fp1_ftunseen_fmneighbor_scaled_initoriginal | 0.135 | 0.028 | 0.014 | 0.533 |
