# Seed spread of the baselines (matched point per seed; seed 0 = the default seed of the main table)

## 1B forget01 GradAscent

| seed | point | Prob | utility | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|
| 0 | lr1e-4_ep5@2 | 0.16 | 0.43 | 0.00 | 0.22 | 0.54 |
| 1 | lr1e-4_ep5_s1@2 | 0.14 | 0.43 | 0.02 | 0.01 | 0.19 |
| 2 | lr1e-4_ep5_s2@2 | 0.13 | 0.43 | 0.10 | 0.06 | 0.25 |
| **all** | mean / sd / range | | | | | 0.33 / 0.19 / 0.19-0.54 |

## 1B forget01 NPO

| seed | point | Prob | utility | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|
| 0 | lr1e-4_ep5@2 | 0.16 | 0.43 | 0.00 | 0.22 | 0.55 |
| 1 | lr1e-4_ep5_s1@10 | 0.02 | 0.44 | 0.10 | 0.28 | 0.57 |
| 2 | lr1e-4_ep5_s2@2 | 0.13 | 0.43 | 0.08 | 0.07 | 0.27 |
| **all** | mean / sd / range | | | | | 0.47 / 0.17 / 0.27-0.57 |

## 3B forget01 GradAscent

| seed | point | Prob | utility | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|
| 0 | lr1e-4_ep5@2 | 0.18 | 0.52 | 0.00 | 0.09 | 0.88 |
| 1 | lr1e-4_ep5_s1@2 | 0.19 | 0.52 | 0.00 | 0.03 | 0.59 |
| 2 | lr1e-4_ep5_s2@2 | 0.17 | 0.52 | 0.00 | 0.08 | 0.60 |
| **all** | mean / sd / range | | | | | 0.69 / 0.16 / 0.59-0.88 |

## 3B forget01 NPO

| seed | point | Prob | utility | ud_bb | wb_unsup | wb_sup |
|---|---|---|---|---|---|---|
| 0 | lr1e-4_ep5@2 | 0.18 | 0.52 | 0.01 | 0.10 | 0.87 |
| 1 | lr1e-4_ep5_s1@2 | 0.19 | 0.52 | 0.00 | 0.04 | 0.59 |
| 2 | lr1e-4_ep5_s2@2 | 0.17 | 0.52 | 0.00 | 0.06 | 0.62 |
| **all** | mean / sd / range | | | | | 0.69 / 0.15 / 0.59-0.87 |

