# 14 LUNAR 与 baseline 调参结果（forget01，1B）

## 问题
调参之后，baseline 与 LUNAR 能否在"遗忘到 retain 水平"的同时保持 utility？与 NEAR 的公平比较。

## 做法
- LUNAR：系数 {0.5, 1, 1.5, 2} × forget01 / forget005（`sweep_coeff.sh`，已有的 c2 在同一目录）。没有跑 `LUNAR_unverified`。
- baseline：`tune.sh`，lr {默认, 5e-5, 1e-4}，共 20 步（forget01 40 条，每 epoch 2 步），按 `baseline_table.py` 取 best epoch。未见到 fp32 的对照目录（`DTYPES` 未跑或名字不带精度后缀）。
- 修了 `baseline_table.py`：某个 run 缺 step-0 评测时（IdkDPO lr1e-4），"utility ≥ 原模型 95%"的筛选会被直接放行，现改为用同一模型/forget 的 step-0 作基准；IdkDPO lr1e-4 现在正确标 `*`。

## 结果

### baseline（best epoch；retain 参考：Prob 0.169，mia 0.483，extraction 0.062；原模型 utility 0.597）
| 方法 / lr | 取的 step | forget Prob | mia | extraction | utility | retain ROUGE |
|---|---|---|---|---|---|---|
| IdkDPO 1e-4 | 20* | 0.132 | 0.461 | 0.040 | 0.414 | 0.339 |
| IdkDPO 5e-5 | 18* | 0.299 | 0.759 | 0.076 | 0.512 | 0.408 |
| RMU 1e-4 | 2* | 0.152 | 0.414 | 0.048 | 0.412 | 0.285 |
| NPO 1e-4 | 2* | 0.184 | 0.438 | 0.071 | 0.444 | 0.329 |
| SimNPO 1e-4 | 20* | 0.306 | 0.811 | 0.075 | 0.427 | 0.359 |
| GradAscent 1e-4 | 2* | 0.297 | 0.712 | 0.065 | 0.462 | 0.350 |
| GradDiff 1e-4 | 2* | 0.273 | 0.721 | 0.053 | 0.477 | 0.341 |

（`*` = 没有任何 epoch 满足 utility ≥ 95% 原模型。）

- 默认 lr 几乎不遗忘（Prob 0.34–0.90），lr 加大后才有遗忘，但 **utility 已经掉到 0.41–0.51**，没有一个满足 95% 约束。
- **IdkDPO 1e-4**：最接近 retain 水平（Prob 0.13、mia 0.46、extraction 0.04），但 utility 0.41、retain ROUGE 0.34（原模型 0.82）；step 2 的 utility=0.0（暂时崩溃）后才回升；forget_quality 0.58（n=40 的离散台阶，不当结论）。
- **RMU 1e-4**：2 步后 Prob 0.152、mia 0.414，utility 0.412；之后继续训练 utility 单调崩（step 12：0.117，step 14 起 utility 缺失）。
- **NPO 1e-4**：过度遗忘（Prob 0.02、mia 0.01–0.04），utility 稳定在 0.38–0.41。
- 评测中没有 UD（`tune.sh` 默认 `PROBE=0`），baseline 的 `ud_*` 仍缺。

### LUNAR（untuned 系数扫描，没有新增）
- forget01：c0.5 Prob 0.796/utility 0.592（几乎不遗忘）；c1 0.518/0.561；c1.5 0.115/0.396（`ud_bb` 0.52）；c2 0.024/0.269。
- forget005（作者 200）：c0.5 0.798/0.591；c1 0.514/0.569；c1.5 0.132/0.442（`ud_bb` 0.475）；c2 0.028/0.449（`ud_bb` 0.25）。
- 即使扫了系数，LUNAR 也是 **遗忘到 retain 水平 ⇔ utility 掉到 0.40–0.44，且 25–52% 的回答被 UD 判为痕迹**。

### 与 NEAR 并排（遗忘到 retain 水平时）
| | Prob | mia | utility | retain ROUGE | `ud_bb` |
|---|---|---|---|---|---|
| NEAR B+ans（c7，l7-9） | 0.206 | 0.531 | **0.569** | 0.588 | 0.046 |
| NEAR + KL（l7-9） | 0.227 | 0.724 | 0.553 | 0.551 | 0.0 |
| IdkDPO 1e-4 | 0.132 | 0.461 | 0.414 | 0.339 | 未测 |
| RMU 1e-4（step 2） | 0.152 | 0.414 | 0.412 | 0.285 | 未测 |
| NPO 1e-4（step 2） | 0.184 | 0.438 | 0.444 | 0.329 | 未测 |
| LUNAR forget01 c1.5 | 0.115 | 0.426 | 0.396 | 0.478 | 0.52 |

## 结论与注意
1. 在"遗忘到 retain 水平"这个匹配点上，NEAR 的 utility（0.55–0.57）比调过 lr 的 baseline（0.41–0.44）和 LUNAR（0.40）高约 0.12–0.17，retain ROUGE 高约 0.12–0.30。这是论文的主结果，【稳】在方向上，**幅度仍需谨慎**：单次运行、n=40。
2. 公平性仍有缺口：baseline 只调了 lr 的两档、固定 20 步、没有 fp32 对照、没有调 batch/epoch/权重（如 GradDiff 的 retain 权重、NPO 的 β、RMU 的层/系数）；LUNAR 只扫了系数，没有换参考集或 `LUNAR_unverified`。论文里要写"在 X 的调参预算内"。
3. baseline 的 UD 缺失；要宣称"更隐蔽"，至少补 IdkDPO 1e-4 / RMU / NPO 的 `PROBE=1`。
4. 现象：baseline 在 step 2 前后 utility 非单调（IdkDPO 1e-4 step 2 为 0），best epoch 选择受 step 粒度影响；评测每 2 步一次，粒度粗。

## 下一步
1. 对 IdkDPO 1e-4、RMU 1e-4、NPO 1e-4 选定 epoch 后用 `PROBE=1` 重跑，得到 `ud_bb`/`entity_bb`/`wb_unsup`。
2. 补一轮更细的 baseline 调参（更小的 lr 区间 5e-5–1e-4，较少步数的 checkpoint；GradDiff/NPO/RMU 的方法特有参数；fp32 对照），避免被质疑 baseline 没调好。
3. 画 Pareto（遗忘 Prob/mia vs utility；vs `ud_bb`），NEAR（含 KL 和各 c）、LUNAR、baseline 同图。
