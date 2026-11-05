## 问题

W_down 对 retain 有损害(retain ROUGE 0.82->0.62)，且 teacher-forced 概率/MIA 仍高。尝试 (a) retain_weight=3/10，(b) wdown.include_answer（forget/retain 都用"问题+答案"全文的 token，all 模式对答案 token 也加位移）。

## 做法

scripts/NEAR/sweep_w_down.sh；c7/L9，K=5，forget01。结果在 saves/eval/NEAR/single/rw*_c7_l9_k5_l9_c7_all/。

## 结果

| | rw1 | rw3 | rw10 | rw1+ans | rw3+ans | retain99 | LUNAR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| forget_quality | 0.405 | 0.266 | 0.097 | 0.405 | 0.165 | 1.0 | 0.919 |
| model_utility | 0.5645 | 0.5684 | 0.5788 | 0.5584 | 0.5671 | 0.599 | 0.440 |
| forget_Q_A_Prob | 0.417 | 0.445 | 0.524 | 0.444 | 0.480 | 0.165 | 0.180 |
| forget ROUGE | 0.401 | 0.418 | 0.479 | 0.423 | 0.441 | 0.407 | 0.199 |
| retain ROUGE(生成) | 0.636 | 0.677 | 0.711 | 0.760 | 0.756 | - | 0.399 |
| mia_loss | 0.846 | 0.861 | 0.908 | 0.868 | 0.889 | 0.484 | 0.491 |
| privleak | -74.7 | -76.7 | -84.2 | -78.3 | -80.8 | +5.9 | -6.1 |
| achieved/intended幅度比 | 0.87 | 0.70 | 0.45 | 0.79 | 0.61 | | |

- retain_weight：干净的权衡曲线——retain 越保（ROUGE 0.64→0.71）遗忘越弱（forget_quality 0.40→0.10，位移幅度比 0.87→0.45）。

- include_answer：retain 生成质量明显更好（0.636→0.760），遗忘指标基本不变；但"答案 token 没训练"这个假设**没被支持**——forget_Q_A_Prob(0.417→0.444)、MIA(0.846→0.868) 没有下降。它的价值是保 retain，不是降 MIA。

- 所有变体 forget_Q_A_Prob 仍 0.42~0.52（retain 0.165）、MIA 0.85~0.91、privleak -75~-84：NEAR 是"问题→回答"的重定向，不擦除知识，teacher-forced 似然层面泄漏仍在（LUNAR 靠大位移范数≈8 压下去，代价是 utility 0.44）。

## 下一步

用户决定。候选：(1) 多层编辑（如 8/9/10 同时，每层位移更小，降痕迹也可能提升遗忘）；(2) include_answer + 更大系数(c10~12) 补回遗忘；(3) 匹配干净分布（per-sample 匹配 neighbor/retain 模型激活）代替常数平移，降低激活痕迹（与 metric_line 的白盒指标汇合）；(4) LUNAR 同口径重跑；(5) forget05 验证。
