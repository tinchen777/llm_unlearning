## 问题

NEAR（eoi、系数1、第11层）几乎没遗忘：forget ROUGE 0.81（retain 参考 0.41）、forget_quality 0.0068（与 full 相同）、MIA≈1、privleak≈-100。

## 排查

W_down 训练没问题（achieved_vs_intended 余弦0.977、幅度比0.88；bf16 舍入无影响）。问题是推理时 steering 本身无效：候选层 unseen_gap≈1.0。r_UV 太小（第11层范数0.72；LUNAR 约4.1 且系数2）。顺带修：hallucination_rate 恒为1 的 bug（text metric 互相看不到）、probe 实验把 retain_logs_path 置 null 导致没有 forget_quality。

## 结果

结论：先要找到 steering 有效的设置，再训练 W_down。

## 下一步

只跑选层（不训练）扫 系数×模式×层：scripts/NEAR/sweep_select.sh、sweep_fine.sh，汇总 summarize_sweep.py --top。
