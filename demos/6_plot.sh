#!/bin/bash
# =============================================================================
# Demo 6: 绘图模块 (src/vis + src/plot.py) — 把实验保存的数据绘制成图表
# -----------------------------------------------------------------------------
# 纯离线读取 run 目录 (saves/{mode}/{task_name}) 里已保存的数据, 不需要 GPU:
#   - trainer_state.json                        -> 训练 loss / 自定义分项 loss 曲线
#   - checkpoint-*/evals/{BENCH}_SUMMARY.json   -> 评测指标随训练的轨迹 / 末次对比
#   - checkpoint-*/evals/{BENCH}_EVAL.json      -> 逐样本分布直方图
#
# 子命令:
#   report    一键生成全部适用图表 (train + evals + compare + tradeoff)
#   train     训练曲线 (--keys 指定 log_history 里的键, 默认自动发现)
#   evals     指标轨迹 (--metrics 指定, 默认全部)
#   compare   末次指标跨方法对比柱状图
#   tradeoff  遗忘-效用散点 (--x model_utility --y forget_quality)
#   dist      逐样本分布 (--metric forget_Q_A_Prob --stat prob [--step N])
#
# 输出: PNG, 默认存到 <第一个run>/plots/, 用 -o DIR 覆盖。
# =============================================================================
set -e
cd "$(dirname "$0")/.."

# 一键报告: 对比两个 run
python src/plot.py report \
  saves/unlearn/demo_unlearn_NCU \
  saves/unlearn/demo_unlearn_graddiff \
  -o saves/plots/ncu_vs_graddiff

# 单独画某张图的例子:
# python src/plot.py train saves/unlearn/demo_unlearn_NCU \
#   --keys loss forget_pull_loss anchor_loss retain_lm_loss
# python src/plot.py evals saves/unlearn/demo_unlearn_NCU --metrics forget_quality model_utility
# python src/plot.py tradeoff saves/unlearn/*
# python src/plot.py dist saves/unlearn/demo_unlearn_NCU --metric forget_Q_A_Prob --stat prob
