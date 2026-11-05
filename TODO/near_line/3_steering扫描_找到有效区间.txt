【问题】哪些 (模式, 系数, 层) 下推理时加 coeff*r_UV 能把 forget 回答改成"像陌生人"（目标 ROUGE 0.39、答案对数概率 -2.57）且不乱码？
【做法】sweep_select.sh / sweep_fine.sh，K=5，forget01，1B 官方 full。
【结果】(saves/NEAR/single/near_tofu_Llama-3.2-1B-Instruct_forget01/k5_c*_*/)
  - eoi 模式：系数1 时第0~13层完全无效；只在最后两层(14,15)有效，且是"改回答开头"（偏浅）。
  - all 模式有明确有效区间：系数12/第6层(unseen_gap 0.26, ROUGE 0.39, 退化0)、系数8/第9层(0.27, 退化0.05)、系数7/第9层(0.31, 退化0)。回答变成通顺的编造/回避式回答，简单事实(如性别)不变。系数10 时第7~9层退化率 12~30%。
【下一步】训练 W_down：K=5 COEFF=12 MODE=all W_DOWN_LAYER=[6]；备选 COEFF=7 W_DOWN_LAYER=[9]（scripts/NEAR/single_run.sh "3 4"）。
