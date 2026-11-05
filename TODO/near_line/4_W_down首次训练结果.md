【问题】把 steering 有效的设置训练成 W_down，遗忘与 retain 效果如何？
【做法】c12/L6 与 c7/L9（all 模式，K=5，forget01），eval=一次 eval.py（TOFU+响应+激活）。
【结果】(saves/eval/NEAR/single/..._k5_l6_c12_all, ..._k5_l9_c7_all) 遗忘开始有效：生成层面接近 retain 模型。
  forget_quality  full0.0068 | c12/L6 0.014 | c7/L9 0.40 | LUNAR 0.92(只训了20QA)
  model_utility   full0.599  | 0.573       | 0.565     | LUNAR 0.440
  forget ROUGE    retain0.41 | 0.42        | 0.40      | LUNAR 0.20
  forget_Q_A_Prob retain0.165| 0.54        | 0.42      | LUNAR 0.18
  privleak        retain+5.9 | -89         | -75       | LUNAR -6;  mia_loss 0.93/0.85 vs retain 0.48
  retain ROUGE 0.80->0.64；holdout/neighbor 不变；无拒答、无退化（回答被换成通顺的编造：人名/父母职业/奖项）。
  注意：编辑后 activation_auc_forget_neighbor 在被编辑层之后升到 0.73~0.93（之前<0.5）= 编辑痕迹，不是遗忘成功的指标。
【下一步】(a) retain_weight 1→3→10 降低 retain 损害；(b) 假设"答案 token 没被训练到所以 teacher-forced 概率/MIA 仍高"→ wdown.include_answer；(c) LUNAR 用 FORGET_HF_SPLIT=train 与 NEAR 同口径(40条)重跑；(d) forget05 验证。
