## 问题

其他 unlearn baseline 跑 IdkDPO 时报 Bad characters ... 'retain??'。

## 原因

configs/eval/tofu.yaml 的 retain_split 默认 'retain??'（必须由实验配置给出）。unlearn/tofu/default.yaml 的 eval.tofu 有 retain_split: ${retain_split}，idk.yaml 与 finetune/tofu/neighbor.yaml 漏了。

## 做法

两个文件补上；新增测试 test_tofu_experiments_hand_their_splits_to_the_evaluation（所有 TOFU 实验的 eval.tofu.*_split 必须来自各自的 *_split）。

## 结果

已修并推送（131f398）；用户重跑 IdkDPO，待确认。其它方法不受影响。输出路径已改为 `saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET_SPLIT>`。

## 下一步

待用户确认 IdkDPO 跑通。另：baseline 目前 forget_quality≈0.003~0.007（≈full，基本没遗忘），疑似 bf16+lr1e-5 更新偏弱（见 `near_line/1`），对比前要先解决。
