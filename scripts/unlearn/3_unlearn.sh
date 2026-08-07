#!/bin/bash
# =============================================================================
# Demo 4: 用一个【已有的】遗忘方法做 unlearning (这里用 GradDiff)
# -----------------------------------------------------------------------------
# 入口: src/train.py  (mode=unlearn)
# experiment=unlearn/tofu/default -> configs/experiment/unlearn/tofu/default.yaml
#   - model:   待遗忘的目标模型 (默认 open-unlearning/tofu_Llama-3.2-1B-Instruct_full)
#   - data:    unlearn -> 同时加载 forget 和 retain 两个数据集
#              collator 把每个 batch 组织成 {"forget": {...}, "retain": {...}}
#   - trainer: 由命令行 trainer=GradDiff 指定 (handler=GradDiff)
#   - eval:    tofu (遗忘过程中/结束后评测)
#
# 遗忘方法的核心只有一个函数: trainer 的 compute_loss(model, inputs)。
#   - GradAscent: loss = -forget_loss
#   - GradDiff:   loss = gamma*(-forget_loss) + alpha*retain_loss
#   - NPO/SimNPO/DPO/RMU/...: 各自不同的 compute_loss
#
# 输出: saves/unlearn/demo_unlearn_graddiff
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-2-7b-hf

echo start MUSE BoundedGradDiff
python src/train.py --config-name=unlearn \
  experiment=unlearn/muse/default \
  model=${MODEL} \
  trainer=BoundedGradDiff \
  trainer.args.per_device_train_batch_size=1 \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.alpha=1.0 \
  trainer.method_args.retain_loss_type=NLL \
  forget_split=forget \
  retain_split=retain1 \
  retain_logs_path=saves/eval/muse_${MODEL}_News_retrain/MUSE_EVAL.json \
  task_name=test_muse/BoundedGradDiff \
  # --cfg job --resolve
echo end MUSE BoundedGradDiff
