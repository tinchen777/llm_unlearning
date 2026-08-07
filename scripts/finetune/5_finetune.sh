#!/bin/bash
# =============================================================================
# Demo 1: 直接（全参数）微调一个模型
# -----------------------------------------------------------------------------
# 入口: src/train.py  (mode=train)
# 通过 Hydra 把以下配置组合在一起:
#   experiment=finetune/tofu/default
#     -> configs/experiment/finetune/tofu/default.yaml
#        - model:   Llama-3.2-1B-Instruct
#        - trainer: finetune  (handler=FinetuneTrainer, configs/trainer/finetune.yaml)
#        - data:    TOFU_QA_full (locuslab/TOFU 的 "full" split)
#        - eval:    tofu  (训练过程中按 epoch 评测)
#
# 训练完的权重会保存到 paths.output_dir，默认 = saves/<mode>/<task_name>
# 即: saves/finetune/demo_finetune_full
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Qwen2.5-3B-Instruct

python src/train.py \
  experiment=finetune/muse/default \
  model=${MODEL} \
  trainer.args.eval_on_start=True \
  trainer.args.per_device_train_batch_size=1 \
  trainer.args.num_train_epochs=5 \
  task_name=test/muse_${MODEL}_full \
  # trainer.args.gradient_checkpointing=true \

echo end finetune ${MODEL}
