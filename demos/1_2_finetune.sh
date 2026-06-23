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
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0


MODEL=phi-1_5

python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=${MODEL} \
  trainer.args.eval_on_start=False \
  trainer.args.num_train_epochs=20 \
  task_name=test/tofu_${MODEL}_full \

# 产物: saves/finetune/tofu_${MODEL}_full
echo end finetune ${MODEL}
