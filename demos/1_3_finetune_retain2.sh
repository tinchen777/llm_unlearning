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

MODEL1=Qwen2.5-1.5B-Instruct
MODEL2=phi-1_5
MODEL3=Llama-3.2-3B-Instruct
MODEL4=Qwen2.5-3B-Instruct

forget_split=forget10
retain_split=retain90
holdout_split=holdout10


python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=${MODEL3} \
  trainer.args.eval_on_start=False \
  trainer.args.num_train_epochs=5  \
  data/datasets@data.train=TOFU_QA_retain \
  data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  task_name=test/tofu_${MODEL3}_${retain_split} \
  # retain_logs_path=test/tofu_${MODEL3}_${retain_split}/TOFU_EVAL.json \
  # trainer.args.per_device_train_batch_size=8 \
  # trainer.args.ddp_find_unused_parameters=true \
  # trainer.args.gradient_checkpointing=true \
  # --cfg job --resolve

python src/train.py --config-name=train.yaml \
  experiment=finetune/tofu/default \
  model=${MODEL4} \
  trainer.args.eval_on_start=False \
  trainer.args.num_train_epochs=5  \
  data/datasets@data.train=TOFU_QA_retain \
  data.train.TOFU_QA_retain.args.hf_args.name=${retain_split} \
  forget_split=${forget_split} \
  holdout_split=${holdout_split} \
  task_name=test/tofu_${MODEL4}_${retain_split} \
  # retain_logs_path=test/tofu_${MODEL4}_${retain_split}/TOFU_EVAL.json \
  # trainer.args.per_device_train_batch_size=8 \
  # trainer.args.ddp_find_unused_parameters=true \
  # trainer.args.gradient_checkpointing=true \
  # --cfg job --resolve