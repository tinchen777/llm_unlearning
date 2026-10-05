#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-1B-Instruct
FORGET_SPLIT=forget10



echo start WGA
python src/train.py --config-name=unlearn \
  experiment=unlearn/muse/default \
  model=${MODEL} \
  trainer=WGA \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.alpha=1.0 \
  trainer.method_args.retain_loss_type=NLL \
  forget_split=forget10 \
  retain_split=retain90 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
  task_name=test/WGA \
  --cfg job --resolve
echo end WGA
