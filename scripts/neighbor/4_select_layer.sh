#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# keep these identical in 1_probe / 3_r_uv / 4_select_layer / 5_w_down / 6_eval (3-5 share TASK_NAME)
MODEL=Llama-3.2-1B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99
HOLDOUT_SPLIT=holdout01
SPLITS="forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}"
TASK_NAME=test/redirect_${MODEL}_${FORGET_SPLIT}

# Step 2b: LUNAR-style layer selection (needs 3_r_uv.sh). For every layer, the forget prompts are redirected at
# inference time (block_out += coeff * r_UV[K][author], what a perfect W_down edit of that layer does) and compared
# with the unsteered model on unseen questions (neighbor / holdout):
#   -> layer_selection.json        per layer: rougeL_recall, answer_logprob, refusal / degenerate rates, unseen_gap;
#                                  + `selected_layer` (min unseen_gap among non-degenerate layers). READ IT.
#   -> responses_steer_l<l>.json   steered forget answers of every layer
# (settings: `select` in configs/experiment/generate/neighbor_probe.yaml; k / coeff / shift_mode follow `wdown`)
echo start select_layer ${MODEL}
python src/select_layer.py \
  experiment=generate/neighbor_probe \
  model=${MODEL} \
  target_model=full \
  ${SPLITS} \
  wdown.k=5 \
  wdown.coeff=1.0 \
  task_name=${TASK_NAME}
  # 'select.layers=[4,5,6,7,8,9,10,11]' \
  # select.criterion=rougeL_recall \
echo end select_layer ${MODEL}
