#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-3B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99

# r_UV per forget author, ablated over the number K of neighbours used as reference:
#   -> r_uv.pt           r_UV[K][author] = [n_pos, n_layers, d], + the "before" activations, neighbour order, ...
#   -> r_uv_summary.json per K and layer: ||r_UV||, ||r_UV|| / ||forget activation||, cos(r_UV[K], r_UV[max K])
# output: saves/neighbor/<task_name>/   (settings: `ruv` in configs/experiment/custom/neighbor_probe.yaml)
# NOTE: `ruv.qa_per_author` / `ruv.author_offset` must make the author numbering of the forget rows match
#       `author_id` of the neighbour csv; this is checked (and explained) before any forward pass.
echo start r_uv ${MODEL}

python src/r_uv.py --config-name=train \
  experiment=custom/neighbor_probe \
  model=${MODEL} \
  forget_split=${FORGET_SPLIT} \
  neighbor_split=${NEIGHBOR_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  'ruv.ks=[1,5,15]' \
  task_name=test/redirect_${MODEL}_${FORGET_SPLIT}
  # ruv.author_offset=0 \
  # ruv.point=block_out \
  # --cfg job --resolve

echo end r_uv ${MODEL}
