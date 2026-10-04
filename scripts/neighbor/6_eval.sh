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

# Step 4: evaluate the edited model of 5_w_down.sh
#   (a) TOFU metrics (forget quality, model utility, privleak, MIA, ...) -> saves/eval/<task>/TOFU_EVAL.json
#       RETAIN_LOGS: TOFU_EVAL.json of the retain model of RETAIN_SPLIT (forget quality / privleak reference)
#   (b) the step-1 probe on the edited model: compare its responses_summary.json / activations_summary.json with
#       the `full` and `retain` probes of 1_probe.sh (forget should now look like neighbor / holdout, as for `retain`)
LAYER=$(python -c "import json; print(json.load(open('saves/neighbor/${TASK_NAME}/layer_selection.json'))['selected_layer'])")
RETAIN_LOGS=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for K in 1 5 15; do
  TAG=k${K}_l${LAYER}_c1
  EDITED=saves/neighbor/${TASK_NAME}/model_${TAG}
  echo start eval ${EDITED}
  python src/eval.py \
    experiment=eval/tofu/default \
    model=${MODEL} \
    model.pretrained.name_or_path=${EDITED} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    task_name=test/redirect_${MODEL}_${FORGET_SPLIT}_${TAG}

  python src/neighbor.py \
    experiment=generate/neighbor_probe \
    model=${MODEL} \
    model.pretrained.name_or_path=${EDITED} \
    ${SPLITS} \
    task_name=test/probe_${MODEL}_${FORGET_SPLIT}_redirect_${TAG}
  echo end eval ${EDITED}
done
