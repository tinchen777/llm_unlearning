#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Step 4: evaluate the edited models of 3_w_down.sh with the TOFU metrics (forget quality, model utility, privleak, MIA, ...)
#   -> saves/eval/<task>/TOFU_EVAL.json
# RETAIN_LOGS: TOFU_EVAL.json of the retain model of RETAIN_SPLIT (forget quality / privleak reference).
# NOTE: the TOFU forget metrics are those of the WHOLE forget split (forget_split=${FORGET_SPLIT}); LUNAR unlearns the
# first 20 QAs (one author) of it, so forget quality is only meaningful if FORGET_HF_SPLIT is "train" (all of it).
RETAIN_LOGS=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for MODE in "${SHIFT_MODES[@]}"; do
  SELECTION=saves/LUNAR/${TASK_NAME}/$(selection_name ${MODE})
  LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
  TAG=$(w_down_tag ${LAYER} ${MODE})
  EDITED=saves/LUNAR/${TASK_NAME}/model_${TAG}
  echo start eval ${EDITED}
  python src/eval.py \
    experiment=eval/tofu/default \
    model=${FULL_MODEL} \
    model.pretrained.name_or_path=${EDITED} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    task_name=test/${TASK_NAME}_${TAG}
  echo end eval ${EDITED}
done
