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
#   -> saves/eval/LUNAR/eval_lunar_tofu_<MODEL>_<FORGET_SPLIT>_<REF>_<TAG>/TOFU_EVAL.json   (mode `eval`: saves/eval/<task_name>)
#      REF = basename of REF_DATASET (LUNAR_harmful | LUNAR_unverified), TAG = c<coeff>_l<layer>_<mode>[_cf]
# RETAIN_LOGS: TOFU_EVAL.json of the retain model of RETAIN_SPLIT (forget quality / privleak reference).
# NOTE: the TOFU forget metrics are those of the WHOLE forget split (forget_split=${FORGET_SPLIT}); LUNAR unlearns the
# first 20 QAs (one author) of it, so forget quality is only meaningful if FORGET_HF_SPLIT is "train" (all of it).
RETAIN_LOGS=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for MODE in "${SHIFT_MODES[@]}"; do
  CONFIG_DIR=saves/LUNAR/${TASK_NAME}/$(config_dir ${MODE})
  SELECTION=${CONFIG_DIR}/layer_selection.json
  LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
  TAG=$(w_down_tag ${LAYER} ${MODE})
  EDITED=${CONFIG_DIR}/$(run_dir ${LAYER})/model
  echo start eval ${EDITED}
  python src/eval.py \
    experiment=eval/tofu/default \
    model=${FULL_MODEL} \
    model.pretrained.name_or_path=${EDITED} \
    forget_split=${FORGET_SPLIT} \
    holdout_split=${HOLDOUT_SPLIT} \
    retain_logs_path=${RETAIN_LOGS} \
    task_name=LUNAR/eval_lunar_tofu_${MODEL}_${FORGET_SPLIT}_$(basename "${REF_DATASET}")_${TAG}
  echo end eval ${EDITED}
done
