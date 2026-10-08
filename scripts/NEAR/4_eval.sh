#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"   # MODEL, splits, FULL_MODEL, TASK_NAME, KS, COEFF, SHIFT_MODES, SOLVER
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# Step 4: evaluate the edited models of 5_w_down.sh
#   (a) TOFU metrics (forget quality, model utility, privleak, MIA, ...) -> saves/eval/<task>/TOFU_EVAL.json
#       RETAIN_LOGS: TOFU_EVAL.json of the retain model of RETAIN_SPLIT (forget quality / privleak reference)
#   (b) the step-1 probe on the edited model: compare its responses_summary.json / activations_summary.json with
#       the `full` and `retain` probes of 1_probe.sh (forget should now look like neighbor / holdout, as for `retain`)
RETAIN_LOGS=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    SELECTION=saves/neighbor/${TASK_NAME}/$(selection_name ${K} ${MODE})
    LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
    TAG=$(w_down_tag ${K} ${LAYER} ${MODE})
    EDITED=saves/neighbor/${TASK_NAME}/model_${TAG}
    echo start eval ${EDITED}
    python src/eval.py \
      experiment=eval/tofu/default \
      model=${FULL_MODEL} \
      model.pretrained.name_or_path=${EDITED} \
      forget_split=${FORGET_SPLIT} \
      holdout_split=${HOLDOUT_SPLIT} \
      retain_logs_path=${RETAIN_LOGS} \
      task_name=test/redirect_${MODEL}_${FORGET_SPLIT}_${TAG}

    # the probe of an edited model: neither forget nor retain data is "seen" in the sense of the full model any more
    python src/neighbor.py \
      experiment=generate/neighbor_probe \
      model=${FULL_MODEL} \
      model.pretrained.name_or_path=${EDITED} \
      ${SPLITS} \
      task_name=test/probe_${MODEL}_${FORGET_SPLIT}_redirect_${TAG}
    echo end eval ${EDITED}
  done
done
