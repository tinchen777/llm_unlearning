#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/_common.sh"
# MODEL, SPLITS, TARGET_MODEL, TASK_NAME, KS, SHIFT_MODES
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# Step 4: evaluate the edited models of 3_w_down.sh
#   (a) TOFU metrics (forget quality, model utility, privleak, MIA, ...) -> saves/eval/TOFU/<task>/TOFU_EVAL.json
#       RETAIN_LOGS: TOFU_EVAL.json of the retain model of RETAIN_SPLIT (forget quality / privleak reference)
#   (b) the step-1 probe on the edited model: compare its responses/responses_summary.json / activations_summary.json with
#       the `full` and `retain` probes of 0_probe.sh (forget should now look like neighbor / holdout, as for `retain`)
RETAIN_LOGS=saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    CONFIG_DIR=saves/NEAR/${TASK_NAME}/$(config_dir ${K} ${MODE})
    SELECTION=${CONFIG_DIR}/layer_selection.json
    LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
    TAG=$(w_down_tag ${K} ${LAYER} ${MODE})
    EDITED=${CONFIG_DIR}/$(run_dir ${LAYER})/model

    echo start eval ${EDITED}
    python src/eval.py \
      experiment=eval/tofu/default \
      model=${TARGET_MODEL} \
      model.pretrained.name_or_path=${EDITED} \
      forget_split=${FORGET_SPLIT} \
      holdout_split=${HOLDOUT_SPLIT} \
      retain_logs_path=${RETAIN_LOGS} \
      task_name=NEAR/test/near_tofu_${MODEL}_${FORGET_SPLIT}_${TAG}

    # the probe of an edited model: neither forget nor retain data is "seen" in the sense of the full model any more
    python src/METHOD/NEAR/probe.py \
      experiment=unlearn/tofu/NEAR \
      model=${TARGET_MODEL} \
      model.pretrained.name_or_path=${EDITED} \
      ${SPLITS} \
      probe.activations.from_r_uv=saves/NEAR/${TASK_NAME}/r_uv.pt \
      task_name=probe/test/probe_near_tofu_${MODEL}_${FORGET_SPLIT}_${TAG}
    echo end eval ${EDITED}
  done
done
