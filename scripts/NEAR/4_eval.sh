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


# Step 4: evaluate the edited models of 3_w_down.sh, ONE evaluation per model (experiment=eval/tofu/probe), all in
# saves/eval/<task_name>/ (TOFU_EVAL.json, TOFU_SUMMARY.json and the files of the probe):
#   - the TOFU metrics: forget quality, model utility, privleak, MIA, ...; the reference of forget quality / privleak is the
#     TOFU_EVAL.json of the retain model of RETAIN_SPLIT (eval.tofu.retain_logs_path, derived from `retain_split`)
#   - responses of forget / retain / holdout / neighbor (forget_responses.json, ..., metrics forget_hallucination_rate, ...)
#   - activations (activations.pt, activations_summary.json, metrics activation_auc_...): the hook point, positions, layers and
#     author numbering are read from r_uv.pt (eval.tofu.metrics.activation_probe.r_uv_path) so they are those of step 1, and
#     the edited layer is the focus layer. Compare them with the `full` and `retain` probes of 0_probe.sh (forget should now
#     look like neighbor / holdout, as for `retain`).

# The edited model should behave like the retain model: it has only "seen" the retain set, forget is as unknown to it as neighbor /
# holdout. The metrics are named activation_auc_seen_unseen_max, ... whatever the sets are. (The default of the config is for the
# full model: seen [forget, retain], unseen [neighbor, holdout].)
SEEN="eval.tofu.metrics.activation_probe.seen=[retain]"
UNSEEN="eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]"

for MODE in "${SHIFT_MODES[@]}"; do
  for K in "${KS[@]}"; do
    CONFIG_DIR=saves/NEAR/${TASK_NAME}/$(config_dir ${K} ${MODE})
    SELECTION=${CONFIG_DIR}/layer_selection.json
    LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
    TAG=$(w_down_tag ${K} ${LAYER} ${MODE})
    EDITED=${CONFIG_DIR}/$(run_dir ${LAYER})/model

    echo start eval ${EDITED}
    python src/eval.py \
      experiment=eval/tofu/probe \
      model=${TARGET_MODEL} \
      model.pretrained.name_or_path=${EDITED} \
      ${SPLITS} \
      eval.tofu.metrics.activation_probe.r_uv_path=saves/NEAR/${TASK_NAME}/r_uv.pt \
      eval.tofu.metrics.activation_probe.focus_layer=${LAYER} \
      "${SEEN}" "${UNSEEN}" \
      task_name=NEAR/test/near_tofu_${MODEL}_${FORGET_SPLIT}_${TAG}
    echo end eval ${EDITED}
  done
done
