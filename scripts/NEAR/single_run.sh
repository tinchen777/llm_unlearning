#!/bin/bash
# ONE NEAR experiment, steps 1-4, with every parameter in the block below (run.sh = many models x splits x K x modes).
#   r_UV -> layer selection -> W_down -> eval of the edited model
#
#   bash scripts/NEAR/single_run.sh [GPU_ID] ["STAGES"]
#   bash scripts/NEAR/single_run.sh 0                  # stages 1 2 3 4
#   bash scripts/NEAR/single_run.sh 0 "3 4"            # only W_down + eval (r_uv.pt and layer_selection.json already exist)
#   MODEL=Llama-3.2-3B-Instruct K=5 COEFF=1.5 bash scripts/NEAR/single_run.sh 0     # every parameter can come from the environment
#   DRY_RUN=1 bash scripts/NEAR/single_run.sh          # only print the commands
#
# Output: saves/NEAR/${TASK_NAME}/k${K}_c${COEFF}_${MODE}/   (layer_selection.json, l<layer>[_cf]/model, ...)
#         saves/eval/NEAR/${TASK_NAME}_<tag>/            (TOFU_SUMMARY.json, TOFU_EVAL.json, *_responses.json, activations*)
# Needs for forget quality / privleak in the eval: saves/eval/.../tofu_<MODEL>_<RETAIN_SPLIT>/TOFU_EVAL.json (see eval.tofu.retain_logs_path).
set -e
cd "$(dirname "$0")/../.." || exit 1

# ============================== parameters of the experiment ==============================
MODEL=${MODEL:-Llama-3.2-1B-Instruct}         # base model; the target is the official TOFU full model of it (configs/model/tofu/)
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-neighbor01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
AUTHOR_OFFSET=${AUTHOR_OFFSET:-199}           # makes the author numbering of the forget rows equal to `author_id` of the neighbour csv (forget01: 199, forget05: 191, forget10: 181)

K=${K:-5}                                     # neighbours per author used as the reference
COEFF=${COEFF:-1.0}                           # forget target = original MLP output + COEFF * r_UV
MODE=${MODE:-eoi}                             # all = the paper (r_UV on every token) | eoi = end-of-instruction tokens only
SOLVER=${SOLVER:-adam}                        # adam | closed_form

LAYER_FRACTION=${LAYER_FRACTION:-[0.4,0.8]}   # candidate layers of the selection (relative depth); LAYERS=[4,5,6] for an explicit list
LAYERS=${LAYERS:-}                            # empty: use LAYER_FRACTION
W_DOWN_LAYER=${W_DOWN_LAYER:-auto}            # auto = the selected layer; or [10]
# extra Hydra overrides of a stage, e.g. EXTRA_W_DOWN="wdown.lr=1e-2 wdown.epochs=40"
EXTRA_R_UV=${EXTRA_R_UV:-}
EXTRA_SELECT=${EXTRA_SELECT:-}
EXTRA_W_DOWN=${EXTRA_W_DOWN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}

# TOFU_EVAL.json of the retain model: the reference of forget quality and privleak (None without it)
RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}

TASK_NAME=${TASK_NAME:-single/near_tofu_${MODEL}_${FORGET_SPLIT}}
# ===========================================================================================

GPU_ID=${1:-0}
STAGES=${2:-"1 2 3 4"}
for stage in ${STAGES}; do
  case "${stage}" in 1|2|3|4) ;; *) echo "unknown stage '${stage}' (valid: 1 2 3 4)" >&2; exit 1 ;; esac
done

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "Using GPU: [${GPU_ID}]  stages: ${STAGES}"

TARGET_MODEL=tofu/tofu_${MODEL}_full
SPLITS=(forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT})
# folders: same rules as config_dir() / run_dir() / w_down_tag() in scripts/NEAR/_common.sh
CONFIG_DIR=saves/NEAR/${TASK_NAME}/k${K}_c$(printf "%g" "${COEFF}")_${MODE}
SELECTION=${CONFIG_DIR}/layer_selection.json
CF=""; if [ "${SOLVER}" = closed_form ]; then CF=_cf; fi

run() {   # run "<name>" cmd...
  local name=$1; shift
  echo "############  START STAGE ${name} | ${MODEL} ${FORGET_SPLIT} | K=${K} COEFF=${COEFF} MODE=${MODE} SOLVER=${SOLVER}  ############"
  if [ -n "${DRY_RUN}" ]; then echo "$@"; else "$@"; fi
  echo "############  END STAGE ${name}  ############"
}

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has_stage 4 && [ ! -f "${RETAIN_LOGS_PATH}" ]; then
  echo "WARNING: ${RETAIN_LOGS_PATH} not found: forget_quality and privleak will be None (RETAIN_LOGS_PATH=... to change it)" >&2
fi

if has_stage 1; then
  # r_UV of every K in ruv.ks (here only K): saves/NEAR/${TASK_NAME}/r_uv.pt
  run "[1] r_uv" python src/METHOD/NEAR/r_uv.py \
    experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    "ruv.ks=[${K}]" basic.author_offset=${AUTHOR_OFFSET} \
    task_name=${TASK_NAME} ${EXTRA_R_UV}
fi

if has_stage 2; then
  LAYER_ARG="select.layer_fraction=${LAYER_FRACTION}"
  if [ -n "${LAYERS}" ]; then LAYER_ARG="select.layers=${LAYERS} select.layer_fraction=null"; fi
  run "[2] select_layer" python src/METHOD/NEAR/select_layer.py \
    experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    basic.k=${K} basic.coeff=${COEFF} basic.shift_mode=${MODE} ${LAYER_ARG} \
    task_name=${TASK_NAME} ${EXTRA_SELECT}
fi

if has_stage 3; then
  # k / coeff / shift_mode / r_uv.pt are read from the layer_selection.json of stage 2
  run "[3] w_down" python src/METHOD/NEAR/w_down.py \
    experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    wdown.layer_selection_path=${SELECTION} wdown.solver=${SOLVER} "wdown.layers=${W_DOWN_LAYER}" \
    task_name=${TASK_NAME} ${EXTRA_W_DOWN}
fi

if has_stage 4; then
  if [ -n "${DRY_RUN}" ]; then
    LAYER='<selected layer>'
  else
    LAYER=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
  fi
  EDITED=${CONFIG_DIR}/l${LAYER}${CF}/model
  TAG=k${K}_l${LAYER}_c$(printf "%g" "${COEFF}")_${MODE}${CF}
  # ONE evaluation: TOFU metrics + responses + activations (positions / layers / point / author numbering from r_uv.pt, the edited layer
  # is the focus layer). The edited model should behave like the retain model: it has only "seen" the retain set.
  run "[4] eval" python src/eval.py \
    experiment=eval/tofu/probe model=${TARGET_MODEL} model.pretrained.name_or_path=${EDITED} "${SPLITS[@]}" \
    eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH} \
    eval.tofu.metrics.activation_probe.r_uv_path=saves/NEAR/${TASK_NAME}/r_uv.pt \
    eval.tofu.metrics.activation_probe.focus_layer=${LAYER} \
    "eval.tofu.metrics.activation_probe.seen=[retain]" \
    "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]" \
    task_name=NEAR/${TASK_NAME}_${TAG} ${EXTRA_EVAL}
fi
echo "################ done: stages ${STAGES} (${CONFIG_DIR}) ################"
