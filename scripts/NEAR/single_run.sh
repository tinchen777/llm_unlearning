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
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-${FORGET_SPLIT/forget/neighbor}}   # forget01 -> neighbor01, forget005 -> neighbor005 (data/tofu/tofu_<split>.csv)
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
case "${FORGET_SPLIT}" in forget005) _OFFSET=200 ;; forget05) _OFFSET=191 ;; forget10) _OFFSET=181 ;; *) _OFFSET=199 ;; esac
AUTHOR_OFFSET=${AUTHOR_OFFSET:-${_OFFSET}}           # makes the author numbering of the forget rows equal to `author_id` of the neighbour csv (forget01: 199, forget005: 200, forget05: 191, forget10: 181)

K=${K:-5}                                     # neighbours per author used as the reference
RUV_KS=${RUV_KS:-${K}}                        # the K values r_uv.pt (stage 1) is computed for, e.g. RUV_KS=1,3,5,10 for the K ablation (K must be one of them)
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
# EVAL_PROFILE=fast: the stage-4 evaluation without the metrics that are not used in sweeps (configs/eval/tofu_fast.yaml: no mia_gradnorm / zlib / min_k(++),
# privleak, exact_memorization, forget_quality); the other values are computed as in the full profile. Default: full.
EVAL_PROFILE=${EVAL_PROFILE:-full}
if [ "${EVAL_PROFILE}" = fast ]; then EXTRA_EVAL="eval=tofu_fast ${EXTRA_EVAL}"; elif [ "${EVAL_PROFILE}" != full ]; then echo "EVAL_PROFILE must be full or fast" >&2; exit 1; fi

# TOFU_EVAL.json of the retain model: the reference of forget quality and privleak (None without it)
RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}

TASK_NAME=${TASK_NAME:-single/near_tofu_${MODEL}_${FORGET_SPLIT}}
# where r_uv.pt and the layer selections (stages 1-2) live; give another TASK_NAME to write the W_down runs / evaluations of a variation (e.g.
# EXTRA_W_DOWN="wdown.retain_weight=3") next to, not over, the ones of the same setting:
#   SELECT_TASK_NAME=single/near_tofu_Llama-3.2-1B-Instruct_forget01 TASK_NAME=single/rw3 EXTRA_W_DOWN="wdown.retain_weight=3" ... "3 4"
SELECT_TASK_NAME=${SELECT_TASK_NAME:-${TASK_NAME}}
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
SELECTION=saves/NEAR/${SELECT_TASK_NAME}/k${K}_c$(printf "%g" "${COEFF}")_${MODE}/layer_selection.json
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
    "ruv.ks=[${RUV_KS}]" basic.author_offset=${AUTHOR_OFFSET} \
    task_name=${SELECT_TASK_NAME} ${EXTRA_R_UV}
fi

if has_stage 2; then
  LAYER_ARG="select.layer_fraction=${LAYER_FRACTION}"
  if [ -n "${LAYERS}" ]; then LAYER_ARG="select.layers=${LAYERS} select.layer_fraction=null"; fi
  run "[2] select_layer" python src/METHOD/NEAR/select_layer.py \
    experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    basic.k=${K} basic.coeff=${COEFF} basic.shift_mode=${MODE} ${LAYER_ARG} \
    task_name=${SELECT_TASK_NAME} ${EXTRA_SELECT}
fi

if has_stage 3; then
  # k / coeff / shift_mode / r_uv.pt are read from the layer_selection.json of stage 2
  run "[3] w_down" python src/METHOD/NEAR/w_down.py \
    experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    wdown.layer_selection_path=${SELECTION} wdown.solver=${SOLVER} "wdown.layers=${W_DOWN_LAYER}" \
    task_name=${TASK_NAME} ${EXTRA_W_DOWN}
fi

if has_stage 4; then
  # the edited layer(s): the selected one (W_DOWN_LAYER=auto) or the explicit list, e.g. [14] or [6,7] (folder l6-7, the first is the focus layer)
  if [ "${W_DOWN_LAYER}" != auto ]; then
    LAYERS_EDITED=$(echo "${W_DOWN_LAYER}" | tr -d '[] ')
  elif [ -n "${DRY_RUN}" ]; then
    LAYERS_EDITED='<selected layer>'
  else
    LAYERS_EDITED=$(python -c "import json; print(json.load(open('${SELECTION}'))['selected_layer'])")
  fi
  LAYER_DIR=${LAYERS_EDITED//,/-}     # name of the run folder
  LAYER=${LAYERS_EDITED%%,*}          # focus layer of the activation probe
  EDITED=${CONFIG_DIR}/l${LAYER_DIR}${CF}/model
  TAG=k${K}_l${LAYER_DIR}_c$(printf "%g" "${COEFF}")_${MODE}${CF}
  # ONE evaluation: TOFU metrics + responses + activations (positions / layers / point / author numbering from r_uv.pt, the edited layer
  # is the focus layer). The edited model should behave like the retain model: it has only "seen" the retain set.
  run "[4] eval" python src/eval.py \
    experiment=eval/tofu/probe model=${TARGET_MODEL} model.pretrained.name_or_path=${EDITED} "${SPLITS[@]}" \
    eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH} \
    eval.tofu.metrics.activation_probe.r_uv_path=saves/NEAR/${SELECT_TASK_NAME}/r_uv.pt \
    eval.tofu.metrics.activation_probe.focus_layer=${LAYER} \
    "eval.tofu.metrics.activation_probe.seen=[retain]" \
    "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]" \
    task_name=NEAR/${TASK_NAME}_${TAG} ${EXTRA_EVAL}
fi
echo "################ done: stages ${STAGES} (${CONFIG_DIR}) ################"
