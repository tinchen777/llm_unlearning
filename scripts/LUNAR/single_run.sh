#!/bin/bash
# ONE LUNAR experiment, steps 1-4, with every parameter in the block below (run.sh = many models x splits x modes).
#   r_UV -> layer selection -> W_down -> eval of the edited model
#
#   bash scripts/LUNAR/single_run.sh [GPU_ID] ["STAGES"]
#   bash scripts/LUNAR/single_run.sh 0                 # stages 1 2 3 4
#   bash scripts/LUNAR/single_run.sh 0 "3 4"           # only W_down + eval (r_uv.pt and layer_selection.json already exist)
#   MODEL=Llama-3.2-3B-Instruct COEFF=2.0 MODE=all bash scripts/LUNAR/single_run.sh 0   # every parameter can come from the environment
#   DRY_RUN=1 bash scripts/LUNAR/single_run.sh         # only print the commands
#
# Output: saves/LUNAR/${TASK_NAME}/c${COEFF}_${MODE}/   (layer_selection.json, l<layer>[_cf]/model, ...)
#         saves/eval/LUNAR/${TASK_NAME}_<tag>/          (TOFU_EVAL.json, ...)
# Needs: python setup/setup_data.py --lunar (the reference set), and for forget quality / privleak the TOFU_EVAL.json of the retain model.
# The forget set is FORGET_SPLIT: forget01 (2 authors) = what NEAR unlearns, so the two are comparable; forget005 = ONE author (author 200 = the last 20 rows
# of forget01; there is no retain995 model, so forget quality / privleak of that run are not meaningful):
#   bash scripts/LUNAR/single_run.sh 0                                   # forget01
#   FORGET_SPLIT=forget005 bash scripts/LUNAR/single_run.sh 0            # one author, trained AND evaluated on forget005
# The edited model is evaluated with the SAME evaluation as NEAR (experiment=eval/tofu/probe: TOFU metrics + responses + activations +
# detectability), so the two are comparable column by column.
set -e
cd "$(dirname "$0")/../.." || exit 1

# ============================== parameters of the experiment ==============================
MODEL=${MODEL:-Llama-3.2-1B-Instruct}         # base model; the target is the official TOFU full model of it (configs/model/tofu/)
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-${FORGET_SPLIT/forget/neighbor}}   # only for the evaluation (the held-out neighbours: neighbor_*, probe, detectability)
case "${FORGET_SPLIT}" in forget005) _OFFSET=200 ;; forget05) _OFFSET=191 ;; forget10) _OFFSET=181 ;; *) _OFFSET=199 ;; esac
AUTHOR_OFFSET=${AUTHOR_OFFSET:-${_OFFSET}}          # author numbering of the forget rows = `author_id` of the neighbour csv (forget01: 199 = its first author, forget005: 200 = its last author, forget05: 191, forget10: 181)
FORGET_HF_SPLIT=${FORGET_HF_SPLIT:-train}           # the HF split of the forget data (all of FORGET_SPLIT)
REF_DATASET=${REF_DATASET:-LUNAR/LUNAR_harmful}     # D_ref: LUNAR/LUNAR_harmful | LUNAR/LUNAR_unverified

COEFF=${COEFF:-2.0}                           # forget target = original MLP output + COEFF * r_UV (LUNAR README: 2.0)
MODE=${MODE:-all}                             # all = the paper (r_UV on every token) | eoi = end-of-instruction tokens only
SOLVER=${SOLVER:-adam}                        # adam | closed_form

LAYER_FRACTION=${LAYER_FRACTION:-[0.4,0.8]}   # candidate layers of the selection (relative depth)
LAYERS=${LAYERS:-}                            # empty: use LAYER_FRACTION; or an explicit list, e.g. [4,5,6]
W_DOWN_LAYER=${W_DOWN_LAYER:-auto}            # auto = the selected layer; or [16]
# extra Hydra overrides of a stage, e.g. EXTRA_W_DOWN="wdown.lr=1e-2 wdown.epochs=40"
EXTRA_R_UV=${EXTRA_R_UV:-}
EXTRA_SELECT=${EXTRA_SELECT:-}
EXTRA_W_DOWN=${EXTRA_W_DOWN:-}
EXTRA_EVAL=${EXTRA_EVAL:-}

# TOFU_EVAL.json of the retain model: the reference of forget quality and privleak (None without it)
RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}

TASK_NAME=${TASK_NAME:-single/lunar_tofu_${MODEL}_${FORGET_SPLIT}_$(basename "${REF_DATASET}")}
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
# forget_hf_split contains `[` / `:`: Hydra only accepts it quoted (an array keeps the quotes through word splitting)
SPLITS=(forget_split=${FORGET_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}
        "forget_hf_split='${FORGET_HF_SPLIT}'" "data/datasets@data.ref.dataset=${REF_DATASET}")
# folders: same rules as config_dir() / run_dir() / w_down_tag() in scripts/LUNAR/_common.sh
CONFIG_DIR=saves/LUNAR/${TASK_NAME}/c$(printf "%g" "${COEFF}")_${MODE}
SELECTION=${CONFIG_DIR}/layer_selection.json
CF=""; if [ "${SOLVER}" = closed_form ]; then CF=_cf; fi

run() {   # run "<name>" cmd...
  local name=$1; shift
  echo "############  START STAGE ${name} | ${MODEL} ${FORGET_SPLIT} ${REF_DATASET} | COEFF=${COEFF} MODE=${MODE} SOLVER=${SOLVER}  ############"
  if [ -n "${DRY_RUN}" ]; then echo "$@"; else "$@"; fi
  echo "############  END STAGE ${name}  ############"
}

has_stage() { [[ " ${STAGES} " == *" $1 "* ]]; }

if has_stage 4 && [ ! -f "${RETAIN_LOGS_PATH}" ]; then
  echo "WARNING: ${RETAIN_LOGS_PATH} not found: forget_quality and privleak will be None (RETAIN_LOGS_PATH=... to change it)" >&2
fi

if has_stage 1; then
  # r_UV = mean(D_ref) - mean(D_forget) of every layer: saves/LUNAR/${TASK_NAME}/r_uv.pt
  run "[1] r_uv" python src/METHOD/LUNAR/r_uv.py \
    experiment=unlearn/tofu/LUNAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    task_name=${TASK_NAME} ${EXTRA_R_UV}
fi

if has_stage 2; then
  LAYER_ARG="select.layer_fraction=${LAYER_FRACTION}"
  if [ -n "${LAYERS}" ]; then LAYER_ARG="select.layers=${LAYERS} select.layer_fraction=null"; fi
  run "[2] select_layer" python src/METHOD/LUNAR/select_layer.py \
    experiment=unlearn/tofu/LUNAR model=${TARGET_MODEL} "${SPLITS[@]}" \
    basic.coeff=${COEFF} basic.shift_mode=${MODE} ${LAYER_ARG} \
    task_name=${TASK_NAME} ${EXTRA_SELECT}
fi

if has_stage 3; then
  # coeff / shift_mode / r_uv.pt are read from the layer_selection.json of stage 2
  run "[3] w_down" python src/METHOD/LUNAR/w_down.py \
    experiment=unlearn/tofu/LUNAR model=${TARGET_MODEL} "${SPLITS[@]}" \
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
  TAG=c$(printf "%g" "${COEFF}")_l${LAYER_DIR}_${MODE}${CF}
  # ONE evaluation, the same as NEAR's: TOFU metrics + responses (forget / retain / holdout / held-out neighbours) + activations + detectability.
  # An edited model should behave like the retain model: seen = [retain]. (LUNAR has no r_uv.pt of the NEAR kind: positions / layers / point
  # of the probe are its defaults, those of the baselines.)
  run "[4] eval" python src/eval.py \
    experiment=eval/tofu/probe model=${TARGET_MODEL} model.pretrained.name_or_path=${EDITED} \
    forget_split=${FORGET_SPLIT} holdout_split=${HOLDOUT_SPLIT} retain_split=${RETAIN_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} \
    eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH} \
    eval.tofu.metrics.activation_probe.author_offset=${AUTHOR_OFFSET} \
    eval.tofu.metrics.activation_probe.focus_layer=${LAYER} \
    "eval.tofu.metrics.activation_probe.seen=[retain]" \
    "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]" \
    task_name=LUNAR/${TASK_NAME}_${TAG} ${EXTRA_EVAL}
fi
echo "################ done: stages ${STAGES} (${CONFIG_DIR}) ################"
