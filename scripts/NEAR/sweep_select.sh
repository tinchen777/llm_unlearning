#!/bin/bash
# Step 2 only (layer selection = INFERENCE-TIME steering, no training), swept over the coefficient and the shift mode:
# does adding coeff * r_UV to the forget prompts change the answers AT ALL? W_down can only reproduce what the steering does, so
# a setting whose steering does nothing (unseen_gap ~ 1) is not worth training W_down for. One run costs the generation of the
# forget answers per candidate layer (a few minutes for 1B), no W_down training.
#
#   bash scripts/NEAR/sweep_select.sh [GPU_ID]
#   COEFFS="3 10" MODES="all" bash scripts/NEAR/sweep_select.sh 0
#   LAYER_FRACTION="[0.4,1.0]" bash scripts/NEAR/sweep_select.sh 0     # a window of relative depths instead of every layer
#   DRY_RUN=1 bash scripts/NEAR/sweep_select.sh                        # only print the commands
#
# Needs the r_uv.pt of stage 1 of the SAME TASK_NAME (default: that of scripts/NEAR/single_run.sh; its K must be in ruv.ks).
# Output: saves/NEAR/${TASK_NAME}/k${K}_c${COEFF}_${MODE}/{layer_selection.json, responses_steer_l<layer>.json}   one folder per setting
# (a run with the same K / coeff / mode overwrites it), then the table of all settings of the task (scripts/NEAR/summarize_sweep.py).
# Next: train W_down for the best setting with  K=.. COEFF=.. MODE=.. bash scripts/NEAR/single_run.sh 0 "3 4".
set -e
cd "$(dirname "$0")/../.." || exit 1

# ============================== parameters ==============================
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-neighbor01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
TASK_NAME=${TASK_NAME:-single/near_tofu_${MODEL}_${FORGET_SPLIT}}

read -r -a KS <<< "${KS_LIST:-5}"                  # neighbours per author (each must be in the ruv.ks of r_uv.pt)
read -r -a COEFFS <<< "${COEFFS:-1 3 10 30}"       # LUNAR's shift is ~8 in norm on the 1B model (coeff 2 * ||r_UV|| ~ 4); NEAR's r_UV is ~0.7
read -r -a MODES <<< "${MODES:-eoi all}"           # eoi = end-of-instruction tokens only | all = every token (the paper)
LAYER_FRACTION=${LAYER_FRACTION:-null}             # null = every layer; [0.4,0.8] = LUNAR's window
LAYERS=${LAYERS:-}                                 # an explicit list instead, e.g. [4,5,6,7,8,9]
EXTRA_SELECT=${EXTRA_SELECT:-}                     # more Hydra overrides, e.g. "select.max_degenerate=0.2"
# ========================================================================

GPU_ID=${1:-0}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "Using GPU: [${GPU_ID}]  task: ${TASK_NAME}  K: ${KS[*]}  coeff: ${COEFFS[*]}  mode: ${MODES[*]}  layers: ${LAYER_FRACTION}"

TARGET_MODEL=tofu/tofu_${MODEL}_full
SPLITS=(forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT})
R_UV=saves/NEAR/${TASK_NAME}/r_uv.pt
if [ -z "${DRY_RUN}" ] && [ ! -f "${R_UV}" ]; then
  echo "${R_UV} not found: run stage 1 first (bash scripts/NEAR/single_run.sh ${GPU_ID} 1)" >&2
  exit 1
fi

for MODE in "${MODES[@]}"; do
  for K in "${KS[@]}"; do
    for COEFF in "${COEFFS[@]}"; do
      echo "############  select_layer | ${MODEL} ${FORGET_SPLIT} | K=${K} COEFF=${COEFF} MODE=${MODE}  ############"
      layer_arg="select.layer_fraction=${LAYER_FRACTION}"
      if [ -n "${LAYERS}" ]; then layer_arg="select.layers=${LAYERS} select.layer_fraction=null"; fi
      cmd=(python src/METHOD/NEAR/select_layer.py
        experiment=unlearn/tofu/NEAR model=${TARGET_MODEL} "${SPLITS[@]}"
        basic.k=${K} basic.coeff=${COEFF} basic.shift_mode=${MODE}
        ${layer_arg}
        task_name=${TASK_NAME} ${EXTRA_SELECT})
      if [ -n "${DRY_RUN}" ]; then echo "${cmd[@]}"; else "${cmd[@]}"; fi
    done
  done
done

if [ -z "${DRY_RUN}" ]; then
  python scripts/NEAR/summarize_sweep.py saves/NEAR/${TASK_NAME}
  python scripts/NEAR/summarize_sweep.py saves/NEAR/${TASK_NAME} --top 12
  echo "per layer: python scripts/NEAR/summarize_sweep.py saves/NEAR/${TASK_NAME} --layers"
fi
