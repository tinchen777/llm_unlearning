#!/bin/bash
# The whole NEAR chain in one go: r_UV -> layer selection -> W_down -> eval (1_r_uv.sh .. 4_eval.sh), in this order.
# All settings (model, splits, K, coeff, shift modes, solver, task name) are in _common.sh and shared by every step.
#
#   bash scripts/NEAR/run_all.sh [GPU_ID] ["STAGES"]
#   bash scripts/NEAR/run_all.sh 0              # stages 1 2 3 4 on GPU 0
#   bash scripts/NEAR/run_all.sh 0 "3 4"        # only W_down + eval (r_uv.pt and the layer selections already exist)
#   DRY_RUN=1 bash scripts/NEAR/run_all.sh      # only print what would run
#
# Stops at the first failing step (set -e), so a later step never reads the outputs of a failed one.
# NOT included: 0_probe.sh / 0_probe_compare.sh (analysis of several models, own model / split lists).
# 4_eval.sh needs the TOFU_EVAL.json of the retain model (RETAIN_LOGS in 4_eval.sh) for forget quality / privleak.

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

GPU_ID=${1:-0}
STAGES=${2:-"1 2 3 4"}

declare -A STAGE_SCRIPT=(
  [1]=1_r_uv.sh
  [2]=2_select_layer.sh
  [3]=3_w_down.sh
  [4]=4_eval.sh
)

for stage in ${STAGES}; do
  script=${STAGE_SCRIPT[${stage}]}
  if [ -z "${script}" ]; then
    echo "unknown stage '${stage}' (valid: 1 2 3 4)" >&2
    exit 1
  fi
  echo "################ stage ${stage}: ${script} (GPU ${GPU_ID}) ################"
  if [ -n "${DRY_RUN}" ]; then
    echo "bash ${SCRIPT_DIR}/${script} ${GPU_ID}"
  else
    bash "${SCRIPT_DIR}/${script}" "${GPU_ID}"
  fi
done
echo "################ done: stages ${STAGES} ################"
