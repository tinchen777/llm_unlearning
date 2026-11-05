#!/bin/bash
# Does a SMALLER shift spread over MORE layers forget as well, with a smaller activation footprint? (near_line/9: the white-box outlier score `ud_wb_unsup`
# of the edited models is ~0.99 at coefficient 7: the forget activations are all pushed the same way and form a cluster away from the benign ones.)
# Grid of  coefficient x layer set  (all of them `increment` + `sequential` + answer tokens, the best configuration so far), each = W_down training +
# ONE evaluation (TOFU metrics + responses + activations + detectability). The coefficients of the steering sweep of `${BASE}_ks` are available (K=5:
# 2 4 5 6 7 8 12), r_uv.pt and the selection files are read from there.
#
#   bash scripts/NEAR/sweep_footprint.sh [GPU_ID]                          # COEFFS="4 5 6"  x  LAYER_SETS="7,8,9 6,7,8,9 5,6,7,8,9"  (9 runs)
#   COEFFS="4" LAYER_SETS="6,7,8,9" bash scripts/NEAR/sweep_footprint.sh 1
#   DRY_RUN=1 bash scripts/NEAR/sweep_footprint.sh
#
# Read the result: forget_Q_A_Prob / mia_loss / extraction_strength (forgetting: the retain model has 0.169 / 0.483 / 0.062), model_utility and the
# retain ROUGE (the cost), ud_wb_unsup / ud_wb_sup / activation_anom_auc_forget_benign_max (the footprint), ud_bb (what a black-box observer sees).
# Output: saves/NEAR/single_v2/rw1_ans_increment_seq_c<coeff>_l<layers>/ and saves/eval/NEAR/single_v2/<same>_k5_l<layers>_c<coeff>_all/
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
read -r -a COEFFS <<< "${COEFFS:-4 5 6}"
read -r -a LAYER_SETS <<< "${LAYER_SETS:-7,8,9 6,7,8,9 5,6,7,8,9}"
GPU_ID=${1:-0}

export OUT_PREFIX=${OUT_PREFIX:-single_v2}
export SELECT_TASK_NAME=${SELECT_TASK_NAME:-${OUT_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}_ks}   # r_uv.pt (K=5) and the selections of those coefficients
export MULTI_LAYER=increment SEQUENTIAL=1 INCLUDE_ANSWER=1 RETAIN_WEIGHTS=1
if [ -z "${DRY_RUN}" ] && [ ! -f "saves/NEAR/${SELECT_TASK_NAME}/r_uv.pt" ]; then
  echo "saves/NEAR/${SELECT_TASK_NAME}/r_uv.pt not found (run the K ablation / steering sweep of that task first)" >&2
  exit 1
fi

for coeff in "${COEFFS[@]}"; do
  if [ -z "${DRY_RUN}" ] && [ ! -f "saves/NEAR/${SELECT_TASK_NAME}/k5_c${coeff}_all/layer_selection.json" ]; then
    echo "no selection for coefficient ${coeff} in ${SELECT_TASK_NAME} (have: $(ls saves/NEAR/${SELECT_TASK_NAME} | grep -o 'c[0-9.]*_all' | tr '\n' ' '))" >&2
    continue
  fi
  for layers in "${LAYER_SETS[@]}"; do
    echo "################  coeff=${coeff} layers=${layers}  ################"
    COEFF=${coeff} LAYER=${layers} bash scripts/NEAR/sweep_w_down.sh "${GPU_ID}" || exit $?
  done
done
