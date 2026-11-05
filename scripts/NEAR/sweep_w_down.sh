#!/bin/bash
# W_down variations of ONE setting (K, coeff, mode, layer from stage 2 of scripts/NEAR/single_run.sh): the weight of the retain loss and,
# optionally, training on the answer tokens too. Each variation is stages 3 + 4 (training + one evaluation) and goes to its own folder,
# next to (not over) the run it is compared with:
#     saves/NEAR/single/<VARIANT>/k<K>_c<coeff>_all/l<layer>/        saves/eval/NEAR/single/<VARIANT>_k<K>_l<layer>_c<coeff>_all/
#
#   bash scripts/NEAR/sweep_w_down.sh [GPU_ID]
#   RETAIN_WEIGHTS="3 10" bash scripts/NEAR/sweep_w_down.sh 0                      # default: coeff 7, layer 9
#   COEFF=12 LAYER=6 RETAIN_WEIGHTS="3" bash scripts/NEAR/sweep_w_down.sh 1
#   INCLUDE_ANSWER=1 RETAIN_WEIGHTS="1 3" bash scripts/NEAR/sweep_w_down.sh 0      # prompt AND answer tokens (variants named ..._ans)
#   LAYER=8,9,10 COEFF=7 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS="1" bash scripts/NEAR/sweep_w_down.sh 0   # several layers
#   DRY_RUN=1 bash scripts/NEAR/sweep_w_down.sh                                    # only print the commands
#
# Several layers (LAYER=8,9,10): MULTI_LAYER=same = every layer adds coeff * its whole r_UV (the lower layers' shift is counted again by the
# layers above); MULTI_LAYER=increment = every layer adds its share, so the TOTAL is coeff * r_UV of the last layer (the effect of editing
# that layer alone, spread over several); SEQUENTIAL=1 = a layer is trained on its inputs AFTER the lower edits (and repairs their side effects).
#
# `retain_weight` is the factor of the retain term of the loss  forget_weight * MSE(forget tokens) + retain_weight * MSE(retain tokens):
# larger = the retain / neighbor tokens keep their original MLP output more strictly, the forgetting gets weaker. Compare with the run
# of retain_weight=1 (saves/NEAR/single/near_tofu_<MODEL>_<FORGET_SPLIT>/): forget_quality, forget_Q_A_Prob / ROUGE vs model_utility,
# retain_rougeL_recall (w_down_summary.json has the same numbers on 32 fixed samples).
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
K=${K:-5}
COEFF=${COEFF:-7}
LAYER=${LAYER:-9}
read -r -a RETAIN_WEIGHTS <<< "${RETAIN_WEIGHTS:-3 10}"
INCLUDE_ANSWER=${INCLUDE_ANSWER:-0}
MULTI_LAYER=${MULTI_LAYER:-increment}     # only with several layers: same | increment
SEQUENTIAL=${SEQUENTIAL:-0}               # only with several layers: 1 = train the layers one after the other

# where r_uv.pt and the layer selection of this setting are (stages 1-2 of the default run), and the folder prefix of the variants
# (saves/NEAR/<OUT_PREFIX>/<variant>/, saves/eval/NEAR/<OUT_PREFIX>/<variant>_k.../): e.g. OUT_PREFIX=single_v2 for the runs of the new neighbour protocol
OUT_PREFIX=${OUT_PREFIX:-single}
export SELECT_TASK_NAME=${SELECT_TASK_NAME:-${OUT_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}}
GPU_ID=${1:-0}

for rw in "${RETAIN_WEIGHTS[@]}"; do
  variant=rw${rw}
  extra="wdown.retain_weight=${rw}"
  if [ "${INCLUDE_ANSWER}" = 1 ]; then
    variant=${variant}_ans
    extra="${extra} wdown.include_answer=true"
  fi
  if [[ "${LAYER}" == *,* ]]; then
    extra="${extra} wdown.multi_layer=${MULTI_LAYER}"
    variant=${variant}_${MULTI_LAYER}
    if [ "${SEQUENTIAL}" = 1 ]; then extra="${extra} wdown.sequential=true"; variant=${variant}_seq; fi
  fi
  variant=${variant}_c${COEFF}_l${LAYER//,/-}
  echo "################  ${variant}: ${extra}  ################"
  TASK_NAME=${OUT_PREFIX}/${variant} EXTRA_W_DOWN="${extra}" \
    K=${K} COEFF=${COEFF} MODE=all W_DOWN_LAYER="[${LAYER}]" \
    bash scripts/NEAR/single_run.sh "${GPU_ID}" "3 4" || exit $?
done
