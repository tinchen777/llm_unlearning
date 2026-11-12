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
ANSWER_COEFF=${ANSWER_COEFF:-}            # only with INCLUDE_ANSWER=1: factor on COEFF for the answer tokens (e.g. 0.5); empty = same as the prompt
KL_WEIGHT=${KL_WEIGHT:-}               # non-empty: stage 2 (end-to-end refinement with the KL term, needs INCLUDE_ANSWER=1), e.g. KL_WEIGHT=1; KL_STEPS / KL_LR optional
KL_STEPS=${KL_STEPS:-}
KL_LR=${KL_LR:-}
FP_WEIGHT=${FP_WEIGHT:-}                 # with KL_WEIGHT: mu of the hinge on the probability of the TRUE forget answer (wdown.kl.forget_prob_weight), FP_TARGET (default 0.17) its level
FP_TARGET=${FP_TARGET:-}                 # a number or `unseen` (= the original model's probability on the holdout answers, the default of the config)
KL_INIT=${KL_INIT:-}                     # stage1 (default) | original: NO steering stage (ablation, near_line/20)
KL_FORGET_W=${KL_FORGET_W:-}             # forget activation-MSE weight in the refinement (0 = no steering anchor)
FP_MODE=${FP_MODE:-}                     # hinge (default) | match | mean | neighbor
KL_NPQ=${KL_NPQ:-}                       # neighbours per forget question in the KL (wdown.kl.neighbors_per_question; empty = all, a random one per step)
KL_MASK_NAME=${KL_MASK_NAME:-}           # 1: leave the neighbour-name tokens out of the KL (wdown.kl.mask_neighbor_name)
RKL_WEIGHT=${RKL_WEIGHT:-}               # retain-answer KL weight (wdown.kl.retain_kl_weight; against the global likelihood drift, 2026-11-12), variant _rkl<w>
DYN_EVERY=${DYN_EVERY:-}                 # dynamic forget target every N steps (wdown.kl.forget_prob_dynamic_every), variant _dyn<N>
DYN_REF=${DYN_REF:-}                     # its unseen reference split (wdown.kl.forget_prob_dynamic_ref, default neighbor_heldout), variant _dr<split>
MULTI_LAYER=${MULTI_LAYER:-increment}     # only with several layers: same | increment
SEQUENTIAL=${SEQUENTIAL:-0}               # only with several layers: 1 = train the layers one after the other

# where r_uv.pt and the layer selection of this setting are (stages 1-2 of the default run), and the folder prefix of the variants
# (saves/NEAR/<OUT_PREFIX>/<variant>/, saves/eval/NEAR/<OUT_PREFIX>/<variant>_k.../): e.g. OUT_PREFIX=single_v2 for the runs of the new neighbour protocol
OUT_PREFIX=${OUT_PREFIX:-single}
# no SELECT_TASK_NAME given: the folder of the default run (saves/NEAR/<OUT_PREFIX>/near_tofu_<MODEL>_<FORGET>) or, if it has no selection for this
# K / COEFF, that of the steering sweep (..._ks, which holds the selections of the other coefficients)
if [ -z "${SELECT_TASK_NAME:-}" ]; then
  SELECT_TASK_NAME=${OUT_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}
  sel_file() { echo "saves/NEAR/$1/k${K}_c$(printf "%g" "${COEFF}")_all/layer_selection.json"; }
  if [ ! -f "$(sel_file "${SELECT_TASK_NAME}")" ] && [ -f "$(sel_file "${SELECT_TASK_NAME}_ks")" ]; then
    echo "no selection for K=${K} COEFF=${COEFF} in ${SELECT_TASK_NAME}: using ${SELECT_TASK_NAME}_ks" >&2
    SELECT_TASK_NAME=${SELECT_TASK_NAME}_ks
  fi
fi
export SELECT_TASK_NAME
GPU_ID=${1:-0}

for rw in "${RETAIN_WEIGHTS[@]}"; do
  variant=rw${rw}
  extra="wdown.retain_weight=${rw}"
  if [ "${INCLUDE_ANSWER}" = 1 ]; then
    variant=${variant}_ans
    extra="${extra} wdown.include_answer=true"
    if [ -n "${ANSWER_COEFF}" ]; then extra="${extra} wdown.answer_coeff=${ANSWER_COEFF}"; variant=${variant}_ac${ANSWER_COEFF}; fi
  fi
  if [ -n "${KL_WEIGHT}" ]; then
    extra="${extra} wdown.kl.enabled=true wdown.kl.weight=${KL_WEIGHT}"
    variant=${variant}_kl${KL_WEIGHT}
    if [ -n "${KL_STEPS}" ]; then extra="${extra} wdown.kl.steps=${KL_STEPS}"; variant=${variant}_s${KL_STEPS}; fi
    if [ -n "${KL_LR}" ]; then extra="${extra} wdown.kl.lr=${KL_LR}"; variant=${variant}_lr${KL_LR}; fi
    if [ -n "${FP_WEIGHT}" ]; then extra="${extra} wdown.kl.forget_prob_weight=${FP_WEIGHT}"; variant=${variant}_fp${FP_WEIGHT}; fi
    if [ -n "${FP_TARGET}" ]; then extra="${extra} wdown.kl.forget_prob_target=${FP_TARGET}"; variant=${variant}_ft${FP_TARGET}; fi
    if [ -n "${FP_MODE}" ]; then extra="${extra} wdown.kl.forget_prob_mode=${FP_MODE}"; variant=${variant}_fm${FP_MODE}; fi
    if [ -n "${KL_INIT}" ]; then extra="${extra} wdown.kl.init=${KL_INIT}"; variant=${variant}_init${KL_INIT}; fi
    if [ -n "${KL_FORGET_W}" ]; then extra="${extra} wdown.kl.forget_weight=${KL_FORGET_W}"; variant=${variant}_kfw${KL_FORGET_W}; fi
    if [ -n "${KL_NPQ}" ]; then extra="${extra} wdown.kl.neighbors_per_question=${KL_NPQ}"; variant=${variant}_npq${KL_NPQ}; fi
    if [ "${KL_MASK_NAME}" = 1 ]; then extra="${extra} wdown.kl.mask_neighbor_name=true"; variant=${variant}_mn; fi
    if [ -n "${RKL_WEIGHT}" ]; then extra="${extra} wdown.kl.retain_kl_weight=${RKL_WEIGHT}"; variant=${variant}_rkl${RKL_WEIGHT}; fi
    if [ -n "${DYN_EVERY}" ]; then extra="${extra} wdown.kl.forget_prob_dynamic_every=${DYN_EVERY}"; variant=${variant}_dyn${DYN_EVERY}; fi
    if [ -n "${DYN_REF}" ]; then extra="${extra} wdown.kl.forget_prob_dynamic_ref=${DYN_REF}"; variant=${variant}_dr${DYN_REF}; fi
  fi
  if [[ "${LAYER}" == *,* ]]; then
    extra="${extra} wdown.multi_layer=${MULTI_LAYER}"
    variant=${variant}_${MULTI_LAYER}
    if [ "${SEQUENTIAL}" = 1 ]; then extra="${extra} wdown.sequential=true"; variant=${variant}_seq; fi
  fi
  extra="${extra} ${EXTRA_WDOWN:-}"      # more Hydra overrides of the W_down stage (not part of the run name), e.g. EXTRA_WDOWN="wdown.kl.fr_batch_size=16"
  variant=${variant}_c${COEFF}_l${LAYER//,/-}
  echo "################  ${variant}: ${extra}  ################"
  TASK_NAME=${OUT_PREFIX}/${variant} EXTRA_W_DOWN="${extra}" \
    K=${K} COEFF=${COEFF} MODE=all W_DOWN_LAYER="[${LAYER}]" \
    bash scripts/NEAR/single_run.sh "${GPU_ID}" "3 4" || exit $?
done
