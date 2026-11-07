#!/bin/bash
# TEMPORARY. Does the KL term (stage 2 of W_down, src/METHOD/NEAR/kl_refine.py) help at a SMALL coefficient, where the activations stay close to
# the neighbours (near_line/13: c7 pushes the forget activations ~6 r_UV BEYOND the neighbour mean = the white-box footprint)?
# Grid of  coefficient x KL weight  at ONE fixed layer set (a controlled comparison: only the KL term and the coefficient change; the layers are NOT
# re-selected per coefficient, see near_line/13), `increment` + `sequential` + answer tokens. KL weight `off` = the stage-1 solution only (the control).
#
#   bash scripts/NEAR/sweep_kl.sh [GPU_ID]                       # COEFFS="2 4"  x  KL_WEIGHTS="off 1 3"  x  LAYER_SETS="7,8,9"   (6 runs, finished ones are skipped)
#   COEFFS="2" KL_WEIGHTS="off 1" LAYER_SETS="11,12" bash scripts/NEAR/sweep_kl.sh 1     # does the KL advantage hold on layers closer to the output?
#   KL_WEIGHTS="0 1" ...    # `0` = end-to-end refinement WITHOUT the KL term (separates "more training" from "the KL term")
#   KL_STEPS=300 KL_LR=2e-5 ...         # more refinement steps / another lr of the KL stage (they are part of the run name)
#   FP_WEIGHT=1 ...                     # + the hinge on the probability of the TRUE forget answer (near_line/17: the KL alone does not lower it)
#   EVAL_PROFILE=fast ...               # the fast evaluation profile (configs/eval/tofu_fast.yaml)
#   DRY_RUN=1 bash scripts/NEAR/sweep_kl.sh                      # only print the runs
#
# Read the result: forget_Q_A_Prob / mia_loss / extraction_strength (forgetting; retain model 0.169 / 0.483 / 0.062), model_utility and retain ROUGE (cost),
# ud_wb_unsup (footprint), ud_bb and the first words of forget_responses.json (what a black-box observer sees).
# Output: saves/NEAR/single_v2/rw1_ans[_kl<w>]_increment_seq_c<coeff>_l<layers>/ and saves/eval/NEAR/single_v2/<same>_k5_l<layers>_c<coeff>_all/
cd "$(dirname "$0")/../.." || exit 1

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
read -r -a COEFFS <<< "${COEFFS:-2 4}"
read -r -a KL_WEIGHTS <<< "${KL_WEIGHTS:-off 1 3}"
read -r -a LAYER_SETS <<< "${LAYER_SETS:-7,8,9}"
SKIP_DONE=${SKIP_DONE:-1}
GPU_ID=${1:-0}

export OUT_PREFIX=${OUT_PREFIX:-single_v2}
export SELECT_TASK_NAME=${SELECT_TASK_NAME:-${OUT_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}_ks}   # r_uv.pt (K=5) and the selections of the coefficients
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
    tag=${layers//,/-}
    for kl in "${KL_WEIGHTS[@]}"; do
      if [ "${kl}" = off ]; then kl_env=""; variant=rw1_ans_increment_seq_c${coeff}_l${tag}; else kl_env=${kl}; variant=rw1_ans_kl${kl}${KL_STEPS:+_s${KL_STEPS}}${KL_LR:+_lr${KL_LR}}${FP_WEIGHT:+_fp${FP_WEIGHT}}${FP_TARGET:+_ft${FP_TARGET}}${FP_MODE:+_fm${FP_MODE}}_increment_seq_c${coeff}_l${tag}; fi   # KL_STEPS / KL_LR: passed on to sweep_w_down.sh (same naming)
      done_file=saves/eval/NEAR/${OUT_PREFIX}/${variant}_k5_l${tag}_c${coeff}_all/TOFU_EVAL.json
      if [ "${SKIP_DONE}" = 1 ] && [ -f "${done_file}" ]; then echo "skip (done): ${variant}"; continue; fi
      echo "################  coeff=${coeff} layers=${layers} kl=${kl}  ->  ${variant}  ################"
      [ -n "${DRY_RUN}" ] && continue
      COEFF=${coeff} LAYER=${layers} KL_WEIGHT=${kl_env} bash scripts/NEAR/sweep_w_down.sh "${GPU_ID}" || exit $?
    done
  done
done
