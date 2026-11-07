#!/bin/bash
# ABLATION: is the activation redirection (steering) needed at all? At c1 the steering alone barely moves Prob (0.88 -> 0.86); KL takes it to 0.56 and the hinge on the true
# answer's probability to 0.08 (near_line/19), so the forgetting may be done by KL + hinge. Layers 11-12-13, c1, all with KL 1 and the hinge (mu 1, target `unseen`, mode `match`),
# the full evaluation (also paraphrased / perturbed questions: the hinge trains on the forget answers themselves, does it generalise?).
#   withsteer steering init (stage 1, c1) + forget activation anchor + KL + hinge          = the method
#   noanchor  steering init, but NO forget activation anchor in the refinement (kl.forget_weight 0)
#   nosteer   NO steering at all: original weights, no anchor, KL + hinge only             = "pure KL + hinge + neighbours"
#   hingeonly NO steering, no anchor, NO KL, hinge only                                    = what the neighbours (KL) add to a bounded unlikelihood
#   klonly    NO steering, no anchor, KL only (no hinge)                                  = control: does KL alone move Prob?
# Folders: saves/NEAR/single_v2/<run>/ and saves/eval/NEAR/single_v2/<run>_k5_l11-12-13_c1_all/ with <run> =
#   withsteer rw1_ans_kl1_fp1_ftunseen_fmmatch_increment_seq_c1_l11-12-13
#   noanchor  rw1_ans_kl1_fp1_ftunseen_fmmatch_kfw0_increment_seq_c1_l11-12-13
#   nosteer   rw1_ans_kl1_fp1_ftunseen_fmmatch_initoriginal_kfw0_increment_seq_c1_l11-12-13
#   hingeonly rw1_ans_kl0_fp1_ftunseen_fmmatch_initoriginal_kfw0_increment_seq_c1_l11-12-13
#   klonly    rw1_ans_kl1_initoriginal_kfw0_increment_seq_c1_l11-12-13
# (the variant label is not part of the folder names; the `_c1_all` at the end is the shift mode `all`, not the TOFU full model)
# (c1 only picks the selection / r_UV file; with init=original and anchor 0 the shift is never used.)
#   bash scripts/NEAR/kl_layers/8_ablate_steering.sh 0                         # VARIANTS="withsteer noanchor nosteer hingeonly klonly"
#   VARIANTS="nosteer hingeonly" bash scripts/NEAR/kl_layers/8_ablate_steering.sh 1
cd "$(dirname "$0")/../../.." || exit 1
read -r -a VARIANTS <<< "${VARIANTS:-withsteer noanchor nosteer hingeonly klonly}"
COMMON=(EVAL_PROFILE=${EVAL_PROFILE:-full} COEFFS="${COEFFS:-1}" LAYER_SETS="${LAYER_SETS:-11,12,13}")
for v in "${VARIANTS[@]}"; do
  case "${v}" in
    withsteer) env "${COMMON[@]}" KL_WEIGHTS=1 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match bash scripts/NEAR/sweep_kl.sh "${1:-0}" ;;
    noanchor)  env "${COMMON[@]}" KL_WEIGHTS=1 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 bash scripts/NEAR/sweep_kl.sh "${1:-0}" ;;
    nosteer)   env "${COMMON[@]}" KL_WEIGHTS=1 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 KL_INIT=original bash scripts/NEAR/sweep_kl.sh "${1:-0}" ;;
    hingeonly) env "${COMMON[@]}" KL_WEIGHTS=0 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 KL_INIT=original bash scripts/NEAR/sweep_kl.sh "${1:-0}" ;;
    klonly)    env "${COMMON[@]}" KL_WEIGHTS=1 KL_FORGET_W=0 KL_INIT=original bash scripts/NEAR/sweep_kl.sh "${1:-0}" ;;
    *) echo "unknown variant ${v}" >&2; exit 1 ;;
  esac
done
