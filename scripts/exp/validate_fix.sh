#!/bin/bash
# Minimal validation of the two fixes for the sampling-level traces of NEAR-Match (mechanism -> fix -> validation, paper_notes/8 §2.3, near_line/25):
#   (a) the neighbour NAMES are distilled into the student      -> mask_neighbor_name (KL_MASK_NAME=1)
#   (b) the student fits the MIXTURE of the neighbours            -> neighbors_per_question (KL_NPQ=m: m fixed neighbours per forget question)
#   (c) the per-sample answer probability is pulled to ONE value  -> forget_prob_mode neighbor (per-sample level) instead of match
# For every variant: train + the standard evaluation (scripts/NEAR/sweep_kl.sh), then ONE sampling evaluation of all of them (scripts/analysis/sample_consistency.py) and the
# spread of the answer probabilities (scripts/analysis/prob_spread.py); a table is printed and written to saves/analysis/validate_<SETTING>/summary.txt.
#
#   bash scripts/exp/validate_fix.sh <GPU>                       # SETTING=1B_f01, the 6 variants of the minimal design
#   SETTING=3B_f01 VARIANTS="E5" bash scripts/exp/validate_fix.sh <GPU>     # confirm one variant at another setting (3B_f01 | 1B_f05 | 1B_f01)
#   NO_SAMPLE=1 VARIANTS="E3" bash scripts/exp/validate_fix.sh <GPU>   # only train + evaluate this variant (run the variants on several GPUs in parallel, then the line below)
#   SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh <GPU>          # only the sampling evaluation + table (the runs exist)
#   DRY_RUN=1 bash scripts/exp/validate_fix.sh                   # print the commands
# Variants (FP_MODE | neighbours per question | mask name):
#   R0 match|all|off  the current method (not trained again: its run exists)   E1 match|all|on   E2 match|1|off   E3 match|1|on
#   E4 neighbor|all|off (calibration alone)   E5 neighbor|1|on   E6 neighbor|3|on
set -e
cd "$(dirname "$0")/../.." || exit 1
GPU=${1:-0}
SETTING=${SETTING:-1B_f01}
VARIANTS=${VARIANTS:-"E1 E2 E3 E4 E5 E6"}
case "${SETTING}" in
  1B_f01) MODEL=Llama-3.2-1B-Instruct; F=forget01; H=holdout01; R=retain99; NB=neighbor01; UP=single_v2; OUT=single_v3; C=1; L=11,12,13; STEPS=; EXTRA=; MH=Llama-3.2-1B-Instruct ;;
  3B_f01) MODEL=Llama-3.2-3B-Instruct; F=forget01; H=holdout01; R=retain99; NB=neighbor01; UP=Llama3B_v2; OUT=Llama3B_v3; C=6; L=19,20,21; STEPS=; EXTRA="wdown.kl.fr_batch_size=16"; MH=Llama-3.2-3B-Instruct ;;
  1B_f05) MODEL=Llama-3.2-1B-Instruct; F=forget05; H=holdout05; R=retain95; NB=neighbor05; UP=forget05_Llama1B_v2; OUT=forget05_Llama1B_v3; C=7; L=11,12,13; STEPS=300; EXTRA="wdown.kl.batch_size=16"; MH=Llama-3.2-1B-Instruct ;;
  *) echo "SETTING must be 1B_f01 | 3B_f01 | 1B_f05" >&2; exit 1 ;;
esac
declare -A MODE_OF=([R0]=match [E1]=match [E2]=match [E3]=match [E4]=neighbor [E5]=neighbor [E6]=neighbor)
declare -A NPQ_OF=([R0]= [E1]= [E2]=1 [E3]=1 [E4]= [E5]=1 [E6]=3)
declare -A MASK_OF=([R0]=0 [E1]=1 [E2]=0 [E3]=1 [E4]=0 [E5]=1 [E6]=1)

# the name of the run folder (must equal the one sweep_w_down.sh / sweep_kl.sh build)
run_name() {  # $1 = variant
  local v=$1 n="rw1_ans_kl1${STEPS:+_s${STEPS}}_fp1_ftunseen_fm${MODE_OF[$v]}_initoriginal_kfw0"
  [ -n "${NPQ_OF[$v]}" ] && n+="_npq${NPQ_OF[$v]}"
  [ "${MASK_OF[$v]}" = 1 ] && n+="_mn"
  echo "${n}_increment_seq_c${C}_l${L//,/-}"
}
export FORGET_SPLIT=${F} RETAIN_SPLIT=${R} HOLDOUT_SPLIT=${H} NEIGHBOR_SPLIT=${NB} MODEL
if [ -z "${SKIP_TRAIN}" ]; then
  for v in ${VARIANTS}; do
    echo "################  ${SETTING} ${v}: calibration=${MODE_OF[$v]} neighbours/question=${NPQ_OF[$v]:-all} mask_name=${MASK_OF[$v]}  ################"
    cmd="OUT_PREFIX=${OUT} UPSTREAM_PREFIX=${UP} ${STEPS:+KL_STEPS=${STEPS}} ${EXTRA:+EXTRA_WDOWN=\"${EXTRA}\"} FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=${MODE_OF[$v]} KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 ${NPQ_OF[$v]:+KL_NPQ=${NPQ_OF[$v]}} $([ "${MASK_OF[$v]}" = 1 ] && echo KL_MASK_NAME=1) EVAL_PROFILE=fast COEFFS=${C} LAYER_SETS=${L} bash scripts/NEAR/sweep_kl.sh ${GPU}"
    echo "${cmd}"
    [ -n "${DRY_RUN}" ] || eval "${cmd}"
  done
fi

if [ -n "${NO_SAMPLE}" ]; then exit 0; fi
# sampling evaluation of R0 (the current run) and every variant, and the answer-probability spread
OUTDIR=saves/analysis/validate_${SETTING}
specs=(); spread=()
for v in R0 ${VARIANTS}; do
  specs+=(--model "${v}=near:saves/NEAR/${OUT}/$(run_name "$v")/k5_c${C}_all/l${L//,/-}")
  spread+=("${v}=saves/eval/NEAR/${OUT}/$(run_name "$v")_k5_l${L//,/-}_c${C}_all")
done
cmd=(python scripts/analysis/sample_consistency.py --gpu "${GPU}" --out "${OUTDIR}" --base "open-unlearning/tofu_${MH}_full" --forget "${F}" --holdout "${H}"
     --model "retain=hf:open-unlearning/tofu_${MH}_${R}" "${specs[@]}")
echo "${cmd[*]}"
if [ -z "${DRY_RUN}" ]; then
  mkdir -p "${OUTDIR}"
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "${cmd[@]}"
  {
    echo; echo "== answer-probability spread (forget / holdout)"
    python scripts/analysis/prob_spread.py "retain=saves/eval/retain/${MH}/${F}" "${spread[@]}"
    echo; echo "== standard evaluation (Prob / mia / extraction / utility / retain generation / ud_bb / wb_unsup)"
    python - "${spread[@]}" <<'PY'
import json, sys
for a in sys.argv[1:]:
    label, _, folder = a.partition("=")
    try:
        d = json.load(open(folder + "/TOFU_SUMMARY.json"))
    except OSError:
        print(f"{label}: no evaluation"); continue
    g = lambda k: d.get(k, float("nan"))
    print(f"{label:4s} {g('forget_Q_A_Prob'):.3f} / {g('mia_loss'):.3f} / {g('extraction_strength'):.3f} / {g('model_utility'):.3f} / {g('retain_rougeL_recall'):.3f} / {g('ud_bb'):.3f} / {g('ud_wb_unsup'):.3f}")
PY
  } | tee "${OUTDIR}/extra.txt"
  cat "${OUTDIR}/summary.md" | tee "${OUTDIR}/summary.txt" >/dev/null
fi
