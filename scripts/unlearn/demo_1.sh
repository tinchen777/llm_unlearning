#!/bin/bash

set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# source "${SCRIPT_DIR}/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

method_args() {
  case "$1" in
    IdkDPO)
      # DPO with "I don't know" answers
      echo "experiment=unlearn/tofu/idk trainer=DPO trainer.args.eval_on_start=False"
      ;;

    PDU)
      # retain_loss_eps has no default
      echo "experiment=unlearn/tofu/default trainer=PDU trainer.method_args.retain_loss_eps=0.3"
      ;;

    GradAscent|GradDiff|NPO|SimNPO|RMU|CEU|WGA|UNDIAL|SatImp|BoundedGradDiff)
      echo "experiment=unlearn/tofu/default trainer=$1"
      ;;

    *)
      echo "unknown method '$1'" >&2
      return 1
      ;;
  esac
}

METHODS=(
    GradAscent
    # GradDiff
    # NPO
    # SimNPO
    # IdkDPO
    # RMU
)
MODEL=Llama-3.2-1B-Instruct
FULL_MODEL=tofu/tofu_${MODEL}_full

FORGET_SPLIT=forget01
HOLDOUT_SPLIT=holdout01
RETAIN_SPLIT=retain99

SPLIT_ARGS=(
  "forget_split=${FORGET_SPLIT}"
  "retain_split=${RETAIN_SPLIT}"
  "holdout_split=${HOLDOUT_SPLIT}"
)
RETAIN_LOGS=saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

for METHOD in "${METHODS[@]}"; do
  echo start ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
  python src/train.py --config-name=unlearn \
    $(method_args ${METHOD}) \
    model=${FULL_MODEL} \
    "${SPLIT_ARGS[@]}" \
    retain_logs_path=${RETAIN_LOGS} \
    "${EXTRA_ARGS[@]}" \
    task_name=ttestt/${METHOD}_1 \
    # --cfg job --resolve
  echo end ${METHOD} ${FULL_MODEL} ${FORGET_SPLIT}
done



FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.2-3B-Instruct TASK_NAME=forget10_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget10_ks COEFFS="14" MODES="all" LAYER_FRACTION="[0.25,0.75]" bash scripts/NEAR/sweep_select.sh 0 && FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=forget10_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget10_ks OUT_PREFIX=forget10_Llama3B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=14 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh 0


FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.1-8B-Instruct TASK_NAME=forget10_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget10_ks COEFFS="7 8 9" MODES="all" LAYERS="[15,16,17,18,19,20]" bash scripts/NEAR/sweep_select.sh 1 && for c in 7 8 9; do L=$(python -c "import json;s=json.load(open('saves/NEAR/forget10_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget10_ks/k5_c${c}_all/layer_selection.json'))['selected_layer'];print(f'{s-2},{s-1},{s}')"); FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="wdown.gen_batch_size=8" SELECT_TASK_NAME=forget10_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget10_ks OUT_PREFIX=forget10_Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=$L COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh 1; done
