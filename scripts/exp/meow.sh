#!/bin/bash
# MEOW baseline (src/METHOD/MEOW/meow.py) + the standard evaluation of this repository of EVERY checkpoint, one setting, one GPU.
#   MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 bash scripts/exp/meow.sh <GPU>                       # MEOW (the released inverted facts), forget01
#   MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 TARGET=neighbor bash scripts/exp/meow.sh <GPU>      # the control MEOW -> neighbour (labels = the train neighbours' answers, same mechanics)
# Data: the authors' released files in data/meow/ (git-ignored), the layout of their repository without the leading data/tofu:
#       data/meow/generated_data/<split>_perturbed_generated.json   (the inverted facts; the only file read by default)
#       data/meow/memo_data/ , data/meow/tofu_data/                  (not needed unless MEMO_JSON points into memo_data)
# Env: MODEL / FORGET_SPLIT / NEIGHBOR_SPLIT / RETAIN_SPLIT / HOLDOUT_SPLIT / AUTHOR_OFFSET (scripts/NEAR/_common.sh), R_UV_PREFIX (saves/NEAR/<prefix>/near_tofu_<MODEL>_<split>_ks/r_uv.pt: the activation probe
#      of the evaluation reads it), TARGET=inverted|neighbor, K (answers per question; default 2 / 3 / 6 for forget01 / 05 / 10: the paper's Llama-2 values), SELECT=largest|smallest|random (largest),
#      SAVE_STEPS (checkpoints; one step = 32 rows = 4 questions repeated 8 times), LR (1e-5), KEEP_STEPS (steps whose weights are kept after their evaluation; default: none, the evaluation
#      stays), MEMO_JSON (use this file of MEMO scores, e.g. the released one of Llama-2, instead of computing them with MODEL), LOWMEM=1 (gradient checkpointing, 2 rows per micro-batch: 8B),
#      FOCUS_LAYER (probe layer; default half of the depth), TAG, EVAL_PROFILE=fast|full, SKIP_TRAIN=1 (only evaluate the existing checkpoints).
# Output: saves/MEOW/<name>/checkpoint-<step>/ (weights removed after the evaluation unless kept), saves/eval/MEOW/<name>_s<step>/ (TOFU_SUMMARY.json, responses, activations, ...).
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/../NEAR/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
TARGET=${TARGET:-inverted}
SELECT=${SELECT:-largest}
LR=${LR:-1e-5}
MEOW_DIR=${MEOW_DIR:-data/meow}
case "${FORGET_SPLIT}" in
  forget01) K_DEFAULT=2; STEPS_DEFAULT="5 10 20 40 80" ;;
  forget05) K_DEFAULT=3; STEPS_DEFAULT="10 25 50 100 200" ;;
  forget10) K_DEFAULT=6; STEPS_DEFAULT="25 50 100 200 400" ;;
  *) echo "FORGET_SPLIT must be forget01 | forget05 | forget10" >&2; exit 1 ;;
esac
K=${K:-${K_DEFAULT}}
read -r -a STEPS_A <<< "${SAVE_STEPS:-${STEPS_DEFAULT}}"
read -r -a KEEP_A <<< "${KEEP_STEPS:-}"
NB=""; if [ "${TARGET}" = neighbor ]; then NB="_nb"; fi
NAME=${MODEL}_${FORGET_SPLIT}_meow${NB}_k${K}_${SELECT}${TAG}
OUT=saves/MEOW/${NAME}
MEMO_JSON=${MEMO_JSON:-saves/MEOW/memo/${MODEL}_${FORGET_SPLIT}_memo.json}
case "${MODEL}" in
  *1B*) FOCUS_LAYER=${FOCUS_LAYER:-8} ;;
  *3B*) FOCUS_LAYER=${FOCUS_LAYER:-14} ;;
  *8B*) FOCUS_LAYER=${FOCUS_LAYER:-16} ;;
  *) FOCUS_LAYER=${FOCUS_LAYER:-8} ;;
esac
EXTRA_TR=""
if [ -n "${LOWMEM}" ]; then EXTRA_TR="--gradient-checkpointing --micro-rows 2"; fi
if [ -z "${SKIP_TRAIN}" ] && [ ! -f "${MEOW_DIR}/generated_data/${FORGET_SPLIT}_perturbed_generated.json" ]; then echo "missing ${MEOW_DIR}/generated_data/${FORGET_SPLIT}_perturbed_generated.json (the released inverted facts of MEOW)" >&2; exit 1; fi
echo "meow.sh effective settings: MODEL=${MODEL} FORGET_SPLIT=${FORGET_SPLIT} RETAIN_SPLIT=${RETAIN_SPLIT} HOLDOUT_SPLIT=${HOLDOUT_SPLIT} NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT} TARGET=${TARGET} K=${K} SELECT=${SELECT} STEPS=${STEPS_A[*]}"
if [ -z "${SKIP_TRAIN}" ]; then
  python src/METHOD/MEOW/meow.py --model open-unlearning/tofu_${MODEL}_full --forget-split ${FORGET_SPLIT} \
    --meow-json ${MEOW_DIR}/generated_data/${FORGET_SPLIT}_perturbed_generated.json --memo-json ${MEMO_JSON} \
    --target ${TARGET} --neighbor-csv data/tofu/tofu_${NEIGHBOR_SPLIT}.csv --k ${K} --select ${SELECT} --lr ${LR} \
    --save-steps "${STEPS_A[@]}" --template-config configs/model/tofu/tofu_${MODEL}_full.yaml --memo-max-new-tokens ${MEMO_NEW_TOKENS:-64} \
    ${EXTRA_TR} --out ${OUT}
fi
EXTRA=""
if [ "${EVAL_PROFILE:-fast}" = fast ]; then EXTRA="eval=tofu_fast"; fi
for STEP in "${STEPS_A[@]}"; do
  CK=${OUT}/checkpoint-${STEP}
  if [ ! -f "${CK}/config.json" ]; then echo "[skip] ${CK} does not exist"; continue; fi
  python src/eval.py experiment=eval/tofu/probe model=${TARGET_MODEL} model.pretrained.name_or_path=${CK} ${SPLITS} \
    eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH} \
    eval.tofu.metrics.activation_probe.r_uv_path=saves/NEAR/${R_UV_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}_ks/r_uv.pt \
    eval.tofu.metrics.activation_probe.focus_layer=${FOCUS_LAYER} \
    "eval.tofu.metrics.activation_probe.seen=[retain]" "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]" \
    task_name=MEOW/${NAME}_s${STEP} ${EXTRA} ${EXTRA_EVAL}
  keep=0; for k in "${KEEP_A[@]}"; do if [ "${k}" = "${STEP}" ]; then keep=1; fi; done
  if [ "${keep}" = 0 ]; then rm -f ${CK}/*.safetensors ${CK}/*.bin; fi
done
