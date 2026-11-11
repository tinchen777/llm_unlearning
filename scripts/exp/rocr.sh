#!/bin/bash
# ROCR baseline (src/METHOD/ROCR/rocr.py, Ye et al. 2025) + the standard evaluation of this repository, one setting, one GPU.
#   MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=famous   bash scripts/exp/rocr.sh <GPU>     # the original: Donald Trump
#   MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor bash scripts/exp/rocr.sh <GPU>     # only the target changed: train_rank 1 neighbour
# Env: MODEL / FORGET_SPLIT / NEIGHBOR_SPLIT / RETAIN_SPLIT / HOLDOUT_SPLIT / AUTHOR_OFFSET (scripts/NEAR/_common.sh), R_UV_PREFIX = the saves/NEAR/<prefix>/ (e.g. single_v2, Llama3B_v2, forget05_Llama1B_v2) whose near_tofu_<MODEL>_<split>_ks/r_uv.pt the
# activation probe of the evaluation reads (any NEAR run of the same model and split), SCALE (factor on the residual, default 1 = ROCR), TAG (extra suffix),
# EVAL_PROFILE=fast|full, EXTRA_ROCR / EXTRA_EVAL (more arguments), SKIP_EDIT=1 (only evaluate), DTYPE (bfloat16 | float32).
set -e
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
source "${SCRIPT_DIR}/../NEAR/_common.sh"
cd "${SCRIPT_DIR}/../.." || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
GPU_ID=${1:-0}
echo "rocr.sh effective settings: MODEL=${MODEL} FORGET_SPLIT=${FORGET_SPLIT} RETAIN_SPLIT=${RETAIN_SPLIT} HOLDOUT_SPLIT=${HOLDOUT_SPLIT} NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT} AUTHOR_OFFSET=${AUTHOR_OFFSET} (an exported value of the terminal wins over the default)"
export CUDA_VISIBLE_DEVICES=${GPU_ID}
TARGET=${TARGET:-famous}
SCALE=${SCALE:-1}
read -r -a LAYERS_A <<< "${LAYERS:?LAYERS, e.g. \"4 5 6\" (Llama-3-8B), \"2 3 4\" (1B), \"3 4 5\" (3B)}"
R_UV_PREFIX=${R_UV_PREFIX:?R_UV_PREFIX, e.g. single_v2 (saves/NEAR/<prefix>/near_tofu_<MODEL>_<split>_ks/r_uv.pt)}
SCALE_TAG=""
if [ "${SCALE}" != 1 ]; then SCALE_TAG="_s${SCALE}"; fi   # (not `$([ .. ] && echo ..)`: its exit status 1 would end the script under set -e)
NAME=${MODEL}_${FORGET_SPLIT}_${TARGET}_l$(IFS=-; echo "${LAYERS_A[*]}")${SCALE_TAG}${TAG}
OUT=saves/ROCR/${NAME}
if [ -z "${SKIP_EDIT}" ]; then
  python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_${MODEL}_full --forget-split ${FORGET_SPLIT} --author-offset ${AUTHOR_OFFSET} \
    --neighbor-csv data/tofu/tofu_${NEIGHBOR_SPLIT}.csv --target ${TARGET} --layers "${LAYERS_A[@]}" --scale ${SCALE} --dtype ${DTYPE:-bfloat16} \
    --out ${OUT} ${EXTRA_ROCR}
fi
EXTRA=""
if [ "${EVAL_PROFILE:-full}" = fast ]; then EXTRA="eval=tofu_fast"; fi
python src/eval.py experiment=eval/tofu/probe model=${TARGET_MODEL} model.pretrained.name_or_path=${OUT}/model ${SPLITS} \
  eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH} \
  eval.tofu.metrics.activation_probe.r_uv_path=saves/NEAR/${R_UV_PREFIX}/near_tofu_${MODEL}_${FORGET_SPLIT}_ks/r_uv.pt \
  eval.tofu.metrics.activation_probe.focus_layer=${LAYERS_A[0]} \
  "eval.tofu.metrics.activation_probe.seen=[retain]" "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]" \
  task_name=ROCR/${NAME} ${EXTRA} ${EXTRA_EVAL}
