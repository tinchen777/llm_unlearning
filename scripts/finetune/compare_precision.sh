#!/bin/bash
# TEMPORARY: is the 1B TOFU-full finetuning limited by pure-bf16 training (weights stored in bf16: Adam steps of lr 1e-5 are
# rounded away)? Trains the same model twice, once with fp32 weights (+ bf16 autocast, i.e. fp32 master weights) and once with the
# default bf16 weights, then prints the training losses side by side. The TOFU metrics of every epoch are in
# saves/finetune/test/<task>/checkpoint-<step>/evals/TOFU_SUMMARY.json (the official full model: forget_Q_A_Prob ~0.90, model_utility ~0.60).
#
#   bash scripts/finetune/compare_precision.sh [GPU_ID] [EPOCHS] [MODEL]
#   bash scripts/finetune/compare_precision.sh 0 3                    # both, 3 epochs, Llama-3.2-1B-Instruct
#   PRECISIONS="fp32" bash scripts/finetune/compare_precision.sh 0    # only one of them
#
# Note: warmup_epochs = 1.0 in the experiment, so the first epoch only warms the learning rate up.
set -e
cd "$(dirname "$0")/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
EPOCHS=${2:-3}
MODEL=${3:-Llama-3.2-1B-Instruct}
read -r -a PRECISIONS <<< "${PRECISIONS:-fp32 bf16}"
echo "Using GPU: [${GPU_ID}]  model: ${MODEL}  epochs: ${EPOCHS}  precisions: ${PRECISIONS[*]}"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

for P in "${PRECISIONS[@]}"; do
  case "${P}" in
    fp32) DTYPE_ARG=(model.pretrained.dtype=float32) ;;
    bf16) DTYPE_ARG=() ;;                     # the default of the model config (bfloat16)
    *) echo "unknown precision '${P}' (fp32 | bf16)" >&2; exit 1 ;;
  esac
  TASK=test/tofu_${MODEL}_ep${EPOCHS}_${P}
  echo "################  ${P}: ${TASK}  ################"
  python src/train.py \
    experiment=finetune/tofu/default \
    model=${MODEL} \
    "${DTYPE_ARG[@]}" \
    trainer.args.num_train_epochs=${EPOCHS} \
    task_name=${TASK}
done

echo "################  training losses  ################"
python - "${MODEL}" "${EPOCHS}" "${PRECISIONS[@]}" <<'PY'
import json, sys
model, epochs, precisions = sys.argv[1], sys.argv[2], sys.argv[3:]
runs = {}
for p in precisions:
    path = f"saves/finetune/test/tofu_{model}_ep{epochs}_{p}/trainer_state.json"
    try:
        runs[p] = {x["step"]: x["loss"] for x in json.load(open(path))["log_history"] if "loss" in x}
    except FileNotFoundError:
        print("missing", path)
steps = sorted(set().union(*[set(r) for r in runs.values()])) if runs else []
every = max(1, len(steps) // 15)
print("step  " + "  ".join(f"{p:>8}" for p in runs))
for s in steps[::every] + ([steps[-1]] if steps and (len(steps) - 1) % every else []):
    print(f"{s:>4}  " + "  ".join(f"{runs[p].get(s, float('nan')):8.4f}" for p in runs))
PY
