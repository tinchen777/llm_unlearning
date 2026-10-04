#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# keep these identical in 1_probe / 3_r_uv / 4_select_layer / 5_w_down / 6_eval (3-5 share TASK_NAME)
MODEL=Llama-3.2-1B-Instruct
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99
HOLDOUT_SPLIT=holdout01
SPLITS="forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}"
TASK_NAME=test/redirect_${MODEL}_${FORGET_SPLIT}

# Step 1: probe the base / TOFU-full / TOFU-retain models on forget, neighbor, retain, holdout.
# Hypothesis: on entities a model has never seen it answers confidently but wrongly (hallucination), not "I don't know".
#   base   : never saw TOFU      -> all four splits should look alike (unseen)
#   full   : saw forget + retain -> forget/retain answered correctly; neighbor/holdout hallucinated
#   retain : saw retain only     -> forget should now look like neighbor/holdout (the target of the method)
# per model -> saves/neighbor/test/probe_<MODEL>_<FORGET_SPLIT>_<target>/
#   responses_<split>.json      generations + rouge + answer_logprob + refusal/degenerate flags
#   responses_summary.json      per split: rougeL_recall, answer_prob, refusal / hallucination / degenerate rates
#   activations_summary.json    per layer: norms, cosines, seen-vs-unseen separability AUC (step 2 diagnostics)
for TARGET in base full retain; do
  echo start probe ${MODEL} ${TARGET}
  python src/neighbor.py \
    experiment=generate/neighbor_probe \
    model=${MODEL} \
    target_model=${TARGET} \
    ${SPLITS} \
    task_name=test/probe_${MODEL}_${FORGET_SPLIT}_${TARGET} \
    # --cfg job --resolve
    # probe.max_gen_samples=null \
    # probe.activations=false \
  echo end probe ${MODEL} ${TARGET}
done
