#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


# BASE_MODEL=Llama-3.2-1B-Instruct

BASE_MODELS=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Llama-3.1-8B-Instruct
)

RETAIN_SPLITS=(
  "forget10 holdout10 retain90 neighbor10"
  "forget05 holdout05 retain95 neighbor05"
  "forget01 holdout01 retain99 neighbor01"
  # "forget10 holdout10 retain90 full"
)

# FORGET_SPLIT=forget01
# NEIGHBOR_SPLIT=neighbor01
# RETAIN_SPLIT=retain99
# HOLDOUT_SPLIT=holdout01

# MODEL=tofu_${BASE_MODEL}_${RETAIN_SPLIT}

# RETAIN_MODEL_PATH=saves/retain/ep_5/tofu_${MODEL}_${RETAIN_SPLIT}

# Step 1: probe the base / TOFU-full / TOFU-retain models on forget, neighbor, retain, holdout.
# Hypothesis: on entities a model has never seen it answers confidently but wrongly (hallucination), not "I don't know".
#   base   : never saw TOFU      -> all four splits should look alike (unseen)
#   full   : saw forget + retain -> forget/retain answered correctly; neighbor/holdout hallucinated
#   retain : saw retain only     -> forget should now look like neighbor/holdout (the target of the method)
# per model -> saves/neighbor/test/probe_<MODEL>_<FORGET_SPLIT>_<target>/
#   responses_<split>.json      generations + rouge + answer_logprob + refusal/degenerate flags
#   responses_summary.json      per split: rougeL_recall, answer_prob, refusal / hallucination / degenerate rates
#   activations_summary.json    per layer: norms, cosines, seen-vs-unseen separability AUC (step 2 diagnostics)

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for split in "${RETAIN_SPLITS[@]}"; do
    read -r forget_split holdout_split retain_split neighbor_split <<< "${split}"
    FORGET_SPLIT=${forget_split}
    HOLDOUT_SPLIT=${holdout_split}
    RETAIN_SPLIT=${retain_split}
    NEIGHBOR_SPLIT=${neighbor_split}

    MODEL=tofu_${BASE_MODEL}_${RETAIN_SPLIT}
    echo start probe ${MODEL}

    python src/neighbor.py \
      experiment=generate/neighbor_probe \
      model=tofu/${MODEL} \
      forget_split=${FORGET_SPLIT} \
      neighbor_split=${NEIGHBOR_SPLIT} \
      retain_split=${RETAIN_SPLIT} \
      holdout_split=${HOLDOUT_SPLIT} \
      task_name=1_probe/responses/${MODEL} \
      # --cfg job --resolve
      # probe.max_gen_samples=null \
      # probe.activations=false \

    echo end probe ${MODEL}
  done
done


# tofu_Llama-3.2-1B-Instruct_full