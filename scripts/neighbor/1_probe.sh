#!/bin/bash

set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


BASE_MODELS=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Llama-3.1-8B-Instruct
)

# "forget_split holdout_split retain_split neighbor_split"
SPLIT_SETS=(
  "forget10 holdout10 retain90 neighbor10"
  "forget05 holdout05 retain95 neighbor05"
  "forget01 holdout01 retain99 neighbor01"
)

# which official checkpoints to probe: `retain` (saw the retain data only = the ideal result of unlearning) and
# `full` (saw forget + retain). The activation AUCs of the SAME pair of the two are compared afterwards
# (scripts/neighbor/1b_probe_compare.sh).
KINDS=(retain full)

# Step 1: ANALYSIS of the official models on forget, neighbor, retain, holdout. Nothing of steps 2-4 reads these outputs.
# Hypothesis: on entities a model has never seen it answers confidently but wrongly (hallucination), not "I don't know".
#   retain : saw retain only     -> forget / neighbor / holdout are all unseen: no refusals, hallucination; forget ~ holdout
#   full   : saw forget + retain -> forget / retain answered correctly; neighbor / holdout hallucinated
# (`model=<base>` = a base model that never saw TOFU is possible too, but the Llama-3.2-1B base checkpoint is broken in
#  some environments, see configs/model/Llama-3.2-1B-Instruct.yaml.)
# per model and forget split -> saves/neighbor/1_probe/<MODEL>_<FORGET_SPLIT>/
#   responses_<split>.json      generations + rouge + answer_logprob + refusal/degenerate flags
#   responses_summary.json      per split: rougeL_recall, answer_prob, refusal / hallucination / degenerate rates
#   activations_summary.json    per layer: norms, r_UV size, AUCs of pairs of question sets (see its `note`: they measure
#                               how different the QUESTIONS are; only forget | neighbor and full - retain mean knowledge)

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for split in "${SPLIT_SETS[@]}"; do
    read -r FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT NEIGHBOR_SPLIT <<< "${split}"
    for KIND in "${KINDS[@]}"; do
      if [ "${KIND}" = full ]; then
        MODEL=tofu_${BASE_MODEL}_full
        # the full model has seen forget and retain, not neighbor / holdout (the defaults of probe.seen / probe.unseen)
        SEEN_ARGS=()
      else
        MODEL=tofu_${BASE_MODEL}_${RETAIN_SPLIT}
        # the retain model has seen the retain data only: forget is unseen for it
        SEEN_ARGS=('probe.seen=[retain]' 'probe.unseen=[forget,neighbor,holdout]')
      fi
      echo start probe ${MODEL} ${FORGET_SPLIT}

      python src/neighbor.py \
        experiment=generate/neighbor_probe \
        model=tofu/${MODEL} \
        forget_split=${FORGET_SPLIT} \
        neighbor_split=${NEIGHBOR_SPLIT} \
        retain_split=${RETAIN_SPLIT} \
        holdout_split=${HOLDOUT_SPLIT} \
        "${SEEN_ARGS[@]}" \
        task_name=1_probe/${MODEL}_${FORGET_SPLIT} \
        # --cfg job --resolve
        # probe.max_gen_samples=null \
        # probe.responses=false \
        # probe.activations=false \

      echo end probe ${MODEL} ${FORGET_SPLIT}
    done
  done
done
