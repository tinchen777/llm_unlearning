#!/bin/bash
# Step 1b: full - retain difference of the activation AUCs of one pair of question sets (no GPU, no model).
# The AUC of a pair mostly reflects how different the questions are, which is the same for both models; the difference
# of the SAME pair (default `forget | neighbor`, the matched one) keeps what exists only because the full model has seen
# the forget data. Layers with a large positive auc_group_delta are a candidate window for 2_select_layer.sh.
# Needs 0_probe.sh (kinds retain AND full) first.
set -e
cd $(dirname "$0")/../.. || exit 1

BASE_MODELS=(Llama-3.2-1B-Instruct Llama-3.2-3B-Instruct Llama-3.1-8B-Instruct)
# "forget_split retain_split"
SPLIT_SETS=("forget10 retain90" "forget05 retain95" "forget01 retain99")
PAIR="forget | neighbor"

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for split in "${SPLIT_SETS[@]}"; do
    read -r FORGET_SPLIT RETAIN_SPLIT <<< "${split}"
    FULL=saves/NEAR/1_probe/tofu_${BASE_MODEL}_full_${FORGET_SPLIT}/activations_summary.json
    RETAIN=saves/NEAR/1_probe/tofu_${BASE_MODEL}_${RETAIN_SPLIT}_${FORGET_SPLIT}/activations_summary.json
    echo "=== ${BASE_MODEL} ${FORGET_SPLIT}"
    python src/METHOD/NEAR/neighbor_compare.py \
      --full "${FULL}" --retain "${RETAIN}" --pair "${PAIR}" \
      --out saves/NEAR/1_probe/compare_${BASE_MODEL}_${FORGET_SPLIT}.json
  done
done
