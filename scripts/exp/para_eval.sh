#!/bin/bash
# E6 with REWORDED QUESTIONS (TODO/exp/22 13b): every method's forget05 model (the models of the A1 v2 / A2 P0 lanes; NEAR = v2, LUNAR 3B = the real forget05 point) evaluated with
# the forget-only profile configs/eval/tofu_para.yaml twice: original questions (`_q`) and TOFU's paraphrased questions (`_pq`). Inference only.
#   bash scripts/exp/para_eval.sh <GPU> <SCALE: 1B|3B|8B>        (DRY_RUN=1 prints; FORCE=1 re-evaluates; a finished model is skipped)
# Output: saves/eval/para/<SCALE>_f05/<label>_{q,pq}/TOFU_EVAL.json -> python scripts/analysis/para_question_table.py
# near: specs (w_down.pt) are materialised to saves/spa/models/para_<SCALE>_<label> for the evaluation and deleted afterwards.
cd "$(dirname "$0")/../.." || exit 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=${1:-0}; S=${2:?scale 1B|3B|8B}; export CUDA_VISIBLE_DEVICES=${G}
V4='rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq'
case ${S} in
  1B) M=Llama-3.2-1B-Instruct; LIST="retain|hf:open-unlearning/tofu_${M}_retain95
full|hf:open-unlearning/tofu_${M}_full
NEAR|near:saves/NEAR/forget05_Llama1B_v4/${V4}_c7_l11-12-13/k5_c7_all/l11-12-13
NEAR-Act|near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9
LUNAR|near:saves/LUNAR/single/lunar_tofu_${M}_forget05_LUNAR_harmful/c1.5_all/l11
ROCR|hf:saves/ROCR/${M}_forget05_famous_l2-3-4/model
GA|hf:saves/tofu_unlearn/GradAscent/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-14
NPO|hf:saves/tofu_unlearn/NPO/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-35
RMU|hf:saves/tofu_unlearn/RMU/tofu_${M}_forget05_lr5e-5_ep5_ckpt
MEOW|hf:saves/MEOW/${M}_forget05_meow_k3_largest_lr3e-5/checkpoint-200
MEOW-nb|hf:saves/MEOW/${M}_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-50" ;;
  3B) M=Llama-3.2-3B-Instruct; LIST="retain|hf:open-unlearning/tofu_${M}_retain95
full|hf:open-unlearning/tofu_${M}_full
NEAR|near:saves/NEAR/forget05_Llama3B_v4/${V4}_c8_l19-20-21/k5_c8_all/l19-20-21
NEAR-Act|near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14
LUNAR|near:saves/LUNAR/single/lunar_tofu_${M}_forget05_LUNAR_harmful/c1.5_all/l12
ROCR|hf:saves/ROCR/${M}_forget05_famous_l3-4-5/model
GA|hf:saves/tofu_unlearn/GradAscent/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-21
NPO|hf:saves/tofu_unlearn/NPO/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-14
RMU|hf:saves/tofu_unlearn/RMU/tofu_${M}_forget05_lr1e-4_ep5_ckpt
MEOW|hf:saves/MEOW/${M}_forget05_meow_k3_largest_lr3e-5/checkpoint-200
MEOW-nb|hf:saves/MEOW/${M}_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-200" ;;
  8B) M=Llama-3.1-8B-Instruct; LIST="retain|hf:open-unlearning/tofu_${M}_retain95
full|hf:open-unlearning/tofu_${M}_full
NEAR|near:saves/NEAR/forget05_Llama8B_v4/${V4}_c7_l16-17-18/k5_c7_all/l16-17-18
NEAR-Act|near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19
LUNAR|near:saves/LUNAR/single/lunar_tofu_${M}_forget05_LUNAR_harmful/c2.8_all/l23
ROCR|hf:saves/ROCR/${M}_forget05_famous_l4-5-6/model
GA|hf:saves/tofu_unlearn/GradAscent/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-14
NPO|hf:saves/tofu_unlearn/NPO/tofu_${M}_forget05_lr2e-5_ep5_ckpt/checkpoint-35
RMU|hf:saves/tofu_unlearn/RMU/tofu_${M}_forget05_lr1e-4_ep5_ckpt" ;;
  *) echo "scale must be 1B, 3B or 8B" >&2; exit 1 ;;
esac
EXTRA=""; [ "${S}" = 8B ] && EXTRA="eval.tofu.batch_size=4"
run() { if [ -n "${DRY_RUN}" ]; then echo "    $*"; else "$@" < /dev/null; fi; }   # (stdin: the model list of the loop)
while IFS='|' read -r label spec; do
  [ -z "${label}" ] && continue
  out=saves/eval/para/${S}_f05/${label}
  if [ -z "${FORCE}" ] && [ -f "${out}_q/TOFU_EVAL.json" ] && [ -f "${out}_pq/TOFU_EVAL.json" ]; then echo "[skip] ${S} ${label}"; continue; fi
  echo "[$(date +%T)] ${S} ${label}: ${spec}"
  path=${spec#*:}; tmp=""
  if [ "${spec%%:*}" = near ]; then
    tmp=saves/spa/models/para_${S}_${label}
    run python scripts/analysis/spa_materialize.py --model "${spec}" --base open-unlearning/tofu_${M}_full --out "${tmp}" || { echo "[FAIL] materialise ${label}" >&2; continue; }
    path=${tmp}
  fi
  for qk in question paraphrased_question; do
    tag=$([ ${qk} = question ] && echo q || echo pq)
    run python src/eval.py experiment=eval/tofu/default model=tofu/tofu_${M}_full model.pretrained.name_or_path=${path} \
      forget_split=forget05 holdout_split=holdout05 retain_split=retain95 eval=tofu_para eval.tofu.question_key=${qk} ${EXTRA} \
      task_name=para/${S}_f05/${label}_${tag} || echo "[FAIL] eval ${label} ${qk}" >&2
  done
  [ -n "${tmp}" ] && [ -z "${DRY_RUN}" ] && rm -rf "${tmp}"
done <<< "${LIST}"
echo "para_eval ${S} finished $(date)"
