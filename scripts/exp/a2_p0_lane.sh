#!/bin/bash
# A2, P0 = the recognition baseline (no shadow model; scripts/analysis/spa.py p0, TODO/exp/A2_SPEC.md §9): on EVERY method's matched forget05 model (the models of the A1 v2 lanes), does a probe
# trained on the model itself (retained authors = known vs strangers) still flag the forgotten authors (Q1: F vs strangers; on the retain model ~0.5), and does it confuse them with retained ones
# (Q2: F vs retained)? `before the edit` = layers below the first edited layer, where the activations equal the original model's. Independent of the NEAR-Dist fix (NEAR-Dist = the ORIGINAL E8 here).
# 3B, then 1B, then 8B (one 70-80 GB card). Per model: one forward pass over 60 names x 10 templates + the answer-probability proxy (~2-5 min). ROCR models are re-made if missing.
# Result: saves/analysis/spa_p0_<3B|1B|8B>/summary.md (+ summary.json). Activations: saves/spa/p0/<scale>/<model>.npz (not uploaded: *.npz is ignored).
#   bash scripts/exp/a2_p0_lane.sh <GPU>        # background, log logs/a2p0/lane.log;   FG=1 foreground;  DRY_RUN=1;  FORCE=1
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/a2p0; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/lane.log" 2>&1 &
  echo "a2_p0_lane started (pid $!), log ${LOGDIR}/lane.log"
  exit 0
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() {  # step <name> < command on stdin
  local name=$1 cmd; cmd=$(cat)
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; return 0; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
SETS=saves/analysis/spa/sets_v2.json
step sets <<'CMD' || true
[ -f saves/analysis/spa/sets_v2.json ] || python scripts/analysis/spa.py sets --out saves/analysis/spa/sets_v2.json
CMD

# label|spec|edited layers (empty = all layers are trained)
models_for() {
  case "$1" in
    3B) M=Llama-3.2-3B-Instruct; cat <<EOT
retain|hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95|
full|hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_full|
NEAR-Act|near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14|12,13,14
NEAR-Dist|near:saves/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21|19,20,21
LUNAR|near:saves/LUNAR/forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks/c1.5_all/l14|14
ROCR|hf:saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model|3,4,5
GA|hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-21|
NPO|hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14|
RMU|hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-3B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14|
MEOW|hf:saves/MEOW/Llama-3.2-3B-Instruct_forget05_meow_k3_largest_lr3e-5/checkpoint-200|
MEOW-nb|hf:saves/MEOW/Llama-3.2-3B-Instruct_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-200|
EOT
    ;;
    1B) M=Llama-3.2-1B-Instruct; cat <<EOT
retain|hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95|
full|hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full|
NEAR-Act|near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9|7,8,9
NEAR-Dist|near:saves/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13|11,12,13
LUNAR|near:saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful/c1.5_all/l11|11
ROCR|hf:saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model|2,3,4
GA|hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14|
NPO|hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35|
RMU|hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget05_lr5e-5_ep5_ckpt/checkpoint-28|
MEOW|hf:saves/MEOW/Llama-3.2-1B-Instruct_forget05_meow_k3_largest_lr3e-5/checkpoint-200|
MEOW-nb|hf:saves/MEOW/Llama-3.2-1B-Instruct_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-50|
EOT
    ;;
    8B) M=Llama-3.1-8B-Instruct; cat <<EOT
retain|hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95|
full|hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_full|
NEAR-Act|near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19|17,18,19
NEAR-Dist|near:saves/NEAR/forget05_Llama8B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18|16,17,18
LUNAR|near:saves/LUNAR/single/lunar_tofu_Llama-3.1-8B-Instruct_forget05_LUNAR_harmful/c2.8_all/l23|23
ROCR|hf:saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model|4,5,6
GA|hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14|
NPO|hf:saves/tofu_unlearn/NPO/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35|
RMU|hf:saves/tofu_unlearn/RMU/tofu_Llama-3.1-8B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14|
EOT
    ;;
  esac
}
rocr_layers() { case "$1" in 1B) echo "2 3 4";; 3B) echo "3 4 5";; 8B) echo "4 5 6";; esac; }

for S in 3B 1B 8B; do
  models_for "${S}" > /dev/null; models_for "${S}" > "${LOGDIR}/models_${S}.txt"
  M=$(case ${S} in 1B) echo Llama-3.2-1B-Instruct;; 3B) echo Llama-3.2-3B-Instruct;; 8B) echo Llama-3.1-8B-Instruct;; esac)
  L=$(rocr_layers "${S}"); LD=${L// /-}
  step rocr_model_${S} <<CMD || true
if [ -f saves/ROCR/${M}_forget05_famous_l${LD}/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_${M}_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers ${L} --scale 1 --dtype bfloat16 --out saves/ROCR/${M}_forget05_famous_l${LD}; fi
CMD
  ACTS=()
  while IFS='|' read -r label spec el; do
    [ -z "${label}" ] && continue
    out=saves/spa/p0/${S}/${label}.npz
    step p0_acts_${S}_${label} <<CMD || true
python scripts/analysis/spa.py collect --sets ${SETS} --model "${spec}" --base open-unlearning/tofu_${M}_full --out ${out}
CMD
    if [ -n "${DRY_RUN}" ] || [ -f "${out}" ]; then ACTS+=("${label}=${out}${el:+#${el}}"); fi
  done < "${LOGDIR}/models_${S}.txt"
  FORCE=1 step p0_attack_${S} <<CMD || true
python scripts/analysis/spa.py p0 --sets ${SETS} --acts ${ACTS[*]} --out saves/analysis/spa_p0_${S}
CMD
done
echo "a2_p0_lane finished $(date)"
