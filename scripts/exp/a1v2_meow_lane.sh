#!/bin/bash
# A1 v2 on the MEOW models (run meow_keep_lane.sh first: the weights) and the two "naturalness" attacks of the MEOW paper (PPL > 200, consistency with "Sorry, I don't know") on the saved responses of ALL methods
# at their forget05 matched points (1B, 3B): the check that natural-looking responses (low attack success) are nevertheless distinguishable (A1 v2). The attack is added into the folders of a1v2_lane.sh
# (same templates, same pool; the analysis of the folder is redone with all the models, summary_k2_max.md); the naturalness tables go to saves/analysis/naturalness_<1B|3B>_f05.md.
# 4 steps; a failed step does not stop the lane; a finished step is marked in logs/next/done/.
#   bash scripts/exp/a1v2_meow_lane.sh <GPU>             # background (nohup), log logs/next/a1v2_meow_lane.log
#   FG=1 bash scripts/exp/a1v2_meow_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/a1v2_meow_lane.log" 2>&1 &
  echo "a1v2_meow_lane started (pid $!), log ${LOGDIR}/a1v2_meow_lane.log"
  exit 0
fi
step() {  # step <name> < command on stdin (the here-document below)
  local name=$1 cmd; cmd=$(cat)
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; return 0; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
echo "a1v2_meow_lane on GPU ${G} starts $(date)"
step a1meow_attack_1b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_1B_f05 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model MEOW=hf:saves/MEOW/Llama-3.2-1B-Instruct_forget05_meow_k3_largest_lr3e-5/checkpoint-200 --model MEOW-nb=hf:saves/MEOW/Llama-3.2-1B-Instruct_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-50
CMD
step a1meow_attack_3b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model MEOW=hf:saves/MEOW/Llama-3.2-3B-Instruct_forget05_meow_k3_largest_lr3e-5/checkpoint-200 --model MEOW-nb=hf:saves/MEOW/Llama-3.2-3B-Instruct_forget05_meow_nb_k3_largest_lr3e-5/checkpoint-200
CMD
step a1meow_naturalness_1b_f05 <<'CMD' || true
python scripts/analysis/naturalness.py --gpu ${G} --ref-model open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95 --out saves/analysis/naturalness_1B_f05.md retain=saves/eval/retain/Llama-3.2-1B-Instruct/forget05 GA=saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14/evals NPO=saves/tofu_unlearn/NPO/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35/evals RMU=saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget05_lr5e-5_ep5_ckpt/checkpoint-28/evals LUNAR=saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful_c1.5_l11_all NEAR-Act=saves/eval/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9_k5_l7-8-9_c7_all NEAR-Dist=saves/eval/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13_k5_l11-12-13_c7_all ROCR=saves/eval/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4 MEOW=saves/eval/MEOW/Llama-3.2-1B-Instruct_forget05_meow_k3_largest_lr3e-5_s200 MEOW-nb=saves/eval/MEOW/Llama-3.2-1B-Instruct_forget05_meow_nb_k3_largest_lr3e-5_s50
CMD
step a1meow_naturalness_3b_f05 <<'CMD' || true
python scripts/analysis/naturalness.py --gpu ${G} --ref-model open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --out saves/analysis/naturalness_3B_f05.md retain=saves/eval/retain/Llama-3.2-3B-Instruct/forget05 GA=saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-21/evals NPO=saves/tofu_unlearn/NPO/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14/evals RMU=saves/tofu_unlearn/RMU/tofu_Llama-3.2-3B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14/evals LUNAR=saves/eval/LUNAR/forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks_c1.5_l14_all NEAR-Act=saves/eval/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14_k5_l12-13-14_c14_all NEAR-Dist=saves/eval/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21_k5_l19-20-21_c8_all ROCR=saves/eval/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5 MEOW=saves/eval/MEOW/Llama-3.2-3B-Instruct_forget05_meow_k3_largest_lr3e-5_s200 MEOW-nb=saves/eval/MEOW/Llama-3.2-3B-Instruct_forget05_meow_nb_k3_largest_lr3e-5_s200
CMD
echo "a1v2_meow_lane finished $(date)"
