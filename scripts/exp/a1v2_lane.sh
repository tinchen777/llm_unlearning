#!/bin/bash
# A1 v2 (scripts/analysis/rank_attack_v2.py) on forget05: the forgotten authors are a MINORITY of every pool (2 + all holdout authors of holdout05 and holdout10), query templates DERIVED FROM THE
# RETAINED DATA, a second attacker with logits (mean log-probability / entropy of the greedy answers), jackknife interval and label-permutation null. Models per size: retain (calibration), full (positive
# control), NEAR-Act, NEAR-Dist (E8), LUNAR (their matched points, rebuilt from w_down.pt) and ROCR (original; the model folder is re-made if it is missing). 3B and 1B first, 8B last (80 GB card).
# FIRST read the "templates:" lines of logs/a1v2/a1v2_templates.log (the forms derived from the retain split) and the "pool:" lines of the attack logs (names extracted from the questions).
# A failed step does not stop the lane; a finished step is marked in logs/a1v2/done/ and skipped on a re-run.
#   bash scripts/exp/a1v2_lane.sh <GPU>             # background (nohup), log logs/a1v2/a1v2_lane.log
#   FG=1 bash scripts/exp/a1v2_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Result: saves/analysis/rank_attack2_<3B|1B|8B>_f05/summary_k2_max.md (re-analyse offline: python scripts/analysis/rank_attack_v2.py --analyze <dir> [--k 2 --agg mean]).
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/a1v2; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/a1v2_lane.log" 2>&1 &
  echo "a1v2_lane started (pid $!), log ${LOGDIR}/a1v2_lane.log"
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
echo "a1v2_lane on GPU ${G} starts $(date)"
step a1v2_templates <<'CMD' || true
python scripts/analysis/rank_attack_v2.py --templates-only --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --out saves/analysis/rank_attack2_templates --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200
CMD
step a1v2_rocr_model_3b <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 3 4 5 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5; fi
CMD
step a1v2_attack_3b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model retain=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14 --model NEAR-Dist=near:saves/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 --model LUNAR=near:saves/LUNAR/forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks/c1.5_all/l14 --model ROCR=hf:saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model
CMD
step a1v2_rocr_model_1b <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 2 3 4 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4; fi
CMD
step a1v2_attack_1b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_1B_f05 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9 --model NEAR-Dist=near:saves/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 --model LUNAR=near:saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful/c1.5_all/l11 --model ROCR=hf:saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model
CMD
step a1v2_rocr_model_8b <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 4 5 6 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6; fi
CMD
step a1v2_attack_8b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_8B_f05 --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model retain=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19 --model NEAR-Dist=near:saves/NEAR/forget05_Llama8B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18 --model LUNAR=near:saves/LUNAR/single/lunar_tofu_Llama-3.1-8B-Instruct_forget05_LUNAR_harmful/c2.8_all/l23 --model ROCR=hf:saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model
CMD
echo "a1v2_lane finished $(date)"
