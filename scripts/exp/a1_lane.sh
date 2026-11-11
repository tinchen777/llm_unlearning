#!/bin/bash
# A1 (the black-box repeated-query ranking attack, scripts/analysis/rank_attack.py) on forget05, 3B and 1B: pool = the 10 forgotten + the 10 holdout authors, models = retain (the
# calibration: AUC ~ 0.5), full, NEAR-Act and NEAR-Dist and LUNAR at their matched points, ROCR (original).  The edited models of NEAR / LUNAR are rebuilt in memory from w_down.pt; the ROCR model
# folder was deleted by the clean-up, so it is re-made first (the null-space projectors are cached in data/rocr, a few minutes).  7 steps (names check, 3B, 1B, 8B), ~ 20 prompts x 10 templates x 11 answers per model.
# A failed step does not stop the lane; a finished step is marked in logs/a1/done/ and skipped on a re-run.  3B / 1B need little memory; the 8B step needs the 80 GB card.
#   bash scripts/exp/a1_lane.sh <GPU>             # background (nohup), log logs/a1/a1.log
#   FG=1 bash scripts/exp/a1_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Result: saves/analysis/rank_attack_<3B|1B>_f05/summary.md.  CHECK FIRST the line "pool: ..." in logs/a1/a1_attack_*.log: the author names are extracted from the questions (not checked on holdout05).
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/a1; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/a1.log" 2>&1 &
  echo "lane a1 started (pid $!), log ${LOGDIR}/a1.log"
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
echo "lane a1 on GPU ${G} starts $(date)"
step a1_names <<'CMD' || true
python scripts/analysis/rank_attack.py --names-only --forget forget05 --holdout holdout05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --out saves/analysis/rank_attack_names
CMD
step a1_rocr_model_3b_f05 <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 3 4 5 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5; fi
CMD
step a1_rocr_model_1b_f05 <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 2 3 4 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4; fi
CMD
step a1_attack_3b_f05 <<'CMD' || true
python scripts/analysis/rank_attack.py --gpu ${G} --out saves/analysis/rank_attack_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget forget05 --holdout holdout05 --model retain=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14 --model NEAR-Dist=near:saves/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 --model LUNAR=near:saves/LUNAR/forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks/c1.5_all/l14 --model ROCR=hf:saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model
CMD
step a1_attack_1b_f05 <<'CMD' || true
python scripts/analysis/rank_attack.py --gpu ${G} --out saves/analysis/rank_attack_1B_f05 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget05 --holdout holdout05 --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9 --model NEAR-Dist=near:saves/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 --model LUNAR=near:saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful/c1.5_all/l11 --model ROCR=hf:saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model
CMD
step a1_rocr_model_8b_f05 <<'CMD' || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${G}; if [ -f saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model/config.json ]; then echo "model exists"; else python src/METHOD/ROCR/rocr.py --model open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget-split forget05 --author-offset 191 --neighbor-csv data/tofu/tofu_neighbor05.csv --target famous --layers 4 5 6 --scale 1 --dtype bfloat16 --out saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6; fi
CMD
step a1_attack_8b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack.py --gpu ${G} --out saves/analysis/rank_attack_8B_f05 --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget forget05 --holdout holdout05 --model retain=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95 --model full=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_full --model NEAR-Act=near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19 --model NEAR-Dist=near:saves/NEAR/forget05_Llama8B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18 --model LUNAR=near:saves/LUNAR/single/lunar_tofu_Llama-3.1-8B-Instruct_forget05_LUNAR_harmful/c2.8_all/l23 --model ROCR=hf:saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model
CMD
echo "lane a1 finished $(date)"
