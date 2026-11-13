#!/bin/bash
# NEAR-Dist v2 ablation: the KL TARGET. Same configuration as the final NEAR-Dist (E8 + retain-answer KL 1), only the target of the KL term changes from the train neighbours'
# answers to "I don't know"-style answers (data/idk.jsonl, the file of the IdkDPO baseline; cross-entropy on their first 8 tokens, 3 answers per question by a fixed rotation).
# The calibration of the true answer's probability (neighbor_scaled), the retain activation term and the retain-answer KL are unchanged. Question: is the anchor ON THE DISTRIBUTION
# enough, or does the TARGET (neighbour vs refusal) matter too? (the missing cell "distribution anchor x refusal target" of the target x anchor table, paper_notes/3).
# Cells: 3B forget05 (the fix rounds' cell; the v2 runs are there) and 1B forget05 (its v2 run comes from dist_final_lane_small.sh). Prefix *_v4, variant ..._rkl1_tidk_...
# (cells.py does NOT count it as NEAR-Dist). One 35-40 GB card; a failed step does not stop the lane; done markers in logs/distabl/done/.
#   bash scripts/exp/dist_ablation_idk_lane.sh <GPU>     # background, log logs/distabl/idk.log;   FG=1 foreground;  DRY_RUN=1;  FORCE=1
# Read: saves/analysis/dist_fix_{3B,1B}_forget05.md (rows "Dist rkl1 (s0)" vs "Dist rkl1_tidk (s0)"): Prob, mia, utility, ud_bb (refusal trace), wb_unsup / wb_sup.
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/distabl; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/idk.log" 2>&1 &
  echo "dist_ablation_idk_lane started (pid $!), log ${LOGDIR}/idk.log"
  exit 0
fi
step() {  # step <name> < command on stdin
  local name=$1 cmd; cmd=$(cat)
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; return 0; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
if [ -z "${DRY_RUN}" ] && [ ! -s data/idk.jsonl ]; then echo "data/idk.jsonl not found (the IdkDPO baseline's file, one answer per line)" >&2; exit 1; fi
DIST='FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 RKL_WEIGHT=1 KL_TARGET=idk EVAL_PROFILE=full'
S05='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05'
echo "dist_ablation_idk_lane on GPU ${G} starts $(date)"
step idk_3b_f05 <<CMD || true
${S05} MODEL=Llama-3.2-3B-Instruct KL_STEPS=300 OUT_PREFIX=forget05_Llama3B_v4 UPSTREAM_PREFIX=forget05_Llama3B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=16 wdown.kl.batch_size=16" ${DIST} COEFFS="8" LAYER_SETS="19,20,21" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step idk_1b_f05 <<CMD || true
${S05} MODEL=Llama-3.2-1B-Instruct KL_STEPS=300 OUT_PREFIX=forget05_Llama1B_v4 UPSTREAM_PREFIX=forget05_Llama1B_v2 EXTRA_WDOWN="wdown.kl.batch_size=16" ${DIST} COEFFS="7" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
FORCE=1 step check <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 3B --split forget05 --out saves/analysis/dist_fix_3B_forget05.md && python scripts/analysis/dist_fix_check.py --size 1B --split forget05 --out saves/analysis/dist_fix_1B_forget05.md
CMD
echo "dist_ablation_idk_lane finished $(date)"
