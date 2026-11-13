#!/bin/bash
# NEAR-Dist v2 ablation of the KL TARGET (paper_notes/3 §2.4; wdown.kl.target): the final NEAR-Dist configuration (E8 + retain-answer KL 1) with ONLY the target of the KL term changed:
#   idk                  "I don't know"-style answers (data/idk.jsonl), cross-entropy on their first 8 tokens        -> a refusal target, shared by all authors
#   shared_neighbor      ONE neighbour (the rank-1 neighbour of the first forget author) for every forget author      -> a shared but non-refusal target (tests "shared target -> white-box trace")
#   mismatched_neighbor  each author gets the next forget author's neighbours                                          -> targets still differ per author but are not attribute-matched
# The probability calibration (per author's own neighbours), the retain terms, layers and steps are those of the cell's NEAR-Dist v2 run (dist_final_lane_*.sh). Output prefix *_v4
# (seeds *_v4_s<n>), variants ..._rkl1_t<target>_... (cells.py does not count them as NEAR-Dist). A failed step does not stop the lane; done markers in logs/abltarget/done/.
# any card (small): 1B forget05 shared / mismatched, 1B forget10 idk; can share the 80 GB card with abl_target_big.
#   bash scripts/exp/abl_target_1b.sh <GPU>     # background, log logs/abltarget/abl_target_1b.log;   FG=1 foreground;  DRY_RUN=1;  FORCE=1
# Read: saves/analysis/dist_fix_<scale>_<split>.md, rows "Dist rkl1 (s0)" (NEAR-Dist v2) vs "Dist rkl1_t<target> (s..)": Prob, mia, utility, ud_bb (refusal trace), wb_unsup / wb_sup.
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/abltarget; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/abl_target_1b.log" 2>&1 &
  echo "abl_target_1b started (pid $!), log ${LOGDIR}/abl_target_1b.log"
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
DIST='FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 RKL_WEIGHT=1 EVAL_PROFILE=full'
S01='FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01'
S05='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05'
S10='FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10'
BIG='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4"'
# the cells (= their NEAR-Dist v2 settings)
C_3B05='${S05} MODEL=Llama-3.2-3B-Instruct KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama3B_v2 ${DIST} COEFFS="8" LAYER_SETS="19,20,21"'
C_1B05='${S05} MODEL=Llama-3.2-1B-Instruct KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama1B_v2 ${DIST} COEFFS="7" LAYER_SETS="11,12,13"'
C_8B05='${S05} MODEL=Llama-3.1-8B-Instruct ${BIG} KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama8B_v2 ${DIST} COEFFS="7" LAYER_SETS="16,17,18"'
C_3B01='${S01} MODEL=Llama-3.2-3B-Instruct UPSTREAM_PREFIX=Llama3B_v2 ${DIST} COEFFS="6" LAYER_SETS="19,20,21"'
C_1B10='${S10} MODEL=Llama-3.2-1B-Instruct KL_STEPS=500 UPSTREAM_PREFIX=forget10_Llama1B_v2 ${DIST} COEFFS="7" LAYER_SETS="11,12,13"'
echo "abl_target_1b on GPU ${G} starts $(date)"
step 1b05_shared <<CMD || true
$(eval echo "${C_1B05}") OUT_PREFIX=forget05_Llama1B_v4 EXTRA_WDOWN="wdown.kl.batch_size=16" KL_TARGET=shared_neighbor bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step 1b05_mismatched <<CMD || true
$(eval echo "${C_1B05}") OUT_PREFIX=forget05_Llama1B_v4 EXTRA_WDOWN="wdown.kl.batch_size=16" KL_TARGET=mismatched_neighbor bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step 1b10_idk <<CMD || true
$(eval echo "${C_1B10}") OUT_PREFIX=forget10_Llama1B_v4 EXTRA_WDOWN="wdown.kl.fr_batch_size=4 wdown.kl.batch_size=16" KL_TARGET=idk bash scripts/NEAR/sweep_kl.sh ${G}
CMD
FORCE=1 step check_abl_target_1b <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 1B --split forget05 --out saves/analysis/dist_fix_1B_forget05.md && python scripts/analysis/dist_fix_check.py --size 1B --split forget10 --out saves/analysis/dist_fix_1B_forget10.md
CMD
echo "abl_target_1b finished $(date)"
