#!/bin/bash
# NEAR-Dist likelihood fix, validation on ONE cell (Llama-3.2-3B, forget05): the E8 configuration of TODO/exp/11 (same layers 19-21, coefficient 8, 300 KL steps, neighbour-scaled
# calibration, no steering anchor, 3 neighbours per question, names masked) + the two new terms of src/METHOD/NEAR/kl_refine.py (TODO/exp/20):
#   V3 rkl1_dyn25  retain-answer KL weight 1 + dynamic forget target every 25 steps   (the most likely fix, first)
#   V1 dyn25       dynamic target only     V2 rkl1  retain KL only     V4 rkl3_dyn25  stronger retain KL
# Outputs under the NEW prefix forget05_Llama3B_v4 (cells.py reads only _v2 / _v3: the main table does not change). Last step: the acceptance table saves/analysis/dist_fix_3B_forget05.md.
# 5 steps, one 40 GB card (3B), each run = KL refinement + full evaluation; a failed step does not stop the lane; finished steps are marked in logs/distfix/done/.
#   bash scripts/exp/dist_fix_lane.sh <GPU>        # background (nohup), log logs/distfix/lane.log
#   FG=1 bash scripts/exp/dist_fix_lane.sh <GPU>   # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/distfix; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/lane.log" 2>&1 &
  echo "dist_fix_lane started (pid $!), log ${LOGDIR}/lane.log"
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
# the E8 command of TODO/exp/11 (3B forget05), with the new prefix
E8='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.2-3B-Instruct KL_STEPS=300 OUT_PREFIX=forget05_Llama3B_v4 UPSTREAM_PREFIX=forget05_Llama3B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=16 wdown.kl.batch_size=16" FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 EVAL_PROFILE=full COEFFS="8" LAYER_SETS="19,20,21"'
echo "dist_fix_lane on GPU ${G} starts $(date)"
step v3_rkl1_dyn25 <<CMD || true
${E8} RKL_WEIGHT=1 DYN_EVERY=25 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step v1_dyn25 <<CMD || true
${E8} DYN_EVERY=25 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step v2_rkl1 <<CMD || true
${E8} RKL_WEIGHT=1 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step v4_rkl3_dyn25 <<CMD || true
${E8} RKL_WEIGHT=3 DYN_EVERY=25 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
FORCE=1 step check <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 3B --split forget05 --out saves/analysis/dist_fix_3B_forget05.md
CMD
echo "dist_fix_lane finished $(date)"
