#!/bin/bash
# NEAR-Dist likelihood fix, round 2 (3B forget05; round 1 = dist_fix_lane.sh, results TODO/exp/20): the retain-answer KL alone fixes the drift (mia 0.32, retain_Q_A_Prob 0.79,
# utility 0.66) but ud_wb_sup rose 0.18 -> 0.25-0.30, above the pre-registered bound 0.23. Questions: (a) does a STRONGER retain KL lower it (rkl3 + dyn gave 0.25)? (b) is +0.1 inside the seed
# noise of ud_wb_sup (8B E8 seeds: 0.49 / 0.36 / 0.35)?
#   1 rkl3           2 rkl10          (retain KL only, no dynamic target: round 1 showed the dynamic target is not needed and alone it is harmful)
#   3 original E8, seed 1    4 rkl1, seed 1    5 original E8, seed 2    6 rkl1, seed 2       (the seed noise, before vs after the fix)
# Seeds as in next_lane_d.sh: EXTRA_WDOWN seed=<n>, prefix forget05_Llama3B_v4_s<n> (cells.py reads only _v2 / _v3). Last step: saves/analysis/dist_fix_3B_forget05.md (all rounds).
#   bash scripts/exp/dist_fix_lane2.sh <GPU>        # background, log logs/distfix/lane2.log;   FG=1 foreground;  DRY_RUN=1;  FORCE=1
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/distfix; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/lane2.log" 2>&1 &
  echo "dist_fix_lane2 started (pid $!), log ${LOGDIR}/lane2.log"
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
E8B='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 MODEL=Llama-3.2-3B-Instruct KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama3B_v2 FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 EVAL_PROFILE=full COEFFS="8" LAYER_SETS="19,20,21"'
BS='wdown.kl.fr_batch_size=16 wdown.kl.batch_size=16'
echo "dist_fix_lane2 on GPU ${G} starts $(date)"
step r2_rkl3 <<CMD || true
${E8B} OUT_PREFIX=forget05_Llama3B_v4 EXTRA_WDOWN="${BS}" RKL_WEIGHT=3 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step r2_rkl10 <<CMD || true
${E8B} OUT_PREFIX=forget05_Llama3B_v4 EXTRA_WDOWN="${BS}" RKL_WEIGHT=10 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
for s in 1 2; do
step r2_e8_seed${s} <<CMD || true
${E8B} OUT_PREFIX=forget05_Llama3B_v4_s${s} EXTRA_WDOWN="seed=${s} ${BS}" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step r2_rkl1_seed${s} <<CMD || true
${E8B} OUT_PREFIX=forget05_Llama3B_v4_s${s} EXTRA_WDOWN="seed=${s} ${BS}" RKL_WEIGHT=1 bash scripts/NEAR/sweep_kl.sh ${G}
CMD
done
FORCE=1 step check2 <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 3B --split forget05 --out saves/analysis/dist_fix_3B_forget05.md
CMD
echo "dist_fix_lane2 finished $(date)"
