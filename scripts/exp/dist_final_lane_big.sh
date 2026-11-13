#!/bin/bash
# NEAR-Dist FINAL configuration (2026-11-13, TODO/exp/20): E8 + the retain-answer KL weight 1 (RKL_WEIGHT=1), on every cell of the main table except 3B forget05 (done in the fix
# rounds: saves/eval/NEAR/forget05_Llama3B_v4/..._rkl1_...). Each cell keeps its E8 settings (layers, coefficient, KL steps, batch sizes of TODO/exp/11 and scripts/exp/validate_fix.sh);
# output prefix <prefix>_v4 (cells.py: NEAR-Dist = _rkl1 runs, the old E8 = NEAR-Dist-v1). Full evaluation everywhere (forget01 E8 had the fast profile).
# 80 GB card: the three 8B cells. A failed step does not stop the lane; done markers in logs/distfinal/done/.
#   bash scripts/exp/dist_final_lane_big.sh <GPU>       # background, log logs/distfinal/big.log;   FG=1 foreground;  DRY_RUN=1;  FORCE=1
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/distfinal; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/big.log" 2>&1 &
  echo "dist_final_lane_big started (pid $!), log ${LOGDIR}/big.log"
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
DIST='FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 RKL_WEIGHT=1 EVAL_PROFILE=full'
S01='FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01'
S05='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05'
S10='FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10'
BIG='PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4"'
echo "dist_final_lane_big on GPU ${G} starts $(date)"
step final_8b_f01 <<CMD || true
${S01} MODEL=Llama-3.1-8B-Instruct ${BIG} OUT_PREFIX=Llama8B_v4 UPSTREAM_PREFIX=Llama8B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=4 wdown.kl.batch_size=4 wdown.gen_batch_size=8" ${DIST} COEFFS="7" LAYER_SETS="16,17,18" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step final_8b_f05 <<CMD || true
${S05} MODEL=Llama-3.1-8B-Instruct ${BIG} KL_STEPS=300 OUT_PREFIX=forget05_Llama8B_v4 UPSTREAM_PREFIX=forget05_Llama8B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8" ${DIST} COEFFS="7" LAYER_SETS="16,17,18" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
step final_8b_f10 <<CMD || true
${S10} MODEL=Llama-3.1-8B-Instruct ${BIG} KL_STEPS=500 OUT_PREFIX=forget10_Llama8B_v4 UPSTREAM_PREFIX=forget10_Llama8B_v2 EXTRA_WDOWN="wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8" ${DIST} COEFFS="7" LAYER_SETS="16,17,18" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
echo "dist_final_lane_big finished $(date)"
