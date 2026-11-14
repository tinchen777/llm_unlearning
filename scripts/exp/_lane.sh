# Shared boilerplate of the lane scripts of batch 1 onwards (b1_*.sh). Source it AFTER setting LANE (name) and G (GPU id); the caller's "$0" / "$@" are re-used.
#   background by default (nohup, log logs/<LANE>/lane.log); FG=1 foreground; DRY_RUN=1 only prints the steps; FORCE=1 ignores the done markers (logs/<LANE>/done/).
#   step <name> <<CMD ... CMD     runs the command in a subshell, log logs/<LANE>/<name>.log; a failed step is reported and the lane goes on (`|| true` at the call).
#   wait_for <lane>               blocks until another lane has FINISHED (marker logs/<lane>/done/_finished, written by lane_end even if steps failed), for order on a shared card.
#   lane_end                      last line of every lane.
source scripts/exp/_clean_env.sh
LOGDIR=logs/${LANE}; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "$@" > "${LOGDIR}/lane.log" 2>&1 &
  rm -f "${DONE}/_finished"
  echo "${LANE} started on GPU ${G} (pid $!), log ${LOGDIR}/lane.log"
  exit 0
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() {
  local name=$1 cmd; cmd=$(cat)
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; return 0; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
wait_for() {
  [ -n "${DRY_RUN}" ] && { echo "    (wait for lane $1 to finish: logs/$1/done/_finished)"; return 0; }
  until [ -f "logs/$1/done/_finished" ]; do sleep 300; done
}
lane_end() {
  [ -n "${DRY_RUN}" ] || touch "${DONE}/_finished"
  echo "${LANE} finished $(date)"
}
# NEAR v2 settings shared by the training lanes (= dist_final_lane_*.sh / abl_target_*.sh)
DIST='FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=neighbor_scaled KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 KL_NPQ=3 KL_MASK_NAME=1 RKL_WEIGHT=1 EVAL_PROFILE=full'
S01='FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01'
S05='FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05'
S10='FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10'
BIG='EXTRA_EVAL="eval.tofu.batch_size=4"'
