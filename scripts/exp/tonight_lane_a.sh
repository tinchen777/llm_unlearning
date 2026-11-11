#!/bin/bash
# Tonight, lane a (80 GB card (shares it with lane b)): E4: E8 training seeds (3B / 1B forget01) and the sampling seeds.  8 steps, one after the other; a failed step does not stop the lane; a finished step is marked in
# logs/tonight/done/ and skipped on a re-run (the same command resumes).  Commands from TODO/exp/15 and TODO/exp/17.
#   bash scripts/exp/tonight_lane_a.sh <GPU>             # starts in the background (nohup), log logs/tonight/a.log
#   FG=1 bash scripts/exp/tonight_lane_a.sh <GPU>        # in the foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Watch: tail -f logs/tonight/a.log ; one log per step: logs/tonight/<step>.log
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/tonight; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/a.log" 2>&1 &
  echo "lane a started (pid $!), log ${LOGDIR}/a.log"
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
echo "lane a on GPU ${G} starts $(date)"
step e4_e8_seed1_3b_f01 <<'CMD' || true
SEED=1 SETTING=3B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_e8_seed2_3b_f01 <<'CMD' || true
SEED=2 SETTING=3B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_e8_seed1_1b_f01 <<'CMD' || true
SEED=1 SETTING=1B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_e8_seed2_1b_f01 <<'CMD' || true
SEED=2 SETTING=1B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_sampleseed_ss1_1b <<'CMD' || true
SAMPLE_SEED=1 SETTING=1B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_sampleseed_ss1_3b <<'CMD' || true
SAMPLE_SEED=1 SETTING=3B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_sampleseed_ss2_1b <<'CMD' || true
SAMPLE_SEED=2 SETTING=1B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_sampleseed_ss2_3b <<'CMD' || true
SAMPLE_SEED=2 SETTING=3B_f01 VARIANTS="E8" SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh ${G}
CMD
echo "lane a finished $(date)"
