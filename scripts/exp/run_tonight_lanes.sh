#!/bin/bash
# Tonight's experiments that need no code change (exp/15 §二 ablation ladder L1-L5, §三 E8 / Act training seeds and sampling seeds, + ROCR residual factors 4/5/6 and an fp32 check),
# all 1B / 3B (no 8B), three lanes of <= 40 GB each: lanes a and b run at the same time on the 80 GB card, lane c on the 40 GB card.
#   bash scripts/exp/run_tonight_lanes.sh <80GB_GPU> <40GB_GPU>      # the three lanes in the background (nohup), logs in logs/tonight/
#   LANE=a bash scripts/exp/run_tonight_lanes.sh 0 1                 # one lane (a | b | c);  FG=1 = foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# The commands are in scripts/exp/tonight_cmds.tsv (lane <TAB> step <TAB> command; ${G} = the card of the lane), copied from TODO/exp/15 and TODO/exp/17.
# A failed step does not stop the lane; a finished step is marked in logs/tonight/done/ and skipped on a re-run. Watch: tail -f logs/tonight/a.log logs/tonight/b.log logs/tonight/c.log
cd "$(dirname "$0")/../.." || exit 1
BIG_GPU=${1:-0}; SMALL_GPU=${2:-1}
LOGDIR=logs/tonight; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  for l in a b c; do
    if [ -n "${LANE}" ] && [ "${LANE}" != "${l}" ]; then continue; fi
    LANE_INNER=${l} nohup bash "$0" "${BIG_GPU}" "${SMALL_GPU}" > "${LOGDIR}/${l}.log" 2>&1 &
    echo "lane ${l} started (pid $!), log ${LOGDIR}/${l}.log"
  done
  exit 0
fi
l=${LANE_INNER:-${LANE:-a}}
if [ "${l}" = c ]; then G=${SMALL_GPU}; else G=${BIG_GPU}; fi
echo "lane ${l} on GPU ${G} starts $(date)"
while IFS=$'\t' read -r lane name cmd; do
  [ "${lane}" = "${l}" ] || continue
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; continue; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; continue; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; fi
done < scripts/exp/tonight_cmds.tsv
echo "lane ${l} finished $(date)"
