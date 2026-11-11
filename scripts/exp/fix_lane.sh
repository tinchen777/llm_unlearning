#!/bin/bash
# Re-runs with the evaluation completed: ROCR -> neighbour, residual factors 5 / 6: forget05 3B factor 6 (its evaluation stopped after the first metrics) and forget10 1B / 3B (the earlier runs of lane c
# were made by mistake with forget10 and have no detectability metrics). Edit + evaluation (the edited models of the clean-up are gone; the null-space projectors are cached). Every command sets its splits.
# 5 steps for a 40 GB card; a failed step does not stop the lane; a finished step is marked in logs/next/done/.
#   bash scripts/exp/fix_lane.sh <GPU>             # background (nohup), log logs/next/fix.log
#   FG=1 bash scripts/exp/fix_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/fix.log" 2>&1 &
  echo "fix_lane started (pid $!), log ${LOGDIR}/fix.log"
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
echo "fix_lane on GPU ${G} starts $(date)"
step fix_rocr_nb_s6_3b_f05 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step fix_rocr_nb_s5_1b_f10 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step fix_rocr_nb_s6_1b_f10 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step fix_rocr_nb_s5_3b_f10 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step fix_rocr_nb_s6_3b_f10 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
echo "fix_lane finished $(date)"
