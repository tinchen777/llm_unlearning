#!/bin/bash
# Redo of what the stale exported variables spoiled in lane f (the baseline seeds ran on forget05 instead of forget01) + ROCR -> neighbour with the residual factors 5 / 6 at forget05
# (forget10 gave Prob 0.31 / 0.08 at 5 / 6 for 1B and 0.31 / 0.09 for 3B; factor 8 collapses the model at forget05 and forget10).  Every command sets its splits itself.
# 8 steps for a 40 GB card; a failed step does not stop the lane; a finished step is marked in logs/next/done/ and skipped on a re-run.
#   bash scripts/exp/next_lane_f.sh <GPU>             # background (nohup), log logs/next/f.log
#   FG=1 bash scripts/exp/next_lane_f.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/f.log" 2>&1 &
  echo "lane f started (pid $!), log ${LOGDIR}/f.log"
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
echo "lane f on GPU ${G} starts $(date)"
step c3b_baseline_seed1_1b_f01 <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199; TAG=_s1 EXTRA_TRAIN="seed=1" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3b_baseline_seed2_1b_f01 <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199; TAG=_s2 EXTRA_TRAIN="seed=2" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3b_baseline_seed1_3b_f01 <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199; TAG=_s1 EXTRA_TRAIN="seed=1" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3b_baseline_seed2_3b_f01 <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199; TAG=_s2 EXTRA_TRAIN="seed=2" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step rocr_nb_s5_1b_f05 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_1b_f05 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s5_3b_f05 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_3b_f05 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
echo "lane f finished $(date)"
