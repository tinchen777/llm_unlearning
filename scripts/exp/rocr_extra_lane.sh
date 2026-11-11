#!/bin/bash
# ROCR -> neighbour with more residual factors, to find the matching region (Prob <= 0.25) where it is still missing: factor 7 at forget01 (1B, 3B: factor 6 gave Prob 0.38 / 0.54, factor 8 collapses the 1B
# model), and the 8B model (so far only factor 8 at forget01): forget01 factors 6, 7, forget05 and forget10 factor 6 (the factor that reached the matching region at 1B / 3B forget05 and forget10).
# 6 steps; 1B / 3B need little memory, the 8B steps (last) about 25 GB: fine next to two small processes on an 80 GB card. A failed step does not stop the lane; a finished step is marked in logs/next/done/.
#   bash scripts/exp/rocr_extra_lane.sh <GPU>             # background (nohup), log logs/next/rocr_extra.log
#   FG=1 bash scripts/exp/rocr_extra_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/rocr_extra.log" 2>&1 &
  echo "rocr_extra_lane started (pid $!), log ${LOGDIR}/rocr_extra.log"
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
echo "rocr_extra_lane on GPU ${G} starts $(date)"
step rocr_nb_s7_1b_f01 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=7 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s7_3b_f01 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=7 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_8b_f01 <<'CMD' || true
MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=Llama8B_v2 LAYERS="4 5 6" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s7_8b_f01 <<'CMD' || true
MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=Llama8B_v2 LAYERS="4 5 6" TARGET=neighbor SCALE=7 EVAL_PROFILE=fast PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_8b_f05 <<'CMD' || true
MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama8B_v2 LAYERS="4 5 6" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_8b_f10 <<'CMD' || true
MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama8B_v2 LAYERS="4 5 6" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" bash scripts/exp/rocr.sh ${G}
CMD
echo "rocr_extra_lane finished $(date)"
