#!/bin/bash
# ROCR baseline (TODO/exp/17): two sequential lanes, BIG = 80 GB card (8B), SMALL = 40 GB card (1B / 3B). Same mechanics as scripts/exp/run_lanes.sh:
# a failed step does not stop the lane, a finished step is marked in logs/rocr/done/ and skipped on a re-run (same command resumes), one log per step.
#
#   bash scripts/exp/run_rocr_lanes.sh <BIG_GPU> <SMALL_GPU>        # both lanes in the background (nohup), logs in logs/rocr/
#   LANE=big bash scripts/exp/run_rocr_lanes.sh 0 1                 # only the big lane (LANE=small: only the small lane)
#   LANE=all bash scripts/exp/run_rocr_lanes.sh 0                   # ONE chain on one card: the big lane, then the small lane
#   FG=1 LANE=small bash scripts/exp/run_rocr_lanes.sh 0 1          # in the foreground
#   DRY_RUN=1 bash scripts/exp/run_rocr_lanes.sh 0 1                # print the steps only
#   FORCE=1 ...                                                     # ignore the done markers
# Watch: tail -f logs/rocr/big.log logs/rocr/small.log ; one log per step: logs/rocr/<step>.log. Results: saves/ROCR/<name>/ (edited model, rocr_summary.json),
# saves/eval/ROCR/<name>/ (evaluation). The null-space projectors are cached in data/rocr/<model>/ by the first run of every model, so the order below matters.
#
# Per setting: `famous` = the original ROCR (Donald Trump), `neighbor` = only the target changed (train_rank 1 neighbour). Extra rows (EVAL_PROFILE=fast): a layer
# scan of `neighbor` and the residual factor SCALE = 3, 8 (NOT ROCR; shows how much more than one full redirection is needed).
cd "$(dirname "$0")/../.." || exit 1
BIG_GPU=${1:-0}; SMALL_GPU=${2:-1}
LOGDIR=logs/rocr; DONE=${LOGDIR}/done; mkdir -p "${DONE}"

if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  for l in big small; do
    if [ "${LANE}" = all ]; then l=all; fi
    if [ -n "${LANE}" ] && [ "${LANE}" != "${l}" ]; then continue; fi
    LANE_INNER=${l} nohup bash "$0" "${BIG_GPU}" "${SMALL_GPU}" > "${LOGDIR}/${l}.log" 2>&1 &
    echo "lane ${l} started (pid $!), log ${LOGDIR}/${l}.log"
    if [ "${l}" = all ]; then break; fi
  done
  exit 0
fi

step() {  # step <name> <command string>; non-zero if it failed
  local name=$1; shift
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    $*"; return 0; fi
  if ( eval "$*" ) > "${LOGDIR}/${name}.log" 2>&1; then
    touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else
    echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1
  fi
}

E1="MODEL=Llama-3.2-1B-Instruct"; E3="MODEL=Llama-3.2-3B-Instruct"
E8="MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL='eval.tofu.batch_size=4'"
S01="FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199"
S05="FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191"
S10="FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181"
RU01="R_UV_PREFIX=single_v2"   # 1B forget01; 3B / 8B forget01: Llama3B_v2 / Llama8B_v2; forget05 / forget10: forget05_Llama{1,3,8}B_v2 / forget10_Llama{1,3,8}B_v2
R() {  # R <model env> <split env> <r_uv prefix> <layers> <target> [extra env]
  echo "export $1 $2; R_UV_PREFIX=$3 LAYERS='$4' TARGET=$5 $6 bash scripts/exp/rocr.sh \${G}"
}

# ======================================= BIG lane (80 GB): 8B =======================================
big() {
  G=${BIG_GPU}
  L8="4 5 6"
  for t in famous neighbor; do step f01_8b_${t} "$(R "${E8}" "${S01}" Llama8B_v2 "${L8}" ${t})" || true; done
  for s in 3 8; do step f01_8b_neighbor_s${s} "$(R "${E8}" "${S01}" Llama8B_v2 "${L8}" neighbor "SCALE=${s} EVAL_PROFILE=fast")" || true; done
  step f01_8b_neighbor_l8-9-10 "$(R "${E8}" "${S01}" Llama8B_v2 "8 9 10" neighbor "EVAL_PROFILE=fast")" || true
  for t in famous neighbor; do step f05_8b_${t} "$(R "${E8}" "${S05}" forget05_Llama8B_v2 "${L8}" ${t})" || true; done
  for t in famous neighbor; do step f10_8b_${t} "$(R "${E8}" "${S10}" forget10_Llama8B_v2 "${L8}" ${t})" || true; done
}

# ======================================= SMALL lane (40 GB): 1B, 3B =======================================
small() {
  G=${SMALL_GPU}
  # --- 1B forget01 (first: also fills the projector cache of layers 2..10)
  for t in famous neighbor; do step f01_1b_${t} "$(R "${E1}" "${S01}" single_v2 "2 3 4" ${t})" || true; done
  for L in "5 6 7" "8 9 10"; do step f01_1b_neighbor_l${L// /-} "$(R "${E1}" "${S01}" single_v2 "${L}" neighbor "EVAL_PROFILE=fast")" || true; done
  for s in 3 8; do step f01_1b_neighbor_s${s} "$(R "${E1}" "${S01}" single_v2 "2 3 4" neighbor "SCALE=${s} EVAL_PROFILE=fast")" || true; done
  # --- 3B forget01
  for t in famous neighbor; do step f01_3b_${t} "$(R "${E3}" "${S01}" Llama3B_v2 "3 4 5" ${t})" || true; done
  for L in "6 7 8" "10 11 12"; do step f01_3b_neighbor_l${L// /-} "$(R "${E3}" "${S01}" Llama3B_v2 "${L}" neighbor "EVAL_PROFILE=fast")" || true; done
  for s in 3 8; do step f01_3b_neighbor_s${s} "$(R "${E3}" "${S01}" Llama3B_v2 "3 4 5" neighbor "SCALE=${s} EVAL_PROFILE=fast")" || true; done
  # --- forget05 (1B, 3B), then forget10 (1B, 3B)
  for t in famous neighbor; do step f05_1b_${t} "$(R "${E1}" "${S05}" forget05_Llama1B_v2 "2 3 4" ${t})" || true; done
  for t in famous neighbor; do step f05_3b_${t} "$(R "${E3}" "${S05}" forget05_Llama3B_v2 "3 4 5" ${t})" || true; done
  for t in famous neighbor; do step f10_1b_${t} "$(R "${E1}" "${S10}" forget10_Llama1B_v2 "2 3 4" ${t})" || true; done
  for t in famous neighbor; do step f10_3b_${t} "$(R "${E3}" "${S10}" forget10_Llama3B_v2 "3 4 5" ${t})" || true; done
}

l=${LANE_INNER:-${LANE:-big}}
if [ "${l}" = all ]; then SMALL_GPU=${BIG_GPU}; echo "lane all starts $(date)"; big; small; echo "lane all finished $(date)"; exit 0; fi
if [ -n "${DRY_RUN}" ] && [ -z "${LANE}" ]; then echo "##### BIG (GPU ${BIG_GPU})"; big; echo "##### SMALL (GPU ${SMALL_GPU})"; small; exit 0; fi
echo "lane ${l} starts $(date)"; ${l}; echo "lane ${l} finished $(date)"
