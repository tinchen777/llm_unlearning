#!/bin/bash
# MEOW baseline (src/METHOD/MEOW/meow.py; Gu et al. 2025: the released inverted facts, MEMO-selected, cross-entropy fine-tuning) and the control MEOW -> neighbour (same mechanics, labels = the answers of the
# author's train neighbours), 3B, splits forget01 / forget05 / forget10; every checkpoint is evaluated with the standard evaluation (saves/eval/MEOW/<name>_s<step>/). Needs the authors' released files in data/meow/
# (generated_data/<split>_perturbed_generated.json; the first step only lists them). 40 GB card (full fine-tuning of 3B with AdamW).
# The MEMO scores are computed with the original 3B model (saves/MEOW/memo/, cached; 64 generated tokens per chunk instead of the 200 of the released code, said in the paper).
# A failed step does not stop the lane; a finished step is marked in logs/meow/done/ and skipped on a re-run.
#   bash scripts/exp/meow_lane_3b.sh <GPU>             # background (nohup), log logs/meow/meow_lane_3b.log
#   FG=1 bash scripts/exp/meow_lane_3b.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/meow; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/meow_lane_3b.log" 2>&1 &
  echo "meow_lane_3b started (pid $!), log ${LOGDIR}/meow_lane_3b.log"
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
echo "meow_lane_3b on GPU ${G} starts $(date)"
step meow_data <<'CMD' || true
ls data/meow/generated_data/*_perturbed_generated.json
CMD
step meow_3b_forget01_inverted <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=Llama3B_v2 TARGET=inverted bash scripts/exp/meow.sh ${G}
CMD
step meow_3b_forget01_neighbor <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=Llama3B_v2 TARGET=neighbor bash scripts/exp/meow.sh ${G}
CMD
step meow_3b_forget05_inverted <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 TARGET=inverted bash scripts/exp/meow.sh ${G}
CMD
step meow_3b_forget05_neighbor <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 TARGET=neighbor bash scripts/exp/meow.sh ${G}
CMD
step meow_3b_forget10_inverted <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama3B_v2 TARGET=inverted bash scripts/exp/meow.sh ${G}
CMD
step meow_3b_forget10_neighbor <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama3B_v2 TARGET=neighbor bash scripts/exp/meow.sh ${G}
CMD
echo "meow_lane_3b finished $(date)"
