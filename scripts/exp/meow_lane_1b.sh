#!/bin/bash
# MEOW baseline (src/METHOD/MEOW/meow.py) and the control MEOW -> neighbour (same mechanics, labels = the answers of the author's train neighbours), 1B, forget01 then forget05, learning rates 1e-4 and 3e-5.
# The first check (1B, forget01, lr 1e-5) showed the implementation works (loss 1.8 -> 0.08, the model answers with the inverted facts) but the Prob of the true answers only fell to 0.48 in 80 steps: the
# learning rate is raised (1e-4 and 3e-5) to look for the matching region (Prob <= 0.25). Every checkpoint is evaluated (saves/eval/MEOW/<name>_s<step>/), its weights deleted afterwards: the matched
# configuration is re-trained (cheap, same seed) with its weights kept for the attacks. One of two processes on an 80 GB card (about 12 GB).  Needs data/meow/generated_data/ (the authors' released inverted facts).
# A failed step does not stop the lane; a finished step is marked in logs/meow/done/ and skipped on a re-run.
#   bash scripts/exp/meow_lane_1b.sh <GPU>             # background (nohup), log logs/meow/meow_lane_1b.log
#   FG=1 bash scripts/exp/meow_lane_1b.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/meow; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/meow_lane_1b.log" 2>&1 &
  echo "meow_lane_1b started (pid $!), log ${LOGDIR}/meow_lane_1b.log"
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
echo "meow_lane_1b on GPU ${G} starts $(date)"
step meow_data <<'CMD' || true
ls data/meow/generated_data/*_perturbed_generated.json
CMD
step meow_1b_forget01_inverted_lr1e-4 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=single_v2 TARGET=inverted LR=1e-4 TAG=_lr1e-4 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget01_neighbor_lr1e-4 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=single_v2 TARGET=neighbor LR=1e-4 TAG=_lr1e-4 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget01_inverted_lr3e-5 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=single_v2 TARGET=inverted LR=3e-5 TAG=_lr3e-5 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget01_neighbor_lr3e-5 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199 R_UV_PREFIX=single_v2 TARGET=neighbor LR=3e-5 TAG=_lr3e-5 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget05_inverted_lr1e-4 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=inverted LR=1e-4 TAG=_lr1e-4 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget05_neighbor_lr1e-4 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=neighbor LR=1e-4 TAG=_lr1e-4 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget05_inverted_lr3e-5 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=inverted LR=3e-5 TAG=_lr3e-5 bash scripts/exp/meow.sh ${G}
CMD
step meow_1b_forget05_neighbor_lr3e-5 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=neighbor LR=3e-5 TAG=_lr3e-5 bash scripts/exp/meow.sh ${G}
CMD
echo "meow_lane_1b finished $(date)"
