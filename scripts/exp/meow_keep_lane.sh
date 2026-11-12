#!/bin/bash
# Re-train the MEOW / MEOW -> neighbour configurations that the attacks use, KEEPING the weights of the one step (the sweep of meow_lane_*.sh deleted the weights after the evaluation); same seed and data, so
# the model should be the one of the sweep (the evaluation is repeated: check forget_Q_A_Prob of saves/eval/MEOW/<name>_s<step> against the sweep, bf16 training is not bit-exact).
# Configurations (lr 3e-5 everywhere): forget05 1B MEOW step 200 (matched, Prob 0.24), 1B MEOW-nb step 50 (0.23), 3B MEOW step 200 (Prob 0.29: the CLOSEST point, no match), 3B MEOW-nb step 200 (0.22).
# 4 steps (a few minutes of training + one evaluation each); about 3 + 3 + 7 + 7 GB of weights are kept in saves/MEOW/<name>/checkpoint-<step>.
#   bash scripts/exp/meow_keep_lane.sh <GPU>             # background (nohup), log logs/next/meow_keep_lane.log
#   FG=1 bash scripts/exp/meow_keep_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/meow_keep_lane.log" 2>&1 &
  echo "meow_keep_lane started (pid $!), log ${LOGDIR}/meow_keep_lane.log"
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
echo "meow_keep_lane on GPU ${G} starts $(date)"
step meowkeep_1b_f05_inverted <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=inverted LR=3e-5 TAG=_lr3e-5 SAVE_STEPS="200" KEEP_STEPS="200" bash scripts/exp/meow.sh ${G}
CMD
step meowkeep_1b_f05_neighbor <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 TARGET=neighbor LR=3e-5 TAG=_lr3e-5 SAVE_STEPS="50" KEEP_STEPS="50" bash scripts/exp/meow.sh ${G}
CMD
step meowkeep_3b_f05_inverted <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 TARGET=inverted LR=3e-5 TAG=_lr3e-5 SAVE_STEPS="200" KEEP_STEPS="200" bash scripts/exp/meow.sh ${G}
CMD
step meowkeep_3b_f05_neighbor <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 TARGET=neighbor LR=3e-5 TAG=_lr3e-5 SAVE_STEPS="200" KEEP_STEPS="200" bash scripts/exp/meow.sh ${G}
CMD
echo "meow_keep_lane finished $(date)"
