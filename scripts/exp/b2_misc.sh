#!/bin/bash
# Batch 2, lane B (card 1, together with b2_ablation.sh): small jobs.
#   1. GA / NPO forget01 seeds 3 and 4 at 1B / 3B (TODO/exp/22 7b, TODO/exp/25: where is seed 42's white-box trace in the seed distribution?). Same commands as next_lane_f.sh.
#      They do NOT enter the main table (cells.py: default seed only); read them with scripts/analysis/seed_spread.py.
#   2. Paraphrased QUESTIONS (TODO/exp/22 13b): scripts/exp/para_eval.sh at 3B, 1B, 8B (inference), then scripts/analysis/para_question_table.py.
#   bash scripts/exp/b2_misc.sh <GPU>      (waits for card 1's last batch-1 lane (b1_8b); NO_WAIT=1 = start now)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b2_misc; source scripts/exp/_lane.sh "$@"
if [ -z "${NO_WAIT}" ]; then echo "waiting for card 1's last batch-1 lane (b1_8b) to finish (NO_WAIT=1 starts at once) ..."; for l in b1_8b; do wait_for $l; done; fi
F01='FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 AUTHOR_OFFSET=199'
for S in 1B 3B; do
  for n in 3 4; do
    step seed${n}_${S}_f01 <<CMD || true
export MODEL=Llama-3.2-${S}-Instruct ${F01}; TAG=_s${n} EXTRA_TRAIN="seed=${n}" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
  done
done
FORCE=1 step seed_spread <<'CMD' || true
python scripts/analysis/seed_spread.py --out saves/analysis/seed_spread.md
CMD
for S in 3B 1B 8B; do
  step para_${S} <<CMD || true
bash scripts/exp/para_eval.sh ${G} ${S}
CMD
done
FORCE=1 step para_table <<'CMD' || true
python scripts/analysis/para_question_table.py
CMD
lane_end
