#!/bin/bash
# Batch 1, lane E (card 1, AFTER b1_blackbox.sh and b1_seeds.sh: it waits for their last steps): the 8B jobs that need a card of their own.
#   1. seeds 1, 2 of NEAR v2 8B forget05 (#8) and of the 8B shared-neighbour target (9b; 8B wb_sup seed noise ~ +-0.07)
#   2. A2 on NEAR v2 at 8B, real + retain control (#5)
#   bash scripts/exp/b1_8b.sh <GPU>         NO_WAIT=1: start at once (e.g. on a free card)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b1_8b; source scripts/exp/_lane.sh "$@"
if [ -z "${NO_WAIT}" ]; then echo "waiting for b1_blackbox and b1_seeds to finish ..."; wait_for b1_blackbox; wait_for b1_seeds; fi
C8='${S05} MODEL=Llama-3.1-8B-Instruct ${BIG} KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama8B_v2 ${DIST} COEFFS="7" LAYER_SETS="16,17,18"'
for n in 1 2; do
  step s${n}_8b05 <<CMD || true
$(eval echo "${C8}") OUT_PREFIX=forget05_Llama8B_v4_s${n} EXTRA_WDOWN="seed=${n} wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
  step s${n}_8b05_shared <<CMD || true
$(eval echo "${C8}") OUT_PREFIX=forget05_Llama8B_v4_s${n} EXTRA_WDOWN="seed=${n} wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8" KL_TARGET=shared_neighbor bash scripts/NEAR/sweep_kl.sh ${G}
CMD
done
FORCE=1 step check_8B_forget05 <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 8B --split forget05 --out saves/analysis/dist_fix_8B_forget05.md
CMD
for C in real retain; do
  step a2near_8B_${C} <<CMD || true
export GPU=${G} CUDA_VISIBLE_DEVICES=${G}; source scripts/exp/spa_lib.sh; spa_run 8B near ${C}
CMD
done
lane_end
