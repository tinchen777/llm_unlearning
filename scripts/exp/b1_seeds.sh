#!/bin/bash
# Batch 1, lane C (card 1, together with b1_blackbox.sh): seeds 1 and 2 of NEAR v2 where the paper needs them (TODO/exp/22 #6, #8, 9b), 3B / 1B trainings only.
#   3B forget10 v2   (#6: is its wb_sup 0.62 a seed effect?)        1B forget10 v2 (#8: error bars)        1B forget05 shared-neighbour target (9b: second source of white-box trace)
# Seeds as in dist_fix_lane2.sh / abl_target_3b.sh: EXTRA_WDOWN seed=<n>, prefix *_v4_s<n> (cells.py reads only *_v4, so the main table is unchanged). Last steps: dist_fix_check tables.
#   bash scripts/exp/b1_seeds.sh <GPU>      Result: saves/analysis/dist_fix_{3B_forget10,1B_forget10,1B_forget05}.md rows "(s1)", "(s2)"
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b1_seeds; source scripts/exp/_lane.sh "$@"
for n in 1 2; do
  step s${n}_3b10 <<CMD || true
${S10} MODEL=Llama-3.2-3B-Instruct KL_STEPS=500 OUT_PREFIX=forget10_Llama3B_v4_s${n} UPSTREAM_PREFIX=forget10_Llama3B_v2 EXTRA_WDOWN="seed=${n} wdown.kl.fr_batch_size=8 wdown.kl.batch_size=16" ${DIST} COEFFS="8" LAYER_SETS="19,20,21" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
  step s${n}_1b10 <<CMD || true
${S10} MODEL=Llama-3.2-1B-Instruct KL_STEPS=500 OUT_PREFIX=forget10_Llama1B_v4_s${n} UPSTREAM_PREFIX=forget10_Llama1B_v2 EXTRA_WDOWN="seed=${n} wdown.kl.fr_batch_size=4 wdown.kl.batch_size=16" ${DIST} COEFFS="7" LAYER_SETS="11,12,13" bash scripts/NEAR/sweep_kl.sh ${G}
CMD
  step s${n}_1b05_shared <<CMD || true
${S05} MODEL=Llama-3.2-1B-Instruct KL_STEPS=300 OUT_PREFIX=forget05_Llama1B_v4_s${n} UPSTREAM_PREFIX=forget05_Llama1B_v2 EXTRA_WDOWN="seed=${n} wdown.kl.batch_size=16" ${DIST} COEFFS="7" LAYER_SETS="11,12,13" KL_TARGET=shared_neighbor bash scripts/NEAR/sweep_kl.sh ${G}
CMD
done
for c in "3B forget10" "1B forget10" "1B forget05"; do
  set -- ${c}
  FORCE=1 step check_$1_$2 <<CMD || true
python scripts/analysis/dist_fix_check.py --size $1 --split $2 --out saves/analysis/dist_fix_$1_$2.md
CMD
done
lane_end
