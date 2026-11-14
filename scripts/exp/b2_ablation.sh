#!/bin/bash
# Batch 2, lane C (card 1, together with b2_misc.sh; 3B trainings, ~40 GB): component ablations of NEAR (v2) at 3B forget05 (TODO/exp/22 #11). Each run changes ONE part of the v2
# configuration (E8 + retain KL 1, layers 19-21, 300 KL steps, seed 0); the v1-era ablations were all at forget01:
#   npqall   every train neighbour, a random one per step (instead of 3 fixed per question)      mnoff  neighbour-name tokens kept in the KL
#   fp0      no probability term (weight 0)            fpmatch  one common target (`match`)       fpnb   per-question neighbour level, NOT rescaled (`neighbor`)
#   l13      layers 13-15 instead of 19-21             steer    start from the steering + regression solution (KL_INIT=stage1) instead of the original weights
# Last step: dist_fix_check 3B forget05 (rows "Abl ...").
#   bash scripts/exp/b2_ablation.sh <GPU>      (waits for card 1's last batch-1 lane (b1_8b); NO_WAIT=1 = start now)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b2_ablation; source scripts/exp/_lane.sh "$@"
if [ -z "${NO_WAIT}" ]; then echo "waiting for card 1's last batch-1 lane (b1_8b) to finish (NO_WAIT=1 starts at once) ..."; for l in b1_8b; do wait_for $l; done; fi
C='${S05} MODEL=Llama-3.2-3B-Instruct KL_STEPS=300 UPSTREAM_PREFIX=forget05_Llama3B_v2 OUT_PREFIX=forget05_Llama3B_v4 ${DIST} COEFFS="8" LAYER_SETS="19,20,21"'
abl() {  # name, overrides placed AFTER the cell settings (the later assignment wins)
  local name=$1; shift
  step abl_${name} <<CMD || true
$(eval echo "${C}") EXTRA_WDOWN="wdown.kl.fr_batch_size=16 wdown.kl.batch_size=16" $* bash scripts/NEAR/sweep_kl.sh ${G}
CMD
}
abl npqall  KL_NPQ=
abl mnoff   KL_MASK_NAME=0
abl fp0     FP_WEIGHT=0
abl fpmatch FP_MODE=match
abl fpnb    FP_MODE=neighbor
abl l13     LAYER_SETS=13,14,15
abl steer   KL_INIT=stage1
FORCE=1 step check <<'CMD' || true
python scripts/analysis/dist_fix_check.py --size 3B --split forget05 --out saves/analysis/dist_fix_3B_forget05.md
CMD
lane_end
