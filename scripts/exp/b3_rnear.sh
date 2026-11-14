#!/bin/bash
# A2 author-batch control (TODO/exp/22 #9, TODO/exp/27). S (authors 181-190) and F (191-200) are the last TOFU authors; the retained authors R of sets_v2 are drawn from 1-180. A Q2 / P2 / P0
# signal can then come from the author BATCH instead of the edit (1B: Q2 0.84-0.90 already before the edit layers; 3B retain control 0.82). sets_v3 adds R_A_near / R_B_near = retained
# authors with ids 151-180 (closest to S / F; 'Cape Town', a mis-extracted name in R_A of v2, is excluded).
#   1. sets_v3.json (spa.py sets --r-near 151 180): CHECK the printed R names are people.
#   2. CHEAP (no shadow model needed): for every finished A2 condition, add the near names to the TARGET activations (collect --extend) and re-run the attack with R_B = near
#      (Q2 = F vs near retained authors; the probes, trained as before, are unchanged) -> saves/analysis/spa_<scale>_<method>_<cond>_rbnear/
#   3. P0 with R_A = R_B = near (the recognition probe is trained on the target itself, so this one is complete) -> saves/analysis/spa_p0_<scale>_rnear/
#   4. FULL for FULL_CELLS (default "3B/act": the Q2 0.95-0.99 of NEAR-Act at 3B is the claim to check): shadows re-done with the v3 sets (P2's negatives R_A = near too),
#      SPA_DIR=saves/spa_v3 -> saves/analysis/spa_<scale>_<method>_<cond>_rnear/   (~3-6 h per scale and method)
#   bash scripts/exp/b3_rnear.sh <GPU>      FULL_CELLS="3B/act 8B/act" or FULL_CELLS="" ;  RN_METHODS / RN_SCALES restrict steps 2-3 (default: all that exist)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b3_rnear; source scripts/exp/_lane.sh "$@"
export CUDA_VISIBLE_DEVICES=${G} GPU=${G}
V3=saves/analysis/spa/sets_v3.json
step sets_v3 <<CMD || true
[ -f ${V3} ] || python scripts/analysis/spa.py sets --out ${V3} --r-near 151 180
CMD
SCALES=${RN_SCALES:-"3B 8B 1B"}; METHODS=${RN_METHODS:-"act near lunar rocr rmu ga npo dist"}
for S in ${SCALES}; do
  base=$(case ${S} in 1B) echo open-unlearning/tofu_Llama-3.2-1B-Instruct;; 3B) echo open-unlearning/tofu_Llama-3.2-3B-Instruct;; 8B) echo open-unlearning/tofu_Llama-3.1-8B-Instruct;; esac)
  # the retain95 target is shared by every method's retain condition: extend it once
  step ext_${S}_retain95 <<CMD || true
f=saves/spa/${S}_retain95_target.npz; [ -f \$f ] || { echo "no \$f"; exit 0; }; python scripts/analysis/spa.py collect --extend --sets ${V3} --model hf:${base}_retain95 --base ${base}_full --out \$f
CMD
  for m in ${METHODS}; do
    for C in real retain; do
      d=saves/spa/${S}_${m}_${C}
      if [ -z "${DRY_RUN}" ] && { [ ! -f ${d}/target.npz ] || [ ! -f ${d}/shadow4.npz ]; }; then continue; fi
      step rbnear_${S}_${m}_${C} <<CMD || true
source scripts/exp/spa_lib.sh; spa_cfg ${S} ${m}
if [ ${C} = real ]; then python scripts/analysis/spa.py collect --extend --sets ${V3} --model hf:saves/spa/models/${S}_${m}_mprime --base ${base}_full --out ${d}/target.npz
else cp saves/spa/${S}_retain95_target.npz ${d}/target.npz; fi
python scripts/analysis/spa.py attack --sets ${V3} --rb-pool near --shadow ${d}/shadow0.npz ${d}/shadow1.npz ${d}/shadow2.npz ${d}/shadow3.npz ${d}/shadow4.npz --target ${d}/target.npz \\
  --out saves/analysis/spa_${S}_${m}_${C}_rbnear --kinds lr dm \${LAYERS:+--edit-layers \${LAYERS//,/ }} --title "A2 ${S} ${m} (${C}), R_B = near retained authors"
CMD
    done
  done
  # P0: every model of the a2_p0 lane (its list logs/a2p0/models_<S>.txt), extended and re-run with both R pools near
  step p0near_${S} <<CMD || true
L=logs/a2p0/models_${S}.txt; [ -f \$L ] || { echo "no \$L (run a2_p0_lane.sh first)"; exit 0; }
ACTS=()
while IFS='|' read -r label spec el; do
  [ -z "\$label" ] && continue; f=saves/spa/p0/${S}/\$label.npz; [ -f \$f ] || continue
  python scripts/analysis/spa.py collect --extend --sets ${V3} --model "\$spec" --base ${base}_full --out \$f < /dev/null || continue
  ACTS+=("\$label=\$f\${el:+#\$el}")
done < \$L
python scripts/analysis/spa.py p0 --sets ${V3} --ra-pool near --rb-pool near --acts "\${ACTS[@]}" --out saves/analysis/spa_p0_${S}_rnear
CMD
done
for cell in ${FULL_CELLS-3B/act}; do
  S=${cell%/*}; m=${cell#*/}
  for C in real retain; do
    step full_${S}_${m}_${C} <<CMD || true
export SPA_SETS_V=v3 SPA_DIR=saves/spa_v3 SPA_OUT_SUFFIX=_rnear SPA_ATTACK_ARGS="--ra-pool near --rb-pool near"; source scripts/exp/spa_lib.sh; spa_run ${S} ${m} ${C}
CMD
  done
done
lane_end
