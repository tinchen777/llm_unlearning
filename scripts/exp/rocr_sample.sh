#!/bin/bash
# Sampling consistency (N=10, T=1: Reuse = cross-question name share, modal name share, self agreement, leak@k) of the ROCR models made by scripts/exp/run_rocr_lanes.sh:
# the original ROCR (`famous`) and ROCR -> neighbour, with the retain and the full model as references (the same model on the holdout questions is the reference inside the table).
# Independent of the lanes (separate file, own markers logs/rocr/done/sample_*): run it on ANY card while / after the lanes run; a setting whose ROCR models do not exist yet is skipped
# (FAIL line, nothing marked), so just run it again later.
#   bash scripts/exp/rocr_sample.sh <GPU>                       # all settings that exist
#   ONLY="1b 3b" bash scripts/exp/rocr_sample.sh <GPU>          # e.g. only the 1B and 3B settings (ONLY=8b: only the 8B ones); SPLITS="forget01" to restrict the split
#   DRY_RUN=1 bash scripts/exp/rocr_sample.sh 0                 # only print the commands
# Output: saves/analysis/rocr_sample_<model>_<split>/summary.md (+ <label>.json); log logs/rocr/sample_<model>_<split>.log
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}
LOGDIR=logs/rocr; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
read -r -a ONLY_A <<< "${ONLY:-1b 3b 8b}"
read -r -a SPLITS_A <<< "${SPLITS:-forget01 forget05 forget10}"
declare -A MODELS=([1b]=Llama-3.2-1B-Instruct [3b]=Llama-3.2-3B-Instruct [8b]=Llama-3.1-8B-Instruct)
declare -A LAYERS=([1b]=2-3-4 [3b]=3-4-5 [8b]=4-5-6)
declare -A HOLD=([forget01]=holdout01 [forget05]=holdout05 [forget10]=holdout10)
declare -A RET=([forget01]=retain99 [forget05]=retain95 [forget10]=retain90)
for sz in "${ONLY_A[@]}"; do
  for f in "${SPLITS_A[@]}"; do
    m=${MODELS[$sz]}; ly=${LAYERS[$sz]}; name=sample_${m}_${f}
    A=saves/ROCR/${m}_${f}_famous_l${ly}/model; B=saves/ROCR/${m}_${f}_neighbor_l${ly}/model
    [ -f "${DONE}/${name}" ] && [ -z "${FORCE}" ] && { echo "[skip] ${name} (done)"; continue; }
    cmd="python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/rocr_${name} --base open-unlearning/tofu_${m}_full --forget ${f} --holdout ${HOLD[$f]} --model retain=hf:open-unlearning/tofu_${m}_${RET[$f]} --model full=hf:open-unlearning/tofu_${m}_full --model ROCR=hf:${A} --model ROCR-nb=hf:${B}"
    if [ -n "${DRY_RUN}" ]; then echo "${name}: ${cmd}"; continue; fi
    if [ ! -d "${A}" ] || [ ! -d "${B}" ]; then echo "[wait] ${name}: ${A} or ${B} does not exist yet"; continue; fi
    echo "[$(date +%T)] START ${name}"
    if ${cmd} > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"; else echo "[$(date +%T)] FAIL  ${name} (see ${LOGDIR}/${name}.log)"; fi
  done
done
