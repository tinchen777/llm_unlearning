#!/bin/bash
# =============================================================================
# The reference models of the NEAR probe: the official TOFU `full` model (knows the forget data) and the official `retain` model
# (never saw it = the ideal result of unlearning), for every (base model, forget set). ONE evaluation each (src/eval.py,
# experiment=eval/tofu/probe): TOFU metrics + responses of forget / retain / holdout / neighbor + activations (per-layer AUCs, r_UV).
#
#   bash scripts/eval/baseline_probe.sh [GPU_ID]
#   KINDS="full" bash scripts/eval/baseline_probe.sh 0       # only some kinds
#   ONLY_SPLITS="forget005 forget01" bash scripts/eval/baseline_probe.sh 0   # only some forget sets
#   ONLY_MODELS="Llama-3.2-1B-Instruct" bash scripts/eval/baseline_probe.sh 0   # only some base models
#   DRY_RUN=1 bash scripts/eval/baseline_probe.sh            # only print what would run
#   SKIP_EXISTING=1 bash scripts/eval/baseline_probe.sh      # skip the runs that already have a TOFU_SUMMARY.json
#   CONTINUE_ON_ERROR=1 bash scripts/eval/baseline_probe.sh  # report the failures at the end instead of stopping
#
# Output: saves/eval/<KIND>/<MODEL>/<FORGET_SPLIT>/   KIND = full | retain      (TOFU_SUMMARY.json, TOFU_EVAL.json, *_responses.json,
#   activations_summary.json, activations.pt). Compare with the edited models of scripts/NEAR/4_eval.sh by SUBTRACTING the same
#   pair of splits (full - edited, retain - edited): the difference removes the content effect and leaves the knowledge effect.
#
# The activations use the defaults (end-of-instruction tokens, every layer, block_out) = those of r_uv.py, so they line up with the
# NEAR runs layer by layer. The `seen | unseen` AUC: full = [forget, retain] | [neighbor, holdout]; retain = [retain] | [forget,
# neighbor, holdout]. The forget authors are numbered with AUTHOR_OFFSET so that they match `author_id` of the neighbour csv (needed
# by the group-wise AUC, which needs >= 4 authors: forget05 and up).
# Needs data/tofu/tofu_<NEIGHBOR_SPLIT>.csv and, for forget quality / privleak, the retain logs (RETAIN_LOGS_PATH, see below).
# =============================================================================
set -e
cd "$(dirname "$0")/../.." || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

BASE_MODELS=(
  Llama-3.2-1B-Instruct
  Llama-3.2-3B-Instruct
  Llama-3.1-8B-Instruct
)
# ONLY_MODELS="Llama-3.2-1B-Instruct" runs just these base models (space separated)
if [ -n "${ONLY_MODELS}" ]; then read -r -a BASE_MODELS <<< "${ONLY_MODELS}"; fi

# "forget_split holdout_split retain_split neighbor_split author_offset"
SPLIT_SETS=(
  "forget01 holdout01 retain99 neighbor01 199"
  "forget05 holdout05 retain95 neighbor05 191"
  # ONE author (author 200 = the last 20 rows of forget01). Its retain model is the official retain99 one (it never saw these authors
  # either), there is no retain995: forget quality / privleak are not meaningful for it. ONLY_SPLITS="forget005" runs just this one.
  "forget005 holdout01 retain99 neighbor005 200"
  "forget10 holdout10 retain90 neighbor10 181"   # ONLY_SPLITS="forget10" runs just this one (the references of the forget10 table)
)

read -r -a KINDS <<< "${KINDS:-retain full}"   # retain first: its probe is the clean reference of the detectability numbers of the others

failed=()
for base in "${BASE_MODELS[@]}"; do
  for split in "${SPLIT_SETS[@]}"; do
    read -r forget holdout retain neighbor offset <<< "${split}"
    if [ -n "${ONLY_SPLITS}" ] && [[ " ${ONLY_SPLITS} " != *" ${forget} "* ]]; then continue; fi
    # TOFU_EVAL.json of the retain model of this split: the reference of forget quality / privleak
    retain_logs=${RETAIN_LOGS_PATH:-saves/eval/TOFU/tofu_${base}_${retain}/TOFU_EVAL.json}
    for kind in "${KINDS[@]}"; do
      case "${kind}" in
        full)   model=tofu/tofu_${base}_full
                seen="[forget,retain]"; unseen="[neighbor,holdout]" ;;
        retain) model=tofu/tofu_${base}_${retain}
                seen="[retain]"; unseen="[forget,neighbor,holdout]" ;;
        *) echo "unknown kind '${kind}' (full | retain)" >&2; exit 1 ;;
      esac
      task=${kind}/${base}/${forget}
      combo="${kind} ${base} ${forget}"
      echo "================ ${combo} -> saves/eval/${task} ================"
      if [ -n "${SKIP_EXISTING}" ] && [ -f "saves/eval/${task}/TOFU_SUMMARY.json" ]; then
        echo "---------------- ${combo}: skipped (exists)"
        continue
      fi
      cmd=(python src/eval.py
        experiment=eval/tofu/probe
        model=${model}
        forget_split=${forget} holdout_split=${holdout} retain_split=${retain} neighbor_split=${neighbor}
        eval.tofu.retain_logs_path=${retain_logs}
        eval.tofu.metrics.activation_probe.author_offset=${offset}
        "eval.tofu.metrics.activation_probe.seen=${seen}"
        "eval.tofu.metrics.activation_probe.unseen=${unseen}"
        task_name=${task})
      if [ -n "${DRY_RUN}" ]; then
        echo "${cmd[@]}"
        continue
      fi
      # NOTE: no `if ( ... )` / `|| ...` around the call: bash would switch `set -e` off inside it
      set +e
      "${cmd[@]}"
      status=$?
      set -e
      if [ ${status} -eq 0 ]; then
        echo "---------------- ${combo}: done"
      else
        failed+=("${combo}")
        echo "---------------- ${combo}: FAILED (exit ${status})" >&2
        [ -z "${CONTINUE_ON_ERROR}" ] && exit ${status}
      fi
    done
  done
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "failed: ${failed[*]}" >&2
  exit 1
fi
echo "################ all done ################"
