#!/bin/bash
# =============================================================================
# Demo: ONE evaluation (src/eval.py, experiment=eval/tofu/probe) that gives
#   - the TOFU metrics (forget quality, model utility, privleak, MIA, ...)
#   - the responses of forget / retain / holdout / neighbor and their text metrics
#     (forget_hallucination_rate, neighbor_refusal_rate, ... : `expose` in configs/eval/tofu_metrics/*_responses.yaml)
#   - the activations (r_UV, norms, AUCs; `activation_probe`) and the metrics exposed from them
#     (activation_auc_forget_neighbor_max, ...)
#
#   bash scripts/eval/demo_probe.sh [GPU_ID] [MODEL_PATH]
#   bash scripts/eval/demo_probe.sh 0                                      # the official TOFU full model
#   bash scripts/eval/demo_probe.sh 0 saves/unlearn/ttestt/GradAscent      # the model of scripts/unlearn/demo_1.sh
#
# Optional (environment variables):
#   R_UV=saves/NEAR/<task>/r_uv.pt   positions / layers / point / author numbering of the probe are read from that r_uv.pt
#                                    (and its forget split / rows are checked), as in scripts/NEAR/4_eval.sh
#   FOCUS_LAYER=7                    layer whose AUCs are also published (`activation_auc_..._focus`)
#   MODEL_KIND=full|unlearned        which sets the model should treat as seen for the `seen | unseen` AUC
#                                    (activation_auc_seen_unseen_max): full = [forget, retain] | [neighbor, holdout] (default when no
#                                    MODEL_PATH), unlearned = [retain] | [forget, neighbor, holdout] (default with a MODEL_PATH; also
#                                    for the retain model)
#
# Needs: data/tofu/tofu_neighbor01.csv (the neighbours of NEAR), TOFU_EVAL.json of the retain model for forget quality / privleak
# (eval.tofu.retain_logs_path, default saves/eval/TOFU/tofu_<MODEL>_<retain split>/TOFU_EVAL.json; without it they are None).
#
# Output: saves/eval/ttestt/probe_demo/
#   TOFU_SUMMARY.json   one number per metric  (look for  forget_hallucination_rate, activation_auc_forget_neighbor_max)
#   TOFU_EVAL.json      the same + per-sample values (value_by_index) + the per-layer table (activation_probe.value_by_layer)
#   forget_responses.json, retain_..., holdout_..., neighbor_...   one record per question
#   activations_summary.json, activations.pt                        the per-layer diagnostics
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}

MODEL=Llama-3.2-1B-Instruct
MODEL_PATH=${2:-}
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99
HOLDOUT_SPLIT=holdout01

RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}
MODEL_KIND=${MODEL_KIND:-$([ -n "${MODEL_PATH}" ] && echo unlearned || echo full)}

EXTRA_ARGS=("eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}")
case "${MODEL_KIND}" in
  full) ;;  # the default of configs/eval/tofu_metrics/activation_probe.yaml
  unlearned)
    EXTRA_ARGS+=("eval.tofu.metrics.activation_probe.seen=[retain]")
    EXTRA_ARGS+=("eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]")
    ;;
  *) echo "MODEL_KIND must be full or unlearned, got '${MODEL_KIND}'" >&2; exit 1 ;;
esac
if [ -n "${MODEL_PATH}" ]; then
  EXTRA_ARGS+=("model.pretrained.name_or_path=${MODEL_PATH}")
fi
if [ -n "${R_UV}" ]; then
  EXTRA_ARGS+=("eval.tofu.metrics.activation_probe.r_uv_path=${R_UV}")
fi
if [ -n "${FOCUS_LAYER}" ]; then
  EXTRA_ARGS+=("eval.tofu.metrics.activation_probe.focus_layer=${FOCUS_LAYER}")
fi

echo start probe demo ${MODEL_PATH:-tofu_${MODEL}_full} \(${MODEL_KIND}\)
python src/eval.py \
  experiment=eval/tofu/probe \
  model=tofu/tofu_${MODEL}_full \
  forget_split=${FORGET_SPLIT} \
  neighbor_split=${NEIGHBOR_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  holdout_split=${HOLDOUT_SPLIT} \
  "${EXTRA_ARGS[@]}" \
  task_name=ttestt/probe_demo
echo end probe demo
echo "summary: saves/eval/ttestt/probe_demo/TOFU_SUMMARY.json"
