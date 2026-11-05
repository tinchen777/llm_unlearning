# Shared settings of scripts/unlearn/1_unlearn.sh and run.sh. Source it, do not run it.
# TOFU baselines of OpenUnlearning (src/train.py, mode=unlearn) on the official TOFU-full models, one (model, forget set)
# at a time; run.sh loops over models and forget sets like scripts/NEAR/run.sh does.
#
# Every value can be overridden from the environment, e.g.
#   MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 HOLDOUT_SPLIT=holdout05 RETAIN_SPLIT=retain95 bash scripts/unlearn/1_unlearn.sh

MODEL=${MODEL:-Llama-3.2-1B-Instruct}       # base model name of the checkpoints in configs/model/tofu/
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-neighbor01}   # only for PROBE=1
AUTHOR_OFFSET=${AUTHOR_OFFSET:-199}            # only for PROBE=1: author numbering of the forget rows = `author_id` of the neighbour csv (forget01: 199, forget05: 191, forget10: 181)
FULL_MODEL=tofu/tofu_${MODEL}_full          # the model that KNOWS the forget data = start of every method

# TOFU_EVAL.json of the official retain model of RETAIN_SPLIT: forget quality / privleak reference
RETAIN_LOGS=saves/eval/TOFU/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json

# Hydra overrides of the data. An ARRAY, used as "${SPLIT_ARGS[@]}".
SPLIT_ARGS=(
  "forget_split=${FORGET_SPLIT}" "retain_split=${RETAIN_SPLIT}" "holdout_split=${HOLDOUT_SPLIT}"
  "retain_logs_path=${RETAIN_LOGS}"
)

# PROBE=1 (default): the evaluation the trainer runs at the start and after every epoch also has the NEAR probe, so the baselines
# are comparable with NEAR / LUNAR (scripts/NEAR/4_eval.sh): responses of the neighbour questions (neighbor_*) and the activations
# (activation_*: AUCs of forget | neighbor, seen | unseen, ...). A baseline unlearns towards the retain model: seen = [retain].
# PROBE=0: only the TOFU metrics (+ forget / retain / holdout responses), faster (the probe costs a few forward passes per epoch).
PROBE=${PROBE:-1}
if [ "${PROBE}" = 1 ]; then
  SPLIT_ARGS+=(
    "+eval/tofu_metrics=[neighbor_responses,activation_probe]"
    "+eval.tofu.neighbor_split=${NEIGHBOR_SPLIT}"
    "eval.tofu.metrics.activation_probe.author_offset=${AUTHOR_OFFSET}"
    "eval.tofu.metrics.activation_probe.seen=[retain]"
    "eval.tofu.metrics.activation_probe.unseen=[forget,neighbor,holdout]"
  )
fi

# The methods to run. Override with e.g.  METHODS="GradDiff NPO" bash scripts/unlearn/1_unlearn.sh
#   core (the baselines of the paper): GradAscent GradDiff NPO SimNPO IdkDPO RMU
#   more in src/trainer: CEU WGA UNDIAL SatImp PDU BoundedGradDiff
if [ -z "${METHODS}" ]; then
  # METHODS=(GradAscent GradDiff NPO SimNPO IdkDPO RMU)
  METHODS=(IdkDPO RMU)
else
  read -r -a METHODS <<< "${METHODS}"
fi

# Optional, for tuning a method (the checkpoint is selected by forget quality with the model utility within 5 % of the
# original, see the paper): LR=5e-6 EPOCHS=5 -> trainer.args.* and a suffix in the output folder name.
SUFFIX=""
EXTRA_ARGS=()
if [ -n "${LR}" ]; then EXTRA_ARGS+=("trainer.args.learning_rate=${LR}"); SUFFIX+="_lr${LR}"; fi
if [ -n "${EPOCHS}" ]; then EXTRA_ARGS+=("trainer.args.num_train_epochs=${EPOCHS}"); SUFFIX+="_ep${EPOCHS}"; fi

# Output: saves/tofu_unlearn/<METHOD>/tofu_<MODEL>_<FORGET_SPLIT><SUFFIX>/   (like saves/NEAR/..., grouped by method, then by model and forget
# set; e.g. all the runs of one method are siblings: +vis.run_folders='[[saves/tofu_unlearn/NPO, "*"]]'). `mode` stays `unlearn` (the
# entry points dispatch on it), so the folder is set through paths.output_dir instead of the default saves/<mode>/<task_name>.
task_name() { printf "%s/tofu_%s_%s%s" "$1" "${MODEL}" "${FORGET_SPLIT}" "${SUFFIX}"; }
output_dir() { printf "saves/tofu_unlearn/%s" "$(task_name "$1")"; }

# experiment + trainer + the method's own hyper-parameters. Everything else comes from configs/trainer/<METHOD>.yaml and
# configs/experiment/unlearn/tofu/*.yaml (learning rate 1e-5, 10 epochs, ...): do NOT repeat the defaults here, an override
# like `trainer.method_args.gamma=1.0` silently changes methods whose own default is different (SimNPO 0.125, SatImp 0.1, ...).
method_args() {
  case "$1" in
    IdkDPO)  # DPO with "I don't know" answers as the preferred responses (the idk experiment adds the idk forget data)
      echo "experiment=unlearn/tofu/idk trainer=DPO trainer.args.eval_on_start=False" ;;
    PDU)     # retain_loss_eps has no default
      echo "experiment=unlearn/tofu/default trainer=PDU trainer.method_args.retain_loss_eps=0.3" ;;
    GradAscent|GradDiff|NPO|SimNPO|RMU|CEU|WGA|UNDIAL|SatImp|BoundedGradDiff)
      echo "experiment=unlearn/tofu/default trainer=$1" ;;
    *)
      echo "unknown method '$1'" >&2; return 1 ;;
  esac
}
