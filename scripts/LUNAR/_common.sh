# Shared settings of scripts/LUNAR/1_r_uv.sh .. 4_eval.sh. Source it, do not run it.
# The chain r_UV -> layer selection -> W_down -> eval must use the SAME values in every script, so they live here.
# Every value can be overridden from the environment (run_all.sh does that to loop over models / forget sets).

MODEL=${MODEL:-Llama-3.2-1B-Instruct}       # base model name of the checkpoints in configs/model/tofu/
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
# LUNAR unlearns ONE author (the first 20 QAs of the forget split); "train" = the whole forget split
FORGET_HF_SPLIT=${FORGET_HF_SPLIT:-train[:20]}
# reference set D_ref: LUNAR/LUNAR_harmful (fine-tuned models) | LUNAR/LUNAR_unverified (fictitious entities)
REF_DATASET=${REF_DATASET:-LUNAR/LUNAR_harmful}
# Hydra overrides of the splits. An ARRAY, used as "${SPLIT_ARGS[@]}": the value of forget_hf_split contains `[` / `:`,
# which Hydra's override grammar only accepts QUOTED (forget_hf_split='train[:20]'); a plain string variable would lose
# the quotes when the shell word-splits it ("mismatched input '[' expecting <EOF>").
SPLIT_ARGS=(
  "forget_split=${FORGET_SPLIT}" "retain_split=${RETAIN_SPLIT}" "holdout_split=${HOLDOUT_SPLIT}"
  "forget_hf_split='${FORGET_HF_SPLIT}'"
)

# The model r_UV, the layer selection and W_down work on: the one that KNOWS the forget data (the official TOFU full model).
FULL_MODEL=tofu/tofu_${MODEL}_full
TASK_NAME=lunar_tofu_${MODEL}_${FORGET_SPLIT}_$(basename "${REF_DATASET}")     # output: saves/LUNAR/${TASK_NAME}/

COEFF=${COEFF:-1.0}               # forget target = original MLP output + COEFF * r_UV (LUNAR README: 2.0)
SHIFT_MODES=(all eoi)             # all = the paper (same r_UV on every token), eoi = end-of-instruction tokens only
SOLVER=${SOLVER:-adam}            # adam | closed_form (LUNAR Eq. 9, no lr / epochs)

# file name of the layer selection of MODE; same rule as selection_file_name() in src/METHOD/LUNAR/_utils/selection.py
selection_name() { printf "layer_selection_c%g_%s.json" "${COEFF}" "$1"; }
# tag of the W_down outputs of (LAYER, MODE); same rule as `tag` in src/METHOD/LUNAR/w_down.py
w_down_tag() { printf "c%g_l%s_%s%s" "${COEFF}" "$1" "$2" "$([ "${SOLVER}" = closed_form ] && echo _cf)"; }
