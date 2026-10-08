# Shared settings of scripts/NEAR/1_r_uv.sh .. 4_eval.sh. Source it, do not run it.
# The chain r_UV -> layer selection -> W_down -> eval must use the SAME values in every script, so they live here.

MODEL=Llama-3.2-1B-Instruct       # base model name of the checkpoints in configs/model/tofu/
# Start small: LUNAR's TOFU experiments unlearn ONE author (20 QAs); forget01 = 2 authors, forget10 = 20 authors.
FORGET_SPLIT=forget01
NEIGHBOR_SPLIT=neighbor01
RETAIN_SPLIT=retain99
HOLDOUT_SPLIT=holdout01
SPLITS="forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}"

# The model r_UV, the layer selection and W_down work on: the one that KNOWS the forget data (the official TOFU full model).
FULL_MODEL=tofu/tofu_${MODEL}_full
TASK_NAME=redirect_${MODEL}_${FORGET_SPLIT}     # output: saves/NEAR/${TASK_NAME}/

KS=(1 5 15)                       # number of neighbours per author used as the reference (must be in ruv.ks)
COEFF=1.0                         # forget target = original MLP output + COEFF * r_UV (LUNAR README: 2.0)
SHIFT_MODES=(eoi all)             # all = the paper (same r_UV on every token), eoi = end-of-instruction tokens only
SOLVER=adam                       # adam | closed_form (LUNAR Eq. 9, no lr / epochs)

# file name of the layer selection of (K, MODE); same rule as selection_file_name() in src/model/redirect.py
selection_name() { printf "layer_selection_k%s_c%g_%s.json" "$1" "${COEFF}" "$2"; }
# tag of the W_down outputs of (K, LAYER, MODE); same rule as `tag` in src/METHOD/NEAR/w_down.py
w_down_tag() { printf "k%s_l%s_c%g_%s%s" "$1" "$2" "${COEFF}" "$3" "$([ "${SOLVER}" = closed_form ] && echo _cf)"; }
