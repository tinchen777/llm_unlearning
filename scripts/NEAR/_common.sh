# Shared settings of scripts/NEAR/1_r_uv.sh .. 4_eval.sh. Source it, do not run it.
# The chain r_UV -> layer selection -> W_down -> eval must use the SAME values in every script, so they live here.

# Every value below can be overridden from the environment (run_all.sh does that to loop over models / splits), e.g. MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 ... bash scripts/NEAR/1_r_uv.sh

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
# base model name of the checkpoints in configs/model/tofu/

FORGET_SPLIT=${FORGET_SPLIT:-forget01}
NEIGHBOR_SPLIT=${NEIGHBOR_SPLIT:-neighbor01}
RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}

AUTHOR_OFFSET=${AUTHOR_OFFSET:-199}
# ruv.author_offset: makes the author numbering of the forget rows (row // 20) equal to `author_id` of the neighbour csv.
# forget01 = authors 199-200 -> 199. (forget05 / forget10: 191 / 181 if the csv follows the same 1..200 numbering; checked against the csv before any forward pass, r_uv.py stops with the numbers it found otherwise.)

SPLITS="forget_split=${FORGET_SPLIT} neighbor_split=${NEIGHBOR_SPLIT} retain_split=${RETAIN_SPLIT} holdout_split=${HOLDOUT_SPLIT}"

TARGET_MODEL=tofu/tofu_${MODEL}_full
# The model r_UV, the layer selection and W_down work on: the one that KNOWS the forget data (the official TOFU full model).

TASK_NAME=test/near_tofu_${MODEL}_${FORGET_SPLIT}

read -r -a KS <<< "${KS_LIST:-1 5 15}"
# number of neighbours per author used as the reference (must be in ruv.ks); e.g. KS_LIST="5"
COEFF=${COEFF:-1.0}
# forget target = original MLP output + COEFF * r_UV (LUNAR README: 2.0)
read -r -a SHIFT_MODES <<< "${MODES:-eoi all}"
# all = the paper (same r_UV on every token), eoi = end-of-instruction tokens only
SOLVER=${SOLVER:-adam}
# adam | closed_form (LUNAR Eq. 9, no lr / epochs)

# folder of one setting (K, MODE) below saves/NEAR/${TASK_NAME}/ and the W_down run folder of (LAYER) inside it; same rules as config_dir() / run_dir() in src/METHOD/NEAR/_utils/neighbors.py (the layer selection is <config_dir>/layer_selection.json, the edited model <config_dir>/<run_dir>/model)
config_dir() {
    printf "k%s_c%g_%s" "$1" "${COEFF}" "$2"
}
run_dir() {
    printf "l%s%s" "$1" "$([ "${SOLVER}" = closed_form ] && echo _cf)"
}

# tag of the W_down run (only used to name the eval outputs); same rule as `tag` in src/METHOD/NEAR/w_down.py
w_down_tag() {
    printf "k%s_l%s_c%g_%s%s" "$1" "$2" "${COEFF}" "$3" "$([ "${SOLVER}" = closed_form ] && echo _cf)"
}
