#!/bin/bash
# Functions of the A2 shadow-probe attack chains (TODO/exp/A2_SPEC.md). Source it, do not run it.  Used by spa_lane_*.sh.
#   spa_cfg SCALE METHOD      -> MODEL COEFF LAYERS TARGET_SPEC DIST KIND (provider's configuration at forget05; the attacker knows the algorithm and its hyper-parameters)
#                                METHOD: act (NEAR-Act) | near (NEAR = NEAR-Dist v2) | dist (NEAR-Dist v1) | lunar (LUNAR) | rocr (ROCR, original: Donald Trump) | rmu (RMU) | ga (GradAscent) | npo (NPO)   (TODO/exp/23, 26)
#   spa_run SCALE METHOD COND -> everything of one condition: M' (COND=real) or the retain95 model (COND=retain, the negative control), 5 shadow edits of 2 authors each, activations, attack.
# Shadow edit of group k: the SAME algorithm on M' with the forget set = rows [40k, 40k+40) of forget10 (= authors 181+2k, 182+2k), their own train neighbours, retain90, provider's layers and
# coefficient. Nothing of F, of the provider's neighbours or of the full model M0 is read. The edited model is deleted after its activations are taken.
# Env: SPA_COEFFS (grid of NEAR-Act / LUNAR), SPA_RMU_EPOCHS, SPA_RMU_EXTRA (Hydra overrides of the RMU trainer, memory), GPU (default 0), SPA_GROUPS (default "0 1 2 3 4"), DRY_RUN=1, FORCE=1 (ignore done markers), SPA_DIR (activations, default saves/spa), OUT_DIR (default saves/analysis),
#      KEEP_EDITED=1 (keep the edited shadow models, ~6 GB each at 3B: activations of new names can then be taken without re-editing).
# Sets v2 (sets_v2.json) add the retained authors R_A / R_B; activation files of v1 lack them, so the activation markers carry the version (.v2.done) and an edit whose model was
# deleted is re-done automatically when its activations are needed.
# The attack marker carries .rules (layer rules + name-length control, TODO/exp/21): an attack done before that is re-run from the existing activations (CPU only).

SPA_DIR=${SPA_DIR:-saves/spa}
OUT_DIR=${OUT_DIR:-saves/analysis}
SETS_V=${SPA_SETS_V:-v2}                     # v3 = + the near retained authors (TODO/exp/27); use with its own SPA_DIR, SPA_OUT_SUFFIX and SPA_ATTACK_ARGS
SETS=${OUT_DIR}/spa/sets_${SETS_V}.json
SPA_OUT_SUFFIX=${SPA_OUT_SUFFIX:-}           # e.g. _rnear: the attack writes saves/analysis/spa_<scale>_<method>_<cond>_rnear
SPA_ATTACK_ARGS=${SPA_ATTACK_ARGS:-}         # e.g. "--ra-pool near --rb-pool near"
read -r -a SPA_GROUPS_A <<< "${SPA_GROUPS:-0 1 2 3 4}"
GPU=${GPU:-0}

spa_cfg() {
  DIST=0; DIST_EXTRA=""; COEFF=""; RMU_LR=""; TRAINER=RMU; RMU_EXTRA=${SPA_RMU_EXTRA:-}     # (reset: the lanes call spa_cfg for several methods in one shell)
  case "$1/$2" in
    1B/act)  MODEL=Llama-3.2-1B-Instruct; COEFF=7;  LAYERS=7,8,9;    TARGET_SPEC=near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9 ;;
    1B/dist) MODEL=Llama-3.2-1B-Instruct; COEFF=7;  LAYERS=11,12,13; DIST=1; KL_STEPS=${KL_STEPS:-300}; TARGET_SPEC=near:saves/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 ;;
    3B/act)  MODEL=Llama-3.2-3B-Instruct; COEFF=14; LAYERS=12,13,14; TARGET_SPEC=near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14 ;;
    3B/dist) MODEL=Llama-3.2-3B-Instruct; COEFF=8;  LAYERS=19,20,21; DIST=1; KL_STEPS=300; TARGET_SPEC=near:saves/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 ;;
    8B/act)  MODEL=Llama-3.1-8B-Instruct; COEFF=7;  LAYERS=17,18,19; TARGET_SPEC=near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19 ;;
    8B/dist) MODEL=Llama-3.1-8B-Instruct; COEFF=7;  LAYERS=16,17,18; DIST=1; KL_STEPS=300; TARGET_SPEC=near:saves/NEAR/forget05_Llama8B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18 ;;
    # NEAR (= NEAR-Dist v2: E8 + retain-answer KL 1, prefix *_v4, the method of the paper); `dist` above is v1 (kept for the earlier markers). KL_STEPS = the provider's forget05 value.
    1B/near) MODEL=Llama-3.2-1B-Instruct; COEFF=7; LAYERS=11,12,13; DIST=1; KL_STEPS=300; DIST_EXTRA="wdown.kl.retain_kl_weight=1"; TARGET_SPEC=near:saves/NEAR/forget05_Llama1B_v4/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 ;;
    3B/near) MODEL=Llama-3.2-3B-Instruct; COEFF=8; LAYERS=19,20,21; DIST=1; KL_STEPS=300; DIST_EXTRA="wdown.kl.retain_kl_weight=1"; TARGET_SPEC=near:saves/NEAR/forget05_Llama3B_v4/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 ;;
    8B/near) MODEL=Llama-3.1-8B-Instruct; COEFF=7; LAYERS=16,17,18; DIST=1; KL_STEPS=300; DIST_EXTRA="wdown.kl.retain_kl_weight=1 wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8"; TARGET_SPEC=near:saves/NEAR/forget05_Llama8B_v4/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18 ;;
    # LUNAR (harmful reference set, ONE edited layer): the matched points of the main table (TODO/exp/23: the 3B point is c1.5 l12, NOT the misfiled folder forget05_Llama3B_v2/..._ks)
    1B/lunar) MODEL=Llama-3.2-1B-Instruct; COEFF=1.5; LAYERS=11; TARGET_SPEC=near:saves/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful/c1.5_all/l11 ;;
    3B/lunar) MODEL=Llama-3.2-3B-Instruct; COEFF=1.5; LAYERS=12; TARGET_SPEC=near:saves/LUNAR/single/lunar_tofu_Llama-3.2-3B-Instruct_forget05_LUNAR_harmful/c1.5_all/l12 ;;
    8B/lunar) MODEL=Llama-3.1-8B-Instruct; COEFF=2.8; LAYERS=23; TARGET_SPEC=near:saves/LUNAR/single/lunar_tofu_Llama-3.1-8B-Instruct_forget05_LUNAR_harmful/c2.8_all/l23 ;;
    # ROCR, original (every author -> Donald Trump, residual factor 1; the provider's run never reaches the matching level, so it has no strength to match: one edit, no grid)
    1B/rocr) MODEL=Llama-3.2-1B-Instruct; LAYERS=2,3,4; TARGET_SPEC=hf:saves/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4/model ;;
    3B/rocr) MODEL=Llama-3.2-3B-Instruct; LAYERS=3,4,5; TARGET_SPEC=hf:saves/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5/model ;;
    8B/rocr) MODEL=Llama-3.1-8B-Instruct; LAYERS=4,5,6; TARGET_SPEC=hf:saves/ROCR/Llama-3.1-8B-Instruct_forget05_famous_l4-5-6/model ;;
    # RMU (steering layer 7, ALL parameters trained; the final model of the _ckpt run = step 35 = the matched point of the main table). LAYERS = the steered layer: the attacker
    # knows it (`module_regex`); post_edit_mean = the layers above it. pre_edit_mean is NOT "M' = the original model" here (every layer is trained).
    1B/rmu) MODEL=Llama-3.2-1B-Instruct; LAYERS=7; RMU_LR=5e-5; TARGET_SPEC=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget05_lr5e-5_ep5_ckpt ;;
    3B/rmu) MODEL=Llama-3.2-3B-Instruct; LAYERS=7; RMU_LR=1e-4; TARGET_SPEC=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-3B-Instruct_forget05_lr1e-4_ep5_ckpt ;;
    8B/rmu) MODEL=Llama-3.1-8B-Instruct; LAYERS=7; RMU_LR=1e-4; TARGET_SPEC=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.1-8B-Instruct_forget05_lr1e-4_ep5_ckpt
            RMU_EXTRA=${SPA_RMU_EXTRA:-"trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"} ;;  # same effective batch 32
    # GA / NPO (TODO/exp/26): the provider's M' = the checkpoints of the A1 v2 / A2 P0 lanes (a2_p0_lane.sh; the main table's matched step can differ by one epoch where no model
    # was kept: 3B GA uses step 21, the main table step 14). No edit layer (all parameters, no target layer): only the cv / cv_deepest rules apply. Same shadow training as RMU.
    1B/ga)  MODEL=Llama-3.2-1B-Instruct; TRAINER=GradAscent; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14 ;;
    3B/ga)  MODEL=Llama-3.2-3B-Instruct; TRAINER=GradAscent; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-21 ;;
    8B/ga)  MODEL=Llama-3.1-8B-Instruct; TRAINER=GradAscent; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14
            RMU_EXTRA=${SPA_RMU_EXTRA:-"trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"} ;;
    1B/npo) MODEL=Llama-3.2-1B-Instruct; TRAINER=NPO; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35 ;;
    3B/npo) MODEL=Llama-3.2-3B-Instruct; TRAINER=NPO; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14 ;;
    8B/npo) MODEL=Llama-3.1-8B-Instruct; TRAINER=NPO; RMU_LR=2e-5; LAYERS=""; TARGET_SPEC=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35
            RMU_EXTRA=${SPA_RMU_EXTRA:-"trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"} ;;
    *) echo "spa_cfg: unknown $1/$2" >&2; return 1 ;;
  esac
  case "$2" in act|dist|near) KIND=near ;; rmu|ga|npo) KIND=train ;; *) KIND=$2 ;; esac
  BASE=open-unlearning/tofu_${MODEL}_full
  # SHADOW selection rule (the provider's rule, applied by the attacker on its own shadow set S_k whose answers it has): the WEAKEST coefficient of the grid whose edit brings the
  # answer probability of S_k to <= SPA_TAU (0.25). The provider's coefficient on M' over-shoots (3B Act c14: S_k 0.78 -> 0.008, holdout 0.13; on retain95 0.97 -> 0.13), so a fixed
  # coefficient does not reproduce the provider's strength. NEAR-Dist calibrates its own level (KL stage): no grid.
  if [ "${KIND}" = rocr ] || [ "${KIND}" = train ]; then SHADOW_COEFFS=x     # one edit (ROCR: no strength knob; RMU: the strength is the training length, chosen inside the run)
  # (3B pilot, group 0: real c0.5 -> S_k 0.31, c1 -> 0.10, so 0.75 was added; retain c1.75 -> 0.26, c2 -> 0.16. 8B starts lower than the provider's c2.8 for the same reason.)
  elif [ "${KIND}" = lunar ]; then case "$1" in 8B) SHADOW_COEFFS=${SPA_COEFFS:-"1 1.5 2 2.4 2.8 3.2 3.6 4"} ;; *) SHADOW_COEFFS=${SPA_COEFFS:-"0.5 0.75 1 1.25 1.5 1.75 2 2.5 3"} ;; esac
  elif [ "${DIST}" = 1 ]; then SHADOW_COEFFS=${SPA_COEFFS:-${COEFF}}
  # (3B pilot, group 0: c1 / c2 change nothing; real chose c8 -> S_k 0.177 = the provider's F on M', retain chose c14 -> 0.134. 1B / 8B: grids go past the provider's c7 for the retain control)
  else case "$1" in 3B) SHADOW_COEFFS=${SPA_COEFFS:-"4 6 8 10 12 14 16"} ;; *) SHADOW_COEFFS=${SPA_COEFFS:-"2 3 4 5 6 7 9 11"} ;; esac; fi
}
SPA_TAU=${SPA_TAU:-0.25}
# RMU shadow: at most SPA_RMU_EPOCHS epochs of the provider's lr / batch (40 rows = 2 optimizer steps per epoch: 18 epochs = 36 steps ~ the provider's 35 steps on forget05), evaluated
# after every epoch on the rows of S_k (configs/eval/tofu_sprob.yaml: forget_Q_A_Prob only); training stops at the first epoch with forget_Q_A_Prob <= SPA_TAU (src/trainer/base.py,
# STOP_AT_METRIC) and that model is kept. Never met -> the last epoch (stop.json: met false).
SPA_RMU_EPOCHS=${SPA_RMU_EPOCHS:-18}

# spa_step NAME DONE_FILE cmd...   (skips when DONE_FILE exists unless FORCE=1; DRY_RUN=1 only prints)
spa_step() {
  local name=$1 done_file=$2; shift 2
  if [ -z "${FORCE}" ] && [ -f "${done_file}" ]; then echo "[skip] ${name}"; return 0; fi
  echo "[run ] ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "       $*"; return 0; fi
  "$@" || { echo "[FAIL] ${name}" >&2; return 1; }
  mkdir -p "$(dirname "${done_file}")"; touch "${done_file}"
}

# one shadow edit of NEAR-Act / NEAR-Dist on the model $mprime (folder or hub id) for group $k; prints the folder of the edited model
spa_edit_near() {   # scale method mprime k cond [coeff]
  local scale=$1 method=$2 mprime=$3 k=$4 cond=$5
  spa_cfg "${scale}" "${method}" || return 1
  local coeff=${6:-${COEFF}}
  local a0=$((181 + 2 * k)) a1=$((182 + 2 * k)) r0=$((40 * k)) r1=$((40 * k + 40))
  local tag=spa/${scale}_${method}_${cond}_g${k}
  local lay="[${LAYERS}]"      # (computed here: in a command prefix `A=.. B=$A` the later assignment sees the earlier one -> [[12,13,14]])
  local ovr="model.pretrained.name_or_path=${mprime} data.forget.dataset.args.hf_args.split=\"train[${r0}:${r1}]\" +data.neighbor.dataset.args.row_filter.author_id=[${a0},${a1}] +data.neighbor_heldout.dataset.args.row_filter.author_id=[${a0},${a1}]"
  local wd="wdown.retain_weight=1 wdown.include_answer=true wdown.multi_layer=increment wdown.sequential=true"
  if [ "${DIST}" = 1 ]; then
    wd="${wd} wdown.kl.enabled=true wdown.kl.weight=1 wdown.kl.steps=${KL_STEPS} wdown.kl.forget_prob_weight=1 wdown.kl.forget_prob_target=unseen wdown.kl.forget_prob_mode=neighbor_scaled wdown.kl.init=original wdown.kl.forget_weight=0 wdown.kl.neighbors_per_question=3 wdown.kl.mask_neighbor_name=true ${DIST_EXTRA} ${EXTRA_DIST}"
  fi
  MODEL=${MODEL} FORGET_SPLIT=forget10 NEIGHBOR_SPLIT=neighbor10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 AUTHOR_OFFSET=${a0} \
  K=5 COEFF=${coeff} MODE=all LAYERS="${lay}" W_DOWN_LAYER="${lay}" TASK_NAME=${tag} \
  EXTRA_R_UV="${ovr}" EXTRA_SELECT="${ovr}" EXTRA_W_DOWN="${wd} ${ovr}" \
  bash scripts/NEAR/single_run.sh "${GPU}" "1 2 3"
}

# LUNAR on $mprime for group k: r_UV of S_k (stage 1, once per group: it does not depend on the coefficient), then layer selection restricted to the provider's layer + W_down (stages 2-3)
spa_edit_lunar() {   # scale method mprime k cond coeff
  local scale=$1 method=$2 mprime=$3 k=$4 cond=$5 coeff=$6
  spa_cfg "${scale}" "${method}" || return 1
  local a0=$((181 + 2 * k)) r0=$((40 * k)) r1=$((40 * k + 40))
  local task=spa/${scale}_${method}_${cond}_g${k}
  local ovr="model.pretrained.name_or_path=${mprime}"
  local env=(MODEL=${MODEL} FORGET_SPLIT=forget10 FORGET_HF_SPLIT="train[${r0}:${r1}]" RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=${a0}
             REF_DATASET=LUNAR/LUNAR_harmful MODE=all LAYERS="[${LAYERS}]" W_DOWN_LAYER="[${LAYERS}]" TASK_NAME=${task}
             EXTRA_R_UV="${ovr}" EXTRA_SELECT="${ovr}" EXTRA_W_DOWN="${ovr}")
  if [ ! -f "saves/LUNAR/${task}/r_uv.pt" ]; then env "${env[@]}" COEFF=${coeff} bash scripts/LUNAR/single_run.sh "${GPU}" "1" || return 1; fi
  env "${env[@]}" COEFF=${coeff} bash scripts/LUNAR/single_run.sh "${GPU}" "2 3"
}

# ROCR (original target, factor 1) on $mprime for group k: forget rows = S_k; the null-space projectors are computed once on $mprime (cache data/rocr/<basename of $mprime>/)
spa_edit_rocr() {    # scale method mprime k cond
  local scale=$1 method=$2 mprime=$3 k=$4 cond=$5
  spa_cfg "${scale}" "${method}" || return 1
  local r0=$((40 * k)) r1=$((40 * k + 40))
  python src/METHOD/ROCR/rocr.py --model "${mprime}" --forget-split forget10 --forget-rows "${r0}:${r1}" --author-offset $((181 + 2 * k)) --target famous \
    --layers ${LAYERS//,/ } --scale 1 --dtype bfloat16 --out "$(dirname "$(spa_edited_dir "${scale}" "${method}" "${cond}" "${k}")")"
}

# RMU / GA / NPO (the provider's trainer config and lr; trainer = TRAINER) on $mprime for group k: forget = S_k, retain90; stops at the first epoch with forget_Q_A_Prob(S_k) <= SPA_TAU (see SPA_RMU_EPOCHS)
spa_edit_train() {   # scale method mprime k cond
  local scale=$1 method=$2 mprime=$3 k=$4 cond=$5
  spa_cfg "${scale}" "${method}" || return 1
  local r0=$((40 * k)) r1=$((40 * k + 40))
  local out; out=$(spa_edited_dir "${scale}" "${method}" "${cond}" "${k}")
  local extra=(); [ -n "${RMU_EXTRA}" ] && read -r -a extra <<< "${RMU_EXTRA}"
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 STOP_AT_METRIC="forget_Q_A_Prob<=${SPA_TAU}" \
  python src/train.py --config-name=unlearn experiment=unlearn/tofu/default trainer=${TRAINER} model=tofu/tofu_${MODEL}_full model.pretrained.name_or_path=${mprime} \
    forget_split=forget10 retain_split=retain90 holdout_split=holdout10 retain_logs_path=null \
    "data.train.dataset.forget.args.hf_args.split='train[${r0}:${r1}]'" eval=tofu_sprob "eval.tofu.forget_hf_split='train[${r0}:${r1}]'" \
    trainer.args.learning_rate=${RMU_LR} trainer.args.num_train_epochs=${SPA_RMU_EPOCHS} "${extra[@]}" \
    task_name=spa/${scale}_${method}_${cond}_g${k} paths.output_dir=${out}
}

spa_edit() {         # scale method mprime k cond coeff   (dispatch on the method)
  spa_cfg "$1" "$2" || return 1
  case "${KIND}" in
    near)  spa_edit_near "$1" "$2" "$3" "$4" "$5" "$6" ;;
    lunar) spa_edit_lunar "$1" "$2" "$3" "$4" "$5" "$6" ;;
    rocr)  spa_edit_rocr "$1" "$2" "$3" "$4" "$5" ;;
    train) spa_edit_train "$1" "$2" "$3" "$4" "$5" ;;
  esac
}

spa_edited_dir() {   # scale method cond k [coeff]  -> the folder of the edited model (hf:)
  spa_cfg "$1" "$2" || return 1
  case "${KIND}" in
    near)  echo "saves/NEAR/spa/$1_$2_$3_g$4/k5_c$(printf "%g" "${5:-${COEFF}}")_all/l${LAYERS//,/-}/model" ;;
    lunar) echo "saves/LUNAR/spa/$1_$2_$3_g$4/c$(printf "%g" "${5:-${COEFF}}")_all/l${LAYERS//,/-}/model" ;;
    rocr)  echo "saves/ROCR/spa/$1_$2_$3_g$4/model" ;;
    train) echo "saves/tofu_unlearn/spa/$1_$2_$3_g$4" ;;
  esac
}

spa_edit_root() {    # scale method cond k [coeff]  -> the folder that is deleted once the activations are taken (model + its small files)
  spa_cfg "$1" "$2" || return 1
  local d; d=$(spa_edited_dir "$@") || return 1
  case "${KIND}" in train) echo "${d}" ;; *) dirname "${d}" ;; esac
}

spa_keep_small() {   # src_dir dst_prefix: copy the small json files of an edit folder to the logs before the folder is deleted (provenance of the shadow strength)
  local f
  for f in "$1"/*.json; do [ -f "${f}" ] && [ "$(stat -c %s "${f}")" -lt 2000000 ] && cp "${f}" "$2_$(basename "${f}")"; done
  return 0
}

spa_shadow() {   # scale method cond k mprime
  local scale=$1 method=$2 cond=$3 k=$4 mprime=$5
  local acts=${SPA_DIR}/${scale}_${method}_${cond}/shadow${k}.npz
  spa_cfg "${scale}" "${method}" || return 1
  local log=logs/spa/${scale}_${method}_${cond}
  local acts_done=${log}/acts_g${k}.${SETS_V}.sel.done      # .sel: chosen by the shadow selection rule (earlier .v2.done markers = the fixed provider coefficient, re-done)
  local sel_file=${log}/coeff_g${k}
  local strength=${OUT_DIR}/spa/shadow_strength_${scale}_${method}_${cond}.tsv
  if [ -z "${FORCE}" ] && [ -f "${acts_done}" ]; then echo "[skip] acts ${scale} ${method} ${cond} g${k}"; return 0; fi
  local chosen="" c edited p root
  for c in ${SHADOW_COEFFS}; do          # weakest first: stop at the first coefficient that reaches the matching level on S_k
    edited=$(spa_edited_dir "${scale}" "${method}" "${cond}" "${k}" "${c}")
    root=$(spa_edit_root "${scale}" "${method}" "${cond}" "${k}" "${c}")
    if [ -z "${DRY_RUN}" ] && [ ! -f "${edited}/config.json" ]; then rm -f "${log}/edit_g${k}_c${c}.done"; fi   # deleted after an earlier collect
    spa_step "edit ${scale} ${method} ${cond} g${k} c${c}" "${log}/edit_g${k}_c${c}.done" spa_edit "${scale}" "${method}" "${mprime}" "${k}" "${cond}" "${c}" || return 1
    chosen=${c}
    if [ "${KIND}" = rocr ]; then break; fi                                                      # one edit; strength = the proxy printed by the collect below
    if [ "${KIND}" = train ]; then                                                                 # the stopping epoch was chosen inside the run (stop.json)
      if [ -n "${DRY_RUN}" ]; then echo "       (stops at the first epoch with forget_Q_A_Prob(S_k) <= ${SPA_TAU}; see ${root}/stop.json)"; break; fi
      p=$(python -c "import json; d=json.load(open('${root}/stop.json')); print(d['step'], d['forget_Q_A_Prob'], d['met'])")
      mkdir -p "$(dirname "${strength}")"; printf "g%s\tstep_Prob_met\t%s\n" "${k}" "${p}" >> "${strength}"
      echo "       g${k} ${TRAINER} stopped at step / forget_Q_A_Prob(S_k) / met: ${p}"
      break
    fi
    if [ -n "${DRY_RUN}" ]; then echo "       (then: S_k answer prob of ${root}/responses_after_summary.json <= ${SPA_TAU} ? stop : next coefficient)"; break; fi
    p=$(python -c "import json; print(json.load(open('${root}/responses_after_summary.json'))['forget']['answer_prob'])")
    mkdir -p "$(dirname "${strength}")"
    printf "g%s\tc%s\tS_k_answer_prob\t%s\n" "${k}" "${c}" "${p}" >> "${strength}"
    echo "       g${k} coefficient ${c}: S_k answer prob ${p}"
    if python -c "import sys; sys.exit(0 if float('${p}') <= ${SPA_TAU} else 1)"; then break; fi
    [ -n "${KEEP_EDITED}" ] || rm -rf "${root}"     # not chosen: free the disk
  done
  [ -n "${DRY_RUN}" ] || { echo "${chosen}" > "${sel_file}"; printf "g%s\tchosen\tc%s\n" "${k}" "${chosen}" >> "${strength}"; }
  edited=$(spa_edited_dir "${scale}" "${method}" "${cond}" "${k}" "${chosen}")
  root=$(spa_edit_root "${scale}" "${method}" "${cond}" "${k}" "${chosen}")
  spa_step "acts ${scale} ${method} ${cond} g${k} (c${chosen})" "${acts_done}" \
    python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${edited}" --base "${BASE}" --out "${acts}" --gpu "${GPU}" || return 1
  if [ -z "${DRY_RUN}" ] && [ -f "${acts_done}" ] && [ -d "${root}" ]; then
    spa_keep_small "${root}" "${log}/g${k}"                                        # summaries / stop.json / rocr_summary.json -> logs/spa/...
    [ -n "${KEEP_EDITED}" ] || rm -rf "${root}"                                    # the edit's folder (model + w_down.pt / checkpoints)
  fi
}

spa_run() {   # scale method cond
  local scale=$1 method=$2 cond=$3
  spa_cfg "${scale}" "${method}" || return 1
  local dir=${SPA_DIR}/${scale}_${method}_${cond}
  if [ "${SETS_V}" = v2 ]; then spa_step "sets ${SETS_V}" "logs/spa/sets_${SETS_V}.done" python scripts/analysis/spa.py sets --out "${SETS}" || return 1
  elif [ -z "${DRY_RUN}" ] && [ ! -f "${SETS}" ]; then echo "${SETS} not found (make it first: spa.py sets --r-near ...)" >&2; return 1; fi
  local mprime
  if [ "${cond}" = real ]; then
    mprime=${SPA_DIR}/models/${scale}_${method}_mprime
    spa_step "materialise M' ${scale} ${method}" "logs/spa/${scale}_${method}_mprime.done" python scripts/analysis/spa_materialize.py --model "${TARGET_SPEC}" --base "${BASE}" --out "${mprime}" --gpu "${GPU}" || return 1
    spa_step "target acts ${scale} ${method} real" "logs/spa/${scale}_${method}_real/target.${SETS_V}.done" python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${mprime}" --base "${BASE}" --out "${dir}/target.npz" --gpu "${GPU}" || return 1
  else
    mprime=open-unlearning/tofu_${MODEL}_retain95
    spa_step "target acts ${scale} retain95" "logs/spa/${scale}_retain95.${SETS_V}.done" python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${mprime}" --base "${BASE}" --out "${SPA_DIR}/${scale}_retain95_target.npz" --gpu "${GPU}" || return 1
    if [ -z "${DRY_RUN}" ]; then mkdir -p "${dir}"; cp "${SPA_DIR}/${scale}_retain95_target.npz" "${dir}/target.npz"; fi
  fi
  for k in "${SPA_GROUPS_A[@]}"; do spa_shadow "${scale}" "${method}" "${cond}" "${k}" "${mprime}" || return 1; done
  if [ "${#SPA_GROUPS_A[@]}" -eq 5 ]; then
    spa_step "attack ${scale} ${method} ${cond}${SPA_OUT_SUFFIX}" "logs/spa/${scale}_${method}_${cond}/attack.${SETS_V}${SPA_OUT_SUFFIX}.rules.done" \
      python scripts/analysis/spa.py attack --sets "${SETS}" --shadow ${dir}/shadow0.npz ${dir}/shadow1.npz ${dir}/shadow2.npz ${dir}/shadow3.npz ${dir}/shadow4.npz \
        --target "${dir}/target.npz" --out "${OUT_DIR}/spa_${scale}_${method}_${cond}${SPA_OUT_SUFFIX}" ${SPA_ATTACK_ARGS} --kinds lr dm ${LAYERS:+--edit-layers ${LAYERS//,/ }} --title "A2 SPA ${scale} ${method} (${cond})"
  else
    echo "[note] SPA_GROUPS=${SPA_GROUPS_A[*]}: no attack (needs all 5 groups); shadow strength is printed by each collect"
  fi
}
