#!/bin/bash
# Functions of the A2 shadow-probe attack chains (TODO/exp/A2_SPEC.md). Source it, do not run it.  Used by spa_lane_*.sh.
#   spa_cfg SCALE METHOD      -> MODEL COEFF LAYERS TARGET_SPEC DIST (provider's configuration of NEAR-Act / NEAR-Dist at forget05; the attacker knows the algorithm and its hyper-parameters)
#   spa_run SCALE METHOD COND -> everything of one condition: M' (COND=real) or the retain95 model (COND=retain, the negative control), 5 shadow edits of 2 authors each, activations, attack.
# Shadow edit of group k: the SAME algorithm on M' with the forget set = rows [40k, 40k+40) of forget10 (= authors 181+2k, 182+2k), their own train neighbours, retain90, provider's layers and
# coefficient. Nothing of F, of the provider's neighbours or of the full model M0 is read. The edited model is deleted after its activations are taken.
# Env: GPU (default 0), SPA_GROUPS (default "0 1 2 3 4"), DRY_RUN=1, FORCE=1 (ignore done markers), SPA_DIR (activations, default saves/spa), OUT_DIR (default saves/analysis).

SPA_DIR=${SPA_DIR:-saves/spa}
OUT_DIR=${OUT_DIR:-saves/analysis}
SETS=${OUT_DIR}/spa/sets.json
read -r -a SPA_GROUPS_A <<< "${SPA_GROUPS:-0 1 2 3 4}"
GPU=${GPU:-0}

spa_cfg() {
  DIST=0
  case "$1/$2" in
    1B/act)  MODEL=Llama-3.2-1B-Instruct; COEFF=7;  LAYERS=7,8,9;    TARGET_SPEC=near:saves/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9 ;;
    1B/dist) MODEL=Llama-3.2-1B-Instruct; COEFF=7;  LAYERS=11,12,13; DIST=1; KL_STEPS=${KL_STEPS:-300}; TARGET_SPEC=near:saves/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 ;;
    3B/act)  MODEL=Llama-3.2-3B-Instruct; COEFF=14; LAYERS=12,13,14; TARGET_SPEC=near:saves/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14/k5_c14_all/l12-13-14 ;;
    3B/dist) MODEL=Llama-3.2-3B-Instruct; COEFF=8;  LAYERS=19,20,21; DIST=1; KL_STEPS=300; TARGET_SPEC=near:saves/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 ;;
    8B/act)  MODEL=Llama-3.1-8B-Instruct; COEFF=7;  LAYERS=17,18,19; TARGET_SPEC=near:saves/NEAR/forget05_Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19 ;;
    8B/dist) MODEL=Llama-3.1-8B-Instruct; COEFF=7;  LAYERS=16,17,18; DIST=1; KL_STEPS=300; TARGET_SPEC=near:saves/NEAR/forget05_Llama8B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18 ;;
    *) echo "spa_cfg: unknown $1/$2" >&2; return 1 ;;
  esac
  BASE=open-unlearning/tofu_${MODEL}_full
}

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
spa_edit_near() {
  local scale=$1 method=$2 mprime=$3 k=$4 cond=$5
  spa_cfg "${scale}" "${method}" || return 1
  local a0=$((181 + 2 * k)) a1=$((182 + 2 * k)) r0=$((40 * k)) r1=$((40 * k + 40))
  local tag=spa/${scale}_${method}_${cond}_g${k}
  local lay="[${LAYERS}]"      # (computed here: in a command prefix `A=.. B=$A` the later assignment sees the earlier one -> [[12,13,14]])
  local ovr="model.pretrained.name_or_path=${mprime} data.forget.dataset.args.hf_args.split=\"train[${r0}:${r1}]\" +data.neighbor.dataset.args.row_filter.author_id=[${a0},${a1}] +data.neighbor_heldout.dataset.args.row_filter.author_id=[${a0},${a1}]"
  local wd="wdown.retain_weight=1 wdown.include_answer=true wdown.multi_layer=increment wdown.sequential=true"
  if [ "${DIST}" = 1 ]; then
    wd="${wd} wdown.kl.enabled=true wdown.kl.weight=1 wdown.kl.steps=${KL_STEPS} wdown.kl.forget_prob_weight=1 wdown.kl.forget_prob_target=unseen wdown.kl.forget_prob_mode=neighbor_scaled wdown.kl.init=original wdown.kl.forget_weight=0 wdown.kl.neighbors_per_question=3 wdown.kl.mask_neighbor_name=true ${EXTRA_DIST}"
  fi
  MODEL=${MODEL} FORGET_SPLIT=forget10 NEIGHBOR_SPLIT=neighbor10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 AUTHOR_OFFSET=${a0} \
  K=5 COEFF=${COEFF} MODE=all LAYERS="${lay}" W_DOWN_LAYER="${lay}" TASK_NAME=${tag} \
  EXTRA_R_UV="${ovr}" EXTRA_SELECT="${ovr}" EXTRA_W_DOWN="${wd} ${ovr}" \
  bash scripts/NEAR/single_run.sh "${GPU}" "1 2 3"
}

spa_edited_dir() {   # scale method cond k
  spa_cfg "$1" "$2" || return 1
  echo "saves/NEAR/spa/$1_$2_$3_g$4/k5_c$(printf "%g" "${COEFF}")_all/l${LAYERS//,/-}/model"
}

spa_shadow() {   # scale method cond k mprime
  local scale=$1 method=$2 cond=$3 k=$4 mprime=$5
  local acts=${SPA_DIR}/${scale}_${method}_${cond}/shadow${k}.npz
  spa_cfg "${scale}" "${method}" || return 1
  local edited; edited=$(spa_edited_dir "${scale}" "${method}" "${cond}" "${k}")
  spa_step "edit ${scale} ${method} ${cond} g${k}" "logs/spa/${scale}_${method}_${cond}/edit_g${k}.done" spa_edit_near "${scale}" "${method}" "${mprime}" "${k}" "${cond}" || return 1
  spa_step "acts ${scale} ${method} ${cond} g${k}" "logs/spa/${scale}_${method}_${cond}/acts_g${k}.done" \
    python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${edited}" --base "${BASE}" --out "${acts}" --gpu "${GPU}" || return 1
  if [ -z "${DRY_RUN}" ] && [ -f "${acts}" ] && [ -d "$(dirname "${edited}")" ]; then rm -rf "$(dirname "${edited}")"; fi   # the edited layers' folder (model + w_down.pt): the activations are what is kept
}

spa_run() {   # scale method cond
  local scale=$1 method=$2 cond=$3
  spa_cfg "${scale}" "${method}" || return 1
  local dir=${SPA_DIR}/${scale}_${method}_${cond}
  spa_step "sets" "logs/spa/sets.done" python scripts/analysis/spa.py sets --out "${SETS}" || return 1
  local mprime
  if [ "${cond}" = real ]; then
    mprime=${SPA_DIR}/models/${scale}_${method}_mprime
    spa_step "materialise M' ${scale} ${method}" "logs/spa/${scale}_${method}_mprime.done" python scripts/analysis/spa_materialize.py --model "${TARGET_SPEC}" --base "${BASE}" --out "${mprime}" --gpu "${GPU}" || return 1
    spa_step "target acts ${scale} ${method} real" "logs/spa/${scale}_${method}_real/target.done" python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${mprime}" --base "${BASE}" --out "${dir}/target.npz" --gpu "${GPU}" || return 1
  else
    mprime=open-unlearning/tofu_${MODEL}_retain95
    spa_step "target acts ${scale} retain95" "logs/spa/${scale}_retain95.done" python scripts/analysis/spa.py collect --sets "${SETS}" --model "hf:${mprime}" --base "${BASE}" --out "${SPA_DIR}/${scale}_retain95_target.npz" --gpu "${GPU}" || return 1
    if [ -z "${DRY_RUN}" ]; then mkdir -p "${dir}"; cp "${SPA_DIR}/${scale}_retain95_target.npz" "${dir}/target.npz"; fi
  fi
  for k in "${SPA_GROUPS_A[@]}"; do spa_shadow "${scale}" "${method}" "${cond}" "${k}" "${mprime}" || return 1; done
  if [ "${#SPA_GROUPS_A[@]}" -eq 5 ]; then
    spa_step "attack ${scale} ${method} ${cond}" "logs/spa/${scale}_${method}_${cond}/attack.done" \
      python scripts/analysis/spa.py attack --sets "${SETS}" --shadow ${dir}/shadow0.npz ${dir}/shadow1.npz ${dir}/shadow2.npz ${dir}/shadow3.npz ${dir}/shadow4.npz \
        --target "${dir}/target.npz" --out "${OUT_DIR}/spa_${scale}_${method}_${cond}" --probes lr dm --title "A2 SPA ${scale} ${method} (${cond})"
  else
    echo "[note] SPA_GROUPS=${SPA_GROUPS_A[*]}: no attack (needs all 5 groups); shadow strength is printed by each collect"
  fi
}
