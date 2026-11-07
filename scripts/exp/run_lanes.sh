#!/bin/bash
# Two sequential "lanes" for one 70 GB card (BIG: 8B and anything heavy) and one 40 GB card (SMALL: 1B / 3B). Each lane runs its steps one after the other,
# in priority order; a failed step does NOT stop the lane (its dependent steps are skipped); a step that finished is marked in logs/exp/done/ and skipped
# on a re-run, so the same command resumes after a crash. Commands = TODO/exp/{1..4}_*.md.
#
#   bash scripts/exp/run_lanes.sh <BIG_GPU> <SMALL_GPU>            # both lanes in the background (nohup), logs in logs/exp/
#   LANE=big bash scripts/exp/run_lanes.sh 0 1                     # only the big lane   (LANE=small: only the small lane)
#   FG=1 LANE=big bash scripts/exp/run_lanes.sh 0 1                # in the foreground
#   DRY_RUN=1 bash scripts/exp/run_lanes.sh 0 1                    # print the steps only
#   WITH_COST=1 ...                                                # also re-run the 8B steering upstream for the cost table (overwrites layer_selection.json with the same result)
#   FORCE=1 ...                                                    # ignore the done markers
# Watch:  tail -f logs/exp/big.log logs/exp/small.log      one log per step: logs/exp/<step>.log
cd "$(dirname "$0")/../.." || exit 1
BIG_GPU=${1:-0}; SMALL_GPU=${2:-1}
LOGDIR=logs/exp; DONE=${LOGDIR}/done; mkdir -p "${DONE}"

if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then   # launcher: start the lanes in the background
  for l in big small; do
    if [ -n "${LANE}" ] && [ "${LANE}" != "${l}" ]; then continue; fi
    LANE_INNER=${l} nohup bash "$0" "${BIG_GPU}" "${SMALL_GPU}" > "${LOGDIR}/${l}.log" 2>&1 &
    echo "lane ${l} started (pid $!), log ${LOGDIR}/${l}.log"
  done
  exit 0
fi

step() {  # step <name> <command string>; returns non-zero if the step failed
  local name=$1; shift
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    $*"; return 0; fi
  if ( eval "$*" ) > "${LOGDIR}/${name}.log" 2>&1; then
    touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else
    echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1
  fi
}

E8="MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
E3="MODEL=Llama-3.2-3B-Instruct"
E1="MODEL=Llama-3.2-1B-Instruct"
S05="FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191"
S10="FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181"
KLCOMMON="FP_WEIGHT=1 FP_TARGET=unseen FP_MODE=match KL_FORGET_W=0 KL_INIT=original KL_WEIGHTS=1 EVAL_PROFILE=full"
LOWMEM="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"

# ======================================= BIG lane (70 GB): 8B =======================================
big() {
  G=${BIG_GPU}
  # --- A. forget05 x 8B NEAR: upstream -> select -> (old, new)
  T=forget05_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget05_ks
  step A_up  "export ${E8} ${S05}; RUV_KS=1,3,5,10 TASK_NAME=${T} bash scripts/NEAR/single_run.sh ${G} 1" \
  && step A_sel "export ${E8} ${S05}; TASK_NAME=${T} COEFFS=7 MODES=all LAYERS='[15,16,17,18,19,20]' bash scripts/NEAR/sweep_select.sh ${G}" \
  && { step A_new "export ${E8} ${S05}; EXTRA_EVAL='eval.tofu.batch_size=4' KL_STEPS=300 OUT_PREFIX=forget05_Llama8B_v3 UPSTREAM_PREFIX=forget05_Llama8B_v2 EXTRA_WDOWN='wdown.kl.fr_batch_size=4 wdown.kl.batch_size=8 wdown.gen_batch_size=8' ${KLCOMMON} COEFFS=7 LAYER_SETS='16,17,18' bash scripts/NEAR/sweep_kl.sh ${G}" || true
       step A_old "export ${E8} ${S05}; EXTRA_WDOWN='wdown.gen_batch_size=8' EXTRA_EVAL='eval.tofu.batch_size=4' SELECT_TASK_NAME=${T} OUT_PREFIX=forget05_Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=7 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}" || true; } || true
  # --- C. forget05 x 8B LUNAR (fixed layer 23)
  step C_lunar "export ${E8} ${S05}; LAYERS='[23]' COEFFS='1.5 2 2.4 2.8' FORGET_SPLITS=forget05 EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}" || true
  # --- B. forget05 x 8B baselines (GA, GD, SimNPO need no reference model; NPO, RMU last)
  step B_ga_gd "export ${E8} ${S05}; LRS=1e-5 METHODS='GradAscent GradDiff' EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step B_simnpo "export ${E8} ${S05}; LRS=2e-5 METHODS=SimNPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step B_npo_rmu "export ${E8} ${S05}; EXTRA_TRAIN='${LOWMEM}' LRS=2e-5 METHODS='NPO RMU' EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  # --- D. (optional) 8B forget01 steering upstream, for the cost table; same layer window as the original selection (8..24)
  if [ -n "${WITH_COST}" ]; then
    T1=Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget01_ks
    step D_up "export ${E8}; RUV_KS=1,3,5,10 TASK_NAME=${T1} bash scripts/NEAR/single_run.sh ${G} 1" \
    && step D_sel "export ${E8}; TASK_NAME=${T1} COEFFS=7 MODES=all LAYERS='[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]' bash scripts/NEAR/sweep_select.sh ${G}" || true
  fi
}

# ======================================= SMALL lane (40 GB): 3B / 1B =======================================
small() {
  G=${SMALL_GPU}
  # --- E. forget05 x 3B NEAR (upstream and selection -> layer 14 were done earlier)
  T=forget05_Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget05_ks
  step E_new "export ${E3} ${S05}; KL_STEPS=300 OUT_PREFIX=forget05_Llama3B_v3 UPSTREAM_PREFIX=forget05_Llama3B_v2 EXTRA_WDOWN='wdown.kl.fr_batch_size=16 wdown.kl.batch_size=16' ${KLCOMMON} COEFFS=8 LAYER_SETS='19,20,21 13,14,15' bash scripts/NEAR/sweep_kl.sh ${G}" || true
  step E_old "export ${E3} ${S05}; SELECT_TASK_NAME=${T} OUT_PREFIX=forget05_Llama3B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=12,13,14 COEFF=8 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}" || true
  # --- forget05 x 1B baselines + LUNAR (lr = the guesses of exp/2; read the trajectories afterwards)
  step F_1b_ga_npo "export ${E1} ${S05}; LRS=2e-5 METHODS=GradAscent EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}; LRS=5e-5 METHODS=NPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step F_1b_lunar "export ${E1} ${S05}; COEFFS='1 1.5 2' FORGET_SPLITS=forget05 EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}" || true
  step F_1b_gd_rmu "export ${E1} ${S05}; LRS=2e-5 METHODS='GradDiff RMU' EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step F_1b_simnpo "export ${E1} ${S05}; LRS=5e-5 METHODS=SimNPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  # --- forget10 x 1B main methods
  T10=forget10_Llama1B_v2/near_tofu_Llama-3.2-1B-Instruct_forget10_ks
  step G_10_up "export ${E1} ${S10}; RUV_KS=1,3,5,10 TASK_NAME=${T10} bash scripts/NEAR/single_run.sh ${G} 1" \
  && step G_10_sel "export ${E1} ${S10}; TASK_NAME=${T10} COEFFS=7 MODES=all LAYERS='[7,8,9]' bash scripts/NEAR/sweep_select.sh ${G}" \
  && { step G_10_new "export ${E1} ${S10}; KL_STEPS=500 EXTRA_WDOWN='wdown.kl.batch_size=16' UPSTREAM_PREFIX=forget10_Llama1B_v2 OUT_PREFIX=forget10_Llama1B_v3 ${KLCOMMON} COEFFS=7 LAYER_SETS='11,12,13' bash scripts/NEAR/sweep_kl.sh ${G}" || true
       step G_10_old "export ${E1} ${S10}; SELECT_TASK_NAME=${T10} OUT_PREFIX=forget10_Llama1B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=7 MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}" || true; } || true
  step G_10_base "export ${E1} ${S10}; LRS=1e-5 METHODS=GradAscent EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}; LRS=2e-5 METHODS=NPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step G_10_lunar "export ${E1} ${S10}; COEFFS='1 1.5 2' FORGET_SPLITS=forget10 EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}" || true
  # --- forget05 x 3B LUNAR + baselines (lr = guesses; 1e-5 for GA / GD / RMU because the 1B GA window is narrow)
  step H_3b_lunar "export ${E3} ${S05}; COEFFS='1 1.5 2' FORGET_SPLITS=forget05 EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}" || true
  step H_3b_ga_npo "export ${E3} ${S05}; LRS=1e-5 METHODS=GradAscent EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}; LRS=5e-5 METHODS=NPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step H_3b_gd_rmu "export ${E3} ${S05}; LRS=1e-5 METHODS='GradDiff RMU' EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
  step H_3b_simnpo "export ${E3} ${S05}; LRS=5e-5 METHODS=SimNPO EVAL_PROFILE=full bash scripts/unlearn/tune.sh ${G}" || true
}

l=${LANE_INNER:-${LANE:-big}}
if [ -n "${DRY_RUN}" ] && [ -z "${LANE}" ]; then echo "##### BIG (GPU ${BIG_GPU})"; big; echo "##### SMALL (GPU ${SMALL_GPU})"; small; exit 0; fi
echo "lane ${l} starts $(date)"; ${l}; echo "lane ${l} finished $(date)"
