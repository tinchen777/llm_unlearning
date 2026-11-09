#!/bin/bash
# Fill the gaps of the main table / Pareto points (TODO/exp/18) + the sampling data of the main-text baselines (X4): THREE sequential lanes, BIG = 80 GB card (8B),
# SMALL and SMALL2 = 40 GB cards (1B / 3B).
# Matching rule decided for the paper: forget_Q_A_Prob <= 0.25 only (mia_loss is reported, not required). Same mechanics as scripts/exp/run_lanes.sh: a failed step does not stop
# the lane, a finished step is marked in logs/fill/done/ and skipped on a re-run, one log per step.
#
#   bash scripts/exp/run_fill_lanes.sh <BIG_GPU> <SMALL_GPU> <SMALL2_GPU>   # the three lanes in the background (nohup), logs in logs/fill/
#   LANE=big bash scripts/exp/run_fill_lanes.sh 0 1 2                       # one lane (LANE=small | small2); LANE=all bash scripts/exp/run_fill_lanes.sh 0 = one chain on one card
#   DRY_RUN=1 bash scripts/exp/run_fill_lanes.sh 0 1 2                      # print the steps;  FG=1 = foreground;  FORCE=1 = ignore the done markers
# Order inside a lane: 1) the methods that FORGET TOO LITTLE (stronger lr / coefficient), 2) the cells with NO points at all, 3) the runs that over-forget (weaker lr).
# Watch: tail -f logs/fill/big.log logs/fill/small.log ; one log per step: logs/fill/<step>.log ; the learning rates are guesses along the monotone Prob(lr) trend of the existing runs.
cd "$(dirname "$0")/../.." || exit 1
BIG_GPU=${1:-0}; SMALL_GPU=${2:-1}; SMALL2_GPU=${3:-${2:-1}}
LOGDIR=logs/fill; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  for l in big small small2; do
    if [ "${LANE}" = all ]; then l=all; fi
    if [ -n "${LANE}" ] && [ "${LANE}" != "${l}" ]; then continue; fi
    LANE_INNER=${l} nohup bash "$0" "${BIG_GPU}" "${SMALL_GPU}" "${SMALL2_GPU}" > "${LOGDIR}/${l}.log" 2>&1 &
    echo "lane ${l} started (pid $!), log ${LOGDIR}/${l}.log"
    if [ "${l}" = all ]; then break; fi
  done
  exit 0
fi
step() {
  local name=$1; shift
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    $*"; return 0; fi
  if ( eval "$*" ) > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
E1="MODEL=Llama-3.2-1B-Instruct"; E3="MODEL=Llama-3.2-3B-Instruct"
E8="MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
S05="FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191"
S10="FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181"
S01="FORGET_SPLIT=forget01"
LOWMEM="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true"
B() {  # B <model env> <split env> <lr> <methods> [low]  -> one baseline run (every epoch is evaluated)
  local x=""; if [ -n "$5" ]; then x="EXTRA_TRAIN='${LOWMEM}' "; fi
  echo "export $1 $2; ${x}LRS=$3 METHODS='$4' EVAL_PROFILE=full bash scripts/unlearn/tune.sh \${G}"
}
LUN() {  # LUN <model env> <split env> <split> <layers> <coeffs>
  local ly=""; if [ -n "$4" ]; then ly="LAYERS='$4' "; fi   # empty = the layer search of the script
  echo "export $1 $2; ${ly}COEFFS='$5' FORGET_SPLITS=$3 EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh \${G}"
}
SAMPX() {  # SAMPX <model> <size tag> <GA lr> <GA ckpt> <NPO lr> <NPO ckpt>  -> sampling consistency of the matched-point GA / NPO checkpoints (forget01; X4), with retain / full
  local m=$1
  local ga=saves/tofu_unlearn/GradAscent/tofu_${m}_forget01_lr$3_ep5_ckpt/checkpoint-$4 npo=saves/tofu_unlearn/NPO/tofu_${m}_forget01_lr$5_ep5_ckpt/checkpoint-$6
  echo "python scripts/analysis/sample_consistency.py --gpu \${G} --out saves/analysis/sample_$2_ga_npo --base open-unlearning/tofu_${m}_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_${m}_retain99 --model full=hf:open-unlearning/tofu_${m}_full --model GA=hf:${ga} --model NPO=hf:${npo}"
}

# ======================================= BIG lane (80 GB): 8B =======================================
big() {
  G=${BIG_GPU}
  # 1) forgets too little
  step b_f05_8b_simnpo "$(B "${E8}" "${S05}" 1e-4 SimNPO low)" || true
  step b_f05_8b_rmu    "$(B "${E8}" "${S05}" 1e-4 RMU low)" || true
  step b_f10_8b_simnpo "$(B "${E8}" "${S10}" 1e-4 SimNPO low)" || true
  step b_f10_8b_act "export ${E8} ${S10}; for c in 8 10 12; do EXTRA_EVAL='eval.tofu.batch_size=4' EXTRA_WDOWN='wdown.gen_batch_size=8' SELECT_TASK_NAME=forget10_Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget10_ks OUT_PREFIX=forget10_Llama8B_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=\$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh \${G}; done" || true
  # 2) no (complete) points: the earlier GD / NPO / RMU runs of 8B forget10 stopped after checkpoint-0 / -13
  step b_f10_8b_gd  "$(B "${E8}" "${S10}" 2e-5 GradDiff low)" || true
  step b_f10_8b_npo "$(B "${E8}" "${S10}" 2e-5 NPO low)" || true
  step b_f10_8b_rmu "$(B "${E8}" "${S10}" 5e-5 RMU low)" || true
  step b_f10_8b_lunar "$(LUN "${E8}" "${S10}" forget10 '[23]' '2 2.4 2.8')" || true
  # 3) over-forgets (utility ~0): weaker
  step b_f01_8b_simnpo_weak "$(B "${E8}" "${S01}" 2e-5 SimNPO low)" || true
  step b_f05_8b_ga_weak "$(B "${E8}" "${S05}" 5e-6 GradAscent low)" || true
  step b_f10_8b_ga_weak "$(B "${E8}" "${S10}" 5e-6 GradAscent low)" || true
  # 4) X4: the sampling data of the main-text baselines at 8B. GA (lr 2e-5, checkpoint-4) exists; NPO needs a re-training that keeps the checkpoint of the matched point (lr 5e-5, step 10 =
  #    the end; save_steps=10 keeps ONLY that one, 16 GB), then the sampling; and the sampling of E8 (the run exists) with the validate_fix.sh protocol
  step x4_8b_npo_ckpt "export ${E8} ${S01}; TAG=_ckpt EXTRA_TRAIN='${LOWMEM} trainer.args.save_strategy=steps +trainer.args.save_steps=10' LRS=5e-5 METHODS=NPO EVAL_PROFILE=fast bash scripts/unlearn/tune.sh \${G}" || true
  step x4_8b_sample "$(SAMPX Llama-3.1-8B-Instruct 8B 2e-5 4 5e-5 10)" || true
  step x4_8b_e8_sample "export ${E8}; SETTING=8B_f01 VARIANTS='E8' SKIP_TRAIN=1 bash scripts/exp/validate_fix.sh \${G}" || true
}

# ======================================= SMALL lane (40 GB): 1B / 3B =======================================
small() {
  G=${SMALL_GPU}
  # 1) forgets too little
  step s_f05_1b_simnpo "$(B "${E1}" "${S05}" 2e-4 SimNPO)" || true
  step s_f05_3b_ga     "$(B "${E3}" "${S05}" 2e-5 GradAscent)" || true
  step s_f05_3b_simnpo "$(B "${E3}" "${S05}" 2e-4 SimNPO)" || true
  step s_f05_3b_rmu    "$(B "${E3}" "${S05}" 1e-4 RMU)" || true
  # 2) no points at all: 1B forget10
  step s_f10_1b_gd     "$(B "${E1}" "${S10}" 2e-5 GradDiff)" || true
  step s_f10_1b_simnpo "$(B "${E1}" "${S10}" 2e-4 SimNPO)" || true
  step s_f10_1b_rmu    "$(B "${E1}" "${S10}" 5e-5 RMU)" || true
  # 3) X4 sampling of the matched-point GA / NPO checkpoints (1B; the checkpoints of exp/16 C2 exist)
  step x4_1b_sample "$(SAMPX Llama-3.2-1B-Instruct 1B 1e-4 2 1e-4 2)" || true
}

# ======================================= SMALL2 lane (40 GB): 3B forget10 =======================================
small2() {
  G=${SMALL2_GPU}
  step s_f10_3b_gd     "$(B "${E3}" "${S10}" 2e-5 GradDiff low)" || true
  step s_f10_3b_simnpo "$(B "${E3}" "${S10}" 1e-4 SimNPO low)" || true
  step s_f10_3b_ga     "$(B "${E3}" "${S10}" 2e-5 GradAscent low)" || true
  step s_f10_3b_npo    "$(B "${E3}" "${S10}" 2e-5 NPO low)" || true
  step s_f10_3b_lunar  "$(LUN "${E3}" "${S10}" forget10 '' '1 1.5 2')" || true
  step x4_3b_sample "$(SAMPX Llama-3.2-3B-Instruct 3B 1e-4 2 1e-4 2)" || true
  # 3) over-forgets: weaker
  step s_f01_3b_simnpo_weak "$(B "${E3}" "${S01}" 5e-5 SimNPO)" || true
}

l=${LANE_INNER:-${LANE:-big}}
if [ "${l}" = all ]; then SMALL_GPU=${BIG_GPU}; SMALL2_GPU=${BIG_GPU}; echo "lane all starts $(date)"; big; small; small2; echo "lane all finished $(date)"; exit 0; fi
if [ -n "${DRY_RUN}" ] && [ -z "${LANE}" ]; then echo "##### BIG (GPU ${BIG_GPU})"; big; echo "##### SMALL (GPU ${SMALL_GPU})"; small; echo "##### SMALL2 (GPU ${SMALL2_GPU})"; small2; exit 0; fi
echo "lane ${l} starts $(date)"; ${l}; echo "lane ${l} finished $(date)"
