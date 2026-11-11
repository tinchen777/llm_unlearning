#!/bin/bash
# Tonight, lane c (40 GB card): E4: Act seeds 1B, E8 seeds 1B forget05; ROCR -> neighbour residual factors 4/5/6 (1B, 3B); ROCR fp32 check (1B).  11 steps, one after the other; a failed step does not stop the lane; a finished step is marked in
# logs/tonight/done/ and skipped on a re-run (the same command resumes).  Commands from TODO/exp/15 and TODO/exp/17.
#   bash scripts/exp/tonight_lane_c.sh <GPU>             # starts in the background (nohup), log logs/tonight/c.log
#   FG=1 bash scripts/exp/tonight_lane_c.sh <GPU>        # in the foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Watch: tail -f logs/tonight/c.log ; one log per step: logs/tonight/<step>.log
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/tonight; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/c.log" 2>&1 &
  echo "lane c started (pid $!), log ${LOGDIR}/c.log"
  exit 0
fi
step() {  # step <name> < command on stdin (the here-document below)
  local name=$1 cmd; cmd=$(cat)
  if [ -z "${FORCE}" ] && [ -f "${DONE}/${name}" ]; then echo "[skip] ${name} (done)"; return 0; fi
  echo "[$(date +%F\ %T)] START ${name}"
  if [ -n "${DRY_RUN}" ]; then echo "    ${cmd}"; return 0; fi
  if ( eval "${cmd}" ) < /dev/null > "${LOGDIR}/${name}.log" 2>&1; then touch "${DONE}/${name}"; echo "[$(date +%T)] OK    ${name}"
  else echo "[$(date +%T)] FAIL  ${name}  (see ${LOGDIR}/${name}.log)"; return 1; fi
}
echo "lane c on GPU ${G} starts $(date)"
step e4_act_seed1_1b_f01 <<'CMD' || true
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_s1 EVAL_PROFILE=full EXTRA_WDOWN="seed=1" INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_act_seed2_1b_f01 <<'CMD' || true
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_s2 EVAL_PROFILE=full EXTRA_WDOWN="seed=2" INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_e8_seed1_1b_f05 <<'CMD' || true
SEED=1 SETTING=1B_f05 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_e8_seed2_1b_f05 <<'CMD' || true
SEED=2 SETTING=1B_f05 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step rocr_nb_s4_1b <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=4 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s5_1b <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_1b <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s4_3b <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct R_UV_PREFIX=Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=4 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s5_3b <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct R_UV_PREFIX=Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=5 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s6_3b <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct R_UV_PREFIX=Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=6 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_fp32_check_1b <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct R_UV_PREFIX=single_v2 LAYERS="2 3 4" TARGET=famous DTYPE=float32 TAG=_fp32 EXTRA_EVAL="model.pretrained.dtype=float32" bash scripts/exp/rocr.sh ${G}
CMD
echo "lane c finished $(date)"
