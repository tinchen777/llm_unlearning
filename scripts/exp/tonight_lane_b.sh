#!/bin/bash
# Tonight, lane b (80 GB card (shares it with lane a)): X1 ablation ladder L1-L5 (1B / 3B) and the Act training seeds of 3B.  11 steps, one after the other; a failed step does not stop the lane; a finished step is marked in
# logs/tonight/done/ and skipped on a re-run (the same command resumes).  Commands from TODO/exp/15 and TODO/exp/17.
#   bash scripts/exp/tonight_lane_b.sh <GPU>             # starts in the background (nohup), log logs/tonight/b.log
#   FG=1 bash scripts/exp/tonight_lane_b.sh <GPU>        # in the foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Watch: tail -f logs/tonight/b.log ; one log per step: logs/tonight/<step>.log
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/tonight; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/b.log" 2>&1 &
  echo "lane b started (pid $!), log ${LOGDIR}/b.log"
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
echo "lane b on GPU ${G} starts $(date)"
step x1_L1_1b <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct REFS="LUNAR/LUNAR_neighbor01" FORGET_SPLITS="forget01" COEFFS="7 12" LAYERS="[9]" EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}
CMD
step x1_L1_3b <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct REFS="LUNAR/LUNAR_neighbor01" FORGET_SPLITS="forget01" COEFFS="10 12 14" LAYERS="[13]" EVAL_PROFILE=full bash scripts/LUNAR/sweep_coeff.sh ${G}
CMD
step x1_L2_1b <<'CMD' || true
for c in 7 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2_abl_L2 EVAL_PROFILE=full EXTRA_WDOWN="wdown.retain_splits=[retain]" LAYER=9 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L2_3b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_abl_L2 EVAL_PROFILE=full EXTRA_WDOWN="wdown.retain_splits=[retain]" LAYER=13 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L3_1b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full LAYER=9 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L3_3b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2 EVAL_PROFILE=full LAYER=13 COEFF=$c RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L4_1b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L4_3b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2 EVAL_PROFILE=full LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step x1_L5_1b <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-1B-Instruct SELECT_TASK_NAME=single_v2/near_tofu_Llama-3.2-1B-Instruct_forget01_ks OUT_PREFIX=single_v2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=7,8,9 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_act_seed1_3b_f01 <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_s1 EVAL_PROFILE=full EXTRA_WDOWN="seed=1" INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_act_seed2_3b_f01 <<'CMD' || true
for c in 12; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.2-3B-Instruct SELECT_TASK_NAME=Llama3B_v2/near_tofu_Llama-3.2-3B-Instruct_forget01_ks OUT_PREFIX=Llama3B_v2_s2 EVAL_PROFILE=full EXTRA_WDOWN="seed=2" INCLUDE_ANSWER=1 LAYER=11,12,13 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
echo "lane b finished $(date)"
