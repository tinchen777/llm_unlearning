#!/bin/bash
# Next batch for a free 40 GB card (1B / 3B only): RMU sampling data at 1B (re-train with the checkpoint of the matched point, lr 5e-5 step 10; then sample), ROCR -> neighbour with the
# residual factor 8 at forget05 / forget10 (1B, 3B; the factor that reached the matching region at forget01), and the baseline seeds (GA, NPO at lr 1e-4, forget01, seeds 1 and 2; 1B, 3B).
# 10 steps, one after the other; a failed step does not stop the lane; a finished step is marked in logs/next/done/ and skipped on a re-run.  No code change.
#   bash scripts/exp/next_lane_e.sh <GPU>             # background (nohup), log logs/next/e.log
#   FG=1 bash scripts/exp/next_lane_e.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Watch: tail -f logs/next/e.log ; one log per step: logs/next/<step>.log
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/e.log" 2>&1 &
  echo "lane e started (pid $!), log ${LOGDIR}/e.log"
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
echo "lane e on GPU ${G} starts $(date)"
step rmu_ckpt_1b <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=10" LRS=5e-5 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step rmu_sample_1b <<'CMD' || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_1B_rmu --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget01_lr5e-5_ep5_ckpt/checkpoint-10
CMD
step rocr_nb_s8_1b_f05 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=8 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s8_3b_f05 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 R_UV_PREFIX=forget05_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=8 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s8_1b_f10 <<'CMD' || true
MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama1B_v2 LAYERS="2 3 4" TARGET=neighbor SCALE=8 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step rocr_nb_s8_3b_f10 <<'CMD' || true
MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget10 RETAIN_SPLIT=retain90 HOLDOUT_SPLIT=holdout10 NEIGHBOR_SPLIT=neighbor10 AUTHOR_OFFSET=181 R_UV_PREFIX=forget10_Llama3B_v2 LAYERS="3 4 5" TARGET=neighbor SCALE=8 EVAL_PROFILE=fast bash scripts/exp/rocr.sh ${G}
CMD
step c3_baseline_seed1_1b_f01 <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct; TAG=_s1 EXTRA_TRAIN="seed=1" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3_baseline_seed2_1b_f01 <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct; TAG=_s2 EXTRA_TRAIN="seed=2" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3_baseline_seed1_3b_f01 <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct; TAG=_s1 EXTRA_TRAIN="seed=1" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step c3_baseline_seed2_3b_f01 <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct; TAG=_s2 EXTRA_TRAIN="seed=2" LRS=1e-4 METHODS="GradAscent NPO" EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
echo "lane e finished $(date)"
