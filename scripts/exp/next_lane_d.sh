#!/bin/bash
# Next batch for a free 80 GB card (8B): E4 seeds at 8B (Act c7 layers 17-18-19, E8) and the sampling data of RMU (X4 / E8 of EXPERIMENTS.md): re-train RMU with the checkpoint of
# the matched point (forget01, lr 5e-5, step 10 = the end; 1B and 8B; the 3B one exists) and sample it.  8 steps, one after the other; a failed step does not stop the lane;
# a finished step is marked in logs/next/done/ and skipped on a re-run.  No code change: every command is an existing script with its documented switches.
#   bash scripts/exp/next_lane_d.sh <GPU>             # background (nohup), log logs/next/d.log
#   FG=1 bash scripts/exp/next_lane_d.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
# Watch: tail -f logs/next/d.log ; one log per step: logs/next/<step>.log
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/next; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/d.log" 2>&1 &
  echo "lane d started (pid $!), log ${LOGDIR}/d.log"
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
echo "lane d on GPU ${G} starts $(date)"
step e4_act_seed1_8b_f01 <<'CMD' || true
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="seed=1 wdown.gen_batch_size=8" SELECT_TASK_NAME=Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget01_ks OUT_PREFIX=Llama8B_v2_s1 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_act_seed2_8b_f01 <<'CMD' || true
for c in 7; do FORGET_SPLIT=forget01 RETAIN_SPLIT=retain99 HOLDOUT_SPLIT=holdout01 NEIGHBOR_SPLIT=neighbor01 MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EXTRA_EVAL="eval.tofu.batch_size=4" EXTRA_WDOWN="seed=2 wdown.gen_batch_size=8" SELECT_TASK_NAME=Llama8B_v2/near_tofu_Llama-3.1-8B-Instruct_forget01_ks OUT_PREFIX=Llama8B_v2_s2 EVAL_PROFILE=full INCLUDE_ANSWER=1 LAYER=17,18,19 COEFF=$c MULTI_LAYER=increment SEQUENTIAL=1 RETAIN_WEIGHTS=1 bash scripts/NEAR/sweep_w_down.sh ${G}; done
CMD
step e4_e8_seed1_8b_f01 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEED=1 SETTING=8B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step e4_e8_seed2_8b_f01 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SEED=2 SETTING=8B_f01 VARIANTS="E8" bash scripts/exp/validate_fix.sh ${G}
CMD
step rmu_ckpt_1b <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget01; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=10" LRS=5e-5 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step rmu_ckpt_8b <<'CMD' || true
export MODEL=Llama-3.1-8B-Instruct PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FORGET_SPLIT=forget01; TAG=_ckpt EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true trainer.args.save_strategy=steps +trainer.args.save_steps=10" LRS=5e-5 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step rmu_sample_1b <<'CMD' || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_1B_rmu --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget01_lr5e-5_ep5_ckpt/checkpoint-10
CMD
step rmu_sample_8b <<'CMD' || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_8B_rmu --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_full --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.1-8B-Instruct_forget01_lr5e-5_ep5_ckpt/checkpoint-10
CMD
echo "lane d finished $(date)"
