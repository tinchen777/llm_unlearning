#!/bin/bash
# A1 v2 for the baselines GA, NPO, RMU on forget05 (1B, 3B, 8B): re-train each at its MATCHED point (the best-utility point with forget_Q_A_Prob <= 0.25 of the main table) keeping the checkpoint of that step
# (`save_steps` = the matched step; matched points: 1B GA lr2e-5@14, NPO lr2e-5@35, RMU lr5e-5@28; 3B GA lr2e-5@21, NPO lr2e-5@14, RMU lr1e-4@14; 8B GA lr2e-5@14 (utility 0.15: a collapsed point), NPO lr2e-5@35,
# RMU lr1e-4@14), then run the attack on them into the SAME folder as a1v2_lane.sh (same templates, same pool; the analysis of the folder is redone with all the models). Run a1v2_lane.sh first.
# 12 steps; 8B needs the 80 GB card (low-memory flags), 1B / 3B are fine on 40 GB. A failed step does not stop the lane; a finished step is marked in logs/a1v2b/done/.
#   bash scripts/exp/a1v2_baselines_lane.sh <GPU>             # background (nohup), log logs/a1v2b/a1v2_baselines_lane.log
#   FG=1 bash scripts/exp/a1v2_baselines_lane.sh <GPU>        # foreground;  DRY_RUN=1 = print the steps;  FORCE=1 = ignore the done markers
cd "$(dirname "$0")/../.." || exit 1
source scripts/exp/_clean_env.sh
G=${1:-0}
LOGDIR=logs/a1v2b; DONE=${LOGDIR}/done; mkdir -p "${DONE}"
if [ -z "${LANE_INNER}" ] && [ -z "${FG}" ] && [ -z "${DRY_RUN}" ]; then
  LANE_INNER=1 nohup bash "$0" "${G}" > "${LOGDIR}/a1v2_baselines_lane.log" 2>&1 &
  echo "a1v2_baselines_lane started (pid $!), log ${LOGDIR}/a1v2_baselines_lane.log"
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
echo "a1v2_baselines_lane on GPU ${G} starts $(date)"
step a1v2_ckpt_1b_gradascent <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=14" LRS=2e-5 METHODS=GradAscent EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_1b_npo <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=35" LRS=2e-5 METHODS=NPO EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_1b_rmu <<'CMD' || true
export MODEL=Llama-3.2-1B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=28" LRS=5e-5 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_attack_baselines_1b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_1B_f05 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model GA=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14 --model NPO=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35 --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget05_lr5e-5_ep5_ckpt/checkpoint-28
CMD
step a1v2_ckpt_3b_gradascent <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=21" LRS=2e-5 METHODS=GradAscent EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_3b_npo <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=14" LRS=2e-5 METHODS=NPO EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_3b_rmu <<'CMD' || true
export MODEL=Llama-3.2-3B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 ; TAG=_ckpt EXTRA_TRAIN="trainer.args.save_strategy=steps +trainer.args.save_steps=14" LRS=1e-4 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_attack_baselines_3b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_3B_f05 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model GA=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-21 --model NPO=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14 --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.2-3B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14
CMD
step a1v2_ckpt_8b_gradascent <<'CMD' || true
export MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; TAG=_ckpt EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true trainer.args.save_strategy=steps +trainer.args.save_steps=14" LRS=2e-5 METHODS=GradAscent EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_8b_npo <<'CMD' || true
export MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; TAG=_ckpt EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true trainer.args.save_strategy=steps +trainer.args.save_steps=35" LRS=2e-5 METHODS=NPO EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_ckpt_8b_rmu <<'CMD' || true
export MODEL=Llama-3.1-8B-Instruct FORGET_SPLIT=forget05 RETAIN_SPLIT=retain95 HOLDOUT_SPLIT=holdout05 NEIGHBOR_SPLIT=neighbor05 AUTHOR_OFFSET=191 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; TAG=_ckpt EXTRA_TRAIN="trainer.args.per_device_train_batch_size=2 trainer.args.gradient_accumulation_steps=16 trainer.args.gradient_checkpointing=true trainer.args.save_strategy=steps +trainer.args.save_steps=14" LRS=1e-4 METHODS=RMU EVAL_PROFILE=fast bash scripts/unlearn/tune.sh ${G}
CMD
step a1v2_attack_baselines_8b_f05 <<'CMD' || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_8B_f05 --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200 --model GA=hf:saves/tofu_unlearn/GradAscent/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14 --model NPO=hf:saves/tofu_unlearn/NPO/tofu_Llama-3.1-8B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35 --model RMU=hf:saves/tofu_unlearn/RMU/tofu_Llama-3.1-8B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14
CMD
echo "a1v2_baselines_lane finished $(date)"
