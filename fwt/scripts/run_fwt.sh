#!/bin/bash
# =============================================================================
# Forgetting Without Telling: 三个阶段的完整流程 (TOFU forget10, Llama-3.2-1B)
# -----------------------------------------------------------------------------
# 阶段 1 邻居生成: 属性抽取 -> 邻居合成 -> 未知性验证 -> 问题改写 -> neighbors.json
# 阶段 2 层选择:   在遗忘前模型上打分, 选出重定向要写入的层
# 阶段 3 训练:     把遗忘问题的激活重定向到邻居激活 (只更新 down_proj)
#
# 全部产物都落在 data/fwt/ 与 saves/ 下, 不改动 src/ 与 configs/ 的任何文件。
# =============================================================================
set -e
cd "$(dirname "$0")/../.." || exit 1

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
echo "Using GPU: [${GPU_ID}]"

MODEL=Llama-3.2-1B-Instruct
TARGET_MODEL=open-unlearning/tofu_Llama-3.2-1B-Instruct_full
FORGET_SPLIT=forget10
RETAIN_SPLIT=retain90
NEIGHBORS=data/fwt/neighbors_${FORGET_SPLIT}.json
SELECTION=saves/fwt/layer_selection_${MODEL}_${FORGET_SPLIT}.json

# -----------------------------------------------------------------------------
# 阶段 1: 属性匹配邻居数据 (M=5)
#   --backend template 完全离线; 换成 --backend llm 并设置 FWT_LLM_BASE_URL /
#   FWT_LLM_API_KEY 即可用外部 LLM 生成 (论文附录报告的就是这一版)。
#   --probe-model 打开"模型确实不认识这些名字"的质控 (D2 的关键一步)。
# -----------------------------------------------------------------------------
echo "=== [1/3] build neighbors ==="
python fwt/scripts/build_neighbors.py \
  --dataset-path locuslab/TOFU \
  --forget-split ${FORGET_SPLIT} \
  --num-neighbors 5 \
  --backend template \
  --block-size 20 \
  --probe-model ${TARGET_MODEL} \
  --out ${NEIGHBORS} \
  --report saves/fwt/neighbors_${FORGET_SPLIT}_report.json

# -----------------------------------------------------------------------------
# 阶段 2: 层选择
#   打分维度: known/unknown 可分性、重定向代价、邻居目标内聚度、retain 的安全边界
# -----------------------------------------------------------------------------
echo "=== [2/3] select layers ==="
python fwt/scripts/select_layers.py \
  --model ${TARGET_MODEL} \
  --neighbors ${NEIGHBORS} \
  --retain-dataset locuslab/TOFU \
  --retain-split ${RETAIN_SPLIT} \
  --num-questions 128 \
  --top-k 3 \
  --out ${SELECTION}

LAYER=$(python -c "import json;print(json.load(open('${SELECTION}'))['selected_layers'][0])")
echo "selected layer: ${LAYER}"

# -----------------------------------------------------------------------------
# 阶段 3: 训练 (默认 MSE + NLL retain, 单层)
#   换消融只改命令行:
#     对比项:      trainer.method_args.contrastive_weight=0.5
#     随机采样目标: trainer.method_args.target_mode=sample
#     多层:        trainer.method_args.layers="[5,6,7]"
#     邻居数量 M:  data.forget.TOFU_QA_forget_neighbor.args.num_neighbors=3
# -----------------------------------------------------------------------------
echo "=== [3/3] unlearn with neighbor redirection ==="
python fwt/train_fwt.py --config-name=unlearn \
  experiment=unlearn/tofu/fwt \
  model=${MODEL} \
  forget_split=${FORGET_SPLIT} \
  retain_split=${RETAIN_SPLIT} \
  holdout_split=holdout10 \
  neighbors_path=${NEIGHBORS} \
  trainer.method_args.layers="[${LAYER}]" \
  trainer.method_args.gamma=1.0 \
  trainer.method_args.alpha=1.0 \
  retain_logs_path=saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json \
  task_name=fwt/redirect_${FORGET_SPLIT}_L${LAYER}

echo "done"
