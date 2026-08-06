#!/bin/bash
# =============================================================================
# 评测一个模型 (TOFU benchmark)
# -----------------------------------------------------------------------------
# 入口: src/eval.py  (mode=eval)
#   1) get_model()      根据 model 配置加载模型+tokenizer
#   2) get_evaluators() 根据 eval=tofu 构造 TOFUEvaluator
#   3) evaluator.evaluate() 逐个跑 configs/eval/tofu.yaml 里 default 列出的指标
#
# 结果文件 (写入 paths.output_dir = saves/eval/<task_name>):
#   TOFU_EVAL.json     每条样本的细粒度分数 (value_by_index)
#   TOFU_SUMMARY.json  每个指标的聚合值 (agg_value)
#
# retain_logs_path: 指向 "retain 参照模型" 的 EVAL.json。
#   forget_quality / privleak 等指标需要它来和参照模型做对比。
#   需要先 `python setup_data.py --eval` 下载官方参照日志，或自己评一个 retain 模型。
# =============================================================================
set -e
cd $(dirname "$0")/../.. || exit 1

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

GPU_ID=${1:-0}
echo "Using GPU: [${GPU_ID}]"
export CUDA_VISIBLE_DEVICES=${GPU_ID}


MODEL=Llama-3.2-3B-Instruct
MODEL_PATH=saves/finetune/test/tofu_Llama-3.2-3B-Instruct_full

python src/eval.py \
  experiment=eval/tofu/default \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
  forget_split=forget10 \
  holdout_split=holdout10 \
  task_name=demo_eval_${MODEL} \

# retain_logs_path=saves/eval/tofu_${MODEL}_retain90/TOFU_EVAL.json \
