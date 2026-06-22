# 实验

## 模型

1. nsloth/Llama-3.2-1B-Instruct
2. Qwen/Qwen2.5-1.5B-Instruct
3. ？？？
4. 


Llama-3.2-1B-Instruct      # ~1.2B
Qwen2.5-1.5B-Instruct      # ~1.5B
phi-1_5                    # ~1.3B
Llama-3.2-3B-Instruct      # ~3B
Qwen2.5-3B-Instruct        # ~3B

## 实验设置

1. `forget_split=forget10` `retain_split=retain90` `holdout_split=holdout10`
2. `forget_split=forget05` `retain_split=retain95` `holdout_split=holdout05`
3. `forget_split=forget01` `retain_split=retain99` `holdout_split=holdout01`

## 实验清单

### 针对 nsloth/Llama-3.2-1B-Instruct

1. from scratch:

    `saves/eval/tofu_Llama-3.2-1B-Instruct_retain90` ✅

    `saves/eval/tofu_Llama-3.2-1B-Instruct_retain95` ✅

    `saves/eval/tofu_Llama-3.2-1B-Instruct_retain99` ✅

### Qwen/Qwen2.5-1.5B-Instruct

1. from scratch:

    TODO
