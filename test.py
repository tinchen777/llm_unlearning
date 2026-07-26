

import sys
sys.path.append('/data/tianzhen/my_projects/LLM/llm_unlearning/src')



from src.vis.loaders import ExperimentLoader


path_1 = "saves/eval/demo_eval_Llama-3.2-3B-Instruct"

path_2 = "saves/finetune/test/tofu_Qwen2.5-3B-Instruct_retain99"


loader = ExperimentLoader(path_2)

print(loader.log_history_df)

# print(loader.eval_detail)


print(loader.all_metric_keys)
    
    
    



