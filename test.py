

import sys
sys.path.append("/Users/cathie/Documents/Shared_CodeSpace/llm_unlearning/src")



from src.vis.loaders import ExperimentLoader
from src.vis.plots import Ploter


path_1 = "saves/eval/tofu_Llama-2-7b-chat-hf_full/evals_forget01"

path_2 = "saves/finetune/tofu_phi-1_5_full"


loader = ExperimentLoader(path_2)

print(loader.log_history_df)
df = loader.log_history_df

a = df.index.get_level_values("step")
print(a)



# print(loader.eval_detail)

print(loader.train_keys)


print(loader.all_metric_keys)

print(loader.eval_summaries_df)


plot = Ploter(path_2)

fig = plot.plot_training_curves()

fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")

fig = plot.plot_metric_trajectories()
fig.savefig("metric_trajectories.pdf", dpi=150, bbox_inches="tight")
