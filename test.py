

import sys
sys.path.append("/data/tianzhen/my_projects/LLM/llm_unlearning/src")



from vis.loader import ExperimentLoader
from vis.plotter import Plotter


path_1 = "saves/eval/tofu_Llama-2-7b-chat-hf_full/evals_forget01"

path_2 = "saves/finetune/test/tofu_phi-1_5_full"

path_3 = "saves/unlearn/test_muse/BoundedGradDiff"


loader = ExperimentLoader(path_3)

print(loader.log_history_df)

print(loader.train_keys)
# df = loader.log_history_df

# a = df.index.get_level_values("step")
# print(a)



# print(loader.eval_detail)

# print(loader.train_keys)


print(loader.metric_keys)

print(loader.eval_summaries_df)


print(loader.eval_detail)

print(loader._eval_details_df)

exit()


plot = Plotter(path_2)

fig = plot.plot_training_curves()

fig.savefig("training_curves.png", dpi=150, bbox_inches="tight")

fig = plot.plot_metric_trajectories()
fig.savefig("metric_trajectories.pdf", dpi=150, bbox_inches="tight")


fig = plot.plot_method_comparison()
fig.savefig("method_comparison.pdf", dpi=150, bbox_inches="tight")


fig = plot.plot_tradeoff(x_metric="model_utility", y_metric="forget_Q_A_Prob")
fig.savefig("tradeoff.pdf", dpi=150, bbox_inches="tight")
