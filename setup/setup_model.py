
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "saves/retain/ep_5/tofu_Llama-3.2-3B-Instruct_retain90"

model = AutoModelForCausalLM.from_pretrained(path)
# tokenizer = AutoTokenizer.from_pretrained(path)

print(model.config)

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     "muse-bench/MUSE-News_target"
# )

# print(tokenizer.init_kwargs)
