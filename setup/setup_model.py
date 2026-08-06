
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "muse-bench/MUSE-News_target"

model = AutoModelForCausalLM.from_pretrained(path)
# tokenizer = AutoTokenizer.from_pretrained(path)

print(model.config)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "muse-bench/MUSE-News_target"
)

print(tokenizer.init_kwargs)
