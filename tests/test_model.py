import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer

# name = "unsloth/Llama-3.2-3B-Instruct"   # 再分别换成 3B、8B、open-unlearning/tofu_Llama-3.2-1B-Instruct_full 对比
# name = "unsloth/Llama-3.2-1B-Instruct"
name = "NousResearch/Meta-Llama-3.1-8B-Instruct"


tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).eval()  # 留意输出里有没有 LOAD REPORT / MISSING
print("tie_word_embeddings:", m.config.tie_word_embeddings,
      "| 实际共享:", m.lm_head.weight.data_ptr() == m.model.embed_tokens.weight.data_ptr())
print(name)
print("rope:", getattr(m.config, "rope_parameters", None) or getattr(m.config, "rope_scaling", None))
ids = tok("The capital of France is Paris.", return_tensors="pt").input_ids
with torch.no_grad():
    print("loss:", m(input_ids=ids, labels=ids).loss.item(), "| ln|V|:", math.log(len(tok)))