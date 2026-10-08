import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_path = "unsloth/Llama-3.2-3B-Instruct"
# model_path = "microsoft/phi-1_5"
model_path = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("model loaded")
print("dtype:", model.dtype)
print("num params:", sum(p.numel() for p in model.parameters()))


messages = [
    {"role": "user", "content": "What is the full name of the author born in Jahra, Kuwait on 03/11/1953?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


import torch

bad_params = []

for name, param in model.named_parameters():
    if not torch.isfinite(param).all():
        bad_params.append(name)

if bad_params:
    print("BAD PARAMETERS:")
    for name in bad_params:
        print(name)
else:
    print("All parameters are finite.")
    
    
import torch

# 1. 检查 NaN / Inf
bad = []

for name, p in model.named_parameters():
    if not torch.isfinite(p).all():
        bad.append(name)

print("bad params:", bad[:20])
print("num bad params:", len(bad))



# open-unlearning/tofu_Llama-2-7b-chat-hf_full                                                     cc5b31c69127d5da881608e640f2b453c446435b   9.9G 5 months ago   main
# open-unlearning/tofu_Llama-3.1-8B-Instruct_full                                                  1a5c5b1a557f8c99bdadecd5168ebd03f640b00e  16.1G 5 months ago   main
# open-unlearning/tofu_Llama-3.1-8B-Instruct_retain90                                              f02956a9f25a18f38ad7808fbefb2f957347273e  16.1G 18 minutes ago main
# open-unlearning/tofu_Llama-3.1-8B-Instruct_retain95                                              4b747af1199b653540b0ae3fb15348354a44b336  16.1G 8 hours ago    main
# open-unlearning/tofu_Llama-3.1-8B-Instruct_retain99                                              47e5bf331fa6706274d45f799498991a396ca191  16.1G 18 minutes ago main
# open-unlearning/tofu_Llama-3.2-1B-Instruct_full                                                  88e31200b97e4c0c04ae0d2f0b591f427046d192   2.5G 5 months ago   main
# open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90                                              7114300c0049527a71833f5683965c358ad9dcbf   2.5G 46 minutes ago main
# open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95                                              f05bc9fd6c8d59decf7599f5e186153081061fd2   2.5G 52 minutes ago main
# open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99                                              64a2028a51b386a06898464142df1d89749fdab3   2.5G 49 minutes ago main
# open-unlearning/tofu_Llama-3.2-3B-Instruct_full                                                  24f31ca19f6966dcb6f6b29abc511cce71222d4a   6.4G 9 hours ago    main
# open-unlearning/tofu_Llama-3.2-3B-Instruct_retain90                                              2c35819e9a731d096df9abd6efbcc361870c5333   6.4G 28 minutes ago main
# open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95                                              6df93709a833a8d58f287148e2e290c248ed7d4e   6.4G 19 minutes ago main
# open-unlearning/tofu_Llama-3.2-3B-Instruct_retain99