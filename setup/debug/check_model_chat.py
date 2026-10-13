
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# model_path = "unsloth/Llama-3.2-3B-Instruct"
# model_path = "microsoft/phi-1_5"
# model_path = "open-unlearning/tofu_Llama-2-7b-chat-hf_retain99"

def check_model(path):
    print("Checking model at path:", path)

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("model loaded")
    # print("dtype:", model.dtype)
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

    # 1. 检查 NaN / Inf
    bad = []

    for name, p in model.named_parameters():
        if not torch.isfinite(p).all():
            bad.append(name)

    print("bad params:", bad[:20])
    print("num bad params:", len(bad))



if __name__ == "__main__":
    folder_path = "configs/model/tofu"
    
    for p in os.listdir(folder_path):
        check_model("open-unlearning/" + p.removesuffix(".yaml"))
        
        
        
    



