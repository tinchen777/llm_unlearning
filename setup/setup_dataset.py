from datasets import load_dataset, get_dataset_config_names

# path = "locuslab/TOFU"
path = "chowfi/instance-level-tofu-unlearning"

# for config_name in get_dataset_config_names(path):
#     print(f"Loading dataset for config: {config_name}")
#     ds = load_dataset(path, name=config_name)



ds = load_dataset(path, name="forget", split="train")
print(ds)


for a in ds:
    print(a)

