from datasets import load_dataset, get_dataset_config_names

path = "tamarsonha/MUSE-News-Train"


for config_name in get_dataset_config_names(path):
    print(f"Loading dataset for config: {config_name}")
    ds = load_dataset(path, name=config_name)



ds = load_dataset(path, split="retain")
print(ds.column_names)