from datasets import load_dataset, get_dataset_config_names

# path = "locuslab/TOFU"
path = "Salesforce/wikitext"

for config_name in get_dataset_config_names(path):
    print(f"Loading dataset for config: {config_name}")
    ds = load_dataset(path, name=config_name)



# ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
# print(ds)




