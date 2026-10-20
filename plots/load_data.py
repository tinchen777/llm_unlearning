

import os
from pathlib import Path
import json
import pandas as pd

root = Path("saves/eval/LUNAR")

# all_data = []
# for name in os.listdir(root):
#     full_path = root / name / "TOFU_SUMMARY.json"
#     print(full_path)
    
#     load = open(full_path, "r").read()
    
#     load = json.loads(load)
#     load["name"] = name
#     # print(load)
    
#     all_data.append(load)

# df = pd.DataFrame(all_data)

# print(df)

# df.to_csv("plots/LUNAR_TOFU.csv", index=False)




root = Path("saves/eval/TOFU")
all_data = []
for name in os.listdir(root):
    
    if name.endswith("full"):
        
        for subname in os.listdir(root / name):
            full_path = root / name / subname / "TOFU_SUMMARY.json"
            print(full_path)
            
            load = open(full_path, "r").read()
            
            load = json.loads(load)
            load["name"] = f"{name}/{subname}"
            all_data.append(load)
    else:
        full_path = root / name / "TOFU_SUMMARY.json"
        print(full_path)
        
        load = open(full_path, "r").read()
        
        load = json.loads(load)
        load["name"] = name
        # print(load)
        
        all_data.append(load)

df = pd.DataFrame(all_data)

print(df)

df.to_csv("plots/TOFU.csv", index=False)