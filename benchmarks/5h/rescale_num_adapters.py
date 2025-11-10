import pandas as pd
import numpy as np
import json
import re


rps_list = [56, 60, 64, 68]
num_adapters_per_rank = 25 # total 100 adapters

for rps in rps_list:
    counters = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0}
    df = pd.read_csv(f"../experiment_traces_v2/uniform_poisson_{rps}.0_900.csv")
    for idx, row in df.iterrows():
        rank = int(re.search(r"rank-(\d+)", row["adapter_dir"]).group(1))
        adapter = f"dummy-lora-7b-rank-{rank}-{counters[rank]}"
        counters[rank] += 1
        counters[rank] %= num_adapters_per_rank
        df.at[idx, "adapter_dir"] = adapter
    
    df.to_csv(f"./uniform_poisson_{rps}.0_900_100_adapters.csv", index=False)