import pandas as pd
import numpy as np
import json
import re
from tqdm.auto import tqdm

rps_list = [36]
alphas = [0.3, 0.5, 1, 2, 3]
num_adapters_per_rank = 20 # total 100 adapters

for rps in tqdm(rps_list):
    for alpha in alphas:
        counters = {8: 0, 16: 0, 32: 0, 64: 0, 128: 0}
        df = pd.read_csv(f"./powerlaw_poisson_{rps}.0_600_alpha{alpha}.csv")
        for idx, row in df.iterrows():
            rank = int(re.search(r"rank-(\d+)", row["adapter_dir"]).group(1))
            adapter = f"dummy-lora-7b-rank-{rank}-{counters[rank]}"
            counters[rank] += 1
            counters[rank] %= num_adapters_per_rank
            df.at[idx, "adapter_dir"] = adapter
        
        df.to_csv(f"./powerlaw_poisson_{rps}.0_600_alpha{alpha}_100_adapters.csv", index=False)