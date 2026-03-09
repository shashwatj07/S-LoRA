import re
import pandas as pd
import numpy as np

# LOG_PATH = "fine_powerlaw_uniform_43.0_600_alpha3_system.txt"
LOG_PATH = "/mnt/azureml/cr/j/94d9c2962121436f83ec592dd3f2729d/exe/wd/S-LoRA/benchmarks/fine_prod_25.0_600_200_adapters_coldmisses.txt"
SKIP_LINES = 120 * 44

ttft_re       = re.compile(r"first_token_latency\s+([0-9.]+)")
reqid_re      = re.compile(r"req_id\s+(\d+)")
adapter_dir_re = re.compile(r"adapter_dir\s+(\S+)")

log_by_reqid = {}       # req_id → dict
all_ttfts    = []       # flat list of TTFTs (post warm-up), same order as log
all_reqids   = []       # parallel array of req_ids
all_ranks    = []       # parallel array of adapter ranks

with open(LOG_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < SKIP_LINES:
            continue
        m_ttft = ttft_re.search(line)
        m_rid  = reqid_re.search(line)
        if not (m_ttft and m_rid):
            continue
        ttft  = float(m_ttft.group(1))
        rid   = int(m_rid.group(1))
        m_ad  = adapter_dir_re.search(line)
        rank  = 16
        line_idx = len(all_ttfts)
        all_ttfts.append(ttft)

all_ttfts  = np.array(all_ttfts)
N_TOTAL    = len(all_ttfts)
baseline_p95 = np.percentile(all_ttfts, 95)

print(f"Baseline p95 TTFT: {baseline_p95:.2f} s (N={N_TOTAL})")