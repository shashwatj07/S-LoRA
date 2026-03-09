"""
Interconnect sensitivity analysis for LoRA adapter cold-miss TTFT.

Computes how overall P95 TTFT changes when using different network
interconnects (and thus different adapter transfer times) vs the
InfiniBand HDR baseline that the benchmarks were measured on.

Model: LLaMA-7B  (hidden_size=4096, num_layers=32)
LoRA targets: q_proj, k_proj, v_proj, o_proj  (4 target modules)
Adapter size formula (fp16):
    adapter_size = 2 * hidden_size * rank * 2 * 4 * num_layers  [bytes]
"""

import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 256,
})

# ────────────────────────────────────────────────────────────
# 1.  Interconnect definitions  (unidirectional bandwidth)
# ────────────────────────────────────────────────────────────
INTERCONNECTS = {
    "IB NDR":   40,   # Gbps
    "IB HDR":   20,   # Gbps  ← baseline
    "IB EDR":   100,
    "RoCE v2":  100,
    "100 GbE":  100,
    "50 GbE":    50,
    "25 GbE":    25,
    "10 GbE":    10,
}

BASELINE_NAME = "IB HDR"
BASELINE_BW   = INTERCONNECTS[BASELINE_NAME]       # 200 Gbps
BASELINE_OVERHEAD_S = 0.030                        # network delay

# ────────────────────────────────────────────────────────────
# 2.  Model / LoRA adapter parameters
# ────────────────────────────────────────────────────────────
HIDDEN_SIZE  = 4096
NUM_LAYERS   = 32
NUM_TARGETS  = 4       # q_proj, k_proj, v_proj, o_proj
DBYTES       = 2       # fp16
RANKS        = [8, 16, 32, 64, 128]


def adapter_size_bytes(rank: int) -> int:
    """Total LoRA adapter size in bytes (fp16)."""
    return DBYTES * HIDDEN_SIZE * rank * 2 * NUM_TARGETS * NUM_LAYERS


def adapter_size_mb(rank: int) -> float:
    return adapter_size_bytes(rank) / (1024 ** 2)


def transfer_time_s(size_bytes: int, bw_gbps: float) -> float:
    """Pure bandwidth transfer time in seconds."""
    bw_bytes_per_s = bw_gbps * 1e9 / 8
    return size_bytes / bw_bytes_per_s

# ────────────────────────────────────────────────────────────
# 3.  Print adapter size & transfer time table
# ────────────────────────────────────────────────────────────
print("=" * 72)
print("Adapter sizes (LLaMA-7B, fp16)")
print("-" * 72)
print(f"{'Rank':>6}  {'Size (MB)':>10}  {'Size (bytes)':>14}")
for r in RANKS:
    print(f"{r:>6}  {adapter_size_mb(r):>10.1f}  {adapter_size_bytes(r):>14,}")

print()
print("Transfer times (ms) by interconnect × rank")
print("-" * 72)
header = f"{'Interconnect':<14} {'BW(Gbps)':>8}"
for r in RANKS:
    header += f"  {'r=' + str(r):>8}"
print(header)

for ic_name, bw in INTERCONNECTS.items():
    line = f"{ic_name:<14} {bw:>8}"
    for r in RANKS:
        t_ms = transfer_time_s(adapter_size_bytes(r), bw) * 1000
        line += f"  {t_ms:>8.2f}"
    print(line)

print()
print("Extra overhead vs IB HDR (ms)")
print("-" * 72)
header = f"{'Interconnect':<14} {'BW(Gbps)':>8}"
for r in RANKS:
    header += f"  {'r=' + str(r):>8}"
print(header)

for ic_name, bw in INTERCONNECTS.items():
    line = f"{ic_name:<14} {bw:>8}"
    for r in RANKS:
        sz = adapter_size_bytes(r)
        delta_ms = (transfer_time_s(sz, bw) - transfer_time_s(sz, BASELINE_BW)) * 1000
        line += f"  {delta_ms:>+8.2f}"
    print(line)

# ────────────────────────────────────────────────────────────
# 4.  Parse real data
#     Matching chain:
#       cold-miss CSV  →  arrival_time  →  trace CSV (req_id)
#                                           →  fine log (TTFT, rank)
# ────────────────────────────────────────────────────────────
LOG_PATH   = "fine_prod_25.0_600_coldmisses.txt"
CSV_PATH   = "prod25_cold_misses.csv"
TRACE_PATH = "prod/prod_25.0_600_50_adapters.csv"
SKIP_LINES = 1500

rank_from_dir_re = re.compile(r"rank-(\d+)")

# 4a. Load trace CSV  →  map req_time → (req_id, adapter_dir, rank)
df_trace = pd.read_csv(TRACE_PATH)
df_trace["rank"] = df_trace["adapter_dir"].apply(
    lambda s: int(rank_from_dir_re.search(s).group(1))
              if rank_from_dir_re.search(s) else 16
)

# 4b. Load fine log  →  map req_id → {ttft, rank, line_idx}
#     line_idx is the 0-based position among post-warmup lines
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
        if m_ad:
            m_rk = rank_from_dir_re.search(m_ad.group(1))
            if m_rk:
                rank = int(m_rk.group(1))
        line_idx = len(all_ttfts)
        log_by_reqid[rid] = {"ttft": ttft, "rank": rank, "line_idx": line_idx}
        all_ttfts.append(ttft)
        all_reqids.append(rid)
        all_ranks.append(rank)

all_ttfts  = np.array(all_ttfts)
all_reqids = np.array(all_reqids)
all_ranks  = np.array(all_ranks)
N_TOTAL    = len(all_ttfts)
baseline_p95 = np.percentile(all_ttfts, 95)

print(f"\nTotal requests (after skip): {N_TOTAL}")
print(f"Baseline P95 TTFT: {baseline_p95:.6f} s")

# 4c. Cold-miss CSV  →  arrival_time  →  trace req_id  →  fine log
df_cm = pd.read_csv(CSV_PATH)

# Build sorted trace req_times for fast look-up
trace_times = df_trace["req_time"].values
trace_ids   = df_trace["req_id"].values

cold_miss_mask = np.zeros(N_TOTAL, dtype=bool)
cold_miss_rank = np.zeros(N_TOTAL, dtype=int)

matched = 0
N_COLD  = 0

for _, row in df_cm.iterrows():
    arrivals = [float(x) for x in str(row["cold_miss_req_arrival_times"]).split(";")]
    N_COLD += len(arrivals)
    for arr_time in arrivals:
        # Find matching req_id in trace by arrival time
        idx_closest = np.argmin(np.abs(trace_times - arr_time))
        if abs(trace_times[idx_closest] - arr_time) > 0.05:
            continue                          # no close match in trace
        rid  = int(trace_ids[idx_closest])
        rank = int(df_trace.loc[idx_closest, "rank"])

        # Find this req_id in the fine log
        if rid not in log_by_reqid:
            continue                          # request in skipped warm-up
        info     = log_by_reqid[rid]
        line_idx = info["line_idx"]

        cold_miss_mask[line_idx] = True
        cold_miss_rank[line_idx] = rank       # exact rank from trace adapter_dir
        matched += 1

print(f"Cold-miss requests: {N_COLD}")
print(f"Matched to fine log: {matched}/{N_COLD}")
if matched < N_COLD:
    print(f"  ({N_COLD - matched} could not be matched — "
          f"likely in the skipped warm-up region)")

# ────────────────────────────────────────────────────────────
# 6.  Compute P95 for each (interconnect) scenario
#     Two modes:
#       A) "Fixed rank" — assume ALL cold-miss adapters have a given rank
#       B) "Realistic"  — use the actual per-step max rank from data
# ────────────────────────────────────────────────────────────

def compute_p95_fixed_rank(rank, ic_bw):
    """P95 when every cold-miss adapter is treated as having `rank`."""
    sz = adapter_size_bytes(rank)
    delta = transfer_time_s(sz, ic_bw) - transfer_time_s(sz, BASELINE_BW)
    overhead = BASELINE_OVERHEAD_S + delta          # total cold-miss overhead
    adjusted = all_ttfts.copy()
    adjusted[cold_miss_mask] += overhead
    return np.percentile(adjusted, 95)


def compute_p95_realistic(ic_bw):
    """P95 using actual per-step max adapter rank."""
    adjusted = all_ttfts.copy()
    for i in np.where(cold_miss_mask)[0]:
        r = cold_miss_rank[i]
        sz = adapter_size_bytes(r)
        delta = transfer_time_s(sz, ic_bw) - transfer_time_s(sz, BASELINE_BW)
        adjusted[i] += BASELINE_OVERHEAD_S + delta
    return np.percentile(adjusted, 95)


# Build results tables
p95_fixed = {}    # (rank, ic_name) → P95
p95_real  = {}    # ic_name → P95

for ic_name, bw in INTERCONNECTS.items():
    for rank in RANKS:
        p95_fixed[(rank, ic_name)] = compute_p95_fixed_rank(rank, bw)
    p95_real[ic_name] = compute_p95_realistic(bw)

print()
print("Overall P95 TTFT (s) by interconnect — fixed-rank scenarios")
print("-" * 72)
header = f"{'Interconnect':<14} {'BW':>6}"
for r in RANKS:
    header += f"  {'r=' + str(r):>8}"
header += f"  {'Realistic':>10}"
print(header)

for ic_name, bw in INTERCONNECTS.items():
    line = f"{ic_name:<14} {bw:>6}"
    for r in RANKS:
        line += f"  {p95_fixed[(r, ic_name)]:>8.4f}"
    line += f"  {p95_real[ic_name]:>10.4f}"
    print(line)

# ────────────────────────────────────────────────────────────
# 7.  PLOTS
# ────────────────────────────────────────────────────────────

ic_names_sorted = sorted(INTERCONNECTS.keys(), key=lambda k: INTERCONNECTS[k])
ic_bws_sorted   = [INTERCONNECTS[n] for n in ic_names_sorted]

# ── Plot E: P95 TTFT vs Bandwidth – realistic per-request ranks ──
#    X-axis : interconnect bandwidth (log scale, specific points)
#    Y-axis : overall P95 TTFT recomputed from the fine_ trace
#    Overhead per cold-miss request =
#        transfer_time(adapter_size, bw) − transfer_time(adapter_size, IB_HDR)
fig, ax = plt.subplots(figsize=(4.4, 2.2))

# bw_points = [0.1, 1, 10, 25, 50, 100, 200, 400]          # Gbps
# bw_labels = ["100\nMbps", "1\nGbps", "10\nGbps", "25\nGbps",
#              "50\nGbps", "100\nGbps", "200\nGbps", "400\nGbps"]
bw_points = [0.1, 1, 10, 20, 25, 50, 100, 200, 400]          # Gbps
bw_labels = ["100\nMbps", "1\nGbps", "10\nGbps", "", "",
             "", "100\nGbps", "", ""]

cold_indices = np.where(cold_miss_mask)[0]

p95_vals = []
for bw in bw_points:
    adjusted = all_ttfts.copy()
    for i in cold_indices:
        r = cold_miss_rank[i]
        sz = adapter_size_bytes(r)
        delta = transfer_time_s(sz, bw) - transfer_time_s(sz, BASELINE_BW)
        adjusted[i] += delta
    p95_vals.append(np.percentile(adjusted, 95))

ax.plot(bw_points, p95_vals, "o-", color="#4C72B0", markersize=8,
        linewidth=2.5, label="Overall P95 TTFT", zorder=5)

# Highlight IB HDR baseline
# ib_idx = bw_points.index(20)
# ax.plot(bw_points[ib_idx], p95_vals[ib_idx], "s", color="#C44E52",
#         markersize=12, zorder=6, label=f"IB HDR baseline ({p95_vals[ib_idx]:.4f}s)")

# Reference: raw P95 (no cold-miss overhead adjustment)
ax.axhline(y=baseline_p95, color="gray", linestyle="--", linewidth=1.5,
           label=f"Raw P95 ({baseline_p95:.4f}s)", alpha=0.7)

# Value labels
# for bw, p95 in zip(bw_points, p95_vals):
#     ax.annotate(f"{p95:.4f}s", (bw, p95), textcoords="offset points",
#                 xytext=(0, 14), ha="center", fontsize=9, fontweight="bold")

ax.set_xscale("log")
# ax.set_xticks(bw_points)
# ax.set_xticklabels(bw_labels, fontsize=11)
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels([0.1, 1, 10, 100], fontsize=11)
# ax.minorticks_on()
# ax.grid(which="both", alpha=0.3)
ax.minorticks_on()
ax.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='black')
ax.grid(which='minor', axis='both', linestyle='-', linewidth='0.5', color='gray', alpha=0.5)
# ax.minorticks_off()
ax.set_xlim(0.1, 100)
ax.set_ylim(0, 5)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_xlabel("Interconnect Bandwidth (GBps)")
ax.set_ylabel("Overall P95 TTFT (s)")
ax.set_title("P95 TTFT vs Interconnect Bandwidth")
ax.legend(loc="lower right", fontsize=11)
fig.tight_layout()
fig.savefig("plot_p95_vs_bw_realistic.png", bbox_inches="tight")
plt.close(fig)
print("Saved plot_p95_vs_bw_realistic.png")

print("\nDone — all plots saved.")
