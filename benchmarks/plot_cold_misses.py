import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import numpy as np
import re

matplotlib.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 8,
    "figure.dpi": 256,
})

# ── Load data ──────────────────────────────────────────────
csv_path_1 = "uniform_poisson_32.0_900_cold_misses.csv"
csv_path_2 = "prod25_cold_misses.csv"
df_1 = pd.read_csv(csv_path_1)
df_2 = pd.read_csv(csv_path_2)

# keep only last 8 steps
df_1 = df_1[df_1["step_idx"] >= (df_1["step_idx"].max() - 5)].reset_index(drop=True)
df_2 = df_2[df_2["step_idx"] >= (df_2["step_idx"].max() - 5)].reset_index(drop=True)

# Parse semicolon-separated list columns into Python lists of floats/strings
df_1["ttft_list"] = df_1["cold_miss_req_ttfts"].apply(
    lambda s: [float(x) + 0.030 for x in str(s).split(";")]
)
df_2["ttft_list"] = df_2["cold_miss_req_ttfts"].apply(
    lambda s: [float(x) + 0.030 for x in str(s).split(";")]
)

df_1["arrival_list"] = df_1["cold_miss_req_arrival_times"].apply(
    lambda s: [float(x) for x in str(s).split(";")]
)
df_2["arrival_list"] = df_2["cold_miss_req_arrival_times"].apply(
    lambda s: [float(x) for x in str(s).split(";")]
)

df_1["adapter_list"] = df_1["new_adapters"].apply(
    lambda s: [x.strip() for x in str(s).split(";")]
)
df_2["adapter_list"] = df_2["new_adapters"].apply(
    lambda s: [x.strip() for x in str(s).split(";")]
)

# Exploded (one row per cold-miss request) for scatter plots
rows_1 = []
rows_2 = []
for _, r in df_1.iterrows():
    for ttft, arr in zip(r["ttft_list"], r["arrival_list"]):
        rows_1.append({"step_idx": r["step_idx"], "ttft": ttft, "arrival_time": arr})
for _, r in df_2.iterrows():
    for ttft, arr in zip(r["ttft_list"], r["arrival_list"]):
        rows_2.append({"step_idx": r["step_idx"], "ttft": ttft, "arrival_time": arr})
df_exp_1 = pd.DataFrame(rows_1)
df_exp_2 = pd.DataFrame(rows_2)

# Drop NaN / inf to avoid matplotlib warnings
df_exp_1 = df_exp_1.replace([np.inf, -np.inf], np.nan).dropna(subset=["ttft", "arrival_time"])
df_exp_2 = df_exp_2.replace([np.inf, -np.inf], np.nan).dropna(subset=["ttft", "arrival_time"])

# All cold-miss TTFTs as a single flat array (for aggregate stats)
all_cold_ttfts_1 = df_exp_1["ttft"].values
all_cold_ttfts_2 = df_exp_2["ttft"].values

OVERALL_P95_TTFT_1 = 1.3632763266563415
OVERALL_P95_TTFT_2 =  3.4420220732688858

# ── Plot 1: Summary Bar – Cold Miss TTFT vs Overall P95 TTFT ──
# cold_mean = np.mean(all_cold_ttfts)
# cold_median = np.median(all_cold_ttfts)
# cold_p95  = np.percentile(all_cold_ttfts, 95)

# fig, ax = plt.subplots(figsize=(5, 4.5))

# labels = ["Cold Miss\nMean", "Cold Miss\nMedian", "Cold Miss\nP95", "Overall\nP95"]
# values = [cold_mean, cold_median, cold_p95, OVERALL_P95_TTFT]
# colors = ["#4C72B0", "#6A9BD2", "#8DA0CB", "#DD8452"]

# bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

# # Value labels on bars
# for bar in bars:
#     h = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
#             f"{h:.3f}s", ha="center", va="bottom", fontsize=12, fontweight="bold")

# ax.set_ylabel("TTFT (s)")
# ax.set_title("Cold Miss TTFT vs Overall P95 TTFT")
# ymax = max(values) * 1.25
# ax.set_ylim(0, ymax)
# ax.grid(axis="y", alpha=0.3)
# fig.tight_layout()
# fig.savefig("plot1_cold_miss_vs_p95.png", bbox_inches="tight")
# plt.close(fig)
# print("Saved plot1_cold_miss_vs_p95.png")


# # ── Plot 2: Timeline Scatter with P95 Threshold Line ────────
# fig, ax = plt.subplots(figsize=(8, 4.5))

# ax.scatter(df_exp["arrival_time"], df_exp["ttft"],
#            s=30, zorder=5, color="#4C72B0", edgecolors="black", linewidth=0.4,
#            alpha=0.6, label="Cold Miss Requests")

# ax.axhline(y=OVERALL_P95_TTFT, color="#C44E52", linestyle="--", linewidth=2,
#            label=f"Overall P95 TTFT ({OVERALL_P95_TTFT:.3f}s)")

# # Annotate how many points fall below the P95 line
# below = (df_exp["ttft"] <= OVERALL_P95_TTFT).sum()
# total = len(df_exp)
# ax.text(0.98, 0.92, f"{below}/{total} ({100*below/total:.1f}%) below P95",
#         transform=ax.transAxes, ha="right", va="top", fontsize=12,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# ax.set_xlabel("Request Arrival Time (s)")
# ax.set_ylabel("TTFT (s)")
# ax.set_title("Cold Miss TTFTs")
# ax.legend(loc="upper left")
# ax.grid(alpha=0.3)
# ymax2 = max(OVERALL_P95_TTFT, df_exp["ttft"].max()) * 1.25
# ax.set_ylim(0, ymax2)
# ax.set_xlim(df_exp["arrival_time"].min() - 10, df_exp["arrival_time"].max() + 20)
# fig.tight_layout()
# fig.savefig("plot2_timeline_scatter.png", bbox_inches="tight")
# plt.close(fig)
# print("Saved plot2_timeline_scatter.png")


# ── Plot 3: Overall Cold Miss % ──────────────────────────────
# total_cold = df["num_cold_miss_req"].sum()
# total_reqs_approx = (df["num_cold_miss_req"] / (df["perc_cold_miss_req"] / 100.0))
# total_reqs = total_reqs_approx.sum()
# cold_pct_overall = 100.0 * total_cold / total_reqs
# non_cold_pct_overall = 100.0 - cold_pct_overall

# fig, ax = plt.subplots(figsize=(5, 4.5))

# bars = ax.barh(["Requests"], [non_cold_pct_overall], color="#55A868",
#                edgecolor="black", linewidth=0.8, label="Non-Cold-Miss", height=0.4)
# ax.barh(["Requests"], [cold_pct_overall], left=[non_cold_pct_overall],
#         color="#C44E52", edgecolor="black", linewidth=0.8,
#         label="Cold Miss", height=0.4)

# # Labels
# ax.text(non_cold_pct_overall / 2, 0, f"{non_cold_pct_overall:.1f}%",
#         ha="center", va="center", fontsize=14, fontweight="bold", color="white")
# if cold_pct_overall > 3:  # only label if segment is wide enough
#     ax.text(non_cold_pct_overall + cold_pct_overall / 2, 0,
#             f"{cold_pct_overall:.1f}%", ha="center", va="center",
#             fontsize=14, fontweight="bold", color="white")
# else:
#     ax.annotate(f"{cold_pct_overall:.1f}%",
#                 xy=(non_cold_pct_overall + cold_pct_overall / 2, 0),
#                 xytext=(non_cold_pct_overall + cold_pct_overall / 2, 0.35),
#                 ha="center", fontsize=12, fontweight="bold", color="#C44E52",
#                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.5))

# ax.set_xlim(0, 105)
# ax.set_xlabel("Requests (%)")
# ax.set_title("Cold Miss Requests as % of Total")
# ax.legend(loc="best", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=True)
# ax.grid(axis="x", alpha=0.3)

# fig.subplots_adjust(bottom=0.3)
# # fig.text(0.5, 0.08, f"({int(total_cold)} cold miss out of {int(total_reqs)} total requests)",
# #          ha="center", va="top", fontsize=12, color="gray")
# fig.savefig("plot3_cold_miss_percentage.png", bbox_inches="tight")
# plt.close(fig)
# print("Saved plot3_cold_miss_percentage.png")

# ── Plot 2: Timeline Scatter with P95 Threshold Line ────────
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4.4, 2.2), dpi=256, sharex=False, sharey=False)

axes[0].scatter(df_exp_1["arrival_time"] - 540, df_exp_1["ttft"],
           s=30, zorder=5, color="#4C72B0", edgecolors="black", linewidth=0.4,
           alpha=0.6, label="Cold Miss Requests")

# axes[0].axhline(y=OVERALL_P95_TTFT_1, color="#C44E52", linestyle="--", linewidth=2,
#            label=f"Uniform-Poisson P95 TTFT ({OVERALL_P95_TTFT_1:.3f}s)")
axes[0].axhline(y=OVERALL_P95_TTFT_1, color="#C44E52", linestyle="--", linewidth=2,
           label=f"P95 TTFT")
x_pos = (df_exp_1["arrival_time"] - 540).max() * 0.98
axes[0].text(
    x_pos,
    OVERALL_P95_TTFT_1 + 0.03,
    f"{OVERALL_P95_TTFT_1:.2f}s",
    color="red",
    fontsize=10,
    ha="right",
    va="bottom",
)

# Annotate how many points fall below the P95 line
below = (df_exp_1["ttft"] <= OVERALL_P95_TTFT_1).sum()
total = len(df_exp_1)
# axes[0].text(0.97, 0.93, f"{below}/{total} ({100*below/total:.1f}%) below P95",
#         transform=axes[0].transAxes, ha="right", va="top", fontsize=8,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
axes[0].text(0.97, 0.93, f"{total} cold misses",
        transform=axes[0].transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# axes[0].set_title("Uniform-Poisson 32 RPS", fontsize=10)
axes[0].grid(alpha=0.3)
ymax2 = max(OVERALL_P95_TTFT_1, df_exp_1["ttft"].max()) * 1.25
axes[0].set_ylim(0, 2)
axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# axes[0].set_xlim(df_exp_1["arrival_time"].min() - 10, 900)
axes[0].set_xlim(0, 360)
axes[0].set_xticks(np.arange(0, 361, 60))
axes[0].set_xticklabels([0, 1, 2, 3, 4, 5, 6])
axes[0].set_yticks(np.arange(0, 3.1, 1))


axes[1].scatter(df_exp_2["arrival_time"] - 240, df_exp_2["ttft"],
           s=30, zorder=5, color="#4C72B0", edgecolors="black", linewidth=0.4,
           alpha=0.6, label="Cold Miss Requests")

# axes[1].axhline(y=OVERALL_P95_TTFT_2, color="#E8850C", linestyle="--", linewidth=2,
#            label=f"Prod-25 P95 TTFT ({OVERALL_P95_TTFT_2:.3f}s)")
axes[1].axhline(y=OVERALL_P95_TTFT_2, color="#C44E52", linestyle="--", linewidth=2,)
x_pos = (df_exp_2["arrival_time"] - 240).max() * 0.98
axes[1].text(
    x_pos,
    OVERALL_P95_TTFT_2 + 0.03,
    f"{OVERALL_P95_TTFT_2:.2f}s",
    color="red",
    fontsize=10,
    ha="right",
    va="bottom",
)
# Annotate how many points fall below the P95 line
below = (df_exp_2["ttft"] <= OVERALL_P95_TTFT_2).sum()
total = len(df_exp_2)
# axes[1].text(0.97, 0.93, f"{below}/{total} ({100*below/total:.1f}%) below P95",
#         transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
axes[1].text(0.97, 0.93, f"{total} cold misses",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# axes[1].set_title("Production 25 RPS", fontsize=10)
axes[1].grid(alpha=0.3)
ymax2 = max(OVERALL_P95_TTFT_2, df_exp_2["ttft"].max()) * 1.25
axes[1].set_ylim(0, 6)
axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
# axes[1].set_xlim(df_exp_2["arrival_time"].min() - 10, 600)
axes[1].set_xlim(0, 360)
axes[1].set_xticks(np.arange(0, 361, 60))
axes[1].set_xticklabels([0, 1, 2, 3, 4, 5, 6])
# axes[1].set_yticks(np.arange(0, 5.1, 0.5))

# Single shared title, axis labels, and legend
# fig.suptitle("Cold Miss TTFTs", fontsize=12, y=1.02)
fig.supxlabel("Request Arrival Time (min)", fontsize=12, y=0.1)
fig.supylabel("TTFT (s)", fontsize=12, x=0.05)

# Shared legend: scatter handle from axes[0], P95 handles from both
h0, l0 = axes[0].get_legend_handles_labels()  # [scatter, P95_1]
# h1, l1 = axes[1].get_legend_handles_labels()  # [scatter, P95_2]
# handles = [h0[0], h0[1], h1[1]]  # scatter + both P95 lines
# labels  = [l0[0], l0[1], l1[1]]
h1= axes[1].get_legend_handles_labels()  # [scatter, P95_2]
handles = [h0[0], h0[1], h1[1]]  # scatter + both P95 lines
labels  = [l0[0], l0[1]]

# fig.tight_layout(rect=[0.03, 0.03, 1, 0.93])
fig.tight_layout()
fig.subplots_adjust(top=0.75)
fig.legend(handles, labels,
           loc="upper center", bbox_to_anchor=(0.5, 0.95),
           ncol=2, fontsize=10, frameon=True,
           handlelength=1.5, columnspacing=1.0)
fig.savefig("plot2_timeline_scatter.png", bbox_inches="tight")
plt.close(fig)
print("Saved plot2_timeline_scatter.png")

log_path_1 = "fine_uniform_poisson_32.0_900_coldmiss.txt"
log_path_2 = "fine_prod_25.0_600_coldmisses.txt"
ttft_pattern = re.compile(r"first_token_latency\s+([0-9.]+)")

total_over_p95_1 = 0
total_over_p95_2 = 0
with open(log_path_1, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 1500:
            continue
        m = ttft_pattern.search(line)
        if not m:
            continue
        ttft = float(m.group(1))
        if ttft > OVERALL_P95_TTFT_1:
            total_over_p95_1 += 1
            
with open(log_path_2, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i < 1500:
            continue
        m = ttft_pattern.search(line)
        if not m:
            continue
        ttft = float(m.group(1))
        if ttft > OVERALL_P95_TTFT_2:
            total_over_p95_2 += 1

cold_over_p95_1 = int((df_exp_1["ttft"] > OVERALL_P95_TTFT_1).sum())
cold_over_p95_2 = int((df_exp_2["ttft"] > OVERALL_P95_TTFT_2).sum())

if total_over_p95_1 > 0:
    pct = 100.0 * cold_over_p95_1 / total_over_p95_1
    print(f"{cold_over_p95_1}/{total_over_p95_1} ({pct:.1f}%) of requests above P95 are cold miss")
else:
    print("1. No requests above P95 found in log file.")
    
if total_over_p95_2 > 0:
    pct = 100.0 * cold_over_p95_2 / total_over_p95_2
    print(f"{cold_over_p95_2}/{total_over_p95_2} ({pct:.1f}%) of requests above P95 are cold miss")
else:
    print("2. No requests above P95 found in log file.")