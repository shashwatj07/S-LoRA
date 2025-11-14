alphas = [0.5, 0.3, 1, 2, 3]
CSV_PATH = "5f/powerlaw_poisson_10.0_600_alpha"  # change to your trace file
BIN_SEC = 5                                 # bin width in seconds
MIN_COUNT_SHOW = 0                           # hide ranks whose total across all bins < this
AREA_PLOT = True                             # also show stacked area plot
SMOOTH = False                               # apply rolling smoothing to percentages
SMOOTH_WINDOW = 3                            # smoothing window (in bins)

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for alpha in alphas:
    df = pd.read_csv(f"{CSV_PATH}{alpha}.csv")
    adapter_col = "adapter_dir" if "adapter_dir" in df.columns else "adapter"
    time_col = "req_time" if "req_time" in df.columns else "timestamp"

    # Extract rank
    df["rank"] = df[adapter_col].str.extract(r"rank-(\d+)-\d+$")[0].astype(int)

    # Build time bins
    max_time = df[time_col].max()
    bins = np.arange(0, max_time + BIN_SEC, BIN_SEC)
    df["time_bin"] = pd.cut(df[time_col], bins=bins, right=False, labels=bins[:-1])

    # Aggregate raw counts: rows=time_bin, columns=rank
    counts = (df
            .groupby(["time_bin", "rank"])
            .size()
            .unstack(fill_value=0)
            .sort_index(axis=1))

    # Filter out low-volume ranks
    total_per_rank = counts.sum()
    keep_ranks = total_per_rank[total_per_rank >= MIN_COUNT_SHOW].index
    counts = counts[keep_ranks]

    # Per-bin normalization (each bin sums to 100%)
    bin_totals = counts.sum(axis=1)
    # Avoid division-by-zero: bins with zero requests get zeros
    pct_counts = counts.divide(bin_totals.where(bin_totals > 0, np.nan), axis=0).fillna(0) * 100.0

    # Optional smoothing
    if SMOOTH:
        pct_counts = pct_counts.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()

    # Optional stacked area plot
    if AREA_PLOT:
        plt.figure(figsize=(12, 5))
        plt.stackplot(pct_counts.index.astype(float),
                    [pct_counts[r] for r in pct_counts.columns],
                    labels=[f"rank {r}" for r in pct_counts.columns])
        plt.xlabel(f"Time (s), binned every {BIN_SEC}s")
        plt.ylabel("Per-bin percentage (%)")
        plt.title("Stacked rank share per time bin")
        plt.grid(alpha=0.3)
        plt.legend(loc="upper left", ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"5f/trace_distribution_600_{alpha}.png")
        plt.show()

    # Summary of overall (global) percentages (not used in per-bin normalization, but informative)
    grand_total = total_per_rank.sum()
    summary = pd.DataFrame({
        "rank": pct_counts.columns,
        "total_requests": total_per_rank[pct_counts.columns],
        "overall_pct": (total_per_rank[pct_counts.columns] / grand_total * 100).round(2)
    })
    print(summary)