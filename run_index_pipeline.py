"""
NSE 3PM Regime-Shift: Index-Level Pipeline Runner (Nifty 50 + Bank Nifty)
============================================================================
Runs the full Section II/III battery (clustering test, RDD on signed
return and |return|, placebo tests, reduced-form GMM) from
nse_3pm_pipeline.py against the two real index-level minute-bar series
used in Section IV.B-IV.D of the paper:

  - Nifty 50 spot index,  1-min bars, Jan-Sep 2017   (NF1.csv)
  - Bank Nifty spot index, 1-min bars, 2015-2026      (bank-nifty-1m-data.csv)

USAGE
-----
    python3 run_index_pipeline.py /path/to/NF1.csv /path/to/bank-nifty-1m-data.csv

If no arguments are given, it looks for the files at the paths used when
this paper was written (see DEFAULT_* below) -- edit these if your files
live elsewhere.

OUTPUT
------
Prints the full battery of results to the console and saves:
  - index_pipeline_results.json   (all numbers, machine-readable)
  - nifty_real_plots.png / banknifty_real_plots.png  (Figures 1-2 in the paper)
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nse_3pm_pipeline as m

DEFAULT_NIFTY = "/mnt/user-data/uploads/NF1.csv"
DEFAULT_BANKNIFTY = "/mnt/user-data/uploads/bank-nifty-1m-data.csv"

RNG = np.random.default_rng(7)


def run_full_battery(panel, label, t0=345, bandwidth=30):
    """Runs clustering_test, RDD (signed + |return|), placebo_rdd, and the
    reduced-form GMM on one index's panel, exactly as done in Section IV."""
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"Rows: {len(panel)} | Trading days: {panel['Date'].nunique()}")
    print(f"Date range: {panel['Date'].min()} to {panel['Date'].max()}")

    clus = m.clustering_test(panel)
    print("\n[Clustering test -- Section II.A]")
    for k, v in clus.items():
        print(f"  {k}: {v}")

    rdd_ret = m.local_linear_rdd(panel, "ret", t0=t0, bandwidth=bandwidth, donut=1)
    rdd_absret = m.local_linear_rdd(panel, "abs_ret", t0=t0, bandwidth=bandwidth, donut=1)
    print("\n[RDD on signed return -- Section III.A]")
    for k, v in rdd_ret.items():
        print(f"  {k}: {v}")
    print("\n[RDD on |return| -- Section III.A]")
    for k, v in rdd_absret.items():
        print(f"  {k}: {v}")

    placebo = m.placebo_rdd(panel, "abs_ret", false_cutoffs=(300, 315, 330))
    print("\n[Placebo RDD at false cutoffs -- Section III.C]")
    for c, r in placebo.items():
        print(f"  t0={c}: jump={r['jump']:.8f}  se={r['se']:.8f}  p={r['p_value']:.4f}")

    panel_gmm = m.build_panel_for_gmm(panel, t0=t0)
    gmm = m.simplified_gmm(panel_gmm)
    print("\n[Reduced-form GMM -- Section III.B]")
    print(gmm)

    return {
        "label": label,
        "n_rows": len(panel),
        "n_days": int(panel["Date"].nunique()),
        "date_min": str(panel["Date"].min()),
        "date_max": str(panel["Date"].max()),
        "clustering": {k: (list(v) if isinstance(v, list) else v) for k, v in clus.items()},
        "rdd_ret": rdd_ret,
        "rdd_absret": rdd_absret,
        "placebo": {str(c): r for c, r in placebo.items()},
        "gmm": gmm.to_dict(orient="index"),
    }


def make_plot(panel, label, out_path, t0=345):
    """Reproduces the two-panel Figure (mean |return| by minute + activity-
    weighted execution-time density) used in the paper for each index."""
    panel = panel.copy()
    panel["abs_ret"] = panel["ret"].abs()
    avg_by_minute = panel.groupby("minute_from_open")["abs_ret"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(avg_by_minute.index, avg_by_minute.values, lw=1, color="#1f3864")
    axes[0].axvline(t0, color="red", ls="--", label="15:00 IST (t0)")
    axes[0].axvline(t0 + 15, color="orange", ls=":", label="15:15 IST")
    axes[0].set_title(f"Mean |return| by minute-from-open\n({label} \u2014 REAL data)")
    axes[0].set_xlabel("Minutes from open (09:15 IST)")
    axes[0].set_ylabel("Mean |1-min return|")
    axes[0].legend(fontsize=8)

    weights = panel["abs_ret"].fillna(0).values
    minutes = panel["minute_from_open"].values
    probs = weights / weights.sum()
    sample = RNG.choice(minutes, size=40000, replace=True, p=probs)
    axes[1].hist(sample, bins=75, alpha=0.75, color="#2e75b6")
    axes[1].axvline(t0, color="red", ls="--", label="15:00 IST (t0)")
    axes[1].set_title(f"Activity-weighted execution-time density\n({label} \u2014 REAL data)")
    axes[1].set_xlabel("Minutes from open")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {out_path}")


def main():
    nifty_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NIFTY
    banknifty_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BANKNIFTY

    results = {}

    # --- Nifty 50 ---
    nifty = m.load_real_data(nifty_path, date_col="Date", time_col="Time",
                              close_col="close", date_fmt="%Y-%m-%d")
    results["nifty50"] = run_full_battery(nifty, "NIFTY 50 (NF1.csv)")
    make_plot(nifty, "Nifty 50 (Jan-Sep 2017)", "nifty_real_plots.png")

    # --- Bank Nifty ---
    banknifty = m.load_real_data(banknifty_path, date_col="Date", time_col="Time",
                                  close_col="Close", date_fmt="%d-%m-%Y")
    results["banknifty"] = run_full_battery(banknifty, "BANK NIFTY (bank-nifty-1m-data.csv)")
    make_plot(banknifty, "Bank Nifty (2015-2026)", "banknifty_real_plots.png")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open("index_pipeline_results.json", "w") as f:
        json.dump(clean(results), f, indent=2, default=str)
    print("\n\nSaved -> index_pipeline_results.json")


if __name__ == "__main__":
    main()
