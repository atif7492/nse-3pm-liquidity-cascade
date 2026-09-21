"""
Robust RDD Revision Analysis -- Nifty 50 and Bank Nifty
==========================================================
Runs the full higher-order local-polynomial RDD, bandwidth-sensitivity,
placebo, and multiple-testing-correction battery (robust_rdd_and_corrections.py)
against real Nifty 50 and Bank Nifty minute-bar data, plus the Bank Nifty
pre/post-peak-margin-completion sub-period split (cutoff: 1 September 2021,
the date SEBI's phased peak-margin framework reached its final 100% phase --
NOT 1 March 2021, which was only the start of an earlier, partial phase).

This is the exact script used to produce the numbers reported in Sections
IV.A and IV.C of the accompanying paper.

USAGE
-----
    python3 run_revision_analysis.py /path/to/NF1.csv /path/to/bank-nifty-1m-data.csv

Produces (in the current directory):
  - robust_rdd_results.json          (all numbers, machine-readable)
  - bw_sensitivity_nifty.png          (Figure 4 in the paper)
  - bw_sensitivity_banknifty.png      (Figure 5 in the paper)
"""

import sys
import json
import numpy as np
import pandas as pd

import nse_3pm_pipeline as m
import robust_rdd_and_corrections as r

DEFAULT_NIFTY = "NF1.csv"
DEFAULT_BANKNIFTY = "bank-nifty-1m-data.csv"

# SEBI peak-margin framework: phased in from December 2020, reaching its
# final 100%-upfront-margin phase on 1 September 2021. This is the correct
# cutoff for the pre/post split below -- an earlier draft of this analysis
# incorrectly used 1 March 2021, which was only the start of an intermediate
# (50%) phase, not the completion date.
PEAK_MARGIN_COMPLETION_DATE = "2021-09-01"

RESULTS = {}


def run_index(csv_path, label, date_col, time_col, close_col, date_fmt, bw_plot_path):
    print("=" * 70, f"\n{label}\n", "=" * 70)
    panel = m.load_real_data(csv_path, date_col=date_col, time_col=time_col,
                              close_col=close_col, date_fmt=date_fmt)
    panel["abs_ret"] = panel["ret"].abs()
    print(f"Rows: {len(panel)} | Days: {panel['Date'].nunique()} | "
          f"Range: {panel['Date'].min()} to {panel['Date'].max()}")

    h_star, cv_scores = r.cv_select_bandwidth(panel, "abs_ret", t0=345)
    print("CV-optimal bandwidth:", h_star)

    lin = r.local_poly_rdd(panel, "abs_ret", t0=345, bandwidth=h_star, order=1)
    quad = r.local_poly_rdd(panel, "abs_ret", t0=345, bandwidth=h_star, order=2)
    print("Linear @ h*:   ", lin)
    print("Higher-order @ h*:", quad)

    sens = r.bandwidth_sensitivity(panel, "abs_ret", t0=345)
    r.plot_bandwidth_sensitivity(sens, label, bw_plot_path)
    print(sens)

    placebo = {}
    for c in (300, 315, 330, 345):
        placebo[c] = r.local_poly_rdd(panel, "abs_ret", t0=c, bandwidth=h_star, order=1)
    pvals = [placebo[c]["p_value"] for c in (300, 315, 330, 345)]
    bonf = r.bonferroni(pvals)
    bh = r.benjamini_hochberg(pvals)
    print("Raw p (14:15, 14:30, 14:45, 15:00-true):", pvals)
    print("Bonferroni:", bonf.tolist())
    print("BH-FDR:    ", bh.tolist())

    return panel, {
        "h_star": h_star, "cv_scores": {str(k): v for k, v in cv_scores.items()},
        "linear": lin, "quadratic": quad,
        "bandwidth_sensitivity": sens.to_dict(orient="records"),
        "placebo_at_hstar": {str(c): v for c, v in placebo.items()},
        "multiple_testing": {"cutoffs": [300, 315, 330, 345], "raw_p": pvals,
                              "bonferroni": bonf.tolist(), "bh_fdr": bh.tolist()},
        "n_days": int(panel["Date"].nunique()),
        "date_min": str(panel["Date"].min()), "date_max": str(panel["Date"].max()),
    }


def main():
    nifty_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NIFTY
    banknifty_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BANKNIFTY

    _, RESULTS["nifty"] = run_index(
        nifty_path, "NIFTY 50", "Date", "Time", "close", "%Y-%m-%d",
        "bw_sensitivity_nifty.png")

    bn, RESULTS["banknifty_full"] = run_index(
        banknifty_path, "BANK NIFTY (full sample)", "Date", "Time", "Close", "%d-%m-%Y",
        "bw_sensitivity_banknifty.png")

    # --- Sub-period split around peak-margin completion ---
    print("\n\n" + "=" * 70, f"\nBANK NIFTY -- sub-period split (cutoff: {PEAK_MARGIN_COMPLETION_DATE})\n", "=" * 70)
    bn_pre = bn[bn["Date"] < PEAK_MARGIN_COMPLETION_DATE].copy()
    bn_post = bn[bn["Date"] >= PEAK_MARGIN_COMPLETION_DATE].copy()
    print(f"Pre-period:  {bn_pre['Date'].min()} to {bn_pre['Date'].max()}  ({bn_pre['Date'].nunique()} days)")
    print(f"Post-period: {bn_post['Date'].min()} to {bn_post['Date'].max()}  ({bn_post['Date'].nunique()} days)")

    sub_results = {}
    for label, sub in [("pre", bn_pre), ("post", bn_post)]:
        h_star_s, _ = r.cv_select_bandwidth(sub, "abs_ret", t0=345)
        lin_s = r.local_poly_rdd(sub, "abs_ret", t0=345, bandwidth=h_star_s, order=1)
        quad_s = r.local_poly_rdd(sub, "abs_ret", t0=345, bandwidth=h_star_s, order=2)
        print(f"[{label}] h*={h_star_s}  linear={lin_s}")
        print(f"[{label}] higher-order={quad_s}")
        sub_results[label] = {
            "h_star": h_star_s, "linear": lin_s, "quadratic": quad_s,
            "n_days": int(sub["Date"].nunique()),
            "date_min": str(sub["Date"].min()), "date_max": str(sub["Date"].max()),
        }
    RESULTS["banknifty_subperiods"] = sub_results

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    with open("robust_rdd_results.json", "w") as f:
        json.dump(clean(RESULTS), f, indent=2, default=str)
    print("\n\nSaved -> robust_rdd_results.json")


if __name__ == "__main__":
    main()
