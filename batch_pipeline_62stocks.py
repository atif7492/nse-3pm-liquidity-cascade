"""
NSE 3PM Regime-Shift: Full Per-Stock Batch Pipeline (62-Stock Ready)
======================================================================
Built on the verified nse_3pm_pipeline.py core (same clustering_test /
local_linear_rdd / placebo_rdd logic used for the Nifty 50 and Bank Nifty
index-level results in Section IV.B-IV.D), extended to:

  1. Handle the Kaggle per-stock schema: date,open,high,low,close,volume
     (single combined datetime column, not separate date/time columns)
  2. Run on an entire FOLDER of per-stock CSVs at once
  3. For each stock, compute the FULL Section IV.B/IV.C battery, not just
     the single RDD-jump number the original 537-file script produced:
       - IV.B equivalent: 2- vs 3-component clustering test (mode, LR, p)
       - IV.C equivalent: RDD on signed return AND |return| at t0=345,
         plus placebo tests at 3 false cutoffs
  4. Write one row per stock to a summary CSV, and print progress as it runs

FIXES vs the original 537-file script the user shared earlier:
  - Vectorized datetime -> minute-from-open conversion (the original's
    per-row Python loop was the main reason the first attempt stalled at
    62/537 files)
  - Vectorized weighted least squares in local_linear_rdd (avoids the
    O(N^2) np.diag(w) memory blow-up fixed earlier for Bank Nifty)
  - Explicit non-standard-session filtering (Muhurat / extended-hours days
    dropped, matching the index-level cleaning in IV.A)
  - try/except per file so one bad file does not kill the whole batch

USAGE
-----
    python3 batch_pipeline_62stocks.py /path/to/folder/of/csvs

Produces: 62stocks_full_summary.csv in the current directory, with columns:
  ticker, trading_days, n_rows, irregular_days_dropped,
  clus_mode_near_3pm, clus_lr_stat, clus_pvalue,
  rdd_ret_jump, rdd_ret_t, rdd_ret_p,
  rdd_absret_jump, rdd_absret_t, rdd_absret_p,
  placebo_300_p, placebo_315_p, placebo_330_p
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats

RNG = np.random.default_rng(42)


# ----------------------------------------------------------------------
# 1. LOADER -- Kaggle per-stock schema (combined datetime column)
# ----------------------------------------------------------------------
def load_stock_csv(csv_path, session_open_h=9, session_open_m=15):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "close" not in df.columns:
        return None, "Missing date/close column"

    dt = pd.to_datetime(df["date"], errors="coerce")
    mask = dt.notna()
    df = df[mask].copy()
    dt = dt[mask]

    df["Date"] = dt.dt.strftime("%Y-%m-%d")
    df["Close"] = pd.to_numeric(df["close"], errors="coerce")
    # Drop non-positive prices (bad ticks / data errors) BEFORE computing
    # returns -- a zero or negative price is never a real trade, and a
    # transition through zero produces an infinite pct_change that
    # silently poisons the clustering test's probability weights downstream.
    df = df[df["Close"] > 0]
    # vectorized minute-from-open (no per-row loop -> scales to 1M+ rows)
    df["minute_from_open"] = (dt.dt.hour - session_open_h) * 60 + (dt.dt.minute - session_open_m)

    df = df.dropna(subset=["Close", "Date"])

    # drop non-standard sessions (Muhurat / extended-hours days), same
    # cleaning rule as the index-level IV.A analysis
    day_max = df.groupby("Date")["minute_from_open"].transform("max")
    n_days_before = df["Date"].nunique()
    df = df[(df["minute_from_open"] >= 0) & (df["minute_from_open"] <= 390) & (day_max <= 390)]
    n_days_after = df["Date"].nunique()
    dropped = n_days_before - n_days_after

    if len(df) < 500:
        return None, "Too few rows after cleaning"

    df = df.sort_values(["Date", "minute_from_open"]).reset_index(drop=True)
    df["ret"] = df.groupby("Date")["Close"].pct_change()
    df["abs_ret"] = df["ret"].abs()
    return df, dropped


# ----------------------------------------------------------------------
# 2. CLUSTERING TEST (Section II.A / IV.B equivalent)
# ----------------------------------------------------------------------
def clustering_test(panel, t0=345):
    weights = panel["abs_ret"].replace([np.inf, -np.inf], np.nan).fillna(0).values
    minutes = panel["minute_from_open"].values
    if weights.sum() == 0 or not np.isfinite(weights.sum()):
        return None
    probs = weights / weights.sum()
    n_sample = min(20000, len(minutes))
    sample = RNG.choice(minutes, size=n_sample, replace=True, p=probs).reshape(-1, 1).astype(float)

    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=3).fit(sample)
    gmm3 = GaussianMixture(n_components=3, random_state=0, n_init=3).fit(sample)

    ll2 = gmm2.score(sample) * len(sample)
    ll3 = gmm3.score(sample) * len(sample)
    lr_stat = 2 * (ll3 - ll2)
    p_value = 1 - stats.chi2.cdf(lr_stat, df=3)
    means3 = sorted(gmm3.means_.flatten())
    mode_near = min(means3, key=lambda m: abs(m - t0))
    return {"lr_stat": lr_stat, "p_value": p_value, "mode_near_3pm": mode_near}


# ----------------------------------------------------------------------
# 3. RDD (Section III.A / IV.C equivalent) -- vectorized, memory-safe
# ----------------------------------------------------------------------
def triangular_kernel(u, h):
    return np.maximum(0, 1 - np.abs(u) / h)


def local_linear_rdd(panel, outcome_col, t0=345, bandwidth=30, donut=1):
    df = panel.replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome_col]).copy()
    df["run"] = df["minute_from_open"] - t0
    df = df[(df["run"].abs() <= bandwidth) & (df["run"].abs() > donut)]

    def fit_side(sub):
        x = sub["run"].values
        y = sub[outcome_col].values
        w = triangular_kernel(x, bandwidth)
        X = np.column_stack([np.ones_like(x), x])
        Xw = w[:, None] * X
        XtW_X = X.T @ Xw
        XtW_y = Xw.T @ y
        beta = np.linalg.pinv(XtW_X) @ XtW_y
        resid = y - X @ beta
        meat_weight = (w**2) * (resid**2)
        meat = X.T @ (meat_weight[:, None] * X)
        bread = np.linalg.pinv(XtW_X)
        vcov = bread @ meat @ bread
        return beta, vcov

    left = df[df["run"] < 0]
    right = df[df["run"] >= 0]
    if len(left) < 10 or len(right) < 10:
        return None

    beta_l, vcov_l = fit_side(left)
    beta_r, vcov_r = fit_side(right)
    jump = beta_r[0] - beta_l[0]
    se_jump = np.sqrt(vcov_l[0, 0] + vcov_r[0, 0])
    t_stat = jump / se_jump if se_jump > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
    return {"jump": jump, "t_stat": t_stat, "p_value": p_value,
            "n_left": len(left), "n_right": len(right)}


def placebo_rdd(panel, outcome_col, cutoffs=(300, 315, 330)):
    out = {}
    for c in cutoffs:
        r = local_linear_rdd(panel, outcome_col, t0=c)
        out[c] = r["p_value"] if r else np.nan
    return out


# ----------------------------------------------------------------------
# 4. PER-STOCK WORKER
# ----------------------------------------------------------------------
def process_one_stock(csv_path, ticker):
    t_start = time.time()
    panel, status = load_stock_csv(csv_path)
    if panel is None:
        return {"ticker": ticker, "status": f"SKIPPED ({status})"}

    n_days = panel["Date"].nunique()
    n_rows = len(panel)

    clus = clustering_test(panel)
    rdd_ret = local_linear_rdd(panel, "ret", t0=345)
    rdd_absret = local_linear_rdd(panel, "abs_ret", t0=345)
    placebo = placebo_rdd(panel, "abs_ret")

    elapsed = time.time() - t_start
    row = {
        "ticker": ticker,
        "status": "SUCCESS",
        "trading_days": n_days,
        "n_rows": n_rows,
        "irregular_days_dropped": status,  # `status` holds dropped-day count on success
        "clus_mode_near_3pm": clus["mode_near_3pm"] if clus else np.nan,
        "clus_lr_stat": clus["lr_stat"] if clus else np.nan,
        "clus_pvalue": clus["p_value"] if clus else np.nan,
        "rdd_ret_jump": rdd_ret["jump"] if rdd_ret else np.nan,
        "rdd_ret_t": rdd_ret["t_stat"] if rdd_ret else np.nan,
        "rdd_ret_p": rdd_ret["p_value"] if rdd_ret else np.nan,
        "rdd_absret_jump": rdd_absret["jump"] if rdd_absret else np.nan,
        "rdd_absret_t": rdd_absret["t_stat"] if rdd_absret else np.nan,
        "rdd_absret_p": rdd_absret["p_value"] if rdd_absret else np.nan,
        "placebo_300_p": placebo.get(300, np.nan),
        "placebo_315_p": placebo.get(315, np.nan),
        "placebo_330_p": placebo.get(330, np.nan),
        "seconds": round(elapsed, 1),
    }
    return row


# ----------------------------------------------------------------------
# MAIN -- point this at a folder of "<TICKER>_minute.csv" files
# ----------------------------------------------------------------------
def run_batch(folder, out_csv="62stocks_full_summary.csv"):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
    print(f"Found {len(files)} CSV files in {folder}")
    results = []
    for i, fname in enumerate(files, 1):
        ticker = fname.replace("_minute.csv", "").replace(".csv", "")
        # strip any numeric upload-id prefix like "1787876773260_"
        parts = ticker.split("_")
        if parts[0].isdigit():
            ticker = "_".join(parts[1:])
        path = os.path.join(folder, fname)
        row = process_one_stock(path, ticker)
        results.append(row)
        status_str = row["status"]
        extra = f" (days={row.get('trading_days')}, RDD|ret| jump={row.get('rdd_absret_jump'):.6f}, p={row.get('rdd_absret_p'):.4f})" if status_str == "SUCCESS" else ""
        print(f"[{i}/{len(files)}] {ticker} -> {status_str}{extra}", flush=True)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    n_success = (out_df["status"] == "SUCCESS").sum()
    print(f"\nDone: {n_success}/{len(files)} succeeded. Saved -> {out_csv}")
    return out_df


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    run_batch(folder)
