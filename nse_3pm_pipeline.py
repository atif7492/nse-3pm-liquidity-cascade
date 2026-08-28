"""
NSE 3:00 PM Regime-Shift Analysis Pipeline
============================================
Implements, from first principles (numpy/scipy/scikit-learn only —
statsmodels/linearmodels/rdrobust are NOT available in this offline
sandbox, so RDD and GMM are hand-rolled rather than imported):

  1. KDE / Gaussian-Mixture clustering test for execution-time density
     (Section II.A of the paper: 2-component vs 3-component mixture,
     likelihood-ratio test for a mode at t=345 min from open = 15:00 IST)

  2. Sharp Regression Discontinuity Design at t0 = 15:00 IST, local-linear
     estimator with a triangular kernel, following the Calonico-Cattaneo-
     Titiunik (2014) logic (manual implementation; no rdrobust package)

  3. A simplified structural GMM estimator for the panel specification in
     Section III.B (two-step GMM via scipy.optimize minimizing the
     quadratic form of the moment conditions -- NOT the full
     Arellano-Bond/Blundell-Bond system-GMM from linearmodels, which is
     unavailable offline; this is a didactic approximation)

IMPORTANT — DATA HONESTY
-------------------------
This script is demonstrated below on SYNTHETIC data (a GBM random walk
with an injected artificial 15:00 liquidity shock) purely to validate
that the pipeline runs end-to-end and the statistics behave as expected
under a KNOWN ground truth. This is a software-correctness check, NOT
an empirical finding about NSE markets. Every synthetic output is
labeled accordingly.

To run this on REAL data: replace `load_data()` with a loader that reads
the actual minute-bar CSV (Date, Time, Open, High, Low, Close columns --
matching e.g. the sandeepkapri/Nifty50-Minute-Data schema) and re-run.
No other code changes should be necessary.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 0. SYNTHETIC DATA GENERATOR (validation only -- NOT real NSE data)
# ----------------------------------------------------------------------

def generate_synthetic_day(date, inject_3pm_shock=True, base_price=22000.0):
    """One synthetic trading day, 09:15-15:30 IST, 1-minute bars.
    375 minutes. GBM base dynamics + optional injected volatility/impact
    shock in the 15:00-15:15 window, to give the detection pipeline a
    KNOWN true signal to recover (software validation, not empirical claim).
    """
    n_min = 375
    minutes = np.arange(n_min)
    sigma_base = 0.0006  # per-minute vol, illustrative
    shocks = RNG.normal(0, sigma_base, n_min)

    if inject_3pm_shock:
        # 15:00 IST = minute 345 from open (09:15 + 345 = 15:00)
        window = (minutes >= 345) & (minutes < 360)
        shocks[window] += RNG.normal(0.0009, 0.0011, window.sum())  # directional + vol bump
        # thin out liquidity -> wider effective moves (impact) in same window
        shocks[window] *= 2.2

    log_ret = shocks
    price = base_price * np.exp(np.cumsum(log_ret))
    close = price
    open_ = np.r_[base_price, price[:-1]]
    high = np.maximum(open_, close) * (1 + RNG.uniform(0, 0.0004, n_min))
    low = np.minimum(open_, close) * (1 - RNG.uniform(0, 0.0004, n_min))

    times = pd.date_range("09:15", periods=n_min, freq="min").time
    df = pd.DataFrame({
        "Date": date, "Time": [t.strftime("%H:%M:%S") for t in times],
        "Open": open_, "High": high, "Low": low, "Close": close,
        "minute_from_open": minutes,
    })
    return df


def load_data(n_days=250, inject_shock=True):
    """Builds a synthetic multi-day panel. USE load_real_data() INSTEAD
    once you have an actual minute-bar CSV -- everything downstream
    (clustering_test, local_linear_rdd, build_panel_for_gmm) is agnostic
    to the data source and takes the same `panel` DataFrame schema."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    frames = [generate_synthetic_day(d.strftime("%Y-%m-%d"), inject_shock)
              for d in dates]
    panel = pd.concat(frames, ignore_index=True)
    panel["ret"] = panel.groupby("Date")["Close"].pct_change()
    return panel


def load_real_data(csv_path, date_col="Date", time_col="Time",
                    close_col="Close", date_fmt="%d-%m-%y",
                    session_open="09:15:00"):
    """Loader for a real minute-bar CSV matching the common NSE schema
    (Date, Time, Open, High, Low, Close -- e.g. the sandeepkapri/
    Nifty50-Minute-Data or comparable Kaggle/vendor exports).

    Usage:
        panel = load_real_data("/path/to/nifty50_candlestick_data.csv")
        ct   = clustering_test(panel)
        rdd  = local_linear_rdd(panel, "abs_ret", t0=345)
        ...

    NOTE: this function does no imputation or survivorship correction --
    inspect for holidays, half-days (e.g. Muhurat trading), and exchange
    circuit-halt gaps before treating results as final. Contracts with
    non-standard closes (pre-2025 15:30 close vs post-2025 15:15/15:30
    split) should be segmented by regime and analyzed separately, since
    pooling both regimes would bias the RDD toward whichever regime has
    more trading days in the sample.
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={date_col: "Date", time_col: "Time", close_col: "Close"})
    dt = pd.to_datetime(df["Date"], format=date_fmt, errors="coerce")
    tm = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce").dt.time
    df["Date"] = dt.dt.strftime("%Y-%m-%d")
    open_dt = pd.to_datetime(session_open, format="%H:%M:%S")
    minute_from_open = []
    for t in tm:
        if pd.isna(t):
            minute_from_open.append(np.nan)
            continue
        t_dt = pd.to_datetime(t.strftime("%H:%M:%S"), format="%H:%M:%S")
        minute_from_open.append(int((t_dt - open_dt).total_seconds() // 60))
    df["minute_from_open"] = minute_from_open
    df = df.dropna(subset=["minute_from_open"])
    df["minute_from_open"] = df["minute_from_open"].astype(int)

    # Drop non-standard sessions: special evening "Muhurat" trading (Diwali)
    # and any other day whose minute-from-open range falls well outside the
    # regular 09:15-15:30 continuous session (0-375 min). These are thin,
    # symbolic, or otherwise structurally different sessions that are not
    # comparable to the regular-day RMS dynamics under study; pooling them
    # in would contaminate the very discontinuity we are trying to isolate.
    day_max = df.groupby("Date")["minute_from_open"].transform("max")
    n_before = df["Date"].nunique()
    df = df[day_max <= 390].copy()
    n_after = df["Date"].nunique()
    n_dropped = n_before - n_after
    if n_dropped > 0:
        print(f"[load_real_data] Dropped {n_dropped} non-standard session day(s) "
              f"(e.g. Muhurat trading) out of {n_before} total days.")

    df = df.sort_values(["Date", "minute_from_open"]).reset_index(drop=True)
    df["ret"] = df.groupby("Date")["Close"].pct_change()
    df["abs_ret"] = df["ret"].abs()
    return df


# ----------------------------------------------------------------------
# 1. KDE / GAUSSIAN-MIXTURE EXECUTION-DENSITY CLUSTERING TEST
# ----------------------------------------------------------------------

def clustering_test(panel):
    """Fit 2-component vs 3-component Gaussian mixtures to the intraday
    time-of-trade distribution (proxied here by |return|-weighted minute
    timestamps, i.e. activity-weighted resampling), run a likelihood-
    ratio test for a third mode at t=345 (15:00 IST)."""
    weights = panel["ret"].abs().fillna(0).values
    minutes = panel["minute_from_open"].values
    # activity-weighted resample: minutes with higher |return| sampled more
    probs = weights / weights.sum()
    sample = RNG.choice(minutes, size=50000, replace=True, p=probs).reshape(-1, 1).astype(float)

    gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(sample)
    gmm3 = GaussianMixture(n_components=3, random_state=0, n_init=5).fit(sample)

    ll2 = gmm2.score(sample) * len(sample)
    ll3 = gmm3.score(sample) * len(sample)
    lr_stat = 2 * (ll3 - ll2)
    # 3 extra free params (mean, var, weight of the 3rd component, net of
    # the simplex constraint) -> df=3 chi-square reference, per Sec. II.A
    p_value = 1 - stats.chi2.cdf(lr_stat, df=3)

    means3 = sorted(gmm3.means_.flatten())
    return {
        "gmm2_means": sorted(gmm2.means_.flatten()),
        "gmm3_means": means3,
        "loglik_2comp": ll2,
        "loglik_3comp": ll3,
        "LR_statistic": lr_stat,
        "p_value": p_value,
        "mode_near_3pm": min(means3, key=lambda m: abs(m - 345)),
    }


# ----------------------------------------------------------------------
# 2. REGRESSION DISCONTINUITY DESIGN AT t0 = 15:00 IST (minute 345)
# ----------------------------------------------------------------------

def triangular_kernel(u, h):
    w = np.maximum(0, 1 - np.abs(u) / h)
    return w


def local_linear_rdd(panel, outcome_col, t0=345, bandwidth=30, donut=1):
    """Sharp RDD: local-linear regression on each side of t0, triangular
    kernel weights, donut exclusion of `donut` minutes either side of t0
    to avoid discreteness/microstructure noise right at the cutoff.
    Returns the estimated jump (beta), its SE (via heteroskedasticity-
    robust sandwich formula), and a t-stat/p-value.
    """
    df = panel.dropna(subset=[outcome_col]).copy()
    df["run"] = df["minute_from_open"] - t0
    df = df[(df["run"].abs() <= bandwidth) & (df["run"].abs() > donut)]

    def fit_side(sub):
        x = sub["run"].values
        y = sub[outcome_col].values
        w = triangular_kernel(x, bandwidth)
        X = np.column_stack([np.ones_like(x), x])
        # Vectorized weighted least squares -- avoid materializing an NxN
        # diagonal weight matrix (that blows up memory for large N; a
        # dense np.diag(w) on ~80k rows tries to allocate ~48 GB and
        # crashes). X.T @ (w[:, None] * X) is algebraically identical to
        # X.T @ diag(w) @ X but costs O(N) memory instead of O(N^2).
        Xw = w[:, None] * X
        XtW_X = X.T @ Xw
        XtW_y = Xw.T @ y
        beta = np.linalg.pinv(XtW_X) @ XtW_y
        resid = y - X @ beta
        # robust (HC1-style) sandwich variance, same vectorization trick
        meat_weight = (w**2) * (resid**2)
        meat = X.T @ (meat_weight[:, None] * X)
        bread = np.linalg.pinv(XtW_X)
        vcov = bread @ meat @ bread
        return beta, vcov

    left = df[df["run"] < 0]
    right = df[df["run"] >= 0]
    beta_l, vcov_l = fit_side(left)
    beta_r, vcov_r = fit_side(right)

    intercept_l, intercept_r = beta_l[0], beta_r[0]
    jump = intercept_r - intercept_l
    se_jump = np.sqrt(vcov_l[0, 0] + vcov_r[0, 0])
    t_stat = jump / se_jump if se_jump > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        "outcome": outcome_col, "bandwidth": bandwidth, "donut": donut,
        "n_left": len(left), "n_right": len(right),
        "intercept_left": intercept_l, "intercept_right": intercept_r,
        "jump": jump, "se": se_jump, "t_stat": t_stat, "p_value": p_value,
    }


def placebo_rdd(panel, outcome_col, false_cutoffs=(315, 330, 300)):
    """Placebo test: re-run the RDD at false cutoffs (e.g. 14:30, 14:45,
    14:00) where no RMS deadline exists. A true SMLC effect should show
    up ONLY at t0=345, not at these placebos."""
    return {c: local_linear_rdd(panel, outcome_col, t0=c) for c in false_cutoffs}


# ----------------------------------------------------------------------
# 3. SIMPLIFIED STRUCTURAL GMM  (Section III.B)
# ----------------------------------------------------------------------

def build_panel_for_gmm(panel, t0=345, window=15):
    """Collapse to one row per trading day x pre/post window, matching
    the D_3PM specification in the paper. abs_ret is the dependent
    variable (proxy for |Delta P|); D3PM is the treatment dummy;
    lag_absret is the autoregressive control."""
    rows = []
    for date, g in panel.groupby("Date"):
        g = g.sort_values("minute_from_open")
        pre = g[(g["minute_from_open"] >= t0 - window) & (g["minute_from_open"] < t0)]
        post = g[(g["minute_from_open"] >= t0) & (g["minute_from_open"] < t0 + window)]
        if len(pre) < 2 or len(post) < 2:
            continue
        rows.append({"Date": date, "D3PM": 0, "abs_ret": pre["ret"].abs().mean(),
                      "lag_absret": g[g["minute_from_open"] < t0 - window]["ret"].abs().mean()})
        rows.append({"Date": date, "D3PM": 1, "abs_ret": post["ret"].abs().mean(),
                      "lag_absret": pre["ret"].abs().mean()})
    return pd.DataFrame(rows).dropna()


def simplified_gmm(panel_gmm):
    """Two-step GMM: moment conditions are that regressors are
    orthogonal to residuals of  abs_ret = b0 + b1*D3PM + b2*lag_absret + e.
    This is a simplified, homoskedastic-instrument analogue of the
    Arellano-Bond estimator in the paper (full dynamic-panel GMM with
    lagged-level/difference instruments requires linearmodels, which is
    unavailable offline). Estimated via scipy.optimize minimizing the
    GMM quadratic form; standard errors via the usual GMM sandwich.
    """
    y = panel_gmm["abs_ret"].values
    X = np.column_stack([np.ones(len(panel_gmm)), panel_gmm["D3PM"].values,
                          panel_gmm["lag_absret"].values])
    Z = X.copy()  # exactly identified case -> GMM reduces to OLS-IV here

    def moments(beta):
        resid = y - X @ beta
        return Z * resid[:, None]

    def objective(beta):
        g = moments(beta).mean(axis=0)
        return g @ g

    beta0 = np.zeros(X.shape[1])
    res = optimize.minimize(objective, beta0, method="BFGS")
    beta_hat = res.x

    resid = y - X @ beta_hat
    n = len(y)
    S = (Z * resid[:, None]).T @ (Z * resid[:, None]) / n
    G = -(Z.T @ X) / n
    vcov = np.linalg.pinv(G.T @ np.linalg.pinv(S) @ G) / n
    se = np.sqrt(np.diag(vcov))
    t_stats = beta_hat / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))

    names = ["const", "D3PM", "lag_absret"]
    return pd.DataFrame({"coef": beta_hat, "se": se, "t": t_stats, "p": p_values}, index=names)


# ----------------------------------------------------------------------
# MAIN — run the full pipeline on SYNTHETIC validation data
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC VALIDATION RUN -- NOT REAL NSE DATA")
    print("=" * 70)

    panel = load_data(n_days=250, inject_shock=True)

    # 1. KDE clustering test
    ct = clustering_test(panel)
    print("\n[1] Execution-density clustering test (GMM, 2- vs 3-component)")
    for k, v in ct.items():
        print(f"    {k}: {v}")

    # 2. RDD on |returns| (volatility) at 15:00
    rdd_vol = local_linear_rdd(panel, "ret", t0=345, bandwidth=30, donut=1)
    print("\n[2] RDD on returns at t0=15:00 (minute 345)")
    for k, v in rdd_vol.items():
        print(f"    {k}: {v}")

    panel["abs_ret"] = panel["ret"].abs()
    rdd_absvol = local_linear_rdd(panel, "abs_ret", t0=345, bandwidth=30, donut=1)
    print("\n[2b] RDD on |returns| (volatility proxy) at t0=15:00")
    for k, v in rdd_absvol.items():
        print(f"    {k}: {v}")

    # placebo cutoffs
    placebos = placebo_rdd(panel, "abs_ret")
    print("\n[2c] Placebo RDDs at false cutoffs (should show weak/no jump)")
    for c, r in placebos.items():
        print(f"    t0={c}: jump={r['jump']:.6f}, p={r['p_value']:.4f}")

    # 3. Simplified structural GMM
    panel_gmm = build_panel_for_gmm(panel)
    gmm_res = simplified_gmm(panel_gmm)
    print("\n[3] Simplified structural GMM (D_3PM specification)")
    print(gmm_res)

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    avg_by_minute = panel.groupby("minute_from_open")["abs_ret"].mean()
    axes[0].plot(avg_by_minute.index, avg_by_minute.values, lw=1)
    axes[0].axvline(345, color="red", ls="--", label="15:00 IST (t0)")
    axes[0].axvline(360, color="orange", ls=":", label="15:15 IST")
    axes[0].set_title("Mean |return| by minute-from-open\n(SYNTHETIC validation data)")
    axes[0].set_xlabel("Minutes from open (09:15 IST)")
    axes[0].set_ylabel("Mean |1-min return|")
    axes[0].legend(fontsize=8)

    sample_for_hist = RNG.choice(panel["minute_from_open"].values, size=20000,
                                  replace=True,
                                  p=(panel["abs_ret"].fillna(0) / panel["abs_ret"].fillna(0).sum()).values)
    axes[1].hist(sample_for_hist, bins=60, alpha=0.7)
    axes[1].axvline(345, color="red", ls="--", label="15:00 IST (t0)")
    axes[1].set_title("Activity-weighted execution-time density\n(SYNTHETIC validation data)")
    axes[1].set_xlabel("Minutes from open")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("/home/claude/synthetic_validation_plots.png", dpi=150)
    print("\nSaved plot -> /home/claude/synthetic_validation_plots.png")
