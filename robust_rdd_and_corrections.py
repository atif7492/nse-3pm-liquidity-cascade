"""
Robust RDD Extension (CCT-inspired) + Multiple-Testing Correction
====================================================================
Implements, on top of the existing nse_3pm_pipeline.py local_linear_rdd():

  1. A data-driven, MSE-optimal bandwidth selector via leave-one-side-out
     cross-validation (not hand-picked at a fixed 30 minutes).
  2. A local-QUADRATIC bias-check estimator at the selected bandwidth,
     following the spirit of Calonico-Cattaneo-Titiunik (2014): a local-
     linear estimator is biased under curvature near the cutoff, and a
     higher-order local polynomial fit at the same bandwidth reveals and
     corrects most of that bias.
  3. A bandwidth-sensitivity sweep (point estimate + 95% CI across a grid
     of bandwidths), the standard robustness display for RDD estimates.
  4. Bonferroni and Benjamini-Hochberg (FDR) multiple-testing correction
     across the 4 RDD tests per index (true cutoff + 3 placebos) and
     across the 62-stock cross-section's 62 independent tests.

HONESTY NOTE: this is a from-scratch, CCT-INSPIRED implementation, not a
port of the `rdrobust` R/Stata/Python package. It captures the two central
ideas -- (a) do not hand-pick the bandwidth, select it in a principled,
data-driven way, and (b) do not rely solely on a linear local fit, which
is biased under curvature -- but does not replicate rdrobust's exact
plug-in bandwidth formula or its precise robust-bias-corrected variance
estimator. This is stated explicitly in the paper text as well.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nse_3pm_pipeline as m


# ----------------------------------------------------------------------
# 1. Local polynomial fit of arbitrary order (order=1 linear, order=2 quad)
# ----------------------------------------------------------------------
def _fit_side_poly(sub, outcome_col, bandwidth, order=1):
    x = sub["run"].values
    y = sub[outcome_col].values
    w = m.triangular_kernel(x, bandwidth)
    cols = [np.ones_like(x)] + [x**k for k in range(1, order + 1)]
    X = np.column_stack(cols)
    Xw = w[:, None] * X
    XtW_X = X.T @ Xw
    XtW_y = Xw.T @ y
    beta = np.linalg.pinv(XtW_X) @ XtW_y
    resid = y - X @ beta
    meat_weight = (w**2) * (resid**2)
    meat = X.T @ (meat_weight[:, None] * X)
    bread = np.linalg.pinv(XtW_X)
    vcov = bread @ meat @ bread
    return beta, vcov  # beta[0] = intercept (the level at the cutoff)


def local_poly_rdd(panel, outcome_col, t0=345, bandwidth=30, donut=1, order=1):
    """Generalizes local_linear_rdd to arbitrary local-polynomial order."""
    df = panel.dropna(subset=[outcome_col]).copy()
    df["run"] = df["minute_from_open"] - t0
    df = df[(df["run"].abs() <= bandwidth) & (df["run"].abs() > donut)]
    left = df[df["run"] < 0]
    right = df[df["run"] >= 0]
    if len(left) < (order + 5) or len(right) < (order + 5):
        return None
    beta_l, vcov_l = _fit_side_poly(left, outcome_col, bandwidth, order)
    beta_r, vcov_r = _fit_side_poly(right, outcome_col, bandwidth, order)
    jump = beta_r[0] - beta_l[0]
    se = np.sqrt(vcov_l[0, 0] + vcov_r[0, 0])
    t_stat = jump / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
    return {"jump": jump, "se": se, "t_stat": t_stat, "p_value": p_value,
            "n_left": len(left), "n_right": len(right), "bandwidth": bandwidth, "order": order}


# ----------------------------------------------------------------------
# 2. MSE-optimal bandwidth via leave-one-side-out cross-validation
# ----------------------------------------------------------------------
def cv_select_bandwidth(panel, outcome_col, t0=345, donut=1,
                         candidate_bandwidths=None, order=1):
    """For each candidate bandwidth, fits the local-polynomial RDD on each
    side and computes leave-one-out cross-validated prediction MSE
    (standard nonparametric bandwidth-selection criterion; Imbens-
    Kalyanaraman/CCT-style optimal bandwidths follow the same MSE-
    minimization logic via an asymptotic plug-in formula instead of
    literal CV -- we use CV here because it requires no additional
    density/derivative estimation machinery and is directly computable
    from the same local-polynomial fits already implemented)."""
    if candidate_bandwidths is None:
        candidate_bandwidths = list(range(10, 65, 5))

    df_all = panel.dropna(subset=[outcome_col]).copy()
    df_all["run"] = df_all["minute_from_open"] - t0

    cv_scores = {}
    for h in candidate_bandwidths:
        sub = df_all[(df_all["run"].abs() <= h) & (df_all["run"].abs() > donut)]
        side_errs = []
        for side_df in (sub[sub["run"] < 0], sub[sub["run"] >= 0]):
            if len(side_df) < (order + 10):
                continue
            x = side_df["run"].values
            y = side_df[outcome_col].values
            w_all = m.triangular_kernel(x, h)
            # leave-one-out via closed-form hat-matrix diagonal (fast,
            # avoids an O(N) refit loop): loo_resid_i = resid_i / (1 - h_ii)
            cols = [np.ones_like(x)] + [x**k for k in range(1, order + 1)]
            X = np.column_stack(cols)
            Xw = w_all[:, None] * X
            XtW_X_inv = np.linalg.pinv(X.T @ Xw)
            beta = XtW_X_inv @ (Xw.T @ y)
            resid = y - X @ beta
            H_diag = np.einsum("ij,jk,ik->i", X, XtW_X_inv, Xw)
            H_diag = np.clip(H_diag, 0, 0.99)  # numerical safety
            loo_resid = resid / (1 - H_diag)
            side_errs.append(np.mean(loo_resid**2))
        if side_errs:
            cv_scores[h] = np.mean(side_errs)

    if not cv_scores:
        return candidate_bandwidths[len(candidate_bandwidths) // 2], cv_scores
    h_star = min(cv_scores, key=cv_scores.get)
    return h_star, cv_scores


def bandwidth_sensitivity(panel, outcome_col, t0=345, donut=1,
                           candidate_bandwidths=None, order=1):
    if candidate_bandwidths is None:
        candidate_bandwidths = list(range(10, 65, 5))
    rows = []
    for h in candidate_bandwidths:
        r = local_poly_rdd(panel, outcome_col, t0=t0, bandwidth=h, donut=donut, order=order)
        if r:
            rows.append({"bandwidth": h, **r})
    return pd.DataFrame(rows)


def plot_bandwidth_sensitivity(sens_df, label, out_path, t0_label="15:00"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ci_lo = sens_df["jump"] - 1.96 * sens_df["se"]
    ci_hi = sens_df["jump"] + 1.96 * sens_df["se"]
    ax.plot(sens_df["bandwidth"], sens_df["jump"], "o-", color="#1f3864", label="RDD jump")
    ax.fill_between(sens_df["bandwidth"], ci_lo, ci_hi, color="#1f3864", alpha=0.15, label="95% CI")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Bandwidth (minutes)")
    ax.set_ylabel("RDD |return| jump estimate")
    ax.set_title(f"Bandwidth sensitivity -- {label} (cutoff = {t0_label})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Multiple-testing correction
# ----------------------------------------------------------------------
def bonferroni(pvals):
    pvals = np.asarray(pvals, dtype=float)
    return np.minimum(pvals * len(pvals), 1.0)


def benjamini_hochberg(pvals):
    """Standard BH step-up FDR-adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity (step-up)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


if __name__ == "__main__":
    print("robust_rdd_and_corrections.py loaded OK -- see run_revision_analysis.py for execution")
