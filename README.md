# NSE 15:00 Volatility Acceleration — Analysis Pipeline (Revised)

Reproducible Python pipeline behind the working paper **"Mandated Broker
Square-Off and Intraday Volatility Acceleration: Evidence from NSE Index
Derivatives"** (Mohd Atif Mansuri, atiftepl7492@gmail.com).

This is a **revised** repository. Compared to an earlier version of this
project, it adds a data-driven-bandwidth, higher-order local-polynomial RDD
estimator, explicit Bonferroni/Benjamini–Hochberg multiple-testing
correction, a Bank Nifty pre/post-peak-margin-completion sub-period split,
and a precise accounting of why an open-interest interaction term could not
be estimated. See `NSE_Paper_REVISED.pdf` for the full writeup — this
README documents the code, not the paper's arguments.

**Central finding, stated plainly:** Bank Nifty and a majority of a 62-stock
cross-section show a **multiple-testing-robust** volatility jump at 15:00
IST. Nifty's own discontinuity is **not** robust: it loses significance
under a linear specification at its own data-driven-optimal bandwidth, and
a 14:30 placebo cutoff is more significant than the true 15:00 cutoff. We
report this fragility rather than concealing it — see Section IV.C of the
paper.

---

## What's in this repo

| File | Purpose |
|---|---|
| `nse_3pm_pipeline.py` | Core statistical library: KDE/GMM clustering test, fixed-bandwidth RDD, placebo tests, reduced-form GMM. |
| `robust_rdd_and_corrections.py` | **New in this revision.** CV-optimal bandwidth selection, higher-order (quadratic) local-polynomial RDD, bandwidth-sensitivity sweep, Bonferroni and BH-FDR multiple-testing correction. |
| `run_index_pipeline.py` | Runs the original fixed-bandwidth battery on Nifty 50 / Bank Nifty. |
| `run_revision_analysis.py` | **New in this revision.** Runs the full robust-RDD battery (CV-optimal bandwidth, higher-order fit, bandwidth sensitivity, multiple-testing-corrected placebos) on Nifty 50 and Bank Nifty, plus the Bank Nifty pre/post-1-September-2021 sub-period split. This is the exact script behind Sections IV.A and IV.C of the paper. |
| `batch_pipeline_62stocks.py` | Runs the full battery across a folder of individual-stock minute-bar CSVs. |
| `62stocks_with_multiple_testing_correction.csv` | Output of the above: all 62 stocks' clustering/RDD/placebo results plus Bonferroni- and BH-FDR-corrected p-values (Section IV.E). |
| `figures/` | All figures used in the paper (see below). |
| `NSE_Paper_REVISED.pdf` | The paper itself. |

## Figures

- `figures/nifty_real_plots.png`, `figures/banknifty_real_plots.png` — mean `|return|` by minute-from-open and activity-weighted execution-time density (Figures 1–2 equivalent).
- `figures/bw_sensitivity_nifty.png`, `figures/bw_sensitivity_banknifty.png` — **new**: RDD point estimate and 95% CI across a 10–60-minute bandwidth grid (Figures 4–5). Compare these two directly: Nifty's confidence band widens toward zero as bandwidth grows; Bank Nifty's does not.
- `figures/banknifty_subperiod_comparison.png` — **new**: pre- vs. post-1-September-2021 higher-order RDD jump (Figure 3).
- `figures/crosssection_62stocks_BHFDR.png` — cross-sectional jump by archetype, with BH-FDR-significant stocks colored (Figure 6).

## A terminology note carried over from the paper

We call our RDD estimator **"higher-order local-polynomial,"** not
**"bias-corrected."** It is CCT (Calonico–Cattaneo–Titiunik, 2014)-inspired:
it selects its bandwidth via cross-validated MSE minimization (the same
logic as CCT's plug-in bandwidth) and reduces linear-specification bias via
a local-quadratic fit at the same bandwidth. It does **not** replicate the
`rdrobust` package's exact plug-in bandwidth formula or its formal robust
bias-corrected variance estimator — a genuinely different, more
conservative object. If you need the exact CCT/rdrobust procedure, use the
`rdrobust` R/Stata/Python package directly; this code is a from-scratch,
dependency-light approximation, not a substitute for it in a context where
the distinction matters.

## Data format expected

**Index-level** (`nse_3pm_pipeline.load_real_data`, used by
`run_index_pipeline.py` and `run_revision_analysis.py`): a CSV with separate
date and time columns, e.g.

```
Date,Time,Open,High,Low,Close
09-01-2015,09:15:00,17200.1,17205.3,17198.0,17203.5
```

**Per-stock** (`batch_pipeline_62stocks.load_stock_csv`): a CSV with a
single combined datetime column, e.g.

```
date,open,high,low,close,volume
2015-02-02 09:15:00,30.7,30.7,30.7,30.8,256
```

Both loaders drop non-standard sessions (e.g. Diwali Muhurat trading) and
non-positive prices (bad ticks, which otherwise produce infinite returns via
`pct_change`) before computing returns.

## Installation

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

No other dependencies (no `rdrobust`, no `pyarrow`). Tested with Python
3.10+.

## Usage

### Original fixed-bandwidth battery

```bash
python3 run_index_pipeline.py /path/to/nifty_minute_data.csv /path/to/banknifty_minute_data.csv
```

### Revised, robust-RDD battery (recommended — this is what the paper reports)

```bash
python3 run_revision_analysis.py /path/to/NF1.csv /path/to/bank-nifty-1m-data.csv
```

Prints the full battery to the console — CV-optimal bandwidth, linear and
higher-order jump estimates, the full bandwidth-sensitivity table, and
Bonferroni/BH-FDR-corrected placebo p-values for both indices, plus the
Bank Nifty pre/post-1-Sept-2021 split — and saves:
- `robust_rdd_results.json` — every number, machine-readable
- `bw_sensitivity_nifty.png`, `bw_sensitivity_banknifty.png`

To use the library directly on your own data:

```python
import nse_3pm_pipeline as m
import robust_rdd_and_corrections as r

panel = m.load_real_data("your_data.csv", date_col="Date", time_col="Time",
                          close_col="Close", date_fmt="%d-%m-%Y")
panel["abs_ret"] = panel["ret"].abs()

h_star, cv_scores = r.cv_select_bandwidth(panel, "abs_ret", t0=345)
linear_estimate = r.local_poly_rdd(panel, "abs_ret", t0=345, bandwidth=h_star, order=1)
higher_order_estimate = r.local_poly_rdd(panel, "abs_ret", t0=345, bandwidth=h_star, order=2)

sensitivity_table = r.bandwidth_sensitivity(panel, "abs_ret", t0=345)
r.plot_bandwidth_sensitivity(sensitivity_table, "My Series", "my_bw_plot.png")

# Multiple-testing correction across a placebo battery
pvals = [0.03, 0.001, 0.04, 0.06]  # e.g. 14:15, 14:30, 14:45, 15:00-true
bonferroni_p = r.bonferroni(pvals)
bh_fdr_p = r.benjamini_hochberg(pvals)
```

### Per-stock batch (folder of individual-stock CSVs)

```bash
python3 batch_pipeline_62stocks.py /path/to/folder_of_stock_csvs/
```

Writes `62stocks_full_summary.csv`. To add multiple-testing correction on
top of that output (as reported in the paper):

```python
import pandas as pd
import robust_rdd_and_corrections as r

df = pd.read_csv("62stocks_full_summary.csv")
df["bonferroni_p"] = r.bonferroni(df["rdd_absret_p"].values)
df["bh_fdr_p"] = r.benjamini_hochberg(df["rdd_absret_p"].values)
```

## Headline findings (from the paper, real data)

- **Nifty 50** (2017, N=65,499): at the CV-optimal 60-minute bandwidth, the
  linear-specification jump at 15:00 is **not** significant (p=0.063,
  Bonferroni p=0.253, BH-FDR p=0.084) — a 14:30 placebo is more significant
  (p<0.001) than the true cutoff. The higher-order (quadratic) estimate at
  the same bandwidth **is** significant (p<0.001). Read together: the
  Nifty finding is specification-dependent, not settled.
- **Bank Nifty** (2015–2026, N=1,045,858): significant at every tested
  bandwidth (10–60 min), under both linear and higher-order specifications,
  and in both the pre- and post-1-September-2021 sub-periods separately.
- **62 individual stocks**, fully re-derived from raw data: 46/62 (74.2%)
  show a significant `|return|` jump raw; 37/62 (59.7%) remain significant
  under conservative Bonferroni correction; all 6 apparent *directional*
  exceptions lose significance under either correction (likely false
  positives). Stocks flagged as higher-retail-leverage show roughly double
  the mean jump of institutional bluechips — the paper's most persuasive
  piece of evidence, though the archetype labels themselves are not
  independently verified.
- **Open interest**: examined but not usable. The one F&O bhavcopy file
  available covers only 35 single-stock contracts (no index-level
  contracts) over 1 Jan–9 Aug 2016 — zero date overlap with the 2017 Nifty
  sample, and only 3 overlapping tickers with the 62-stock panel, in
  non-overlapping periods. The `OI_j^ret` interaction term in the paper's
  GMM specification is explicitly deferred, not estimated.

## Known limitations (see the paper's Section IV.F for the full list)

- Minute-bar OHLC only — no tick-level execution timestamps.
- The exact `rdrobust`/CCT robust bias-corrected variance estimator is not
  implemented (see terminology note above).
- The 62-stock cross-section uses the linear specification only; per-stock
  higher-order robustness (analogous to the Nifty/Bank Nifty check) is not
  yet run.
- No options/implied-volatility data — the gamma cross-diffusion mechanism
  in the paper's theory section is untested.

## License

Add a license of your choice (e.g. MIT) before publishing.
