# NSE 3:00 PM Regime-Shift Analysis Pipeline

Reproducible Python pipeline behind the working paper **"Intraday Liquidity
Cascades, Order Book Imbalance, and Gamma Explosions: A Quantitative
Analysis of the 3:00 PM Regime Shift in NSE Index Derivatives"** (Mohd Atif
Mansuri, atiftepl7492@gmail.com).

It tests whether NSE index/stock prices show a statistical discontinuity
around **15:00 IST**, associated with mandatory broker Risk Management
System (RMS) auto-square-off of retail intraday leverage and, historically,
the 15:00–15:30 IST derivatives closing-price VWAP window.

All results in the paper were produced by the scripts in this repo, run
directly against real NSE minute-bar data — no numbers in the paper were
hand-entered.

---

## What's in this repo

| File | Purpose |
|---|---|
| `nse_3pm_pipeline.py` | Core statistical library: KDE/GMM clustering test, regression discontinuity design (RDD), placebo tests, reduced-form GMM. Imported by both runner scripts below. |
| `run_index_pipeline.py` | Runs the full battery on **index-level** data (used for Nifty 50 and Bank Nifty in the paper). |
| `batch_pipeline_62stocks.py` | Runs the full battery across a **folder of individual-stock** minute-bar CSVs and writes one summary row per stock. |
| `figures/` | Committed reference charts from the paper (see below) — regenerate at any time by re-running the scripts on your own data. |
| `README.md` | This file. |

## Figures

`figures/nifty_real_plots.png` and `figures/banknifty_real_plots.png` —
mean `|return|` by minute-from-open, and the activity-weighted
execution-time density, for Nifty 50 (2017) and Bank Nifty (2015–2026)
respectively.

`figures/62stocks_cross_section.png` — mean RDD `|return|` jump by
author-supplied archetype label, and the full distribution across all 62
individually re-derived stocks (blue = statistically significant at 5%).

## What the pipeline tests

1. **Clustering test** (`clustering_test`) — Section II.A. Fits a 2-component
   vs. 3-component Gaussian mixture to the intraday, activity-weighted
   execution-time distribution and runs a likelihood-ratio test for a third
   mode anchored near 15:00 (minute 345 from a 09:15 open).
2. **Regression discontinuity design** (`local_linear_rdd`) — Section III.A.
   Local-linear regression with a triangular kernel and donut exclusion
   around a cutoff (default t₀ = 345, i.e. 15:00 IST), on both signed
   returns and `|return|` (a volatility proxy). Reports the jump, t-stat,
   and p-value.
3. **Placebo tests** (`placebo_rdd`) — Section III.C. Re-runs the RDD at
   false cutoffs (default 14:15 / 14:30 / 14:45) to check whether any
   detected effect is specific to 15:00 or part of a broader trend.
4. **Reduced-form panel GMM** (`simplified_gmm`) — Section III.B. A
   two-step GMM comparing mean `|return|` in the 15:00–15:15 window against
   the preceding 15 minutes, with an autoregressive control.

## Data format expected

**Index-level** (`nse_3pm_pipeline.load_real_data`): a CSV with separate
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

Both loaders drop non-standard sessions (e.g. Diwali Muhurat trading, or
any day whose minute-from-open range exceeds ~390 minutes) and non-positive
prices (bad ticks) before computing returns.

## Installation

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

No other dependencies. Tested with Python 3.10+.

## Usage

### Index-level (Nifty 50 / Bank Nifty style data)

```bash
python3 run_index_pipeline.py /path/to/nifty_minute_data.csv /path/to/banknifty_minute_data.csv
```

Prints the full battery for both series to the console and saves:
- `index_pipeline_results.json` — every number, machine-readable
- `nifty_real_plots.png`, `banknifty_real_plots.png` — mean `|return|` by
  minute-from-open, and the activity-weighted execution-time density

To use the library directly on your own data:

```python
import nse_3pm_pipeline as m

panel = m.load_real_data("your_data.csv", date_col="Date", time_col="Time",
                          close_col="Close", date_fmt="%d-%m-%Y")

clustering = m.clustering_test(panel)
rdd_vol = m.local_linear_rdd(panel, "abs_ret", t0=345, bandwidth=30)
placebo = m.placebo_rdd(panel, "abs_ret")
gmm = m.simplified_gmm(m.build_panel_for_gmm(panel))
```

### Per-stock batch (folder of individual-stock CSVs)

```bash
python3 batch_pipeline_62stocks.py /path/to/folder_of_stock_csvs/
```

Each file should be named `<TICKER>_minute.csv` (an optional numeric
upload-id prefix like `12345_TICKER_minute.csv` is stripped automatically).
Writes `62stocks_full_summary.csv` with one row per stock:

```
ticker, status, trading_days, n_rows, irregular_days_dropped,
clus_mode_near_3pm, clus_lr_stat, clus_pvalue,
rdd_ret_jump, rdd_ret_t, rdd_ret_p,
rdd_absret_jump, rdd_absret_t, rdd_absret_p,
placebo_300_p, placebo_315_p, placebo_330_p, seconds
```

## Headline findings (from the paper, real data)

- **Nifty 50** (2017, N=65,499) and **Bank Nifty** (2015–2026, N=1,045,858):
  significant jump in `|return|` at 15:00 (t=3.82 and t=13.90
  respectively), but **no** significant jump in signed returns.
- **62 individual NSE stocks**, fully re-derived from raw data: 46/62
  (74.2%) show a significant `|return|` jump; only 6/62 (9.7%) show a
  significant *directional* jump; stocks flagged as higher-retail-leverage
  names show roughly double the mean jump of institutional bluechips.
- **Important honest caveat**: placebo cutoffs (14:30, 14:45) are *also*
  frequently significant — in 47/62 stocks and both indices. This means the
  evidence supports "volatility accelerates through the last trading hour"
  more cleanly than it supports "a sharp discontinuity unique to 15:00."
  A higher-order RDD specification with a formal bandwidth selector
  (Calonico–Cattaneo–Titiunik, 2014) is flagged as the natural next step to
  sharpen this.

See the full paper for the theoretical model (KDE, order-book-imbalance
predatory-trading model, gamma cross-diffusion SDE) and complete
methodological discussion.

## Known limitations

- Minute-bar OHLC only — no tick-level execution timestamps, so the
  clustering test uses an `|return|`-weighted resampling proxy, not literal
  trade-time KDE.
- No open-interest data — the interaction term in the full panel GMM
  specification (retail leverage × treatment window) is not estimated here.
- RDD uses a fixed 30-minute bandwidth and a linear control function; see
  the placebo-test caveat above.

## License

Add a license of your choice (e.g. MIT) before publishing.
