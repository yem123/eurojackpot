# EuroJackpot ML Analysis (2018–2026)

## Running a step

```bash
pip install -r requirements.txt

# Step 0 (only needed once, or after new raw data arrives): validate raw draws
python src/data_validation.py

# Step 6: Model Comparison
python scripts/run_step6.py

# Step 7: Feature Engineering
python scripts/run_step7.py

# Step 8: Advanced ML (validation-only model selection; holdout untouched)
python scripts/run_step8.py

# Rebuild the audit ledger (run any time, reads current results/*.json)
python scripts/build_experiment_ledger.py
```

Each runner reads from `data/processed/`, never from `data/raw/` directly — raw
data only ever passes through `data_validation.py`.

## Status

| Step | Description | Status |
|---|---|---|
| 1–5 | Data collection, cleaning, exploratory analysis, backtesting, first ML backtest | Done (source: `data/raw/*.xlsx`, pre-existing) |
| 6 | Model Comparison (Logistic Regression / Random Forest / Gradient Boosting vs frequency & random) | **Done** — see `results/step6/`. No model shows a reproducible edge over random once probabilities are properly scored (log loss/Brier) and results are checked for significance. |
| 7 | Feature Engineering (15 new features: multi-window frequency, gaps, trend, recency weighting, pair/triplet co-occurrence, adjacency, odd/even, low/high, draw-context sum/odd/low) | **Done** — see `results/step7/`. Correlation of every feature with the outcome is ~0. Re-running Step 6's models with the expanded set barely moves anything. **Caveat:** evaluated on the same last-100-draw window as Step 6 — see "Data discipline" above; not a clean holdout test. |
| 8 | Advanced ML (XGBoost, LightGBM, averaging ensemble, post-hoc calibration) | **Done** — see `results/step8/`. Run as a predefined validation-only experiment: model selection (lowest validation log loss) happens on a dedicated validation window, the holdout is never touched, and the winning model per pool is frozen to `results/model_spec.json`. Winning margins are tiny (main: LogReg vs RF, Δ log loss ≈ 0.00004) and neither winner's validation permutation p-value is anywhere near significant (0.79 / 0.84) — consistent with no real signal. |
| 9 | Walk-Forward Optimization | Not started |
| 9 | Walk-Forward Optimization | Not started |
| 10 | Statistical Significance | Not started |
| 11–21 | Holdout test, stability analysis, probability model, combination generation, ensemble, final selection, final backtest, report, production pipeline, final package | Not started |
