# EuroJackpot ML Analysis (2018–2026)

Investigates whether there is any reproducible statistical signal in EuroJackpot
draws, using a strict train/test discipline throughout. **The default expectation
is that there is no exploitable signal** — every step is designed to prove that
empirically rather than assume it.

## Project layout

```
eurojackpot_project/
├── data/
│   ├── raw/            Original source workbooks, untouched (step3-5 legacy + clean draws)
│   └── processed/       Validated draws + generated feature tables (CSV)
│
├── notebooks/           Ad-hoc exploration only. Nothing here is a dependency
│                        of results/ — if an analysis matters, it belongs in src/.
│
├── src/                 Reusable pipeline code. No step-specific hacks.
│   ├── data_validation.py   Load + validate raw draws -> data/processed/clean_draws.csv
│   ├── features.py          Feature engineering (trailing counts now; Step 7 adds more here)
│   ├── models.py             Model + baseline definitions (sklearn models, frequency/random baselines)
│   ├── backtesting.py        Train/test split + comparison harness, calls models.py + statistics.py
│   └── statistics.py         Log loss, Brier, calibration, permutation testing, significance correction
│
├── scripts/              One thin runner per step. Orchestration only — logic lives in src/.
│   ├── run_step6.py        Step 6: Model Comparison
│   ├── run_step7.py        Step 7: Feature Engineering
│   ├── run_step8.py        Step 8: Advanced ML (validation-only model selection)
│   └── build_experiment_ledger.py   Rebuilds results/EXPERIMENT_LEDGER.md from disk
│
├── results/               One folder per step, never overwritten by later steps.
│   ├── EXPERIMENT_LEDGER.md   Auto-generated audit: model/feature/test counts, holdout exposure
│   ├── model_spec.json        Frozen model selection per pool (written by Step 8, gates holdout use)
│   ├── step6/
│   │   ├── model_comparison.xlsx   Main deliverable (formatted workbook)
│   │   ├── main_results.json       Raw metrics, main pool
│   │   ├── euro_results.json       Raw metrics, euro pool
│   │   ├── significance.json       Permutation test results
│   │   └── calibration.csv         Calibration bin data
│   ├── step7/
│   │   ├── feature_engineering.xlsx  Main deliverable (catalog, correlations, model rerun, importances)
│   │   ├── main_results.json         Rich-feature model metrics, main pool
│   │   ├── euro_results.json         Rich-feature model metrics, euro pool
│   │   ├── significance.json         Permutation test results
│   │   ├── feature_importances.json  RF/GB importances + LogReg coefficients
│   │   └── step6_vs_step7.json       Direct before/after comparison
│   └── step8/
│       ├── advanced_ml_validation.xlsx   Main deliverable — VALIDATION window only, no holdout numbers
│       ├── main_validation_results.json  Validation metrics, main pool
│       ├── euro_validation_results.json  Validation metrics, euro pool
│       ├── significance.json             Permutation tests (validation window)
│       └── feature_importances.json      XGBoost/LightGBM importances
│
├── requirements.txt
└── README.md
```

## Why this structure

- **No monolithic script.** Each concern (validation, features, models, backtesting,
  statistics) is its own module, imported by whichever step needs it. A step's
  runner script in `scripts/` is a few dozen lines of orchestration, not logic.
- **`src/features.py` is the one place features get added.** Step 7 (gaps, pairs,
  co-occurrence, odd/even structure, etc.) extends this file; `models.py` and
  `backtesting.py` don't need to change to pick up new features.
- **`src/models.py` is the one place models get added.** Step 8 (XGBoost/LightGBM,
  ensembles) extends `get_ml_models()`/`get_advanced_models()`; the backtesting harness
  doesn't change.
- **`src/statistics.py` is the one place significance logic lives.** Step 10
  (confidence intervals, bootstrap, multiple-testing correction, effect sizes)
  extends this file rather than duplicating test logic per step.
- **`src/splits.py` is the one place the train/validation/holdout boundary is defined.**
  Every script imports `three_way_split()` from here rather than re-deriving draw
  ranges, so the holdout can't silently drift or get redefined per script.
- **`src/experiment_log.py` gates holdout access in code, not just by convention.**
  `freeze_model_spec()` / `require_frozen_spec()` mean a script literally cannot
  score the holdout for a pool until model selection for that pool is frozen.
- **`results/stepN/` is append-only.** Nothing later overwrites an earlier step's
  output, so the full history of the investigation stays reproducible and auditable.
- **`results/EXPERIMENT_LEDGER.md` is regenerated, never hand-edited.** It counts
  models/strategies/features/statistical tests directly from the JSON files on disk,
  so the count is always trustworthy — see "Data discipline" below.

## Data discipline (read before adding a step)

The dataset is small (690 draws total). Steps 6 and 7 both evaluated every strategy
against "the last 100 draws," calling it a test set — that window got scored by
**20 separate permutation tests** across those two steps (see
`results/EXPERIMENT_LEDGER.md` for the exact, auto-computed count). No model was
tuned or dropped based on those numbers, but repeated exposure like that means it
can no longer be honestly called an untouched final holdout.

**From Step 8 onward, the split is three-way** (`src/splits.py`):

| Split | Draw range | Size | Used for |
|---|---|---|---|
| Train | draw_idx 1-489 | 489 draws | Fitting every model |
| Validation | draw_idx 490-589 | 100 draws | **Model selection** |
| Holdout | draw_idx 590-689 | 100 draws | Reserved — one final evaluation only |

The holdout is carved out before any model touches the data and asserted untouched
in code (`Splits.assert_holdout_untouched`), not just by convention. Model selection
happens entirely on the validation window, with the selection criterion fixed
*before* results are computed (see `scripts/run_step8.py`). The winning model per
pool is then frozen to `results/model_spec.json` via `experiment_log.freeze_model_spec()`
— an immutable record that any future holdout-touching script must check for via
`experiment_log.require_frozen_spec()` before it's allowed to run.

Run `python scripts/build_experiment_ledger.py` any time to regenerate an exact,
disk-derived count of every model, strategy, feature, and statistical test run so
far, and which of them touched the holdout window.

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

## Key methodological rules (apply to every future step)

1. **No leakage.** A feature for draw *i* may only use information from draws
   strictly before *i*. `features.trailing_count_features` already enforces this
   (first draw has no valid features and is dropped).
2. **Fixed or walk-forward split, never shuffled.** Draws are chronological;
   `backtesting.chronological_split` always holds out the *most recent* draws as
   test data.
3. **Compare against random expectation and simple frequency, always.** A model
   is only interesting relative to these baselines, not in isolation.
4. **Report log loss/Brier, not just hit count.** Hit count on 5-of-50 picks is
   extremely noisy over 100 draws; probability-quality metrics are far more
   sensitive to whether a model is doing anything real.
5. **Correct for multiple testing before calling anything significant.** See
   `statistics.bonferroni_significant`. A p-value under 0.05 across a table of
   10 strategies is expected roughly half the time by chance alone.
