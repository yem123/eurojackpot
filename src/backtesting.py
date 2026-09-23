"""
backtesting.py

Chronological train/test splitting and the model-comparison harness used
by each backtest step. Every model (ML or baseline) is scored on the same
held-out draws using the metrics in statistics.py.

Step 9 (walk-forward optimization) will add a rolling/expanding-window
variant here; run_comparison() below is the fixed-split version used
through Step 6-8.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from features import to_long_format
from models import (
    get_ml_models,
    get_advanced_models,
    AveragingEnsemble,
    calibrate_probabilities,
    HistoricalFrequencyBaseline,
    RecentFrequencyBaseline,
    RandomExpectationBaseline,
)
from statistics import score_probabilities, hits_per_draw, permutation_test


def chronological_split(draws: pd.DataFrame, n_test: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a chronologically-sorted draws table into train/test, test = last n_test draws."""
    assert draws["draw_date"].is_monotonic_increasing, "draws must be sorted ascending by date"
    return draws.iloc[:-n_test].reset_index(drop=True), draws.iloc[-n_test:].reset_index(drop=True)


def run_comparison(draws: pd.DataFrame, feature_wide: pd.DataFrame, drawn_cols: list[str],
                    pool_size: int, k_pick: int, window: int = 20, n_test: int = 100,
                    n_perm: int = 1000, random_state: int = 42) -> dict:
    """
    Full Step 6-style comparison for one pool (main or euro):
    trains Logistic Regression / Random Forest / Gradient Boosting plus three
    baselines on the same chronological split, scores all of them on the same
    held-out draws, and runs a permutation significance check on each.

    Returns {'results': {name: metrics}, 'predictions': {name: np.array},
             'test_long': DataFrame, 'significance': {name: {...}}}
    """
    long_df = to_long_format(draws, feature_wide, drawn_cols, pool_size, window)
    feat_col = f"feat_prev{window}"

    n_train_draws = len(draws) - n_test
    train_long = long_df[long_df["draw_idx"] < n_train_draws].reset_index(drop=True)
    test_long = long_df[long_df["draw_idx"] >= n_train_draws].reset_index(drop=True)

    Xtr = train_long[[feat_col]].to_numpy(dtype=float)
    ytr = train_long["target"].to_numpy()
    Xte = test_long[[feat_col]].to_numpy(dtype=float)
    yte = test_long["target"].to_numpy()

    predictions, results, significance = {}, {}, {}

    for name, model in get_ml_models(random_state=random_state).items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        predictions[name] = proba

    hist_freq = HistoricalFrequencyBaseline().fit(train_long["number"].to_numpy(), ytr)
    predictions["Historical Frequency"] = hist_freq.predict_proba(test_long["number"].to_numpy())[:, 1]

    recent_freq = RecentFrequencyBaseline(window=window).fit(Xtr, ytr)
    predictions["Recent-N Frequency"] = recent_freq.predict_proba(test_long[feat_col].to_numpy())[:, 1]

    rand_baseline = RandomExpectationBaseline(k_pick=k_pick, pool_size=pool_size).fit(Xtr, ytr)
    predictions["Random Expectation"] = rand_baseline.predict_proba(Xte)[:, 1]

    random_exp_hits = k_pick * k_pick / pool_size

    for name, proba in predictions.items():
        metrics = score_probabilities(yte, proba)
        if name == "Random Expectation":
            metrics.update({"draws": n_test, "avg_hits": random_exp_hits,
                             "zero": None, "one_plus": None, "two_plus": None, "max_hits": None})
        else:
            hits = hits_per_draw(test_long, proba, k_pick)
            metrics.update({
                "draws": len(hits), "avg_hits": float(hits.mean()),
                "zero": int((hits == 0).sum()), "one_plus": int((hits >= 1).sum()),
                "two_plus": int((hits >= 2).sum()), "max_hits": int(hits.max()),
            })
            significance[name] = permutation_test(test_long, proba, k_pick, n_perm=n_perm, seed=random_state)
        results[name] = metrics

    return {
        "results": results,
        "predictions": predictions,
        "test_long": test_long,
        "significance": significance,
        "random_expectation_avg_hits": random_exp_hits,
    }


# ---------------------------------------------------------------------------
# Step 7+ — generalized comparison over an arbitrary long-format feature table
#
# run_comparison() above is kept exactly as-is (it built its own single-feature
# long table internally) so results/step6 stays reproducible. This version
# takes an already-built long table (e.g. from features.build_rich_long_features)
# with any number of feature columns, so Step 7/8 model runs don't need to
# duplicate the train/test/score/permutation-test logic.
# ---------------------------------------------------------------------------

def run_model_comparison(long_df: pd.DataFrame, feature_cols: list[str], baseline_feature_col: str,
                          k_pick: int, pool_size: int, n_test_draws: int, n_perm: int = 1000,
                          random_state: int = 42) -> dict:
    """
    Same evaluation as run_comparison(), but generalized to any feature set.

    long_df must have columns: draw_idx, number, target, plus every column in
    feature_cols (and baseline_feature_col, normally one of feature_cols).
    Split is chronological: the last n_test_draws distinct draw_idx values are
    the test set.

    Returns the same shape as run_comparison(): results / predictions /
    test_long / significance / random_expectation_avg_hits, plus
    'feature_importances' for the tree models when available.
    """
    draw_ids = np.sort(long_df["draw_idx"].unique())
    test_ids = set(draw_ids[-n_test_draws:])
    train_long = long_df[~long_df["draw_idx"].isin(test_ids)].reset_index(drop=True)
    test_long = long_df[long_df["draw_idx"].isin(test_ids)].reset_index(drop=True)

    Xtr = train_long[feature_cols].to_numpy(dtype=float)
    ytr = train_long["target"].to_numpy()
    Xte = test_long[feature_cols].to_numpy(dtype=float)
    yte = test_long["target"].to_numpy()

    predictions, results, significance, importances = {}, {}, {}, {}

    for name, model in get_ml_models(random_state=random_state).items():
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        predictions[name] = proba
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_cols, model.feature_importances_.tolist()))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_cols, model.coef_[0].tolist()))

    hist_freq = HistoricalFrequencyBaseline().fit(train_long["number"].to_numpy(), ytr)
    predictions["Historical Frequency"] = hist_freq.predict_proba(test_long["number"].to_numpy())[:, 1]

    baseline_window = int(round(train_long[baseline_feature_col].max())) or 20
    recent_freq = RecentFrequencyBaseline(window=max(baseline_window, 1)).fit(Xtr, ytr)
    predictions["Recent-N Frequency"] = recent_freq.predict_proba(test_long[baseline_feature_col].to_numpy())[:, 1]

    rand_baseline = RandomExpectationBaseline(k_pick=k_pick, pool_size=pool_size).fit(Xtr, ytr)
    predictions["Random Expectation"] = rand_baseline.predict_proba(Xte)[:, 1]

    random_exp_hits = k_pick * k_pick / pool_size

    for name, proba in predictions.items():
        metrics = score_probabilities(yte, proba)
        if name == "Random Expectation":
            metrics.update({"draws": n_test_draws, "avg_hits": random_exp_hits,
                             "zero": None, "one_plus": None, "two_plus": None, "max_hits": None})
        else:
            hits = hits_per_draw(test_long, proba, k_pick)
            metrics.update({
                "draws": len(hits), "avg_hits": float(hits.mean()),
                "zero": int((hits == 0).sum()), "one_plus": int((hits >= 1).sum()),
                "two_plus": int((hits >= 2).sum()), "max_hits": int(hits.max()),
            })
            significance[name] = permutation_test(test_long, proba, k_pick, n_perm=n_perm, seed=random_state)
        results[name] = metrics

    return {
        "results": results,
        "predictions": predictions,
        "test_long": test_long,
        "significance": significance,
        "random_expectation_avg_hits": random_exp_hits,
        "feature_importances": importances,
    }


# ---------------------------------------------------------------------------
# Step 8 — advanced models (XGBoost/LightGBM where available), an averaging
# ensemble over every fitted model, and post-hoc probability calibration.
# Reuses the same long-format feature table and train/test split logic as
# run_model_comparison() (Step 7), so it's a straight extension, not a
# reimplementation.
# ---------------------------------------------------------------------------

def run_advanced_comparison(long_df: pd.DataFrame, feature_cols: list[str], baseline_feature_col: str,
                             k_pick: int, pool_size: int, n_test_draws: int, n_perm: int = 1000,
                             random_state: int = 42, calibration_method: str = "isotonic") -> dict:
    """
    Trains every Step 6 model (Logistic Regression, Random Forest, Gradient
    Boosting) plus every Step 8 model available (XGBoost, LightGBM), builds a
    simple probability-averaging ensemble over all of them, and — for every
    model including the ensemble — fits a post-hoc calibrator (isotonic or
    sigmoid) on that model's own training-set predictions and applies it to
    the test-set predictions.

    Returns:
      results / predictions / test_long / significance / random_expectation_avg_hits
          — same shape as run_model_comparison(), for the RAW (uncalibrated) predictions.
      calibrated_results / calibrated_predictions
          — same metrics computed on the calibrated probabilities.
      feature_importances — for every tree-based model.
    """
    draw_ids = np.sort(long_df["draw_idx"].unique())
    test_ids = set(draw_ids[-n_test_draws:])
    train_long = long_df[~long_df["draw_idx"].isin(test_ids)].reset_index(drop=True)
    test_long = long_df[long_df["draw_idx"].isin(test_ids)].reset_index(drop=True)

    Xtr = train_long[feature_cols].to_numpy(dtype=float)
    ytr = train_long["target"].to_numpy()
    Xte = test_long[feature_cols].to_numpy(dtype=float)
    yte = test_long["target"].to_numpy()

    all_models = {**get_ml_models(random_state=random_state), **get_advanced_models(random_state=random_state)}

    fitted, predictions, importances = {}, {}, {}
    train_predictions = {}
    for name, model in all_models.items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        predictions[name] = model.predict_proba(Xte)[:, 1]
        train_predictions[name] = model.predict_proba(Xtr)[:, 1]
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_cols, np.asarray(model.feature_importances_).tolist()))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_cols, np.asarray(model.coef_[0]).tolist()))

    ensemble = AveragingEnsemble(fitted)
    predictions["Averaging Ensemble"] = ensemble.predict_proba(Xte)[:, 1]
    train_predictions["Averaging Ensemble"] = ensemble.predict_proba(Xtr)[:, 1]

    hist_freq = HistoricalFrequencyBaseline().fit(train_long["number"].to_numpy(), ytr)
    predictions["Historical Frequency"] = hist_freq.predict_proba(test_long["number"].to_numpy())[:, 1]

    baseline_window = int(round(train_long[baseline_feature_col].max())) or 20
    recent_freq = RecentFrequencyBaseline(window=max(baseline_window, 1)).fit(Xtr, ytr)
    predictions["Recent-N Frequency"] = recent_freq.predict_proba(test_long[baseline_feature_col].to_numpy())[:, 1]

    rand_baseline = RandomExpectationBaseline(k_pick=k_pick, pool_size=pool_size).fit(Xtr, ytr)
    predictions["Random Expectation"] = rand_baseline.predict_proba(Xte)[:, 1]

    random_exp_hits = k_pick * k_pick / pool_size
    learned_models = list(all_models.keys()) + ["Averaging Ensemble"]

    def _score_all(preds_dict):
        results, significance = {}, {}
        for name, proba in preds_dict.items():
            metrics = score_probabilities(yte, proba)
            if name == "Random Expectation":
                metrics.update({"draws": n_test_draws, "avg_hits": random_exp_hits,
                                 "zero": None, "one_plus": None, "two_plus": None, "max_hits": None})
            else:
                hits = hits_per_draw(test_long, proba, k_pick)
                metrics.update({
                    "draws": len(hits), "avg_hits": float(hits.mean()),
                    "zero": int((hits == 0).sum()), "one_plus": int((hits >= 1).sum()),
                    "two_plus": int((hits >= 2).sum()), "max_hits": int(hits.max()),
                })
                if name in learned_models:
                    significance[name] = permutation_test(test_long, proba, k_pick, n_perm=n_perm, seed=random_state)
            results[name] = metrics
        return results, significance

    results, significance = _score_all(predictions)

    # Post-hoc calibration: fit on each model's own train predictions, apply to test.
    calibrated_predictions = {}
    for name in learned_models:
        try:
            calibrated_predictions[name] = calibrate_probabilities(
                train_predictions[name], ytr, predictions[name], method=calibration_method)
        except Exception:
            calibrated_predictions[name] = predictions[name]  # fall back to raw if calibration fails
    # baselines pass through unchanged (nothing to calibrate against their own train fit)
    for name in ("Historical Frequency", "Recent-N Frequency", "Random Expectation"):
        calibrated_predictions[name] = predictions[name]

    calibrated_results, _ = _score_all(calibrated_predictions)

    return {
        "results": results,
        "predictions": predictions,
        "calibrated_results": calibrated_results,
        "calibrated_predictions": calibrated_predictions,
        "test_long": test_long,
        "significance": significance,
        "random_expectation_avg_hits": random_exp_hits,
        "feature_importances": importances,
        "models_used": list(all_models.keys()),
    }


# ---------------------------------------------------------------------------
# Step 8, corrected discipline — model SELECTION happens only on a validation
# window carved out of the training period. The holdout (the same draws
# Steps 6-7 called "test") is excluded from this function entirely: it is
# never loaded into Xtr/Xte, never scored, never permutation-tested. See
# splits.py for why, and experiment_log.py for the freeze mechanism that
# gates any later use of the holdout.
# ---------------------------------------------------------------------------

def run_step8_validation(long_df: pd.DataFrame, feature_cols: list[str], baseline_feature_col: str,
                          k_pick: int, pool_size: int, validation_draws: int = 100, holdout_draws: int = 100,
                          n_perm: int = 1000, random_state: int = 42,
                          calibration_method: str = "isotonic") -> dict:
    """
    Same models/ensemble/calibration as run_advanced_comparison(), but:
      - fit on splits.train only
      - scored on splits.validation only
      - splits.holdout is carved out up front and never touched below
        (asserted via Splits.assert_holdout_untouched)

    Returns the same shape as run_advanced_comparison(), computed entirely
    on the validation window, plus 'holdout_size' and 'validation_size' for
    the record, and 'used_draw_ids' (train ∪ validation) so callers can
    double check nothing else leaked in.
    """
    from splits import three_way_split  # local import to avoid a hard dependency at module load time

    splits = three_way_split(long_df, validation_draws=validation_draws, holdout_draws=holdout_draws)
    used_ids = splits.train_ids | splits.validation_ids
    splits.assert_holdout_untouched(used_ids)

    train_long, val_long = splits.train, splits.validation

    Xtr = train_long[feature_cols].to_numpy(dtype=float)
    ytr = train_long["target"].to_numpy()
    Xval = val_long[feature_cols].to_numpy(dtype=float)
    yval = val_long["target"].to_numpy()

    all_models = {**get_ml_models(random_state=random_state), **get_advanced_models(random_state=random_state)}

    fitted, predictions, importances, train_predictions = {}, {}, {}, {}
    for name, model in all_models.items():
        model.fit(Xtr, ytr)
        fitted[name] = model
        predictions[name] = model.predict_proba(Xval)[:, 1]
        train_predictions[name] = model.predict_proba(Xtr)[:, 1]
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_cols, np.asarray(model.feature_importances_).tolist()))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_cols, np.asarray(model.coef_[0]).tolist()))

    ensemble = AveragingEnsemble(fitted)
    predictions["Averaging Ensemble"] = ensemble.predict_proba(Xval)[:, 1]
    train_predictions["Averaging Ensemble"] = ensemble.predict_proba(Xtr)[:, 1]

    hist_freq = HistoricalFrequencyBaseline().fit(train_long["number"].to_numpy(), ytr)
    predictions["Historical Frequency"] = hist_freq.predict_proba(val_long["number"].to_numpy())[:, 1]

    baseline_window = int(round(train_long[baseline_feature_col].max())) or 20
    recent_freq = RecentFrequencyBaseline(window=max(baseline_window, 1)).fit(Xtr, ytr)
    predictions["Recent-N Frequency"] = recent_freq.predict_proba(val_long[baseline_feature_col].to_numpy())[:, 1]

    rand_baseline = RandomExpectationBaseline(k_pick=k_pick, pool_size=pool_size).fit(Xtr, ytr)
    predictions["Random Expectation"] = rand_baseline.predict_proba(Xval)[:, 1]

    random_exp_hits = k_pick * k_pick / pool_size
    learned_models = list(all_models.keys()) + ["Averaging Ensemble"]

    def _score_all(preds_dict):
        results, significance = {}, {}
        for name, proba in preds_dict.items():
            metrics = score_probabilities(yval, proba)
            if name == "Random Expectation":
                metrics.update({"draws": validation_draws, "avg_hits": random_exp_hits,
                                 "zero": None, "one_plus": None, "two_plus": None, "max_hits": None})
            else:
                hits = hits_per_draw(val_long, proba, k_pick)
                metrics.update({
                    "draws": len(hits), "avg_hits": float(hits.mean()),
                    "zero": int((hits == 0).sum()), "one_plus": int((hits >= 1).sum()),
                    "two_plus": int((hits >= 2).sum()), "max_hits": int(hits.max()),
                })
                if name in learned_models:
                    significance[name] = permutation_test(val_long, proba, k_pick, n_perm=n_perm, seed=random_state)
            results[name] = metrics
        return results, significance

    results, significance = _score_all(predictions)

    calibrated_predictions = {}
    for name in learned_models:
        try:
            calibrated_predictions[name] = calibrate_probabilities(
                train_predictions[name], ytr, predictions[name], method=calibration_method)
        except Exception:
            calibrated_predictions[name] = predictions[name]
    for name in ("Historical Frequency", "Recent-N Frequency", "Random Expectation"):
        calibrated_predictions[name] = predictions[name]

    calibrated_results, _ = _score_all(calibrated_predictions)

    # Re-assert after the fact: nothing above should have touched holdout draw_idx values.
    splits.assert_holdout_untouched(set(train_long["draw_idx"]) | set(val_long["draw_idx"]))

    return {
        "results": results,
        "calibrated_results": calibrated_results,
        "significance": significance,
        "feature_importances": importances,
        "random_expectation_avg_hits": random_exp_hits,
        "models_used": list(all_models.keys()),
        "train_size": len(splits.train_ids),
        "validation_size": len(splits.validation_ids),
        "holdout_size": len(splits.holdout_ids),
        "holdout_draw_ids": sorted(splits.holdout_ids),
    }
