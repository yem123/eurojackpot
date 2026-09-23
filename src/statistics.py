"""
statistics.py

Evaluation and significance-testing helpers shared by every backtest:
log loss, Brier score, calibration tables, and a permutation test for
whether an observed hit rate could plausibly be chance.

Kept separate from backtesting.py so Step 10 (the full statistical
significance workstream: CIs, bootstrap, multiple-testing correction,
effect sizes) extends this file rather than rewriting the backtest loop.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss


def score_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Log loss and Brier score over every (draw, number) pair."""
    return {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bins predictions into n_bins quantile buckets and compares mean predicted
    probability against the actual observed hit rate in each bucket."""
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df["bin"] = pd.cut(df["p"], bins=n_bins)
    grp = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"), mean_predicted=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )
    return grp


def hits_per_draw(test_long: pd.DataFrame, proba: np.ndarray, k_pick: int) -> np.ndarray:
    """Given a long-format test set (draw_idx, number, target) and predicted
    probabilities aligned row-for-row, returns an array of hit counts per draw
    when picking the top k_pick numbers by probability within each draw."""
    order, target_grid, pool_size, n_draws = _prepare_draw_groups(test_long)
    proba_grid = np.asarray(proba, dtype=float)[order].reshape(n_draws, pool_size)
    return _hits_grid(proba_grid, target_grid, k_pick)


def _prepare_draw_groups(test_long: pd.DataFrame):
    """Precompute the (draw x number) grid structure once so repeated calls
    (e.g. inside a 1000-iteration permutation test) don't pay pandas groupby
    overhead every time. Requires every draw to have the same number of rows
    (true here: every draw gets one row per pool number) — falls back with a
    clear error otherwise."""
    draw_idx = test_long["draw_idx"].to_numpy()
    order = np.argsort(draw_idx, kind="stable")
    draw_idx_sorted = draw_idx[order]
    target_sorted = test_long["target"].to_numpy()[order]
    unique_draws, counts = np.unique(draw_idx_sorted, return_counts=True)
    pool_size = int(counts[0])
    if not np.all(counts == pool_size):
        raise ValueError("hits_per_draw/permutation_test require every draw to have the same "
                          "number of (draw, number) rows — got uneven group sizes.")
    n_draws = len(unique_draws)
    target_grid = target_sorted.reshape(n_draws, pool_size)
    return order, target_grid, pool_size, n_draws


def _hits_grid(proba_grid: np.ndarray, target_grid: np.ndarray, k_pick: int) -> np.ndarray:
    """Vectorized top-k hit count across all draws at once: (n_draws, pool_size) -> (n_draws,).

    Tie-break rule (explicit, deterministic): when two numbers have equal
    predicted probability, the lower number is preferred. This is implemented
    via a stable sort on -probability — since each row's columns are already
    in ascending-number order by construction, a stable sort's tie-breaking
    naturally falls back to that original (ascending-number) order.
    """
    if k_pick >= proba_grid.shape[1]:
        return target_grid.sum(axis=1)
    order = np.argsort(-proba_grid, axis=1, kind="stable")[:, :k_pick]
    return np.take_along_axis(target_grid, order, axis=1).sum(axis=1)


def permutation_test(test_long: pd.DataFrame, proba: np.ndarray, k_pick: int,
                      n_perm: int = 1000, seed: int = 0) -> dict:
    """
    Null hypothesis: the model's predicted probabilities carry no real
    information about which numbers get drawn. Builds a null distribution
    of avg hits by randomly shuffling the probability values across
    (draw, number) pairs n_perm times, and reports where the observed
    avg hits falls in that null distribution.

    Vectorized: the (draw x number) grid is built once, and each permutation
    reshapes a shuffled probability vector into that grid rather than
    re-grouping with pandas — same statistic, much faster for n_perm in the
    thousands across several models.

    NOTE: does not correct for testing multiple models/strategies at once.
    Apply a multiple-testing correction (e.g. Bonferroni, Benjamini-Hochberg)
    across the full set of comparisons before treating any single result as
    significant — see Step 10.
    """
    rng = np.random.default_rng(seed)
    order, target_grid, pool_size, n_draws = _prepare_draw_groups(test_long)
    proba = np.asarray(proba, dtype=float)
    n = len(proba)

    observed_grid = proba[order].reshape(n_draws, pool_size)
    observed = _hits_grid(observed_grid, target_grid, k_pick).mean()

    count_ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(n)
        perm_grid = proba[shuffled][order].reshape(n_draws, pool_size)
        stat = _hits_grid(perm_grid, target_grid, k_pick).mean()
        if stat >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {"avg_hits": float(observed), "perm_p_value": float(p_value), "n_perm": n_perm}


def bonferroni_significant(p_value: float, n_tests: int, alpha: float = 0.05) -> bool:
    return p_value < (alpha / n_tests)
