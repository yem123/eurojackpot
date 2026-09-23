"""
features.py

Builds model-ready features from the canonical clean-draws table
(data/processed/clean_draws.csv, produced by data_validation.py).

Current features (Step 6 baseline):
    - trailing count: how many times each number appeared in the previous
      `window` draws (default 20).

This module is intentionally the single place new features get added in
Step 7 (gaps since last appearance, pairs/triplets, odd/even & low/high
structure, sum/range, recency weighting, etc.) — models.py and
backtesting.py should not need to change when a feature is added here.
"""

from pathlib import Path
import numpy as np
import pandas as pd

MAIN_COLS = [f"main_{i}" for i in range(1, 6)]
EURO_COLS = [f"euro_{i}" for i in range(1, 3)]
MAIN_POOL = 50
EURO_POOL = 12


def trailing_count_features(draws: pd.DataFrame, drawn_cols: list[str], pool_size: int,
                             window: int = 20) -> pd.DataFrame:
    """
    For each draw i and each number n in [1, pool_size], count how many times
    n appeared in the `window` draws strictly before draw i (0 for numbers
    with no prior history). Returns a wide DataFrame: one row per draw,
    one column per number, named f"n{n}_prev{window}".
    """
    n_draws = len(draws)
    counts = np.zeros((n_draws, pool_size), dtype=int)
    drawn_matrix = draws[drawn_cols].to_numpy()

    # indicator[i, n-1] = 1 if number n was drawn at draw i
    indicator = np.zeros((n_draws, pool_size), dtype=int)
    for i in range(n_draws):
        for v in drawn_matrix[i]:
            indicator[i, int(v) - 1] = 1

    for i in range(n_draws):
        start = max(0, i - window)
        if i == 0:
            counts[i, :] = 0
        else:
            counts[i, :] = indicator[start:i, :].sum(axis=0)

    cols = [f"n{n}_prev{window}" for n in range(1, pool_size + 1)]
    out = pd.DataFrame(counts, columns=cols)
    out.insert(0, "draw_date", draws["draw_date"].values)
    # first draw has no history at all -> mark as unusable downstream
    out.loc[0, cols] = np.nan
    return out


def build_feature_set(draws: pd.DataFrame, window: int = 20) -> dict:
    """Returns {'main': wide_df, 'euro': wide_df} of trailing-count features."""
    return {
        "main": trailing_count_features(draws, MAIN_COLS, MAIN_POOL, window),
        "euro": trailing_count_features(draws, EURO_COLS, EURO_POOL, window),
    }


def to_long_format(draws: pd.DataFrame, feature_wide: pd.DataFrame, drawn_cols: list[str],
                    pool_size: int, window: int) -> pd.DataFrame:
    """
    Reshape to one row per (draw_idx, number): draw_idx, number, feat_prev{window}, target.
    target = 1 if `number` appears in that draw's drawn_cols.
    Drops rows with no history (first draw).
    """
    feat_cols = [f"n{n}_prev{window}" for n in range(1, pool_size + 1)]
    records = []
    drawn_sets = draws[drawn_cols].apply(lambda r: set(r.tolist()), axis=1)
    for draw_idx in range(len(draws)):
        if feature_wide.loc[draw_idx, feat_cols].isna().any():
            continue
        drawn = drawn_sets.iloc[draw_idx]
        for n in range(1, pool_size + 1):
            records.append((
                draw_idx,
                n,
                feature_wide.loc[draw_idx, f"n{n}_prev{window}"],
                1 if n in drawn else 0,
            ))
    return pd.DataFrame(records, columns=["draw_idx", "number", f"feat_prev{window}", "target"])


# ---------------------------------------------------------------------------
# Step 7 — expanded feature set
#
# Everything below is additive: build_feature_set()/to_long_format() above
# (Step 6) are untouched, so results/step6 stays exactly reproducible.
# build_rich_long_features() is the new entry point for Step 7 onward.
# ---------------------------------------------------------------------------

from collections import defaultdict
from itertools import combinations


def _drawn_indicator(draws: pd.DataFrame, drawn_cols: list[str], pool_size: int) -> np.ndarray:
    """indicator[i, n-1] = 1 if number n was drawn at draw i."""
    n_draws = len(draws)
    indicator = np.zeros((n_draws, pool_size), dtype=int)
    drawn_matrix = draws[drawn_cols].to_numpy().astype(int)
    for i in range(n_draws):
        for v in drawn_matrix[i]:
            indicator[i, v - 1] = 1
    return indicator


def _window_frequency(indicator: np.ndarray, window: int) -> np.ndarray:
    """count[i, n] = appearances of number n in the `window` draws strictly before draw i."""
    n_draws, pool_size = indicator.shape
    out = np.zeros((n_draws, pool_size), dtype=int)
    for i in range(1, n_draws):
        start = max(0, i - window)
        out[i, :] = indicator[start:i, :].sum(axis=0)
    return out


def _cumulative_frequency(indicator: np.ndarray) -> np.ndarray:
    """count[i, n] = appearances of number n in ALL draws strictly before draw i."""
    n_draws, pool_size = indicator.shape
    out = np.zeros((n_draws, pool_size), dtype=int)
    cum = np.zeros(pool_size, dtype=int)
    for i in range(n_draws):
        out[i, :] = cum
        cum = cum + indicator[i, :]
    return out


def _gap_since_last(indicator: np.ndarray) -> np.ndarray:
    """draws since number n last appeared, strictly before draw i.
    If n has never appeared before draw i, gap = i (distance back to the start of history)."""
    n_draws, pool_size = indicator.shape
    out = np.zeros((n_draws, pool_size), dtype=int)
    last_seen = np.full(pool_size, -1)
    for i in range(n_draws):
        out[i, :] = np.where(last_seen >= 0, i - last_seen, i)
        last_seen = np.where(indicator[i, :] == 1, i, last_seen)
    return out


def _recency_weighted(indicator: np.ndarray, half_life: int) -> np.ndarray:
    """Exponentially decayed count of past appearances (half_life in draws), strictly before draw i."""
    n_draws, pool_size = indicator.shape
    decay = 0.5 ** (1.0 / half_life)
    out = np.zeros((n_draws, pool_size), dtype=float)
    running = np.zeros(pool_size, dtype=float)
    for i in range(n_draws):
        out[i, :] = running
        running = running * decay + indicator[i, :]
    return out


def _cooccurrence_features(draws: pd.DataFrame, drawn_cols: list[str], pool_size: int):
    """
    For each draw i and number n (strictly using history before draw i):
      pair_score[i, n]     = sum of historical co-occurrence counts between n and each
                              number that appeared in draw i-1 (the immediately preceding draw)
      triplet_score[i, n]  = sum of historical triplet co-occurrence counts between n and
                              each pair of numbers from draw i-1
      adjacent_flag[i, n]  = 1 if n-1 or n+1 appeared in draw i-1, else 0
    """
    n_draws = len(draws)
    drawn_matrix = draws[drawn_cols].to_numpy().astype(int)

    pair_counts = np.zeros((pool_size, pool_size), dtype=int)
    triplet_counter = defaultdict(int)

    pair_score = np.zeros((n_draws, pool_size), dtype=int)
    triplet_score = np.zeros((n_draws, pool_size), dtype=int)
    adjacent_flag = np.zeros((n_draws, pool_size), dtype=int)

    prev_numbers = None
    for i in range(n_draws):
        if prev_numbers is not None:
            prev_set = set(prev_numbers)
            for n in range(1, pool_size + 1):
                others = [m for m in prev_numbers if m != n]
                pair_score[i, n - 1] = sum(pair_counts[n - 1, m - 1] for m in others)
                ts = 0
                for m1, m2 in combinations(others, 2):
                    ts += triplet_counter.get(frozenset({n, m1, m2}), 0)
                triplet_score[i, n - 1] = ts
                adjacent_flag[i, n - 1] = int((n - 1) in prev_set or (n + 1) in prev_set)

        cur = drawn_matrix[i].tolist()
        for a, b in combinations(cur, 2):
            pair_counts[a - 1, b - 1] += 1
            pair_counts[b - 1, a - 1] += 1
        for a, b, c in combinations(cur, 3):
            triplet_counter[frozenset({a, b, c})] += 1
        prev_numbers = cur

    return pair_score, triplet_score, adjacent_flag


def draw_context_features(draws: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Draw-level structural context (sum, odd/even mix, low/high mix), each expressed as a
    trailing rolling average over `window` draws strictly before draw i (leak-safe).
    Broadcast onto every number-row for that draw when merged into the long table.
    """
    sums = draws["main_sum"] if "main_sum" in draws.columns else draws[MAIN_COLS].sum(axis=1)
    odd_counts = draws["main_odd_count"] if "main_odd_count" in draws.columns else \
        draws[MAIN_COLS].apply(lambda r: sum(v % 2 == 1 for v in r), axis=1)
    low_counts = draws["main_low_1_25_count"] if "main_low_1_25_count" in draws.columns else \
        draws[MAIN_COLS].apply(lambda r: sum(v <= 25 for v in r), axis=1)

    df = pd.DataFrame({
        "draw_idx": range(len(draws)),
        "ctx_avg_sum": sums.rolling(window, min_periods=1).mean().shift(1),
        "ctx_avg_odd_count": odd_counts.rolling(window, min_periods=1).mean().shift(1),
        "ctx_avg_low_count": low_counts.rolling(window, min_periods=1).mean().shift(1),
    })
    # draw 0 has no prior draws -> NaN from shift(1); fill with the draw's own values as a neutral start
    df.loc[0, ["ctx_avg_sum", "ctx_avg_odd_count", "ctx_avg_low_count"]] = [
        sums.iloc[0], odd_counts.iloc[0], low_counts.iloc[0]
    ]
    return df


def build_rich_long_features(draws: pd.DataFrame, drawn_cols: list[str], pool_size: int,
                              short_window: int = 10, medium_window: int = 20, long_window: int = 50,
                              half_life: int = 10, include_context: bool = True) -> pd.DataFrame:
    """
    Step 7 feature set. Returns one row per (draw_idx, number) with:
      freq_short, freq_medium, freq_long   trailing counts over `short/medium/long_window` draws
      freq_all                              cumulative count over all draws before this one
      freq_trend                            short-term rate minus long-term rate (freq_short/short_window - freq_long/long_window)
      gap_since_last                        draws since number last appeared (= draw_idx if never seen)
      recency_weighted                      exponentially decayed appearance count (half_life draws)
      pair_score                            historical co-occurrence strength with the previous draw's numbers
      triplet_score                         historical triplet co-occurrence strength with the previous draw's numbers
      adjacent_flag                         1 if number-1 or number+1 was in the previous draw
      is_odd, is_low                        static structural properties of the number itself
      ctx_avg_sum, ctx_avg_odd_count,       trailing rolling averages of draw-level sum/odd-count/low-count
      ctx_avg_low_count                     (same value for every number within a given draw)
      target                                1 if the number was drawn

    Draw 0 is dropped (no history). All other rows are leak-safe: every feature for draw i
    is computed only from draws strictly before i (plus draw i's own static number properties).
    """
    n_draws = len(draws)
    indicator = _drawn_indicator(draws, drawn_cols, pool_size)

    freq_short = _window_frequency(indicator, short_window)
    freq_medium = _window_frequency(indicator, medium_window)
    freq_long = _window_frequency(indicator, long_window)
    freq_all = _cumulative_frequency(indicator)
    gap = _gap_since_last(indicator)
    rw = _recency_weighted(indicator, half_life)
    pair_score, triplet_score, adjacent_flag = _cooccurrence_features(draws, drawn_cols, pool_size)

    freq_trend = freq_short / short_window - freq_long / long_window

    ctx = draw_context_features(draws, window=medium_window) if include_context else None

    records = []
    for i in range(1, n_draws):  # drop draw 0 (no history)
        ctx_row = ctx.iloc[i] if ctx is not None else None
        drawn_set = set(draws.iloc[i][drawn_cols].astype(int).tolist())
        for n in range(1, pool_size + 1):
            rec = {
                "draw_idx": i,
                "number": n,
                "freq_short": freq_short[i, n - 1],
                "freq_medium": freq_medium[i, n - 1],
                "freq_long": freq_long[i, n - 1],
                "freq_all": freq_all[i, n - 1],
                "freq_trend": freq_trend[i, n - 1],
                "gap_since_last": gap[i, n - 1],
                "recency_weighted": rw[i, n - 1],
                "pair_score": pair_score[i, n - 1],
                "triplet_score": triplet_score[i, n - 1],
                "adjacent_flag": adjacent_flag[i, n - 1],
                "is_odd": int(n % 2 == 1),
                "is_low": int(n <= pool_size // 2),
                "target": int(n in drawn_set),
            }
            if ctx_row is not None:
                rec["ctx_avg_sum"] = ctx_row["ctx_avg_sum"]
                rec["ctx_avg_odd_count"] = ctx_row["ctx_avg_odd_count"]
                rec["ctx_avg_low_count"] = ctx_row["ctx_avg_low_count"]
            records.append(rec)

    return pd.DataFrame.from_records(records)


RICH_FEATURE_COLUMNS = [
    "freq_short", "freq_medium", "freq_long", "freq_all", "freq_trend", "gap_since_last",
    "recency_weighted", "pair_score", "triplet_score", "adjacent_flag", "is_odd", "is_low",
    "ctx_avg_sum", "ctx_avg_odd_count", "ctx_avg_low_count",
]


# ---------------------------------------------------------------------------
# FROZEN FEATURE SPEC — locked at the end of Step 7.
#
# These are the exact feature sets every model from Step 8 onward is trained
# on. Do not add, remove, or redefine a feature here without: (1) bumping
# FEATURE_SPEC_VERSION, (2) re-running scripts/run_step6.py and
# scripts/run_step7.py so the baseline comparisons stay valid for the new
# spec, and (3) re-running scripts/build_experiment_ledger.py. Silently
# editing this list defeats the point of freezing it.
#
# EURO_FROZEN excludes:
#   - the three ctx_* draw-context features (they're derived from the Main
#     draw's sum/odd-count/low-count and don't apply to the Euro pool)
#   - triplet_score (structurally constant at 0 for Euro: a 2-number draw
#     never has two "other" numbers left after excluding the candidate, so
#     the feature carries no information and would just be dead weight)
# ---------------------------------------------------------------------------
FEATURE_SPEC_VERSION = "1.0-step7-frozen"

FROZEN_MAIN_FEATURES = list(RICH_FEATURE_COLUMNS)
FROZEN_EURO_FEATURES = [c for c in RICH_FEATURE_COLUMNS if not c.startswith("ctx_") and c != "triplet_score"]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    draws = pd.read_csv(root / "data" / "processed" / "clean_draws.csv", parse_dates=["draw_date"])

    # Step 6 baseline feature (kept for reproducibility of results/step6)
    feats = build_feature_set(draws, window=20)
    out_dir = root / "data" / "processed"
    feats["main"].to_csv(out_dir / "main_features_prev20.csv", index=False)
    feats["euro"].to_csv(out_dir / "euro_features_prev20.csv", index=False)
    print(f"[Step 6] Main features: {feats['main'].shape}, Euro features: {feats['euro'].shape}")

    # Step 7 rich feature set
    main_rich = build_rich_long_features(draws, MAIN_COLS, MAIN_POOL)
    euro_rich = build_rich_long_features(draws, EURO_COLS, EURO_POOL, include_context=False)
    main_rich.to_csv(out_dir / "main_features_rich.csv", index=False)
    euro_rich.to_csv(out_dir / "euro_features_rich.csv", index=False)
    print(f"[Step 7] Main rich features: {main_rich.shape}, Euro rich features: {euro_rich.shape}")
    print(f"Saved -> {out_dir}")
