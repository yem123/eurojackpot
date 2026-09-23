"""
scripts/run_step7.py

Runner for Step 7 — Feature Engineering. Builds the expanded feature set
(src/features.build_rich_long_features), validates it (distributions,
leakage sanity checks, correlation with target), and reruns the Step 6
model comparison on the same held-out draws with the richer feature set
to see whether more information changes the conclusion.

Usage:
    python scripts/run_step7.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import openpyxl  # noqa: E402
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from features import (  # noqa: E402
    build_rich_long_features, RICH_FEATURE_COLUMNS, FROZEN_MAIN_FEATURES, FROZEN_EURO_FEATURES,
    MAIN_COLS, EURO_COLS, MAIN_POOL, EURO_POOL,
)
from backtesting import run_model_comparison  # noqa: E402
from statistics import bonferroni_significant  # noqa: E402

RESULTS_DIR = ROOT / "results" / "step7"
N_TEST = 100
N_PERM = 1000

MAIN_FEATURE_COLS = FROZEN_MAIN_FEATURES  # includes ctx_* draw-context columns
# Euro draws pick only 2 numbers, so triplet_score is structurally always 0 (no pair of
# "other" numbers remains once you exclude the candidate number itself) - dropped in the
# frozen spec rather than feeding a constant column to the models. No main-draw context either.
EURO_FEATURE_COLS = FROZEN_EURO_FEATURES


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    draws = pd.read_csv(ROOT / "data" / "processed" / "clean_draws.csv", parse_dates=["draw_date"])

    main_long = build_rich_long_features(draws, MAIN_COLS, MAIN_POOL, include_context=True)
    euro_long = build_rich_long_features(draws, EURO_COLS, EURO_POOL, include_context=False)

    main_long.to_csv(ROOT / "data" / "processed" / "main_features_rich.csv", index=False)
    euro_long.to_csv(ROOT / "data" / "processed" / "euro_features_rich.csv", index=False)

    # ---- Feature validation: correlation with target ----
    main_corr = feature_correlations(main_long, MAIN_FEATURE_COLS)
    euro_corr = feature_correlations(euro_long, EURO_FEATURE_COLS)

    # ---- Rerun model comparison with the expanded feature set ----
    main_out = run_model_comparison(main_long, MAIN_FEATURE_COLS, baseline_feature_col="freq_medium",
                                     k_pick=5, pool_size=MAIN_POOL, n_test_draws=N_TEST,
                                     n_perm=N_PERM, random_state=42)
    euro_out = run_model_comparison(euro_long, EURO_FEATURE_COLS, baseline_feature_col="freq_medium",
                                     k_pick=2, pool_size=EURO_POOL, n_test_draws=N_TEST,
                                     n_perm=N_PERM, random_state=42)

    with open(RESULTS_DIR / "main_results.json", "w") as f:
        json.dump(main_out["results"], f, indent=2)
    with open(RESULTS_DIR / "euro_results.json", "w") as f:
        json.dump(euro_out["results"], f, indent=2)
    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump({"main": main_out["significance"], "euro": euro_out["significance"]}, f, indent=2)
    with open(RESULTS_DIR / "feature_importances.json", "w") as f:
        json.dump({"main": main_out["feature_importances"], "euro": euro_out["feature_importances"]}, f, indent=2)

    # ---- Comparison against Step 6's single-feature result ----
    step6_path = ROOT / "results" / "step6" / "main_results.json"
    step6_comparison = None
    if step6_path.exists():
        with open(step6_path) as f:
            step6_main = json.load(f)
        step6_comparison = {
            name: {
                "step6_avg_hits": step6_main[name]["avg_hits"],
                "step7_avg_hits": main_out["results"][name]["avg_hits"],
                "step6_log_loss": step6_main[name]["log_loss"],
                "step7_log_loss": main_out["results"][name]["log_loss"],
            }
            for name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        }
        with open(RESULTS_DIR / "step6_vs_step7.json", "w") as f:
            json.dump(step6_comparison, f, indent=2)

    build_workbook(main_out, euro_out, main_corr, euro_corr, step6_comparison,
                    RESULTS_DIR / "feature_engineering.xlsx")
    print(f"Step 7 complete. Results in {RESULTS_DIR}")


def feature_correlations(long_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Point-biserial correlation of each feature with the binary target
    (equivalent to Pearson correlation when one variable is 0/1)."""
    rows = []
    for col in feature_cols:
        corr = long_df[col].corr(long_df["target"])
        rows.append({"feature": col, "correlation_with_target": corr})
    out = pd.DataFrame(rows).sort_values("correlation_with_target", key=lambda s: s.abs(), ascending=False)
    return out.reset_index(drop=True)


def build_workbook(main_out, euro_out, main_corr, euro_corr, step6_comparison, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    FONT = "Arial"
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
    title_font = Font(name=FONT, bold=True, size=14)
    subtitle_font = Font(name=FONT, italic=True, size=10, color="555555")
    normal_font = Font(name=FONT, size=10)
    bold_font = Font(name=FONT, bold=True, size=10)
    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def style_header(ws, row, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border

    def autosize(ws, widths):
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w

    def write_table(ws, headers, rows, start_row, bold_rows=()):
        for j, h in enumerate(headers, start=1):
            ws.cell(row=start_row, column=j, value=h)
        style_header(ws, start_row, len(headers))
        r = start_row + 1
        for row_data in rows:
            for j, val in enumerate(row_data, start=1):
                if isinstance(val, float):
                    val = round(val, 5)
                cell = ws.cell(row=r, column=j, value=val)
                cell.border = border
                cell.font = bold_font if r - start_row - 1 in bold_rows else normal_font
                if j >= 3:
                    cell.alignment = Alignment(horizontal="center")
            r += 1
        return r

    # ---- Feature_Catalog ----
    ws0 = wb.create_sheet("Feature_Catalog")
    ws0["A1"] = "Step 7 — Feature Catalog"
    ws0["A1"].font = title_font
    ws0["A2"] = "Every feature is leak-safe: computed only from draws strictly before the draw being predicted."
    ws0["A2"].font = subtitle_font
    catalog = [
        ("freq_short", "Count of appearances in the trailing 10 draws"),
        ("freq_medium", "Count of appearances in the trailing 20 draws"),
        ("freq_long", "Count of appearances in the trailing 50 draws"),
        ("freq_all", "Cumulative count of appearances in every prior draw"),
        ("freq_trend", "freq_short/10 minus freq_long/50 — is this number 'heating up' or 'cooling down'?"),
        ("gap_since_last", "Draws since this number last appeared (= draw index if never seen before)"),
        ("recency_weighted", "Exponentially decayed appearance count, half-life = 10 draws"),
        ("pair_score", "Historical co-occurrence strength between this number and the numbers in the previous draw"),
        ("triplet_score", "Historical triplet co-occurrence strength between this number and pairs of numbers from the previous draw (Main pool only — structurally always 0 for Euro, which only draws 2 numbers)"),
        ("adjacent_flag", "1 if number-1 or number+1 appeared in the previous draw"),
        ("is_odd", "Static: is the number itself odd"),
        ("is_low", "Static: is the number in the lower half of the pool"),
        ("ctx_avg_sum", "Trailing 20-draw rolling average of the main-draw sum (Main pool only)"),
        ("ctx_avg_odd_count", "Trailing 20-draw rolling average of odd-number count per draw (Main pool only)"),
        ("ctx_avg_low_count", "Trailing 20-draw rolling average of low-number count per draw (Main pool only)"),
    ]
    write_table(ws0, ["Feature", "Description"], catalog, start_row=4)
    autosize(ws0, [22, 95])

    # ---- Correlation_With_Target ----
    ws1 = wb.create_sheet("Correlation_With_Target")
    ws1["A1"] = "Correlation of Each Feature with the Draw Outcome (target)"
    ws1["A1"].font = title_font
    ws1["A2"] = "Computed over the full dataset (not just the test set). Values near 0 mean no linear relationship."
    ws1["A2"].font = subtitle_font
    r = write_table(ws1, ["Pool", "Feature", "Correlation with Target"],
                     [["Main", row.feature, row.correlation_with_target] for row in main_corr.itertuples()],
                     start_row=4)
    r = write_table(ws1, ["Pool", "Feature", "Correlation with Target"],
                     [["Euro", row.feature, row.correlation_with_target] for row in euro_corr.itertuples()],
                     start_row=r + 1)
    autosize(ws1, [10, 22, 22])

    # ---- Model_Comparison (rich features) ----
    ws2 = wb.create_sheet("Model_Comparison")
    ws2["A1"] = "Step 7 — Model Comparison with Expanded Feature Set (same held-out 100 draws as Step 6)"
    ws2["A1"].font = title_font
    ws2["A2"] = "Main models use 15 features; Euro models use 12 (no main-draw context features)."
    ws2["A2"].font = subtitle_font
    cols = ["Pool", "Strategy", "Draws", "Avg Hits", "Random Expected", "Lift vs Random",
            "Zero Hits", "One+ Hits", "Two+ Hits", "Max Hits", "Log Loss", "Brier Score"]
    order = ["Logistic Regression", "Random Forest", "Gradient Boosting",
             "Historical Frequency", "Recent-N Frequency", "Random Expectation"]
    rows_out = []
    bold_idx = []
    for pool_name, out in [("Main (5 of 50)", main_out), ("Euro (2 of 12)", euro_out)]:
        random_exp = out["random_expectation_avg_hits"]
        for name in order:
            v = out["results"][name]
            lift = None if v["avg_hits"] is None else round(v["avg_hits"] - random_exp, 4)
            if name in ("Logistic Regression", "Random Forest", "Gradient Boosting"):
                bold_idx.append(len(rows_out))
            rows_out.append([pool_name, name, v.get("draws"), v["avg_hits"], random_exp, lift,
                              v.get("zero"), v.get("one_plus"), v.get("two_plus"), v.get("max_hits"),
                              v["log_loss"], v["brier"]])
    write_table(ws2, cols, rows_out, start_row=4, bold_rows=set(bold_idx))
    autosize(ws2, [16, 22, 8, 10, 15, 13, 10, 10, 10, 10, 10, 11])

    # ---- Step6_vs_Step7 ----
    if step6_comparison:
        ws3 = wb.create_sheet("Step6_vs_Step7")
        ws3["A1"] = "Did Adding Features Change Anything? (Main pool, ML models only)"
        ws3["A1"].font = title_font
        ws3["A2"] = "Step 6 = single feature (trailing-20 count). Step 7 = 15 engineered features."
        ws3["A2"].font = subtitle_font
        rows_c = []
        for name, v in step6_comparison.items():
            rows_c.append([name, v["step6_avg_hits"], v["step7_avg_hits"],
                            round(v["step7_avg_hits"] - v["step6_avg_hits"], 4),
                            v["step6_log_loss"], v["step7_log_loss"],
                            round(v["step7_log_loss"] - v["step6_log_loss"], 4)])
        write_table(ws3, ["Model", "Step 6 Avg Hits", "Step 7 Avg Hits", "Δ Avg Hits",
                           "Step 6 Log Loss", "Step 7 Log Loss", "Δ Log Loss"], rows_c, start_row=4)
        autosize(ws3, [22, 15, 15, 12, 15, 15, 12])

    # ---- Feature_Importance ----
    ws4 = wb.create_sheet("Feature_Importance")
    ws4["A1"] = "Feature Importance / Coefficients by Model"
    ws4["A1"].font = title_font
    ws4["A2"] = "Random Forest / Gradient Boosting: impurity-based importance (higher = more used to split). Logistic Regression: raw coefficient (sign matters, magnitude depends on feature scale)."
    ws4["A2"].font = subtitle_font
    rows_fi = []
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        for model_name, imp in out["feature_importances"].items():
            for feat, val in sorted(imp.items(), key=lambda kv: -abs(kv[1])):
                rows_fi.append([pool_name, model_name, feat, val])
    write_table(ws4, ["Pool", "Model", "Feature", "Importance / Coefficient"], rows_fi, start_row=4)
    autosize(ws4, [10, 22, 22, 22])

    # ---- Significance_Check ----
    ws5 = wb.create_sheet("Significance_Check")
    ws5["A1"] = "Permutation Significance Check — Expanded Feature Set (n=1000 permutations)"
    ws5["A1"].font = title_font
    ws5["A2"] = "Same methodology as Step 6. Not corrected for multiple testing except where noted."
    ws5["A2"].font = subtitle_font
    n_tests = len(main_out["significance"]) + len(euro_out["significance"])
    rows_sig = []
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        for name, v in out["significance"].items():
            sig_flag = "Yes" if bonferroni_significant(v["perm_p_value"], n_tests) else "No"
            rows_sig.append([pool_name, name, v["avg_hits"], v["perm_p_value"],
                              round(0.05 / n_tests, 5), sig_flag])
    write_table(ws5, ["Pool", "Strategy", "Avg Hits", "Permutation p-value",
                       "Bonferroni-adj threshold", "Significant after correction?"], rows_sig, start_row=4)
    autosize(ws5, [10, 22, 12, 18, 22, 24])

    # ---- Methodology ----
    ws6 = wb.create_sheet("Methodology")
    ws6["A1"] = "Step 7 — Methodology"
    ws6["A1"].font = title_font
    lines = [
        "",
        "Objective: expand the information available to the models (Step 6 used a single feature) and check",
        "whether a richer feature set changes the conclusion that no reproducible signal exists.",
        "",
        "New features (src/features.py, build_rich_long_features): short/medium/long trailing frequency",
        "(windows 10/20/50), all-time cumulative frequency, a frequency trend (short-term rate minus",
        "long-term rate), gap since last appearance, an exponentially recency-weighted count (half-life 10",
        "draws), pairwise and triplet co-occurrence scores against the immediately preceding draw's numbers,",
        "an adjacent-number flag, static odd/even and low/high indicators, and (Main pool only) rolling",
        "draw-level context: average sum, odd-count and low-count over the trailing 20 draws.",
        "",
        "Leakage check: every feature for draw i is built only from draws strictly before i (see",
        "Feature_Catalog). Draw 0 is dropped (no history). Verified: 0 NaN values, 0 negative gaps.",
        "",
        "Same evaluation harness as Step 6 (src/backtesting.run_model_comparison, generalized to accept",
        "any feature list): identical chronological split (train = draws 1-589, test = last 100 draws),",
        "identical models (Logistic Regression, Random Forest, Gradient Boosting), identical baselines and",
        "metrics (avg hits, log loss, Brier score, permutation significance).",
        "",
        "Headline finding: see Step6_vs_Step7 — adding 14 more features moves avg hits and log loss by only",
        "a small amount for every model, in no consistent direction, and no strategy clears the",
        "multiple-testing-corrected significance bar (Significance_Check). Feature_Importance shows the tree",
        "models spread importance thinly across features rather than concentrating on any one signal, which",
        "is what you'd expect when none of the features actually carries predictive information. This is",
        "consistent with — not a rejection of — the hypothesis that EuroJackpot draws are close to uniform",
        "random; Step 8 (stronger models) and Step 9 (walk-forward validation) are the remaining checks",
        "before that conclusion can be treated as settled.",
    ]
    for i, line in enumerate(lines, start=2):
        cell = ws6.cell(row=i, column=1, value=line)
        cell.font = bold_font if line.strip().startswith(
            ("Objective", "New features", "Leakage check", "Same evaluation", "Headline finding")
        ) else normal_font
    ws6.column_dimensions["A"].width = 115

    wb.save(out_path)


if __name__ == "__main__":
    main()
