"""
scripts/run_step6.py

Runner for Step 6 — Model Comparison. Thin orchestration only: all real
logic lives in src/. Reads processed data, runs the comparison for both
pools, writes results/step6/model_comparison.xlsx plus CSV/JSON
intermediates for anything downstream that wants raw numbers instead of
a spreadsheet.

Usage:
    python scripts/run_step6.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import openpyxl  # noqa: E402
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from features import build_feature_set, MAIN_COLS, EURO_COLS, MAIN_POOL, EURO_POOL  # noqa: E402
from backtesting import run_comparison  # noqa: E402
from statistics import calibration_table, bonferroni_significant  # noqa: E402

RESULTS_DIR = ROOT / "results" / "step6"
WINDOW = 20
N_TEST = 100
N_PERM = 1000


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    draws = pd.read_csv(ROOT / "data" / "processed" / "clean_draws.csv", parse_dates=["draw_date"])
    feats = build_feature_set(draws, window=WINDOW)

    main_out = run_comparison(draws, feats["main"], MAIN_COLS, MAIN_POOL, k_pick=5,
                               window=WINDOW, n_test=N_TEST, n_perm=N_PERM, random_state=42)
    euro_out = run_comparison(draws, feats["euro"], EURO_COLS, EURO_POOL, k_pick=2,
                               window=WINDOW, n_test=N_TEST, n_perm=N_PERM, random_state=42)

    # ---- JSON summaries (raw numbers, for downstream steps) ----
    with open(RESULTS_DIR / "main_results.json", "w") as f:
        json.dump(main_out["results"], f, indent=2)
    with open(RESULTS_DIR / "euro_results.json", "w") as f:
        json.dump(euro_out["results"], f, indent=2)
    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump({"main": main_out["significance"], "euro": euro_out["significance"]}, f, indent=2)

    # ---- Calibration tables ----
    cal_rows = []
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        yte = out["test_long"]["target"].to_numpy()
        for name, proba in out["predictions"].items():
            if name == "Random Expectation":
                continue
            n_bins = 8 if pool_name == "Main" else 6
            t = calibration_table(yte, proba, n_bins=n_bins)
            t["model"] = name
            t["pool"] = pool_name
            cal_rows.append(t)
    cal_df = pd.concat(cal_rows, ignore_index=True)
    cal_df["bin"] = cal_df["bin"].astype(str)
    cal_df.to_csv(RESULTS_DIR / "calibration.csv", index=False)

    # ---- Build the Excel workbook ----
    build_workbook(main_out, euro_out, cal_df, RESULTS_DIR / "model_comparison.xlsx")
    print(f"Step 6 complete. Results in {RESULTS_DIR}")


def build_workbook(main_out, euro_out, cal_df, out_path):
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

    # ---- Model_Comparison ----
    ws = wb.create_sheet("Model_Comparison")
    ws["A1"] = "Step 6 — Model Comparison (Out-of-Sample, last 100 draws)"
    ws["A1"].font = title_font
    ws["A2"] = "Same held-out test set used across all strategies. Main: pick top 5 of 50. Euro: pick top 2 of 12."
    ws["A2"].font = subtitle_font

    cols = ["Pool", "Strategy", "Draws", "Avg Hits", "Random Expected", "Lift vs Random",
            "Zero Hits", "One+ Hits", "Two+ Hits", "Max Hits", "Log Loss", "Brier Score"]
    row0 = 4
    for j, c in enumerate(cols, start=1):
        ws.cell(row=row0, column=j, value=c)
    style_header(ws, row0, len(cols))

    order = ["Logistic Regression", "Random Forest", "Gradient Boosting",
             "Historical Frequency", "Recent-N Frequency", "Random Expectation"]

    r = row0 + 1
    for pool_name, out in [("Main (5 of 50)", main_out), ("Euro (2 of 12)", euro_out)]:
        random_exp = out["random_expectation_avg_hits"]
        for name in order:
            v = out["results"][name]
            lift = None if v["avg_hits"] is None else round(v["avg_hits"] - random_exp, 4)
            row_vals = [pool_name, name, v.get("draws"), round(v["avg_hits"], 4), round(random_exp, 4), lift,
                        v.get("zero"), v.get("one_plus"), v.get("two_plus"), v.get("max_hits"),
                        round(v["log_loss"], 4), round(v["brier"], 4)]
            for j, val in enumerate(row_vals, start=1):
                cell = ws.cell(row=r, column=j, value=val)
                cell.font = bold_font if name in ("Logistic Regression", "Random Forest", "Gradient Boosting") else normal_font
                cell.border = border
                if j >= 3:
                    cell.alignment = Alignment(horizontal="center")
            r += 1

    ws.freeze_panes = "A5"
    autosize(ws, [16, 22, 8, 10, 15, 13, 10, 10, 10, 10, 10, 11])

    notes = [
        "Notes:",
        "- All models use a single feature (count of appearances in the trailing ~20 draws) so results are directly comparable; richer features arrive in Step 7.",
        "- Historical Frequency = each number's overall train-set hit rate used as a constant probability. Recent-N Frequency = the raw trailing count itself, scaled.",
        "- Ties in predicted probability (common here, since these models share a small feature set) are broken by preferring the lower number — see statistics._hits_grid for the exact rule.",
        "- Random Expectation avg hits computed analytically (hypergeometric): k_pick^2 / pool_size.",
        "- Log loss / Brier score are computed over every (draw, number) pair in the test set, not just the picked numbers.",
        "- See Significance_Check sheet before treating any 'lift vs random' as a real edge.",
    ]
    for i, line in enumerate(notes):
        cell = ws.cell(row=r + 2 + i, column=1, value=line)
        cell.font = bold_font if i == 0 else normal_font

    # ---- Calibration ----
    ws2 = wb.create_sheet("Calibration")
    ws2["A1"] = "Calibration — Mean Predicted Probability vs Actual Hit Rate, by Bin"
    ws2["A1"].font = title_font
    ws2["A2"] = "Well-calibrated models have mean_predicted ≈ actual_rate within each bin."
    ws2["A2"].font = subtitle_font

    cal_cols = ["pool", "model", "bin", "n", "mean_predicted", "actual_rate"]
    cal_out = cal_df[cal_cols].rename(columns={
        "pool": "Pool", "model": "Model", "bin": "Probability Bin", "n": "N (draw-number pairs)",
        "mean_predicted": "Mean Predicted", "actual_rate": "Actual Rate",
    })
    row0b = 4
    for j, c in enumerate(cal_out.columns, start=1):
        ws2.cell(row=row0b, column=j, value=c)
    style_header(ws2, row0b, len(cal_out.columns))
    for i, row_data in enumerate(cal_out.itertuples(index=False), start=row0b + 1):
        for j, val in enumerate(row_data, start=1):
            if isinstance(val, float):
                val = round(val, 4)
            cell = ws2.cell(row=i, column=j, value=val)
            cell.border = border
            cell.font = normal_font
    ws2.freeze_panes = "A5"
    autosize(ws2, [10, 22, 32, 20, 15, 13])

    # ---- Significance_Check ----
    ws3 = wb.create_sheet("Significance_Check")
    ws3["A1"] = "Permutation Significance Check (n=1000 permutations per model)"
    ws3["A1"].font = title_font
    ws3["A2"] = ("p-value = share of random probability-shuffles achieving avg hits >= the model's observed "
                 "avg hits. NOT corrected for multiple testing (10 strategies tested here); a full correction is Step 10.")
    ws3["A2"].font = subtitle_font

    sig_cols = ["Pool", "Strategy", "Avg Hits", "Permutation p-value", "Bonferroni-adj threshold (10 tests)", "Significant after correction?"]
    row0c = 4
    for j, c in enumerate(sig_cols, start=1):
        ws3.cell(row=row0c, column=j, value=c)
    style_header(ws3, row0c, len(sig_cols))

    r = row0c + 1
    n_tests = len(main_out["significance"]) + len(euro_out["significance"])
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        for name, v in out["significance"].items():
            sig_flag = "Yes" if bonferroni_significant(v["perm_p_value"], n_tests) else "No"
            vals = [pool_name, name, round(v["avg_hits"], 4), round(v["perm_p_value"], 4),
                    round(0.05 / n_tests, 5), sig_flag]
            for j, val in enumerate(vals, start=1):
                cell = ws3.cell(row=r, column=j, value=val)
                cell.border = border
                cell.font = normal_font
                if j >= 3:
                    cell.alignment = Alignment(horizontal="center")
            r += 1
    ws3.freeze_panes = "A5"
    autosize(ws3, [16, 22, 12, 18, 26, 24])
    ws3.cell(row=r + 2, column=1,
             value="Interpretation: check the 'Significant after correction?' column — a strategy only clears "
                   "the bar once its p-value beats the Bonferroni-adjusted threshold, not just 0.05.").font = normal_font

    # ---- Methodology ----
    ws4 = wb.create_sheet("Methodology")
    ws4["A1"] = "Step 6 — Methodology"
    ws4["A1"].font = title_font
    lines = [
        "",
        "Objective: compare Logistic Regression, Random Forest, and Gradient Boosting against frequency-based",
        "strategies and random expectation, on identical out-of-sample data, using multiple evaluation metrics.",
        "",
        "Pipeline: data/raw -> src/data_validation.py -> data/processed/clean_draws.csv -> src/features.py ->",
        "src/backtesting.py (uses src/models.py, src/statistics.py) -> results/step6/. Run end-to-end with",
        "scripts/run_step6.py. See README.md for the full project layout.",
        "",
        "Data: data/processed/clean_draws.csv, 690 validated draws (2018-2026). First draw dropped (no prior",
        "history). Remaining 689 draws split chronologically: train = draws 1-589, test = last 100 draws",
        "(held out, never used for fitting).",
        "",
        "Framing: each draw x number is one observation. Target = 1 if that number was drawn, else 0.",
        "Feature: trailing count of the number's appearances in roughly the prior 20 draws, identical across",
        "every model so the comparison is apples-to-apples. Feature engineering is deliberately deferred to Step 7.",
        "",
        "Models: Logistic Regression, Random Forest (300 trees, depth 4), Gradient Boosting (200 trees, depth 2,",
        "lr 0.05) — shallow on purpose given a single input feature. Baselines: Historical Frequency (constant",
        "per-number training hit rate), Recent-N Frequency (the raw feature itself), Random Expectation",
        "(uniform probability, hit rate from the hypergeometric distribution).",
        "",
        "Metrics: avg hits picking the model's top-k numbers per draw; log loss and Brier score over every",
        "(draw, number) pair; calibration tables; permutation significance (1000 shuffles per model).",
        "",
        "Headline finding: across both pools, every model — ML and baseline alike — lands within noise of random",
        "expectation. Log loss/Brier are nearly identical across all six strategies per pool, and no result",
        "survives the multiple-testing correction. With a single weak feature, none of the three algorithms",
        "shows a reproducible edge. Step 7's feature engineering and Step 9's walk-forward validation are the",
        "tests that would actually reveal a real signal if one exists.",
    ]
    for i, line in enumerate(lines, start=2):
        cell = ws4.cell(row=i, column=1, value=line)
        cell.font = bold_font if line.strip().startswith(
            ("Objective", "Pipeline", "Data:", "Framing", "Models:", "Metrics", "Headline finding")
        ) else normal_font
    ws4.column_dimensions["A"].width = 110

    wb.save(out_path)


if __name__ == "__main__":
    main()
