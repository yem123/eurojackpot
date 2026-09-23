"""
scripts/run_step8.py

Runner for Step 8 — Advanced ML, run as a PREDEFINED experiment:

  1. Data is split train / validation / holdout (src/splits.py). The holdout
     (draw_idx 590-689 — numerically the same draws Steps 6-7 called "test")
     is carved out up front and never loaded into any model-fitting or
     scoring call in this script. src/backtesting.run_step8_validation()
     asserts this at the code level, not just by convention.

  2. Selection criterion is fixed BEFORE looking at any result:
        primary   = lowest validation log loss among the six learned
                    strategies (Logistic Regression, Random Forest,
                    Gradient Boosting, XGBoost, LightGBM, Averaging Ensemble)
        tie-break = higher validation permutation p-value is WORSE, i.e.
                    prefer the model whose validation result is LEAST
                    consistent with random chance (lowest p-value), but only
                    as a tie-break on log loss, not as the primary criterion
                    (avg hits over 100 draws is too noisy to select on).
     This is written down here, in the script, before results exist.

  3. The winning model per pool is frozen via experiment_log.freeze_model_spec
     — an immutable record of which model, which hyperparameters, and which
     validation metrics justified the choice. No later step may use the
     holdout for this pool without that frozen spec existing on disk.

  4. This script produces VALIDATION results only. There is no holdout
     number anywhere in its output. A separate, later, one-time run against
     the holdout (once specs for both pools are frozen) is a distinct script
     by design, so "run the comparison" and "touch the holdout" can never be
     the same action.

Usage:
    python scripts/run_step8.py
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

from features import (  # noqa: E402
    build_rich_long_features, FROZEN_MAIN_FEATURES, FROZEN_EURO_FEATURES, FEATURE_SPEC_VERSION,
    MAIN_COLS, EURO_COLS, MAIN_POOL, EURO_POOL,
)
from backtesting import run_step8_validation  # noqa: E402
from statistics import bonferroni_significant  # noqa: E402
from models import _HAS_XGBOOST, _HAS_LIGHTGBM  # noqa: E402
from experiment_log import ModelSpec, freeze_model_spec  # noqa: E402

RESULTS_DIR = ROOT / "results" / "step8"
VALIDATION_DRAWS = 100
HOLDOUT_DRAWS = 100
N_PERM = 1000

LEARNED_MODELS = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Averaging Ensemble"]
BASELINE_MODELS = ["Historical Frequency", "Recent-N Frequency", "Random Expectation"]
MODEL_ORDER = LEARNED_MODELS + BASELINE_MODELS

HYPERPARAMS = {
    "Logistic Regression": {"max_iter": 1000},
    "Random Forest": {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 50},
    "Gradient Boosting": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05},
    "XGBoost": {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "LightGBM": {"n_estimators": 200, "max_depth": 2, "num_leaves": 7, "learning_rate": 0.05, "subsample": 0.8},
    "Averaging Ensemble": {"components": LEARNED_MODELS[:5]},
}


def select_winner(results: dict, significance: dict) -> tuple[str, str]:
    """Predefined selection rule (see module docstring). Returns (winner, note)."""
    candidates = [m for m in LEARNED_MODELS if m in results]
    ranked = sorted(candidates, key=lambda m: results[m]["log_loss"])
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    note = f"Selected by lowest validation log loss ({results[winner]['log_loss']:.5f})."
    if runner_up is not None:
        gap = results[runner_up]["log_loss"] - results[winner]["log_loss"]
        note += f" Runner-up: {runner_up} (log loss {results[runner_up]['log_loss']:.5f}, gap {gap:.5f})."
    if winner in significance:
        note += f" Validation permutation p-value: {significance[winner]['perm_p_value']:.4f}."
    return winner, note


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"XGBoost available: {_HAS_XGBOOST} | LightGBM available: {_HAS_LIGHTGBM}")
    print(f"Frozen feature spec: {FEATURE_SPEC_VERSION} "
          f"(Main: {len(FROZEN_MAIN_FEATURES)} features, Euro: {len(FROZEN_EURO_FEATURES)} features)")

    draws = pd.read_csv(ROOT / "data" / "processed" / "clean_draws.csv", parse_dates=["draw_date"])
    main_long = build_rich_long_features(draws, MAIN_COLS, MAIN_POOL, include_context=True)
    euro_long = build_rich_long_features(draws, EURO_COLS, EURO_POOL, include_context=False)

    main_out = run_step8_validation(main_long, FROZEN_MAIN_FEATURES, baseline_feature_col="freq_medium",
                                     k_pick=5, pool_size=MAIN_POOL, validation_draws=VALIDATION_DRAWS,
                                     holdout_draws=HOLDOUT_DRAWS, n_perm=N_PERM, random_state=42)
    euro_out = run_step8_validation(euro_long, FROZEN_EURO_FEATURES, baseline_feature_col="freq_medium",
                                     k_pick=2, pool_size=EURO_POOL, validation_draws=VALIDATION_DRAWS,
                                     holdout_draws=HOLDOUT_DRAWS, n_perm=N_PERM, random_state=42)

    print(f"Main:  train={main_out['train_size']} validation={main_out['validation_size']} "
          f"holdout={main_out['holdout_size']} (untouched)")
    print(f"Euro:  train={euro_out['train_size']} validation={euro_out['validation_size']} "
          f"holdout={euro_out['holdout_size']} (untouched)")

    with open(RESULTS_DIR / "main_validation_results.json", "w") as f:
        json.dump(main_out["results"], f, indent=2)
    with open(RESULTS_DIR / "euro_validation_results.json", "w") as f:
        json.dump(euro_out["results"], f, indent=2)
    with open(RESULTS_DIR / "main_validation_calibrated_results.json", "w") as f:
        json.dump(main_out["calibrated_results"], f, indent=2)
    with open(RESULTS_DIR / "euro_validation_calibrated_results.json", "w") as f:
        json.dump(euro_out["calibrated_results"], f, indent=2)
    with open(RESULTS_DIR / "significance.json", "w") as f:
        json.dump({"main": main_out["significance"], "euro": euro_out["significance"]}, f, indent=2)
    with open(RESULTS_DIR / "feature_importances.json", "w") as f:
        json.dump({"main": main_out["feature_importances"], "euro": euro_out["feature_importances"]}, f, indent=2)

    # ---- Predefined model selection (validation only) ----
    main_winner, main_note = select_winner(main_out["results"], main_out["significance"])
    euro_winner, euro_note = select_winner(euro_out["results"], euro_out["significance"])
    print(f"Main winner:  {main_winner} — {main_note}")
    print(f"Euro winner:  {euro_winner} — {euro_note}")

    for pool, out, winner, note in [("main", main_out, main_winner, main_note), ("euro", euro_out, euro_winner, euro_note)]:
        spec = ModelSpec(
            frozen=True,
            feature_spec_version=FEATURE_SPEC_VERSION,
            pool=pool,
            selected_model=winner,
            selection_criterion="lowest validation log loss (validation window = 100 draws, "
                                 "immediately preceding the untouched holdout); tie-break on "
                                 "permutation p-value, not used here",
            hyperparameters=HYPERPARAMS.get(winner, {}),
            validation_metrics=out["results"][winner],
            selection_notes=note,
        )
        freeze_model_spec(spec, overwrite=True)

    build_workbook(main_out, euro_out, main_winner, euro_winner, main_note, euro_note,
                    RESULTS_DIR / "advanced_ml_validation.xlsx")
    print(f"Step 8 complete. Results in {RESULTS_DIR}. Model spec frozen at results/model_spec.json.")
    print("Holdout (draw_idx 590-689) was not loaded, scored, or referenced anywhere in this run.")


def build_workbook(main_out, euro_out, main_winner, euro_winner, main_note, euro_note, out_path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    FONT = "Arial"
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(name=FONT, bold=True, color="FFFFFF", size=11)
    title_font = Font(name=FONT, bold=True, size=14)
    subtitle_font = Font(name=FONT, italic=True, size=10, color="555555")
    normal_font = Font(name=FONT, size=10)
    bold_font = Font(name=FONT, bold=True, size=10)
    winner_fill = PatternFill(start_color="D9EAD3", end_color="D9EAD3", fill_type="solid")
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

    def write_table(ws, headers, rows, start_row, highlight_values=()):
        for j, h in enumerate(headers, start=1):
            ws.cell(row=start_row, column=j, value=h)
        style_header(ws, start_row, len(headers))
        r = start_row + 1
        for row_data in rows:
            is_winner_row = len(row_data) > 1 and row_data[1] in highlight_values
            for j, val in enumerate(row_data, start=1):
                if isinstance(val, float):
                    val = round(val, 5)
                cell = ws.cell(row=r, column=j, value=val)
                cell.border = border
                cell.font = bold_font if is_winner_row else normal_font
                if is_winner_row:
                    cell.fill = winner_fill
                if j >= 3:
                    cell.alignment = Alignment(horizontal="center")
            r += 1
        return r

    # ---- Cover / discipline notice ----
    ws0 = wb.create_sheet("Read_Me_First")
    ws0["A1"] = "Step 8 — Validation-Only Model Comparison"
    ws0["A1"].font = title_font
    lines = [
        "",
        "IMPORTANT: everything in this workbook is computed on the VALIDATION window only",
        "(draw_idx 490-589, 100 draws). The HOLDOUT window (draw_idx 590-689, 100 draws — the",
        "same draws Steps 6-7 called 'test') was never loaded into any model in this run.",
        "src/backtesting.run_step8_validation() asserts this in code, not just in this note.",
        "",
        f"Main pool winner:  {main_winner}",
        f"  {main_note}",
        "",
        f"Euro pool winner:  {euro_winner}",
        f"  {euro_note}",
        "",
        "Both selections are now frozen in results/model_spec.json. Any future script that",
        "wants to evaluate on the holdout must call experiment_log.require_frozen_spec() for",
        "the relevant pool, which will succeed now that these specs exist — but the point of",
        "freezing is that the choice above will NOT be revisited after seeing holdout results.",
    ]
    for i, line in enumerate(lines, start=2):
        cell = ws0.cell(row=i, column=1, value=line)
        cell.font = bold_font if line.startswith(("IMPORTANT", "Main pool winner", "Euro pool winner")) else normal_font
    ws0.column_dimensions["A"].width = 100

    # ---- Validation_Comparison ----
    ws1 = wb.create_sheet("Validation_Comparison")
    ws1["A1"] = "Model Comparison — Validation Window Only (100 draws)"
    ws1["A1"].font = title_font
    ws1["A2"] = "Winning model per pool highlighted in green. Selection criterion: lowest validation log loss."
    ws1["A2"].font = subtitle_font
    cols = ["Pool", "Strategy", "Draws", "Avg Hits", "Random Expected", "Lift vs Random",
            "Zero Hits", "One+ Hits", "Two+ Hits", "Max Hits", "Log Loss", "Brier Score"]
    rows_out = []
    for pool_name, out in [("Main (5 of 50)", main_out), ("Euro (2 of 12)", euro_out)]:
        random_exp = out["random_expectation_avg_hits"]
        present = [m for m in MODEL_ORDER if m in out["results"]]
        for name in present:
            v = out["results"][name]
            lift = None if v["avg_hits"] is None else round(v["avg_hits"] - random_exp, 4)
            rows_out.append([pool_name, name, v.get("draws"), v["avg_hits"], random_exp, lift,
                              v.get("zero"), v.get("one_plus"), v.get("two_plus"), v.get("max_hits"),
                              v["log_loss"], v["brier"]])
    write_table(ws1, cols, rows_out, start_row=4, highlight_values={main_winner, euro_winner})
    autosize(ws1, [16, 22, 8, 10, 15, 13, 10, 10, 10, 10, 10, 11])

    # ---- Calibration_Effect ----
    ws2 = wb.create_sheet("Calibration_Effect")
    ws2["A1"] = "Effect of Post-Hoc Probability Calibration (isotonic, validation window)"
    ws2["A1"].font = title_font
    ws2["A2"] = "Calibrator fit on each model's own TRAINING predictions, applied to validation predictions."
    ws2["A2"].font = subtitle_font
    cols2 = ["Pool", "Model", "Avg Hits (raw)", "Avg Hits (calibrated)", "Log Loss (raw)", "Log Loss (calibrated)",
             "Δ Log Loss", "Brier (raw)", "Brier (calibrated)", "Δ Brier"]
    rows2 = []
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        present = [m for m in LEARNED_MODELS if m in out["results"]]
        for name in present:
            raw = out["results"][name]
            cal = out["calibrated_results"][name]
            rows2.append([pool_name, name, raw["avg_hits"], cal["avg_hits"],
                          raw["log_loss"], cal["log_loss"], round(cal["log_loss"] - raw["log_loss"], 5),
                          raw["brier"], cal["brier"], round(cal["brier"] - raw["brier"], 5)])
    write_table(ws2, cols2, rows2, start_row=4)
    autosize(ws2, [10, 20, 14, 18, 14, 18, 12, 12, 16, 12])

    # ---- Significance_Check ----
    ws3 = wb.create_sheet("Significance_Check")
    ws3["A1"] = "Permutation Significance Check — Validation Window (n=1000 permutations)"
    ws3["A1"].font = title_font
    ws3["A2"] = "Bonferroni-adjusted across all tests run this step. This is diagnostic, not the selection rule (see Read_Me_First)."
    ws3["A2"].font = subtitle_font
    n_tests = len(main_out["significance"]) + len(euro_out["significance"])
    rows3 = []
    for pool_name, out in [("Main", main_out), ("Euro", euro_out)]:
        for name, v in out["significance"].items():
            sig_flag = "Yes" if bonferroni_significant(v["perm_p_value"], n_tests) else "No"
            rows3.append([pool_name, name, v["avg_hits"], v["perm_p_value"], round(0.05 / n_tests, 5), sig_flag])
    write_table(ws3, ["Pool", "Strategy", "Avg Hits", "Permutation p-value", "Bonferroni-adj threshold",
                       "Significant after correction?"], rows3, start_row=4)
    autosize(ws3, [10, 22, 12, 18, 22, 24])

    # ---- Methodology ----
    ws4 = wb.create_sheet("Methodology")
    ws4["A1"] = "Step 8 — Methodology (Corrected Discipline)"
    ws4["A1"].font = title_font
    lines = [
        "",
        "Why this step looks different from Steps 6-7: those steps repeatedly evaluated every",
        "strategy against 'the last 100 draws', calling it a test set. That window was never used",
        "to tune hyperparameters or drop models, but it WAS scored 20 times across two steps (see",
        "results/EXPERIMENT_LEDGER.md). That is enough repeated exposure that it can't honestly be",
        "called an untouched final holdout anymore. Step 8 corrects course.",
        "",
        "New split (src/splits.py): train = draw_idx 1-489 (489 draws), validation = draw_idx",
        "490-589 (100 draws), holdout = draw_idx 590-689 (100 draws, same window Steps 6-7 used).",
        "Everything in this workbook is computed on validation only. The holdout is carved out",
        "before any model touches the data and is asserted untouched twice in",
        "backtesting.run_step8_validation().",
        "",
        "Selection criterion, fixed BEFORE computing results (see scripts/run_step8.py,",
        "select_winner()): the learned strategy with the lowest validation log loss wins. Avg hits",
        "is reported for context but is not the selection criterion — over 100 draws with 5-of-50",
        "or 2-of-12 picks, hit count is a very high-variance statistic; log loss uses every",
        "(draw, number) pair's probability, not just the top-k picks, so it is far less noisy.",
        "",
        "Models: identical to the first version of Step 8 — Logistic Regression, Random Forest,",
        "Gradient Boosting, XGBoost, LightGBM, and an unweighted probability-averaging ensemble",
        "over all five, plus the three Step 6/7 baselines. Same frozen Step 7 feature spec.",
        "",
        "Output: the winning model per pool is frozen to results/model_spec.json via",
        "experiment_log.freeze_model_spec(). That file is now the single source of truth for",
        "'which model won' — no later step may pick a different model for this pool without",
        "explicitly re-opening the question (freeze_model_spec(..., overwrite=True), which is",
        "deliberately not the default).",
        "",
        "What's still open: the holdout has not been touched. A final, one-time holdout",
        "evaluation of the frozen spec(s) is a separate future script by design — this keeps",
        "'compare models' and 'touch the holdout' from ever being the same action.",
    ]
    for i, line in enumerate(lines, start=2):
        cell = ws4.cell(row=i, column=1, value=line)
        cell.font = bold_font if line.startswith(
            ("Why this step", "New split", "Selection criterion", "Models:", "Output:", "What's still open")
        ) else normal_font
    ws4.column_dimensions["A"].width = 112

    wb.save(out_path)


if __name__ == "__main__":
    main()
