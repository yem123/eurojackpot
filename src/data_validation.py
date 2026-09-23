"""
data_validation.py

Loads raw EuroJackpot draw data and validates it before anything downstream
(features, models, backtests) is allowed to touch it.

Canonical clean draw format (one row per draw):
    draw_date, main_1, main_2, main_3, main_4, main_5, euro_1, euro_2

Rules enforced:
    - main_1..main_5 are 5 distinct integers in [1, 50]
    - euro_1, euro_2 are 2 distinct integers in [1, 12]
    - draw_date is a valid date, draws are chronologically ordered, no duplicate dates
    - no missing values in the required columns
"""

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

MAIN_COLS = [f"main_{i}" for i in range(1, 6)]
EURO_COLS = [f"euro_{i}" for i in range(1, 3)]
REQUIRED_COLS = ["draw_date"] + MAIN_COLS + EURO_COLS

MAIN_POOL = 50
EURO_POOL = 12


@dataclass
class ValidationResult:
    is_valid: bool
    n_rows: int
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Rows checked: {self.n_rows}", f"Valid: {self.is_valid}"]
        if self.errors:
            lines.append(f"Errors ({len(self.errors)}):")
            lines += [f"  - {e}" for e in self.errors]
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            lines += [f"  - {w}" for w in self.warnings]
        return "\n".join(lines)


def load_clean_draws(path: str | Path, sheet_name: str = "Clean_Draws") -> pd.DataFrame:
    """Load the canonical clean-draws table from an Excel workbook."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df = df[REQUIRED_COLS].copy()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df = df.sort_values("draw_date").reset_index(drop=True)
    return df


def validate_draws(df: pd.DataFrame) -> ValidationResult:
    """Run integrity checks on a clean-draws DataFrame. Never raises; returns a report."""
    errors, warnings = [], []

    for col in REQUIRED_COLS:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    if errors:
        return ValidationResult(False, len(df), errors, warnings)

    if df[REQUIRED_COLS].isna().any().any():
        bad = df[df[REQUIRED_COLS].isna().any(axis=1)]
        errors.append(f"{len(bad)} row(s) with missing values, e.g. dates: "
                       f"{bad['draw_date'].head(5).tolist()}")

    for i, row in df.iterrows():
        mains = row[MAIN_COLS].tolist()
        if len(set(mains)) != 5:
            errors.append(f"Row {i} ({row['draw_date'].date()}): main numbers not 5 distinct values: {mains}")
        if any(not (1 <= m <= MAIN_POOL) for m in mains):
            errors.append(f"Row {i} ({row['draw_date'].date()}): main number out of range 1-{MAIN_POOL}: {mains}")

        euros = row[EURO_COLS].tolist()
        if len(set(euros)) != 2:
            errors.append(f"Row {i} ({row['draw_date'].date()}): euro numbers not 2 distinct values: {euros}")
        if any(not (1 <= e <= EURO_POOL) for e in euros):
            errors.append(f"Row {i} ({row['draw_date'].date()}): euro number out of range 1-{EURO_POOL}: {euros}")

    if df["draw_date"].duplicated().any():
        dupes = df.loc[df["draw_date"].duplicated(), "draw_date"].tolist()
        errors.append(f"Duplicate draw dates: {dupes}")

    if not df["draw_date"].is_monotonic_increasing:
        warnings.append("draw_date is not sorted ascending (will be sorted automatically downstream).")

    gaps = df["draw_date"].diff().dt.days.dropna()
    unusual_gaps = gaps[(gaps < 2) | (gaps > 10)]
    if len(unusual_gaps) > 0:
        warnings.append(f"{len(unusual_gaps)} draw(s) with an unusual gap (<2 or >10 days) since the prior draw; "
                         "EuroJackpot is normally weekly. This can be legitimate (rule-change periods) but is worth eyeballing.")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid, len(df), errors, warnings)


def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "eurojackpot_uploaded_clean_2018_2026.xlsx"
    out_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_draws.csv"

    draws = load_clean_draws(raw_path)
    result = validate_draws(draws)
    print(result.summary())
    if result.is_valid:
        save_processed(draws, out_path)
        print(f"Saved validated draws -> {out_path}")
    else:
        raise SystemExit("Validation failed — see errors above. Not writing processed data.")
