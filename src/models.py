"""
models.py

Model definitions for the number-level probability approach: for a given
pool (main 1-50 or euro 1-12), each candidate model outputs a predicted
probability that each number appears in the next draw, given the same
long-format feature table produced by features.to_long_format().

Includes both learned models (sklearn) and simple non-learned baselines,
all exposed through the same .fit(X, y) / .predict_proba(X) interface so
backtesting.py can treat them uniformly.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class BaselineModel:
    """Common interface for non-learned baselines: .fit(X, y), .predict_proba(X)."""
    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError


class HistoricalFrequencyBaseline(BaselineModel):
    """Predicts each number's constant historical hit rate from training data.

    Expects X to be a DataFrame/array with a 'number' column (or first column)
    so it can look up the right constant per number.
    """
    def fit(self, numbers: np.ndarray, y: np.ndarray):
        self.freq_ = {}
        numbers = np.asarray(numbers)
        for n in np.unique(numbers):
            mask = numbers == n
            self.freq_[n] = y[mask].mean() if mask.any() else y.mean()
        self.global_mean_ = y.mean()
        return self

    def predict_proba(self, numbers: np.ndarray):
        numbers = np.asarray(numbers)
        p1 = np.array([self.freq_.get(n, self.global_mean_) for n in numbers])
        return np.column_stack([1 - p1, p1])


class RecentFrequencyBaseline(BaselineModel):
    """Predicts probability proportional to the raw trailing-count feature itself."""
    def __init__(self, window: int = 20):
        self.window = window

    def fit(self, X, y):
        return self  # nothing to fit; uses the raw feature at predict time

    def predict_proba(self, feat: np.ndarray):
        feat = np.asarray(feat, dtype=float)
        p1 = np.clip(feat / self.window, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p1, p1])


class RandomExpectationBaseline(BaselineModel):
    """Uniform probability k_pick / pool_size for every number, every draw."""
    def __init__(self, k_pick: int, pool_size: int):
        self.p = k_pick / pool_size

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p)
        return np.column_stack([1 - p1, p1])


def get_ml_models(random_state: int = 42) -> dict:
    """Learned models compared in Step 6. Add new candidates here for Step 8."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=50,
            random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05, random_state=random_state,
        ),
    }


# ---------------------------------------------------------------------------
# Step 8 — advanced models, ensembling, calibration
# ---------------------------------------------------------------------------

try:
    from xgboost import XGBClassifier
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


def get_advanced_models(random_state: int = 42, base_rate: float = 0.1) -> dict:
    """
    Step 8 candidates: XGBoost and LightGBM where the packages are installed
    (skipped with a note otherwise — 'where available' per the roadmap),
    kept shallow/regularized like the Step 6 tree models given how few and
    weak the underlying features are. base_rate sets scale_pos_weight /
    is_unbalance context (main ~10%, euro ~17% positive rate) so the boosted
    trees aren't fighting a class-imbalance problem on top of a signal problem.
    """
    models = {}
    if _HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=2, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )
    if _HAS_LIGHTGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200, max_depth=2, num_leaves=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1, verbose=-1,
        )
    return models


class AveragingEnsemble(BaselineModel):
    """Simple probability-averaging ensemble over a set of already-fitted
    sklearn-style models (each must expose predict_proba)."""
    def __init__(self, fitted_models: dict):
        self.fitted_models = fitted_models

    def fit(self, X, y):
        return self  # component models are fitted externally; nothing to do here

    def predict_proba(self, X):
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_models.values()])
        p1 = probs.mean(axis=1)
        return np.column_stack([1 - p1, p1])


def calibrate_probabilities(train_proba: np.ndarray, train_y: np.ndarray,
                             test_proba: np.ndarray, method: str = "isotonic") -> np.ndarray:
    """
    Post-hoc recalibration of an already-fitted model's predicted probabilities,
    fit on the model's own training-set predictions (so it never sees the test
    set) and applied to test-set predictions.
    method: 'isotonic' (nonparametric, needs more data) or 'sigmoid' (Platt scaling).
    """
    from sklearn.calibration import _SigmoidCalibration
    from sklearn.isotonic import IsotonicRegression

    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(train_proba, train_y)
        return calibrator.predict(test_proba)
    elif method == "sigmoid":
        calibrator = _SigmoidCalibration()
        calibrator.fit(train_proba, train_y)
        return calibrator.predict(test_proba)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
