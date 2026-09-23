"""
experiment_log.py

Two jobs:

1. Model-spec freezing. Once Step 8's validation-only comparison picks a
   winning model, freeze_model_spec() writes an immutable record of exactly
   what was chosen (feature spec version, model name, hyperparameters,
   selection criterion, validation metrics) to results/model_spec.json.
   Any script that touches the holdout should call require_frozen_spec()
   first and refuse to run without it — this is the code-level enforcement
   of "the holdout stays untouched until the spec is frozen."

2. A running counter of how many models, strategies, features, and
   statistical tests have been evaluated across the whole project. This
   matters for multiple-testing correction: a p-value that looks significant
   against "the 5 tests in this step" can be unremarkable against "the 30
   tests run across the whole project so far." build_experiment_ledger.py
   uses CUMULATIVE_TEST_LOG below to keep an honest running total.
"""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional


SPEC_PATH_DEFAULT = Path(__file__).resolve().parents[1] / "results" / "model_spec.json"


@dataclass
class ModelSpec:
    frozen: bool
    feature_spec_version: str
    pool: str                      # "main" or "euro"
    selected_model: str            # e.g. "Gradient Boosting", "Averaging Ensemble"
    selection_criterion: str       # e.g. "lowest validation log loss"
    hyperparameters: dict
    validation_metrics: dict
    selection_notes: str = ""


def freeze_model_spec(spec: ModelSpec, path: Path = SPEC_PATH_DEFAULT, overwrite: bool = False) -> Path:
    """Writes the model spec as frozen. Refuses to silently clobber an
    existing frozen spec for the same pool unless overwrite=True — freezing
    is meant to be a one-way door per pool."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if path.exists():
        existing = json.loads(path.read_text())
    if spec.pool in existing and not overwrite:
        raise RuntimeError(
            f"A frozen spec for pool='{spec.pool}' already exists at {path}. "
            f"Pass overwrite=True if you deliberately intend to replace it "
            f"(and understand this re-opens the holdout question)."
        )
    spec_dict = asdict(spec)
    spec_dict["frozen"] = True
    existing[spec.pool] = spec_dict
    path.write_text(json.dumps(existing, indent=2))
    return path


def load_frozen_spec(pool: str, path: Path = SPEC_PATH_DEFAULT) -> Optional[ModelSpec]:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if pool not in data:
        return None
    return ModelSpec(**data[pool])


def require_frozen_spec(pool: str, path: Path = SPEC_PATH_DEFAULT) -> ModelSpec:
    """Call this at the top of any script that touches the holdout. Raises
    if no frozen spec exists for this pool — i.e. model selection is not
    yet finished, so the holdout must stay untouched."""
    spec = load_frozen_spec(pool, path)
    if spec is None or not spec.frozen:
        raise RuntimeError(
            f"No frozen model spec found for pool='{pool}'. The holdout may not be "
            f"used until model selection is complete and freeze_model_spec() has been "
            f"called for this pool. Run the validation-only comparison (Step 8) first."
        )
    return spec


# ---------------------------------------------------------------------------
# Cumulative counts, filled in by build_experiment_ledger.py from the actual
# results/*.json files on disk (not hand-maintained — see that script). This
# constant is the *shape* of the log the ledger builder fills in.
# ---------------------------------------------------------------------------
LOG_SCHEMA = {
    "step": None,               # e.g. "step6"
    "pools_evaluated": None,    # e.g. ["main", "euro"]
    "n_models": None,           # distinct learned models (not counting baselines)
    "n_strategies": None,       # models + baselines
    "n_features": None,         # per pool, dict
    "n_significance_tests": None,
    "used_holdout": None,       # bool — did this step touch what is now the frozen holdout window?
}
