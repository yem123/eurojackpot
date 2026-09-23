"""
splits.py

Three-way chronological split: train / validation / holdout, with the
holdout treated as sacred — reserved for a single, final, no-further-tuning
evaluation once a model specification is frozen.

Background: Steps 6 and 7 both used "the last 100 draws" as a repeated
comparison set for every strategy tried. That set was never used to tune
a model's hyperparameters or to decide which model to drop, but it WAS
looked at over and over across two steps (20 permutation tests total —
see results/EXPERIMENT_LEDGER.md). That is enough repeated exposure that
it can no longer be treated as a clean, never-seen final test.

From Step 8 onward:
    - HOLDOUT  = the most recent `holdout_draws` draws. Off-limits for
      everything except a single final run, once model selection is done.
    - VALIDATION = the `validation_draws` draws immediately before that.
      This is where model/strategy SELECTION happens.
    - TRAIN = everything before validation. This is where models are FIT.

With the current dataset (689 usable draws after Step 7's feature build
drops draw 0), validation_draws=100 and holdout_draws=100 gives:
    train:      draw_idx   1 - 489   (489 draws)
    validation: draw_idx 490 - 589   (100 draws)
    holdout:    draw_idx 590 - 689   (100 draws)  <- same draws Steps 6-7 used as "test"

Note the holdout window here is numerically identical to the Step 6/7 test
window. It has NOT been reset to fresh data (there isn't any — draw 689 is
the most recent draw in the dataset). What changes is discipline, not data:
from Step 8 on, nothing in this window is used for model selection, and it
does not appear in any result until model_spec.freeze() has been called
(see experiment_log.py). Steps 6/7's numbers on this window remain on the
record as exploratory, not as the final holdout test.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Splits:
    train: pd.DataFrame
    validation: pd.DataFrame
    holdout: pd.DataFrame
    train_ids: set
    validation_ids: set
    holdout_ids: set

    def assert_holdout_untouched(self, used_ids: set):
        """Call this with whatever draw_idx values a computation actually used.
        Raises if any holdout draw leaked in."""
        leaked = used_ids & self.holdout_ids
        if leaked:
            raise RuntimeError(
                f"Holdout leakage detected: {len(leaked)} holdout draw(s) were used "
                f"in a computation that should only touch train/validation: {sorted(leaked)[:5]}..."
            )


def three_way_split(long_df: pd.DataFrame, validation_draws: int = 100, holdout_draws: int = 100) -> Splits:
    draw_ids = np.sort(long_df["draw_idx"].unique())
    n = len(draw_ids)
    if n <= validation_draws + holdout_draws:
        raise ValueError(f"Not enough draws ({n}) for validation_draws={validation_draws} "
                          f"+ holdout_draws={holdout_draws}.")

    holdout_ids = set(draw_ids[-holdout_draws:])
    validation_ids = set(draw_ids[-(holdout_draws + validation_draws):-holdout_draws])
    train_ids = set(draw_ids[:-(holdout_draws + validation_draws)])

    assert not (train_ids & validation_ids), "train/validation overlap"
    assert not (validation_ids & holdout_ids), "validation/holdout overlap"
    assert not (train_ids & holdout_ids), "train/holdout overlap"

    return Splits(
        train=long_df[long_df["draw_idx"].isin(train_ids)].reset_index(drop=True),
        validation=long_df[long_df["draw_idx"].isin(validation_ids)].reset_index(drop=True),
        holdout=long_df[long_df["draw_idx"].isin(holdout_ids)].reset_index(drop=True),
        train_ids=train_ids, validation_ids=validation_ids, holdout_ids=holdout_ids,
    )
