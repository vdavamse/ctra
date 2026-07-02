"""Task split generation for trial outcome prediction.

Generates stratified train/val/test splits that can be consumed by the
MCTS evaluation loop and the benchmark comparison pipeline.  Splits are
stored as lightweight Parquet files containing ``(trial_id, label)`` pairs.

Stratification by **phase** and optionally **therapeutic area** ensures
that class balance and phase distribution are preserved across splits,
which is critical because base rates differ substantially across phases
(Phase I ~65 % success, Phase II ~35 %, Phase III ~55 %).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class TaskSplitGenerator:
    """Generate stratified train/val/test splits for trial outcome prediction.

    Generates stratified splits that preserve class balance and phase distribution
    (base rates differ across phases: Phase I ~65%, Phase II ~35%, Phase III ~55%
    success). Splits are stored as lightweight Parquet with (trial_id, label) pairs.

    Args:
        random_state: Random seed for reproducibility.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialize the split generator.

        Args:
            random_state: Random seed for reproducibility (default 42).
        """
        self.random_state = random_state

    def generate_splits(
        self,
        trial_ids: list[str],
        labels: list[int],
        phases: list[str] | None = None,
        therapeutic_areas: list[str] | None = None,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> dict[str, pl.DataFrame]:
        """Generate stratified train/val/test splits.

        Stratification is performed on a composite key of
        ``(label, phase, therapeutic_area)`` when those columns are
        provided.  If a stratification group has fewer than 2 members,
        those samples are assigned to the training set.

        Args:
            trial_ids: Unique trial identifiers.
            labels: Binary outcome labels (0=failure, 1=success).
            phases: Trial phases for stratification (e.g. ``["PHASE2"]``).
            therapeutic_areas: Therapeutic areas for stratification.
            train_ratio: Fraction of data for training (default 0.6).
            val_ratio: Fraction of data for validation (default 0.2).

        Returns:
            ``{"train": df, "val": df, "test": df}`` where each DataFrame
            has columns ``[trial_id, label]``.

        Raises:
            ValueError: If inputs have mismatched lengths or invalid ratios.
        """
        # -- Input validation --
        n = len(trial_ids)
        if len(labels) != n:
            raise ValueError(
                f"trial_ids ({n}) and labels ({len(labels)}) must have the same length"
            )
        if phases is not None and len(phases) != n:
            raise ValueError(f"phases ({len(phases)}) must match trial_ids ({n})")
        if therapeutic_areas is not None and len(therapeutic_areas) != n:
            raise ValueError(
                f"therapeutic_areas ({len(therapeutic_areas)}) must match trial_ids ({n})"
            )

        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0 or train_ratio < 0 or val_ratio < 0:
            raise ValueError(
                f"Ratios must be non-negative and sum to <= 1.0.  "
                f"Got train={train_ratio}, val={val_ratio}, "
                f"test={test_ratio:.4f}"
            )

        if n == 0:
            empty = pl.DataFrame({"trial_id": [], "label": []})
            return {"train": empty, "val": empty, "test": empty}

        # -- Build stratification key --
        strat_parts: list[list[str]] = [
            [str(lbl) for lbl in labels],
        ]
        if phases is not None:
            strat_parts.append([str(p) for p in phases])
        if therapeutic_areas is not None:
            strat_parts.append([str(ta) for ta in therapeutic_areas])

        strat_key = np.array(["_".join(parts) for parts in zip(*strat_parts, strict=True)])

        # Handle groups with fewer than 2 members: move them to a
        # catch-all group so StratifiedShuffleSplit doesn't fail.
        unique, counts = np.unique(strat_key, return_counts=True)
        small_groups = set(unique[counts < 2])
        if small_groups:
            strat_key = np.array(["__small__" if k in small_groups else k for k in strat_key])
            logger.debug(
                "Merged %d small stratification groups into __small__",
                len(small_groups),
            )

        indices = np.arange(n)
        trial_ids_arr = np.array(trial_ids)
        labels_arr = np.array(labels)

        # -- First split: train+val vs test --
        train_val_ratio = train_ratio + val_ratio
        if test_ratio > 0 and n >= 2:
            splitter1 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio,
                random_state=self.random_state,
            )
            train_val_idx, test_idx = next(splitter1.split(indices, strat_key))
        else:
            train_val_idx = indices
            test_idx = np.array([], dtype=int)

        # -- Second split: train vs val (within train+val) --
        if val_ratio > 0 and len(train_val_idx) >= 2:
            relative_val = val_ratio / train_val_ratio
            strat_tv = strat_key[train_val_idx]

            # Re-check for small groups in the reduced set
            unique_tv, counts_tv = np.unique(strat_tv, return_counts=True)
            small_tv = set(unique_tv[counts_tv < 2])
            if small_tv:
                strat_tv = np.array(["__small__" if k in small_tv else k for k in strat_tv])

            splitter2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=relative_val,
                random_state=self.random_state,
            )
            train_local, val_local = next(splitter2.split(train_val_idx, strat_tv))
            train_idx = train_val_idx[train_local]
            val_idx = train_val_idx[val_local]
        else:
            train_idx = train_val_idx
            val_idx = np.array([], dtype=int)

        def _make_df(idx: NDArray[Any]) -> pl.DataFrame:
            """Create a split DataFrame from index array.

            Parameters:
                idx: Array of row indices to select.

            Returns:
                DataFrame with columns [trial_id, label].
            """
            return pl.DataFrame(
                {
                    "trial_id": trial_ids_arr[idx].tolist(),
                    "label": labels_arr[idx].tolist(),
                }
            )

        splits = {
            "train": _make_df(train_idx),
            "val": _make_df(val_idx),
            "test": _make_df(test_idx),
        }

        logger.info(
            "Generated splits: train=%d, val=%d, test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )
        return splits

    def save_splits(
        self,
        splits: dict[str, pl.DataFrame],
        output_dir: str | Path,
        prefix: str = "",
    ) -> dict[str, Path]:
        """Save splits as Parquet files.

        Args:
            splits: Dict from :meth:`generate_splits`.
            output_dir: Target directory.
            prefix: Optional filename prefix (e.g. ``"phase2_"``).

        Returns:
            Dict mapping split name to the written file path.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}
        for name, df in splits.items():
            fname = f"{prefix}{name}_data.parquet"
            path = out / fname
            df.write_parquet(path)
            paths[name] = path
            logger.info("Saved %s split (%d rows) to %s", name, len(df), path)
        return paths

    def load_splits(
        self,
        splits_dir: str | Path,
        prefix: str = "",
    ) -> dict[str, pl.DataFrame]:
        """Load previously saved splits.

        Args:
            splits_dir: Directory containing split Parquet files.
            prefix: Filename prefix used during :meth:`save_splits`.

        Returns:
            ``{"train": df, "val": df, "test": df}``.

        Raises:
            FileNotFoundError: If a required split file is missing.
        """
        base = Path(splits_dir)
        splits: dict[str, pl.DataFrame] = {}
        for name in ("train", "val", "test"):
            fname = f"{prefix}{name}_data.parquet"
            path = base / fname
            if not path.exists():
                raise FileNotFoundError(f"Split file not found: {path}")
            splits[name] = pl.read_parquet(path)
            logger.info(
                "Loaded %s split (%d rows) from %s",
                name,
                len(splits[name]),
                path,
            )
        return splits
