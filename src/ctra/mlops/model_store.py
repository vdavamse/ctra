"""Model versioning and artifact management.

Provides a local model store that saves trained models alongside their
feature plans, objective results, and metadata.  Models are versioned
with ISO-8601 timestamps and can be promoted to ``"staging"`` or
``"production"`` stages.

Directory layout::

    models/
        v_20260327T120000/
            xgboost_model.joblib
            tabpfn_model.pkl          (if TabPFN was trained)
            feature_plans.json
            objective_result.json
            metadata.json             (config, timestamps, metrics, stage)
        v_20260328T090000/
            ...
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from ctra.models.model_registry import ModelRegistry
    from ctra.search.objectives import ObjectiveResult

logger = logging.getLogger(__name__)


class ModelStore:
    """Manages model artifacts, feature plans, and versioning.

    Args:
        store_dir: Root directory for the model store. Defaults to ``"models/"``.
    """

    def __init__(self, store_dir: str = "models/") -> None:
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_model(
        self,
        model_name: str,
        registry: ModelRegistry,
        feature_plans: list[dict[str, Any]],
        objective_result: ObjectiveResult,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a trained model with its feature plans.

        Args:
            model_name: Name of the best classifier (``"xgboost"`` or ``"tabpfn"``).
            registry: The fitted ``ModelRegistry`` containing trained wrappers.
            feature_plans: List of feature plan dicts (each with ``name``, ``dtype``,
                ``description``, ``extraction_prompt``, ``sources``).
            objective_result: The objective evaluation for the saved model.
            metadata: Optional extra metadata (training config, dataset info, etc.).

        Returns:
            Version string (e.g. ``"v_20260327T120000"``).
        """
        version = self._make_version()
        version_dir = self._store_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # -- Save model artifact(s) via joblib --
        # Always try to save the requested model
        try:
            wrapper = registry.get_wrapper(model_name)
            artifact_name = f"{model_name}_model.joblib"
            joblib.dump(wrapper, version_dir / artifact_name)
            logger.info("Saved %s to %s", artifact_name, version_dir)
        except KeyError:
            logger.warning(
                "Wrapper for %s not found in registry; skipping artifact save",
                model_name,
            )

        # Also save the other model if it was trained (for comparison)
        for clf_name in ("xgboost", "tabpfn"):
            if clf_name == model_name:
                continue
            try:
                other_wrapper = registry.get_wrapper(clf_name)
                artifact_name = f"{clf_name}_model.joblib"
                joblib.dump(other_wrapper, version_dir / artifact_name)
                logger.info("Saved companion model %s", artifact_name)
            except KeyError:
                pass  # Not trained -- fine

        # -- Feature plans --
        with open(version_dir / "feature_plans.json", "w") as f:
            json.dump(feature_plans, f, indent=2, default=str)

        # -- Objective result --
        obj_dict = {
            "values": objective_result.values.tolist(),
            "names": objective_result.names,
            "details": {
                k: (float(v) if hasattr(v, "__float__") else v)
                for k, v in objective_result.details.items()
            },
        }
        with open(version_dir / "objective_result.json", "w") as f:
            json.dump(obj_dict, f, indent=2)

        # -- Metadata --
        meta: dict[str, Any] = {
            "version": version,
            "model_name": model_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "feature_count": len(feature_plans),
            "objective_names": objective_result.names,
            "objective_values": objective_result.values.tolist(),
            "stage": "none",
        }
        if metadata:
            meta.update(metadata)

        with open(version_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info("Model version %s saved to %s", version, version_dir)
        return version

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_model(
        self,
        version: str | None = None,
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Load a model version.

        Args:
            version: Version string to load. ``None`` loads the latest version.

        Returns:
            ``(model_wrapper, feature_plans)`` where ``model_wrapper`` is the
            deserialized classifier wrapper and ``feature_plans`` is the list
            of feature plan dicts.

        Raises:
            FileNotFoundError: If the requested version does not exist or has
                no model artifact.
        """
        if version is None:
            version = self.get_latest_version()

        version_dir = self._store_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Model version {version} not found at {version_dir}")

        # Determine which model to load from metadata
        meta_path = version_dir / "metadata.json"
        model_name = "xgboost"  # default fallback
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            model_name = meta.get("model_name", "xgboost")

        # Load the primary model wrapper
        artifact_path = version_dir / f"{model_name}_model.joblib"
        if not artifact_path.exists():
            # Try finding any .joblib file
            joblib_files = list(version_dir.glob("*_model.joblib"))
            if not joblib_files:
                raise FileNotFoundError(f"No model artifact found in {version_dir}")
            artifact_path = joblib_files[0]
            logger.warning(
                "Primary model %s not found; loading %s instead",
                model_name,
                artifact_path.name,
            )

        wrapper = joblib.load(artifact_path)

        # Load feature plans
        fp_path = version_dir / "feature_plans.json"
        feature_plans: list[dict[str, Any]] = []
        if fp_path.exists():
            with open(fp_path) as f:
                feature_plans = json.load(f)

        logger.info("Loaded model version %s (%s)", version, artifact_path.name)
        return wrapper, feature_plans

    # ------------------------------------------------------------------
    # Listing and versioning
    # ------------------------------------------------------------------

    def list_versions(self) -> list[dict[str, Any]]:
        """List all saved model versions with metadata.

        Returns:
            List of metadata dicts, sorted by version (newest first).
        """
        versions: list[dict[str, Any]] = []

        for version_dir in sorted(self._store_dir.iterdir(), reverse=True):
            if not version_dir.is_dir() or not version_dir.name.startswith("v_"):
                continue
            meta_path = version_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                versions.append(meta)
            else:
                # Minimal entry for versions without metadata
                versions.append(
                    {
                        "version": version_dir.name,
                        "stage": "none",
                    }
                )

        return versions

    def get_latest_version(self) -> str:
        """Get the latest model version string.

        Returns:
            The most recent version string.

        Raises:
            FileNotFoundError: If no model versions exist.
        """
        version_dirs = sorted(
            [d for d in self._store_dir.iterdir() if d.is_dir() and d.name.startswith("v_")],
            reverse=True,
        )
        if not version_dirs:
            raise FileNotFoundError(f"No model versions found in {self._store_dir}")
        return version_dirs[0].name

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote(self, version: str, stage: str = "production") -> None:
        """Mark a version as production/staging.

        If another version currently holds the target stage, it is demoted
        to ``"none"``.

        Args:
            version: Version string to promote.
            stage: Target stage: ``"production"``, ``"staging"``, or ``"none"``.
        """
        if stage not in ("production", "staging", "none"):
            raise ValueError(
                f"Invalid stage {stage!r}. Must be 'production', 'staging', or 'none'."
            )

        # Demote any existing version at the target stage
        if stage != "none":
            for meta in self.list_versions():
                if meta.get("stage") == stage and meta.get("version") != version:
                    self._update_stage(meta["version"], "none")

        self._update_stage(version, stage)
        logger.info("Promoted version %s to stage %s", version, stage)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_version() -> str:
        """Generate a unique version identifier from the current UTC timestamp.

        Used by ``save_model`` to create a version directory like
        ``v_20260405T154230``.

        Returns:
            A version string in ISO format with 'v_' prefix.
        """
        return f"v_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"

    def _update_stage(self, version: str, stage: str) -> None:
        """Update the deployment stage in a version's metadata JSON.

        Used by ``promote`` to transition a model version from
        ``"staging"`` to ``"production"`` (or other stages).

        Args:
            version: The version identifier (e.g., ``v_20260405T154230``).
            stage: The new stage name (e.g., ``"production"``).
        """
        meta_path = self._store_dir / version / "metadata.json"
        if not meta_path.exists():
            logger.warning("Metadata not found for version %s", version)
            return

        with open(meta_path) as f:
            meta = json.load(f)

        meta["stage"] = stage

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
