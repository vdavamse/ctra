"""Tests for ctra.agents.runner — agent execution abstraction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ctra.agents.data_models import (
    FeatureSource,
    FeatureType,
)
from ctra.agents.runner import (
    load_feature_plans_from_json,
)

# ---------------------------------------------------------------------------
# load_feature_plans_from_json
# ---------------------------------------------------------------------------


def _make_plan_dict() -> dict:
    return {
        "drug_mechanism": {
            "feature_name": "drug_mechanism",
            "feature_idea": "Type of drug mechanism",
            "feature_type": {"value": "categorical"},
            "data_sources": ["chembl"],
            "example_values": [{"value": "kinase_inhibitor"}],
            "possible_values": {"value": ["kinase_inhibitor", "antibody"]},
            "feature_instructions": "Look up mechanism in ChEMBL.",
        },
        "prior_trials": {
            "feature_name": "prior_trials",
            "feature_idea": "Number of prior trials for this drug",
            "feature_type": {"value": "integer"},
            "data_sources": ["related_clinical_trials"],
            "example_values": [{"value": "5"}],
            "possible_values": {},
            "feature_instructions": "Count related trials.",
        },
    }


class TestLoadFeaturePlansFromJson:
    def test_round_trip(self, tmp_path: Path) -> None:
        plans_dict = _make_plan_dict()
        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans_dict, indent=2))

        plans = load_feature_plans_from_json(path)

        assert len(plans) == 2
        assert "drug_mechanism" in plans
        assert "prior_trials" in plans

    def test_feature_types_are_enums(self, tmp_path: Path) -> None:
        plans_dict = _make_plan_dict()
        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans_dict, indent=2))

        plans = load_feature_plans_from_json(path)

        assert plans["drug_mechanism"].feature_type["value"] == FeatureType.CATEGORICAL
        assert plans["prior_trials"].feature_type["value"] == FeatureType.INTEGER

    def test_data_sources_are_enums(self, tmp_path: Path) -> None:
        plans_dict = _make_plan_dict()
        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans_dict, indent=2))

        plans = load_feature_plans_from_json(path)

        assert plans["drug_mechanism"].data_sources == [FeatureSource.CHEMBL]
        assert plans["prior_trials"].data_sources == [FeatureSource.RELATED_CLINICAL_TRIALS]

    def test_preserves_possible_values(self, tmp_path: Path) -> None:
        plans_dict = _make_plan_dict()
        path = tmp_path / "plans.json"
        path.write_text(json.dumps(plans_dict, indent=2))

        plans = load_feature_plans_from_json(path)

        assert plans["drug_mechanism"].possible_values == {
            "value": ["kinase_inhibitor", "antibody"]
        }
        assert plans["prior_trials"].possible_values == {}

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.json"
        path.write_text("{}")

        plans = load_feature_plans_from_json(path)
        assert plans == {}
