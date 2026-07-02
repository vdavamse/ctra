"""Data models and entity type mappings for NER evaluation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class MatchStrategy(str, enum.Enum):
    """Entity matching strategy for NER evaluation."""

    STRICT = "strict"  # Exact span match (start + end + label)
    RELAXED = "relaxed"  # Overlapping span + correct label
    TEXT_ONLY = "text_only"  # Surface form match (most relevant for LinearRAG)


class EvalMode(str, enum.Enum):
    """Evaluation mode."""

    STANDARD = "standard"  # Mode 1: benchmark's native labels
    CTRA = "ctra"  # Mode 2: CTRA's 16 labels
    BOTH = "both"  # Run both modes


class Benchmark(str, enum.Enum):
    """Supported NER benchmarks."""

    CHIA = "chia"
    N2C2 = "n2c2"
    TAC = "tac"


@dataclass(frozen=True, slots=True)
class GoldAnnotation:
    """A single gold-standard entity annotation from a benchmark.

    Normalized from benchmark-specific formats (BRAT, XML, bigbio KB)
    into a common schema for evaluation.
    """

    doc_id: str
    start_char: int
    end_char: int
    text: str
    gold_label: str
    mapped_ctra_label: str | None = None  # None = unmappable type


@dataclass(frozen=True, slots=True)
class PredictedEntity:
    """A single NER prediction from GLiNER-BioMed."""

    doc_id: str
    start_char: int
    end_char: int
    text: str
    label: str
    score: float = 0.0


@dataclass
class PerLabelMetrics:
    """Precision/recall/F1 for a single entity label."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int  # Number of gold entities for this label
    predicted_count: int  # Number of predictions for this label


@dataclass
class EvalResult:
    """Complete evaluation result for one benchmark + mode combination."""

    benchmark: str
    mode: str
    match_strategy: str
    per_label: list[PerLabelMetrics]
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    micro_precision: float
    micro_recall: float
    total_gold: int
    total_predicted: int
    total_matched: int
    metadata: dict[str, str] = field(default_factory=dict)


# CHIA benchmark -> CTRA label mapping
#
# Mapping rationale (validated by full-scale diagnostic over 36,393 entities):
# - "Condition" maps to MULTIPLE CTRA labels: Disease is primary, but GLiNER
#   correctly sub-classifies some as Symptom (321 entities), Adverse event (121),
#   or Anatomical structure (57). We accept all as valid matches.
# - "Measurement" was mapped to Clinical endpoint but this was wrong for most
#   entities. Measurement includes biomarkers (platelets, Hb), diagnostic tests
#   (ECOG), and time-related measures (postnatal age). Mapped to Biomarker as
#   the best single fit; GLiNER may also label as Diagnostic test or Clinical
#   endpoint — all are valid.
# - "Value" was mapped to Drug dosage but most Values are numeric qualifiers
#   ("< 800 g", "0-1 score", "Mild-to-moderate"), not drug dosages. Excluded
#   from mapping (None) since no single CTRA label fits.
# - "Observation" was mapped to Symptom but includes non-symptom entities like
#   "history of", "confirmed diagnosis". Excluded (None) — too heterogeneous.
# - "Device" includes both diagnostic devices and therapeutic devices. Mapped
#   to None since no single CTRA label fits (could be Diagnostic test OR
#   Therapeutic procedure depending on context).
CHIA_TO_CTRA: dict[str, str | None] = {
    "Condition": "Disease",
    "Drug": "Drug",
    "Procedure": "Therapeutic procedure",
    "Device": None,  # Heterogeneous: some diagnostic, some therapeutic
    "Measurement": "Biomarker",  # Was Clinical endpoint; Biomarker fits better
    "Observation": None,  # Heterogeneous: "history of", "diagnosis", "bleeding risk"
    "Temporal": "Time period",
    "Person": "Patient population",
    "Value": None,  # Numeric qualifiers, not drug dosages
    "Mood": None,  # Modifier type, no CTRA equivalent
    "Negation": None,  # Modifier type, no CTRA equivalent
    "Qualifier": None,  # Modifier type, no CTRA equivalent
    "Scope": None,  # Structural type, no CTRA equivalent
    "Visit": "Time period",
    "Reference_point": "Time period",
    "Multiplier": None,
}

# Condition accepts multiple CTRA labels as valid matches.
# GLiNER correctly sub-classifies conditions as Disease, Symptom, Adverse event,
# or Anatomical structure. All should count as matches in Mode 2 evaluation.
CHIA_CONDITION_VALID_LABELS: set[str] = {
    "Disease",
    "Symptom",
    "Adverse event",
    "Anatomical structure",
    "Clinical endpoint",
}

# N2C2 2018 benchmark -> CTRA label mapping
N2C2_TO_CTRA: dict[str, str | None] = {
    "Drug": "Drug",
    "Strength": "Drug dosage",
    "Dosage": "Drug dosage",
    "Route": None,  # Was Mechanism of action; route != mechanism
    "Form": None,  # "tablet", "injection" — no CTRA equivalent
    "Frequency": "Time period",
    "Duration": "Time period",
    "Reason": "Disease",  # Primary mapping; some are Symptom
    "ADE": "Adverse event",
}

# TAC 2017 (spl_adr_200db) benchmark -> CTRA label mapping
TAC_TO_CTRA: dict[str, str | None] = {
    "AdverseReaction": "Adverse event",
    "Severity": None,  # Modifier, no direct CTRA label
    "Negation": None,  # Modifier, no direct CTRA label
    "DrugClass": "Drug",
    "Animal": None,  # No CTRA equivalent
    "Factor": None,  # Contextual modifier
}

# Mapping from Benchmark enum to its mapping dict
BENCHMARK_MAPPINGS: dict[Benchmark, dict[str, str | None]] = {
    Benchmark.CHIA: CHIA_TO_CTRA,
    Benchmark.N2C2: N2C2_TO_CTRA,
    Benchmark.TAC: TAC_TO_CTRA,
}

# Native entity labels per benchmark (for Mode 1)
BENCHMARK_NATIVE_LABELS: dict[Benchmark, list[str]] = {
    Benchmark.CHIA: [
        "Condition",
        "Drug",
        "Procedure",
        "Device",
        "Measurement",
        "Observation",
        "Temporal",
        "Person",
        "Value",
        "Mood",
        "Negation",
        "Qualifier",
        "Scope",
        "Visit",
        "Reference_point",
        "Multiplier",
    ],
    Benchmark.N2C2: [
        "Drug",
        "Strength",
        "Dosage",
        "Route",
        "Form",
        "Frequency",
        "Duration",
        "Reason",
        "ADE",
    ],
    Benchmark.TAC: [
        "AdverseReaction",
        "Severity",
        "Negation",
        "DrugClass",
        "Animal",
        "Factor",
    ],
}

# Label phrasing variants for ablation study
LABEL_VARIANTS: dict[str, list[str]] = {
    "Drug": ["Drug or medication", "Pharmaceutical compound"],
    "Mechanism of action": ["Pharmacological mechanism", "Drug mechanism"],
    "Drug dosage": ["Dose or dosage", "Drug dose and administration"],
    "Disease": ["Disease or condition", "Medical condition"],
    "Symptom": ["Sign or symptom", "Clinical symptom"],
    "Adverse event": ["Adverse drug reaction", "Side effect"],
    "Gene or protein": ["Gene or protein name", "Genetic marker"],
    "Biomarker": ["Biological marker", "Diagnostic biomarker"],
    "Cell type": ["Cell or cell line", "Biological cell type"],
    "Anatomical structure": ["Body part or organ", "Anatomical location"],
    "Clinical endpoint": ["Outcome measure", "Clinical outcome"],
    "Therapeutic procedure": ["Medical procedure", "Treatment procedure"],
    "Diagnostic test": ["Laboratory test", "Diagnostic examination"],
    "Patient population": ["Patient group or cohort", "Study population"],
    "Organization": ["Institution or company", "Healthcare organization"],
    "Time period": ["Duration or time interval", "Temporal expression"],
}


def _get_ctra_labels() -> list[str]:
    """Return CTRA's entity labels from the canonical RAGConfig source."""
    from ctra.config.settings import RAGConfig

    return RAGConfig().ner_labels


# CTRA's core entity labels — sourced from RAGConfig.ner_labels to avoid drift
CTRA_LABELS: list[str] = _get_ctra_labels()
