"""Central configuration for the CTRA pipeline.

All settings are loaded from environment variables (prefixed ``CTRA_``) or a
``.env`` file.  Pydantic Settings validates types and provides defaults that
match the research plan (see ``research/implementation-plan.md``).

.. rubric:: Design Review (ARCHITECT)

**1. Interface contracts -- RAG-engineer to AutoCT-integrator**

The RAG layer (``linearrag_wrapper.py``) returns a list of ``RetrievedChunk``
dataclass instances, each containing:

    - ``text: str``       -- the passage text
    - ``source: str``     -- origin identifier (``"ctg"`` | ``"pubmed"`` | ``"chembl"`` | ``"faers"``)
    - ``doc_id: str``     -- unique document id (NCT ID, PMID, ChEMBL ID, ...)
    - ``date: date | None`` -- publication / posting date (used for leakage filter)
    - ``score: float``    -- relevance score from Personalized PageRank

The feature-builder agent (``feature_builder.py``) calls ``rag.search()`` with
a query string, a ``before_date`` cutoff, and optional source filters.  It
receives ``list[RetrievedChunk]`` and must coerce extracted values to the types
declared in the feature schema produced by ``feature_planner.py``.

The feature schema is a ``list[FeatureSpec]`` where each ``FeatureSpec`` has:

    - ``name: str``             -- machine-readable feature name
    - ``dtype: Literal["float", "int", "bool", "categorical"]``
    - ``description: str``      -- human-readable description for SHAP labels
    - ``extraction_prompt: str`` -- instruction the builder agent uses per trial
    - ``sources: list[str]``    -- which RAG indexes to query

The model wrappers (``tabpfn_classifier.py``, ``xgboost_classifier.py``)
consume a ``pd.DataFrame`` whose columns correspond to ``FeatureSpec.name``
values.  TabPFN accepts the raw DataFrame directly (native missing-value and
categorical handling).  XGBoost receives the same DataFrame after a
``ColumnTransformer`` pass.

**2. Label leakage risk flags**

Label leakage can enter at three points:

    a. **RAG retrieval** -- If a PubMed article or ClinicalTrials.gov posting
       published *after* the trial's ``statusModule.startDateStruct.date``
       contains outcome information, the agent will see the answer.
       Mitigation: ``linearrag_wrapper.search()`` accepts ``before_date`` and
       filters passages before returning.  The date field is populated at
       index time from the source metadata (``protocolSection.statusModule``
       for CTG, PubMed publication date, ChEMBL ``first_approval`` field, FAERS
       report quarter).  **Risk:** Documents that are *updated* after the
       cutoff may have had their original content modified.  ClinicalTrials.gov
       records frequently receive post-completion updates to ``statusModule``
       and ``resultsSection``.  The indexer must snapshot ``protocolSection``
       fields only and exclude ``resultsSection`` entirely.

    b. **Feature values derived from future trials** -- If the builder agent
       retrieves information from a Phase III trial when predicting Phase II
       for the same drug, the Phase III metadata (which exists only because
       Phase II succeeded) leaks the label.  Mitigation: filter by
       ``startDateStruct.date`` of the *queried* trial, not just publication
       date.

    c. **TrialBench temporal splits** -- The benchmark uses time-based splits.
       If CTRA adds new data sources (ChEMBL, FAERS) without respecting the
       same temporal boundary, features for test-set trials may include
       post-split information.  Mitigation: pass the split boundary date from
       the dataset loader into the RAG search.

**3. TabPFN edge cases**

    - **Feature count limit:** TabPFN v2.5 supports up to 2,000 features.
      AutoCT typically produces 10-30 features, well within bounds.  However,
      if MCTS explores configurations with >100 engineered features (unlikely
      but possible with aggressive ``Add`` operations), TabPFN will still work
      but inference time scales quadratically with feature count.  The MCTS
      parsimony objective (minimize feature count) naturally prevents this.

    - **Missing values:** Both TabPFN and XGBoost handle ``pd.NA`` / ``np.nan``
      natively.  The orchestrator's ``build_feature_type_transformer`` uses
      ``"passthrough"`` for all numeric types.  ``SimpleImputer`` is only used
      in the standalone ``XGBoostWrapper`` (not the orchestrator pipeline).

    - **Row count:** TabPFN v2.5 supports up to 100K rows.  CTRA trains on
      100-500 rows -- no issue.  However, ``fit_mode='fit_with_cache'`` should
      be used during MCTS rollouts to avoid re-encoding the training set on
      each evaluation (saves ~60% latency per MCTS node).

    - **Categorical cardinality:** TabPFN handles categoricals natively but
      high-cardinality columns (>50 unique values) degrade performance.  The
      feature planner should constrain categoricals to common therapeutic
      areas, phases, and design types.

    - **Reproducibility:** TabPFN is non-deterministic by default (ensemble
      sampling).  Set ``random_state=42`` in ``TabPFNClassifier()`` for
      reproducible MCTS evaluations.

**4. LinearRAG chunk format compatibility**

    LinearRAG indexes documents as passages (using spaCy sentence splitting to
    form multi-sentence passages) and builds a graph of entity co-occurrence
    edges.  Retrieval returns *passages* ranked by Personalized PageRank
    seeded from the query entities.

    Compatibility considerations:

    - **ClinicalTrials.gov ``protocolSection``:** The ``protocolSection`` JSON
      must be flattened into prose sentences before indexing.  Key fields:
      ``identificationModule.briefTitle``, ``descriptionModule.briefSummary``,
      ``designModule``, ``armsInterventionsModule``,
      ``eligibilityModule.eligibilityCriteria``.  Each sentence should carry
      the NCT ID and field path as metadata for traceability.

    - **PubMed abstracts:** Already sentence-structured.  Index title +
      abstract as separate sentences.

    - **ChEMBL:** Structured data (targets, mechanisms, clinical phases) should
      be converted to declarative sentences (e.g., "Aspirin inhibits
      cyclooxygenase-1 (COX-1) and cyclooxygenase-2 (COX-2).") for
      compatibility with LinearRAG's NER pipeline.  Raw database fields are not
      suitable for entity extraction.

    - **FAERS:** Adverse event reports are semi-structured.  Convert to
      sentences: "Drug X was associated with Y adverse events reported to
      FAERS in Q1 2023."

    - **Entity overlap:** LinearRAG's graph requires entity co-occurrence
      across documents for multi-hop retrieval.  Drug names must be normalized
      across all sources (ClinicalTrials.gov intervention names, PubMed
      mentions, ChEMBL names, FAERS ``medicinalproduct``).  Use ChEMBL's
      molecule_synonyms table as the canonical mapping.

    - **Chunk granularity:** AutoCT's original txtai setup embedded full
      paragraphs.  LinearRAG's passage-level granularity is finer, which
      means more hops are needed to gather equivalent context.  The
      ``top_k`` parameter in ``search()`` should be set higher (20-50) than
      the original txtai default (5-10) to compensate.
"""

from __future__ import annotations

import enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Phase(str, enum.Enum):
    """Clinical trial phases supported by CTRA."""

    PHASE_1 = "phase1"
    PHASE_2 = "phase2"
    PHASE_3 = "phase3"


class ClassifierType(str, enum.Enum):
    """Supported downstream classifiers."""

    XGBOOST = "xgboost"
    TABPFN = "tabpfn"

    @property
    def short_name(self) -> Literal["xgb", "tabpfn"]:
        """Short model-type string used by feature_utils and shapiq."""
        _map: dict[str, Literal["xgb", "tabpfn"]] = {
            "xgboost": "xgb",
            "tabpfn": "tabpfn",
        }
        return _map[self.value]


class DataSource(str, enum.Enum):
    """RAG data sources."""

    CTG = "ctg"
    PUBMED = "pubmed"
    FAERS = "faers"
    AACT = "aact"
    CHEMBL = "chembl"
    PRIMEKG = "primekg"
    DRUGSFDA = "drugsfda"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class LLMConfig(BaseSettings):
    """LLM backbone configuration (Claude Opus 4.6 via DSPy).

    Uses LiteLLM model ID format (``provider/model-name``) for DSPy
    compatibility.  The primary model (Opus) is used for proposer, planner,
    and evaluator agents.  The budget model (Sonnet) is used for the
    high-volume feature builder agent to reduce cost.
    """

    model_config = SettingsConfigDict(env_prefix="CTRA_LLM_")

    # Primary model for proposer / planner / evaluator agents
    model_name: str = "anthropic/claude-opus-4-6"
    # Budget model for high-volume feature builder agent
    budget_model_name: str = "anthropic/claude-sonnet-4-6"

    api_key_env_var: str = "ANTHROPIC_API_KEY"
    max_tokens: int = 16_384
    temperature: float = 0.0

    # Prompt caching -- reuses system prompts across MCTS iterations.
    # Anthropic prompt caching reduces input cost by ~90% for repeated
    # system messages, which is significant during MCTS rollouts where
    # the same DSPy Signatures are called repeatedly.
    cache_control: bool = True

    # Cost tracking (USD per million tokens -- Anthropic API pricing)
    # Primary model (Opus on Bedrock)
    cost_per_mtok_input: float = 5.0
    cost_per_mtok_output: float = 25.0
    # Budget model (Sonnet on Bedrock)
    budget_cost_per_mtok_input: float = 3.0
    budget_cost_per_mtok_output: float = 15.0


class RAGConfig(BaseSettings):
    """LinearRAG retrieval configuration."""

    model_config = SettingsConfigDict(env_prefix="CTRA_RAG_")

    index_dir: Path = Path("datasets/linearrag-index")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ner_model: str = "gliner_bio"
    top_k: int = 30
    # Note: LinearRAG uses its own internal BFS parameters for graph traversal.
    # PPR alpha/iterations were removed as they duplicated LinearRAG internals.
    date_filter_enabled: bool = True
    sources: list[DataSource] = Field(
        default_factory=lambda: [
            DataSource.CTG,
            DataSource.PUBMED,
            DataSource.CHEMBL,
            DataSource.FAERS,
            DataSource.AACT,
            DataSource.PRIMEKG,
            DataSource.DRUGSFDA,
        ]
    )

    # GLiNER-BioMed NER configuration (used when ner_model == "gliner_bio").
    # Labels are zero-shot natural language prompts passed to the GLiNER
    # model (Ihor/gliner-biomed-large-v1.0).  The model computes similarity
    # between label text and candidate entity spans, so phrasing matters.
    # Title Case matches the model's training convention.
    #
    # Label design decisions (from GLiNER-BioMed benchmark analysis):
    # - "Gene or protein" merged: BioASQ showed Gene vs Protein causes high
    #   confusion; single label validated in BioNLP13CG
    # - No "Chemical": Drug vs Chemical is the #1 confusion pair (BioASQ);
    #   non-drug chemicals are low-value for trial outcome prediction
    # - Uni-encoder degrades beyond ~30 labels; 16 labels is well within limit
    ner_labels: list[str] = Field(
        default_factory=lambda: [
            # Core pharmacological
            "Drug",
            "Mechanism of action",
            "Drug dosage",
            # Disease & clinical
            "Disease",
            "Symptom",
            "Adverse event",
            # Biological
            "Gene or protein",
            "Biomarker",
            "Cell type",
            "Anatomical structure",
            # Clinical trial
            "Clinical endpoint",
            "Therapeutic procedure",
            "Diagnostic test",
            "Patient population",
            # Organizational
            "Organization",
            "Time period",
        ]
    )
    ner_threshold: float = 0.4


class DataConfig(BaseSettings):
    """Paths to data artifacts."""

    model_config = SettingsConfigDict(env_prefix="CTRA_DATA_")

    base_dir: Path = Path("datasets")
    ctg_parquet: Path = Path("datasets/ctg-studies.parquet")
    internal_trials_parquet: Path = Path("datasets/internal-trials.parquet")
    pubmed_parquet: Path = Path("datasets/pubmed-cleaned.parquet")
    faers_parquet: Path = Path("datasets/faers.parquet")
    tasks_dir: Path = Path("tasks")

    # AACT (Aggregate Analysis of ClinicalTrials.gov)
    aact_parquet: Path = Path("datasets/aact-studies.parquet")
    aact_csv_dir: Path = Path("datasets/aact-csv")

    # ChEMBL (drug mechanisms/targets)
    chembl_parquet: Path = Path("datasets/chembl.parquet")
    chembl_sqlite_path: Path = Path("datasets/chembl_35.db")

    # PrimeKG (biological knowledge graph)
    primekg_parquet: Path = Path("datasets/primekg.parquet")

    # Drugs@FDA (FDA approval history via openFDA)
    drugsfda_parquet: Path = Path("datasets/drugsfda.parquet")
    drugsfda_api_base: str = "https://api.fda.gov/drug/drugsfda.json"

    # Entity resolution (not a RAG source — preprocessing only)
    pubchem_synonyms_parquet: Path = Path("datasets/pubchem-synonyms.parquet")

    # ClinicalTrials.gov API
    ctg_api_base: str = "https://clinicaltrials.gov/api/v2"

    # PubMed E-utilities
    pubmed_api_base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    pubmed_api_key_env_var: str = "NCBI_API_KEY"
    pubmed_rate_limit: float = 3.0  # requests/sec without API key

    # OpenFDA (FAERS + Drugs@FDA — shared API key and rate limit)
    openfda_api_base: str = "https://api.fda.gov/drug/event.json"
    openfda_api_key_env_var: str = "OPENFDA_API_KEY"
    openfda_rate_limit: float = 4.0  # requests/sec (240/min with key)


class ModelConfig(BaseSettings):
    """Downstream classifier configuration."""

    model_config = SettingsConfigDict(env_prefix="CTRA_MODEL_")

    classifiers: list[ClassifierType] = Field(
        default_factory=lambda: [ClassifierType.XGBOOST, ClassifierType.TABPFN]
    )
    shap_enabled: bool = True
    shap_max_samples: int = 100  # max samples for SHAP computation

    # shapiq interaction value computation
    shapiq_max_order: int = 2  # pairwise interactions by default
    shapiq_max_samples: int = (
        50  # max validation samples to explain (higher = slower but more stable)
    )
    shapiq_budget: int = 2048  # approximation budget for TabPFN (ignored for XGBoost TreeExplainer)

    # XGBoost defaults
    xgb_n_estimators: int = 300
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_early_stopping_rounds: int = 20

    # TabPFN defaults
    tabpfn_device: str = "cuda"  # CUDA required for TabPFN
    tabpfn_n_estimators: int = 8
    tabpfn_fit_mode: str = "fit_with_cache"  # use KV-cache for MCTS rollouts
    tabpfn_random_state: int = 42
    tabpfn_finetune: bool = False  # enable fine-tuning (requires A100/H100)


class MCTSConfig(BaseSettings):
    """MCTS search configuration for feature set optimization."""

    model_config = SettingsConfigDict(env_prefix="CTRA_MCTS_")

    num_rollouts: int = 20
    max_depth: int = 10
    exploration_constant: float = 1.414  # UCT C_p

    # Multi-objective (Pareto)
    objectives: list[Literal["accuracy", "parsimony"]] = Field(
        default_factory=lambda: ["accuracy", "parsimony"]  # type: ignore[arg-type]
    )

    # Maximum feature count (used for parsimony normalization)
    max_features: int = 50

    # Pareto hypervolume reference point (worst-case values, all objectives
    # are "higher is better" and normalized to [0, 1])
    reference_point: list[float] = Field(default_factory=lambda: [0.0, 0.0])

    # AB-MCTS adaptive branching
    adaptive_branching: bool = True
    min_branch_factor: int = 2
    max_branch_factor: int = 8

    # Deep rollout simulation (AutoCT-style)
    # When enabled, each rollout expands and evaluates nodes from the selected
    # leaf all the way to max_depth, creating a full depth-first path.
    # The reward backpropagated is the best found on the entire path.
    # When disabled, each rollout evaluates a single node (shallow mode).
    deep_simulation: bool = True

    # Subprocess execution
    subprocess_timeout: int = 3600
    """Timeout in seconds for each subprocess agent evaluation."""

    # Legacy dir kept for agent_cache derivation in runner.py (runner.py:110
    # uses feature_cache_dir.parent / "agent_cache").
    feature_cache_dir: Path = Path("output/feature_cache")

    # Global feature value store (cross-branch reuse via per-feature granularity)
    feature_store_enabled: bool = True
    feature_store_dir: Path = Path("output/feature_store")


class IngestionConfig(BaseSettings):
    """Data ingestion service configuration."""

    model_config = SettingsConfigDict(env_prefix="CTRA_INGESTION_")

    state_file: Path = Path("datasets/.ingestion_state.json")
    ctg_page_size: int = 100
    ctg_rate_limit: float = 10.0  # requests per second
    pubmed_batch_size: int = 500
    faers_rate_limit: float = 4.0  # requests per second (240/min with key)
    openfda_api_key: str | None = None


class MLOpsConfig(BaseSettings):
    """MLOps: experiment tracking, model store, monitoring, and retraining."""

    model_config = SettingsConfigDict(env_prefix="CTRA_MLOPS_")

    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "ctra"
    model_store_dir: Path = Path("models")
    monitoring_dir: Path = Path("monitoring")
    retrain_on_drift: bool = False
    drift_check_interval_hours: int = 24


class APIConfig(BaseSettings):
    """FastAPI service configuration."""

    model_config = SettingsConfigDict(env_prefix="CTRA_API_")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    log_level: str = "info"
    api_key: str | None = None  # Optional bearer-token authentication
    rate_limit_per_minute: int = 60


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root configuration aggregating all sub-configs.

    Load order (highest priority first):
        1. Environment variables prefixed ``CTRA_``
        2. ``.env`` file in the project root
        3. Defaults defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="CTRA_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Project metadata
    project_name: str = "ctra"
    version: str = "0.1.0"
    debug: bool = False
    output_dir: Path = Path("output")

    # Sub-configs
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mcts: MCTSConfig = Field(default_factory=MCTSConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    api: APIConfig = Field(default_factory=APIConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
