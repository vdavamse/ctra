# Open Questions

## 1. Generalization to Internal Data

Academic models are trained and evaluated on the TrialBench dataset (derived from ClinicalTrials.gov public data). Internal trial data may differ in:
- Disease distribution (concentrated in specific therapeutic areas)
- Label quality (internal outcome tracking vs. public registry reporting)
- Data completeness (richer internal data vs. abbreviated public records)
- Schema differences (internal trial management systems may not map cleanly to ClinicalTrials.gov `protocolSection` format used by the feature building agents)

The internal data alignment ETL addresses this with a dedicated pipeline to map internal trial data to ClinicalTrials.gov's `protocolSection` format.

## 2. Biologics and Non-Small-Molecule Trials

Deep learning SOTA models encode drugs via SMILES molecular fingerprints — this only works for small-molecule drugs. Biologics (antibodies, cell therapies), devices, and behavioral interventions have no SMILES representation. AutoCT's text-based feature engineering partially addresses this, since LLM agents are not restricted to SMILES encodings and can construct features from any textual trial description. However, this is untested for non-small-molecule interventions and needs validation.

## 3. Temporal Validity and Concept Drift

Models trained on historical trial data may not generalize to future trials as:
- Standard of care evolves (a comparator arm that was appropriate in 2020 may not be in 2026)
- Regulatory requirements change (ICH E9(R1) estimand framework, evolving FDA guidance)
- Trial design patterns shift (adaptive designs, decentralized trials, platform trials)

Periodic retraining on recent data is necessary.

## 4. Interpretability vs. Accuracy Tradeoff — Resolved

**Decision:** AutoCT selected as primary approach. Interpretability via SHAP on human-readable features is critical for clinical decision-making and stakeholder communication. AutoCT achieves competitive accuracy (highest Phase I PR-AUC on TrialBench) while providing full per-prediction transparency. Deep learning models (MEXA-CTP, LIFTED) remain as accuracy benchmarks. Jain & Wallace showed that attention weights — the primary "interpretability" mechanism in deep learning models — do not reliably explain predictions, reinforcing the choice of SHAP on explicit features.

## 5. Multi-Document Protocol Challenge

Real-world trial assessment may involve multiple documents (protocol, SAP, IB, amendments). Current models use only ClinicalTrials.gov structured fields. Extending to full-document analysis requires:
- PDF parsing (Docling)
- Long-context encoding (Clinical-Longformer)
- Section-level feature extraction
