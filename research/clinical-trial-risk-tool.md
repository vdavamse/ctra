
# Clinical Trial Risk Tool (CTRA)

## Project Details
- **Repository:** `/mnt/c/Users/sanjor11/projects/crra/clinical_trial_risk`
- **Paper:** "Clinical Trial Risk Tool: software application using natural language processing to identify the risk of trial uninformativeness" by Thomas A. Wood & Douglas McNair, Gates Open Research 2023
- **Live Tool:** https://app.clinicaltrialrisk.org
- **Source Code:** https://github.com/fastdatascience/clinical_trial_risk
- **License:** MIT
- **Funded by:** Bill & Melinda Gates Foundation [INV-050345]

## Context & Background

A majority of clinical trials end without delivering useful results -- a problem called **"uninformativeness"**. Only ~25% of RCTs inform clinical practice (Hutchinson et al., 2022). This wastes resources and raises ethical concerns per the Declaration of Helsinki.

### Zarin et al. (2019) Criteria for an Informative Trial
1. Study hypothesis addresses an important, unresolved question
2. Study designed to provide evidence related to the question
3. Study is feasible
4. Conducted and analyzed in a scientifically valid manner
5. Results reported accurately and promptly

### Common Causes of Uninformativeness
- Underpowering / inadequate sample size
- Lack of Statistical Analysis Plan (SAP)
- Safety and commercial factors

## What the Tool Does

A browser-based NLP application that:
1. Accepts a clinical trial protocol PDF (drag & drop)
2. Parses text via Apache Tika
3. Extracts 8 key features using ML/NLP
4. Feeds features into a linear risk model
5. Outputs HIGH / MEDIUM / LOW risk rating
6. Generates PDF/Excel report

Currently focused on HIV and TB trials.

## 8 Extracted Features & ML Techniques

| Feature | Technique |
|---|---|
| Pathology (HIV/TB/Other) | Naive Bayes (3-class) |
| Trial phase | Ensemble: CNN + Random Forest |
| SAP present? | Naive Bayes (binary) |
| Effect estimate disclosed? | Rule-based + Naive Bayes |
| Number of subjects (sample size) | Rule-based + Random Forest |
| Number of arms | Ensemble: spaCy + Random Forest |
| Countries of investigation | Ensemble: regex + CNN + Random Forest |
| Simulation for sample size? | Naive Bayes (page-level) |

## Risk Scoring Formula

```
score = 26*SAP + 16*effect_estimate + 10*sample_size_tertile
      + 10*international + 10*simulation + 5*phase + 2*arms - 7
```

- **>=50** --> LOW risk
- **40-49** --> MEDIUM risk
- **<40** --> HIGH risk

SAP is the strongest predictor (weight=26). A trial lacking an SAP is extremely unlikely to succeed.

## Implementation Notes

### Tech Stack
- **Language:** Python 3.9
- **Frontend:** Plotly Dash 2.4.1
- **NLP:** spaCy 3.7.2, NLTK, scikit-learn 1.1.1
- **PDF Parsing:** Apache Tika (Java server)
- **Infrastructure:** Docker (two containers: frontend + Tika)
- **Auth:** Auth0
- **Deployment options:** Docker, Google App Engine, Heroku, Azure

### Repository Structure

```
clinical_trial_risk/
├── front_end/
│   ├── application.py          # Main Dash app entry point
│   ├── layout/body.py          # All UI components
│   ├── processors/             # 23 NLP extractor modules
│   ├── util/
│   │   ├── protocol_master_processor.py  # Orchestrates NLP pipeline
│   │   ├── risk_assessor.py              # Linear risk score
│   │   ├── pdf_report_generator.py       # PDF report output
│   │   └── auth0.py                      # OAuth2
│   ├── models/                 # Pre-trained .pkl.bz2 + spaCy models
│   ├── tests/                  # Unit tests for extractors
│   ├── requirements.txt
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/                       # ClinicalTrials.gov pipeline
├── train/                      # ML training scripts & notebooks
├── validation/                 # Validation notebooks
└── notebooks/                  # Exploratory experiments
```

### Pre-trained Models

All stored in `front_end/models/` as `.pkl.bz2` compressed pickles plus spaCy model directories:
- `condition_classifier.pkl.bz2`
- `sap_classifier.pkl.bz2`
- `effect_estimate_classifier.pkl.bz2`
- `num_subjects_classifier.pkl.bz2`
- `phase_rf_classifier.pkl.bz2`
- `arms_classifier_document_level.pkl.bz2`
- `country_ensemble_model.pkl.bz2`
- `simulation_classifier.pkl.bz2`
- spaCy models: `spacy-textcat-phase-04-model-best/`, `spacy-textcat-arms-21-model-best/`, `spacy-textcat-country-16-model-best/`

## Validation Results

### Manual Dataset (300 protocols)

| Component | Accuracy | AUC |
|---|---|---|
| Condition (pathology) | 88% | 100% |
| SAP | 85% | 87% |
| Effect Estimate | 73% | 95% |
| Number of Subjects | 69% | N/A |
| Simulation | 94% | 98% |

### ClinicalTrials.gov Dataset (11,925 protocols)

| Component | Accuracy |
|---|---|
| Phase | 75% |
| SAP | 82% |
| Number of Subjects | 13% (noisy gold standard) |
| Number of Arms | 58% |
| Countries | 87% AUC |

### Hutchinson et al. Dataset (6 protocols)
- 100% AUC

## Local Testing
- Docker-based: `docker-compose up` spins up frontend + Tika containers
- Requires Java for Apache Tika server
- See `front_end/Dockerfile` and `front_end/docker-compose.yml`

## Use Cases
1. **Triage** -- funders quickly filter high-risk protocols
2. **Standardization** -- consistent reviewer calibration
3. **Pre-submission vetting** -- investigators self-check before submitting
4. **Training** -- upskilling junior reviewers
5. **Auto-populate risk questionnaire** -- via API
6. **Domain adaptation** -- fork and extend to oncology, cost estimation

## Blockers & Questions

### Known Limitations
- Sample size extraction is weakest component (13% on ClinicalTrials.gov due to noisy labels)
- Only trained for HIV and TB pathologies
- Linear risk model weights set by expert judgment, not data-driven
- Small risk model validation set (only 6 protocols with informativeness ground truth)

### Future Work
- Expand to more pathologies (oncology, cardiovascular)
- NCT# lookup --> auto-retrieve sample size from ClinicalTrials.gov API
- Number of sites, primary duration, number of endpoints
- Prevalence estimate detection
- Platform/master protocol detection
- Multi-document support (separate Protocol + SAP PDFs)
- Batch processing of multiple protocols
- Expose as REST API/library
- Case management system integration

## Learnings
<!-- What did you learn? Link to permanent notes if reusable -->


## Related Work
<!-- Links to related work notes -->
- [Literature Review 2023-2026](./literature-review-2023-2026.md) — Literature review of 20+ related papers (2023–2026)
- [SOTA Architecture](../README.md) — Synthesized SOTA architecture combining MEXA-CTP, LIFTED, CLaDMoP, and Docling for next-gen clinical trial risk assessment


## Citations

**Total citations (as of March 2026): 3** (OpenAlex) — 2 distinct academic citing works

### Citing Paper 1 — Scoping Review (Published + Preprint)

**Title:** A Scoping Review of Artificial Intelligence Applications in Clinical Trial Risk Assessment
**Authors:** Douglas Teodoro, Nona Naderi, Anthony Yazdani, Boya Zhang, Alban Bornet
**Affiliations:** University of Geneva; Université Paris-Saclay
**Journal:** npj Digital Medicine — published July 30, 2025
**DOI:** 10.1038/s41746-025-01886-7 | PubMed: 40731070
**Preprint:** medRxiv, Jan 22, 2025 — DOI: 10.1101/2025.01.21.25320310
**Context:** Comprehensive scoping review of 142 studies (2013–2024) on AI in clinical trial risk assessment. Wood & McNair cited among 184 references in the category of NLP tools applied to trial protocols.

### Citing Paper 2 — Cost Drivers Study

**Title:** Cost Drivers and Predictive Modeling of Clinical Trial Costs: Analysis of 101 Global Health Trials
**Authors:** Theresia Yiallourou, Thy Pham, Lindsey Baker, G.S. Dissanayake, Joshua L. Proctor, Hil Lyons et al.
**Venue:** VeriXiv (preprint), 2025
**DOI:** 10.12688/verixiv.1282.1
**Context:** Analyzes cost drivers across 101 global health trials and builds cost prediction models. Cites CTRA paper as related methodology for extracting protocol features. Cross-referenced on clinicaltrialrisk.org — both within the Gates Foundation / Fast Data Science ecosystem.

### Non-Academic Mention

**DAC Best Practices website** (dac-trials.org/resources/key-publications/) — Bill & Melinda Gates Foundation's Design, Analyze, Communicate initiative lists the paper as item #15 in their curated key publications list.

### Citation Metrics

| Database | Count |
|---|---|
| OpenAlex | 3 (preprint + published counted separately) |
| Semantic Scholar | 1 |
| PubMed / Europe PMC | Not indexed |

> OpenAlex: paper is at the **74th percentile** for citation impact in its field. Main discoverability gap: not indexed in PubMed.

---
Tags: #work #crra #nlp #clinical-trials #python
