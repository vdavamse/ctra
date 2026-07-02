# A Scoping Review of Artificial Intelligence Applications in Clinical Trial Risk Assessment

## Authors and Affiliations

- **Douglas Teodoro** (corresponding) -- Department of Radiology and Medical Informatics, Faculty of Medicine, University of Geneva, Geneva, Switzerland
- **Nona Naderi** -- Laboratoire Interdisciplinaire des Sciences du Numerique, CNRS, Universite Paris-Saclay, Gif-sur-Yvette, France
- **Anthony Yazdani** -- Department of Radiology and Medical Informatics, Faculty of Medicine, University of Geneva, Geneva, Switzerland
- **Boya Zhang** -- Department of Radiology and Medical Informatics, Faculty of Medicine, University of Geneva, Geneva, Switzerland
- **Alban Bornet** -- Department of Radiology and Medical Informatics, Faculty of Medicine, University of Geneva, Geneva, Switzerland

## Venue/Journal

- **Journal:** npj Digital Medicine (published in partnership with Seoul National University Bundang Hospital)
- **Year:** 2025
- **Volume/Article:** (2025) 8:486
- **DOI:** [https://doi.org/10.1038/s41746-025-01886-7](https://doi.org/10.1038/s41746-025-01886-7)

## Abstract

Artificial intelligence (AI) is increasingly applied to clinical trial risk assessment, aiming to improve safety and efficiency. This scoping review analyzed 142 studies published between 2013 and 2024, focusing on safety (n=55), efficacy (n=46), and operational (n=45) risk prediction. AI techniques, including traditional machine learning, deep learning (e.g., graph neural networks, transformers), and causal machine learning, are used for tasks like adverse drug event prediction, treatment effect estimation, and phase transition prediction. These methods utilize diverse data sources, from molecular structures and clinical trial protocols to patient data and scientific publications. Recently, large language models (LLMs) have seen a surge in applications, featuring in 7 out of 33 studies in 2023. While some models achieve high performance (AUROC up to 96%), challenges remain, including selection bias, limited prospective studies, and data quality issues. Despite these limitations, AI-based risk assessment holds substantial promise for transforming clinical trials, particularly through improved risk-based monitoring frameworks.

## Problem Statement

The literature lacked a comprehensive scoping review highlighting the specific role of AI in the assessment of risks in clinical trials at large, particularly in light of recent advances in the field of machine learning. Clinical trial risks encompass safety (participant well-being), efficacy (ability to deliver intended benefits), and operational effectiveness (smooth execution up to drug approval). Identifying and mitigating these diverse risks is crucial but challenging due to issues around participant safety, maintaining trial integrity, and meeting regulatory obligations.

## Key Contributions

1. Conducted a systematic scoping review of 142 studies (2013--2024) on AI-based clinical trial risk assessment, following PRISMA-ScR guidelines.
2. Categorized AI applications into three risk types -- safety, efficacy, and operational -- with 10 sub-categories (ADE, severity, toxicity, drug response, outcome, survival, treatment effect, likelihood of approval, phase success, and other operational risks).
3. Mapped the landscape of AI algorithms used: traditional ML (dominant, n=61 using random forest), deep learning (GNNs n=14, transformers n=12, CNNs n=10), survival analysis, causal ML, and nascent approaches (relational learning, quantum ML).
4. Identified the rise of LLMs, accounting for ~20% of studies in 2023, though generative LLM use remains minimal (only 3 studies through July 2024).
5. Analyzed datasets and evaluation strategies, identifying key issues: most studies use AUROC (n=94), which is not robust to class imbalance; safety and operational studies primarily use public data while efficacy studies rely on private data.
6. Documented five key limitations: selection bias, poor evaluation strategies, data quality and availability issues, predominance of retrospective studies, and siloed risk models.
7. Proposed recommendations: more diverse datasets, improved evaluation metrics (F1, MCC), prospective studies, real-time data integration, and multi-task learning to simultaneously predict multiple risk types.

## Methodology/Architecture

**Review methodology:**
- Search conducted between October 2023 and July 2024
- Databases: PubMed (n=1,605), Web of Science (n=1,086), Google Scholar (n=1,628); total 4,328 records
- After de-duplication (1,026 removed) and screening (3,108 excluded), 142 studies were included
- Used PRISMA-ScR guidelines and CHARMS checklist for data extraction
- Four researchers extracted data independently; an independent reviewer normalized results

**Risk taxonomy (based on Badwan et al. framework):**
- **Safety risk** (n=55): ADE prediction (n=18), severity prediction (n=7), toxicity prediction (n=32)
- **Efficacy risk** (n=46): drug response (n=11), outcome (n=13), survival (n=16), treatment effect (n=16)
- **Operational risk** (n=45): likelihood of approval (n=16), phase success (n=23), other (n=9, including enrollment, duration, informativeness)

**AI paradigms identified:**
- Traditional ML (dominant): Random forest (n=61), SVM (n=28), XGBoost (n=26)
- Deep learning: GNNs (n=14), transformers (n=12), CNNs (n=10)
- Survival analysis: Cox proportional hazards, DeepSurv
- Causal ML: counterfactual prediction for treatment effect estimation
- LLMs: encoder-based (BERT for protocol encoding) and generative (GPT-3.5, BART, TWIN-GPT)

## Datasets

| Risk Type | Primary Data Source | Median Size | Public/Private |
|-----------|-------------------|-------------|----------------|
| Safety | SIDER (adverse drug reactions from FAERS) | 1,063 compounds | 20 public / 0 private |
| Efficacy | Individual clinical trials | 1 trial, 1,250 participants | 5 public / 29 private |
| Operational | ClinicalTrials.gov | 17,538 protocols | 33 public / 3 private |

Additional data sources include molecular structure databases (PubChem), genomic data, electronic health records, and synthetic data from virtual/in silico trials.

## Results

**Performance (AUROC metric, selected comparable subsets):**
- **ADE prediction** (SIDER dataset): Top models achieved 96.6% (Masumshah et al.), 93.1% (Zhao et al.), 92.0% (Galeano et al., Zhong et al.)
- **Outcome prediction** (individual trials): Top models achieved 84.0%--87.4% AUROC
- **Phase success** (ClinicalTrials.gov): Ferdowsi et al. achieved 92.3%--92.7% AUROC

**Growth trends:**
- Exponential growth in publications from 2013 to 2024
- Important jump between 2020 and 2021
- US institutions dominate (n=53), followed by China (n=16), UK (n=10)
- 90% published in journals, 10% in conferences
- LLM usage surged to ~20% of studies by 2023

**Conditions studied:** Neoplasms most common (n=29), followed by mental disorders (n=6) and infections (n=5). Most disease-specific studies focused on efficacy (n=41 out of 64).

## Limitations

**Limitations of reviewed studies (identified by authors):**
1. **Selection bias** -- Limited compounds compared to drug-like chemical space (O(10^3) vs O(10^60)); efficacy models evaluated on median of 1 trial
2. **Evaluation strategy** -- Top metrics (AUROC, accuracy, recall) are not robust to imbalanced datasets; a naive classifier achieved 91% AUROC on SIDER
3. **Data quality and availability** -- Operational studies lack real-world trial data; safety studies often ignore dosage and route of administration
4. **Retrospective studies** -- Most studies use only retrospective data; some prospective studies use proprietary data
5. **Siloed risk models** -- Risks assessed separately despite interconnection; only a few studies combine categories

**Limitations of the review itself:**
- Broad AI field makes keyword selection challenging; some relevant studies may have been missed
- Risk categorization guided by a single framework (Badwan et al.)
- Unable to fully compare model effectiveness due to lack of common benchmarks

## Relevance to CTRA

This review is **highly relevant** to the CTRA project in several ways:

1. **Directly covers operational risk prediction**, which includes "informativeness of the protocol" -- the exact target of CTRA (predicting protocol uninformativeness). Reference [165] (Wood & McNair, 2023, "Clinical Trial Risk Tool") specifically addresses trial uninformativeness using NLP.
2. **Maps the full landscape** of AI approaches for clinical trial risk, providing context for where CTRA fits within the broader field -- specifically in the "operational risk / other" sub-category.
3. **Identifies key data sources** relevant to CTRA: ClinicalTrials.gov is the dominant source for operational risk studies, with a median of 17,538 protocols.
4. **Highlights the rise of LLMs** for encoding clinical trial protocols, which is directly applicable to CTRA's approach of analyzing protocol text.
5. **Documents the gap** in holistic risk assessment -- no study integrates safety, efficacy, and operational risks simultaneously, suggesting CTRA could contribute to this research avenue.
6. **Identifies evaluation pitfalls** relevant to CTRA: the importance of using metrics robust to class imbalance (F1, MCC) rather than AUROC alone.
7. **Notes that operational risk studies** tend to be phase-specific and use large representative datasets, which aligns with CTRA's approach.

## Code/Data Availability

- No datasets were generated or analyzed during this study (it is a review).
- No public repository mentioned.
- Supplementary Data 1 contains the full list of records identified, screened, and included.
- PRISMA-ScR checklist provided in Supplementary Table 2.

## Citation

Teodoro, D., Naderi, N., Yazdani, A., Zhang, B., & Bornet, A. (2025). A scoping review of artificial intelligence applications in clinical trial risk assessment. *npj Digital Medicine*, 8, 486. https://doi.org/10.1038/s41746-025-01886-7
