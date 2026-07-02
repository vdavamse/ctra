# Large Language Model Analysis of Reporting Quality of Randomized Clinical Trial Articles: A Systematic Review

## Authors and Affiliations

- **Apoorva Srinivasan, MS** -- Department of Computational Biomedicine, Cedars-Sinai Medical Center, Los Angeles, CA
- **Jacob Berkowitz, BS** -- Department of Computational Biomedicine, Cedars-Sinai Medical Center, Los Angeles, CA
- **Nadine A. Friedrich, MD** -- Department of Computational Biomedicine, Cedars-Sinai Medical Center, Los Angeles, CA
- **Sophia Kivelson, MS** -- Department of Computational Biomedicine, Cedars-Sinai Medical Center, Los Angeles, CA
- **Nicholas P. Tatonetti, PhD** -- Department of Computational Biomedicine, Cedars-Sinai Medical Center, Los Angeles, CA; Cedars-Sinai Cancer, Cedars-Sinai Medical Center, Los Angeles, CA

**Corresponding Authors:** Nicholas P. Tatonetti (Nicholas.Tatonetti@cshs.org) and Apoorva Srinivasan (Apoorva.Srinivasan@cshs.org)

## Venue/Journal

- **Journal:** JAMA Network Open
- **Year:** 2025
- **Volume/Issue:** 8(8):e2529418
- **DOI:** [10.1001/jamanetworkopen.2025.29418](https://doi.org/10.1001/jamanetworkopen.2025.29418)
- **Published:** August 28, 2025
- **Accepted:** July 2, 2025
- **Type:** Original Investigation -- Health Informatics (Systematic Review)
- **License:** CC-BY (Open Access)

## Abstract

Incomplete reporting in randomized clinical trials (RCTs) obscures bias and limits reproducibility. Manual audits for adherence to the Consolidated Standards of Reporting Trials (CONSORT) guideline cannot keep pace with publication volume. This study builds and validates a zero-shot large-language-model (LLM) pipeline for automated CONSORT assessment and maps reporting quality over time, biomedical disciplines, and trial features. Of 53,137 screened PDFs, 21,041 RCTs (median publication year 2014; 30 disciplines) were included. In the 70-article validation set (2,210 decisions), LLM outputs matched experts 91.7% of the time; the macro F1 score on CONSORT-TM was 0.86 (95% CI, 0.84-0.87). Mean CONSORT compliance increased from 27.3% (1966-1990) to 57.0% (2010-2024). However, critical elements remained poorly reported, including allocation-concealment mechanism (16.1%) and external-validity discussion (1.6%). Compliance varied across disciplines from 35.2% (pharmacology) to 63.4% (urology), with only negligible associations with clinical trial characteristics (all Cramer V < 0.10).

## Problem Statement

Randomized clinical trials are the cornerstone of evidence-based medicine, but many have methodological flaws and biased results. Poor reporting obscures these biases, complicates replication, and undermines the trustworthiness of biomedical science. The CONSORT statement (first published 1996, updated 2001 and 2010) was created to improve reporting completeness, but adherence remains inconsistent. Traditional manual CONSORT compliance assessments are limited to small samples and cannot scale to the volume of published RCTs. Previous automated approaches using rule-based algorithms or traditional ML performed poorly, and prior LLM attempts achieved only F1 = 0.51 in zero-shot settings. A scalable, accurate automated assessment method is needed.

## Key Contributions

1. Developed a **zero-shot LLM framework** (using GPT-4o-mini) for automated CONSORT compliance assessment that achieved state-of-the-art performance (macro F1 = 0.86), exceeding prior systems by 40+ percentage points.
2. Validated the pipeline via **human expert review** of 70 randomly sampled articles (91.7% agreement rate, Cohen kappa = 0.64).
3. Conducted the **largest-scale analysis of CONSORT reporting quality** to date, covering 21,041 open-access RCTs spanning 1966-2024 across 30 biomedical disciplines.
4. Identified **persistent reporting gaps** in critical methodological items (e.g., allocation concealment 16.1%, external validity discussion 1.6%, protocol access 2.2%) despite overall improvement over time.
5. Mapped **disciplinary variation** in compliance (35%-63%) and found that trial-level factors (FDA regulation, phase, funding) had negligible practical associations with reporting quality (Cramer V < 0.10).
6. Introduced a **confidence-filtering mechanism** where high-confidence LLM outputs (90.8% of all judgments) achieved macro F1 = 0.95.

## Methodology/Architecture

### Pipeline Design
- **PDF Processing:** PubMed open-access PDFs were converted to XML using PyMuPDF v1.26.3.
- **Metadata Enrichment:** Articles were linked with Semantic Scholar and ClinicalTrials.gov metadata; Scimago journal rankings used for discipline classification.
- **LLM Assessment:** Each CONSORT criterion was assessed independently per article. The entire article content was fed into the model one criterion at a time in a zero-shot fashion.
- **Prompt Design:** Chain-of-thought reasoning prompts yielded JSON output with 4 elements: criterion, rationale, decision (MET/NOT MET), and confidence (Low/Medium/High).

### Models Tested
| Model | Description |
|-------|-------------|
| LLM 1 (GPT-4) | Highest overall performance: macro F1 = 0.89 |
| LLM 2 (GPT-4o) | Macro F1 = 0.84 |
| LLM 3 (GPT-4o-mini) | Best speed-to-accuracy trade-off: macro F1 = 0.86, precision = 0.97 -- **selected for deployment** |
| LLM 4 (Llama-2-7B-chat) | Lowest: macro F1 = 0.74 |

All models were used in zero-shot mode. Proprietary models accessed via Azure HIPAA-compliant endpoint; Llama-2 run locally.

### Validation
- **Benchmark:** 50-article CONSORT-TM corpus (sentence-level annotations with 37 items).
- **Human Validation:** 70 articles stratified across 30 specialties and 4 time periods, reviewed by 4 experts (1 clinician, 3 data scientists). Results: 81.2% correct, 10.4% partially correct, 8.4% incorrect.
- **Confidence Filtering:** High-confidence decisions (90.8% of total) showed macro F1 = 0.95; medium-confidence decisions were unreliable (F1 = 0.31) and excluded from final analyses.
- **Four CONSORT items excluded** due to systematic misclassification of absent-event items: 3b (method changes), 6b (outcome changes), 7b (interim analyses), 14b (reasons for termination).

### Statistical Analysis
- Wilson method for 95% CIs on proportions
- Chi-squared tests (Fisher exact when expected cell count < 5) for group differences
- Cramer V for effect sizes
- Pearson correlation for continuous measures
- Python 3.8 with pandas 2.0, SciPy 1.10, statsmodels 0.14

## Datasets

| Dataset | Size | Description |
|---------|------|-------------|
| CONSORT-TM corpus | 50 articles | Benchmark dataset with sentence-level annotations for 37 CONSORT items |
| Full analytic corpus | 21,041 RCTs | Open-access human RCTs from PubMed (1966-2024) across 30 specialties |
| ClinicalTrials.gov subset | 1,790 articles | Subset with NCT numbers linked to trial characteristics |
| Human validation set | 70 articles | Stratified random sample for expert review |
| Total criteria evaluated | 886,788 items | 21 CONSORT items x 21,041 articles (after exclusions) |

**Time periods:** 1966-1990 (2,771 articles), 1990-2000 (1,969), 2000-2010 (3,765), 2010-2024 (10,447).

## Results

### Model Performance (CONSORT-TM Benchmark)

| Model | Accuracy | Precision | Recall | Macro F1 |
|-------|----------|-----------|--------|----------|
| GPT-4 (LLM 1) | 0.84 | 0.93 | 0.85 | 0.89 |
| GPT-4o (LLM 2) | 0.78 | 0.94 | 0.75 | 0.84 |
| GPT-4o-mini (LLM 3) | 0.81 | 0.97 | 0.77 | **0.86** |
| Llama-2-7B (LLM 4) | 0.67 | 0.90 | 0.63 | 0.74 |
| Prior SOTA (Jiang et al. 2024) | N/A | 0.48 | 0.54 | 0.51 |

### Temporal Trends in CONSORT Compliance
- **1966-1990:** 27.3% (95% CI, 27.0%-27.6%)
- **1990-2000:** 33.9% (95% CI, 33.5%-34.3%) -- 24.3% relative increase
- **2000-2010:** 45.0% (estimated from trend)
- **2010-2024:** 57.0% (95% CI, 56.8%-57.2%) -- 26.7% relative increase

### Best- and Worst-Reported Items
- **Best:** Scientific background/rationale (95.9%), Objectives/hypotheses (89.2%)
- **Worst:** External validity discussion (1.6%), Protocol access (2.2%), Allocation concealment (16.1%), Randomization type (7.5%)

### Disciplinary Variation
- **Highest compliance:** Urology/nephrology (63.4%), Critical care (62.3%)
- **Lowest compliance:** Pharmacology (35.2%), Radiology (40.5%)

### Trial Characteristics
- Phase 2 trials had highest compliance (66.6%); Phase 1 lowest (59.0%)
- European trials highest by continent (67.2%); North American second (63.8%)
- All trial-level factor associations were negligible (Cramer V < 0.10)

## Limitations

1. **LLM hallucination risk** -- mitigated via confidence filtering and expert validation, but further refinement of uncertainty quantification is needed.
2. **Open-access bias** -- analysis limited to open-access articles, which may not represent all published RCTs.
3. **Presence vs. quality** -- assessed whether reporting elements were present, not their quality or accuracy.
4. **Evolving standards** -- CONSORT did not exist before 1996, so lower compliance in earlier eras reflects absence of guidance rather than non-adherence.
5. **Single-document limitation** -- only assessed principal results manuscripts; many trials report methods in separate protocol/rationale articles or on registry websites.
6. **Excluded items** -- 4 CONSORT items (3b, 6b, 7b, 14b) were dropped due to systematic misclassification of absent events, which may inflate overall performance estimates.

## Relevance to CTRA

This paper is highly relevant to the CTRA project in several ways:

- **Shared focus on protocol/reporting quality:** Both this paper and CTRA address the quality and completeness of clinical trial documentation. While CTRA predicts protocol uninformativeness prospectively, this paper assesses reporting completeness retrospectively using CONSORT guidelines.
- **LLM-based assessment of trial documents:** The zero-shot LLM pipeline demonstrated here validates the approach of using large language models to automatically assess clinical trial text at scale -- a methodology directly applicable to CTRA's goal of parsing protocol text to identify risk factors.
- **Identification of reporting gaps as risk indicators:** The persistent gaps found (e.g., missing allocation concealment, sample size justification, statistical methods) overlap with features that CTRA's risk model considers when predicting uninformativeness. Poor reporting of these items in published results may correlate with their absence in protocols.
- **Scalability demonstration:** The paper shows that LLM-based assessment can work at scale (21,000+ articles), supporting the viability of automated protocol assessment systems like CTRA.
- **Complementary perspective:** This paper assesses published articles (post-hoc), while CTRA operates at the protocol stage (prospective). Together, they address the trial quality pipeline from both ends.

## Code/Data Availability

- **No public code repository** mentioned for the pipeline itself.
- Data sharing statement referenced in Supplement 2.
- Pre-training corpus sourced from PubMed open-access articles.
- CONSORT-TM benchmark dataset referenced from Kilicoglu et al. (2021).
- Funded by NIH grant R35GM131905.

## Citation

Srinivasan A, Berkowitz J, Friedrich NA, Kivelson S, Tatonetti NP. Large Language Model Analysis of Reporting Quality of Randomized Clinical Trial Articles: A Systematic Review. *JAMA Netw Open.* 2025;8(8):e2529418. doi:10.1001/jamanetworkopen.2025.29418
