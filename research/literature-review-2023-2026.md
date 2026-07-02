
# Clinical Trial NLP/ML — Literature Review (2023–2026)

Research into the latest papers related to NLP and machine learning applied to clinical trials, with focus on areas similar to the Clinical Trial Risk Tool (CTRT).

## Key Architecture Trend

The field has shifted from **rule-based NLP + classical ML** (CTRT's approach) → **Transformer/BERT + GNN ensembles** → **LLM agents with interpretable outputs**. CTRT's Naive Bayes + linear scoring model represents the prior generation, but its interpretability-first design remains valued.

---

## 1. Protocol Risk Prediction (Most Direct Competitors)

### Deep learning-based risk prediction from protocol design
- **Authors:** Sohrab Ferdowsi, Julien Knafou, Nikolay Borissov, David Vicente Alvarez, Rahul Mishra, Poorya Amini, Douglas Teodoro
- **Venue:** Patterns (Cell Press), Vol. 4, Issue 3
- **Year:** 2023
- **DOI:** 10.1016/j.patter.2023.100689
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10028430/
- **Key Contribution:** First work combining BERT-based transformers with Graph Neural Networks (GNN) to predict clinical trial risk directly from protocol text. Uses protocol amendment history to derive a ternary (low/medium/high) risk label. Ensemble achieves AUROC=0.8453 (ternary) and 0.9234 (binary). Integrated gradients analysis shows "Study Design" and "Contacts and Locations" sections are most predictive.
- **Relevance to CTRT:** Directly parallel — both analyze full protocol document for risk. More sophisticated ML architecture than CTRT's Naive Bayes + linear model.

### Improving clinical trial design using interpretable ML for early trial termination
- **Authors:** Ece Kavalci, Anthony Hartshorn
- **Venue:** Scientific Reports, Vol. 13, Article 121
- **Year:** 2023
- **DOI:** 10.1038/s41598-023-27416-7
- **PMCID:** PMC9813129
- **Key Contribution:** Gradient Boosting on 420,268 ClinicalTrials.gov records + eligibility criteria features. SHAP values provide per-feature explanations. AUROC=0.80 for early termination prediction. Designed as an iterative design feedback tool.
- **Relevance to CTRT:** Closest structural parallel — supervised model on registry-level features producing interpretable risk signal. SHAP-based explainability is a direct enhancement path for CTRT's linear weights.

### ClinicalRisk: A New Therapy-related Clinical Trial Dataset
- **Authors:** Junyu Luo, Zhi Qiao, Lucas Glass, Cao Xiao, Fenglong Ma
- **Venue:** ACM CIKM
- **Year:** 2023
- **DOI:** 10.1145/3583780.3615113
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11005852/
- **Key Contribution:** Labeled dataset of 12,717 trials (7,021 successful / 5,696 failed) with annotated failure reasons: funding, design, enrollment, outcome. Section-level text processing outperforms document-level concatenation.
- **Relevance to CTRT:** Failure reason taxonomy maps closely to CTRT's risk indicators. Dataset is a potential benchmark for extending CTRT beyond binary uninformativeness.

### AutoCT: Automating Interpretable Clinical Trial Prediction with LLM Agents
- **Authors:** Fengze Liu, Haoyu Wang, Joonhyuk Cho, Dan Roth, Andrew W. Lo
- **Venue:** EMNLP 2025
- **arXiv:** 2506.04293
- **Key Contribution:** LLM agents (Monte Carlo Tree Search) autonomously generate, evaluate, and refine tabular features from public trial data, then feed those features into interpretable classical ML models. State-of-the-art performance with transparency. Designed for high-stakes biomedical contexts where black-box models are inappropriate.
- **Relevance to CTRT:** Philosophically closest to CTRT — automated feature engineering from protocol text for interpretable risk scoring. Represents a formal framework for doing what CTRT does.

### ClinicalReTrial: A Self-Evolving AI Agent for Clinical Trial Protocol Optimization
- **Authors:** Sixue Xing, Xuanye Xia, Kerui Wu, Meng Jiang, Jintai Chen, Tianfan Fu
- **Venue:** arXiv preprint
- **Year:** 2026
- **arXiv:** 2601.00290
- **Key Contribution:** Self-evolving agent that iteratively redesigns clinical trial protocols using outcome prediction as a simulation environment. Combines failure diagnosis, safety-aware modification, and candidate evaluation in a closed-loop optimization framework with hierarchical memory. Improved 83.3% of protocols with mean +5.7% success probability gain.
- **Relevance to CTRT:** Extends protocol risk prediction into actionable protocol optimization. Uses outcome prediction (CTRA's core capability) as the reward signal for protocol re-design. Direct reference for CTRA's future protocol suggestions feature.

---

## 2. Trial Outcome Prediction (Multimodal)

### HINT: Hierarchical Interaction Network for Clinical Trial Outcome Predictions
- **Authors:** Tianfan Fu, Kexin Huang, Cao Xiao, Lucas Glass, Jimeng Sun
- **Venue:** Patterns (Cell Press), Vol. 3
- **Year:** 2022 (widely cited 2023–2025)
- **DOI:** 10.1016/j.patter.2022.100445
- **Key Contribution:** Landmark multimodal framework encoding drug molecules (SMILES), disease codes (ICD), and eligibility criteria text into a hierarchical interaction graph. Created the TOP (Trial Outcome Prediction) benchmark dataset. F1 scores: 0.665/0.620/0.847 across phases.
- **Relevance to CTRT:** Established benchmark that later papers all compare against. Demonstrates that adding molecular and disease ontology features substantially improves prediction beyond protocol text alone.

### SPOT: Sequential Predictive Modeling of Clinical Trial Outcome with Meta-Learning
- **Authors:** Zifeng Wang, Cao Xiao, Jimeng Sun
- **Venue:** ACM-BCB 2023
- **arXiv:** 2304.05352
- **Key Contribution:** Clusters multi-source trial data into "topics," generates temporally ordered trial embeddings, applies meta-learning for rapid adaptation. Achieves +21.5% PR-AUC lift on Phase I vs HINT. Topic discovery provides interpretable groupings of trial types.
- **Relevance to CTRT:** Sequential and meta-learning approaches outperform prior models. Topic clustering relevant for CTRT's disease-specific risk stratification.

### LIFTED: Multimodal Clinical Trial Outcome Prediction with Large Language Models
- **Authors:** Wenhao Zheng, Liaoyaqi Wang, Dongshen Peng, Hongxia Xu, Yun Li, Hongtu Zhu, Tianfan Fu, Huaxiu Yao
- **Venue:** EMNLP 2025 Findings
- **arXiv:** 2402.06512
- **Key Contribution:** Mixture-of-Experts (MoE) framework converting all trial modalities into natural language descriptions, then processing with noise-resilient encoders and sparse MoE integrator. State-of-the-art across all three trial phases. First MoE-based approach to multimodal trial outcome prediction.
- **Relevance to CTRT:** "Convert everything to text" approach applicable to CTRT's protocol PDF analysis. MoE integration aligns with multi-feature linear scoring model.

### CTP-LLM: Clinical Trial Phase Transition Prediction Using Large Language Models
- **Authors:** Michael Reinisch, Jianfeng He, Chenxi Liao, Sauleh Ahmad Siddiqui, Bei Xiao
- **Venue:** arXiv preprint
- **Year:** 2024
- **arXiv:** 2408.10995
- **Key Contribution:** First LLM-based framework for CTOP. Fine-tunes GPT-3.5 on 20,000 trials linked to phase advancement data. 67% accuracy across all phases, 75% for Phase III→approval. Recruitment criteria and study descriptions drive success more than drug properties.
- **Relevance to CTRT:** Validates CTRT's core hypothesis that protocol text itself is predictive. PhaseTransition dataset is a valuable benchmarking resource.

---

## 3. Feature Extraction from Protocol Text

### Methodological Information Extraction from RCT Publications
- **Authors:** Linh Hoang, Yingjun Guan, Halil Kilicoglu
- **Venue:** AMIA Annual Symposium Proceedings
- **Year:** 2023
- **PMCID:** PMC10148349
- **Key Contribution:** PubMedBERT + CRF NER for trial methodological characteristics (randomization, blinding, allocation concealment, sample size, trial design). F1=0.90 at document level. Annotated corpus of 70 full-text RCTs released publicly.
- **Relevance to CTRT:** Targets the same SAP/randomization/sample size feature set but using NER rather than document-level classification. Could upgrade CTRT from bag-of-words to structured entity extraction.

### Oncology Efficacy Endpoint Extraction with Deep NLP
- **Authors:** Aline Gendrin-Brokmann et al.
- **Venue:** arXiv preprint
- **Year:** 2023
- **arXiv:** 2311.04925
- **Key Contribution:** Multi-label classification model predicting 25 endpoint-related classes from scientific text. F1=96.4% on test set, 93.7–93.9% on held-out case studies. Strong agreement with subject matter experts.
- **Relevance to CTRT:** Directly addresses one of CTRT's key features. 25-class endpoint taxonomy is richer than CTRT's current binary endpoint features.

### Optimizing Clinical Trial Eligibility Design Using NLP and Real-World Data
- **Authors:** Kyeryoung Lee et al.
- **Venue:** JMIR AI
- **Year:** 2024
- **DOI:** 10.2196/50800 | PMCID: PMC11319878
- **Key Contribution:** BiLSTM-CRF pipeline achieving precision=0.91, recall=0.79, F1=0.83 across 3,281 trials and 6 disease areas. Integrates with EHR data to show how changing individual criteria affects real-world enrollment pool size.
- **Relevance to CTRT:** Extends eligibility criteria parsing toward impact simulation — a natural next step for uninformativeness risk assessment.

### LLM Analysis of RCT Reporting Quality (CONSORT at Scale)
- **Authors:** Apoorva Srinivasan, Jacob Berkowitz, Nadine A. Friedrich, Sophia Kivelson, Nicholas P. Tatonetti
- **Venue:** JAMA Network Open
- **Year:** 2025
- **DOI:** 10.1001/jamanetworkopen.2025.29418 | PMCID: PMC12395317
- **Key Contribution:** Zero-shot GPT-4o-mini pipeline assesses CONSORT compliance across 21,041 open-access RCTs (1966–2024). Macro F1=0.86, 91.7% expert agreement. CONSORT compliance rose from 27% (pre-1990) to 57% (post-2010) but critical items remain underreported.
- **Relevance to CTRT:** LLMs can assess protocol reporting quality at scale with near-expert accuracy — same approach applied to protocol documents (not publications) could automate CTRT's SAP/quality assessment without hand-coded NLP features.

---

## 4. Trial Complexity, Duration & Enrollment

### Clinical Trials Are Becoming More Complex: ML Analysis of 16,000+ Trials
- **Authors:** Nigel Markey, Ben Howitt, Ilyass El-Mansouri et al.
- **Venue:** Scientific Reports, Vol. 14, Article 3514
- **Year:** 2024
- **DOI:** 10.1038/s41598-024-53211-z | PMCID: PMC10861486
- **Key Contribution:** ML-derived "Trial Complexity Score" on 16K+ protocols. Complexity increased >10 percentage points over the past decade. A 10-point increase correlates with ~36% longer trial duration. Oncology consistently has the highest complexity.
- **Relevance to CTRT:** The complexity score is a close cousin to CTRT's uninformativeness risk score. Both use automated feature extraction from protocols.

### TrialDura: Hierarchical Attention Transformer for Clinical Trial Duration Prediction
- **Authors:** Leo Yue, Zihan Li, Tianfan Fu et al.
- **Venue:** ACM-BCB 2024
- **arXiv:** 2404.13235
- **Key Contribution:** BioBERT embeddings + hierarchical attention mechanism predicts trial duration. MAE=1.04 years, RMSE=1.39 years. Hierarchical attention provides interpretability into which features drive duration estimates.
- **Relevance to CTRT:** Duration/timeline is a component of trial informativeness risk. Flagging trials with unrealistic timelines would use exactly this framework.

### TrialEnroll: Predicting Clinical Trial Enrollment Success
- **Authors:** Leo Yue, Zihan Li, Tianfan Fu et al.
- **Venue:** ACM-BCB 2024
- **arXiv:** 2407.13115 | DOI: 10.1145/3698587.3701375
- **Key Contribution:** Deep & Cross Network (DCN) + LLM-derived semantic embeddings of eligibility criteria. PR-AUC=0.7002, ROC-AUC=0.7352. Token-level attention reveals which eligibility criteria words impede enrollment.
- **Relevance to CTRT:** Enrollment failure is a leading cause of trial uninformativeness. Models this risk dimension using eligibility criteria — a core CTRT feature.

---

## 5. Why Trials Stop / Failure Reasons

### Genetic Factors Associated with Reasons for Clinical Trial Stoppage
- **Authors:** David Ochoa, Olesya Razuvayevskaya, Irene Lopez, Ian Dunham et al. (Open Targets / EMBL-EBI)
- **Venue:** Nature Genetics
- **Year:** 2024
- **DOI:** 10.1038/s41588-024-01854-z | PMID: 39075208
- **Key Contribution:** NLP classifier categorizing free-text stopping reasons for 28,561 stopped trials into 17 semantic categories. Strong genetic evidence halves the odds of early stoppage. First large-scale NLP analysis of trial stopping reasons using ClinicalTrials.gov "why stopped" field.
- **Relevance to CTRT:** Extends the uninformativeness/failure prediction axis from protocol features to stopping-reason classification. The 17-category taxonomy is a practical reference for extending CTRT's risk labels.

### From RAGs to Riches: Utilizing LLMs to Write Documents for Clinical Trials
- **Authors:** Nigel Markey, Ilyass El-Mansouri, Gaetan Rensonnet, Casper van Langen, Christoph Meier
- **Venue:** Clinical Trials (SAGE), Vol. 22(5), pp. 626–631
- **Year:** 2025
- **DOI:** 10.1177/17407745251320806 | PMCID: PMC12476469
- **Key Contribution:** Evaluates GPT-4 (base and RAG-augmented) for generating clinical trial protocol sections. Base LLM scores ~40% on clinical logic; RAG improves it to ~80%. First rigorous evaluation of LLM-generated protocol sections against expert review.
- **Relevance to CTRT:** If CTRT is extended with LLM-based improvement suggestions, RAG augmentation is the recommended architecture.

---

## 6. Scoping Reviews & Benchmarks

### A Scoping Review of Artificial Intelligence Applications in Clinical Trial Risk Assessment
- **Authors:** Douglas Teodoro, Nona Naderi, Anthony Yazdani, Boya Zhang, Alban Bornet
- **Venue:** npj Digital Medicine, Vol. 8, Article 486
- **Year:** 2025
- **DOI:** 10.1038/s41746-025-01886-7 | PMID: 40731070
- **Key Contribution:** Review of 142 studies (2013–2024) categorized into safety risk (n=55), efficacy risk (n=46), and operational risk (n=45). Best models reach AUROC up to 0.96. LLMs appeared in 7 of 33 studies in 2023 alone. **Cites the CTRT paper.**
- **Relevance to CTRT:** Maps the entire landscape in which CTRT sits.

### TrialBench: Multi-Modal AI-Ready Clinical Trial Datasets
- **Authors:** Tianfan Fu et al.
- **Venue:** Scientific Data
- **Year:** 2025
- **DOI:** https://www.nature.com/articles/s41597-025-05680-8
- **Key Contribution:** 23 AI-ready datasets covering 8 trial prediction tasks: dropout, adverse events, approval outcome, failure reason, duration, enrollment, etc.
- **Relevance to CTRT:** Critical resource for training and benchmarking CTRT extensions beyond HIV/TB.

---

## 7. MCTS and Search Methods

### Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models
- **Authors:** Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang
- **Venue:** International Conference on Machine Learning (ICML)
- **Year:** 2024
- **arXiv:** 2310.04406
- **Key Contribution:** Tree search algorithm unifying language model reasoning (complex decomposition), acting (tool execution), and planning (state tracking). Demonstrates that skipping simulation when direct evaluation is available improves sample efficiency. Foundation for AutoCT's MCTS fixes.
- **Relevance to CTRA:** Establishes the precedent for informed simulation selection in LLM-based MCTS. Key insight for feature engineering search efficiency.

### I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search
- **Authors:** Zujie Liang, Feng Wei, Wujiang Xu, Lin Chen, Yuxi Qian, Xinhui Wu
- **Venue:** Findings of EACL
- **Year:** 2026
- **arXiv:** 2502.14693
- **Key Contribution:** Introspective MCTS with sibling analysis — LLM evaluates what worked/failed in neighboring feature sets before proposing next candidates. Component attribution reveals feature importance. Hybrid value model initially considered but rejected for biomedical domains.
- **Relevance to CTRA:** Introspective sibling analysis directly applicable to feature engineering MCTS. Shows how LLMs can learn from partial exploration history.

### SEA-TS: Self-Evolving Agent for Autonomous Code Generation of Time Series Forecasting Algorithms
- **Authors:** Longkun Xu, Xiaochun Zhang, Qiantu Tuo, Rui Li
- **Venue:** arXiv preprint
- **Year:** 2026
- **arXiv:** 2603.04873
- **Key Contribution:** Self-evolving agent using metric-advantage MCTS with z-score normalized rewards. Code review penalty prevents reward hacking. Demonstrates how MCTS search can be hardened for noisy empirical feedback.
- **Relevance to CTRA:** Error penalty framework (R=-1 for failed features) and reward normalization improve robustness of feature engineering search under variance.

---

## 8. NER and Entity Recognition

### GLiNER-BioMed: A Suite of Efficient Models for Open Biomedical Named Entity Recognition
- **Authors:** Anthony Yazdani, Ihor Stepanov, Douglas Teodoro
- **Venue:** arXiv preprint
- **Year:** 2025
- **arXiv:** 2504.00676
- **Key Contribution:** Zero-shot biomedical NER with 16 entity types (disease, gene, symptom, drug, pathway, protein, etc.). +5.96% F1 over domain-adapted baselines. Models available in four sizes (nano to large). Integrated directly into LinearRAG entity graph construction.
- **Relevance to CTRA:** NER backbone for feature extraction from clinical trial sources. Replaces en_core_sci_scibert for entity recognition at scale without retraining.

---

## Key Takeaways for CTRT Development

1. **BERT + GNN is the proven upgrade** over Naive Bayes for protocol risk prediction (Ferdowsi et al., AUROC 0.92)
2. **SHAP explainability** should replace the current linear weight formula for interpretable feature importance
3. **LLM-based CONSORT assessment** (Srinivasan et al.) can automate SAP/reporting quality at scale with near-expert accuracy (F1=0.86)
4. **RAG + GPT-4** is the practical path for extracting richer protocol features without retraining from scratch
5. **TrialBench & ClinicalRisk** datasets are ready-to-use benchmarks for extending CTRT beyond HIV/TB
6. **AutoCT's** LLM agent + interpretable ML approach is architecturally closest to CTRT's philosophy
7. **Enrollment failure** (TrialEnroll) and **stopping reason classification** (Ochoa et al.) are two high-value feature dimensions currently missing from CTRT
8. **CTRT's uninformativeness framing is unique** — no other paper uses this angle; most frame outcome as success/failure or phase transition
9. **BERT + GNN deep dive** — see [BERT+GNN Clinical Trial Research](./bert-gnn-clinical-trial-research.md) for the full Ferdowsi series, independent papers, and architecture comparison table
10. **Post-2024 LLM trend** — see [Post-2024 LLM Clinical Trial Prediction](./post-2024-llm-clinical-trial-prediction.md) for deep dive on MEXA-CTP, LIFTED, CLaDMoP (cross-attention + MoE replacing GNNs)
11. **GLiNER-BioMed is the NER choice** — zero-shot biomedical NER with 16 entity types replaces domain-adapted BERT for entity extraction at scale without retraining
12. **MCTS search benefits from** introspective sibling analysis (I-MCTS), z-score reward normalization (SEA-TS MA-MCTS), adaptive exploration constants (Global Std with C=2), and error penalties (R=-1 for failed nodes) — all adopted or considered for CTRA's feature engineering search

## Related Notes
- [Clinical Trial Risk Tool](./clinical-trial-risk-tool.md)
- [BERT+GNN Clinical Trial Research](./bert-gnn-clinical-trial-research.md)
- [Post-2024 LLM Clinical Trial Prediction](./post-2024-llm-clinical-trial-prediction.md)
