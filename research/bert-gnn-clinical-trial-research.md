
# BERT + GNN for Clinical Trial Prediction

Research into papers combining BERT/transformers with Graph Neural Networks for clinical trial risk and outcome prediction. Prompted by the Ferdowsi et al. (Patterns 2023) paper which is the most direct architectural peer of the Clinical Trial Risk Tool.

---

## The Ferdowsi Research Group (Direct Series)

The Patterns 2023 paper is paper 3 in a series by the same group:

### 1. Classification of hierarchical text using geometric deep learning
- **Authors:** Sohrab Ferdowsi, Nikolay Borissov, Julien Knafou, Poorya Amini, Douglas Teodoro
- **Venue:** EMNLP 2021
- **URL:** https://aclanthology.org/2021.emnlp-main.48 | arXiv: 2110.15710
- **Architecture:** BERT embeddings as GNN node features on protocol hierarchy graph. Selective graph pooling exploiting invariant hierarchical structure common across CT protocols.
- **Task:** Binary — completed vs. terminated trials
- **Note:** Direct architectural precursor to Ferdowsi 2023

### 2. On Graph Construction for Classification of Clinical Trial Protocols Using GNN
- **Authors:** Sohrab Ferdowsi
- **Venue:** AIME 2022, Lecture Notes in Computer Science vol. 13263, Springer
- **URL:** https://link.springer.com/chapter/10.1007/978-3-031-09342-5_24
- **Architecture:** Ablation study of graph topology choices with BERT node features. Shows domain-knowledge edges improve GNN classification.
- **Task:** Binary — high vs. low risk
- **Note:** Intermediate ablation study between EMNLP 2021 and Patterns 2023

### 3. Deep learning-based risk prediction from protocol design ← primary reference
- **Authors:** Sohrab Ferdowsi, Julien Knafou, Nikolay Borissov, David Vicente Alvarez, Rahul Mishra, Poorya Amini, Douglas Teodoro
- **Venue:** Patterns (Cell Press), Vol. 4, Issue 3, 2023
- **DOI:** 10.1016/j.patter.2023.100689
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10028430/
- **Architecture:** BERT section encoder + GNN on protocol hierarchy + ensemble. BERT embeddings deeply integrated as GNN node features.
- **Task:** Ternary risk (low/medium/high) using protocol amendment history as label
- **Results:** AUROC 0.8453 (ternary), 0.9234 (binary)

---

## Independent BERT + GNN Papers

### HINT: Hierarchical Interaction Network for Clinical Trial Outcome Predictions
- **Authors:** Tianfan Fu, Kexin Huang, Cao Xiao, Lucas Glass, Jimeng Sun
- **Venue:** Patterns (Cell Press), 2022
- **DOI:** 10.1016/j.patter.2022.100445
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9024011/
- **Architecture:** ClinicalBERT (eligibility criteria) + MPNN (drug molecules) + GRAM (ICD-10 disease ontology graph) → attentive hierarchical GCN
- **Task:** Binary trial success/failure per phase (I/II/III) on 17,538 trials
- **Results:** F1 0.665/0.620/0.847 across phases
- **Note:** Published in same journal as Ferdowsi 2023. Created the **TOP benchmark** that all subsequent papers compare against.

### PlaNet / PlaNetLM — Predicting drug outcome via clinical knowledge graph
- **Authors:** Maximilian Haug et al. (Stanford / EPFL)
- **Venue:** medRxiv preprint, 2024
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10942490/
- **Architecture:** R-GCN on massive clinical KG (330K nodes, 14M edges, 9 databases) + PubMedBERT encoding trial arm text — embeddings concatenated before prediction head
- **Task:** Efficacy (AUROC 0.70), serious adverse events (0.79), ADE categories (0.85 avg across 554 categories)
- **Key difference:** Operates on clinical trial arm outcomes (not protocol text); models drug-disease-population triplets at knowledge graph scale

### GATher: Graph Attention Based Predictions of Gene-Disease Links
- **Authors:** David Narganes-Carlon, Anniek Myatt, Mani Mudaliar, Daniel J. Crowther (Exscientia / University of Dundee)
- **Venue:** arXiv 2409.16327, September 2024
- **URL:** https://arxiv.org/abs/2409.16327
- **Architecture:** GATv3 (Transformer-inspired dot-product attention replacing LeakyReLU) on heterogeneous biomedical graph + GPT embeddings for disease nodes
- **Task:** Gene-disease link prediction → maximum achievable trial phase (upstream of trials)
- **Results:** ROC AUC 0.81 for positive efficacy in Phase 2→3 transitions

### MultiGML: Multimodal Graph ML for Prediction of Adverse Drug Events
- **Authors:** Sophia Krix et al. (Fraunhofer SCAI)
- **Venue:** Heliyon, August 2023
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC10481305/
- **Architecture:** RGCN + RGAT on heterogeneous KG (20K nodes, 420K edges, 14 databases) + ESM-1b protein transformer as node features
- **Task:** Drug adverse event prediction (pre-clinical safety screening)

### TrialBench: Multi-Modal AI-Ready Datasets
- **Authors:** Yue Yu, Zifeng Wang, Cao Xiao, Jimeng Sun et al.
- **Venue:** Nature Scientific Data, 2025
- **URL:** https://www.nature.com/articles/s41597-025-05680-8
- **Architecture (baseline):** Bio-BERT + MPNN (drug) + GRAM (disease) — late fusion (parallel → concatenate → MLP)
- **Task:** 8 trial prediction tasks (duration, dropout, SAE, approval, failure reason, etc.)

---

## Architecture Comparison

| Paper | BERT Component | GNN Type | Integration | Task |
|---|---|---|---|---|
| Ferdowsi 2021 | BERT section encoder | Hierarchical GNN | BERT → GNN node features | Binary terminated/completed |
| Ferdowsi 2023 | BERT section encoder | Protocol hierarchy GNN | Deep BERT-into-GNN | Ternary risk |
| HINT | ClinicalBERT | MPNN + GRAM + GCN | Parallel → hierarchical interaction | Trial success/failure |
| PlaNetLM | PubMedBERT | R-GCN (14M edges) | Concatenated with R-GCN | Efficacy + safety |
| GATher | GPT embeddings | GATv3 (Transformer-GAT) | GPT features as node attrs | Gene-disease → trial phase |
| MultiGML | ESM-1b (protein) | RGCN + RGAT | Transformer features in KG | Drug-ADE prediction |
| TrialBench baseline | Bio-BERT | MPNN + GRAM | Late fusion | 8 trial tasks |

---

## Post-2024 Trend: Moving Away from GNN

The frontier papers (2024–2025) are abandoning explicit GNNs in favor of pure transformer/LLM approaches. Cross-attention between modalities is replacing graph message passing as the dominant integration mechanism.

| Paper | Venue | Architecture | Key improvement over HINT |
|---|---|---|---|
| MEXA-CTP | SDM 2025 | BioBERT + masked cross-attention experts | +27.9% Phase III PR-AUC |
| LIFTED | EMNLP 2025 | All modalities as text → MoE transformers | Current SOTA across all phases |
| CLaDMoP | KDD 2025 | BioGPT + contrastive pre-train + LoRA | +10.5% PR-AUC over MEXA-CTP, +13.6% new-disease generalization |

> See: [Post-2024 LLM Clinical Trial Prediction](./post-2024-llm-clinical-trial-prediction.md) for deep dive into MEXA-CTP, LIFTED, CLaDMoP

---

## Related Notes
- [Clinical Trial Risk Tool](./clinical-trial-risk-tool.md)
- [Literature Review 2023-2026](./literature-review-2023-2026.md)
- [Post-2024 LLM Clinical Trial Prediction](./post-2024-llm-clinical-trial-prediction.md)
