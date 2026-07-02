
# Post-2024 Trend: Moving Away from GNN — Deep Dive

The frontier papers (2024–2025) abandon explicit GNNs in favor of cross-attention and LLM-based approaches. All three papers below beat HINT (the GNN baseline) substantially.

## Why GNNs Are Being Replaced

**HINT's limitations** (the GNN baseline all three papers address):
1. Requires expensive wet-lab ADMET pharmacokinetic data unavailable at prediction time
2. Encodes eligibility criteria at paragraph level — collapses inclusion/exclusion criteria together
3. Human-designed interaction graph encodes investigator priors/biases directly into model structure
4. Only handles small-molecule interventional trials (not biologics or devices)
5. Potential look-ahead bias: pretrained embeddings may incorporate post-cutoff data
6. Binary success/failure label collapses complex outcome granularity

---

## Paper 1: MEXA-CTP

**Title:** MEXA-CTP: Mode Experts Cross-Attention for Clinical Trial Outcome Prediction
**Authors:** Yiqing Zhang, Xiaozhong Liu, Fabricio Murai (Worcester Polytechnic Institute)
**Venue:** SDM 2025 (SIAM International Conference on Data Mining)
**arXiv:** https://arxiv.org/abs/2501.06823
**DOI:** 10.1137/1.9781611978520.51
**Code:** https://github.com/murai-lab/MEXA-CTP

### Problem Statement
Targets three specific HINT failures:
- Paragraph-level aggregation loses inclusion vs. exclusion criteria distinction
- Requires wet-lab ADMET data (not available at design time)
- Hard-coded graph encodes predetermined interaction assumptions

### Architecture — Four Stages

**Stage 1: Encoding**
- **Drug:** DeepChem encodes SMILES → molecular fingerprints
- **Disease:** icdcodex maps ICD-10 codes → embeddings (no external KG needed)
- **Eligibility:** BioBERT generates **statement-level** embeddings separately for each inclusion/exclusion criterion (key departure from HINT's paragraph-level)

**Stage 2: Knowledge Embedding**
- Multi-layer transformer encoders (2 layers, 2 heads, hidden=32) enrich each modality independently
- **Siamese transformer** for eligibility criteria: same weights applied to inclusion and exclusion sequences with sinusoidal positional embeddings
- Final criteria embedding = concatenate(processed inclusion + exclusion)

**Stage 3: Mode Experts Module (core innovation)**

Three pairwise experts: (drug↔disease), (drug↔criteria), (disease↔criteria)

Each expert:
1. **Token Selection:** `p(S) = sigmoid(S · Wp)` with hard/soft margin thresholds
2. **Cauchy sparsity loss:** `L_cauchy = Σ log(1 + p_s² / ε)` — drives selection toward 0 or 1
3. **Bidirectional cross-attention:** Mode A attends over Mode B's tokens and vice versa → 6 total cross-attention outputs
4. **Contrastive loss:** Two directions of each pair treated as positive pairs (NT-Xent) → consistent mutual representations

**Stage 4: Prediction**
- Concatenate all 6 interaction outputs → mean pool → sigmoid prediction head
- Loss: `L = L_cls + λ₁·L_cauchy + λ₂·L_contrastive`
- Class-weighted BCE to address label imbalance (Phase I 70% success, Phases II/III ~30%)

### Results vs. HINT (TOP Benchmark)

| Phase | Metric | HINT | MEXA-CTP | Δ |
|---|---|---|---|---|
| I | F1 | .598±.011 | **.713±.027** | +19.2% |
| I | PR-AUC | .581±.021 | **.605±.014** | +4.1% |
| II | F1 | .635±.011 | **.695±.008** | +9.4% |
| II | PR-AUC | .607±.012 | **.635±.015** | +4.6% |
| III | F1 | .814±.013 | **.857±.007** | +5.3% |
| III | PR-AUC | .603±.014 | **.771±.016** | +27.9% |

**Key ablation insight:** Random token selection collapses F1 to 0.328 — token selection is critical. Removing Cauchy loss drops F1 from 0.857 to 0.790.

### Limitations
- No pre-training — starts from scratch on small labeled dataset
- Single benchmark (TOP only)
- Poor generalization to new diseases (Phase I accuracy: 47.55% — barely above random)
- Hyperparameter t (selection threshold) requires grid search

### Relevance to CTRT
- Statement-level criteria encoding directly applicable to CTRT's eligibility criteria extraction
- Learned cross-modal attention could capture interactions between drug type, indication, and protocol without hard-coded domain rules
- Cauchy-penalized token selection → attention-based feature importance (which criteria sentences drive failure risk)
- No wet-lab data required — aligns with CTRT's deployment constraints

---

## Paper 2: LIFTED

**Title:** Multimodal Clinical Trial Outcome Prediction with Large Language Models
**Authors:** Wenhao Zheng, Liaoyaqi Wang, Dongshen Peng, Hongxia Xu, Yun Li, Hongtu Zhu, Tianfan Fu, Huaxiu Yao
**Venue:** EMNLP 2025 Findings, pages 7503–7517
**arXiv:** https://arxiv.org/abs/2402.06512
**ACL Anthology:** https://aclanthology.org/2025.findings-emnlp.396/

### Problem Statement
Targets **rigid, modality-specific encoders** in prior work:
- Adding a new modality requires designing a new encoder from scratch
- Each encoder learns in isolation — similar information across modalities not recognized
- Fragile to missing/noisy modality data — one bad encoder corrupts the full prediction

### Architecture — Four Stages

**Stage 1: Language Unification**

All heterogeneous data converted to natural language descriptions via LLM (GPT-3.5):
- Disease list, drug name, drug description, drug SMILES, eligibility criteria
- Plus one synthesized **summarization modality** (all features → holistic description)
- Total: **6 modality streams** (all as natural language text)

**Stage 2: Unified Representation Learning (K+1 Transformer Encoders)**

Each modality gets a dedicated transformer encoder:
- Modality-specific tokenizer (SMILES uses chemistry-aware tokenizer; others use standard)
- Modality-specific embedding layers
- Learnable `[cls]_k` token as summary representation
- Output: `U_{i,k} = Encoder_k(tokens)`

**Stage 3: Sparse Mixture-of-Experts (SMoE) Refinement (core innovation)**

Pool of R expert networks; each modality embedding routed to top-k experts:
```
G(U_{i,k}) = Softmax(TopK(P(U_{i,k}), k))
P(U_{i,k}) = U_{i,k} · W_g + ε · softplus(U_{i,k} · W_noise)
Ũ_{i,k} = Σ_r G^r(U_{i,k}) · R^r(U_{i,k})
```
Noise term enables load balancing. Modalities with similar information route to the **same expert** → cross-modal pattern consolidation without explicit attention.

**Noise resilience:** Random perturbation of embeddings + consistency loss:
`L_con = (1/N(K+1)) Σ_{i,k} || Ũ_{i,k} - Ṽ_{i,k} ||²_F`

**Stage 4: Dynamic Integration (Disease-Driven Weighting)**

Disease modality alone generates importance weights for all modalities:
```
W_{i,k} = Softmax(C(⊕ U_{i,disease}) · γ_k)
U_i = Σ_k W_{i,k} · Ũ_{i,k}
```
Ablation confirms disease is most discriminative meta-signal.

**Training:** AdamW lr=3×10⁻⁴, CosineAnnealing, 5 epochs, batch=32, ~2h on single 4090

### Results — HINT Benchmark

| Phase | Metric | HINT | SPOT | LIFTED | Best Δ over HINT |
|---|---|---|---|---|---|
| I | PR-AUC | 58.4% | 69.8% | **70.7%±2.3** | +21.1% |
| I | F1 | 68.2% | 68.4% | **71.6%±1.4** | +5.0% |
| II | PR-AUC | 59.1% | 62.6% | **69.8%±1.8** | +18.1% |
| III | PR-AUC | 85.9% | 81.7% | **88.3%±1.1** | +2.8% |
| III | F1 | 80.9% | 81.0% | **83.8%±0.8** | +3.6% |

**Single-modality ablation (Phase I PR-AUC):**
- Criteria alone: **68.0%** (most informative single modality)
- Disease alone: 65.3%
- Summarization: 63.2%
- SMILES alone: 62.8%
- Drug name alone: 61.1% (least informative)
- Full LIFTED: **70.7%** (synergistic benefit)

**Datasets used:** HINT benchmark + CTOD benchmark (~125K trials, much larger than TOP)

### Limitations
- GPT-3.5 preprocessing is hidden compute cost (API calls at scale)
- Specific LLM not named → reproducibility uncertainty
- Not evaluated on TOP benchmark (used by MEXA-CTP and CLaDMoP)
- No failure case analysis or confidence calibration

### Relevance to CTRT
- **Language unification** is immediately applicable: convert all CTRT inputs (ICD codes, drug IDs, free text) to natural language → single unified encoder architecture
- **MoE-based modality weighting** → automatically up-weight most reliable modalities per trial (trust disease signal when drug data is sparse)
- **Noise-resilient encoders** → handles CTRT's real-world problem of missing/incomplete trial data fields
- **Criteria alone achieves 68% PR-AUC** → validates CTRT's central focus on protocol text
- **CTOD benchmark** (125K trials) is a massive training resource for CTRT extensions

---

## Paper 3: CLaDMoP

**Title:** CLaDMoP: Learning Transferrable Models from Successful Clinical Trials via LLMs
**Authors:** Yiqing Zhang, Xiaozhong Liu, Fabricio Murai (WPI — same group as MEXA-CTP)
**Venue:** KDD 2025 (Toronto, August 2025)
**arXiv:** https://arxiv.org/abs/2505.18527
**Code + SCT dataset:** https://github.com/murai-lab/CLaDMoP

### Problem Statement
Addresses what MEXA-CTP fails to fix: **poor generalization to out-of-distribution diseases**:
- All existing methods (HINT, MEXA-CTP, LIFTED) overfit to the TOP benchmark's disease distribution
- MEXA-CTP Phase I accuracy on new diseases: **47.55%** — barely above random
- Untapped knowledge in successful trials beyond TOP is ignored
- Task-specific BCE loss prevents learning transferable representations

### Architecture — Two-Stage System

**Stage 1: Self-Supervised Pre-Training on SCT Dataset**

Two branches:

**LLM Branch (frozen):**
- Encoder: **BioGPT** (347M params, trained on 15M PubMed abstracts + 3M PMC full-text)
- Input: Eligibility criteria text (~364 words avg)
- Output: Embeddings at **3 hierarchical levels**:
  - Coarse: transformer blocks 1–6 (syntactic)
  - Medium: blocks 7–12 (semantic)
  - Fine: blocks 13–18 (domain-specific)

**DM Branch (Drug-Molecule, trained during pre-training):**
- Lightweight transformer processing drug molecules + diseases
- Fuses LLM embeddings via multi-level **Grouping Blocks**

**Grouping Block (prevents sequence length explosion):**
```
U_agg = softmax( U_centroid · W^Q · (U · W^K)^T / √dk ) · U · W^V
```
Trainable centroid tokens as queries (initialized N(0,1)), input tokens as keys/values.
G=3 layers: 100 centroids → 50 → 25 (progressive compression)
S=2 self-attention layers between each grouping layer.

**Pre-Training Objective (InfoNCE pair matching):**
```
logits = exp(τ) · f_C × f_DM^T    (τ=0.6)
L_pretrain = (L_rows + L_cols) / 2
```
Symmetric cross-entropy: match eligibility criteria embeddings to drug-disease embeddings within a batch of successful trials. LLM branch is **frozen** — only DM branch learns to align with criteria representations.

**Stage 2: PEFT Fine-Tuning**
- **LoRA** (rank=8) applied to all attention layers in DM branch
- 3-layer residual prediction head (ReLU)
- lr: 10⁻² (head), 5×10⁻² (LoRA Phases I-II), 10⁻³ (Phase III)

**SCT Dataset (newly introduced):**
- Constructed from ClinicalTrials.gov + DrugBank drug synonyms
- Success assumption: all phases of FDA-approved drugs are successful
- 4,289 unique drugs, 3,326 unique diseases, avg 364 words/criteria
- No leakage with TOP test set

### Results

**vs. MEXA-CTP and HINT (TOP Benchmark):**

| Phase | Metric | HINT | MEXA-CTP | CLaDMoP | Δ vs MEXA-CTP |
|---|---|---|---|---|---|
| I | F1 | .604 | .713 | **.713±.019** | 0% |
| I | PR-AUC | .581 | .605 | **.680±.016** | +12.4% |
| I | ROC-AUC | .575 | .593 | **.642±.015** | +8.3% |
| II | F1 | .635 | .695 | .685±.015 | −1.4% |
| II | PR-AUC | .608 | .635 | **.683±.012** | +7.6% |
| III | F1 | .814 | .857 | **.861±.014** | +0.5% |
| III | PR-AUC | .603 | .771 | **.860±.011** | +11.5% |
| III | ROC-AUC | .685 | .693 | **.702±.018** | +1.3% |

**Generalization to New Diseases (accuracy):**

| Phase | CLaDMoP | MEXA-CTP | Δ |
|---|---|---|---|
| I | **54.34%** | 47.55% | **+13.6%** |
| II | **59.41%** | 55.00% | **+8.0%** |
| III | 54.34% | **55.55%** | −2.2% |

**Pre-training contribution ablation (Phase III):**
| Config | F1 | PR-AUC |
|---|---|---|
| No pre-train, no fine-tune | .357 | .466 |
| Fine-tune only | .800 | .766 |
| Pre-train + LoRA | **.861** | **.860** |

**Grouping Block ablation (G layers, S self-attn layers):**
| G | S | F1 | PR-AUC |
|---|---|---|---|
| 1 | 2 | .797 | .808 |
| 3 | 0 | .612 | .669 |
| **3** | **2** | **.861** | **.860** |

### Limitations
- SCT success assumption introduces noise (drugs may have failed early before eventual approval)
- BioGPT frozen → criteria representations are static regardless of trial-specific context
- Phase III new-disease generalization slightly worse than MEXA-CTP
- Not tested on CTOD benchmark (used by LIFTED)
- No interpretation of what DM branch learns from pair-matching

### Relevance to CTRT
- **Two-stage pre-train + LoRA fine-tune** is the most sophisticated directly applicable pattern for CTRT
- **SCT construction methodology** shows how to build large weakly-labeled pre-training sets from ClinicalTrials.gov + DrugBank — reproducible for CTRT
- **Multi-level BioGPT fusion** applicable to CTRT's protocol processing: extract coarse/medium/fine embeddings for richer criteria representations
- **+13.6% new-disease generalization in Phase I** — critical for CTRT's use case of first-in-disease trials where historical comparators are scarce
- **LoRA** makes multi-disease/phase adaptation low-cost

---

## Cross-Paper Comparison

| Dimension                  | HINT (GNN baseline)            | MEXA-CTP                                   | LIFTED                            | CLaDMoP                                    |
| -------------------------- | ------------------------------ | ------------------------------------------ | --------------------------------- | ------------------------------------------ |
| Architecture               | GNN + hierarchical interaction | Pairwise cross-attention + token selection | Unified LLM → SMoE                | BioGPT + DM branch + contrastive pre-train |
| Criteria encoding          | Paragraph-level BERT           | Statement-level BioBERT (incl/excl split)  | GPT-3.5 description → transformer | BioGPT 3-level extraction (frozen)         |
| Cross-modal interaction    | Hard-coded graph               | Learned pairwise cross-attention           | SMoE routing                      | InfoNCE pair matching + grouping blocks    |
| Pre-training               | ADMET (wet-lab required)       | None                                       | None                              | SCT pair-matching (no wet-lab)             |
| Phase III PR-AUC           | .603                           | .771                                       | .883 (diff split)                 | **.860**                                   |
| New disease generalization | Not tested                     | Poor (47.55% Ph I)                         | Not tested                        | Good (54.34% Ph I)                         |
| Benchmark                  | TOP                            | TOP                                        | HINT + CTOD                       | TOP                                        |
| Wet-lab data required      | Yes                            | No                                         | No                                | No                                         |
| Key limitation             | Biased architecture            | No pre-training                            | LLM API cost                      | Frozen LLM, SCT noise                      |

---

## Synthesis: Recommended Architecture for CTRT Upgrade

Combining insights from all three papers:

1. **Input processing:** Convert all modalities to natural language (LIFTED) + split eligibility criteria at sentence level with inclusion/exclusion separation (MEXA-CTP)
2. **Encoding:** Frozen BioGPT with 3-level extraction — coarse/medium/fine (CLaDMoP)
3. **Cross-modal fusion:** Pairwise cross-attention with learned token selection (MEXA-CTP) OR SMoE with disease-driven weighting (LIFTED)
4. **Pre-training:** InfoNCE pair-matching on SCT-equivalent dataset from ClinicalTrials.gov + DrugBank (CLaDMoP) — biggest gain for new-disease generalization
5. **Fine-tuning:** LoRA (rank=8) on labeled risk dataset (CLaDMoP)

> The single most impactful step for CTRT would be **CLaDMoP's pre-training strategy** — building a weakly-labeled successful trial dataset and using contrastive pair-matching before fine-tuning on CTRT's labeled protocols.

---

## Related Notes
- [Clinical Trial Risk Tool](./clinical-trial-risk-tool.md)
- [Literature Review 2023-2026](./literature-review-2023-2026.md)
- [BERT+GNN Clinical Trial Research](./bert-gnn-clinical-trial-research.md)
