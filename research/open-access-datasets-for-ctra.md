# Open-Access Datasets for Clinical Trial Risk Assessment (CTRA)

**Research Date:** 2026-03-28
**Purpose:** Identify free, open-access datasets to supplement the CTRA project's existing data sources for predicting clinical trial success/failure.

## Already In Use

The project currently leverages:
- **TrialBench** (23 datasets, 8 tasks, ClinicalTrials.gov + DrugBank + TrialTrove)
- **ClinicalTrials.gov** (trial records via public API)
- **PubMed** (literature abstracts)
- **FAERS/OpenFDA** (adverse event reports)
- **TOP benchmark** (17,538 trials from HINT repo)
- **CTOD benchmark** (12,477 trials)

---

## Category 1: Drug / Pharmacology Databases

### 1.1 ChEMBL
- **URL:** https://www.ebi.ac.uk/chembl/
- **What it contains:** Manually curated database of bioactive molecules with drug-like properties. Contains bioactivity measurements (IC50, Ki, EC50, etc.) for >2.4M compounds against >15K targets, sourced from >88K publications. Includes drug mechanism of action, indication, and clinical development phase data.
- **License:** Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)
- **Format:** REST API, bulk PostgreSQL/MySQL/SQLite downloads, Python client library
- **CTRA benefit:** Provides quantitative bioactivity data linking compounds to molecular targets. AutoCT agents can retrieve target potency, selectivity profiles, and mechanism of action for trial drugs, which are strong predictors of clinical success. Phase-transition data in ChEMBL also provides historical attrition information.
- **Python package:** `chembl_webresource_client` (pip install)

### 1.2 PubChem
- **URL:** https://pubchem.ncbi.nlm.nih.gov/
- **What it contains:** The largest open repository of chemical information: >116M compound records, >290M bioassay data points, 1.7M+ bioassays, plus cross-links to patents, literature, and gene/protein data. Includes the BioAssay database with high-throughput screening data from NIH, pharma, and academic labs.
- **License:** Public domain (US Government work)
- **Format:** REST API (PUG-REST), bulk FTP downloads (SDF, JSON, CSV), SMILES/InChI
- **CTRA benefit:** Drug structure retrieval and property calculation for any compound mentioned in a trial. BioAssay data provides high-throughput screening results that can indicate mechanism viability. Cross-links to patents may signal commercial confidence.
- **Python package:** `pubchempy` (pip install)

### 1.3 Therapeutic Target Database (TTD) 2026
- **URL:** https://idrblab.org/ttd/
- **What it contains:** 3,798 targets, 40,398 drugs, 306,247 target-disease associations, 17,806 drug activity landscapes (cytotoxic, antimicrobial, molecular-level), 10,506 perturbation profiles, and clinical profiles for 2,234 approved drugs. Includes target druggability information and multi-target agent data.
- **License:** Free access, no login required
- **Format:** Bulk TSV/CSV download (https://ttd.idrblab.cn/full-data-download)
- **CTRA benefit:** Directly links drug targets to diseases with druggability scores. Multi-target agent data helps predict off-target effects. Clinical profiles of approved drugs provide validation benchmarks. Target perturbation profiles can indicate biological plausibility of a trial's mechanism.
- **Python package:** None official; parse downloaded TSV files with pandas

### 1.4 ChEBI (Chemical Entities of Biological Interest)
- **URL:** https://www.ebi.ac.uk/chebi/
- **What it contains:** >195,000 entries of biologically relevant small molecules with an ontological classification system. Provides chemical role classifications (e.g., "anti-inflammatory agent," "kinase inhibitor"), structural data, and cross-references to other databases.
- **License:** CC BY 4.0 (ELIXIR Core Data Resource)
- **Format:** MySQL dumps, OBO/OWL flat files, REST API
- **CTRA benefit:** The ontological role classification enables AutoCT to categorize trial drugs by pharmacological class and mechanism, which is a key feature for trial outcome prediction. The hierarchical structure supports feature engineering at multiple granularity levels.
- **Python package:** `libchebipy` (pip install)

### 1.5 DailyMed / Structured Product Labels (SPL)
- **URL:** https://dailymed.nlm.nih.gov/
- **What it contains:** The most recent FDA-submitted drug labeling for prescription/OTC drugs, biologics, medical devices, and dietary supplements. Full-text labels in XML (SPL format) including indications, contraindications, warnings, dosing, clinical pharmacology, and clinical study results.
- **License:** Public domain (US Government)
- **Format:** Bulk XML download (daily/weekly/monthly updates), REST API
- **CTRA benefit:** Drug labels contain structured summaries of pivotal clinical trial results, safety signals, and pharmacokinetic data. AutoCT agents can extract prior trial outcomes for the same drug/drug class, warnings that predict future trial risks, and dosing information relevant to trial design adequacy.
- **Python package:** None official; use `requests` + `lxml` for XML parsing

### 1.6 RxNorm
- **URL:** https://www.nlm.nih.gov/research/umls/rxnorm/
- **What it contains:** Normalized naming system for clinical drugs that links brand names, generic names, ingredients, and dose forms across >200 drug vocabularies (First Databank, Micromedex, DrugBank, etc.). Coverage of ~99.995% of prescribable drugs.
- **License:** Free (UMLS license, no charge)
- **Format:** UMLS download, REST API (RxNav), RxNorm API
- **CTRA benefit:** Essential for entity resolution -- mapping drug names in clinical trial records to canonical identifiers. Without RxNorm normalization, the same drug may appear under dozens of different names across ClinicalTrials.gov, FAERS, and literature, causing data fragmentation. This is a critical preprocessing step for RAG retrieval accuracy.
- **Python package:** `rxnorm-api` or direct REST calls via `requests`

### 1.7 OnSIDES (ON-label SIDE effectS)
- **URL:** https://github.com/tatonetti-lab/onsides
- **What it contains:** 7.1M+ drug-ADE (adverse drug event) pairs for 4,097 drug ingredients extracted from 51,460 labels across FDA (DailyMed), EMA, EMC (UK), and KEGG (Japan) using a fine-tuned PubMedBERT model. International coverage with standardized MedDRA coding. F1=0.90, AUROC=0.92.
- **License:** Open source on GitHub (periodically updated)
- **Format:** TSV/CSV files on GitHub, computational pipeline for updates
- **CTRA benefit:** Complements FAERS with label-derived ADE data. While FAERS captures post-market spontaneous reports, OnSIDES provides the manufacturer-acknowledged safety profile. Comparing the two can reveal safety signal discrepancies that predict regulatory risk. Multi-country coverage (US, EU, UK, Japan) enables geographic safety signal analysis.
- **Python package:** None; download directly from GitHub, load with pandas

---

## Category 2: Disease / Phenotype Databases

### 2.1 Mondo Disease Ontology
- **URL:** https://mondo.monarchinitiative.org/
- **What it contains:** A unified disease ontology harmonizing OMIM, Orphanet, EFO, Disease Ontology (DOID), ICD-11, and NCIt neoplasm branches. Provides precisely annotated mappings with strict semantics (equivalent vs. related), hierarchical disease classification, and cross-database linking.
- **License:** CC BY 4.0
- **Format:** OWL, OBO, JSON (GitHub download), EBI Ontology Lookup Service
- **CTRA benefit:** Mondo solves the disease entity resolution problem -- mapping diseases across different trial registries and datasets to a single canonical identifier. The hierarchical structure enables feature engineering at different disease granularity levels (e.g., "breast cancer" vs. "HER2-positive breast cancer"). Critical for cross-dataset trial outcome aggregation.
- **Python package:** `pronto` (OBO/OWL parser), `oaklib` (Ontology Access Kit)

### 2.2 Human Phenotype Ontology (HPO)
- **URL:** https://hpo.jax.org/
- **What it contains:** Standardized vocabulary of >18,000 phenotypic abnormalities in human disease, with annotations linking phenotypes to >12,468 rare diseases (OMIM, Orphanet, DECIPHER). Also provides gene-to-phenotype and phenotype-to-gene mapping files updated monthly.
- **License:** Open access (custom HPO license, free for research)
- **Format:** OBO/OWL download, REST API (NLM Clinical Tables), gene-phenotype TSV files
- **CTRA benefit:** Enables phenotype-based characterization of target diseases in clinical trials. For rare disease trials, HPO terms can capture the phenotypic spectrum better than ICD codes. Gene-phenotype links help identify whether a trial's target is biologically connected to the disease phenotype, which predicts mechanistic validity.
- **Python package:** `hpo-toolkit` (pip install), `pyhpo`

### 2.3 Disease Ontology (DO / DOID)
- **URL:** https://disease-ontology.org/
- **What it contains:** >12,000 disease concepts with 15 relationship types, part of the OBO Foundry. Maps diseases to MeSH, ICD-9/10, SNOMED CT, OMIM, and NCI Thesaurus. Provides human-readable, machine-interpretable disease classification.
- **License:** CC0 1.0 (Public Domain)
- **Format:** OBO/OWL download, SPARQL endpoint
- **CTRA benefit:** Provides another axis of disease classification complementary to ICD-10 and Mondo. Its cross-mappings enable linking trial diseases to genetic (OMIM) and literature (MeSH) contexts. The CC0 license makes it the most permissive option.
- **Python package:** `pronto` (OBO parser)

### 2.4 Orphanet / OrphaData
- **URL:** https://www.orpha.net/ (portal), https://www.orphadata.com/ (datasets)
- **What it contains:** Comprehensive rare disease database covering ~6,000 rare diseases with associated genes, phenotypes, prevalence data, clinical trial information, and orphan drug designations. The OrphaData portal provides bulk download of structured datasets. Includes the ORDO (Orphanet Rare Disease Ontology).
- **License:** Free access, CC BY 4.0 for OrphaData datasets
- **Format:** XML, JSON bulk downloads from OrphaData; ORDO in OWL format
- **CTRA benefit:** Rare disease trials have distinct success/failure profiles. Orphanet provides prevalence data (affects enrollment feasibility), genetic basis information (predicts biomarker strategy viability), and historical orphan drug approval data. Clinical trial registry cross-links connect Orphanet diseases directly to registered trials.
- **Python package:** None official; parse XML/JSON with standard libraries

### 2.5 MeSH (Medical Subject Headings)
- **URL:** https://www.nlm.nih.gov/mesh/meshhome.html
- **What it contains:** ~30,000+ controlled vocabulary terms organized hierarchically, used by ClinicalTrials.gov to classify trial conditions and by PubMed for literature indexing. Updated annually.
- **License:** Public domain (US Government)
- **Format:** RDF (N-Triples), OBO, OWL, XML, ASCII; downloadable from NLM
- **CTRA benefit:** MeSH terms are already the primary disease classification in ClinicalTrials.gov. Using the full MeSH hierarchy enables feature engineering based on disease class relationships (e.g., all "Neoplasms" trials share structural features). MeSH-PubMed linkage enables retrieval of all literature for a disease concept during RAG.
- **Python package:** `pymesh` (community), or parse RDF with `rdflib`

---

## Category 3: Clinical Trial Outcome / Regulatory Datasets

### 3.1 AACT (Aggregate Analysis of ClinicalTrials.gov)
- **URL:** https://aact.ctti-clinicaltrials.org/
- **What it contains:** A publicly available relational database containing ALL protocol and result data elements from every study registered in ClinicalTrials.gov. Updated nightly. Structured as a PostgreSQL database with ~50 tables covering study design, eligibility criteria, interventions, outcomes, results, adverse events, and more.
- **License:** Free, open access (requires free account for direct DB access)
- **Format:** PostgreSQL database (direct connection), daily/static database snapshots (downloadable), CSV pipe-delimited files
- **CTRA benefit:** This is arguably the single most valuable addition to the CTRA pipeline. While ClinicalTrials.gov API provides trial records one at a time, AACT provides the entire database for batch analysis. This enables: (1) population-level feature engineering across all trials, (2) historical success rate calculation by sponsor/disease/phase, (3) enrollment pattern analysis, and (4) result data extraction at scale. The relational structure is far more analysis-friendly than the ClinicalTrials.gov API.
- **Python package:** Connect via `psycopg2` or `sqlalchemy`; download snapshots and load with pandas

### 3.2 Drugs@FDA / openFDA
- **URL:** https://open.fda.gov/ (API), https://www.fda.gov/drugs/drug-approvals-and-databases (portal)
- **What it contains:** Approval history for all FDA-approved drugs since 1939, including approval letters, review documents, patient information, and labels. The openFDA platform provides REST APIs for drug labels, adverse events, enforcement actions, and NDC (National Drug Code) data. Separate endpoints for drugs, devices, food, and animal/veterinary products.
- **License:** Public domain (US Government)
- **Format:** REST API (JSON responses), bulk JSON downloads
- **CTRA benefit:** Historical drug approval data is a direct ground-truth label source -- did a drug ultimately get approved? Review documents contain FDA's assessment of clinical trial quality, which can inform features about regulatory risk factors. The combination of openFDA adverse events + approval status provides a regulatory outcome dataset.
- **Python package:** `python-openfda` (community), or direct REST calls

### 3.3 EMA Clinical Data Publication
- **URL:** https://clinicaldata.ema.europa.eu/
- **What it contains:** Clinical data submitted by pharmaceutical companies to support regulatory applications under the EMA centralised procedure. Includes clinical study reports, individual patient data (in some cases), and analysis datasets for medicines authorized since October 2016.
- **License:** Free access (requires EMA account registration, subject to terms of use)
- **Format:** On-screen view or full download (PDF/datasets), EU Clinical Trials Register (CTIS) for trial-level metadata
- **CTRA benefit:** Provides European regulatory perspective complementary to FDA data. For drugs submitted to both FDA and EMA, comparing regulatory outcomes reveals regional decision-making differences. Clinical study reports contain far more detail than ClinicalTrials.gov results summaries.
- **Python package:** None; web scraping or manual download

### 3.4 EU Clinical Trials Register / CTIS
- **URL:** https://euclinicaltrials.eu/ (CTIS), https://www.clinicaltrialsregister.eu/ (legacy)
- **What it contains:** Information on interventional clinical trials for medicines authorized in the EEA, including protocol information, results summaries, and trial status. The Clinical Trials Information System (CTIS) is the newer platform mandated by the EU Clinical Trials Regulation.
- **License:** Free public access
- **Format:** Web search interface, structured data for trial protocols
- **CTRA benefit:** Provides European trial data not always captured in ClinicalTrials.gov. European-only trials represent a significant portion of global clinical development. Cross-referencing EU and US trial registries improves coverage for multinational trials and helps identify regional trial design differences that affect outcomes.
- **Python package:** None official; web scraping with `requests`/`beautifulsoup4`

### 3.5 FDA Drug Trials Snapshots
- **URL:** https://www.fda.gov/drugs/drug-approvals-and-databases/drug-trials-snapshots
- **What it contains:** Demographic data from clinical trials supporting FDA approvals, including age, sex, race/ethnicity breakdowns of trial participants vs. disease population. Published for each new drug approval.
- **License:** Public domain (US Government)
- **Format:** Web pages, PDF snapshots
- **CTRA benefit:** Trial demographic representativeness is emerging as a predictor of regulatory scrutiny and post-market outcomes. Trials with poor demographic diversity face increasing regulatory challenges. This data enables features around enrollment diversity.
- **Python package:** None; web scraping required

---

## Category 4: Biomedical Knowledge Graphs

### 4.1 PrimeKG (Precision Medicine Knowledge Graph)
- **URL:** https://github.com/mims-harvard/PrimeKG
- **What it contains:** Integrates 20 high-quality resources to describe 17,080 diseases with 4,050,249 relationships across 10 biological scales: disease-associated protein perturbations, biological processes, pathways, anatomical regions, phenotypes, and the full range of approved drugs with therapeutic action. Single CSV file format.
- **License:** MIT (code), dataset license varies by source
- **Format:** Single CSV file via Harvard Dataverse, loads in <6 seconds on standard CPU
- **CTRA benefit:** The most directly useful knowledge graph for CTRA. Its multi-scale biological context (drug -> target -> pathway -> biological process -> phenotype -> disease) provides exactly the feature hierarchy AutoCT agents need to assess mechanistic plausibility. The single-CSV format makes it trivially integrable into the RAG pipeline.
- **Python package:** `from tdc.resource import PrimeKG` (via Therapeutics Data Commons), also NetworkX/iGraph compatible

### 4.2 Hetionet
- **URL:** https://github.com/hetio/hetionet, https://het.io
- **What it contains:** 47,031 nodes (11 types) and 2,250,197 relationships (24 types), integrating 29 public resources. Connects compounds, diseases, genes, anatomies, pathways, biological processes, molecular functions, cellular components, pharmacologic classes, side effects, and symptoms.
- **License:** CC0 1.0 (Public Domain)
- **Format:** Neo4j database, JSON, TSV downloads
- **CTRA benefit:** Originally designed for drug repurposing via systematic prediction of compound-disease treatment probabilities. The meta-path framework (e.g., Drug -> treats -> Disease -> associates -> Gene -> participates -> Pathway) provides exactly the kind of multi-hop reasoning that LinearRAG can leverage. The CC0 license is maximally permissive.
- **Python package:** `hetnetpy` (pip install), Neo4j via `py2neo`

### 4.3 DRKG (Drug Repurposing Knowledge Graph)
- **URL:** https://github.com/gnn4dr/DRKG
- **What it contains:** ~97,000 entities of 13 types and ~5,870,000 triplets of 107 relation types. Integrates data from DrugBank, Hetionet, GNBR, STRING, IntAct, and DGIdb, plus COVID-19 literature. Pre-trained TransE embeddings provided.
- **License:** Apache 2.0 (code), individual source licenses for data
- **Format:** TSV files (drkg.tsv), pre-trained embeddings, Jupyter notebooks for analysis
- **CTRA benefit:** Pre-trained entity embeddings can serve as immediate features for drugs and diseases in trial outcome prediction without training from scratch. The comprehensive relation types (drug-drug interaction, drug-side effect, drug-target, gene-disease, etc.) capture the biological context relevant to trial outcomes.
- **Python package:** DGL (Deep Graph Library) for graph operations, notebooks provided

### 4.4 PharMeBINet
- **URL:** https://zenodo.org/record/6578218 (data), GitHub (code)
- **What it contains:** 2,869,407 nodes (66 labels), 15,883,653 relationships (208 edge types). Built on Hetionet + 19 additional databases including CTD, DrugBank, ClinVar, PharmGKB. Covers ADRs, diseases, drugs, genes, gene variations, proteins, pathways, and phenotypes.
- **License:** Open source
- **Format:** Neo4j database dump
- **CTRA benefit:** The most comprehensive single knowledge graph, combining pharmacological, medical, and biological data. Its inclusion of gene variations and ADR data makes it particularly suited for predicting safety-related trial failures. The gene variation data enables pharmacogenomic feature engineering.
- **Python package:** Neo4j via `py2neo` or `neo4j` Python driver

### 4.5 Open Targets Platform
- **URL:** https://platform.opentargets.org/
- **What it contains:** Target-disease evidence from 23 independent public sources covering genetic associations, somatic mutations, known drugs, clinical trial evidence, pathways, literature mining, and animal models. Released 5 times per year (latest: 25.12, January 2026). Includes evidence scores and tractability assessments.
- **License:** Open access (CC0 for data)
- **Format:** GraphQL API, Google BigQuery, bulk Parquet/JSON downloads
- **CTRA benefit:** Provides pre-computed evidence scores linking drug targets to diseases, which directly quantifies the strength of biological rationale for a trial. The tractability assessments predict whether a target is druggable. Clinical trial evidence integration means Open Targets has already curated success/failure outcomes for many target-disease pairs.
- **Python package:** `opentargets-py` (community), or GraphQL via `requests`/`gql`

### 4.6 Comparative Toxicogenomics Database (CTD)
- **URL:** https://ctdbase.org/
- **What it contains:** 3.8M manually curated direct interactions from 149,000+ articles: chemical-gene/protein interactions, chemical-phenotype interactions, chemical-disease associations, and gene-disease associations across 17,700+ chemicals, 55,400+ genes, 7,200+ diseases, and 630+ species. Generates 48M+ inferred relationships.
- **License:** Free for academic/non-commercial use
- **Format:** Bulk CSV/TSV downloads, REST API, batch query tools
- **CTRA benefit:** The chemical-gene interaction data reveals whether a trial drug's molecular mechanism is well-characterized, which predicts trial success. Chemical-disease direct associations (from literature) provide evidence strength for the therapeutic hypothesis. Toxicogenomic data can predict safety-related failures.
- **Python package:** None official; download CSVs and parse with pandas

---

## Category 5: Real-World Evidence Datasets

### 5.1 MIMIC-IV
- **URL:** https://physionet.org/content/mimiciv/3.1/
- **What it contains:** De-identified EHR data for 364,627 unique patients, 546,028 hospitalizations, and 94,458 ICU stays from Beth Israel Deaconess Medical Center (2008-2022). Includes diagnoses, procedures, lab results, medications, vital signs, clinical notes, and imaging data. MIMIC-IV v3.1 released 2025.
- **License:** PhysioNet Credentialed Health Data License (free, requires CITI training completion)
- **Format:** CSV files in a relational schema, hosted on PhysioNet and AWS
- **CTRA benefit:** Provides real-world treatment outcomes and comorbidity patterns for diseases studied in clinical trials. AutoCT agents can compare trial eligibility criteria against real-world patient distributions to predict enrollment feasibility. Lab value distributions help assess whether trial endpoints are realistic. Drug combination patterns reveal standard of care context.
- **Python package:** `wfdb` (PhysioNet tools), direct CSV loading with pandas

### 5.2 OMOP CDM / OHDSI
- **URL:** https://www.ohdsi.org/data-standardization/
- **What it contains:** Not a single dataset, but a standardized data model used by 100+ institutions worldwide. The OHDSI community provides open-source analytics tools (ATLAS, HADES, CohortDiagnostics) that work across any OMOP-formatted dataset. The Eunomia test dataset provides a synthetic but realistic OMOP database for development.
- **License:** Apache 2.0 (tools), individual datasets have own licenses
- **Format:** PostgreSQL/SQL Server schema, R/Python packages for analysis
- **CTRA benefit:** If CTRA gains access to any OMOP-formatted institutional data, the OHDSI toolchain provides immediate analytics capability. The Eunomia synthetic dataset enables development and testing. OHDSI's population-level effect estimation tools can generate real-world evidence benchmarks for drugs in clinical trials.
- **Python package:** `ohdsi-common` (community), `Eunomia` (synthetic dataset)

### 5.3 SynPUF (CMS Synthetic Public Use Files)
- **URL:** https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files
- **What it contains:** Synthetic Medicare claims data modeled on real beneficiary patterns. Includes demographics, inpatient/outpatient claims, prescription drug events, and carrier claims for ~2.3M synthetic beneficiaries. Designed to preserve statistical properties while protecting privacy.
- **License:** Public domain (US Government)
- **Format:** CSV files
- **CTRA benefit:** Provides a proxy for real-world drug utilization and disease prevalence patterns in the elderly US population. Useful for modeling enrollment feasibility for trials targeting conditions prevalent in Medicare beneficiaries, and for understanding concomitant medication patterns that affect trial design.
- **Python package:** Load directly with pandas

---

## Category 6: Genomics / Biomarker Databases

### 6.1 PharmGKB (Pharmacogenomics Knowledge Base)
- **URL:** https://www.pharmgkb.org/
- **What it contains:** Curated gene-drug-disease relationships with clinical annotations. Contains variant data for >1,700 genes and >700 drugs, 9,000+ literature annotations, 153 drug-centered PK/PD pathways, clinical dosing guidelines, and drug label annotations from US, EU, Japan, Canada, and Switzerland. Includes FDA-recognized pharmacogenomic biomarkers.
- **License:** CC BY-SA 4.0
- **Format:** Bulk TSV downloads, REST API
- **CTRA benefit:** Pharmacogenomic biomarkers directly predict trial outcomes in precision medicine trials. If a trial drug has known PGx associations, this data can predict which patient populations will respond. Clinical dosing guideline data indicates whether dosing in a trial protocol aligns with pharmacogenomic evidence. Drug label PGx annotations signal regulatory expectations for biomarker-driven enrollment.
- **Python package:** `PharmCAT` (clinical annotation tool), or parse downloads with pandas

### 6.2 GWAS Catalog
- **URL:** https://www.ebi.ac.uk/gwas/
- **What it contains:** The largest public GWAS resource: >7,400 curated publications, >1,040,000 SNP-trait associations, >45,000 GWAS across >5,000 human traits, plus >40,000 full summary statistics datasets. Curated by NHGRI-EBI.
- **License:** Open access, FAIR-compliant
- **Format:** Spreadsheet downloads, REST API, summary statistics FTP
- **CTRA benefit:** Genetic associations between variants and diseases reveal the genetic architecture of trial target diseases. Trials targeting genetically validated mechanisms have ~2x higher success rates (per published analyses). GWAS data enables AutoCT to assess whether a trial's therapeutic hypothesis has genetic support, which is one of the strongest predictors of clinical success.
- **Python package:** `gwas-catalog` (community), REST API via `requests`

### 6.3 ClinVar
- **URL:** https://www.ncbi.nlm.nih.gov/clinvar/
- **What it contains:** >3M variant submissions from >2,800 organizations linking human genetic variants to clinical significance (pathogenic, benign, uncertain, etc.) and associated diseases. Includes germline and somatic variants.
- **License:** Public domain (US Government, NCBI)
- **Format:** FTP downloads (XML, VCF, TXT), REST API (E-utilities)
- **CTRA benefit:** Identifies disease-causing variants that inform biomarker strategies in clinical trials. For gene therapy and precision medicine trials, ClinVar data indicates whether the target variant is established as pathogenic, which predicts regulatory and scientific validity of the trial approach.
- **Python package:** `CANVAR` (annotation tool), or parse VCF with `cyvcf2`/`pysam`

### 6.4 GDSC (Genomics of Drug Sensitivity in Cancer)
- **URL:** https://www.cancerrxgene.org/
- **What it contains:** Drug sensitivity data for ~75,000 experiments covering 138 anticancer drugs across ~700 cancer cell lines, integrated with genomic data (mutations, copy number, gene expression) from the COSMIC database. Provides dose-response curves and IC50 values.
- **License:** Free access, no restrictions on data use
- **Format:** Bulk CSV downloads, FTP site
- **CTRA benefit:** For oncology trials, GDSC provides preclinical evidence of drug sensitivity linked to genomic markers. This predicts whether a cancer trial's biomarker strategy is supported by preclinical data. Drugs with strong preclinical genomic-response correlations are more likely to succeed in biomarker-selected trials.
- **Python package:** `gdscdata` (R package; Python via pandas CSV loading)

### 6.5 Gene Expression Omnibus (GEO)
- **URL:** https://www.ncbi.nlm.nih.gov/geo/
- **What it contains:** Public repository of >200K gene expression and functional genomics datasets (microarray, RNA-seq, ChIP-seq, methylation) from >6,000 organisms. Includes datasets on drug response, disease profiling, and biomarker discovery.
- **License:** Public domain (US Government, NCBI)
- **Format:** SOFT/MINiML format, bulk FTP downloads, GEO2R analysis tool
- **CTRA benefit:** Drug response gene expression signatures from GEO can predict whether a trial drug produces the expected molecular effect. Disease expression profiles help characterize target populations. Biomarker discovery datasets may reveal whether proposed trial biomarkers have been validated in prior studies.
- **Python package:** `GEOparse` (pip install)

### 6.6 cBioPortal
- **URL:** https://www.cbioportal.org/
- **What it contains:** Genomic and clinical data from >400 cancer studies including TCGA, AACR GENIE, and institutional datasets. Covers somatic mutations, copy-number alterations, mRNA expression, DNA methylation, protein abundance, and clinical outcomes (survival, treatment response).
- **License:** Open access (individual study licenses vary, most are open)
- **Format:** REST API (Swagger/OpenAPI), web interface, bulk downloads
- **CTRA benefit:** For oncology trials, cBioPortal provides the genomic landscape of the target cancer type, including mutation frequencies and co-occurrence patterns. This helps predict whether a trial's patient selection strategy is feasible (e.g., what % of patients carry the target mutation). Clinical outcome data provides real-world benchmarks for trial endpoints.
- **Python package:** `pyBioPortal` (pip install), `cbio_py` (pip install)

---

## Category 7 (Bonus): Cross-Cutting Infrastructure Resources

### 7.1 UMLS (Unified Medical Language System)
- **URL:** https://www.nlm.nih.gov/research/umls/
- **What it contains:** Metathesaurus linking >200 biomedical vocabularies (SNOMED CT, ICD-10, MeSH, RxNorm, MedDRA, HPO, etc.) with concept-level mappings. Semantic Network with 127 semantic types and 54 relationships.
- **License:** Free (requires UMLS license agreement, no charge)
- **Format:** Rich Release Format (RRF), REST API (UMLS Terminology Services)
- **CTRA benefit:** The master Rosetta Stone for biomedical terminology. UMLS enables mapping between all the different vocabularies used in ClinicalTrials.gov, FAERS, PubMed, EHRs, and knowledge graphs. Essential for AutoCT's RAG pipeline to retrieve relevant information regardless of the terminology used in the source.
- **Python package:** `umls-api` (community), REST API via `requests`

### 7.2 UniChem
- **URL:** https://www.ebi.ac.uk/unichem/
- **What it contains:** Compound identifier cross-referencing service mapping between ChEMBL, ChEBI, DrugBank, PubChem, ZINC, and many other chemical databases using Standard InChI as the linking key.
- **License:** Free, open access
- **Format:** REST API, bulk download of source-to-source mapping files
- **CTRA benefit:** When AutoCT encounters a drug identifier from one database, UniChem instantly resolves it to identifiers in all other databases. This eliminates the need for manual mapping when integrating information across ChEMBL, PubChem, DrugBank, and other chemical sources.
- **Python package:** REST API via `requests`

### 7.3 Therapeutics Data Commons (TDC)
- **URL:** https://tdcommons.ai/
- **What it contains:** 66 AI-ready datasets across 22 learning tasks spanning drug discovery and development. Includes ADMET prediction, drug-target interaction, drug combination, clinical trial outcome prediction tasks. Provides standardized data splits, evaluation metrics, and leaderboards. Hosts PrimeKG access.
- **License:** MIT (code), individual dataset licenses vary
- **Format:** Python API with automatic download and preprocessing
- **CTRA benefit:** TDC provides the closest thing to a "standard benchmark suite" for the CTRA problem space. Its clinical trial outcome prediction task provides direct comparison baselines. The standardized evaluation framework ensures reproducible benchmarking. The fact that it also hosts PrimeKG makes it a one-stop-shop for knowledge graph + benchmark data.
- **Python package:** `pytdc` (pip install), `from tdc.multi_pred import TrialOutcome`

---

## Protein Interaction Databases

### STRING
- **URL:** https://string-db.org/
- **What it contains:** Known and predicted protein-protein interactions for >14,000 organisms. Integrates experimental data, computational predictions, co-expression evidence, and text-mining results. Provides confidence-scored interaction edges.
- **License:** CC BY 4.0
- **Format:** Flat file downloads (TSV), REST API, Cytoscape integration
- **CTRA benefit:** Protein interaction networks contextualize drug targets within their biological neighborhood. A drug target with many high-confidence interaction partners in disease-relevant pathways is more likely to produce a therapeutic effect. Network topology features (degree, betweenness, clustering) are predictive of target tractability.
- **Python package:** `stringdb` (R; Python via REST API or TSV loading)

### BioGRID
- **URL:** https://thebiogrid.org/
- **What it contains:** 2,933,176 protein and genetic interactions, 31,540 chemical interactions, 1,128,339 post-translational modifications from 87,975 publications. Includes chemical-protein interactions from DrugBank.
- **License:** MIT
- **Format:** Tab-delimited downloads (MITAB, BioGRID Tab), REST API
- **CTRA benefit:** Chemical-protein interaction data supplements DrugBank for drug-target mapping. Post-translational modification data is relevant for trials targeting specific protein modifications. The genetic interaction data helps predict synthetic lethality-based trial approaches in oncology.
- **Python package:** Download and parse with pandas; REST API via `requests`

### Reactome
- **URL:** https://reactome.org/
- **What it contains:** Peer-reviewed pathway database covering >2,700 human pathways including metabolic, signaling, immune, and disease pathways. Includes Drug ADME (absorption, distribution, metabolism, excretion) pathways for selected drugs. Cross-references to >100 bioinformatics resources.
- **License:** CC BY 4.0
- **Format:** BioPAX, SBML, MySQL dump, flat files (https://reactome.org/download-data)
- **CTRA benefit:** Pathway context for trial drugs -- knowing which pathways a drug's target participates in predicts both efficacy potential and off-target toxicity risks. Drug ADME pathway data informs pharmacokinetic viability predictions. Pathway-level features enable grouping of mechanistically similar trials.
- **Python package:** `reactome2py` (pip install)

---

## Integration Priority Recommendations

Based on impact-to-effort ratio for the CTRA AutoCT pipeline:

### Tier 1 -- Integrate Immediately (High impact, low effort)
| Dataset | Why |
|---------|-----|
| **AACT** | Replaces per-record ClinicalTrials.gov API with full relational database. Enables population-level feature engineering. |
| **PrimeKG** | Single CSV, loads in seconds, provides multi-scale drug-disease-gene-pathway context for RAG. |
| **Open Targets** | Pre-computed target-disease evidence scores usable as direct features. Parquet downloads. |
| **RxNorm** | Essential drug name normalization for entity resolution across all data sources. |
| **Mondo** | Disease entity resolution across datasets. CC BY 4.0. |

### Tier 2 -- Integrate in Phase 2 (High impact, moderate effort)
| Dataset | Why |
|---------|-----|
| **ChEMBL** | Quantitative bioactivity data and phase transition history. Python client available. |
| **OnSIDES** | Complements FAERS with label-derived ADE data from 4 countries. |
| **PharmGKB** | Pharmacogenomic biomarker data directly predicts precision medicine trial outcomes. |
| **GWAS Catalog** | Genetic validation of drug targets (~2x success rate for genetically validated targets). |
| **CTD** | Chemical-gene interactions and toxicogenomics for safety prediction. |
| **TDC** | Benchmark framework for standardized evaluation. |

### Tier 3 -- Integrate in Phase 3+ (Moderate impact, variable effort)
| Dataset | Why |
|---------|-----|
| **Drugs@FDA / openFDA** | Regulatory outcome ground truth and review documents. |
| **DailyMed** | Full drug label text for RAG retrieval of prior trial results and safety data. |
| **MIMIC-IV** | Real-world treatment outcomes for enrollment feasibility and endpoint realism. |
| **Hetionet / DRKG** | Pre-trained embeddings as features; meta-path reasoning. |
| **TTD** | Target druggability scores and perturbation profiles. |
| **HPO** | Phenotype-based disease characterization for rare disease trials. |
| **Orphanet** | Rare disease prevalence and genetics. |
| **GEO** | Drug response gene expression signatures. |
| **UMLS** | Master vocabulary mapping for all-source integration. |

---

## Summary Statistics

| Category | Datasets Identified | Key Additions |
|----------|-------------------|---------------|
| Drug/Pharmacology | 7 | ChEMBL, PubChem, TTD, ChEBI, DailyMed, RxNorm, OnSIDES |
| Disease/Phenotype | 5 | Mondo, HPO, Disease Ontology, Orphanet, MeSH |
| Regulatory/Outcome | 5 | AACT, Drugs@FDA/openFDA, EMA Clinical Data, EU CTR/CTIS, FDA Snapshots |
| Knowledge Graphs | 6 | PrimeKG, Hetionet, DRKG, PharMeBINet, Open Targets, CTD |
| Real-World Evidence | 3 | MIMIC-IV, OMOP/OHDSI, CMS SynPUF |
| Genomics/Biomarkers | 6 | PharmGKB, GWAS Catalog, ClinVar, GDSC, GEO, cBioPortal |
| Cross-Cutting | 3 | UMLS, UniChem, TDC |
| Protein/Pathway | 3 | STRING, BioGRID, Reactome |
| **Total** | **38** | |

All datasets listed are free/open-access for academic and research use.
