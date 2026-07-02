# DrugBank Alternatives Research: Free/Open-Access Drug Data Sources for CTRA

**Date:** 2026-03-28
**Context:** DrugBank requires a paid license (academic license required for XML/bulk data). This document evaluates free, open-access alternatives that can supply the drug data CTRA needs for its LinearRAG pipeline, specifically: mechanisms of action, drug targets, drug-drug interactions, approval status, SMILES molecular structures, and drug name synonyms for cross-source entity normalization.

**Current DrugBank usage in CTRA** (see `src/ctra/data/drugbank_loader.py`):
- XML parsing into structured DataFrame with fields: `drugbank_id`, `name`, `description`, `indication`, `pharmacodynamics`, `mechanism_of_action`, `toxicity`, `metabolism`, `absorption`, `half_life`, `route_of_elimination`, `targets`, `enzymes`, `drug_interactions`, `categories`, `synonyms`, `approval_status`, `updated_date`
- Text passage generation for LinearRAG indexing (drug name, description, indication, mechanism, targets, toxicity, categories)
- Drug synonym list used as canonical name mapping across ClinicalTrials.gov, PubMed, and FAERS

---

## Executive Summary: Recommended Replacement Strategy

**Primary replacement: ChEMBL + DrugCentral + PubChem (combined)**

No single free database replicates DrugBank's breadth. However, the combination of these three provides full coverage of every data field CTRA currently extracts from DrugBank:

| CTRA Data Need | ChEMBL | DrugCentral | PubChem | Best Source |
|---|---|---|---|---|
| Mechanisms of action | Yes (5,392 drugs annotated) | Yes (~724 MoA targets) | Partial | ChEMBL + DrugCentral |
| Drug targets | Yes (extensive) | Yes | Partial | ChEMBL |
| Drug-drug interactions | No | Yes (from labels) | Partial | DrugCentral |
| Approval status | Yes (17,500 drugs with pipeline status) | Yes (4,959 approved APIs) | Partial | ChEMBL + DrugCentral |
| SMILES structures | Yes | Yes (SDF, SMILES, InChI) | Yes (115M+ compounds) | PubChem or ChEMBL |
| Drug synonyms | Yes (synonyms, trade names) | Limited | Yes (extensive) | PubChem + ChEMBL |
| Drug description/indication | Yes | Yes (14,300+ on/off-label uses) | Yes | DrugCentral |
| Pharmacology text (PD, PK, toxicity) | Partial | Yes (full label data) | Partial | DrugCentral |
| Enzymes/metabolism | Partial | Partial | Yes (BioAssays) | ChEMBL |
| Categories/classification | Yes | Yes | Yes | ChEMBL |

**Supplementary sources for enrichment:** Open Targets Platform (disease-target links), UniProt (target protein details), DGIdb (drug-gene interactions meta-aggregator), TTD (target-disease associations).

**Cross-reference glue:** UniChem (EBI) provides free identifier mapping across ChEMBL, PubChem, DrugCentral, KEGG, BindingDB, and 40+ other sources using Standard InChI keys.

---

## Tier 1: Primary Replacements (Recommended)

### 1. ChEMBL (EMBL-EBI)

**URL:** https://www.ebi.ac.uk/chembl/

**What it provides:**
- 2.2M+ compounds with bioactivity data, 18M+ activity records
- 17,500 approved drugs and clinical pipeline compounds (ChEMBL 35+)
- Mechanism of action annotations for 5,392 compounds
- Indications for 7,590 compounds
- Black box warnings for 592 compounds; 202 withdrawal records
- Drug targets (proteins, nucleic acids), enzymes, transporters
- SMILES, InChI, molecular properties for all small molecules
- Biological sequences for biotherapeutics
- Clinical trial phase progression data
- Chemical probe dataset (new in ChEMBL 36)

**License:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC-BY-SA 3.0). Free for all uses including commercial, with attribution and share-alike requirements.

**Download format:**
- SQLite database (~12 GB uncompressed) -- simplest option, no server needed
- PostgreSQL and MySQL dumps
- SDF files (chemical structures)
- FASTA files (protein sequences)
- REST API with 25 endpoints (paginated, no auth required)
- FTP bulk downloads from EBI

**Python access:**
- `chembl_webresource_client` (official, pip install) -- Django QuerySet-style lazy API, local caching
- `chembl-downloader` (pip install) -- automated SQLite download and extraction
- Direct REST API via `requests`

**Size/coverage:** Largest open bioactivity database. 2.2M+ compounds, 15K+ targets, 1.6M+ assays, 18M+ activities. Updated approximately annually (ChEMBL 36 released July 2025).

**Relevance to CTRA:**
- **HIGH.** Directly replaces DrugBank for: mechanisms of action, drug targets, approval status, SMILES, drug synonyms/trade names, clinical phase data
- Missing: full drug-drug interaction lists (use DrugCentral), detailed PK/PD text narratives (use DrugCentral)
- The `mechanism` endpoint provides curated MoA data in a structured format that maps directly to CTRA's `mechanism_of_action` field
- The `drug` endpoint provides approval status, indications, and clinical phase data

**Migration effort:** MODERATE. ChEMBL's relational schema is more complex than DrugBank's flat XML. Requires writing a new loader class that queries the SQLite database or REST API and maps to the same output schema as `DrugBankLoader`.

---

### 2. DrugCentral (University of New Mexico)

**URL:** https://drugcentral.org/

**What it provides:**
- 4,959 active pharmaceutical ingredients (4,805 human, 396 veterinary)
- ~724 mechanism-of-action targets (curated)
- ~20,000 bioactivity data points
- 14,300+ on- and off-label indications/uses
- 27,000+ contraindications
- ~340,000 adverse drug events (pharmacovigilance)
- Drug-drug interactions (from FDA labels)
- Chemical structures (SMILES, InChI, SDF in MOL V2000/V3000)
- FDA, EMA, and PMDA approval data with dates
- Drug labels text (dosage, warnings, precautions, PK/PD)
- Regulatory monitoring for new approvals

**License:** Creative Commons Attribution-ShareAlike 4.0 (CC-BY-SA 4.0). Fully open access, no registration required.

**Download format:**
- PostgreSQL database dump (full relational database)
- TSV files (drug-target interactions)
- CSV files (approved drug lists by agency)
- SDF/SMILES/InChI structure files
- Public PostgreSQL instance at `drugcentral:unmtid-dbs.net:5433`
- Smart API for programmatic access

**Python access:**
- Direct PostgreSQL connection via `psycopg2` or `sqlalchemy`
- REST/Smart API
- No dedicated Python package (but straightforward SQL queries)

**Size/coverage:** ~5,000 FDA/EMA/PMDA-approved drugs. Smaller than ChEMBL but focused specifically on approved/clinical drugs with rich clinical annotations.

**Relevance to CTRA:**
- **HIGH.** Best free source for: drug-drug interactions, detailed clinical text (indications, contraindications, PK, PD, toxicity), FDA approval status and dates
- The clinical label text is ideal for LinearRAG passage generation -- already structured as natural language
- Directly replaces DrugBank for: `drug_interactions`, `indication`, `pharmacodynamics`, `toxicity`, `metabolism`, `absorption`, `half_life`, `route_of_elimination`, `approval_status`
- Approval date tracking enables temporal filtering for leakage prevention

**Migration effort:** LOW-MODERATE. PostgreSQL dump can be loaded locally. Schema is relational but well-documented. The drug label text fields map almost 1:1 to DrugBank's XML fields.

---

### 3. PubChem (NCBI/NIH)

**URL:** https://pubchem.ncbi.nlm.nih.gov/

**What it provides:**
- 115M+ compounds (world's largest free chemical database)
- SMILES, InChI, InChIKey, SDF for all compounds
- Molecular properties, descriptors, fingerprints, 3D conformers
- 1.5M+ bioassay records with bioactivity results
- Drug-target binding data
- Pharmacological actions
- Drug synonyms (very comprehensive -- often the best source for name normalization)
- Patent references
- Safety and toxicity data
- Literature references (linked to PubMed)

**License:** Public domain (US government work). No restrictions whatsoever.

**Download format:**
- PUG-REST API (no auth, no API key, free unlimited use)
- PUG-View API (full compound records)
- FTP bulk downloads (compounds, substances, bioassays as XML, SDF, JSON)
- SMILES/InChI text files

**Python access:**
- `PubChemPy` (pip install, latest v1.0.5 Sep 2025) -- clean Pythonic interface
- `requests` against PUG-REST API directly
- Biopython has partial PubChem support

**Size/coverage:** 115M+ compounds, 300M+ substances, 1.5M+ bioassays. By far the largest chemical database. Coverage of approved drugs is comprehensive but embedded in a much larger dataset.

**Relevance to CTRA:**
- **HIGH for structures and synonyms.** Best source for: SMILES retrieval by drug name, comprehensive synonym lists for entity normalization, molecular properties
- **MODERATE for mechanisms/targets.** Bioassay data is extensive but less curated than ChEMBL for mechanism annotations
- PubChem CID serves as a universal compound identifier that links to nearly all other databases
- The synonym service is especially valuable for CTRA's cross-source entity normalization (mapping ClinicalTrials.gov intervention names to canonical drug identifiers)

**Migration effort:** LOW. PubChemPy provides a simple API. Bulk data can be fetched per-compound or downloaded via FTP.

---

## Tier 2: Valuable Supplementary Sources

### 4. Open Targets Platform

**URL:** https://platform.opentargets.org/

**What it provides:**
- Disease-target-drug associations with evidence scores
- Drug mechanisms of action (curated from ChEMBL)
- Approved and experimental drug indications
- Clinical trial data linked from ClinicalTrials.gov
- Pharmacovigilance data
- Target tractability assessments
- Genetic evidence linking targets to diseases

**License:** Open access. Data available under EMBL-EBI terms of use. Quarterly updates.

**Download format:**
- GraphQL API
- Parquet bulk downloads (post v25.03, Parquet only)
- Google Cloud datasets
- EMBL-EBI FTP server

**Python access:**
- `opentargets` (PyPI, but archived/legacy)
- GraphQL queries via `requests`/`gql`
- Direct Parquet reading via `polars`/`pandas`

**Size/coverage:** Integrates data from 20+ sources. Covers thousands of targets, diseases, and drugs with scored associations.

**Relevance to CTRA:**
- **MODERATE-HIGH.** Excellent for: disease-target evidence scoring (which targets are most validated for which diseases), linking drugs to genetic evidence
- The evidence scoring could be a powerful feature for trial outcome prediction (trials targeting genetically-validated targets have higher success rates)
- Drug mechanism data is sourced from ChEMBL, so overlaps with direct ChEMBL access
- Clinical trial linking could help validate/enrich ClinicalTrials.gov data

---

### 5. UniProt

**URL:** https://www.uniprot.org/

**What it provides:**
- Comprehensive protein sequence and function database
- Drug target protein details: function, subcellular location, post-translational modifications
- Protein-protein interactions
- Disease associations
- Gene Ontology annotations
- Cross-references to ChEMBL, PDB, and 200+ databases

**License:** Creative Commons Attribution 4.0 (CC-BY 4.0). Free, open access, no login required.

**Download format:**
- REST API (powerful query language, multiple output formats)
- Bulk downloads (XML, FASTA, TSV, JSON, RDF)
- FTP site

**Python access:**
- `uniprot` (PyPI, v1.4.1, Jan 2026)
- `UniProtMapper` (PyPI, ID mapping and field queries)
- Direct REST API

**Size/coverage:** 250M+ protein sequences (UniProtKB). ~570K reviewed entries (Swiss-Prot).

**Relevance to CTRA:**
- **MODERATE.** Provides detailed protein target information that ChEMBL references but doesn't fully describe
- UniProt accession IDs are the standard identifiers for drug targets across ChEMBL, Open Targets, and TTD
- Useful for enriching target-level features (e.g., target protein family, subcellular location, druggability)
- Not a direct DrugBank replacement, but valuable for target characterization

---

### 6. DGIdb (Drug Gene Interaction Database)

**URL:** https://dgidb.org/

**What it provides:**
- Meta-aggregator of drug-gene interactions from 40+ sources
- Gene druggability categories
- Drug-gene interaction types (inhibitor, activator, etc.)
- Sources include: ChEMBL, DrugBank (public subset), PharmGKB, TEND, CIViC, OncoKB, and more

**License:** Free, open access. Code on GitHub under MIT license.

**Download format:**
- GraphQL API
- TSV bulk downloads (genes, drugs, interactions, categories)
- Monthly TSV snapshots

**Python access:**
- `DGIpy` (official Python client)
- GraphQL queries via `requests`
- `R-DGIdb` (R package)

**Size/coverage:** Aggregates from 40+ sources. DGIdb 5.0 (latest) rebuilt for precision medicine pipelines.

**Relevance to CTRA:**
- **MODERATE.** Useful as a meta-aggregator that combines drug-target interaction data from multiple sources in one query
- Can supplement ChEMBL's target data with additional druggability annotations
- The interaction type categorization (inhibitor, activator, etc.) is useful for feature engineering
- Not a primary source but a good validation/enrichment layer

---

### 7. Guide to Pharmacology (IUPHAR/BPS GtoPdb)

**URL:** https://www.guidetopharmacology.org/

**What it provides:**
- 3,127 human pharmacological targets (2026.1 release, March 2026)
- 13,721 ligands with 9,895 having curated quantitative target interactions
- 2,196 approved drugs (1,238 with curated quantitative interactions)
- Expert-curated quantitative pharmacology data (Ki, IC50, EC50, Kd)
- Receptor nomenclature and classification
- Detailed mechanism descriptions

**License:** Open Database License (ODbL) for database; CC-BY-SA 4.0 for contents. Fully open.

**Download format:**
- CSV/TSV downloads (targets, ligands, interactions, ID mappings)
- REST API (JSON format) at `https://www.guidetopharmacology.org/services/`
- RDF/N3 linked data files
- Cross-references to HGNC, UniProt

**Python access:**
- REST API via `requests`
- No dedicated Python package

**Size/coverage:** Smaller than ChEMBL but very high quality expert curation. Focus on established pharmacological targets.

**Relevance to CTRA:**
- **MODERATE.** Best source for: quantitative binding affinity data, expert pharmacological classifications
- The quantitative interaction data (Ki, IC50) could be valuable features for predicting trial outcomes
- Smaller coverage means it won't replace ChEMBL/DrugCentral for breadth

---

### 8. TTD (Therapeutic Target Database)

**URL:** https://idrblab.org/ttd/

**What it provides:**
- 3,798 targets, 40,398 drugs (TTD 2026 release)
- 306,247 target-disease associations covering 2,912 targets
- 10,506 perturbation profiles on 2,368 targets
- Multidimensional activity landscapes for 17,806 drugs
- Clinical profiles for 2,234 approved drugs (indications, dosage, PK, PD, mechanism, drug interactions, adverse reactions, clinical studies)
- Target validation status (successful, clinical trial, research)

**License:** Free, no login required. Academic use.

**Download format:**
- Flat file downloads (TSV)
- Web search interface
- Rebuilt on Vue3/Django (2026)

**Python access:**
- No dedicated Python package
- Download files and parse with `pandas`/`polars`

**Size/coverage:** 40K+ drugs, 3.8K targets. Strong on target-disease associations and approved drug clinical profiles.

**Relevance to CTRA:**
- **MODERATE-HIGH.** The target-disease association data and clinical profiles are directly relevant to trial outcome prediction
- Target validation status (successful target vs. research-only) is a potentially powerful predictive feature
- The 306K target-disease associations could enrich CTRA's feature engineering
- Clinical profiles overlap with DrugCentral but include additional structured fields

---

## Tier 3: Specialized/Niche Sources

### 9. BindingDB

**URL:** https://www.bindingdb.org/

**What it provides:**
- 3.2M binding affinity measurements for 1.4M compounds against 11.4K targets
- Quantitative binding data (Ki, IC50, EC50, Kd)
- SMILES structures for all compounds
- Target protein sequences

**License:** CC-BY 3.0 for curated data. Free access.

**Download format:** Bulk downloads (TSV, SDF), web services, SMILES/InChI queries.

**Python access:** REST API; no dedicated package.

**Relevance to CTRA:** LOW-MODERATE. Niche but valuable for quantitative binding data that could inform drug potency features. Overlaps significantly with ChEMBL bioactivity data.

---

### 10. PharmGKB

**URL:** https://www.pharmgkb.org/

**What it provides:**
- Pharmacogenomic annotations: gene-drug-disease relationships
- Clinical guidelines for pharmacogenomic testing
- Drug label annotations (FDA, EMA, HCSC, CPIC)
- Variant-drug associations
- 715 drugs, 1,761 genes, 227 diseases annotated

**License:** Creative Commons license. Free, no registration required.

**Download format:** Zipped spreadsheets (TSV), API.

**Python access:** REST API via `requests`.

**Relevance to CTRA:** LOW-MODERATE. Pharmacogenomics is a niche concern for trial prediction. However, pharmacogenomic complexity of a drug could be a useful feature (drugs with many PGx interactions may have more variable outcomes).

---

### 11. KEGG DRUG

**URL:** https://www.genome.jp/kegg/drug/

**What it provides:**
- Drug structures, therapeutic targets, metabolizing enzymes
- Drug-drug interaction networks
- Pathway-level drug mechanism mapping

**License:** FREE for academic use only. Commercial use requires a paid license. Bulk FTP download requires paid subscription (since 2011). API limited to 10 entries per request.

**Download format:** REST API (rate-limited, 10 entries/request), web interface. No free bulk download.

**Python access:** `Bio.KEGG.REST` (Biopython), `KEGGREST` (R/Bioconductor).

**Relevance to CTRA:** LOW. The severe API rate limits and no bulk download make it impractical for building a RAG index. Pathway data is available elsewhere (Reactome, Open Targets). **Not recommended** as a primary source.

---

### 12. RxNorm (NLM)

**URL:** https://lhncbc.nlm.nih.gov/RxNav/

**What it provides:**
- Normalized drug names and identifiers
- Links between drug vocabularies (NDC, SNOMED CT, MeSH, ATC)
- Drug-drug interactions (sourced from ONCHigh and DrugBank)
- RxCUI universal drug identifiers

**License:** Free (non-proprietary NLM vocabulary). No API key needed. 20 req/sec rate limit. Bulk download requires free UMLS license agreement.

**Download format:** REST API, bulk RRF files (with UMLS license), Docker container (RxNav-in-a-Box).

**Python access:** REST API via `requests`.

**Relevance to CTRA:** MODERATE for entity normalization. RxNorm's normalized drug naming and cross-vocabulary mapping could replace DrugBank synonyms for entity normalization. The drug-drug interaction data partially comes from DrugBank, however.

---

### 13. FDA Orange Book / OpenFDA

**URL:** https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files | https://open.fda.gov/

**What it provides:**
- FDA-approved drug products with therapeutic equivalence
- Patent and exclusivity data
- Approval dates and application numbers
- OpenFDA: adverse events (FAERS), recalls, labeling, NDC directory

**License:** Public domain (US government). Fully free.

**Download format:** Orange Book: ZIP files (CSV). OpenFDA: Elasticsearch-based REST API (1,000 records/call), bulk JSON downloads.

**Python access:** `requests` against OpenFDA API. No dedicated package.

**Relevance to CTRA:** MODERATE. Approval dates are useful for temporal filtering. FAERS adverse event data (via OpenFDA) is already planned as a CTRA data source. Orange Book patent data could be a novel predictive feature (patent cliff timing vs. trial initiation).

---

### 14. ChemicalProbes.org

**URL:** https://www.chemicalprobes.org/

**What it provides:**
- Expert-reviewed chemical probes (high-quality tool compounds)
- 570+ human protein targets covered
- Probe quality ratings, usage recommendations, caveats
- Bioactivity data, selectivity data

**License:** Free, open access. PostgreSQL/SQLite database dump available.

**Download format:** PostgreSQL dump, SQLite database, web exports.

**Relevance to CTRA:** LOW. Very specialized (chemical probes, not approved drugs). Only ~570 targets. Not relevant for trial outcome prediction.

---

### 15. STITCH (EMBL)

**URL:** http://stitch.embl.de/

**What it provides:**
- Chemical-protein interaction networks
- 9.6M proteins, 1.6B interactions
- Integration of experimental, pathway, text-mining, and predicted interactions
- SMILES-based querying

**License:** Creative Commons (non-commercial subset has separate licensing).

**Download format:** Bulk downloads, REST API, web interface.

**Relevance to CTRA:** LOW-MODERATE. Large-scale interaction network could enrich drug-target relationships. However, much of its data comes from ChEMBL and other sources already covered.

---

### 16. UniChem (EMBL-EBI)

**URL:** https://www.ebi.ac.uk/unichem/

**What it provides:**
- Cross-reference mapping between 40+ chemical databases using Standard InChI
- Maps between: ChEMBL, PubChem, DrugCentral, KEGG, BindingDB, DrugBank, and more
- Identifier tracking (current vs. obsolete IDs)

**License:** Free, open access (EBI resource).

**Download format:** REST API, bulk source maps, weekly updates.

**Relevance to CTRA:** **HIGH as infrastructure.** UniChem is the glue that connects all the other databases. When CTRA encounters a drug name from ClinicalTrials.gov, UniChem can map it to ChEMBL IDs, PubChem CIDs, and DrugCentral IDs in one step. Essential for the multi-source replacement strategy.

---

## Recommended Implementation Plan

### Phase 1: Core Replacement (replaces DrugBank data loader)

Create a new `MultiSourceDrugLoader` that replaces `DrugBankLoader` and combines data from ChEMBL + DrugCentral + PubChem:

```
Data field mapping:
---------------------------------------------------------------
DrugBank field          -> Primary source      -> Fallback
---------------------------------------------------------------
drugbank_id             -> ChEMBL CHEMBL_ID    -> PubChem CID
name                    -> ChEMBL pref_name    -> PubChem IUPAC name
description             -> DrugCentral label   -> ChEMBL description
indication              -> DrugCentral         -> ChEMBL indication
pharmacodynamics        -> DrugCentral label   -> (omit)
mechanism_of_action     -> ChEMBL mechanism    -> DrugCentral MoA
toxicity                -> DrugCentral label   -> (omit)
metabolism              -> DrugCentral label   -> (omit)
absorption              -> DrugCentral label   -> (omit)
half_life               -> DrugCentral label   -> (omit)
route_of_elimination    -> DrugCentral label   -> (omit)
targets                 -> ChEMBL targets      -> DGIdb
enzymes                 -> ChEMBL mechanisms   -> (omit)
drug_interactions       -> DrugCentral         -> (omit)
categories              -> ChEMBL ATC codes    -> DrugCentral
synonyms                -> PubChem synonyms    -> ChEMBL synonyms
approval_status         -> ChEMBL max_phase    -> DrugCentral
updated_date            -> ChEMBL/DrugCentral  -> (latest)
```

### Phase 2: Entity Normalization (replaces DrugBank synonym mapping)

1. Use **PubChem synonyms API** as primary synonym source (most comprehensive)
2. Use **UniChem** to map between identifier systems (ClinicalTrials.gov drug name -> PubChem CID -> ChEMBL ID -> DrugCentral ID)
3. Use **RxNorm** RxCUI for clinical drug name normalization

### Phase 3: Enrichment (new features not available from DrugBank)

1. **Open Targets**: evidence scores for target-disease associations (novel predictive feature)
2. **TTD**: target validation status (successful/clinical/research)
3. **GtoPdb**: quantitative binding affinity data (Ki/IC50)

### Local Database Strategy

For LinearRAG indexing, bulk-download and store locally:
1. **ChEMBL SQLite** (~12 GB) -- single file, no server needed
2. **DrugCentral PostgreSQL dump** -- load into local PostgreSQL or convert to SQLite
3. **PubChem**: download approved drug subset only via PUG-REST (not full 115M compound database)

### Python Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
chembl-webresource-client = ">=0.10.9"
chembl-downloader = ">=0.4.0"
PubChemPy = ">=1.0.5"
psycopg2-binary = ">=2.9"    # for DrugCentral PostgreSQL
uniprot = ">=1.4"             # optional, for target enrichment
```

---

## Comparison Matrix: All Sources at a Glance

| Database | License | MoA | Targets | DDI | SMILES | Approval | Synonyms | Bulk DL | Python Pkg | Maintained |
|---|---|---|---|---|---|---|---|---|---|---|
| **ChEMBL** | CC-BY-SA 3.0 | +++ | +++ | - | +++ | +++ | ++ | SQLite/PG/MySQL | Yes | Yes (2025) |
| **DrugCentral** | CC-BY-SA 4.0 | ++ | ++ | +++ | ++ | +++ | + | PostgreSQL | No (SQL) | Yes |
| **PubChem** | Public domain | + | + | + | +++ | + | +++ | FTP/API | Yes | Yes |
| **Open Targets** | Open access | ++ | +++ | - | - | ++ | + | Parquet/API | Partial | Yes (quarterly) |
| **UniProt** | CC-BY 4.0 | - | +++ | - | - | - | + | FTP/API | Yes | Yes |
| **DGIdb** | MIT/open | + | +++ | - | - | - | + | TSV | Yes | Yes (5.0) |
| **GtoPdb** | ODbL/CC-BY-SA 4.0 | ++ | +++ | - | ++ | ++ | + | CSV/API | No (REST) | Yes (2026) |
| **TTD** | Free/academic | ++ | +++ | + | + | ++ | + | TSV | No | Yes (2026) |
| **BindingDB** | CC-BY 3.0 | + | ++ | - | +++ | - | + | TSV/SDF | No (REST) | Yes |
| **PharmGKB** | CC | + | + | - | - | + | + | TSV | No (REST) | Yes |
| **KEGG DRUG** | Academic only | ++ | ++ | ++ | + | + | + | **No free bulk** | Biopython | Yes |
| **RxNorm** | Free (NLM) | - | - | ++ | - | + | +++ | RRF (UMLS lic) | No (REST) | Yes |
| **FDA/OpenFDA** | Public domain | - | - | - | - | +++ | + | ZIP/API | No | Yes |
| **ChemicalProbes** | Open | + | + | - | + | - | - | SQLite/PG | No | Yes |
| **STITCH** | CC (partial) | + | ++ | - | ++ | - | + | Bulk/API | No | Unclear |
| **UniChem** | Free (EBI) | - | - | - | - | - | +++ (cross-ref) | Bulk/API | No | Yes (weekly) |

Legend: `+++` = comprehensive, `++` = good, `+` = partial, `-` = not available

---

## Key Risks and Mitigations

1. **No single source replaces DrugBank**: Mitigation -- the three-source strategy (ChEMBL + DrugCentral + PubChem) covers all fields. UniChem handles cross-referencing.

2. **Drug-drug interaction coverage**: DrugCentral is the only fully free source with curated DDI data from labels. RxNorm's DDI API partially sources from DrugBank. Mitigation -- DrugCentral DDI data is sufficient for CTRA's RAG passages.

3. **Entity normalization without DrugBank synonyms**: PubChem synonym API is more comprehensive than DrugBank for name variants. Combined with UniChem cross-referencing, this is actually an upgrade.

4. **Data freshness**: ChEMBL updates ~annually, DrugCentral monitors regulatory agencies continuously, PubChem updates daily. Combined, freshness matches or exceeds DrugBank's quarterly XML releases.

5. **TrialBench compatibility**: TrialBench was constructed using DrugBank drug synonyms for linking ClinicalTrials.gov to drug records. The replacement strategy must produce equivalent drug-trial links. Mitigation -- use PubChem + UniChem synonym mapping and validate against TrialBench's existing drug-trial pairs.

---

## Sources

- [ChEMBL Database](https://www.ebi.ac.uk/chembl/)
- [ChEMBL Data Web Services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
- [ChEMBL Downloads](https://chembl.gitbook.io/chembl-interface-documentation/downloads)
- [ChEMBL 36 Release](https://chembl.blogspot.com/2025/09/chembl-36-is-out.html)
- [chembl_webresource_client (GitHub)](https://github.com/chembl/chembl_webresource_client)
- [chembl-downloader (PyPI)](https://pypi.org/project/chembl-downloader/)
- [Drug and Clinical Candidate Drug Data in ChEMBL (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12516679/)
- [DrugCentral](https://drugcentral.org/)
- [DrugCentral Download](https://drugcentral.org/download)
- [DrugCentral 2023 (NAR)](https://academic.oup.com/nar/article/51/D1/D1276/6885038)
- [Exploring DrugCentral: from molecular structures to clinical effects](https://pmc.ncbi.nlm.nih.gov/articles/PMC10692006/)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
- [PubChem Downloads](https://pubchem.ncbi.nlm.nih.gov/docs/downloads)
- [PubChemPy (PyPI)](https://pypi.org/project/PubChemPy/)
- [Open Targets Platform](https://platform.opentargets.org/)
- [Open Targets Platform Documentation](https://platform-docs.opentargets.org/)
- [Open Targets Drug Endpoint](https://platform-docs.opentargets.org/drug)
- [UniProt](https://www.uniprot.org/)
- [UniProt API Documentation](https://www.uniprot.org/api-documentation)
- [uniprot (PyPI)](https://pypi.org/project/uniprot/)
- [DGIdb](https://dgidb.org/)
- [DGIdb 5.0 (NAR 2024)](https://academic.oup.com/nar/article/52/D1/D1227/7416371)
- [Guide to Pharmacology](https://www.guidetopharmacology.org/)
- [GtoPdb 2026 (NAR)](https://academic.oup.com/nar/article/54/D1/D1446/8306131)
- [GtoPdb Downloads](https://www.guidetopharmacology.org/download.jsp)
- [TTD 2026 (NAR)](https://academic.oup.com/nar/article/54/D1/D1692/8324952)
- [BindingDB](https://www.bindingdb.org/)
- [BindingDB in 2024 (NAR)](https://academic.oup.com/nar/article/53/D1/D1633/7906836)
- [PharmGKB](https://www.pharmgkb.org/)
- [KEGG DRUG](https://www.genome.jp/kegg/drug/)
- [KEGG API](https://www.kegg.jp/kegg/rest/keggapi.html)
- [RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/index.html)
- [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html)
- [FDA Orange Book Data Files](https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files)
- [OpenFDA APIs](https://open.fda.gov/apis/)
- [Chemical Probes Portal](https://www.chemicalprobes.org/)
- [STITCH](http://stitch.embl.de/)
- [UniChem](https://www.ebi.ac.uk/unichem/)
- [Biostars: Alternatives to DrugBank Database](https://www.biostars.org/p/490100/)
- [Assessing the public landscape of clinical-stage pharmaceuticals (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7315808/)
