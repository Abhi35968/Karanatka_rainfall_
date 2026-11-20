# Remaining Issues and Author Decisions: Stage 9 Final Manuscript

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Document Version:** Post-Stage 9 Revision  
**Location:** `Stage9_Final_Revised_Manuscript/remaining_issues_and_author_decisions.md`

---

## Purpose

This document itemizes all non-computational, structural, and administrative decisions that must be resolved by the human authors prior to formal submission to a peer-reviewed scientific journal. All empirical experiments, model evaluations, statistical audits, figures, and tables are frozen and finalized.

---

## 1. Author and Institutional Metadata

* **Current Status in Manuscript:**  
  The manuscript currently utilizes standardized blinded placeholders:
  - *Authors:* `[Author Names Blinded for Peer Review / To Be Inserted by Author]`
  - *Affiliations:* `[Department of Atmospheric and Oceanic Sciences / Hydrology / Geoinformatics, University / Institute, City, Country]`
  - *Corresponding Author:* `[Author correspondence email to be inserted upon acceptance]`
* **Action Required:**
  1. Determine whether the chosen target journal operates under a **double-blind** or **single-blind** review process:
     - *If Double-Blind (e.g., certain AMS or IEEE formats):* Retain the existing blinded placeholders in the main manuscript text and submit a separate Title Page containing author names, affiliations, and ORCID iDs.
     - *If Single-Blind (e.g., most Elsevier and Springer journals):* Replace the placeholders directly in `Karnataka_Rainfall_Research_Paper_Revised.docx` with full author names, institutional addresses, ORCID links, and the designated corresponding author's email.
  2. Verify author order and contribution roles in accordance with the CRediT (Contributor Roles Taxonomy) guidelines.

---

## 2. Target Journal Selection and Formatting Alignment

Four peer-reviewed journals represent high-impact, highly aligned venues for this manuscript. The authors should select the primary target venue:

| Journal | Publisher | Impact Factor (approx.) | Scope Alignment & Key Considerations |
| :--- | :--- | :--- | :--- |
| **Journal of Hydrometeorology (JHM)** | American Meteorological Society (AMS) | ~3.5–4.0 | **Primary Recommendation.** Premier venue for the interface of hydrometeorology, land-surface processes, and reanalysis downscaling. Prefers AMS citation style and explicit hydrological evaluation metrics (NSE, KGE). Word count limit typically ~7,500 words for main text. |
| **Atmospheric Research** | Elsevier | ~4.5–5.0 | **Strong Alternative.** Broad readership in monsoon dynamics, convective precipitation, and statistical downscaling. Accepts Elsevier standard formatting with flexible word count (~8,500–10,000 words). |
| **Computers & Geosciences** | Elsevier | ~4.0–4.5 | **Methodological Focus.** Excellent fit if emphasizing the open-source pipeline, machine learning reproducibility, and algorithmic trade-offs (Tweedie vs. Hurdle vs. CatBoost). |
| **Journal of Hydrology** | Elsevier | ~6.0–6.5 | Highly cited hydrology journal. Welcomes catchment-scale and regional water-balance reconstructions, but may request additional river basin streamflow routing verification if submitted as a full hydrological paper. |

* **Action Required:**
  - Select target journal and adjust font style, reference citation syntax (e.g., author-year vs. numbered), and title page layout to match the specific journal's Guide for Authors.

---

## 3. Data Availability and Code Accessibility Statement

* **Current Status in Manuscript:**  
  The manuscript states:
  > *"All modeling workflows, feature extraction scripts, trained models, and validation routines are available in the project reproducibility repository. The gridded daily IMD precipitation data are maintained by the National Data Centre (NDC), India Meteorological Department, Pune (accessible upon user registration). ERA5-Land reanalysis datasets are publicly available from the Copernicus Climate Data Store under an open license."*
* **Action Required:**
  1. **Code Repository:** Upload the frozen replication codebase (the scripts created in Stages 2 through 7) to a public platform such as GitHub or GitLab.
  2. **Data Archival / DOI Minting:** For permanent archival, archive the repository and preprocessed tabular Parquet files (`final_train_dataset.parquet`, `final_validation_dataset.parquet`, `final_test_dataset.parquet`) on **Zenodo** or **Figshare** to obtain a permanent Digital Object Identifier (DOI).
  3. Insert the specific DOI / URL into Section 6 (*Data and Code Availability*) of the DOCX manuscript before submission.

---

## 4. Funding, Grant Numbers, and Acknowledgments

* **Current Status in Manuscript:**  
  Section 7 contains a standard acknowledgment thanking the India Meteorological Department (IMD) for observational data and ECMWF/Copernicus for ERA5-Land reanalysis.
* **Action Required:**
  - Insert explicit funding agency names and grant/project numbers (e.g., Ministry of Earth Sciences [MoES], Department of Science and Technology [DST], Council of Scientific and Industrial Research [CSIR], or University Research Grants) that supported the study.
  - Acknowledge institutional High-Performance Computing (HPC) facilities or cloud computing credits if utilized.

---

## 5. Supplementary Information Strategy

* **Current Status:**  
  The manuscript contains 10 core figures and 8 primary tables embedded directly in the main body.
* **Author Decision:**
  - Check whether the target journal enforces strict limits on display items (many journals permit a maximum of 8 to 12 total figures and tables).
  - *If Display Item Limit is Exceeded:*
    - Relocate Table 2 (Feature Matrix Architecture, 63 variables) and Table 7 (Ablation Experiment Matrix) to a separate **Supplementary Information (SI)** document (`Supplementary_Material.docx`).
    - Retain Figures 1–10 and Tables 1, 3, 4, 5, 6A, 6B, and 8 in the main text.

---

## 6. Declaration of Competing Interests

* **Confirmation Needed:**
  - Confirm that all co-authors agree with the standard declaration:
    > *"The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."*

---
*End of Remaining Issues and Author Decisions.*

