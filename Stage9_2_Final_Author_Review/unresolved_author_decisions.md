# Unresolved Author Decisions and Pre-Submission Roadmap

**Project:** Machine-Learning-Based Retrospective Daily Rainfall Reconstruction over Karnataka Using ERA5-Land Reanalysis and IMD Gridded Observations (2019–2025)  
**Location:** `Stage9_2_Final_Author_Review/unresolved_author_decisions.md`  
**Date:** November 2026  
**Status:** All empirical, mathematical, computational, and statistical validations are **100% complete and frozen**. The decisions itemized below are reserved strictly for the human authors prior to journal upload.

---

## 1. Author and Institutional Metadata

* **Current Status in Manuscript:**  
  The manuscript contains standardized blinded placeholders:
  - *Authors:* `[Author Names Blinded for Peer Review / To Be Inserted by Author]`
  - *Affiliations:* `[Department / Institutional Affiliation To Be Inserted by Author]`
  - *Correspondence:* `[Correspondence Email To Be Inserted Upon Acceptance]`
* **Action Required by Authors:**
  1. Determine the target journal's peer-review model:
     - **If Double-Blind (e.g., AMS Journal of Hydrometeorology):** Keep the blinded placeholders in `Karnataka_Rainfall_Research_Paper_Final.docx`. Prepare a separate Title Page document containing author names, institutional affiliations, ORCID iDs, and corresponding author contact details.
     - **If Single-Blind (e.g., Elsevier Atmospheric Research):** Replace the bracketed placeholders directly in the DOCX title header with the full author list, institutional affiliations, and corresponding author email.
  2. Confirm author sequence and assign CRediT roles (Conceptualization, Methodology, Software, Formal Analysis, Writing - Original Draft, Writing - Review & Editing).

---

## 2. Target Journal Selection

Four peer-reviewed journals represent high-impact, highly aligned venues for this study. The authors should select the primary target venue:

| Journal | Publisher | Impact Factor (approx.) | Scope Alignment & Key Considerations |
| :--- | :--- | :--- | :--- |
| **Journal of Hydrometeorology (JHM)** | American Meteorological Society (AMS) | ~3.5–4.0 | **Primary Recommendation.** Premier venue for the interface of hydrometeorology, land-surface processes, and reanalysis downscaling. High appreciation for hydrological evaluation metrics (NSE, KGE-2012) and rigorous error decomposition. Typical length: ~7,500 words for main text. |
| **Atmospheric Research** | Elsevier | ~4.5–5.0 | **Strong Alternative.** Broad readership across monsoon dynamics, complex terrain orography, and statistical downscaling. Welcomes detailed diagnostic figures and comprehensive sensitivity analyses. Flexible length (~8,500–10,000 words). |
| **Computers & Geosciences** | Elsevier | ~4.0–4.5 | **Methodological Focus.** Excellent fit if highlighting open-source machine learning workflows, multi-objective optimization (Tweedie vs. Hurdle vs. CatBoost), and end-to-end reproducibility. |
| **Journal of Hydrology** | Elsevier | ~6.0–6.5 | Highly cited venue. Excellent for catchment water-balance reconstructions, but reviewers may request streamflow discharge simulation routing if submitted as a full hydrological paper. |

---

## 3. Data Archival and Code Availability Statement

* **Current Status in Manuscript:**  
  Section 7 outlines data sources and repository structure:
  > *"All modeling workflows, feature extraction scripts, trained model joblib files, and validation routines are available in the project reproducibility package. Gridded daily IMD precipitation data are maintained by the National Data Centre (NDC), India Meteorological Department, Pune (accessible upon user registration). ERA5-Land reanalysis datasets are publicly available from the Copernicus Climate Data Store under an open license."*
* **Action Required by Authors:**
  1. Upload the frozen replication repository (scripts from Stages 2 through 7) to GitHub or GitLab.
  2. Archive the repository alongside the preprocessed tabular Parquet files (`final_train_dataset.parquet`, `final_validation_dataset.parquet`, `final_test_dataset.parquet`) on **Zenodo** or **Figshare** to mint a permanent Digital Object Identifier (DOI).
  3. Insert the minted Zenodo DOI link into Section 7 of the final DOCX manuscript.

---

## 4. Funding, Grants, and Acknowledgments

* **Current Status:**  
  Section 8 contains formal acknowledgments to IMD and ECMWF/Copernicus, with bracketed funding placeholders.
* **Action Required by Authors:**
  - Insert explicit grant numbers and funding agencies (e.g., Ministry of Earth Sciences [MoES], Department of Science and Technology [DST], Council of Scientific and Industrial Research [CSIR], or University Research Grants) that supported the authors during this study.
  - Acknowledge High-Performance Computing (HPC) facilities or computational clusters if utilized.

---

## 5. Supplementary Information (SI) Packaging Strategy

* **Current Status:**  
  The main manuscript contains 9 primary figures and 8 comprehensive tables embedded directly in the main body.
* **Author Decision:**
  - Check the target journal's display item limits (many journals allow 8 to 10 total figures/tables in the main text).
  - *If a display item limit is enforced:*
    - Move Table 2 (Feature Matrix Architecture, 63 primary predictors) and Table 5 (Feature Ablation Matrix) to a separate `Supplementary_Information.docx` document.
    - Retain Figures 1–9 and Tables 1, 3, 4, 6, 7A, 7B, and 8 in the main manuscript text.

---

## 6. Declaration of Competing Interests

* **Confirmation Needed:**
  - Verify that all co-authors endorse the standard ethical declaration:
    > *"The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."*

---
*End of Unresolved Author Decisions and Pre-Submission Roadmap.*

