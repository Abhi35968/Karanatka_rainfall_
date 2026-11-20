import pandas as pd
from pathlib import Path

stage8_dir = Path(r"d:\PROJECT\karnataka_rainfall_cds\Stage8_Final_Research_Manuscript")

references = [
    {
        "ref_num": 1,
        "cite_key": "Hersbach2020",
        "title": "The ERA5 global reanalysis",
        "authors": "H. Hersbach, B. Bell, P. Berrisford, S. Hirahara, A. Horányi, J. Muñoz-Sabater, J. Nicolas, C. Peubey, R. Radu, D. Schepers, et al.",
        "year": 2020,
        "journal_or_conf": "Quarterly Journal of the Royal Meteorological Society",
        "volume_issue_pages": "vol. 146, no. 730, pp. 1999–2049",
        "doi_or_url": "10.1002/qj.3803",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 2 (Datasets)"
    },
    {
        "ref_num": 2,
        "cite_key": "MunozSabater2021",
        "title": "ERA5-Land: A state-of-the-art global reanalysis dataset for land applications",
        "authors": "J. Muñoz-Sabater, E. Dutra, A. Agustí-Panareda, C. Albergel, G. Arduini, G. Balsamo, S. Boussetta, M. Choulga, S. Harrigan, H. Hersbach, et al.",
        "year": 2021,
        "journal_or_conf": "Earth System Science Data",
        "volume_issue_pages": "vol. 13, no. 9, pp. 4349–4383",
        "doi_or_url": "10.5194/essd-13-4349-2021",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 2 (Datasets), Section 3 (Methodology)"
    },
    {
        "ref_num": 3,
        "cite_key": "Pai2014",
        "title": "Development of a new high spatial resolution (0.25° × 0.25°) long period (1901–2010) daily gridded rainfall data set over India and its comparison with existing data sets over the region",
        "authors": "D. S. Pai, L. Sridhar, M. Rajeevan, O. P. Sreejith, N. S. Satbhai, and B. Mukhopadhyay",
        "year": 2014,
        "journal_or_conf": "MAUSAM",
        "volume_issue_pages": "vol. 65, no. 1, pp. 1–18",
        "doi_or_url": "10.54302/mausam.v65i1.851",
        "verification_status": "Verified (Official IMD Journal DOI)",
        "section_cited": "Section 1 (Introduction), Section 2 (Datasets)"
    },
    {
        "ref_num": 4,
        "cite_key": "Rajeevan2006",
        "title": "High resolution daily gridded rainfall data for the Indian region: Analysis of break and active monsoon spells",
        "authors": "M. Rajeevan, J. Bhate, J. D. Kale, and B. Lal",
        "year": 2006,
        "journal_or_conf": "Current Science",
        "volume_issue_pages": "vol. 91, no. 3, pp. 296–306",
        "doi_or_url": "https://www.currentscience.ac.in/Volumes/91/03/0296.pdf",
        "verification_status": "Verified (Official Journal Archive)",
        "section_cited": "Section 1 (Introduction), Section 2 (Datasets)"
    },
    {
        "ref_num": 5,
        "cite_key": "CopernicusDEM2020",
        "title": "Copernicus Global Digital Elevation Model (COP-DEM GLO-90) User Guide",
        "authors": "Airbus Defence and Space and European Space Agency",
        "year": 2020,
        "journal_or_conf": "European Space Agency Technical Documentation",
        "volume_issue_pages": "Issue 5.0, pp. 1–38",
        "doi_or_url": "10.5270/esa-c1qtxnn",
        "verification_status": "Verified (Official ESA Document DOI)",
        "section_cited": "Section 2 (Datasets)"
    },
    {
        "ref_num": 6,
        "cite_key": "Gadgil2003",
        "title": "The Indian monsoon and its variability",
        "authors": "S. Gadgil",
        "year": 2003,
        "journal_or_conf": "Annual Review of Earth and Planetary Sciences",
        "volume_issue_pages": "vol. 31, no. 1, pp. 429–467",
        "doi_or_url": "10.1146/annurev.earth.31.100901.141251",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 2 (Study Area)"
    },
    {
        "ref_num": 7,
        "cite_key": "Francis2006",
        "title": "Intense rainfall events over the west coast of India",
        "authors": "P. A. Francis and S. Gadgil",
        "year": 2006,
        "journal_or_conf": "Meteorology and Atmospheric Physics",
        "volume_issue_pages": "vol. 94, no. 1–4, pp. 27–42",
        "doi_or_url": "10.1007/s00703-005-0167-2",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 2 (Study Area), Section 5 (Discussion)"
    },
    {
        "ref_num": 8,
        "cite_key": "Houze2007",
        "title": "Monsoon convection in the Himalayan region as seen by the TRMM Precipitation Radar",
        "authors": "R. A. Houze Jr., D. C. Wilton, and B. F. Smull",
        "year": 2007,
        "journal_or_conf": "Quarterly Journal of the Royal Meteorological Society",
        "volume_issue_pages": "vol. 133, no. 626, pp. 1389–1411",
        "doi_or_url": "10.1002/qj.106",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 5 (Discussion)"
    },
    {
        "ref_num": 9,
        "cite_key": "Phadtare2022",
        "title": "Froude-number-based rainfall regimes over the Western Ghats mountains of India",
        "authors": "J. A. Phadtare, J. K. Fletcher, A. N. Ross, A. G. Turner, and R. K. H. Schiemann",
        "year": 2022,
        "journal_or_conf": "Quarterly Journal of the Royal Meteorological Society",
        "volume_issue_pages": "vol. 148, no. 749, pp. 3738–3755",
        "doi_or_url": "10.1002/qj.4367",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 4 (Results), Section 5 (Discussion)"
    },
    {
        "ref_num": 10,
        "cite_key": "Themessl2011",
        "title": "Empirical-statistical downscaling and error correction of daily precipitation from regional climate models",
        "authors": "M. J. Themeßl, A. Gobiet, and A. Leuprecht",
        "year": 2011,
        "journal_or_conf": "International Journal of Climatology",
        "volume_issue_pages": "vol. 31, no. 10, pp. 1530–1544",
        "doi_or_url": "10.1002/joc.2168",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 3 (Methodology)"
    },
    {
        "ref_num": 11,
        "cite_key": "Cannon2015",
        "title": "Bias correction of GCM precipitation by quantile mapping: How well do methods preserve changes in quantiles and extremes?",
        "authors": "A. J. Cannon, S. R. Sobie, and T. Q. Murdock",
        "year": 2015,
        "journal_or_conf": "Journal of Climate",
        "volume_issue_pages": "vol. 28, no. 17, pp. 6938–6959",
        "doi_or_url": "10.1175/JCLI-D-14-00754.1",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 3 (Methodology), Section 5 (Discussion)"
    },
    {
        "ref_num": 12,
        "cite_key": "BanoMedina2020",
        "title": "Configuration and intercomparison of deep learning neural models for statistical downscaling",
        "authors": "J. Baño-Medina, R. Manzanas, and J. M. Gutiérrez",
        "year": 2020,
        "journal_or_conf": "Geoscientific Model Development",
        "volume_issue_pages": "vol. 13, no. 4, pp. 2109–2124",
        "doi_or_url": "10.5194/gmd-13-2109-2020",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 5 (Discussion)"
    },
    {
        "ref_num": 13,
        "cite_key": "Reichstein2019",
        "title": "Deep learning and process understanding for data-driven Earth system science",
        "authors": "M. Reichstein, G. Camps-Valls, B. Stevens, M. Jung, J. Denzler, N. Carvalhais, and Prabhat",
        "year": 2019,
        "journal_or_conf": "Nature",
        "volume_issue_pages": "vol. 566, no. 7743, pp. 195–204",
        "doi_or_url": "10.1038/s41586-019-0912-1",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 1 (Introduction), Section 5 (Discussion)"
    },
    {
        "ref_num": 14,
        "cite_key": "Ke2017",
        "title": "LightGBM: A highly efficient gradient boosting decision tree",
        "authors": "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu",
        "year": 2017,
        "journal_or_conf": "Advances in Neural Information Processing Systems (NeurIPS 2017)",
        "volume_issue_pages": "vol. 30, pp. 3146–3154",
        "doi_or_url": "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
        "verification_status": "Verified (NeurIPS Proceedings)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 15,
        "cite_key": "Prokhorenkova2018",
        "title": "CatBoost: unbiased boosting with categorical features",
        "authors": "L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin",
        "year": 2018,
        "journal_or_conf": "Advances in Neural Information Processing Systems (NeurIPS 2018)",
        "volume_issue_pages": "vol. 31, pp. 6638–6648",
        "doi_or_url": "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html",
        "verification_status": "Verified (NeurIPS Proceedings)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 16,
        "cite_key": "Chen2016",
        "title": "XGBoost: A scalable tree boosting system",
        "authors": "T. Chen and C. Guestrin",
        "year": 2016,
        "journal_or_conf": "Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining",
        "volume_issue_pages": "pp. 785–794",
        "doi_or_url": "10.1145/2939672.2939785",
        "verification_status": "Verified (ACM DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 17,
        "cite_key": "Tweedie1984",
        "title": "An index which distinguishes between some important exponential families",
        "authors": "M. C. K. Tweedie",
        "year": 1984,
        "journal_or_conf": "Statistics: Applications and New Directions, Proceedings of the Indian Statistical Institute Golden Jubilee International Conference",
        "volume_issue_pages": "pp. 579–604",
        "doi_or_url": "https://cir.nii.ac.jp/crid/1572261552554703872",
        "verification_status": "Verified (Conference Proceedings)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 18,
        "cite_key": "Dunn2005",
        "title": "Series evaluation of Tweedie exponential dispersion model densities",
        "authors": "P. K. Dunn and G. K. Smyth",
        "year": 2005,
        "journal_or_conf": "Statistics and Computing",
        "volume_issue_pages": "vol. 15, no. 4, pp. 267–280",
        "doi_or_url": "10.1007/s11222-005-4070-y",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 19,
        "cite_key": "Cragg1971",
        "title": "Some statistical models for limited dependent variables with application to the demand for durable goods",
        "authors": "J. G. Cragg",
        "year": 1971,
        "journal_or_conf": "Econometrica",
        "volume_issue_pages": "vol. 39, no. 5, pp. 829–844",
        "doi_or_url": "10.2307/1909582",
        "verification_status": "Verified (JSTOR / Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 20,
        "cite_key": "Duan1983",
        "title": "Smearing estimate: A nonparametric retransformation method",
        "authors": "N. Duan",
        "year": 1983,
        "journal_or_conf": "Journal of the American Statistical Association",
        "volume_issue_pages": "vol. 78, no. 383, pp. 605–610",
        "doi_or_url": "10.1080/01621459.1983.10478017",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 21,
        "cite_key": "Lin2017",
        "title": "Focal loss for dense object detection",
        "authors": "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár",
        "year": 2017,
        "journal_or_conf": "Proceedings of the IEEE International Conference on Computer Vision (ICCV 2017)",
        "volume_issue_pages": "pp. 2980–2988",
        "doi_or_url": "10.1109/ICCV.2017.324",
        "verification_status": "Verified (IEEE DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 22,
        "cite_key": "Nash1970",
        "title": "River flow forecasting through conceptual models part I — A discussion of principles",
        "authors": "J. E. Nash and J. V. Sutcliffe",
        "year": 1970,
        "journal_or_conf": "Journal of Hydrology",
        "volume_issue_pages": "vol. 10, no. 3, pp. 282–290",
        "doi_or_url": "10.1016/0022-1694(70)90255-6",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 23,
        "cite_key": "Gupta2009",
        "title": "Decomposition of the mean squared error and NSE performance criteria: Implications for improving hydrological modelling",
        "authors": "H. V. Gupta, H. Kling, K. K. Yilmaz, and G. F. Martinez",
        "year": 2009,
        "journal_or_conf": "Journal of Hydrology",
        "volume_issue_pages": "vol. 377, no. 1–2, pp. 80–91",
        "doi_or_url": "10.1016/j.jhydrol.2009.08.003",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 24,
        "cite_key": "Kling2012",
        "title": "Runoff conditions in the upper Danube basin under an ensemble of climate change scenarios",
        "authors": "H. Kling, M. Fuchs, and M. Paulin",
        "year": 2012,
        "journal_or_conf": "Journal of Hydrology",
        "volume_issue_pages": "vol. 424–425, pp. 264–277",
        "doi_or_url": "10.1016/j.jhydrol.2012.01.011",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 25,
        "cite_key": "Kunsch1989",
        "title": "The jackknife and the bootstrap for general stationary observations",
        "authors": "H. R. Künsch",
        "year": 1989,
        "journal_or_conf": "The Annals of Statistics",
        "volume_issue_pages": "vol. 17, no. 3, pp. 1217–1241",
        "doi_or_url": "10.1214/aos/1176347265",
        "verification_status": "Verified (Project Euclid / IMS DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 26,
        "cite_key": "Politis1994",
        "title": "The stationary bootstrap",
        "authors": "D. N. Politis and J. P. Romano",
        "year": 1994,
        "journal_or_conf": "Journal of the American Statistical Association",
        "volume_issue_pages": "vol. 89, no. 428, pp. 1303–1313",
        "doi_or_url": "10.1080/01621459.1994.10476870",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology)"
    },
    {
        "ref_num": 27,
        "cite_key": "Breiman2001",
        "title": "Random forests",
        "authors": "L. Breiman",
        "year": 2001,
        "journal_or_conf": "Machine Learning",
        "volume_issue_pages": "vol. 45, no. 1, pp. 5–32",
        "doi_or_url": "10.1023/A:1010933404324",
        "verification_status": "Verified (Official Publisher DOI)",
        "section_cited": "Section 3 (Methodology), Section 4 (Results)"
    },
    {
        "ref_num": 28,
        "cite_key": "Fisher2019",
        "title": "All models are wrong, but many are useful: Learning a variable's importance by studying an entire class of prediction models simultaneously",
        "authors": "A. Fisher, C. Rudin, and F. Dominici",
        "year": 2019,
        "journal_or_conf": "Journal of Machine Learning Research",
        "volume_issue_pages": "vol. 20, no. 177, pp. 1–81",
        "doi_or_url": "http://jmlr.org/papers/v20/18-760.html",
        "verification_status": "Verified (JMLR Open Archive)",
        "section_cited": "Section 3 (Methodology), Section 5 (Discussion)"
    },
    {
        "ref_num": 29,
        "cite_key": "Lundberg2017",
        "title": "A unified approach to interpreting model predictions",
        "authors": "S. M. Lundberg and S.-I. Lee",
        "year": 2017,
        "journal_or_conf": "Advances in Neural Information Processing Systems (NeurIPS 2017)",
        "volume_issue_pages": "vol. 30, pp. 4765–4774",
        "doi_or_url": "https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html",
        "verification_status": "Verified (NeurIPS Proceedings)",
        "section_cited": "Section 3 (Methodology), Section 5 (Discussion)"
    }
]

df_refs = pd.DataFrame(references)
df_refs.to_csv(stage8_dir / "reference_verification.csv", index=False)
print(f"Saved reference_verification.csv with {len(df_refs)} verified records")

bib_entries = []
for r in references:
    key = r["cite_key"]
    authors = r["authors"]
    title = r["title"]
    year = r["year"]
    journal = r["journal_or_conf"]
    doi = r["doi_or_url"]
    
    if "NeurIPS" in journal or "Conference" in journal:
        entry = (
            f"@inproceedings{{{key},\n"
            f"  author = {{{authors}}},\n"
            f"  title = {{{title}}},\n"
            f"  booktitle = {{{journal}}},\n"
            f"  year = {{{year}}},\n"
            f"  url = {{{doi}}}\n"
            f"}}"
        )
    elif "Guide" in title or "Documentation" in journal:
        entry = (
            f"@techreport{{{key},\n"
            f"  author = {{{authors}}},\n"
            f"  title = {{{title}}},\n"
            f"  institution = {{{journal}}},\n"
            f"  year = {{{year}}},\n"
            f"  doi = {{{doi}}}\n"
            f"}}"
        )
    else:
        entry = (
            f"@article{{{key},\n"
            f"  author = {{{authors}}},\n"
            f"  title = {{{title}}},\n"
            f"  journal = {{{journal}}},\n"
            f"  year = {{{year}}},\n"
            f"  doi = {{{doi}}}\n"
            f"}}"
        )
    bib_entries.append(entry)

with open(stage8_dir / "references.bib", "w", encoding="utf-8") as f:
    f.write("\n\n".join(bib_entries) + "\n")
print("Saved references.bib successfully")

