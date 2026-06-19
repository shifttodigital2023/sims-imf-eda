# SIMS IMF — Exploratory Data Analysis

Statistical audit and EDA contribution to a Q1 geosciences manuscript on
Instrumental Mass Fractionation (IMF) prediction by SIMS using machine learning.

**Task:** Statistical review
**Pipeline authors:** Geology team (not included in this repository).

> [!note] 
> Place the `.xlsx` file in `../data/` before running the scripts.

---

## Scientific Context

Secondary Ion Mass Spectrometry (SIMS) introduces a systematic bias known as
**Instrumental Mass Fractionation** (IMF, ‰) that shifts the measured δ¹⁸O
value relative to the true value:

> δ¹⁸O_measured = δ¹⁸O_true + IMF (‰)

Classical correction uses polynomial surfaces over mineral composition. This
work audits a multi-model ML pipeline (GBM, XGBoost, SVR, RF, PolyReg) as
a more flexible alternative and provides the statistical evidence base for
preferring non-linear models.

**Dataset:** Śliwińśki et al. (2015) — *not distributed here, cite the original
publication.

---

## Dataset Structure (n = 271, 18 columns)

| Block | Columns | Role |
|---|---|---|
| Labels (7) | File, WiscSIMS, δ¹⁸O true, ¹⁸O/¹⁶O true, δ¹⁸O raw, ¹⁸O/¹⁶O raw, IMF | Omitted from training |
| Inputs (10) | IP, X, Y, DTFX, DTFY, ¹⁶O¹H/¹⁶O, MgCO₃, CaCO₃, MnCO₃, FeCO₃ | Features |
| Target (1) | IMF per mil | Response variable |



---

## Repository Structure

```
sims-imf-eda/
├── scripts/
│   ├── 01_eda_descriptive.py          # Descriptive stats, VIF, Pearson matrix,
│   │                                  # histograms, spatial scatter
│   ├── 02_eda_advanced_validation.py  # Four critical findings against linear models
│   └── 03_generate_notebook.py        # Regenerates the Jupyter notebook from source
├── notebooks/
│   └── EDA_SIMS_IMF_Notebook.ipynb    # Interactive version of the EDA
├── output/
│   ├── Fig_EDA_*.png                  # Figures (300 DPI, publication-ready)
│   └── Table_*.csv                    # Statistical tables
└── review/
    └── pipeline_review_notes.md       # Audit notes on the ML pipeline
```

---

## Four Critical Findings (against linear models)

| # | Finding | Key result |
|---|---|---|
| H1 | Chemical space sparsity | Dolomite end-member: only n=9 (3.3%); no calcite Mg-poor samples |
| H2 | Hardware non-normality | IP skew = −2.08 (SW p < 10⁻¹⁵); ¹⁶O¹H/¹⁶O Box-Cox λ = 0.19 |
| H3 | Spatial non-stationarity | X ↔ DTFX: ρ = −0.31 (centre) vs +0.06 (periphery), Δρ = +0.37 |
| H4 | High dimensionality | Only 57.3% variance in 2D PCA; R² gain +9.6 pp with quadratic PC1 |

---

## Reproducibility

```bash
pip install -r requirements.txt

# Place dataset in ../data/ then run:
python scripts/01_eda_descriptive.py
python scripts/02_eda_advanced_validation.py

# Regenerate notebook (optional):
python scripts/03_generate_notebook.py
```

All outputs land in `output/` automatically.
