# Pipeline Review Notes — Multi_pipeline_v114

Audit of the geology team's ML pipeline (`Multi_pipeline_v114.py/.ipynb`).
Reference version received: 2026-06-18.

---

## Scope

- File reviewed: `Multi_pipeline_v114.py` (Cell 3 — Training)
- Models: GBM, XGBoost, SVR, Random Forest, Polynomial Ridge Regression
- Dataset: n=271, 10 inputs, 1 target (IMF per mil)

---

## Issues Identified

*(to be completed during pipeline review phase)*

### P1 — [Title]
- **Severity:** [Critical / Major / Minor]
- **Location:** Line X
- **Description:**
- **Statistical impact:**
- **Proposed correction:**

---

## EDA-to-Pipeline Linkage

The following EDA findings have direct implications for the pipeline configuration:

| EDA Finding | Pipeline implication | Status |
|---|---|---|
| H2: IP strongly left-skewed (skew=−2.08) | StandardScaler suboptimal for IP; consider RobustScaler | Pending |
| H2: ¹⁶O¹H/¹⁶O Box-Cox λ=0.19 (≈log) | Log transform before scaling recommended | Pending |
| H3: X↔DTFX non-stationarity | Spatial features must not be linearly combined with positional coords | Pending |
| H1: CaCO₃ quasi-constant (σ=0.60) | Near-degenerate feature — verify VIF guard is active for PolyReg | Pending |
| H4: 57.3% variance in 2D PCA | Confirms need for non-linear models; validates GBM/XGB/ANN choice | Confirmed |

---

## Corrections Applied

*(log of changes proposed or implemented)*

| Date | Issue | Change | File |
|---|---|---|---|
| — | — | — | — |
