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

### P1 — XGB_PARAMS mutated in-place by adaptive guard
- **Severity:** Major
- **Location:** Lines 1967–2014 (inside XGB `_run` loop)
- **Description:** The adaptive overfitting guard modifies the global dict `XGB_PARAMS` (e.g., `XGB_PARAMS["max_depth"] = _depth_cap`) during each run. Because the dict is shared, mutations from run k persist into run k+1. Run 1 may start with `max_depth=2` instead of 3 if run 0 triggered the cap, even if run 1's train size would not require it.
- **Statistical impact:** The reported mean performance across 10 runs is a function of execution order, not only of data partitions. The hyperparameter configuration is not constant across runs, violating the independence assumption of the stability analysis.
- **Proposed correction:** Work on a copy of the dict at the start of each run:
  ```python
  for _run in range(N_RUNS):
      _xgb_p = XGB_PARAMS.copy()   # ← add this line
      # replace all XGB_PARAMS[...] = ... references with _xgb_p[...] = ...
      model_xgb = XGBRegressor(**_xgb_p)
  ```

### P2 — StandardScaler on skewed inputs (SVR + PolyReg)
- **Severity:** Minor / design choice
- **Location:** Lines 2380–2385 (SVR outer split); 2459–2463 (SVR inner CV); line 3092 (PolyReg Pipeline)
- **Description:** See EDA-to-Pipeline Linkage below and P3 for full analysis.
- **Proposed correction:** See P3.

### P3 — Missing variance-stabilising transforms before scaling
- **Severity:** Minor (SVR) / Moderate (PolyReg)
- **Location:** Lines 2379–2385 (SVR scaler block); line 3092 (PolyReg Pipeline definition)
- **Description:** Two inputs identified in the EDA have strongly non-normal distributions that distort distance-based and polynomial models:
  - **IP** (skew = −2.08, SW p < 10⁻¹⁵): heavy left tail; extreme values pull the mean and std, compressing the central mass after StandardScaler.
  - **¹⁶O¹H/¹⁶O** (Box-Cox λ = 0.187 ≈ log): right-skewed; log transform produces a near-symmetric distribution.
  Switching to RobustScaler (median + IQR) is sometimes proposed as a fix, but it does not change the shape of the distribution — skewness remains identical after centering and rescaling. The SVR RBF kernel and polynomial feature expansion both operate on the geometry of the feature space; an asymmetric distribution distorts inter-point distances regardless of the centring statistic used. Additionally, CaCO₃ (σ = 0.60, IQR ≈ 0.7–0.9 mol%) is near-degenerate: its very small IQR makes RobustScaler's denominator fragile at n = 271.
- **Statistical impact:**
  - **SVR**: gamma='scale' is computed as 1/(n_features × Var(X)); if X contains untransformed skewed variables, the effective bandwidth is miscalibrated for the majority of the data.
  - **PolyReg**: polynomial terms amplify skewness (x² of a left-skewed variable is even more asymmetric). Applying StandardScaler after polynomial expansion does not undo the shape distortion introduced by the raw inputs.
- **Proposed correction:**
  1. Add a preprocessing step before the scaler in each affected model's run loop:
     ```python
     import numpy as np

     # Indices (positional) of IP and 16OH/16O in X
     IP_COL   = 0   # verify against column order
     OH_COL   = 5   # verify against column order

     X_train_t = X_train_orig.copy().astype(float)
     X_test_t  = X_test_orig.copy().astype(float)

     # Log-transform (fit shift on train only to avoid leakage)
     for col in [IP_COL, OH_COL]:
         shift = max(0.0, -X_train_t[:, col].min()) + 1e-6
         X_train_t[:, col] = np.log1p(X_train_t[:, col] + shift)
         X_test_t[:, col]  = np.log1p(X_test_t[:, col]  + shift)

     # Then apply StandardScaler as usual
     scaler_X = StandardScaler()
     X_train_sc = scaler_X.fit_transform(X_train_t)
     X_test_sc  = scaler_X.transform(X_test_t)
     ```
  2. For PolyReg, apply the same transforms to the raw inputs **before** the sklearn Pipeline (i.e., transform → Pipeline(PolynomialFeatures → StandardScaler → Ridge)).
  3. The same per-fold logic must be applied inside the CV inner loop (fit shift and scaler on fold train only).
  - **Note:** The shift parameter must be estimated on the training fold only and applied to the validation/test fold — fitting it on the full dataset would constitute data leakage.

### P4 — Positional column mapping with no name validation (+ N_INPUTS misconfigured)
- **Severity:** Critical
- **Location:** Lines 705–706 (constants); lines 1644–1654 (data loading block)
- **Description:** The pipeline extracts features using purely positional slicing:
  ```python
  N_LABELS  = 7
  N_INPUTS  = 9          # ← placeholder; actual dataset has 10 inputs
  ...
  inputs_df = df.iloc[:, N_LABELS : N_LABELS + N_INPUTS]   # cols 7–15
  y_series  = df.iloc[:, -1]                                # last col
  ```
  Two distinct problems share this root cause:

  **4a — N_INPUTS is 9, not 10 (script is unrunnable on the actual dataset)**
  The dataset filename encodes the structure: `7labels_10inputs_n271.xlsx`. With N_INPUTS=9, `expected_cols = 7+9+1 = 17`, but the file has 18 columns. The validation at line 1644 raises `ValueError: Column count mismatch: 18 columns found, 17 expected`. The `.py` file has never had its STEP 1 constants updated from their default placeholder values (`N_INPUTS=9`, `FILE_PATH="data.xlsx"`). The geology team presumably ran from the `.ipynb` session with manual edits; the `.py` script as committed cannot be executed on the actual dataset.

  **4b — Column identity is never verified, only column count**
  After fixing N_INPUTS to 10, `inputs_df` becomes `df.iloc[:, 7:17]`. There is no check that column 7 is IP, column 8 is X, etc. If a user reorders columns in Excel (e.g., moves IP to position 9, or inserts a new metadata column), the slice silently picks up different variables. The model trains without error but on the wrong features. This failure mode is completely invisible: predictions are produced, metrics look plausible, and no exception is raised.

- **Statistical impact:**
  - 4a: pipeline halts on execution — no output produced.
  - 4b: if columns are reordered, all trained models silently learn the wrong input→output mapping. Feature importance, SHAP attributions, and predictions are all corrupted with no diagnostic signal.

- **Proposed correction:**
  ```python
  # STEP 1 — update placeholders to match actual dataset
  FILE_PATH = "Data_Sliwinski_2015_ no vacuum_7labels_10inputs_n271.xlsx"
  N_LABELS  = 7
  N_INPUTS  = 10   # ← was 9

  # After loading, add explicit column-name validation:
  EXPECTED_INPUT_COLS = [
      "IP", "X", "Y", "DTFX", "DTFY",
      "16OH/16O", "MgCO3", "CaCO3", "MnCO3", "FeCO3",
  ]   # verify exact header strings against the file
  actual_inputs = df.columns[N_LABELS : N_LABELS + N_INPUTS].tolist()
  if actual_inputs != EXPECTED_INPUT_COLS:
      raise ValueError(
          f"Input column mismatch.\n"
          f"  Expected: {EXPECTED_INPUT_COLS}\n"
          f"  Found:    {actual_inputs}\n"
          "Check column order in the source file."
      )
  ```
  This converts a silent data-corruption failure into an explicit error at load time.

---

## EDA-to-Pipeline Linkage

The following EDA findings have direct implications for the pipeline configuration:

| EDA Finding | Pipeline implication | Status |
|---|---|---|
| H2: IP strongly left-skewed (skew=−2.08) | Log transform before scaling; RobustScaler alone insufficient (see P3) | Pending |
| H2: ¹⁶O¹H/¹⁶O Box-Cox λ=0.19 (≈log) | Log transform before scaling; apply shift on train fold only (see P3) | Pending |
| H3: X↔DTFX non-stationarity | Spatial features must not be linearly combined with positional coords | Pending |
| H1: CaCO₃ quasi-constant (σ=0.60) | Near-degenerate feature — verify VIF guard is active for PolyReg | Pending |
| H4: 57.3% variance in 2D PCA | Confirms need for non-linear models; validates GBM/XGB/ANN choice | Confirmed |

---

## Corrections Applied

*(log of changes proposed or implemented)*

| Date | Issue | Change | File |
|---|---|---|---|
| — | — | — | — |
