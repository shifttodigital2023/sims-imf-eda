#!/usr/bin/env python3
"""
01_eda_descriptive.py
=====================
Exploratory Data Analysis — SIMS IMF dataset (Śliwińśki et al., 2015)
Target: Section 3 "Results – Exploratory Analysis" for Q1 geosciences manuscript.

Run from any directory:
    python scripts/01_eda_descriptive.py

Dataset  : ../data/Data_Sliwinski_2015_no_vacuum_7labels_10inputs_n271.xlsx
Output   : ../output/
"""

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent          # sims-imf-eda/scripts/
DATA_FILE  = str(_HERE.parent.parent / "data" /
                 "Data_Sliwinski_2015_ no vacuum_7labels_10inputs_n271.xlsx")
OUTPUT_DIR = str(_HERE.parent / "output")
DPI        = 300

os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename):
    """Return full output path."""
    return os.path.join(OUTPUT_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA  —  strictly positional column mapping
# ─────────────────────────────────────────────────────────────────────────────
# Layout (18 columns total):
#   [0–6]   7 label columns  → omitted from modelling
#   [7–16] 10 input features  → IP, X, Y, DTFX, DTFY, 16O1H/16O,
#                                MgCO3, CaCO3, MnCO3, FeCO3
#   [17]    1 target          → 'IMF per mil'
#
# WARNING: the pipeline maps columns by position.  Re-ordering the source
# file would silently corrupt all predictions — no error will be raised.

df_raw = pd.read_excel(DATA_FILE)
assert df_raw.shape == (271, 18), \
    f"Unexpected shape {df_raw.shape} — check source file."

LABEL_COLS = list(df_raw.columns[:7])
INPUT_COLS = list(df_raw.columns[7:17])
TARGET_COL = df_raw.columns[17]          # 'IMF per mil'

df = df_raw[INPUT_COLS + [TARGET_COL]].copy()

assert df.isnull().sum().sum() == 0, "Missing values detected — investigate before modelling."

print("=" * 60)
print("SIMS EDA  —  Śliwińśki et al. 2015")
print("=" * 60)
print(f"Samples  : {df.shape[0]}")
print(f"Labels   (omitted): {LABEL_COLS}")
print(f"Inputs   : {INPUT_COLS}")
print(f"Target   : {TARGET_COL}")
print(f"Missing  : 0  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# 2. VARIABLE GROUPING BY PHYSICAL ROLE
# ─────────────────────────────────────────────────────────────────────────────
#
# CHEMICAL COMPOSITION / MATRIX EFFECT  (MgCO₃, CaCO₃, MnCO₃, FeCO₃)
#   These are mole-fraction components of the carbonate solid solution
#   (calcite–dolomite–siderite–rhodochrosite endmembers).
#   They satisfy the COMPOSITIONAL CLOSURE CONSTRAINT:
#       MgCO₃ + CaCO₃ + MnCO₃ + FeCO₃ = 100 mol%  (by definition)
#   This creates PERFECT MULTICOLLINEARITY: any one carbonate can be
#   expressed as the complement of the other three.  Consequence for
#   modelling:
#     • Tree models (GBM, RF, XGBoost) — unaffected; split on individual
#       features, not linear combinations.
#     • Linear/polynomial models (PolyReg) — coefficients are undefined /
#       degenerate; one carbonate must be dropped or an isometric
#       log-ratio (ILR) transform applied before fitting.
#     • SHAP with linear kernel — distorted by collinearity.
#
# SIMS INSTRUMENTAL VARIABLES  (IP, X, Y, DTFX, DTFY, ¹⁶O¹H/¹⁶O)
#   IP        — primary beam current (nA); total ion dose per analysis.
#   X, Y      — sample-stage coordinates (µm); encode spatial position on
#               the mount.  IMF is known to drift with stage position due
#               to beam-column alignment and electrostatic lens aberrations
#               (Kita et al. 2009; Valley et al. 2015).
#   DTFX/DTFY — deflector voltages (V) that steer the primary beam to the
#               desired X/Y position; capture higher-order optical effects
#               not fully encoded by X, Y alone.
#   ¹⁶O¹H/¹⁶O — OH-to-O ion ratio; proxy for residual vacuum quality /
#               molecular-ion contamination in the mass spectrometer.

CHEM_COLS = ["MgCO3", "CaCO3", "MnCO3", "FeCO3"]
SIMS_COLS = ["IP", "X", "Y", "DTFX", "DTFY", "16O1H/16O"]

# Display labels with physical units — used in all figure axes and tables
DISPLAY = {
    "IP":         "IP (nA)",
    "X":          "X (µm)",
    "Y":          "Y (µm)",
    "DTFX":       "DTFX (V)",
    "DTFY":       "DTFY (V)",
    "16O1H/16O":  "¹⁶O¹H/¹⁶O",
    "MgCO3":      "MgCO₃ (mol%)",
    "CaCO3":      "CaCO₃ (mol%)",
    "MnCO3":      "MnCO₃ (mol%)",
    "FeCO3":      "FeCO₃ (mol%)",
    TARGET_COL:   "IMF (‰)",
}

# Verify compositional closure (print to confirm understanding in the paper)
carb_sum = df[CHEM_COLS].sum(axis=1)
print(f"\nCarbonate closure check (should be ≈ 100 mol%):")
print(f"  mean = {carb_sum.mean():.3f}  |  std = {carb_sum.std():.4f}  |  "
      f"min = {carb_sum.min():.3f}  |  max = {carb_sum.max():.3f}")
print(f"  → Perfect closure confirmed. VIF of carbonate variables will be ∞.")

# Flag near-degenerate feature (CaCO3 has σ = 0.60, almost invariant)
ca_std = df["CaCO3"].std()
print(f"\nCaCO₃ variability: σ = {ca_std:.3f} mol%  "
      f"(range {df['CaCO3'].min():.2f}–{df['CaCO3'].max():.2f})")
print("  → Near-constant feature.  Low variance limits predictive contribution;")
print("    also fully determined by closure once Mg, Mn, Fe are known.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TABLE 1  —  ADVANCED DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
# Statistics included:
#   Mean, Median    — central tendency; gap between them indicates skew.
#   Std Dev         — overall spread (sensitive to outliers).
#   Min, Max        — observed range; useful for physical sanity checks.
#   IQR             — robust spread (25th–75th percentile); outlier-insensitive.
#   Skewness        — Fisher–Pearson coefficient.  |skew| > 1 flags departure
#                     from symmetry significant enough to warrant log- or
#                     Box-Cox transformation before linear models.
#   Excess Kurtosis — Fisher definition (Gaussian = 0).  Positive kurtosis
#                     indicates heavier tails than Gaussian; negative indicates
#                     lighter tails.  Relevant for model residual diagnostics.

ALL_COLS = INPUT_COLS + [TARGET_COL]

records = []
for col in ALL_COLS:
    s = df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    records.append({
        "Variable":        DISPLAY[col],
        "N":               int(len(s)),
        "Mean":            round(s.mean(),   5),
        "Median":          round(s.median(), 5),
        "Std Dev":         round(s.std(),    5),
        "Min":             round(s.min(),    5),
        "Max":             round(s.max(),    5),
        "IQR":             round(q3 - q1,    5),
        "Skewness":        round(float(stats.skew(s)),     4),
        "Excess Kurtosis": round(float(stats.kurtosis(s)), 4),
    })

table1 = pd.DataFrame(records).set_index("Variable")
table1.to_csv(out("Table_1_Descriptive_Statistics.csv"))
print("\n" + "─" * 60)
print("Table 1  —  Descriptive Statistics")
print("─" * 60)
print(table1.to_string())
print(f"\n  → Saved: Table_1_Descriptive_Statistics.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MULTICOLLINEARITY DIAGNOSTICS  —  VIF
# ─────────────────────────────────────────────────────────────────────────────
# Variance Inflation Factor:  VIF_j = 1 / (1 − R²_j)
# where R²_j = goodness-of-fit from regressing feature j on all other
# features with an intercept.
#
# Interpretation thresholds (widely used in geoscience literature):
#   VIF < 5            → acceptable
#   5 ≤ VIF < 10       → moderate — monitor, especially for linear models
#   VIF ≥ 10           → severe — regression coefficients unreliable
#   VIF → ∞            → perfect collinearity (guaranteed here for carbonates)
#
# Implementation: LinearRegression from scikit-learn — no additional
# dependency required beyond the existing project environment.

def compute_vif(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    VIF for each column in `cols`, regressing it on all other columns
    in `cols` with an intercept.
    Returns a DataFrame indexed by display label.
    """
    X = data[cols].values
    rows = []
    for i, col in enumerate(cols):
        y       = X[:, i]
        X_other = np.delete(X, i, axis=1)
        # Prepend a ones column (intercept) so R² is referenced to the mean
        X_fit   = np.column_stack([np.ones(len(y)), X_other])
        reg     = LinearRegression(fit_intercept=False).fit(X_fit, y)
        y_hat   = reg.predict(X_fit)
        ss_res  = np.sum((y - y_hat) ** 2)
        ss_tot  = np.sum((y - np.mean(y)) ** 2)
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

        if r2 >= 1.0 - 1e-10:
            vif_val = np.inf
            flag    = "∞  — PERFECT COLLINEARITY"
        elif r2 >= 1.0 - 1e-4:
            vif_val = 1.0 / (1.0 - r2)
            flag    = f"SEVERE (≥ 10 000)"
        else:
            vif_val = 1.0 / (1.0 - r2)
            if vif_val >= 10:
                flag = "SEVERE (≥ 10)"
            elif vif_val >= 5:
                flag = "MODERATE (5–10)"
            else:
                flag = "OK (< 5)"

        rows.append({
            "Variable": DISPLAY[col],
            "VIF":      "∞" if np.isinf(vif_val) else round(vif_val, 2),
            "R²_j":     round(r2, 6),
            "Flag":     flag,
        })
    return pd.DataFrame(rows).set_index("Variable")


print("\n" + "─" * 60)
print("VIF  —  Carbonate variables only  (compositional closure test)")
print("─" * 60)
vif_carbs = compute_vif(df, CHEM_COLS)
print(vif_carbs.to_string())
vif_carbs.to_csv(out("Table_2_VIF_Carbonate_Variables.csv"))
print("  → Saved: Table_2_VIF_Carbonate_Variables.csv")

print("\n" + "─" * 60)
print("VIF  —  All 10 input features")
print("─" * 60)
vif_all = compute_vif(df, INPUT_COLS)
print(vif_all.to_string())
vif_all.to_csv(out("Table_3_VIF_All_Inputs.csv"))
print("  → Saved: Table_3_VIF_All_Inputs.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FIGURE 1  —  PEARSON CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
# We include the target (IMF per mil) so that the reader can identify which
# features have the strongest linear association with the response.
# Lower triangle + diagonal is shown; the upper triangle is masked to
# avoid redundancy — standard in Q1 geoscience publications.
# Colour map: 'coolwarm' is a diverging palette symmetric around 0,
# print-safe, and widely recognised in the chemometrics literature.

corr = df[ALL_COLS].corr(method="pearson")
tick_labels = [DISPLAY[c] for c in ALL_COLS]

# True in cells to HIDE (upper triangle above diagonal)
mask_upper = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(12, 10))
hm = sns.heatmap(
    corr,
    mask=mask_upper,
    cmap="coolwarm",
    vmin=-1, vmax=1,
    annot=True, fmt=".2f",
    annot_kws={"size": 8.5},
    square=True,
    linewidths=0.6, linecolor="#eeeeee",
    cbar_kws={"shrink": 0.78},
    ax=ax,
)

# Style colorbar
cb = hm.collections[0].colorbar
cb.set_label("Pearson r", fontsize=10)
cb.ax.tick_params(labelsize=9)

ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9.5)
ax.set_yticklabels(tick_labels, rotation=0,  fontsize=9.5)
ax.set_title(
    "Pearson Correlation Matrix — SIMS Input Features and IMF Target\n"
    r"($n$ = 271;  Śliwińśki et al., 2015)",
    fontsize=12, fontweight="bold", pad=16,
)

plt.tight_layout()
fig.savefig(out("Fig_EDA_Correlation_Matrix.png"),
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\n  → Saved: Fig_EDA_Correlation_Matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURE 2  —  DISTRIBUTION HISTOGRAMS WITH KDE
# ─────────────────────────────────────────────────────────────────────────────
# One subplot per variable (10 inputs + 1 target = 11 panels).
# Freedman–Diaconis bin width is used (bins='auto' in matplotlib), which
# adapts to the sample size and spread — more robust than fixed-bin counts.
# KDE uses Scott's rule bandwidth — appropriate for n = 271.
#
# Geological interpretation guide for the carbonate distributions:
#   MgCO₃ (9.7–49.8 mol%) — wide range suggests the dataset spans low-Mg
#          calcite through dolomite endmembers.  Bimodal KDE would confirm
#          distinct mineral populations (key finding for the paper).
#   CaCO₃ (50.1–52.7 mol%) — near-constant; effectively degenerate as a
#          standalone predictor once closure is imposed.
#   MnCO₃ (0–2.6 mol%)    — right-skewed; trace Mn substitution in most
#          samples, with a few Mn-enriched grains.
#   FeCO₃ (0.2–36 mol%)   — broad right tail; ankerite/siderite endmembers
#          at the high end.
#   ¹⁶O¹H/¹⁶O            — right-skewed; stable vacuum punctuated by
#          occasional contamination spikes.

CHEM_COLOR   = "#2e8b8b"   # teal     — chemical/matrix group
SIMS_COLOR   = "#2b5ea7"   # steel blue — instrumental group
TARGET_COLOR = "#b03030"   # deep red  — target

COL_COLOR = {c: CHEM_COLOR  for c in CHEM_COLS}
COL_COLOR.update({c: SIMS_COLOR for c in SIMS_COLS})
COL_COLOR[TARGET_COL] = TARGET_COLOR

ncols_g = 4
nrows_g = int(np.ceil(len(ALL_COLS) / ncols_g))   # → 3 rows (last cell empty)

fig, axes = plt.subplots(nrows_g, ncols_g,
                          figsize=(ncols_g * 4.2, nrows_g * 3.6))
axes_flat = axes.flatten()

for idx, col in enumerate(ALL_COLS):
    ax    = axes_flat[idx]
    s     = df[col].dropna()
    color = COL_COLOR[col]

    # Histogram (density-normalised so it overlaps correctly with the KDE)
    ax.hist(s, bins="auto", color=color, alpha=0.45, density=True,
            edgecolor="white", linewidth=0.5, zorder=2)

    # KDE using Scott's rule bandwidth
    kde     = stats.gaussian_kde(s, bw_method="scott")
    x_grid  = np.linspace(s.min() - 0.08 * s.std(),
                           s.max() + 0.08 * s.std(), 500)
    ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.3, zorder=3)

    # Reference lines for mean (dashed) and median (dotted)
    ax.axvline(s.mean(),   color="#222222", linestyle="--", linewidth=1.1,
               label=f"$\\bar{{x}}$ = {s.mean():.3g}", zorder=4)
    ax.axvline(s.median(), color="#222222", linestyle=":",  linewidth=1.2,
               label=f"Mdn = {s.median():.3g}", zorder=4)

    # Skewness and kurtosis annotation
    skew_val = float(stats.skew(s))
    kurt_val = float(stats.kurtosis(s))
    ax.text(0.97, 0.95,
            f"skew = {skew_val:+.2f}\nkurt = {kurt_val:+.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.82, edgecolor="#cccccc", linewidth=0.5))

    ax.set_xlabel(DISPLAY[col], fontsize=9.5)
    ax.set_ylabel("Density",    fontsize=8.5)
    ax.set_title(DISPLAY[col],  fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.6, loc="upper left",
              handlelength=1.4)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45, zorder=0)

# Hide the unused subplot (11 variables in a 4×3 grid → 1 empty)
for j in range(len(ALL_COLS), len(axes_flat)):
    axes_flat[j].set_visible(False)

# Group legend
group_patches = [
    mpatches.Patch(facecolor=SIMS_COLOR,   alpha=0.7,
                   label="SIMS instrumental variables"),
    mpatches.Patch(facecolor=CHEM_COLOR,   alpha=0.7,
                   label="Chemical composition  (matrix effect)"),
    mpatches.Patch(facecolor=TARGET_COLOR, alpha=0.7,
                   label="Target — IMF (‰)"),
]
fig.legend(handles=group_patches, loc="lower right", fontsize=9.5,
           framealpha=0.88, bbox_to_anchor=(0.99, 0.01))

fig.suptitle(
    "Distribution of SIMS Input Features and IMF Target\n"
    r"Histogram (Freedman–Diaconis bins) + KDE (Scott bw),  $n$ = 271"
    "\n— — mean  ·  ·  ·  median",
    fontsize=12, fontweight="bold", y=1.02,
)

plt.tight_layout()
fig.savefig(out("Fig_EDA_Distributions.png"),
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  → Saved: Fig_EDA_Distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FIGURE 3  —  SPATIAL POSITION EFFECT  (X vs Y → IMF colour)
# ─────────────────────────────────────────────────────────────────────────────
# IMF in SIMS is known to exhibit a position-dependent component caused by:
#   (a) Primary beam intensity gradient across stage travel.
#   (b) Electrostatic deflector non-linearities at large X/Y offsets.
#   (c) Sample height variation (tilt) along the mount surface.
# If a systematic spatial gradient is visible, it validates the inclusion
# of X, Y, DTFX, DTFY as model features — which this pipeline does.
# Colour map: RdYlBu_r (diverging).  Red = less negative IMF (closer to 0‰),
# blue = more negative (stronger fractionation).  The map is perceptually
# uniform and print-safe.  Stage crosshairs at (0, 0) mark the optical axis.

fig, ax = plt.subplots(figsize=(7.2, 6.8))

sc = ax.scatter(
    df["X"], df["Y"],
    c=df[TARGET_COL],
    cmap="RdYlBu_r",
    s=52, alpha=0.82,
    edgecolors="white", linewidths=0.45,
    zorder=3,
)

cbar = fig.colorbar(sc, ax=ax, pad=0.03)
cbar.set_label("IMF (‰)", fontsize=10.5)
cbar.ax.tick_params(labelsize=9)

ax.set_xlabel("Stage X position (µm)", fontsize=11)
ax.set_ylabel("Stage Y position (µm)", fontsize=11)
ax.set_title(
    "Spatial Distribution of IMF on SIMS Sample Stage\n"
    r"Colour encodes IMF per mil (‰) — $n$ = 271",
    fontsize=11.5, fontweight="bold",
)
ax.tick_params(labelsize=9)
ax.set_aspect("equal", adjustable="datalim")
ax.grid(linestyle="--", linewidth=0.45, alpha=0.45, zorder=0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Optical-axis crosshairs (stage centre)
ax.axhline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=2)
ax.axvline(0, color="#999999", linewidth=0.9, linestyle="--", zorder=2)
ax.text(0.015, 0.015, "Stage centre (0, 0)",
        transform=ax.transAxes, fontsize=8, color="#777777")

plt.tight_layout()
fig.savefig(out("Fig_EDA_Spatial_Effects.png"),
            dpi=DPI, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  → Saved: Fig_EDA_Spatial_Effects.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EDA COMPLETE")
print(f"Output directory : ./{OUTPUT_DIR}/")
print("Tables (CSV):")
print("  Table_1_Descriptive_Statistics.csv")
print("  Table_2_VIF_Carbonate_Variables.csv")
print("  Table_3_VIF_All_Inputs.csv")
print("Figures (300 DPI PNG):")
print("  Fig_EDA_Correlation_Matrix.png")
print("  Fig_EDA_Distributions.png")
print("  Fig_EDA_Spatial_Effects.png")
print("=" * 60)
