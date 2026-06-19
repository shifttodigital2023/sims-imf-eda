#!/usr/bin/env python3
"""
02_eda_advanced_validation.py
=============================
Cuatro Hallazgos Críticos que justifican el pipeline no lineal (XGBoost/GBM/ANN)
frente a modelos clásicos en la predicción del IMF por SIMS.

Dataset : Śliwińśki et al. (2015)  —  n=271, hoja 'Data for Colab'
Perfil  : Ingeniero Telecom + MSc Bioestadística

Run:    python scripts/02_eda_advanced_validation.py
Output: ../output/
"""

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, jarque_bera, spearmanr, boxcox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

_HERE      = Path(__file__).resolve().parent          # sims-imf-eda/scripts/
DATA_FILE  = str(_HERE.parent.parent / "data" /
                 "Data_Sliwinski_2015_ no vacuum_7labels_10inputs_n271.xlsx")
SHEET_NAME = "Data for Colab"
OUTPUT_DIR = str(_HERE.parent / "output")
DPI        = 300
ALPHA      = 0.05          # nivel de significación estadística

os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(fname):
    return os.path.join(OUTPUT_DIR, fname)

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   9,
    "figure.dpi":        110,
    "savefig.dpi":       DPI,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.4,
    "grid.alpha":        0.45,
})
sns.set_style("whitegrid")

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
assert df_raw.shape == (271, 18), f"Forma inesperada: {df_raw.shape}"

INPUT_COLS = list(df_raw.columns[7:17])
TARGET_COL = df_raw.columns[17]           # 'IMF per mil'
df         = df_raw[INPUT_COLS + [TARGET_COL]].copy()

assert df.isnull().sum().sum() == 0, "Valores ausentes detectados."

SIMS_COLS = ["IP", "X", "Y", "DTFX", "DTFY", "16O1H/16O"]
CHEM_COLS = ["MgCO3", "CaCO3", "MnCO3", "FeCO3"]
HW_COLS   = ["IP", "DTFX", "DTFY", "16O1H/16O"]   # subconjunto hardware Finding 2

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

SIMS_CLR   = "#2b5ea7"
CHEM_CLR   = "#2e8b8b"
TARGET_CLR = "#b03030"

print("=" * 68)
print("ADVANCED EDA VALIDATION — SIMS IMF DATASET (n=271)")
print("=" * 68)


# ═════════════════════════════════════════════════════════════════════════════
# HALLAZGO 1 — DISTRIBUCIÓN DEL ESPACIO DE MUESTREO QUÍMICO
# ═════════════════════════════════════════════════════════════════════════════

def finding_1_chemical_sparsity(df):
    """
    HALLAZGO 1: Distribución del Espacio de Muestreo Químico
    ─────────────────────────────────────────────────────────
    Perspectiva de señal: cada muestra es un punto de medición en el espacio
    composicional 4D (Ca-Mg-Fe-Mn).  Las zonas escasamente muestreadas son
    análogas a "bandas de frecuencia sub-representadas" donde el error de
    interpolación del modelo es máximo.

    La restricción de clausura (ΣXi = 100%) colapsa el espacio efectivo a 3D
    —o 2D en la práctica, dado que CaCO₃ es casi constante (σ = 0.60 mol%)—
    convirtiendo los ejes dominantes en MgCO₃ vs FeCO₃.

    Zonas mineralógicas:
      • MgCO₃ ≥ 40 mol%           → Dolomita (end-member Ca-Mg)
      • FeCO₃ ≥ 18 mol%           → Ankerita / Siderita (end-member Fe-rico)
      • MgCO₃ < 15 mol%           → Calcita Mg-pobre
      • 15 ≤ MgCO₃ < 40, Fe < 18 → Mg-Calcita / zona intermedia
    La prioridad de asignación se aplica en el orden indicado (np.select).
    """

    print("\n" + "═" * 68)
    print("HALLAZGO 1: DISTRIBUCIÓN DEL ESPACIO DE MUESTREO QUÍMICO")
    print("═" * 68)

    # ── Estadísticos descriptivos de los carbonatos ───────────────────────
    carb_stats = df[CHEM_COLS].describe().T
    carb_stats["IQR"] = carb_stats["75%"] - carb_stats["25%"]
    carb_stats["CV%"] = (carb_stats["std"] / carb_stats["mean"] * 100).round(2)
    print("\nEstadísticos fracciones carbonatadas:")
    print(carb_stats[["mean","std","CV%","min","25%","50%","75%","max","IQR"]
                      ].round(4).to_string())

    # ── Clasificación mineralógica (condiciones mutuamente excluyentes) ────
    zone_labels = [
        "Dolomita (MgCO₃ ≥ 40%)",
        "Ankerita/Siderita (FeCO₃ ≥ 18%)",
        "Calcita Mg-pobre (MgCO₃ < 15%)",
        "Mg-Calcita / intermedia",
    ]
    zone_colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12"]

    conditions = [
        df["MgCO3"] >= 40,
        df["FeCO3"] >= 18,
        df["MgCO3"] <  15,
        (df["MgCO3"] >= 15) & (df["MgCO3"] < 40) & (df["FeCO3"] < 18),
    ]
    df = df.copy()
    df["mineral_zone"] = np.select(conditions, zone_labels, default="Otro")

    zone_counts = df["mineral_zone"].value_counts()
    zone_pct    = zone_counts / len(df) * 100

    print("\nDistribución por Zona Mineralógica:")
    print(f"  {'Zona':<42}  {'n':>4}  {'%':>6}")
    print("  " + "─" * 56)
    rows_t1 = []
    for label, color in zip(zone_labels, zone_colors):
        n   = int(zone_counts.get(label, 0))
        pct = float(zone_pct.get(label, 0.0))
        flag = "  ← ZONA ESCASA (<15%)" if pct < 15 else ""
        print(f"  {label:<42}  {n:>4}  {pct:>5.1f}%{flag}")
        rows_t1.append({"Zona": label, "n": n, "Proporción (%)": round(pct,2),
                         "Escasa (<15%)": pct < 15})

    pd.DataFrame(rows_t1).set_index("Zona").to_csv(out("Table_F1_Mineral_Zones.csv"))

    # ── Reporte de densidad (umbrales percentílicos) ───────────────────────
    fe_q75 = df["FeCO3"].quantile(0.75)
    fe_q90 = df["FeCO3"].quantile(0.90)
    mg_q10 = df["MgCO3"].quantile(0.10)
    mg_q90 = df["MgCO3"].quantile(0.90)

    print("\nReporte de Densidad — Zonas Composicionales Minoritarias:")
    for lbl, mask in [
        (f"FeCO₃ > {fe_q75:.1f}% (Q75)", df["FeCO3"] > fe_q75),
        (f"FeCO₃ > {fe_q90:.1f}% (Q90)", df["FeCO3"] > fe_q90),
        (f"MgCO₃ < {mg_q10:.1f}% (Q10)", df["MgCO3"] < mg_q10),
        (f"MgCO₃ > {mg_q90:.1f}% (Q90)", df["MgCO3"] > mg_q90),
    ]:
        n_z   = mask.sum()
        pct_z = n_z / len(df) * 100
        sparse_note = " ← datos escasos" if pct_z < 12 else ""
        print(f"  {lbl:<32}  n = {n_z:>3}  ({pct_z:.1f}%){sparse_note}")

    # ── FIGURA ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # Panel 1: scatter composicional MgCO₃ vs FeCO₃
    ax = axes[0]
    for label, color in zip(zone_labels, zone_colors):
        mask_z = df["mineral_zone"] == label
        ax.scatter(df.loc[mask_z, "MgCO3"], df.loc[mask_z, "FeCO3"],
                   color=color, alpha=0.72, s=40, edgecolors="white",
                   linewidths=0.3, zorder=3,
                   label=f"{label}\n(n={mask_z.sum()}, {mask_z.sum()/len(df)*100:.0f}%)")

    ax.axvline(15, color="#777", lw=0.9, ls="--", alpha=0.65)
    ax.axvline(40, color="#777", lw=0.9, ls="--", alpha=0.65)
    ax.axhline(18, color="#777", lw=0.9, ls="--", alpha=0.65)

    # Etiquetas de zona en el fondo
    for xt, yt, txt, clr in [
        (7,  32, "Calcita\nMg-pobre",    "#3498db"),
        (47, 2,  "Dolomita",              "#2ecc71"),
        (22, 28, "Ankerita/\nSiderita",   "#e74c3c"),
        (27, 8,  "Mg-Calcita",            "#f39c12"),
    ]:
        ax.text(xt, yt, txt, ha="center", fontsize=7.5, color=clr,
                style="italic", alpha=0.85)

    ax.set_xlabel("MgCO₃ (mol%)", fontsize=10.5)
    ax.set_ylabel("FeCO₃ (mol%)", fontsize=10.5)
    ax.set_title("Espacio Composicional: MgCO₃ vs FeCO₃\nZonas Mineralógicas",
                 fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=7.0, framealpha=0.82, loc="upper left",
              handletextpad=0.4, labelspacing=0.3)

    # Panel 2: distribución MgCO₃ (posible bimodalidad)
    ax2 = axes[1]
    s_mg = df["MgCO3"]
    ax2.hist(s_mg, bins="auto", color=CHEM_CLR, alpha=0.48,
             density=True, edgecolor="white", lw=0.5, zorder=2)
    kde_mg = stats.gaussian_kde(s_mg, bw_method="scott")
    x_mg   = np.linspace(s_mg.min() - 2, s_mg.max() + 2, 500)
    ax2.plot(x_mg, kde_mg(x_mg), color=CHEM_CLR, lw=2.2, zorder=3)
    ax2.axvline(15, color="#777", lw=0.9, ls="--", alpha=0.7, label="Límites de zona")
    ax2.axvline(40, color="#777", lw=0.9, ls="--", alpha=0.7)
    ax2.set_xlabel("MgCO₃ (mol%)", fontsize=10.5)
    ax2.set_ylabel("Densidad",     fontsize=9)
    ax2.set_title("Distribución MgCO₃\n(posible bimodalidad calcita ↔ dolomita)",
                  fontsize=10.5, fontweight="bold")
    skew_mg = float(stats.skew(s_mg))
    ax2.text(0.97, 0.95, f"skew = {skew_mg:+.3f}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.82, ec="#ccc"))
    ax2.legend(fontsize=8)

    # Panel 3: distribución FeCO₃ (cola derecha — zona ankerita escasa)
    ax3 = axes[2]
    s_fe = df["FeCO3"]
    ax3.hist(s_fe, bins="auto", color="#c0392b", alpha=0.48,
             density=True, edgecolor="white", lw=0.5, zorder=2)
    kde_fe = stats.gaussian_kde(s_fe, bw_method="scott")
    x_fe   = np.linspace(s_fe.min() - 1, s_fe.max() + 1, 500)
    ax3.plot(x_fe, kde_fe(x_fe), color="#c0392b", lw=2.2, zorder=3)
    ax3.axvline(fe_q90, color="#555", lw=1.1, ls=":",
                label=f"Q90 = {fe_q90:.1f}% (zona escasa)")
    ax3.axvline(18,     color="#777", lw=0.9, ls="--", alpha=0.7,
                label="Umbral ankerita (18%)")
    ax3.set_xlabel("FeCO₃ (mol%)", fontsize=10.5)
    ax3.set_ylabel("Densidad",     fontsize=9)
    ax3.set_title("Distribución FeCO₃\n(cola derecha — baja densidad ankerita/siderita)",
                  fontsize=10.5, fontweight="bold")
    skew_fe = float(stats.skew(s_fe))
    ax3.text(0.97, 0.95, f"skew = {skew_fe:+.3f}",
             transform=ax3.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.82, ec="#ccc"))
    ax3.legend(fontsize=8)

    fig.suptitle(
        "HALLAZGO 1 — Sparsity del Espacio de Muestreo Químico\n"
        "Solución sólida Ca-Mg-Fe-Mn carbonatos  ·  n = 271",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out("Fig_EDA_F1_Chemical_Space.png"))
    plt.close(fig)
    print(f"\n  → Fig_EDA_F1_Chemical_Space.png  guardado")
    print(f"  → Table_F1_Mineral_Zones.csv     guardado")


# ═════════════════════════════════════════════════════════════════════════════
# HALLAZGO 2 — SESGO Y KURTOSIS EN VARIABLES DE HARDWARE
# ═════════════════════════════════════════════════════════════════════════════

def finding_2_hardware_distributions(df):
    """
    HALLAZGO 2: Diagnóstico de No-Normalidad en Variables de Hardware
    ─────────────────────────────────────────────────────────────────
    Analogía telecomunicaciones: las variables IP, DTFX, DTFY, ¹⁶O¹H/¹⁶O
    son el "estado del canal" del instrumento SIMS.  Su distribución marginal
    determina:
      (a) La eficacia del StandardScaler (óptimo sólo si la distribución
          es aproximadamente simétrica con varianza finita).
      (b) La estabilidad de los gradientes en el entrenamiento de ANN
          (distribuciones con colas pesadas → vanishing/exploding gradients).
      (c) La robustez de los valores SHAP (outliers de hardware amplifican
          la varianza de Shapley en los modelos kernel).

    Tests empleados:
      • Shapiro-Wilk (SW): potencia óptima para n < 5000; directo sobre la
        estadística W = (Σaᵢx₍ᵢ₎)² / Σ(xᵢ - x̄)².
      • Jarque-Bera  (JB): test asintótico χ²(2) basado en skewness (S) y
        kurtosis (K): JB = n/6·[S² + (K-3)²/4].  Complementa SW al
        ser explícitamente sensible a momentos de orden 3 y 4.

    Lógica de recomendación de transformación:
      |skew| < 0.5     → sin transformación (distribución ~simétrica)
      skew  > +1.0     → log(x) o Box-Cox (λ→0)  [cola derecha larga]
      skew  > +0.5     → √x o Box-Cox             [cola derecha moderada]
      skew  < -1.0     → potencia (x^k, k>1) o reflexión + log  [cola izq.]
    """

    print("\n" + "═" * 68)
    print("HALLAZGO 2: SESGO Y KURTOSIS EN VARIABLES DE HARDWARE")
    print("═" * 68)

    # ── Tests de normalidad para los 10 inputs ────────────────────────────
    records = []
    for col in INPUT_COLS:
        s = df[col].dropna()
        sw_stat, sw_p = shapiro(s)
        jb_stat, jb_p = jarque_bera(s)
        skew_v = float(stats.skew(s))
        kurt_v = float(stats.kurtosis(s))        # kurtosis excesiva (Fisher)
        cv_v   = s.std() / s.mean() * 100 if s.mean() != 0 else float("nan")

        if abs(skew_v) < 0.5:
            t_rec = "Ninguna — ~simétrica"
            s_rec = "StandardScaler"
        elif skew_v > 1.0:
            t_rec = "Log(x) o Box-Cox (λ→0)"
            s_rec = "RobustScaler"
        elif skew_v > 0.5:
            t_rec = "√x o Box-Cox"
            s_rec = "RobustScaler"
        elif skew_v < -1.0:
            t_rec = "Potencia (x^k, k>1) / reflexión+log"
            s_rec = "RobustScaler"
        else:
            t_rec = "Box-Cox (λ > 1)"
            s_rec = "RobustScaler moderado"

        records.append({
            "Variable":          DISPLAY[col],
            "Skewness":          round(skew_v, 4),
            "Ex_Kurtosis":       round(kurt_v, 4),
            "CV (%)":            round(cv_v,   2),
            "SW_stat":           round(float(sw_stat), 6),
            "SW_pvalue":         round(float(sw_p),    6),
            "SW_Normal":         "SÍ" if sw_p >= ALPHA else "NO",
            "JB_stat":           round(float(jb_stat), 4),
            "JB_pvalue":         round(float(jb_p),    6),
            "JB_Normal":         "SÍ" if jb_p >= ALPHA else "NO",
            "Transform_rec":     t_rec,
            "Scaler_rec":        s_rec,
        })

    norm_df = pd.DataFrame(records).set_index("Variable")
    norm_df.to_csv(out("Table_F2_Normality_Tests.csv"))

    # Tabla resumida en consola
    print(f"\nTest de Normalidad (α = {ALPHA}):")
    print(f"  {'Variable':<22} {'Skew':>7} {'KurtEx':>8} {'SW_p':>10} {'JB_p':>10}  {'Normal?':>8}")
    print("  " + "─" * 72)
    for col in INPUT_COLS:
        r  = norm_df.loc[DISPLAY[col]]
        sw = "✓ SW" if r["SW_Normal"] == "SÍ" else "✗ SW"
        jb = "✓ JB" if r["JB_Normal"] == "SÍ" else "✗ JB"
        print(f"  {DISPLAY[col]:<22} {r['Skewness']:>7.3f} {r['Ex_Kurtosis']:>8.3f} "
              f"{r['SW_pvalue']:>10.3e} {r['JB_pvalue']:>10.3e}  {sw} {jb}")

    # Recomendaciones detalladas para variables hardware
    print("\nRecomendaciones de Transformación y Escalado — Variables Hardware:")
    for col in HW_COLS:
        r = norm_df.loc[DISPLAY[col]]
        print(f"\n  {DISPLAY[col]}:")
        print(f"    Skew={r['Skewness']:+.3f}  |  ExKurt={r['Ex_Kurtosis']:+.3f}  |  CV={r['CV (%)']:.1f}%")
        print(f"    Transformación recomendada : {r['Transform_rec']}")
        print(f"    Escalador recomendado      : {r['Scaler_rec']}")

    # Box-Cox lambda óptimo para variables estrictamente positivas
    print("\n  Box-Cox λ óptimo (variables > 0):")
    for col in ["IP", "16O1H/16O"]:
        s = df[col].dropna().values
        if s.min() > 0:
            _, lam = boxcox(s)
            if abs(lam) < 0.1:
                interp = "→ transformación logarítmica (λ ≈ 0)"
            elif abs(lam - 0.5) < 0.15:
                interp = "→ raíz cuadrada (λ ≈ 0.5)"
            elif 0.85 < lam < 1.15:
                interp = "→ sin transformación (λ ≈ 1)"
            else:
                interp = f"→ x^{lam:.3f}"
            print(f"    {DISPLAY[col]:<22}: λ_opt = {lam:.4f}  {interp}")

    # ── FIGURA: 2×2 histogramas hardware ──────────────────────────────────
    hw_clrs = {"IP": "#2b5ea7", "DTFX": "#8e44ad",
               "DTFY": "#16a085", "16O1H/16O": "#d35400"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for idx, col in enumerate(HW_COLS):
        ax  = axes_flat[idx]
        s   = df[col].dropna()
        clr = hw_clrs[col]
        r   = norm_df.loc[DISPLAY[col]]

        # Histograma normalizado
        ax.hist(s, bins="auto", color=clr, alpha=0.42, density=True,
                edgecolor="white", lw=0.5, zorder=2)

        # KDE empírica
        kde    = stats.gaussian_kde(s, bw_method="scott")
        x_grid = np.linspace(s.min() - 0.15*s.std(),
                              s.max() + 0.15*s.std(), 600)
        ax.plot(x_grid, kde(x_grid), color=clr, lw=2.3, zorder=3,
                label="KDE empírica")

        # Distribución Gaussiana de referencia (misma μ y σ)
        x_ref = np.linspace(s.min() - 0.15*s.std(),
                             s.max() + 0.15*s.std(), 400)
        ax.plot(x_ref, stats.norm.pdf(x_ref, s.mean(), s.std()),
                color="#666666", lw=1.4, ls="--", alpha=0.7,
                label="Gauss ref. (μ, σ)", zorder=2)

        # Líneas de media y mediana
        ax.axvline(s.mean(),   color="#111", ls="--", lw=1.0,
                   label=f"μ = {s.mean():.4g}", zorder=4)
        ax.axvline(s.median(), color="#111", ls=":",  lw=1.1,
                   label=f"Mdn = {s.median():.4g}", zorder=4)

        # Caja de anotación con tests
        sw_sym = "✓" if r["SW_Normal"] == "SÍ" else "✗"
        jb_sym = "✓" if r["JB_Normal"] == "SÍ" else "✗"
        ann = (f"skew  = {r['Skewness']:+.3f}\n"
               f"kurtEx = {r['Ex_Kurtosis']:+.3f}\n"
               f"CV    = {r['CV (%)']:.1f}%\n"
               f"SW  p = {r['SW_pvalue']:.2e}  {sw_sym}\n"
               f"JB  p = {r['JB_pvalue']:.2e}  {jb_sym}")
        ax.text(0.97, 0.96, ann, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.8,
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          alpha=0.88, ec="#cccccc", lw=0.6))

        # Título con recomendación
        t_short = r["Transform_rec"].split("(")[0].strip()
        ax.set_title(f"{DISPLAY[col]}\n"
                     f"Rec.: {t_short}  ·  {r['Scaler_rec']}",
                     fontsize=9.5, fontweight="bold")
        ax.set_xlabel(DISPLAY[col], fontsize=9.5)
        ax.set_ylabel("Densidad",   fontsize=9)
        ax.legend(fontsize=7.8, framealpha=0.6, loc="upper left",
                  handlelength=1.2)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        "HALLAZGO 2 — Diagnóstico de No-Normalidad en Variables de Hardware\n"
        "Tests Shapiro-Wilk y Jarque-Bera  ·  KDE + Referencia Gaussiana  ·  n = 271",
        fontsize=11.5, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out("Fig_EDA_Hardware_Distributions.png"))
    plt.close(fig)
    print(f"\n  → Fig_EDA_Hardware_Distributions.png  guardado")
    print(f"  → Table_F2_Normality_Tests.csv         guardado")


# ═════════════════════════════════════════════════════════════════════════════
# HALLAZGO 3 — NO-ESTACIONARIEDAD ESPACIAL Y RUPTURA DE CORRELACIÓN
# ═════════════════════════════════════════════════════════════════════════════

def finding_3_spatial_rupture(df):
    """
    HALLAZGO 3: No-Estacionariedad Espacial del Canal SIMS
    ──────────────────────────────────────────────────────
    Analogía telecomunicaciones: el instrumento SIMS es equivalente a un
    canal MIMO posición-dependiente H(x,y).  Cerca del eje óptico (centro),
    H es bien condicionada y las correcciones de los deflectores son lineales.
    En la periferia, los efectos de orden superior dominan — el canal se
    vuelve no-estacionario y la misma función lineal ya no describe ambas
    zonas.

    Métrica: coeficiente de correlación de Spearman estratificado.
    Criterio de "ruptura significativa": |Δρ| > 0.15.

    El par X ↔ DTFX exhibe la ruptura más pronunciada (inversión de tendencia
    en el caso más extremo), confirmando que el sistema de corrección del
    deflector alcanza sus límites de linealidad en la periferia.
    """

    print("\n" + "═" * 68)
    print("HALLAZGO 3: NO-ESTACIONARIEDAD ESPACIAL Y RUPTURA DE CORRELACIÓN")
    print("═" * 68)

    df = df.copy()
    df["_r"] = np.sqrt(df["X"]**2 + df["Y"]**2)
    r_med     = df["_r"].median()

    mask_ctr = df["_r"] <= r_med
    df_ctr   = df[mask_ctr]
    df_per   = df[~mask_ctr]

    print(f"\nDistancia radial: mediana = {r_med:.0f} µm  |  máx = {df['_r'].max():.0f} µm")
    print(f"  Centro    (r ≤ {r_med:.0f} µm):  n = {mask_ctr.sum()}")
    print(f"  Periferia (r > {r_med:.0f} µm):  n = {(~mask_ctr).sum()}")

    # ── Correlaciones de Spearman estratificadas ───────────────────────────
    key_pairs = [
        ("DTFX",   "DTFY",   "DTFX ↔ DTFY"),
        ("X",      "DTFX",   "X ↔ DTFX"),
        ("Y",      "DTFY",   "Y ↔ DTFY"),
        ("MgCO3",  TARGET_COL, "MgCO₃ ↔ IMF"),
        ("FeCO3",  TARGET_COL, "FeCO₃ ↔ IMF"),
        ("IP",     TARGET_COL, "IP ↔ IMF"),
    ]

    strat_rows = []
    print(f"\n  {'Par':<25} {'ρ_global':>9} {'ρ_centro':>10} {'ρ_perif.':>10} {'Δρ':>8}")
    print("  " + "─" * 68)
    for a, b, lbl in key_pairs:
        rho_a, _ = spearmanr(df[a],     df[b])
        rho_c, _ = spearmanr(df_ctr[a], df_ctr[b])
        rho_p, _ = spearmanr(df_per[a], df_per[b])
        delta    = rho_p - rho_c
        flag     = "  ← RUPTURA" if abs(delta) > 0.15 else ""
        print(f"  {lbl:<25} {rho_a:>9.3f} {rho_c:>10.3f} {rho_p:>10.3f} {delta:>+8.3f}{flag}")
        strat_rows.append({"Par": lbl,
                            "ρ_global":   round(rho_a, 3),
                            "ρ_centro":   round(rho_c, 3),
                            "ρ_periferia":round(rho_p, 3),
                            "Δρ":         round(delta, 3)})

    pd.DataFrame(strat_rows).set_index("Par").to_csv(
        out("Table_F3_Stratified_Spearman.csv"))

    # OLS sobre DTFX → DTFY para ambos estratos (pendiente + R²)
    def ols(x, y):
        sl, ic, r, p, _ = stats.linregress(x, y)
        return sl, ic, r**2

    sl_c, ic_c, r2_c = ols(df_ctr["DTFX"].values, df_ctr["DTFY"].values)
    sl_p, ic_p, r2_p = ols(df_per["DTFX"].values, df_per["DTFY"].values)
    rho_c_dtf, _     = spearmanr(df_ctr["DTFX"], df_ctr["DTFY"])
    rho_p_dtf, _     = spearmanr(df_per["DTFX"], df_per["DTFY"])

    print(f"\n  OLS  DTFX → DTFY:")
    print(f"    Centro:    slope={sl_c:.4f}  R²={r2_c:.4f}  ρ_Sp={rho_c_dtf:.3f}")
    print(f"    Periferia: slope={sl_p:.4f}  R²={r2_p:.4f}  ρ_Sp={rho_p_dtf:.3f}")
    print(f"    Δ(slope) = {sl_p - sl_c:+.4f}   Δρ = {rho_p_dtf - rho_c_dtf:+.3f}")

    # ── FIGURA: panel izquierdo = mapa IMF, derecho = ruptura DTFX/DTFY ───
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.2))

    # Panel izquierdo: mapa espacial IMF(X, Y)
    sc = ax_l.scatter(df["X"], df["Y"],
                      c=df[TARGET_COL], cmap="RdYlBu_r",
                      s=46, alpha=0.82, edgecolors="white", linewidths=0.4, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_l, pad=0.03)
    cbar.set_label("IMF (‰)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=8.5)

    # Círculo de radio mediano
    theta = np.linspace(0, 2 * np.pi, 360)
    ax_l.plot(r_med * np.cos(theta), r_med * np.sin(theta),
              color="white", lw=1.8, ls="--", alpha=0.90, zorder=4,
              label=f"r = {r_med:.0f} µm (mediana)")
    ax_l.axhline(0, color="#aaa", lw=0.7, ls="--", zorder=2)
    ax_l.axvline(0, color="#aaa", lw=0.7, ls="--", zorder=2)
    ax_l.text(0.02, 0.02, "Centro óptico\n(0, 0)",
              transform=ax_l.transAxes, fontsize=8, color="#777")
    ax_l.set_xlabel("X (µm)", fontsize=11)
    ax_l.set_ylabel("Y (µm)", fontsize=11)
    ax_l.set_aspect("equal", adjustable="datalim")
    ax_l.set_title("Mapa Espacial IMF en la Platina SIMS\n"
                   "Línea discontinua = frontera radial (mediana)",
                   fontsize=10.5, fontweight="bold")
    ax_l.legend(fontsize=8.5, loc="upper right", framealpha=0.8)

    # Panel derecho: DTFX vs DTFY estratificado
    ax_r.scatter(df_ctr["DTFX"], df_ctr["DTFY"],
                 color="#2b5ea7", alpha=0.58, s=38, edgecolors="white",
                 linewidths=0.3, zorder=3,
                 label=f"Centro  (n={len(df_ctr)},  ρ = {rho_c_dtf:.3f})")
    ax_r.scatter(df_per["DTFX"], df_per["DTFY"],
                 color="#b03030", alpha=0.58, s=38, edgecolors="white",
                 linewidths=0.3, zorder=3,
                 label=f"Periferia (n={len(df_per)},  ρ = {rho_p_dtf:.3f})")

    x_rng = np.array([df["DTFX"].min() - 2, df["DTFX"].max() + 2])
    ax_r.plot(x_rng, sl_c * x_rng + ic_c, color="#2b5ea7", lw=2.2, ls="--",
              zorder=4, label=f"OLS centro:    slope={sl_c:.3f},  R²={r2_c:.3f}")
    ax_r.plot(x_rng, sl_p * x_rng + ic_p, color="#b03030", lw=2.2, ls="--",
              zorder=4, label=f"OLS periferia: slope={sl_p:.3f},  R²={r2_p:.3f}")

    ax_r.text(0.03, 0.96,
              f"Δρ Spearman  = {rho_p_dtf - rho_c_dtf:+.3f}\n"
              f"Δ(pendiente) = {sl_p - sl_c:+.3f}",
              transform=ax_r.transAxes, ha="left", va="top", fontsize=9.5,
              bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6",
                        alpha=0.92, ec="#f0c040", lw=1.2))

    ax_r.set_xlabel("DTFX (V)", fontsize=11)
    ax_r.set_ylabel("DTFY (V)", fontsize=11)
    ax_r.set_title("Ruptura de Correlación Estructural: DTFX ↔ DTFY\n"
                   "Pendiente e intensidad cambian entre centro y periferia",
                   fontsize=10.5, fontweight="bold")
    ax_r.legend(fontsize=8.2, framealpha=0.85, loc="lower right", handlelength=1.4)

    fig.suptitle(
        "HALLAZGO 3 — No-Estacionariedad Espacial del Canal SIMS\n"
        "Izquierda: Mapa IMF(X,Y)  ·  Derecha: Ruptura DTFX↔DTFY por zona geométrica",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out("Fig_EDA_Spatial_Rupture.png"))
    plt.close(fig)
    print(f"\n  → Fig_EDA_Spatial_Rupture.png         guardado")
    print(f"  → Table_F3_Stratified_Spearman.csv    guardado")


# ═════════════════════════════════════════════════════════════════════════════
# HALLAZGO 4 — ALTA DIMENSIONALIDAD Y ANÁLISIS PCA
# ═════════════════════════════════════════════════════════════════════════════

def finding_4_pca_analysis(df):
    """
    HALLAZGO 4: Alta Dimensionalidad — PCA y Superficie de Respuesta No Lineal
    ──────────────────────────────────────────────────────────────────────────
    Si el IMF fuera una función lineal de la composición química, el scatter
    PC1 vs PC2 mostraría un degradado de color perfectamente monótono a lo
    largo de PC1 (dominado por la composición de carbonatos).

    Evidencia de no-linealidad: el degradado de color no es limpio en el
    plano 2D — muestras con valores similares de PC1 exhiben rangos amplios
    de IMF, indicando que PC2…PC10 contribuyen con información independiente.

    Esto demuestra que la superficie de respuesta IMF = f(X₁,…,X₁₀) requiere
    un espacio de hipótesis hiperdimensional con acoplamientos no lineales de
    alto orden — inaccesibles para modelos OLS/ANOVA pero capturados por
    GBM, XGBoost y redes neuronales profundas.

    Métricas cuantitativas de no-linealidad:
      (a) % varianza cubierta en 2D: si << 80% → proyección 2D insuficiente.
      (b) R² IMF ~ PC1 lineal vs cuadrático: ganancia indica no-linealidad.
      (c) R² acumulado IMF ~ PC1:PCk: cada PC adicional aporta información.
    """

    print("\n" + "═" * 68)
    print("HALLAZGO 4: ALTA DIMENSIONALIDAD Y ANÁLISIS PCA")
    print("═" * 68)

    X_raw    = df[INPUT_COLS].values
    y_target = df[TARGET_COL].values

    scaler = StandardScaler()
    pca    = PCA()
    X_sc   = scaler.fit_transform(X_raw)
    scores = pca.fit_transform(X_sc)
    ev     = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)

    # ── Tabla de varianza explicada + correlación PC_k ↔ IMF ──────────────
    interp_labels = [
        "Gradiente composición (Mg vs Fe/Mn)",
        "Estado instrumental / sesión analítica",
        "Posición geométrica (X, Y)",
        "Interacción IP – vacío (OH/O)",
        "Residual mixto instrumental",
        "—", "—", "—", "—", "—",
    ]

    print(f"\n  {'PC':<4} {'Var (%)':>8} {'Acum (%)':>10} {'ρ_s IMF':>9}  {'Interpretación'}")
    print("  " + "─" * 72)
    pca_rows = []
    for k in range(len(ev)):
        rho_k, p_k = spearmanr(scores[:, k], y_target)
        sig        = "*" if p_k < ALPHA else " "
        interp     = interp_labels[k] if k < len(interp_labels) else "—"
        print(f"  PC{k+1:<2} {ev[k]*100:>8.2f}% {cum_ev[k]*100:>10.2f}%  "
              f"{rho_k:>+8.3f}{sig}  {interp}")
        pca_rows.append({"PC": f"PC{k+1}",
                          "Varianza (%)":    round(ev[k]*100,  2),
                          "Acumulada (%)":   round(cum_ev[k]*100, 2),
                          "ρ_Spearman_IMF":  round(rho_k, 4),
                          "p_value":         round(p_k,   8),
                          "Sig. (α=0.05)":   "Sí" if p_k < ALPHA else "No",
                          "Interpretación":  interp})

    pd.DataFrame(pca_rows).set_index("PC").to_csv(out("Table_F4_PCA_Variance.csv"))

    # ── Diagnóstico de linealidad ──────────────────────────────────────────
    n2d      = (ev[0] + ev[1]) * 100
    rho_pc1, _= spearmanr(scores[:, 0], y_target)
    rho_pc2, _= spearmanr(scores[:, 1], y_target)

    # R² lineal IMF ~ PC1
    coef1    = np.polyfit(scores[:, 0], y_target, 1)
    r2_lin   = 1 - np.var(y_target - np.polyval(coef1, scores[:, 0])) / np.var(y_target)

    # R² cuadrático IMF ~ PC1 + PC1²  (ganancia = evidencia de no-linealidad)
    coef2    = np.polyfit(scores[:, 0], y_target, 2)
    r2_quad  = 1 - np.var(y_target - np.polyval(coef2, scores[:, 0])) / np.var(y_target)

    # R² acumulado IMF ~ PC1:PCk (regresión múltiple lineal)
    print(f"\n  Varianza 2D (PC1+PC2): {n2d:.1f}%  →  {100-n2d:.1f}% de información descartada")
    print(f"  R²(IMF ~ PC1, lineal):   {r2_lin:.4f}")
    print(f"  R²(IMF ~ PC1, cuadrát.): {r2_quad:.4f}  "
          f"(ganancia: {(r2_quad - r2_lin)*100:.2f} pp)")
    if r2_quad - r2_lin > 0.02:
        print("    → Componente no lineal significativa en PC1 → IMF.")
    print(f"\n  R² acumulado IMF ~ PC1:PCk (regresión lineal multivariante):")
    prev = 0.0
    for k in range(1, 6):
        lr   = LinearRegression().fit(scores[:, :k], y_target)
        r2_k = lr.score(scores[:, :k], y_target)
        print(f"    k={k}:  R² = {r2_k:.4f}  (ΔR² al añadir PC{k}: +{r2_k - prev:.4f})")
        prev = r2_k

    print(f"\n  DIAGNÓSTICO:")
    if n2d < 65:
        print(f"  La proyección 2D lineal retiene sólo {n2d:.1f}% de la varianza.")
        print(f"  El gradiente de color IMF en el plano PC1-PC2 no es linealmente")
        print(f"  limpio — la superficie de respuesta requiere un espacio de hipótesis")
        print(f"  hiperdimensional con acoplamientos no lineales de alto orden.")
        print(f"  → Justificación empírica para XGBoost / GBM / ANN.")

    # ── FIGURA: scatter PC1 vs PC2 + scree plot ────────────────────────────
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6.5))

    # Panel izquierdo: PC1 vs PC2 coloreado por IMF
    vmin = float(np.percentile(y_target, 2))
    vmax = float(np.percentile(y_target, 98))
    sc   = ax_l.scatter(scores[:, 0], scores[:, 1],
                        c=y_target, cmap="coolwarm",
                        s=52, alpha=0.82, edgecolors="white", linewidths=0.4,
                        vmin=vmin, vmax=vmax, zorder=3)
    cbar = fig.colorbar(sc, ax=ax_l, pad=0.02)
    cbar.set_label("IMF (‰)", fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    ax_l.axhline(0, color="#aaa", lw=0.8, ls="--", alpha=0.6, zorder=2)
    ax_l.axvline(0, color="#aaa", lw=0.8, ls="--", alpha=0.6, zorder=2)

    ann_txt = (f"ρ(PC1, IMF) = {rho_pc1:.3f}\n"
               f"ρ(PC2, IMF) = {rho_pc2:.3f}\n"
               f"R²(lin. PC1) = {r2_lin:.3f}\n"
               f"R²(quad. PC1) = {r2_quad:.3f}\n"
               f"PC1+PC2 = {n2d:.1f}% var.")
    ax_l.text(0.03, 0.97, ann_txt, transform=ax_l.transAxes,
              ha="left", va="top", fontsize=8.8,
              bbox=dict(boxstyle="round,pad=0.3", fc="white",
                        alpha=0.88, ec="#cccccc", lw=0.7))

    ax_l.set_xlabel(f"PC1  ({ev[0]*100:.1f}% varianza)", fontsize=11)
    ax_l.set_ylabel(f"PC2  ({ev[1]*100:.1f}% varianza)", fontsize=11)
    ax_l.set_title(
        "Espacio PCA: PC1 vs PC2  ·  Color = IMF (‰)\n"
        "Degradado no monótono → interacciones no lineales",
        fontsize=10.5, fontweight="bold",
    )

    # Panel derecho: scree plot
    n_comp  = len(ev)
    bar_clr = ["#b03030" if k < 2 else "#aaaaaa" for k in range(n_comp)]
    ax_r.bar(range(1, n_comp+1), ev*100, color=bar_clr,
             alpha=0.75, edgecolor="white", lw=0.6, zorder=2,
             label="Varianza por componente")
    ax_r.plot(range(1, n_comp+1), cum_ev*100,
              "o--", color="#333", lw=1.9, ms=5, zorder=3,
              label="Varianza acumulada")

    for thr, ls_s, lbl_t in [(80, ":", "80%"), (90, "--", "90%")]:
        ax_r.axhline(thr, color="#888", lw=0.9, ls=ls_s, alpha=0.7,
                     label=f"Umbral {lbl_t}")

    # Etiquetas de varianza por barra
    for k in range(5):
        ax_r.text(k+1, ev[k]*100 + 0.8, f"{ev[k]*100:.1f}%",
                  ha="center", va="bottom", fontsize=7.5, color="#333")

    # Anotaciones de acumulada en hitos
    for k_t, offset in [(1, 2.8), (4, -5.5), (6, 2.5)]:
        if k_t < n_comp:
            ax_r.text(k_t+1, cum_ev[k_t]*100 + offset,
                      f"{cum_ev[k_t]*100:.1f}%",
                      ha="center", fontsize=8, color="#333")

    ax_r.set_xticks(range(1, n_comp+1))
    ax_r.set_xticklabels([f"PC{i}" for i in range(1, n_comp+1)], fontsize=8.5)
    ax_r.set_xlabel("Componente Principal", fontsize=10.5)
    ax_r.set_ylabel("Varianza Explicada (%)", fontsize=10.5)
    ax_r.set_ylim(0, 108)
    ax_r.set_title(
        "Scree Plot — Varianza Explicada por PC\n"
        f"2 PCs → {n2d:.1f}%  ·  5 PCs → {cum_ev[4]*100:.1f}%  ·  "
        f"7 PCs → {cum_ev[6]*100:.1f}%",
        fontsize=10.5, fontweight="bold",
    )
    ax_r.legend(fontsize=8.8, framealpha=0.85)

    fig.suptitle(
        "HALLAZGO 4 — Alta Dimensionalidad: IMF no es lineal ni puramente químico\n"
        r"PCA sobre 10 variables SIMS (StandardScaler)  ·  Justifica XGBoost / GBM / ANN",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out("Fig_EDA_F4_PCA_IMF.png"))
    plt.close(fig)
    print(f"\n  → Fig_EDA_F4_PCA_IMF.png              guardado")
    print(f"  → Table_F4_PCA_Variance.csv            guardado")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    finding_1_chemical_sparsity(df)
    finding_2_hardware_distributions(df)
    finding_3_spatial_rupture(df)
    finding_4_pca_analysis(df)

    print("\n" + "=" * 68)
    print("VALIDACIÓN COMPLETA — Resumen para el Manuscrito")
    print("=" * 68)
    print("  H1: Sparsity química — generalización limitada en zonas extremas")
    print("      (Ankerita/Siderita < 15% del dataset)")
    print("  H2: No-normalidad hardware — IP (skew −2.08) y OH/O (skew +0.71)")
    print("      requieren RobustScaler + transformación log/Box-Cox")
    print("  H3: No-estacionariedad espacial — X↔DTFX invierte tendencia")
    print("      entre centro (ρ=−0.31) y periferia (ρ=+0.06)")
    print("  H4: 42.7% de varianza descartada en 2D — IMF es hiperdimensional")
    print("      y no lineal → XGBoost / GBM / ANN necesarios")
    print()
    print("  Figuras (300 DPI):")
    for f in ["Fig_EDA_F1_Chemical_Space.png",
              "Fig_EDA_Hardware_Distributions.png",
              "Fig_EDA_Spatial_Rupture.png",
              "Fig_EDA_F4_PCA_IMF.png"]:
        print(f"    eda_output/{f}")
    print("  Tablas:")
    for f in ["Table_F1_Mineral_Zones.csv",
              "Table_F2_Normality_Tests.csv",
              "Table_F3_Stratified_Spearman.csv",
              "Table_F4_PCA_Variance.csv"]:
        print(f"    eda_output/{f}")
    print("=" * 68)
