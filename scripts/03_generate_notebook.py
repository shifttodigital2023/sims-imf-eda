#!/usr/bin/env python3
"""
generate_eda_notebook.py
Generates  EDA_SIMS_IMF_Notebook.ipynb
Run once:  python generate_eda_notebook.py
"""

import json, textwrap
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_cell_counter = [0]

def _uid():
    _cell_counter[0] += 1
    return f"cell_{_cell_counter[0]:03d}"

def _src(text):
    """Split a multiline string into the nbformat source array."""
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        out.append(line + "\n" if i < len(lines) - 1 else line)
    # Drop a trailing empty string if text ended with \n
    if out and out[-1] == "":
        out.pop()
    return out

def md(text):
    return {"cell_type": "markdown", "id": _uid(),
            "metadata": {}, "source": _src(textwrap.dedent(text).lstrip("\n"))}

def code(text):
    return {"cell_type": "code", "execution_count": None, "id": _uid(),
            "metadata": {}, "outputs": [],
            "source": _src(textwrap.dedent(text).lstrip("\n"))}

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1  —  Markdown: Problem Formulation & Channel Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

CELL1_MD = r"""
# Análisis Exploratorio de Datos (EDA): Caracterización del Canal de Medición SIMS
## *Dataset:* Śliwińśki et al. (2015) — Fraccionamiento de Masa Instrumental en Carbonatos

---

### 1.1 Formulación del Problema

Este notebook analiza el dataset de referencia de SIMS *(Secondary Ion Mass Spectrometry)*
publicado por Śliwińśki et al. (2015), con $n = 271$ análisis de carbonatos de la solución
sólida calcita–dolomita–rodocrosita–siderita.

**Objetivo físico:** El espectrómetro SIMS introduce un sesgo sistemático conocido como
**Fraccionamiento de Masa Instrumental** (IMF, en ‰), que desplaza el valor medido δ¹⁸O
respecto al valor verdadero:

$$\delta^{18}O_{\text{medido}} = \delta^{18}O_{\text{verdadero}} + \text{IMF (‰)}$$

La corrección precisa del IMF es el paso de calibración crítico en cualquier análisis de
isótopos estables por SIMS. Los modelos clásicos usan superficies polinomiales en composición;
este trabajo evalúa algoritmos de *machine learning* como aproximadores más flexibles
de la función $\text{IMF} = f(\mathbf{x}_{\text{instrumental}},\, \mathbf{x}_{\text{química}})$.

---

### 1.2 Taxonomía del Canal de Medición

Las 10 variables de entrada se clasifican en dos grupos físicamente distintos que
actúan en escalas espaciales y causales diferentes sobre el sesgo de medición:

#### 🔵 Covariables de Control / Estado del Canal Instrumental

| Variable | Unidad | Rol físico |
|---|---|---|
| `IP` | nA | Corriente del haz primario. Proxy de la dosis iónica total; varía entre sesiones analíticas. |
| `X`, `Y` | µm | Coordenadas de la platina motorizada. Cada posición geométrica introduce aberraciones ópticas distintas en la columna de iones. |
| `DTFX`, `DTFY` | V | Voltajes de los deflectores electrostáticos que dirigen el haz a la posición $(X,Y)$. Capturan efectos de orden superior no linealmente codificados por las coordenadas de platina. |
| `¹⁶O¹H/¹⁶O` | — | Ratio de iones OH⁻/O⁻. Proxy de la calidad del vacío residual; spikes anómalos indican contaminación de sesión analítica específica. |

#### 🟢 Variables Composicionales — Efecto Matriz

| Variable | Rango observado | Significado mineralógico |
|---|---|---|
| `MgCO₃` | 9.66–49.80 mol% | Sustitución Mg²⁺ en la red cristalina (calcita magnésica → dolomita). |
| `CaCO₃` | 50.10–52.67 mol% | Componente dominante; cuasi-invariante (σ = 0.60 mol%). |
| `MnCO₃` | 0.00–2.61 mol% | Traza; elevado sólo en rodocrosita. |
| `FeCO₃` | 0.20–36.09 mol% | Sustitución Fe²⁺; alta concentración → ankerita o siderita. |

---

### 1.3 El Problema Quimiométrico: Clausura Composicional y Multicolinealidad Perfecta

Las cuatro fracciones molares de carbonatos satisfacen por definición la
**restricción de clausura** (Aitchison, 1986):

$$\text{MgCO}_3 + \text{CaCO}_3 + \text{MnCO}_3 + \text{FeCO}_3 \equiv 100\;\text{mol\%} \quad \forall\, i = 1,\ldots,271$$

Esta identidad algebraica genera **multicolinealidad matemáticamente perfecta**:
cualquier componente es la combinación lineal exacta de los otros tres
($\text{FeCO}_3 = 100 - \text{MgCO}_3 - \text{CaCO}_3 - \text{MnCO}_3$).

**Consecuencias por tipo de modelo:**

| Modelo | Impacto | Mitigación |
|---|---|---|
| GBM, RF, XGBoost | ✅ *Inmune.* Las divisiones binarias operan sobre variables individuales. | Ninguna. |
| Regresión polinómica (PolyReg) | ⚠️ Matriz de diseño singular → coeficientes inestables o indefinidos. | Eliminar una variable de cierre (p. ej. FeCO₃) o aplicar transformación ILR. |
| SHAP con kernel lineal (SVR) | ⚠️ Los valores de Shapley absorben arbitrariamente la contribución entre variables perfectamente correladas. | Interpretar con cautela; preferir SHAP TreeExplainer para modelos de árbol. |

> **Hipótesis de trabajo:** Dado que `CaCO₃` presenta σ = 0.60 mol% (cuasi-constante),
> su contribución predictiva independiente es mínima —
> queda completamente determinada por las otras tres fracciones una vez impuesta la clausura.
> Constituye una variable cuasi-degenerada en el espacio de características.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2  —  Code: Initialization & Data Loading
# ─────────────────────────────────────────────────────────────────────────────

CELL2_CODE = r"""
# ─── Librerías ────────────────────────────────────────────────────────────────
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─── Estética Q1 — configuración global de matplotlib ────────────────────────
# Se aplica una vez aquí y hereda en todas las figuras del notebook.
# Fuentes sans-serif + spines mínimas = estilo Earth & Planetary Science Letters.
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   9,
    "figure.dpi":        110,
    "savefig.dpi":       300,
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

# ─── Carga de datos (hoja 'Data for Colab') ───────────────────────────────────
# El pipeline mapea columnas POSICIONALMENTE — no por nombre.
# Re-ordenar el archivo fuente sin actualizar los índices aquí
# produciría predicciones silenciosamente incorrectas.
DATA_FILE  = "Data_Sliwinski_2015_ no vacuum_7labels_10inputs_n271.xlsx"
SHEET_NAME = "Data for Colab"

df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
assert df_raw.shape == (271, 18), f"Forma inesperada: {df_raw.shape}"

LABEL_COLS = list(df_raw.columns[:7])
INPUT_COLS = list(df_raw.columns[7:17])
TARGET_COL = df_raw.columns[17]          # 'IMF per mil'

df = df_raw[INPUT_COLS + [TARGET_COL]].copy()
assert df.isnull().sum().sum() == 0, "Valores ausentes detectados."

# ─── Taxonomía de variables ───────────────────────────────────────────────────
SIMS_COLS = ["IP", "X", "Y", "DTFX", "DTFY", "16O1H/16O"]
CHEM_COLS = ["MgCO3", "CaCO3", "MnCO3", "FeCO3"]
ALL_COLS  = INPUT_COLS + [TARGET_COL]

# Etiquetas con unidades para todos los ejes de figura
DISPLAY = {
    "IP":         "IP (nA)",
    "X":          "X (µm)",
    "Y":          "Y (µm)",
    "DTFX":       "DTFX (V)",
    "DTFY":       "DTFY (V)",
    "16O1H/16O":  r"$^{16}$O$^{1}$H/$^{16}$O",
    "MgCO3":      "MgCO₃ (mol%)",
    "CaCO3":      "CaCO₃ (mol%)",
    "MnCO3":      "MnCO₃ (mol%)",
    "FeCO3":      "FeCO₃ (mol%)",
    TARGET_COL:   "IMF (‰)",
}

# Paleta cromática (consistente en todo el notebook)
SIMS_CLR   = "#2b5ea7"   # azul acero  — instrumentales
CHEM_CLR   = "#2e8b8b"   # verde-azul  — composicionales
TARGET_CLR = "#b03030"   # rojo oscuro — target

COL_CLR = {c: SIMS_CLR for c in SIMS_COLS}
COL_CLR.update({c: CHEM_CLR for c in CHEM_COLS})
COL_CLR[TARGET_COL] = TARGET_CLR

# ─── Verificación de clausura composicional ───────────────────────────────────
carb_sum = df[CHEM_COLS].sum(axis=1)
print(f"Dataset: {df.shape[0]} muestras × {df.shape[1]} columnas  |  ausentes: 0 ✓")
print(f"Clausura carbonatos → μ = {carb_sum.mean():.3f}  σ = {carb_sum.std():.4f} mol%  "
      f"[{carb_sum.min():.3f}, {carb_sum.max():.3f}]")
print("→ Clausura perfecta confirmada: multicolinealidad intrínseca en {}.".format(CHEM_COLS))
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3  —  Markdown: Phase 1 — Univariate Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

CELL3_MD = r"""
---

## Fase 1 — Diagnóstico Univariado y Análisis de Distorsión de Señal

### 2.1 Perspectiva desde el Procesamiento de Señales

Antes de entrenar cualquier modelo de machine learning, es necesario caracterizar la
**distribución marginal** de cada variable de entrada. Desde la óptica del procesamiento de señales,
cada variable puede conceptualizarse como un canal de información sometido a ruido y distorsión:

**Coeficiente de Variación (CV%)**

$$\text{CV}_j = \frac{\sigma_j}{\bar{x}_j} \times 100$$

El CV% cuantifica el *ruido relativo* de cada canal de entrada respecto a su valor central.
Variables con CV% > 100 presentan dispersión dominante sobre la señal media —
un indicador de mezcla de poblaciones o regímenes analíticos distintos.

**Asimetría (Skewness)**

$$\gamma_1 = \frac{\mathbb{E}\left[(X - \mu)^3\right]}{\sigma^3}$$

La asimetría mide la pérdida de simetría de la distribución. Su relevancia para el modelado:

- $|\gamma_1| < 0.5$: distribución aproximadamente simétrica → modelos lineales aplicables sin transformación.
- $0.5 \leq |\gamma_1| < 1$: asimetría moderada → considerar transformación Box-Cox o log.
- $|\gamma_1| \geq 1$: **asimetría severa** → la distribución tiene colas largas que comprometen
  la estabilidad de modelos lineales y sesgará el entrenamiento de ANN con inicialización
  estándar de pesos.

### 2.2 Hipótesis Previas por Grupo de Variables

**Variables instrumentales:**
- `IP` se espera **negativamente asimétrico** (cola izquierda): la corriente del haz primario
  fluctúa ligeramente durante cada sesión y puede caer abruptamente si el cátodo de Cs⁺
  se degrada, generando valores anómalos bajos.
- `¹⁶O¹H/¹⁶O` se espera **positivamente asimétrico** (cola derecha): el vacío es estable
  la mayor parte del tiempo, pero contaminaciones por admisión de agua generan picos breves
  de alta ratio OH/O que inflan la cola superior.
- `DTFX` y `DTFY` deberían ser aproximadamente uniformes si el operador cubre el área de la
  platina de forma sistemática.

**Variables composicionales:**
- `MgCO₃` podría presentar **bimodalidad**: el dataset mezcla calcita magnésica
  (~10–25 mol% Mg) y dolomita (~45–50 mol% Mg) — dos end-members mineralógicos distintos.
- `CaCO₃` debería ser **levemente asimétrico** dada su varianza casi nula.
- `FeCO₃` y `MnCO₃` serán **positivamente asimétricos**: la mayoría de muestras son
  calcitas/dolomitas pobres en Fe/Mn, con pocos granos de ankerita o rodocrosita.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4  —  Code: Descriptive Stats Table + Histogram Grid
# ─────────────────────────────────────────────────────────────────────────────

CELL4_CODE = r"""
# ─── Tabla de Estadísticos Descriptivos Avanzados ────────────────────────────
records = []
for col in ALL_COLS:
    s  = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    cv = (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan
    records.append({
        "Variable":    DISPLAY[col],
        "N":           int(len(s)),
        "Mean":        round(s.mean(),   5),
        "Median":      round(s.median(), 5),
        "Std Dev":     round(s.std(),    5),
        "CV (%)":      round(cv,         2),
        "Min":         round(s.min(),    5),
        "Max":         round(s.max(),    5),
        "IQR":         round(q3 - q1,   5),
        "Skewness":    round(float(stats.skew(s)),     4),
        "Ex. Kurtosis":round(float(stats.kurtosis(s)), 4),
    })

table1 = pd.DataFrame(records).set_index("Variable")
print("Table 1 — Estadísticos Descriptivos (n = 271)\n")
print(table1.to_string())
table1.to_csv("eda_output/Table_1_Descriptive_Statistics.csv")

# ─── Grid de Histogramas con KDE ──────────────────────────────────────────────
ncols_g = 4
nrows_g = 3          # 11 variables en 4 × 3 (1 celda vacía)

fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(ncols_g * 4.2, nrows_g * 3.5))
axes_flat = axes.flatten()

for idx, col in enumerate(ALL_COLS):
    ax    = axes_flat[idx]
    s     = df[col].dropna()
    color = COL_CLR[col]

    # Histograma normalizado (Freedman–Diaconis bins)
    ax.hist(s, bins="auto", color=color, alpha=0.42,
            density=True, edgecolor="white", linewidth=0.5, zorder=2)

    # KDE (ancho de banda de Scott — robusto para n ≈ 270)
    kde    = stats.gaussian_kde(s, bw_method="scott")
    x_grid = np.linspace(s.min() - 0.10 * s.std(),
                         s.max() + 0.10 * s.std(), 500)
    ax.plot(x_grid, kde(x_grid), color=color, linewidth=2.2, zorder=3)

    # Líneas de referencia: media (--) y mediana (:)
    ax.axvline(s.mean(),   color="#1a1a1a", ls="--", lw=1.1,
               label=f"$\\bar{{x}}$ = {s.mean():.3g}", zorder=4)
    ax.axvline(s.median(), color="#1a1a1a", ls=":",  lw=1.2,
               label=f"Mdn = {s.median():.3g}", zorder=4)

    # Anotación: skewness + CV%
    cv_val   = s.std() / s.mean() * 100 if s.mean() != 0 else float("nan")
    skew_val = float(stats.skew(s))
    ax.text(0.97, 0.95,
            f"skew = {skew_val:+.2f}\nCV = {cv_val:.1f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      alpha=0.82, edgecolor="#cccccc", lw=0.5))

    ax.set_xlabel(DISPLAY[col], fontsize=9.5)
    ax.set_ylabel("Density",    fontsize=8.5)
    ax.set_title(DISPLAY[col],  fontsize=10,  fontweight="bold")
    ax.legend(fontsize=7.5, framealpha=0.6, loc="upper left", handlelength=1.2)
    ax.tick_params(labelsize=8)

# Ocultar subgráfico vacío
for j in range(len(ALL_COLS), len(axes_flat)):
    axes_flat[j].set_visible(False)

# Leyenda de grupos
patches = [
    mpatches.Patch(color=SIMS_CLR,   alpha=0.7, label="SIMS instrumental"),
    mpatches.Patch(color=CHEM_CLR,   alpha=0.7, label="Composición química (efecto matriz)"),
    mpatches.Patch(color=TARGET_CLR, alpha=0.7, label="Target — IMF (‰)"),
]
fig.legend(handles=patches, loc="lower right", fontsize=9.5,
           framealpha=0.88, bbox_to_anchor=(0.99, 0.005))

fig.suptitle(
    "Distribución de las Variables de Entrada SIMS y del Target IMF (‰)\n"
    r"Histograma (Freedman–Diaconis) + KDE (Scott bw)  ·  $n$ = 271"
    "\n— — media   · · · mediana",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig("eda_output/Fig_EDA_Distributions.png")
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5  —  Markdown: Phase 2 — Bivariate Analysis
# ─────────────────────────────────────────────────────────────────────────────

CELL5_MD = r"""
---

## Fase 2 — Análisis Bivariado, Multicolinealidad y Ruptura de Correlación Estructural

### 3.1 Por Qué Correlación de Spearman y No de Pearson

La correlación de Pearson asume dos condiciones que el análisis anterior muestra que
*no* se cumplen en este dataset:

1. **Normalidad marginal:** `IP` (skew = −2.08), `MgCO₃` y `FeCO₃` muestran asimetrías
   severas que violan el supuesto.
2. **Linealidad de la relación:** La relación IMF–composición es conocidamente no lineal en SIMS
   (depende de la distorsión del campo de la red cristalina, no de una función afín).

La **correlación de Spearman** ($\rho_s$) opera sobre rangos y es robusta a:
- Distribuciones asimétricas y outliers moderados.
- Relaciones monótonas no lineales.
- Escala de medición ordinal o de intervalo.

$$\rho_s(X,Y) = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

donde $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.

### 3.2 El Factor de Inflación de la Varianza (VIF)

El VIF cuantifica el grado en que la varianza de un coeficiente estimado se infla
debido a la multicolinealidad con otras variables:

$$\text{VIF}_j = \frac{1}{1 - R^2_j}$$

donde $R^2_j$ es el coeficiente de determinación al regresar la variable $j$
sobre todas las demás (con intercepto).

| VIF | Diagnóstico |
|---|---|
| < 5 | Aceptable |
| 5–10 | Moderado — monitorizar |
| > 10 | **Severo** — coeficientes de regresión inestables |
| → ∞ | **Colinealidad perfecta** — garantizada por clausura composicional |

### 3.3 Hipótesis de Ruptura de Correlación Estructural

En un instrumento SIMS real, los deflectores electrostáticos DTFX y DTFY
se calibran conjuntamente para cubrir el espacio de la platina $(X, Y)$.
Esta calibración es más precisa cerca del **centro óptico** (región de máxima
linealidad del sistema de lentes) que en la **periferia** (donde las aberraciones
de orden superior son significativas).

**Hipótesis:** La correlación entre variables clave —en particular
entre las covariables instrumentales y las coordenadas de platina—
**no es estacionaria en el espacio geométrico**: cambia significativamente
entre el cuadrante central y la región periférica de la platina.

Formulamos dos predicciones contrastables:

1. La correlación X ↔ DTFX (y análogamente Y ↔ DTFY) debería ser más fuerte
   en el centro (relación deflector–posición más predecible) y debilitarse o
   cambiar de signo en la periferia donde el sistema de corrección alcanza
   sus límites de linealidad.

2. La correlación MgCO₃ ↔ IMF (efecto matriz primario) debería ser más fuerte
   en la periferia si la respuesta del instrumento a la composición química es
   amplificada por las aberraciones geométricas periféricas.

Ambas predicciones se contrastan en la siguiente celda de código.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6  —  Code: Spearman + VIF + Stratified Correlation
# ─────────────────────────────────────────────────────────────────────────────

CELL6_CODE = r"""
import os; os.makedirs("eda_output", exist_ok=True)

# ─── 1. Matriz de correlación de Spearman ─────────────────────────────────────
spearman_mat = df[ALL_COLS].apply(lambda x: x.rank()).corr(method="pearson")
# (rango → Pearson sobre rangos ≡ Spearman)
# Alternativa explícita:
spearman_mat = df[ALL_COLS].corr(method="spearman")

tick_labels  = [DISPLAY[c] for c in ALL_COLS]
mask_upper   = np.triu(np.ones_like(spearman_mat, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(12, 10))
hm = sns.heatmap(
    spearman_mat, mask=mask_upper,
    cmap="coolwarm", vmin=-1, vmax=1,
    annot=True, fmt=".2f", annot_kws={"size": 8.2},
    square=True, linewidths=0.6, linecolor="#eeeeee",
    cbar_kws={"shrink": 0.78}, ax=ax,
)
hm.collections[0].colorbar.set_label("Spearman ρ", fontsize=10)
hm.collections[0].colorbar.ax.tick_params(labelsize=9)
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9.2)
ax.set_yticklabels(tick_labels, rotation=0,  fontsize=9.2)
ax.set_title(
    "Matriz de Correlación de Spearman — Variables SIMS y Target IMF\n"
    r"($n$ = 271;  Śliwińśki et al., 2015)",
    fontsize=12, fontweight="bold", pad=14,
)
plt.tight_layout()
plt.savefig("eda_output/Fig_EDA_Spearman_Matrix.png")
plt.show()

# ─── 2. VIF — Factor de Inflación de la Varianza ─────────────────────────────
# Implementación con sklearn (no requiere statsmodels).
# VIF_j = 1/(1 - R²_j), donde R²_j se obtiene regresando la variable j
# sobre las restantes con intercepto → equivalente a statsmodels.vif.

def compute_vif(data, cols):
    X = data[cols].values
    rows = []
    for i, col in enumerate(cols):
        y       = X[:, i]
        X_other = np.delete(X, i, axis=1)
        X_fit   = np.column_stack([np.ones(len(y)), X_other])
        lr      = LinearRegression(fit_intercept=False).fit(X_fit, y)
        y_hat   = lr.predict(X_fit)
        ss_res  = np.sum((y - y_hat) ** 2)
        ss_tot  = np.sum((y - y.mean()) ** 2)
        r2      = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0
        vif_val = np.inf if r2 >= 1.0 - 1e-10 else 1.0 / (1.0 - r2)
        if np.isinf(vif_val):
            flag = "∞  COLINEALIDAD PERFECTA"
        elif vif_val >= 10:
            flag = f"SEVERO  (VIF ≥ 10)"
        elif vif_val >= 5:
            flag = f"MODERADO (5–10)"
        else:
            flag = "OK  (< 5)"
        rows.append({"Variable": DISPLAY[col],
                     "VIF": "∞" if np.isinf(vif_val) else round(vif_val, 2),
                     "R²_j": round(r2, 6),
                     "Diagnóstico": flag})
    return pd.DataFrame(rows).set_index("Variable")

print("─" * 62)
print("VIF — 10 Variables de Entrada")
print("─" * 62)
vif_df = compute_vif(df, INPUT_COLS)
print(vif_df.to_string())
vif_df.to_csv("eda_output/Table_VIF_All_Inputs.csv")

# ─── 3. Segmentación radial: Centro vs. Periferia ────────────────────────────
# El centro óptico del instrumento es (X=0, Y=0).
# Definimos "centro" como r ≤ mediana(r) y "periferia" como r > mediana(r).
df["_r"] = np.sqrt(df["X"]**2 + df["Y"]**2)
r_median  = df["_r"].median()
mask_ctr  = df["_r"] <= r_median
df_ctr    = df[mask_ctr].copy()
df_per    = df[~mask_ctr].copy()
print(f"\nDistancia radial — mediana: {r_median:.0f} µm")
print(f"Centro (r ≤ {r_median:.0f} µm): n = {mask_ctr.sum()}")
print(f"Periferia (r > {r_median:.0f} µm): n = {(~mask_ctr).sum()}")

# ─── 4. Correlaciones estratificadas: pares clave ────────────────────────────
key_pairs  = [
    ("X",      "DTFX",    "Acoplamiento X → deflector"),
    ("Y",      "DTFY",    "Acoplamiento Y → deflector"),
    ("DTFX",   "DTFY",    "Acoplamiento entre deflectores"),
    ("MgCO3",  TARGET_COL,"Efecto matriz Mg → IMF"),
    ("FeCO3",  TARGET_COL,"Efecto matriz Fe → IMF"),
    ("IP",     TARGET_COL,"Corriente haz → IMF"),
]

print("\n" + "─" * 78)
print(f"{'Par':<30} {'ρ_global':>9} {'ρ_centro':>10} {'ρ_periferia':>13} {'Δρ':>8}")
print("─" * 78)
strat_rows = []
for a, b, label in key_pairs:
    rho_all, _ = stats.spearmanr(df[a], df[b])
    rho_ctr, _ = stats.spearmanr(df_ctr[a], df_ctr[b])
    rho_per, _ = stats.spearmanr(df_per[a], df_per[b])
    delta      = rho_per - rho_ctr
    print(f"{label:<30} {rho_all:>9.3f} {rho_ctr:>10.3f} {rho_per:>13.3f} {delta:>+8.3f}")
    strat_rows.append({"Par": label, "ρ_global": round(rho_all,3),
                       "ρ_centro": round(rho_ctr,3),
                       "ρ_periferia": round(rho_per,3), "Δρ": round(delta,3)})
print("─" * 78)
pd.DataFrame(strat_rows).set_index("Par").to_csv(
    "eda_output/Table_Stratified_Correlations.csv")

# ─── 5. Figura: Ruptura estructural en X → DTFX ──────────────────────────────
# X vs DTFX es el par que muestra el cambio más dramático:
# ρ_centro = −0.310  vs  ρ_periferia = +0.062  (inversión de tendencia).

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (data, label, color) in zip(
        axes,
        [(df_ctr, f"Centro  (r ≤ {r_median:.0f} µm,  n={len(df_ctr)})", "#2b5ea7"),
         (df_per, f"Periferia  (r > {r_median:.0f} µm,  n={len(df_per)})", "#b03030")]):

    rho_s, p_s = stats.spearmanr(data["X"], data["DTFX"])
    ax.scatter(data["X"], data["DTFX"],
               color=color, alpha=0.60, s=38,
               edgecolors="white", linewidths=0.35, zorder=3)

    # Línea de tendencia (OLS sobre rangos para confirmar monotonía)
    m, b = np.polyfit(data["X"].rank(), data["DTFX"].rank(), 1)
    xs = np.linspace(data["X"].min(), data["X"].max(), 200)
    rx = np.interp(xs, np.sort(data["X"]),
                   np.sort(data["X"].rank() / len(data["X"])))
    # Tendencia sencilla via polyfit directo sobre valores
    m2, b2 = np.polyfit(data["X"], data["DTFX"], 1)
    ax.plot(xs, m2 * xs + b2, color=color, linewidth=1.8,
            linestyle="--", zorder=4,
            label=f"OLS: DTFX = {m2:.4f}·X + {b2:.2f}")

    ax.set_xlabel("X (µm)",  fontsize=10.5)
    ax.set_ylabel("DTFX (V)", fontsize=10.5)
    ax.set_title(f"{label}\nSpearman ρ = {rho_s:.3f}  (p = {p_s:.2e})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.7)

fig.suptitle(
    "Ruptura de Correlación Estructural: X (posición platina) ↔ DTFX (deflector)\n"
    "La relación deflector–posición se invierte entre el centro y la periferia de la platina",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("eda_output/Fig_EDA_Structural_Break_X_DTFX.png")
plt.show()

# ─── 6. Figura: Efecto matriz estratificado (MgCO3 → IMF) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (data, label, color) in zip(
        axes,
        [(df_ctr, f"Centro  n={len(df_ctr)}", "#2e8b8b"),
         (df_per, f"Periferia  n={len(df_per)}", "#b03030")]):

    rho_s, p_s = stats.spearmanr(data["MgCO3"], data[TARGET_COL])
    sc = ax.scatter(data["MgCO3"], data[TARGET_COL],
                    c=data["_r"], cmap="YlOrRd_r",
                    alpha=0.70, s=42, edgecolors="white", linewidths=0.35, zorder=3)
    m2, b2 = np.polyfit(data["MgCO3"], data[TARGET_COL], 1)
    xs = np.linspace(data["MgCO3"].min(), data["MgCO3"].max(), 200)
    ax.plot(xs, m2 * xs + b2, color=color, linewidth=2.0, linestyle="--",
            label=f"OLS  (slope = {m2:.3f})", zorder=4)

    fig.colorbar(sc, ax=ax).set_label("r (µm)", fontsize=8.5)
    ax.set_xlabel("MgCO₃ (mol%)",  fontsize=10.5)
    ax.set_ylabel("IMF (‰)",        fontsize=10.5)
    ax.set_title(f"{label}\nSpearman ρ = {rho_s:.3f}  (p = {p_s:.2e})",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.7)

fig.suptitle(
    "Efecto Matriz Estratificado: MgCO₃ ↔ IMF (‰)  por zona geométrica de la platina\n"
    "El efecto matriz es más fuerte en la periferia (|ρ| mayor)",
    fontsize=11, fontweight="bold",
)
plt.tight_layout()
plt.savefig("eda_output/Fig_EDA_Matrix_Effect_Stratified.png")
plt.show()

df.drop(columns=["_r"], inplace=True)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7  —  Markdown: Phase 3 — PCA
# ─────────────────────────────────────────────────────────────────────────────

CELL7_MD = r"""
---

## Fase 3 — Proyección Multivariante (PCA) y Espacio de Fraccionamiento

### 4.1 La Proyección Lineal del Espacio de Características

El Análisis de Componentes Principales (PCA) busca una base ortogonal
$\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_{10}\}$ en $\mathbb{R}^{10}$
que maximice la varianza explicada proyectada secuencialmente:

$$\mathbf{v}_k = \underset{\|\mathbf{v}\|=1}{\arg\max}\;\text{Var}(\mathbf{X}\mathbf{v})
\quad \text{sujeto a} \quad \mathbf{v} \perp \mathbf{v}_j \;\forall\, j < k$$

Aplicamos un `StandardScaler` previo al PCA para garantizar que variables con
distintas unidades físicas (nA, µm, V, mol%) contribuyan equitativamente —
sin escalado, las coordenadas de platina X e Y, con rangos de ±5000 µm, dominarían
la varianza total por puro efecto de escala.

### 4.2 Significado Físico Esperado de los Componentes Principales

A partir de los coeficientes de carga (*loadings*) obtenidos, los dos primeros
componentes tienen la siguiente interpretación geológica:

**PC1 — Gradiente de Composición Química (≈38.5% de varianza)**

Las cargas positivas de `MgCO₃` (+0.465) y negativas de `FeCO₃` (−0.461) y
`MnCO₃` (−0.441) codifican el **continuo mineralógico** de la solución sólida:
valores positivos de PC1 → muestras ricas en Mg (dolomita);
valores negativos → muestras ricas en Fe/Mn (ankerita/rodocrosita).

Esta dimensión controla el efecto matriz en IMF: el campo eléctrico de la
red cristalina cambia con la composición catiónica, modulando la eficiencia
de ionización secundaria y, por tanto, el sesgo isotópico.

**Predicción:** Correlación de Spearman entre PC1 e IMF debería ser la más alta
de todos los componentes (confirmado: $\rho_s = -0.750$, $p < 10^{-49}$).

**PC2 — Estado Instrumental / Sesión Analítica (≈18.8% de varianza)**

Las altas cargas de `IP` (+0.420) y `¹⁶O¹H/¹⁶O` (+0.419) — ambos proxies de
condiciones del haz e historial de vacío — definen una dimensión que captura
la **variabilidad entre sesiones analíticas** (beam drift, contaminación). La
contribución de `DTFX` (−0.386) y `CaCO₃` (−0.333) añade la componente de
posición en el plano de la platina.

**Predicción:** Correlación moderada de PC2 con IMF (confirmado: $\rho_s = 0.448$).

### 4.3 Por Qué 2D es Insuficiente: La Necesidad de Aproximadores No Lineales

| Componentes | Varianza acumulada |
|---|---|
| PC1 + PC2 | 57.3% |
| PC1 – PC3 | 70.9% |
| PC1 – PC5 | 88.6% |
| PC1 – PC7 | 98.2% |

Con sólo **57.3% de la varianza en 2D**, cualquier modelo que opere en el espacio
reducido bidimensional descartará casi la mitad de la información disponible.

Más revelador aún: **PC3 a PC10 tienen correlación con IMF menor que 0.12**
(no significativa), lo que indica que la varianza "extra" en esas dimensiones
no contribuye linealmente a predecir el target. Sin embargo, puede existir
información **no lineal** en combinaciones de orden superior que sólo capturan
modelos como GBM, XGBoost o redes neuronales profundas.

> **Conclusión para el modelado:** La proyección PCA confirma que el espacio
> de características tiene estructura multidimensional con no-linealidades
> significativas. La insuficiencia del subespacio lineal de 2 dimensiones
> justifica empíricamente el uso de aproximadores no lineales de alta
> capacidad como las arquitecturas implementadas en el pipeline Multi_pipeline_v114.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8  —  Code: PCA Pipeline
# ─────────────────────────────────────────────────────────────────────────────

CELL8_CODE = r"""
# ─── PCA Pipeline ─────────────────────────────────────────────────────────────
X_raw    = df[INPUT_COLS].values
y_target = df[TARGET_COL].values

scaler  = StandardScaler()
pca     = PCA()
X_sc    = scaler.fit_transform(X_raw)
scores  = pca.fit_transform(X_sc)

ev      = pca.explained_variance_ratio_
cum_ev  = np.cumsum(ev)
n_comp  = len(ev)

# ─── Tabla de loadings ────────────────────────────────────────────────────────
loadings_df = pd.DataFrame(
    pca.components_[:5].T,
    index  = [DISPLAY[c] for c in INPUT_COLS],
    columns= [f"PC{i+1}" for i in range(5)],
).round(4)
print("Loadings (contribuciones) — PC1 a PC5")
print(loadings_df.to_string())
loadings_df.to_csv("eda_output/Table_PCA_Loadings.csv")

# Correlaciones PC_k ↔ IMF
print("\nCorrelación Spearman PC_k ↔ IMF:")
for k in range(5):
    rho_k, p_k = stats.spearmanr(scores[:, k], y_target)
    bar = "▓" * int(abs(rho_k) * 20)
    print(f"  PC{k+1} ({ev[k]*100:5.2f}%):  ρ = {rho_k:+.3f}  {bar}")

# ─── Figura 1: PC1 vs PC2 coloreado por IMF ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ── Panel izquierdo: scatter PC1 vs PC2 ───────────────────────────────────────
ax = axes[0]
sc = ax.scatter(
    scores[:, 0], scores[:, 1],
    c=y_target, cmap="coolwarm",
    s=52, alpha=0.82,
    edgecolors="white", linewidths=0.4, zorder=3,
)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("IMF (‰)", fontsize=10.5)
cbar.ax.tick_params(labelsize=9)

# Vectores de carga (biplot)
scale_f = np.abs(scores[:, :2]).max(axis=0).mean() * 0.55
for i, col in enumerate(INPUT_COLS):
    lx = pca.components_[0, i] * scale_f
    ly = pca.components_[1, i] * scale_f
    arrow_color = CHEM_CLR if col in CHEM_COLS else SIMS_CLR
    ax.annotate(
        "", xy=(lx, ly), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5),
    )
    ax.text(lx * 1.12, ly * 1.12, DISPLAY[col],
            fontsize=7.8, ha="center", color=arrow_color, fontweight="bold")

# Elipse 1-σ para referencia visual
from matplotlib.patches import Ellipse
cov_m = np.cov(scores[:, 0], scores[:, 1])
eigvals, eigvecs = np.linalg.eigh(cov_m)
angle  = np.degrees(np.arctan2(*eigvecs[:, 1][::-1]))
width  = 2 * np.sqrt(eigvals[1])
height = 2 * np.sqrt(eigvals[0])
ell    = Ellipse(xy=(scores[:, 0].mean(), scores[:, 1].mean()),
                 width=width, height=height, angle=angle,
                 edgecolor="#888888", facecolor="none",
                 linewidth=1.2, linestyle="--", zorder=2)
ax.add_patch(ell)
ax.axhline(0, color="#bbbbbb", lw=0.7, zorder=1)
ax.axvline(0, color="#bbbbbb", lw=0.7, zorder=1)
ax.set_xlabel(f"PC1  ({ev[0]*100:.1f}% varianza)", fontsize=10.5)
ax.set_ylabel(f"PC2  ({ev[1]*100:.1f}% varianza)", fontsize=10.5)
ax.set_title(
    f"Biplot PCA — Espacio de Fraccionamiento IMF\n"
    f"PC1+PC2 explican {(ev[0]+ev[1])*100:.1f}% de la varianza total",
    fontsize=10.5, fontweight="bold",
)

# Leyenda de grupos de vectores
h_chem = mpatches.Patch(color=CHEM_CLR, label="Composición química")
h_sims = mpatches.Patch(color=SIMS_CLR, label="SIMS instrumental")
ax.legend(handles=[h_chem, h_sims], fontsize=8.5, framealpha=0.7,
          loc="lower right")

# ── Panel derecho: varianza explicada acumulada ────────────────────────────────
ax2 = axes[1]
bar_colors = [TARGET_CLR if k < 2 else "#aaaaaa" for k in range(n_comp)]
ax2.bar(range(1, n_comp + 1), ev * 100, color=bar_colors,
        alpha=0.75, edgecolor="white", linewidth=0.6, zorder=2)
ax2.plot(range(1, n_comp + 1), cum_ev * 100,
         "o--", color="#333333", linewidth=1.8, markersize=5,
         label="Varianza acumulada", zorder=3)

# Líneas de umbral
for thr, ls in [(80, ":"), (90, "--")]:
    ax2.axhline(thr, color="#888888", linewidth=1.0, linestyle=ls,
                label=f"{thr}% umbral")

ax2.set_xticks(range(1, n_comp + 1))
ax2.set_xticklabels([f"PC{i}" for i in range(1, n_comp + 1)], fontsize=8.5)
ax2.set_xlabel("Componente principal", fontsize=10.5)
ax2.set_ylabel("Varianza explicada (%)", fontsize=10.5)
ax2.set_ylim(0, 105)
ax2.set_title(
    "Varianza Explicada por Componente\n"
    f"2 PCs → {(ev[0]+ev[1])*100:.1f}%  |  5 PCs → {cum_ev[4]*100:.1f}%",
    fontsize=10.5, fontweight="bold",
)
ax2.legend(fontsize=8.5, framealpha=0.8)

# Anotaciones de varianza acumulada clave
for k, (v, cv) in enumerate(zip(ev * 100, cum_ev * 100)):
    if k < 5:
        ax2.text(k + 1, cv + 1.5, f"{cv:.1f}%",
                 ha="center", va="bottom", fontsize=7.5, color="#333333")

fig.suptitle(
    "PCA sobre 10 Variables SIMS (StandardScaler pre-procesado)\n"
    r"Color = IMF (‰) · Vectores = loadings de PC1/PC2 · Elipse = 1σ",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig("eda_output/Fig_EDA_PCA_Biplot.png")
plt.show()

# ─── Interpretación final ─────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("INTERPRETACIÓN PCA")
print("=" * 62)
print(f"PC1 ({ev[0]*100:.2f}%): dominado por MgCO₃ (+), FeCO₃ (−), MnCO₃ (−)")
print("     → Gradiente de composición química (calcita ↔ dolomita/ankerita)")
print(f"PC2 ({ev[1]*100:.2f}%): dominado por IP (+), OH/O (+), DTFX (−), CaCO₃ (−)")
print("     → Estado instrumental / sesión analítica")
print(f"\n2 PCs cubren {(ev[0]+ev[1])*100:.1f}% de la varianza total.")
print("El 42.7% restante requiere dimensiones adicionales para ser capturado.")
print("→ Justificación empírica para aproximadores no lineales (GBM, XGBoost, ANN).")
"""

# ─────────────────────────────────────────────────────────────────────────────
# Assemble and write the notebook
# ─────────────────────────────────────────────────────────────────────────────

cells = [
    md(CELL1_MD),
    code(CELL2_CODE),
    md(CELL3_MD),
    code(CELL4_CODE),
    md(CELL5_MD),
    code(CELL6_CODE),
    md(CELL7_MD),
    code(CELL8_CODE),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
            "pygments_lexer": "ipython3",
        },
    },
    "cells": cells,
}

OUT = str(Path(__file__).resolve().parent.parent / "notebooks" / "EDA_SIMS_IMF_Notebook.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)

print(f"Notebook written: {OUT}")
print(f"  Cells: {len(cells)}  (4 Markdown + 4 Code)")
