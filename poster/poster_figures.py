"""Figures of the French poster, read from the data and result tables of this repository.

f0_carte.png      2025 defoliation severity (MRNF) over the Sentinel-1 VV composite, AOI outline
f1_rvi.png        RVI per polygon by severity class (OS1 samples)
f2_f1.png         F1 per class: hand-crafted amplitude descriptors vs SSL4EO-S12 embeddings (OS4)

Run from the repository root: TBE_AOI=saguenay_lsj python poster/poster_figures.py
The F1 values are those printed by scripts/os1_exploratory_analysis.py (5-fold CV, n = 1 329) and
scripts/os4_foundation_model_probe.py (5-fold CV, n = 1 413); OS1 was re-run with nisar_s1 0.5.0 on 2026-11-26.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from config import AOI_POLYGON, PROC_DIR, RESULT_DIR  # noqa: E402
from os1_join_defoliation import load_tbe_polygons  # noqa: E402

OUT = ROOT / "poster" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
W = 15.3
plt.rcParams.update({"font.size": 23, "axes.titlesize": 23, "axes.labelsize": 23, "xtick.labelsize": 21,
                     "ytick.labelsize": 21, "legend.fontsize": 20, "font.family": "DejaVu Sans"})
VIRG = mticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
SEV = {0: ("non affecté", "#7fbf7b"), 1: ("léger", "#fdd835"), 2: ("modéré", "#fb8c00"), 3: ("grave", "#d32f2f")}
AMP, SSL = "#5b7083", "#1f5fae"


def tidy(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.9); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#aaaaaa")


# 0. Map ------------------------------------------------------------------------------------------
print("[1/3] carte (lecture de l'archive MRNF)", flush=True)
with rasterio.open(PROC_DIR / "vv_db_2025.tif") as ds:
    vv = ds.read(1).astype(float); vv[(vv == ds.nodata) | ~np.isfinite(vv)] = np.nan
    crs, b = ds.crs, ds.bounds
tbe = load_tbe_polygons()
t25 = tbe[tbe.year == 2025].to_crs(crs)
import geopandas as gpd  # noqa: E402
aoi = gpd.GeoSeries([AOI_POLYGON], crs=4326).to_crs(crs)
fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.5), constrained_layout=True)
cm = plt.get_cmap("gray").copy(); cm.set_bad("#e3e3e3")          # outside the Sentinel-1 swath
ax.imshow(vv, cmap=cm, vmin=-14, vmax=-4, extent=(b.left, b.right, b.bottom, b.top))
for k in (1, 2, 3):
    sub = t25[t25.severity == k]
    if len(sub):
        sub.plot(ax=ax, color=SEV[k][1], alpha=0.55, edgecolor="none")
aoi.boundary.plot(ax=ax, color="#00acc1", lw=2.5)
ax.set_xlim(b.left, b.right); ax.set_ylim(b.bottom, b.top)
ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
x0, y0 = b.left + 0.06 * (b.right - b.left), b.bottom + 0.05 * (b.top - b.bottom)
ax.plot([x0, x0 + 10000], [y0, y0], color="black", lw=5); ax.text(x0, y0 + 900, "10 km", fontsize=18)
ax.legend(handles=[Patch(color=SEV[k][1], alpha=0.7, label=SEV[k][0]) for k in (1, 2, 3)]
          + [Patch(facecolor="#e3e3e3", edgecolor="#999999", label="hors fauchée S1"),
             Patch(facecolor="none", edgecolor="#00acc1", lw=2.5, label="zone d'étude")],
          loc="upper right", fontsize=16, framealpha=0.9)
fig.savefig(OUT / "f0_carte.png", dpi=200); plt.close(fig)

# 1. RVI by severity ------------------------------------------------------------------------------
print("[2/3] RVI", flush=True)
s = pd.read_csv(RESULT_DIR / "sar_defoliation_samples.csv")
fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.5), constrained_layout=True)
data = [s[s.severity == k].rvi_mean.dropna().values for k in (0, 1, 2, 3)]
bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color="black", lw=2.5))
for patch, k in zip(bp["boxes"], (0, 1, 2, 3), strict=True):
    patch.set_facecolor(SEV[k][1]); patch.set_alpha(0.85)
ax.set_xticks([1, 2, 3, 4], [f"{SEV[k][0]}\n(n = {len(d)})" for k, d in zip((0, 1, 2, 3), data, strict=True)], fontsize=18)
ax.set_ylabel("RVI moyen du polygone"); ax.yaxis.set_major_formatter(VIRG)
tidy(ax)
fig.savefig(OUT / "f1_rvi.png", dpi=200); plt.close(fig)
pd.DataFrame({"classe": [SEV[k][0] for k in (0, 1, 2, 3)], "n": [len(d) for d in data],
              "mediane_rvi": [np.median(d) for d in data]}).round(3).to_csv(OUT / "f1_rvi.csv", index=False)

# 2. F1 per class ---------------------------------------------------------------------------------
print("[3/3] F1", flush=True)
F1 = pd.DataFrame({"classe": ["léger", "modéré", "grave"],
                   "amplitude": [0.553, 0.270, 0.101],      # os1_exploratory_analysis.py
                   "ssl4eo": [0.72, 0.58, 0.39]})            # os4_foundation_model_probe.py
fig, ax = plt.subplots(figsize=(W * 0.5, W * 0.5), constrained_layout=True)
x = np.arange(len(F1)); w = 0.36
for k, (col, lab, c) in enumerate((("amplitude", "VV, VH, RVI (forêt aléatoire)", AMP),
                                   ("ssl4eo", "modèle SSL4EO-S12 (sonde linéaire)", SSL))):
    bars = ax.bar(x + (k - 0.5) * w, F1[col], w, color=c, label=lab)
    for bb, v in zip(bars, F1[col], strict=True):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 0.01, f"{v:.2f}".replace(".", ","), ha="center", fontsize=17)
ax.set_xticks(x, F1.classe); ax.set_ylabel("F1 (validation croisée)"); ax.set_ylim(0, 0.85)
ax.yaxis.set_major_formatter(VIRG); ax.legend(loc="upper right", fontsize=16, frameon=False)
tidy(ax)
fig.savefig(OUT / "f2_f1.png", dpi=200); plt.close(fig)
F1.to_csv(OUT / "f2_f1.csv", index=False)
print("ok")
