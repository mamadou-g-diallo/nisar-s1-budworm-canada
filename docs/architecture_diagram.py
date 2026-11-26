"""Draw docs/architecture.svg, the pipeline diagram shown in the README (static image: it renders
everywhere, unlike Mermaid). Run from the repository root: python docs/architecture_diagram.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STYLE = {"src": ("#dbe9f7", "#3a6ea5"), "pipe": ("#fdf0d5", "#b8860b"), "out": ("#dff2df", "#3c8c3c")}
fig, ax = plt.subplots(figsize=(12, 7.2))
ax.set_xlim(0, 120); ax.set_ylim(0, 72); ax.axis("off")


def band(y, h, title, kind):
    f, e = STYLE[kind]
    ax.add_patch(FancyBboxPatch((1, y), 118, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=f, ec=e, lw=1.2, alpha=0.35))
    ax.text(3, y + h - 1.6, title, fontsize=11, fontweight="bold", color=e, va="top")


def box(x, y, w, h, text, kind, title=None):
    f, e = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1", fc=f, ec=e, lw=1.5))
    if title:
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.6, fontweight="bold",
                family="DejaVu Sans Mono")
        ax.text(x + w / 2, y + h - 6, text, ha="center", va="top", fontsize=9, linespacing=1.4)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, linespacing=1.4)
    return x, y, w, h


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#444444"))


band(52, 19, "External data sources (open, no account needed)", "src")
band(22, 28, "Pipeline (this repository, scripts/ driven by scripts/config.py)", "pipe")
band(1, 19, "Outputs (results/<aoi>/)", "out")

s1 = box(10, 54, 40, 12, "Sentinel-1 RTC\nMicrosoft Planetary Computer STAC", "src")
tbe = box(70, 54, 40, 12, "MRNF spruce budworm defoliation polygons\nDonnées Québec, CC-BY 4.0", "src")

w, h, y = 26, 20, 25
xs = [4, 33, 62, 91]
pipe = [("os1_download_data.py", "same relative orbit\nacross years; median\nsummer composite\n(nisar_s1)"),
        ("os1_feature_extraction.py", "VV/VH to dB, RVI\n(nisar_s1)"),
        ("os1_join_defoliation.py", "zonal statistics on\nTBE polygons, plus\nnegative sampling"),
        ("os1_exploratory_analysis.py", "Random Forest,\n5-fold CV, SHAP")]
for x, (t, d) in zip(xs, pipe, strict=True):
    box(x, y, w, h, d, "pipe", title=t)
for a, b in zip(xs[:-1], xs[1:], strict=True):
    arrow((a + w, y + h / 2), (b, y + h / 2))

arrow((30, 54), (xs[0] + w / 2, y + h))
arrow((90, 54), (xs[2] + w / 2, y + h))

outs = [(30, "sar_defoliation_samples.csv"), (60, "rvi_by_severity.png"), (90, "shap_summary.png")]
for x, t in outs:
    box(x, 4, 27, 9, t, "out")
arrow((xs[2] + w / 2, y), (30 + 13.5, 13))       # join -> samples table (read by the analysis step)
arrow((xs[3] + w / 2 - 3, y), (60 + 13.5, 13))
arrow((xs[3] + w / 2 + 2, y), (90 + 13.5, 13))

out = Path(__file__).resolve().parent / "architecture.svg"
fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
print(out)
