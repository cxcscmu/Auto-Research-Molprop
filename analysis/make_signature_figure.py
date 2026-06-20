"""Generate the two-non-transfer-signature figure for the main manuscript.

The values are the published held-out aggregates already reported in the paper:
the TDC model-axis retained-snapshot trajectory (Table tab:trajtest) and the
Polaris data-axis aggregate (Table tab:heldout). This figure only replots
existing reported numbers and performs no recomputation.

Run from the repository root:
    MPLCONFIGDIR=/tmp/matplotlib .venv_drug/bin/python analysis/make_signature_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

VAL_COLOR = "#9AA0A6"   # validation, neutral grey (not an intervention-axis colour)
TEST_COLOR = "#222222"  # held-out test, near-black

# TDC model-axis retained-snapshot trajectory (Table tab:trajtest).
SNAP = [6, 12, 17, 22, 28, 39, 44]
VAL = [0.021, 0.024, 0.029, 0.032, 0.034, 0.034, 0.035]
TEST = [-0.014, -0.012, -0.006, -0.001, 0.000, 0.000, 0.002]

# Polaris data-axis aggregate (Table tab:heldout).
POL_VAL = 0.022
POL_TEST = -0.019

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)

fig, (a, b) = plt.subplots(1, 2, figsize=(7.25, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})

# Panel A: selection-variance non-transfer (validation climbs, held-out flat).
a.axhline(0, color="#999999", lw=0.8, ls="--", zorder=0)
a.plot(SNAP, VAL, color=VAL_COLOR, lw=1.8, marker="o", markerfacecolor="white",
       markeredgecolor=VAL_COLOR, markeredgewidth=1.5, markersize=5.5, zorder=3,
       label="Validation")
a.plot(SNAP, TEST, color=TEST_COLOR, lw=1.8, marker="o", markersize=5, zorder=3,
       label="Held-out test")
a.annotate("", xy=(44, VAL[-1]), xytext=(44, TEST[-1]),
           arrowprops={"arrowstyle": "<->", "color": "#777777", "lw": 0.9})
a.text(42.5, (VAL[-1] + TEST[-1]) / 2, "gap", color="#777777", fontsize=7.4,
       ha="right", va="center")
a.set_xlabel("Retained model-axis snapshot (trial)")
a.set_ylabel("Aggregate normalised improvement")
a.set_title("Selection variance, TDC model axis", fontweight="bold")
a.set_ylim(-0.03, 0.046)
a.grid(axis="y", color="#ECECEC", lw=0.7, zorder=0)
a.spines[["top", "right"]].set_visible(False)
a.legend(frameon=False, loc="center right")

# Panel B: distribution-shift non-transfer (validation positive, held-out negative).
b.axhline(0, color="#333333", lw=1.0, zorder=2)
b.bar([0, 1], [POL_VAL, POL_TEST], width=0.6, color=[VAL_COLOR, TEST_COLOR], zorder=3)
b.set_xticks([0, 1], ["Validation", "Held-out\ntest"])
b.set_title("Distribution shift, Polaris data axis", fontweight="bold")
b.set_ylim(-0.03, 0.046)
for x, v in [(0, POL_VAL), (1, POL_TEST)]:
    b.text(x, v + (0.0015 if v >= 0 else -0.0015), f"{v:+.3f}", ha="center",
           va="bottom" if v >= 0 else "top", fontsize=8.5, fontweight="bold",
           color=VAL_COLOR if v >= 0 else TEST_COLOR)
b.grid(axis="y", color="#ECECEC", lw=0.7, zorder=0)
b.spines[["top", "right", "left"]].set_visible(False)
b.tick_params(axis="y", length=0, labelleft=False)

fig.suptitle("Held-out certification separates two non-transfer signatures",
             y=1.05, fontsize=10.5, fontweight="bold")
fig.subplots_adjust(wspace=0.16)
fig.savefig(FIGURES / "fig_signatures.pdf", metadata={"CreationDate": None})
fig.savefig(FIGURES / "fig_signatures.png", dpi=220)
plt.close(fig)
print("wrote results/figures/fig_signatures.pdf and .png")
