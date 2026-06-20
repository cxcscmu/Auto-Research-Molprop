"""SUPERSEDED workspace script (earlier intervention-atlas / allocation framing) —
NOT part of the current paper. The manuscript figures are produced by
``make_transfer_figures.py`` and ``make_signature_figure.py``; this script is kept
only as exploratory history and writes to a local scratch directory.

  fig_atlas.pdf       TDC intervention atlas heatmap (22 endpoints x 3 axes)
  fig_crossbench.pdf  cross-benchmark allocation profile (grouped bars)
  fig_foresight.pdf   n_train vs best axis (foresight, 36 endpoints)

Run from the repo root:  .venv_drug/bin/python analysis/make_figures.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from foresight import build_rows, BENCH

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch_figures")
os.makedirs(FIGDIR, exist_ok=True)

# unified axis colours across all figures (Okabe-Ito, colour-blind safe: blue/orange/green)
AX = {"feature": "#0072B2", "model": "#E69F00", "data": "#009E73", "flat": "#9AA0A6"}

DISPLAY = {
    "caco2_wang": "Caco-2 Wang", "hia_hou": "HIA Hou", "pgp_broccatelli": "P-gp Broccatelli",
    "bioavailability_ma": "Bioavailability Ma", "lipophilicity_astrazeneca": "Lipophilicity AZ",
    "solubility_aqsoldb": "Solubility AqSolDB", "bbb_martins": "BBB Martins", "ppbr_az": "PPBR AZ",
    "vdss_lombardo": "VDss Lombardo", "cyp2d6_veith": "CYP2D6 Veith", "cyp3a4_veith": "CYP3A4 Veith",
    "cyp2c9_veith": "CYP2C9 Veith", "cyp2d6_substrate_carbonmangels": "CYP2D6 substrate",
    "cyp3a4_substrate_carbonmangels": "CYP3A4 substrate", "cyp2c9_substrate_carbonmangels": "CYP2C9 substrate",
    "half_life_obach": "Half-life Obach", "clearance_microsome_az": "Microsomal clearance",
    "clearance_hepatocyte_az": "Hepatocyte clearance", "herg": "hERG", "ames": "AMES",
    "dili": "DILI", "ld50_zhu": "LD50 Zhu",
    "freesolv": "FreeSolv", "esol": "ESOL", "bace": "BACE", "hiv": "HIV",
    "tox21_nr_ar": "Tox21 NR-AR", "tox21_sr_mmp": "Tox21 SR-MMP", "tox21_sr_p53": "Tox21 SR-p53",
    "sider_hepatobiliary": "SIDER hepatobiliary", "sider_reproductive": "SIDER reproductive",
    "clintox_ct_tox": "ClinTox", "adme_solu": "Solubility (Polaris)", "adme_mdr1": "MDR1-MDCK (Polaris)",
    "adme_rlm": "RLM clearance (Polaris)", "adme_hlm": "HLM clearance (Polaris)",
}

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=160)
    plt.close(fig)
    print("wrote", name + ".pdf / .png")


def fig_atlas(rows):
    tdc = [r for r in rows if r["benchmark"] == "TDC"]
    tdc.sort(key=lambda r: r["n_train"])
    M = np.array([[r["feature"], r["model"], r["data"]] for r in tdc])
    ylab = [f"{DISPLAY[r['endpoint']]}  (n={r['n_train']})" for r in tdc]
    cols = ["Feature", "Model", "Data"]
    fig, ax = plt.subplots(figsize=(4.4, 7.6))
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd",
                   norm=PowerNorm(gamma=0.5, vmin=0, vmax=M.max()))
    ax.set_xticks(range(3)); ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks(range(len(tdc))); ax.set_yticklabels(ylab, fontsize=8.5)
    ax.xaxis.tick_top()
    axis_idx = {"feature": 0, "model": 1, "data": 2}
    for i, r in enumerate(tdc):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.6,
                    color="black" if v < 0.6 * M.max() else "white")
        if r["best_axis"] in axis_idx:  # highlight selected axis cell
            j = axis_idx[r["best_axis"]]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="#1a1a1a", lw=2.0))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Normalised improvement", fontsize=8)
    ax.set_title("TDC ADMET intervention atlas", fontsize=10, pad=18)
    save(fig, "fig_atlas")


def fig_crossbench():
    # aggregate values taken from tab:transfer (single-pipeline best per axis)
    data = {
        "TDC ADMET (22)": {"feature": 0.013, "model": 0.041, "data": 0.032},
        "MoleculeNet (10)": {"feature": 0.020, "model": 0.018, "data": 0.001},
        "Polaris (4)": {"feature": 0.028, "model": 0.066, "data": 0.022},
    }
    benches = list(data); axes = ["feature", "model", "data"]
    x = np.arange(len(benches)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for k, a in enumerate(axes):
        vals = [data[b][a] for b in benches]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=a.capitalize(), color=AX[a],
                      edgecolor="white", linewidth=0.6)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0008,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(benches, fontsize=9.5)
    ax.set_ylabel("Aggregate normalised improvement", fontsize=9)
    ax.set_ylim(0, 0.075)
    ax.legend(title="Intervention axis", frameon=False, fontsize=8.5, title_fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Cross-benchmark allocation profile", fontsize=10)
    save(fig, "fig_crossbench")


def fig_foresight(rows):
    markers = {"TDC": "o", "MolNet": "s", "Polaris": "^"}
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    # data-candidate zone shading (n_train < 800)
    ax.axvspan(200, 800, color="#f2f2f2", zorder=0)
    ax.axvline(1500, ls="--", lw=1.0, color="#666666", zorder=1)
    for r in rows:
        best = max(r["feature"], r["model"], r["data"])
        y = max(best, 0.001)
        ax.scatter(r["n_train"], y, c=AX[r["best_axis"]], marker=markers[r["benchmark"]],
                   s=46, edgecolor="white", linewidth=0.5, zorder=3)
    for ep, dx, dy in [("freesolv", 1.0, 1.18), ("esol", 1.05, 0.80),
                       ("cyp2c9_substrate_carbonmangels", 0.55, 1.0)]:
        r = next(r for r in rows if r["endpoint"] == ep)
        best = max(r["feature"], r["model"], r["data"])
        ax.annotate(DISPLAY[ep], (r["n_train"], best), fontsize=6.8,
                    xytext=(r["n_train"] * dx, best * dy), color="#333333")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Training-set size  $n_{train}$  (log)", fontsize=9.5)
    ax.set_ylabel("Best-axis normalised improvement (log)", fontsize=9.5)
    ax.set_title("Forecasting the productive axis from prior attributes", fontsize=10)
    ax.text(430, 0.0013, "data-candidate zone\n6 model : 6 data", fontsize=7,
            ha="center", color="#555555", style="italic")
    ax.text(1600, 0.0013, r"$n_{train}\geq1500$: model 13/15", fontsize=7,
            ha="left", color="#555555", style="italic")
    ax.spines[["top", "right"]].set_visible(False)
    axis_leg = [Patch(facecolor=AX[a], label=a.capitalize()) for a in ["feature", "model", "data", "flat"]]
    mk_leg = [Line2D([0], [0], marker=m, color="#444", linestyle="", markersize=7, label=b)
              for b, m in markers.items()]
    leg1 = ax.legend(handles=axis_leg, title="Best axis", loc="upper right",
                     frameon=False, fontsize=8, title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=mk_leg, title="Benchmark", loc="lower right",
              frameon=False, fontsize=8, title_fontsize=8)
    save(fig, "fig_foresight")


def fig_trajectory():
    """Best-so-far aggregate over trials for the three TDC axes."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for axis, d in BENCH["TDC"].items():
        pts = []
        for f in glob.glob(f"{d}/blackboard/snapshots/*/eval/run_seed0.jsonl"):
            exp = f.split("/snapshots/")[1].split("/")[0]
            try:
                tn = int(exp.split("_")[0])
                agg = json.loads(open(f).readline()).get("aggregate_score")
            except Exception:
                continue
            if agg is not None:
                pts.append((tn, agg))
        pts.sort()
        xs, ys, best = [0], [0.0], 0.0
        for tn, agg in pts:
            best = max(best, agg)
            xs.append(tn)
            ys.append(best)
        ax.step(xs, ys, where="post", color=AX[axis], lw=1.9, label=axis.capitalize())
    ax.set_xlabel("Trial")
    ax.set_ylabel("Best-so-far aggregate improvement")
    ax.set_title("Search trajectories on TDC ADMET")
    ax.legend(frameon=False, title="Intervention axis")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_trajectory")


if __name__ == "__main__":
    rows = build_rows()
    fig_atlas(rows)
    fig_crossbench()
    fig_foresight(rows)
    fig_trajectory()
    print("done ->", os.path.abspath(FIGDIR))
