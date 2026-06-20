"""Generate the held-out-result figures used by the main manuscript.

The figures are derived from the same frozen-configuration JSON files and data
axis run logs used by ``gen_results_md.py`` and ``data_audit_table.py``.

Run from the repository root:
    MPLCONFIGDIR=/tmp/matplotlib .venv_drug/bin/python analysis/make_transfer_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

import test_eval as te  # noqa: E402


ANALYSIS = ROOT / "results" / "heldout"
FIGURES = ROOT / "results" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

BENCHES = [
    ("tdc", "TDC ADMET"),
    ("molnet", "MoleculeNet"),
    ("polaris", "Polaris ADME"),
]
AXES = ["feature", "model", "data"]
COLORS = {
    "feature": "#0072B2",
    "model": "#E69F00",
    "data": "#009E73",
    "routed": "#172A57",
    "none": "#9AA0A6",
}
DISPLAY = {
    "caco2_wang": "Caco-2",
    "lipophilicity_astrazeneca": "Lipophilicity",
    "clearance_microsome_az": "Microsomal clearance",
    "cyp2c9_substrate_carbonmangels": "CYP2C9 substrate",
    "half_life_obach": "Half-life",
    "freesolv": "FreeSolv",
    "esol": "ESOL",
    "adme_hlm": "HLM clearance",
}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def load_result(bench: str, config: str) -> dict:
    return json.loads((ANALYSIS / f"{bench}_test_{config}.json").read_text())


def collect_results():
    aggregate = {}
    endpoints = []
    for bench, label in BENCHES:
        provider = te.get_provider(bench)
        data = {name: load_result(bench, name) for name in ["baseline", *AXES]}
        per_axis = {axis: {"val": [], "test": []} for axis in AXES}
        routed_val, routed_test = [], []

        for endpoint, baseline in data["baseline"].items():
            metric = baseline["metric"]
            val_gain, test_gain = {}, {}
            for axis in AXES:
                record = data[axis][endpoint]
                val_gain[axis] = provider.normalise(
                    record["val"], baseline["val"], metric
                )
                test_gain[axis] = provider.normalise(
                    record["test"], baseline["test"], metric
                )
                per_axis[axis]["val"].append(val_gain[axis])
                per_axis[axis]["test"].append(test_gain[axis])

            selected = max(AXES, key=val_gain.get)
            if val_gain[selected] <= 0.005:
                selected = "none"
                selected_val = selected_test = 0.0
            else:
                selected_val = val_gain[selected]
                selected_test = test_gain[selected]

            routed_val.append(selected_val)
            routed_test.append(selected_test)
            endpoints.append(
                {
                    "bench": bench,
                    "benchmark": label,
                    "endpoint": endpoint,
                    "axis": selected,
                    "val": selected_val,
                    "test": selected_test,
                }
            )

        aggregate[label] = {
            axis: {
                split: float(np.mean(values))
                for split, values in splits.items()
            }
            for axis, splits in per_axis.items()
        }
        aggregate[label]["routed"] = {
            "val": float(np.mean(routed_val)),
            "test": float(np.mean(routed_test)),
        }
    return aggregate, endpoints


def save(fig, name: str):
    # CreationDate=None drops the wall-clock timestamp so the PDF is byte-reproducible.
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGURES / f"{name}.png", dpi=220)
    plt.close(fig)
    print(f"wrote results/figures/{name}.pdf and .png")


def figure_transfer_map(aggregate):
    rows = ["feature", "model", "data", "routed"]
    row_labels = ["Feature", "Model", "Data", "Routed"]
    y = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(7.25, 3.05), sharex=True, sharey=True)

    for ax, (_, label) in zip(axes, BENCHES):
        ax.axvline(0, color="#555555", lw=0.8, ls="--", zorder=0)
        for yi, row in zip(y, rows):
            val = aggregate[label][row]["val"]
            test = aggregate[label][row]["test"]
            color = COLORS[row]
            ax.plot([val, test], [yi, yi], color=color, lw=2.0, alpha=0.72, zorder=1)
            ax.scatter(
                val,
                yi,
                s=43,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=3,
            )
            ax.scatter(test, yi, s=43, color=color, edgecolor="white", linewidth=0.4, zorder=4)
            if row == "routed":
                offset = 0.002 if test >= 0 else -0.002
                ax.text(
                    test + offset,
                    yi + 0.18,
                    f"{test:+.3f}",
                    color=color,
                    ha="left" if test >= 0 else "right",
                    va="center",
                    fontsize=7.2,
                )
        ax.set_title(label, pad=7, fontweight="bold")
        ax.set_yticks(y, row_labels)
        ax.set_xlim(-0.027, 0.073)
        ax.set_xticks([-0.02, 0.00, 0.02, 0.04, 0.06])
        ax.grid(axis="x", color="#E7E7E7", lw=0.7, zorder=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[1].set_xlabel("Aggregate normalised improvement over the strong baseline")
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor="#333333", markeredgewidth=1.3, label="Validation"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#333333",
               markeredgecolor="white", label="Held-out test"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False)
    fig.suptitle("Validation-selected improvements transfer differently across regimes",
                 y=1.12, fontsize=11, fontweight="bold")
    fig.subplots_adjust(wspace=0.12)
    save(fig, "fig_transfer_map")


def figure_endpoint_distribution(endpoints, aggregate):
    fig, ax = plt.subplots(figsize=(7.25, 3.35))
    base_y = {"tdc": 2, "molnet": 1, "polaris": 0}
    jitter = {bench: iter(np.linspace(-0.23, 0.23, sum(r["bench"] == bench for r in endpoints)))
              for bench, _ in BENCHES}

    point_position = {}
    for row in endpoints:
        x = row["test"]
        y = base_y[row["bench"]] + next(jitter[row["bench"]])
        point_position[(row["bench"], row["endpoint"])] = (x, y)
        ax.scatter(
            x,
            y,
            s=38,
            color=COLORS[row["axis"]],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )

    for bench, label in BENCHES:
        mean = aggregate[label]["routed"]["test"]
        ax.scatter(mean, base_y[bench], marker="D", s=58, color="#111111",
                   edgecolor="white", linewidth=0.6, zorder=5)
        positive = sum(r["bench"] == bench and r["test"] > 0 for r in endpoints)
        total = sum(r["bench"] == bench for r in endpoints)
        ax.text(0.183, base_y[bench] + 0.43, f"{positive}/{total} positive", ha="right",
                va="center", fontsize=7.3, color="#444444",
                bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88})

    annotations = [
        ("tdc", "cyp2c9_substrate_carbonmangels", (5, 8)),
        ("tdc", "lipophilicity_astrazeneca", (5, -15)),
        ("tdc", "caco2_wang", (6, 9)),
        ("tdc", "clearance_microsome_az", (6, -15)),
        ("molnet", "freesolv", (-5, 10)),
        ("molnet", "esol", (-5, -15)),
        ("polaris", "adme_hlm", (5, 8)),
    ]
    for bench, endpoint, offset in annotations:
        x, y = point_position[(bench, endpoint)]
        ax.annotate(
            f"{DISPLAY[endpoint]} {x:+.3f}",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="left" if offset[0] > 0 else "right",
            va="bottom" if offset[1] > 0 else "top",
            fontsize=7.0,
            color="#333333",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6},
        )

    ax.axvline(0, color="#555555", lw=0.9, ls="--", zorder=0)
    ax.set_xlim(-0.11, 0.19)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks([2, 1, 0], ["TDC ADMET", "MoleculeNet", "Polaris ADME"])
    ax.set_xlabel("Held-out normalised gain of the validation-routed intervention")
    ax.grid(axis="x", color="#E8E8E8", lw=0.7, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS[a], label=a.capitalize())
        for a in AXES
    ] + [
        Line2D([0], [0], marker="o", linestyle="", color=COLORS["none"], label="No route"),
        Line2D([0], [0], marker="D", linestyle="", color="#111111", label="Suite mean"),
    ]
    ax.legend(handles=handles, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    ax.set_title("Transfer is broad but heterogeneous across 36 endpoints",
                 pad=31, fontsize=11, fontweight="bold")
    save(fig, "fig_endpoint_transfer")


LINEAGES = [
    ("TDC ADMET", "drug_dev_maplight_data_v2"),
    ("MoleculeNet", "molnet_dev_data"),
    ("Polaris ADME", "polaris_dev_data"),
]
RETRACT = re.compile(r"^\s*(placeholder|cleared|empty override|removing)\b", re.I)


def snapshot_files(lineage: str):
    files = (ROOT / lineage / "blackboard" / "snapshots").glob(
        "[0-9]*_*/eval/run_seed0.jsonl"
    )
    return sorted(files, key=lambda p: int(p.parent.parent.name.split("_")[0]))


def read_snapshot(path: Path):
    try:
        return json.loads(path.read_text().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def collect_audit(aggregate):
    accepted = []
    rejected = {}
    for label, lineage in LINEAGES:
        snapshots = snapshot_files(lineage)
        records = [record for record in map(read_snapshot, snapshots) if record]
        best = max(records, key=lambda record: record.get("aggregate_score", -9))
        sources = []
        for endpoint, task in (best.get("per_task") or {}).items():
            aug = task.get("data_aug") or {}
            source = aug.get("source") or ""
            if (
                aug.get("verdict") == "accepted"
                and (aug.get("merged_rows") or 0) > 0
                and not RETRACT.match(source)
            ):
                sources.append(aug)
        accepted.append(
            {
                "label": label,
                "sources": len(sources),
                "rows": sum(source.get("merged_rows", 0) for source in sources),
                "gain": aggregate[label]["data"]["test"],
            }
        )

        for record in records:
            for endpoint, task in (record.get("per_task") or {}).items():
                aug = task.get("data_aug") or {}
                if aug.get("verdict") == "rejected_same_source":
                    key = (label, endpoint, (aug.get("source") or "")[:50])
                    rejected[key] = aug.get("test_overlap_rate")
    return accepted, [(label, endpoint, overlap) for (label, endpoint, _), overlap in rejected.items()]


def figure_data_audit(aggregate):
    accepted, rejected = collect_audit(aggregate)
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.25, 3.25), gridspec_kw={"width_ratios": [1.05, 1]})
    bench_colors = ["#5A5A5A", "#5A5A5A", "#5A5A5A"]

    left.axhline(0, color="#444444", lw=1.1, ls="--", zorder=0)
    left.text(2000, 0.0, "zero gain", color="#444444", fontsize=7.0, ha="right", va="bottom")
    for row, color in zip(accepted, bench_colors):
        left.scatter(row["rows"], row["gain"], s=80,
                     color=color, edgecolor="white", linewidth=0.7, zorder=3)
        left.annotate(
            f"{row['label']}\n{row['sources']} sources, +{row['rows']} rows",
            (row["rows"], row["gain"]),
            xytext=(6, 7 if row["gain"] >= 0 else -9),
            textcoords="offset points",
            fontsize=7.3,
            va="bottom" if row["gain"] >= 0 else "top",
        )
    left.set_xscale("log")
    left.set_xlim(80, 2000)
    left.set_ylim(-0.025, 0.018)
    left.set_xlabel("External rows admitted after filtering (log scale)")
    left.set_ylabel("Data-axis held-out gain")
    left.set_title("Admitted volume does not determine transfer", fontweight="bold", fontsize=9.5)
    left.grid(color="#E8E8E8", lw=0.7, zorder=0)
    left.spines[["top", "right"]].set_visible(False)

    endpoint_names = {
        "ames": "AMES / Hansen",
        "half_life_obach": "Half-life / PKSmart",
        "vdss_lombardo": "VDss / PKSmart",
    }
    rejected = sorted(rejected, key=lambda row: row[2] or 0)
    names = [endpoint_names.get(endpoint, endpoint) for _, endpoint, _ in rejected]
    overlaps = [100 * overlap for _, _, overlap in rejected]
    y = np.arange(len(names))
    bars = right.barh(y, overlaps, color="#C94C4C", height=0.58)
    right.axvline(5, color="#172A57", lw=1.2, ls="--", label="5% rejection threshold")
    for bar, value in zip(bars, overlaps):
        right.text(value + 1.5, bar.get_y() + bar.get_height() / 2,
                   f"{value:.0f}%", va="center", fontsize=8, fontweight="bold")
    right.set_yticks(y, names)
    right.set_xlim(0, 100)
    right.set_xlabel("Overlap with held-out test structures")
    right.set_title("High-overlap sources are rejected", fontweight="bold", fontsize=9.5)
    right.legend(frameon=False, loc="lower right")
    right.grid(axis="x", color="#E8E8E8", lw=0.7, zorder=0)
    right.spines[["top", "right", "left"]].set_visible(False)
    right.tick_params(axis="y", length=0)

    fig.suptitle("Auditable data acquisition separates contamination control from transfer",
                 y=1.04, fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.48)
    save(fig, "fig_data_audit")


def main():
    aggregate, endpoints = collect_results()
    figure_transfer_map(aggregate)
    figure_endpoint_distribution(endpoints, aggregate)
    # fig_data_audit additionally needs the data-axis agent run logs (the *_dev_*
    # lineage snapshot trees), which are large and not shipped in this repository.
    # It is provided as a static output in results/figures/; regenerate it only
    # where those logs are present.
    if (ROOT / LINEAGES[0][1] / "blackboard" / "snapshots").is_dir():
        figure_data_audit(aggregate)
    else:
        print("skip fig_data_audit: data-axis run logs not present "
              "(shipped as a static output in results/figures/)")


if __name__ == "__main__":
    main()
