import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pathlib import Path


OUTDIR = Path(__file__).resolve().parent

COLORS = {
    "data": "#E5E7EB",
    "label": "#D9EAF7",
    "positive": "#DCEFD5",
    "engineered": "#DCEFD5",
    "esm2": "#FCE5C3",
    "alphafold": "#E8DDF3",
    "analysis": "#E8EEF4",
    "discovery": "#FCE5C3",
    "mifs": "#DCEFD5",
    "outcome": "#E8DDF3",
    "edge": "#4B5563",
    "text": "#111827",
}


def add_box(ax, x, y, width, height, text, facecolor, fontsize=10.5,
            weight="normal", edgecolor=None, linewidth=1.1):
    edgecolor = edgecolor or COLORS["edge"]
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
        linespacing=1.12,
    )
    return patch


def add_arrow(ax, start, end, connectionstyle="arc3", linestyle="-", mutation=12):
    arrow = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=1.1,
        color=COLORS["edge"],
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        clip_on=False,
    )
    ax.add_patch(arrow)


def add_line(ax, start, end, linestyle="-"):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        transform=ax.transAxes,
        color=COLORS["edge"],
        linewidth=1.1,
        linestyle=linestyle,
        clip_on=False,
    )


plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(13.5, 7.2),
    gridspec_kw={"width_ratios": [0.82, 1.65]},
)

for ax in axes:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

ax_a, ax_b = axes

# -----------------------------------------------------------------------------
# Panel A: dataset construction
# -----------------------------------------------------------------------------
ax_a.text(-0.03, 1.01, "A", transform=ax_a.transAxes,
          fontsize=16, fontweight="bold", va="top")
ax_a.text(0.04, 1.01, "Dataset construction", transform=ax_a.transAxes,
          fontsize=14, fontweight="bold", va="top")

add_box(
    ax_a, 0.15, 0.84, 0.70, 0.095,
    "Annotated membrane-protein landscape\n5,028 TMDs",
    COLORS["data"], fontsize=11.2, weight="bold",
)

add_box(
    ax_a, 0.03, 0.65, 0.42, 0.105,
    "WT-lost TMD events\n121",
    COLORS["positive"], fontsize=11.0, weight="bold",
)
add_box(
    ax_a, 0.55, 0.65, 0.42, 0.105,
    "Other annotated TMDs\n4,907",
    COLORS["data"], fontsize=11.0,
)

add_arrow(ax_a, (0.50, 0.84), (0.24, 0.755))
add_arrow(ax_a, (0.50, 0.84), (0.76, 0.755))

add_box(
    ax_a, 0.22, 0.49, 0.56, 0.085,
    "Aggregate TMD labels by gene",
    COLORS["label"], fontsize=10.8, weight="bold",
)
add_arrow(ax_a, (0.24, 0.65), (0.43, 0.575))
add_arrow(ax_a, (0.76, 0.65), (0.57, 0.575))

add_box(
    ax_a, 0.15, 0.32, 0.70, 0.095,
    "Protein-level prediction dataset\n997 membrane proteins",
    COLORS["data"], fontsize=11.2, weight="bold",
)
add_arrow(ax_a, (0.50, 0.49), (0.50, 0.415))

add_box(
    ax_a, 0.03, 0.13, 0.42, 0.105,
    "WT-lost positive proteins\n74",
    COLORS["positive"], fontsize=10.8, weight="bold",
)
add_box(
    ax_a, 0.55, 0.13, 0.42, 0.105,
    "Background proteins\n923",
    COLORS["data"], fontsize=10.8,
)
add_arrow(ax_a, (0.50, 0.32), (0.24, 0.235))
add_arrow(ax_a, (0.50, 0.32), (0.76, 0.235))

ax_a.text(
    0.50, 0.035,
    "Positive protein: contains at least one WT-lost TMD event",
    transform=ax_a.transAxes,
    ha="center", va="center", fontsize=9.2, color=COLORS["edge"],
)

# Panel separator
ax_a.plot([1.035, 1.035], [0.04, 0.96], transform=ax_a.transAxes,
          color="#D1D5DB", linewidth=1.1, clip_on=False)

# -----------------------------------------------------------------------------
# Panel B: analysis framework
# -----------------------------------------------------------------------------
ax_b.text(-0.005, 1.01, "B", transform=ax_b.transAxes,
          fontsize=16, fontweight="bold", va="top")
ax_b.text(0.055, 1.01, "Representation and interpretation framework",
          transform=ax_b.transAxes, fontsize=14, fontweight="bold", va="top")

representation_specs = [
    (0.02, COLORS["engineered"],
     "Engineered biological\nrepresentation\n46 interpretable descriptors"),
    (0.35, COLORS["esm2"],
     "Sequence-based learned\nrepresentation\nESM2 embeddings"),
    (0.68, COLORS["alphafold"],
     "Structure-informed\nrepresentation\nAlphaFold-derived descriptors"),
]

for x, color, text in representation_specs:
    add_box(ax_b, x, 0.79, 0.30, 0.145, text, color, fontsize=10.3, weight="bold")

# Shared connector from the three representations to the two analysis scales.
centers = [0.17, 0.50, 0.83]
for center in centers:
    add_line(ax_b, (center, 0.79), (center, 0.735))
add_line(ax_b, (centers[0], 0.735), (centers[-1], 0.735))

add_box(
    ax_b, 0.08, 0.55, 0.38, 0.115,
    "Primary protein-level benchmark\n997 proteins • stratified 5-fold CV",
    COLORS["analysis"], fontsize=10.2, weight="bold",
)
add_box(
    ax_b, 0.54, 0.55, 0.38, 0.115,
    "Secondary TMD-level analyses\nGene-grouped cross-validation",
    COLORS["analysis"], fontsize=10.2, weight="bold",
)
add_arrow(ax_b, (0.36, 0.735), (0.27, 0.665))
add_arrow(ax_b, (0.64, 0.735), (0.73, 0.665))

add_box(
    ax_b, 0.24, 0.38, 0.52, 0.095,
    "Comparative benchmarking across\nrepresentations, models, and biological scale",
    COLORS["data"], fontsize=10.5, weight="bold",
)
add_arrow(ax_b, (0.27, 0.55), (0.40, 0.475))
add_arrow(ax_b, (0.73, 0.55), (0.60, 0.475))

add_box(
    ax_b, 0.07, 0.20, 0.40, 0.105,
    "Engineered protein-level\nfeature discovery\nXGBoost ranking • transporter ablation",
    COLORS["discovery"], fontsize=9.8, weight="bold",
)
add_arrow(ax_b, (0.40, 0.38), (0.29, 0.305))

add_box(
    ax_b, 0.56, 0.20, 0.37, 0.105,
    "MIFS + stability assessment\nFour interpretable descriptors\nFold-wise feature selection",
    COLORS["mifs"], fontsize=9.8, weight="bold",
)
add_arrow(ax_b, (0.47, 0.252), (0.56, 0.252))

add_box(
    ax_b, 0.25, 0.035, 0.50, 0.095,
    "Hypothesis-generating interpretation\nWT-lost remodeling susceptibility",
    COLORS["outcome"], fontsize=10.6, weight="bold",
)
add_arrow(ax_b, (0.745, 0.20), (0.60, 0.13))

plt.subplots_adjust(left=0.035, right=0.985, bottom=0.035, top=0.95, wspace=0.09)

for extension, kwargs in {
    "pdf": {},
    "png": {"dpi": 600},
    "svg": {},
}.items():
    fig.savefig(
        OUTDIR / f"Figure1_integrated_framework_v1.{extension}",
        bbox_inches="tight",
        **kwargs,
    )

(OUTDIR / "Figure1_source_counts.tsv").write_text(
    "quantity\tcount\n"
    "annotated_TMDs\t5028\n"
    "WT_lost_TMD_events\t121\n"
    "other_annotated_TMDs\t4907\n"
    "membrane_proteins\t997\n"
    "WT_lost_positive_proteins\t74\n"
    "background_proteins\t923\n",
    encoding="utf-8",
)

print(f"Saved Figure 1 v1 to: {OUTDIR}")
