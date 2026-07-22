from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTDIR = Path("Paper2_ML/ProteinLM_JCIM/06_figures/Figure5_Conceptual_MIFS_v1")
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ink": "#172033",
    "line": "#566273",
    "blue": "#CFE3F4",
    "orange": "#F8DEB5",
    "green": "#D8EBCF",
    "purple": "#E3D7F1",
    "note": "#F1F3F6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
    }
)


def add_box(ax, x, y, width, height, text, facecolor, fontsize=11, bold_lines=1):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=1.25,
        edgecolor=COLORS["line"],
        facecolor=facecolor,
    )
    ax.add_patch(patch)

    lines = text.split("\n")
    line_step = height / (len(lines) + 1)
    top = y + height - line_step
    for index, line in enumerate(lines):
        ax.text(
            x + width / 2,
            top - index * line_step,
            line,
            ha="center",
            va="center",
            fontsize=fontsize if index < bold_lines else fontsize - 0.4,
            fontweight="bold" if index < bold_lines else "normal",
            color=COLORS["ink"],
        )
    return patch


def add_arrow(ax, start, end, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.4,
            linestyle="--" if dashed else "-",
            color=COLORS["line"],
            shrinkA=2,
            shrinkB=2,
        )
    )


fig, ax = plt.subplots(figsize=(13.0, 7.6), facecolor="white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(
    0.5,
    0.96,
    "Conceptual interpretation of the full-data-derived MIFS",
    ha="center",
    va="top",
    fontsize=17,
    fontweight="bold",
    color=COLORS["ink"],
)

# Descriptor boxes
left_x = 0.055
left_width = 0.31
box_height = 0.165
left_ys = [0.69, 0.46, 0.23]

left_texts = [
    "Membrane topology\nTMD number (n_tmd)\nPositive association  β = +0.373",
    "Physicochemical heterogeneity\nTMD negative-charge variability\nPositive association  β = +0.241",
    "TMD organization\nMean inter-TMD gap (mean_tmd_gap)\nNegative association  β = −0.067",
]

for y, text in zip(left_ys, left_texts):
    add_box(ax, left_x, y, left_width, box_height, text, COLORS["blue"], fontsize=10.8)

right_x = 0.73
right_y = 0.46
right_width = 0.225
right_height = 0.165
add_box(
    ax,
    right_x,
    right_y,
    right_width,
    right_height,
    "Cellular context\nER localization (loc_er)\nNegative association  β = −0.314",
    COLORS["orange"],
    fontsize=10.8,
)

# Central MIFS representation
center_x = 0.43
center_y = 0.46
center_width = 0.235
center_height = 0.185
add_box(
    ax,
    center_x,
    center_y,
    center_width,
    center_height,
    "Full-data-derived MIFS\nFour interpretable descriptors\nArchitecture + cellular context",
    COLORS["green"],
    fontsize=11.2,
)

# Descriptor-to-MIFS arrows describe composition, not causality.
for y in left_ys:
    add_arrow(
        ax,
        (left_x + left_width + 0.008, y + box_height / 2),
        (center_x - 0.008, center_y + center_height / 2),
    )

add_arrow(
    ax,
    (right_x - 0.008, right_y + right_height / 2),
    (center_x + center_width + 0.008, center_y + center_height / 2),
)

# Susceptibility outcome and explicitly labeled statistical relationship.
outcome_x = 0.445
outcome_y = 0.20
outcome_width = 0.205
outcome_height = 0.125
add_box(
    ax,
    outcome_x,
    outcome_y,
    outcome_width,
    outcome_height,
    "WT-lost remodeling\nsusceptibility",
    COLORS["purple"],
    fontsize=11.4,
    bold_lines=2,
)

add_arrow(
    ax,
    (center_x + center_width / 2, center_y - 0.006),
    (outcome_x + outcome_width / 2, outcome_y + outcome_height + 0.006),
    dashed=True,
)
ax.text(
    center_x + center_width / 2 + 0.015,
    0.385,
    "statistical association",
    ha="left",
    va="center",
    fontsize=9.4,
    color=COLORS["line"],
    style="italic",
)

# Stability caveat, aligned with Figure 4C.
note = FancyBboxPatch(
    (0.17, 0.035),
    0.66,
    0.105,
    boxstyle="round,pad=0.018,rounding_size=0.025",
    linewidth=1.0,
    edgecolor="#9AA3AE",
    facecolor=COLORS["note"],
)
ax.add_patch(note)
ax.text(
    0.5,
    0.087,
    "Hypothesis-generating framework: exact descriptor selection varied across folds.\nInterpretation therefore emphasizes biological dimensions rather than a unique causal feature set.",
    ha="center",
    va="center",
    fontsize=9.3,
    color=COLORS["ink"],
)

pdf = OUTDIR / "Figure5_conceptual_mifs_v1.pdf"
png = OUTDIR / "Figure5_conceptual_mifs_v1.png"
svg = OUTDIR / "Figure5_conceptual_mifs_v1.svg"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")

print("Saved:")
print(pdf)
print(png)
print(svg)
