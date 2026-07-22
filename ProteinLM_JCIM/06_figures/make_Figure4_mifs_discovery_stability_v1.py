from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd


ROOT = Path("Paper2_ML/ProteinLM_JCIM")
OUTDIR = ROOT / "06_figures/Figure4_MIFS_Discovery_Stability_v1"
OUTDIR.mkdir(parents=True, exist_ok=True)

TOPN_FILE = ROOT / "09_feature_selection/results/topN_feature_retraining_results.tsv"
MIFS_PERF_FILE = ROOT / "12_Model_Interpretation/01_results/mifs_interpretation_model_performance.tsv"
COEF_FILE = ROOT / "12_Model_Interpretation/01_results/mifs_logistic_coefficients.tsv"
FREQ_FILE = ROOT / "09_feature_selection/results_nested_mifs/nested_mifs_feature_selection_frequency.tsv"
STABILITY_FILE = ROOT / "09_feature_selection/results_nested_mifs/nested_mifs_summary.tsv"

topn = pd.read_csv(TOPN_FILE, sep="\t")
mifs_perf = pd.read_csv(MIFS_PERF_FILE, sep="\t")
coef = pd.read_csv(COEF_FILE, sep="\t")
freq = pd.read_csv(FREQ_FILE, sep="\t")
stability = pd.read_csv(STABILITY_FILE, sep="\t")

top5_lr = topn[(topn["feature_set"] == "top5") & (topn["model"] == "LogisticRegression")].iloc[0]
mifs_lr = mifs_perf[mifs_perf["model"] == "LogisticRegression_MIFS"].iloc[0]
stability_row = stability.iloc[0]

mifs_features = {
    "n_tmd",
    "tmd_negative_fraction_std",
    "mean_tmd_gap",
    "loc_er",
}

display_names = {
    "n_tmd": "TMD number",
    "tmd_negative_fraction_std": "TMD negative-charge variability",
    "mean_tmd_gap": "Mean inter-TMD gap",
    "loc_er": "ER localization",
    "mean_relative_tmd_position": "Mean relative TMD position",
    "min_relative_tmd_position": "Minimum relative TMD position",
    "tmd_hydrophobicity_min": "Minimum TMD hydrophobicity",
    "loc_mitochondrion": "Mitochondrial localization",
    "protein_DE_fraction": "Protein acidic-residue fraction",
    "tmd_length_std": "TMD length variability",
    "tmd_positive_fraction_mean": "Mean TMD positive-charge fraction",
}

# Explicitly retain ER localization in the stability panel even though it was
# not selected in any of the five training folds.
if "loc_er" not in set(freq["feature"]):
    freq = pd.concat(
        [
            freq,
            pd.DataFrame(
                [{"feature": "loc_er", "selection_count": 0, "selection_frequency": 0.0}]
            ),
        ],
        ignore_index=True,
    )

freq["display_name"] = freq["feature"].map(display_names).fillna(freq["feature"])
freq["is_full_data_mifs"] = freq["feature"].isin(mifs_features)
freq["plot_label"] = np.where(
    freq["is_full_data_mifs"],
    freq["display_name"] + " (MIFS)",
    freq["display_name"],
)
freq = freq.sort_values(
    ["selection_count", "is_full_data_mifs", "display_name"],
    ascending=[False, False, True],
).reset_index(drop=True)

coef["display_name"] = coef["feature"].map(display_names).fillna(coef["feature"])
coef_order = ["n_tmd", "tmd_negative_fraction_std", "loc_er", "mean_tmd_gap"]
coef["order"] = coef["feature"].map({feature: index for index, feature in enumerate(coef_order)})
coef = coef.sort_values("order")

top5_source = pd.DataFrame(
    [
        {
            "stage": "Top5 fixed subset",
            "roc_auc_mean": top5_lr["roc_auc_mean"],
            "roc_auc_sd": top5_lr["roc_auc_sd"],
            "pr_auc_mean": top5_lr["pr_auc_mean"],
            "pr_auc_sd": top5_lr["pr_auc_sd"],
            "features": top5_lr["features"],
        },
        {
            "stage": "Four-feature MIFS fixed subset",
            "roc_auc_mean": mifs_lr["roc_auc_mean"],
            "roc_auc_sd": mifs_lr["roc_auc_sd"],
            "pr_auc_mean": mifs_lr["pr_auc_mean"],
            "pr_auc_sd": mifs_lr["pr_auc_sd"],
            "features": ",".join(sorted(mifs_features)),
        },
    ]
)

top5_source.to_csv(OUTDIR / "Figure4A_discovery_source.tsv", sep="\t", index=False)
coef.drop(columns=["order"]).to_csv(OUTDIR / "Figure4B_coefficients_source.tsv", sep="\t", index=False)
freq.to_csv(OUTDIR / "Figure4C_selection_stability_source.tsv", sep="\t", index=False)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)

COLORS = {
    "ink": "#172033",
    "line": "#566273",
    "neutral": "#E6E9EE",
    "blue": "#CFE3F4",
    "blue_dark": "#3D76B8",
    "orange": "#F8DEB5",
    "orange_dark": "#D28B2C",
    "green": "#D8EBCF",
    "green_dark": "#57934E",
    "gray_bar": "#A9B1BB",
    "note": "#F4F5F7",
}


def add_panel_label(ax, label, title):
    ax.text(-0.01, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    ax.text(0.045, 1.08, title, transform=ax.transAxes, fontsize=12.5, fontweight="bold", va="top")


def add_box(ax, x, y, width, height, text, facecolor, fontsize=9.2, bold=False):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.2,
        edgecolor=COLORS["line"],
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if bold else "normal",
        color=COLORS["ink"],
        linespacing=1.05,
    )
    return patch


def add_arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color=COLORS["line"],
            shrinkA=1,
            shrinkB=1,
        )
    )


fig = plt.figure(figsize=(14.2, 8.5), facecolor="white")
grid = fig.add_gridspec(
    2,
    2,
    height_ratios=[0.39, 0.61],
    width_ratios=[0.46, 0.54],
    hspace=0.38,
    wspace=0.52,
)

# -----------------------------------------------------------------------------
# Panel A: feature discovery workflow
# -----------------------------------------------------------------------------
ax_a = fig.add_subplot(grid[0, :])
ax_a.set_xlim(0, 1)
ax_a.set_ylim(0, 1)
ax_a.axis("off")
add_panel_label(ax_a, "A", "Exploratory feature discovery and MIFS definition")

box_width = 0.165
box_height = 0.42
box_y = 0.33
box_x = [0.02, 0.22, 0.42, 0.62, 0.82]

texts = [
    "46 engineered\ndescriptors",
    "Full-data XGBoost\nfeature ranking",
    f"Top5 + Logistic Regression\nFixed-subset ROC-AUC {top5_lr['roc_auc_mean']:.3f}",
    "Remove transporter annotation\nManually curated functional label",
    f"Four-feature MIFS\nFixed-subset ROC-AUC {mifs_lr['roc_auc_mean']:.3f}",
]
fills = [COLORS["blue"], COLORS["orange"], COLORS["orange"], COLORS["neutral"], COLORS["green"]]

for index, (x, text, fill) in enumerate(zip(box_x, texts, fills)):
    add_box(
        ax_a,
        x,
        box_y,
        box_width,
        box_height,
        text,
        fill,
        fontsize=9.0,
        bold=index in {0, 2, 4},
    )
    if index < len(box_x) - 1:
        add_arrow(
            ax_a,
            x + box_width + 0.006,
            box_y + box_height / 2,
            box_x[index + 1] - 0.006,
            box_y + box_height / 2,
        )

ax_a.text(
    0.5,
    0.18,
    "Feature discovery preceded fixed-subset cross-validation; fold-wise stability was assessed separately (panel C).",
    ha="center",
    va="center",
    fontsize=9.2,
    color=COLORS["line"],
    style="italic",
)

# -----------------------------------------------------------------------------
# Panel B: standardized coefficients from full-data MIFS model
# -----------------------------------------------------------------------------
ax_b = fig.add_subplot(grid[1, 0])
add_panel_label(ax_b, "B", "Associations in the full-data MIFS model")

y_b = np.arange(len(coef))
values = coef["standardized_coefficient"].to_numpy()
bar_colors = [COLORS["blue_dark"] if value > 0 else COLORS["orange_dark"] for value in values]
ax_b.barh(y_b, values, color=bar_colors, height=0.62, edgecolor="none")
ax_b.axvline(0, color=COLORS["line"], linewidth=1.0)
ax_b.set_yticks(y_b)
ax_b.set_yticklabels(coef["display_name"], fontsize=9.5)
ax_b.invert_yaxis()
ax_b.set_xlim(-0.43, 0.47)
ax_b.set_xlabel("Standardized Logistic Regression coefficient (β)", fontsize=10)
ax_b.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
ax_b.grid(axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.8)
ax_b.set_axisbelow(True)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.spines["left"].set_visible(False)
ax_b.tick_params(axis="y", length=0)

for y_value, coefficient in zip(y_b, values):
    offset = 0.015 if coefficient >= 0 else -0.015
    ax_b.text(
        coefficient + offset,
        y_value,
        f"{coefficient:+.3f}",
        va="center",
        ha="left" if coefficient >= 0 else "right",
        fontsize=9.2,
        color=COLORS["ink"],
    )

ax_b.text(
    0.5,
    -0.22,
    "Coefficient direction describes association, not causation.",
    transform=ax_b.transAxes,
    ha="center",
    va="top",
    fontsize=8.8,
    style="italic",
    color=COLORS["line"],
)

# -----------------------------------------------------------------------------
# Panel C: fold-wise selection stability
# -----------------------------------------------------------------------------
ax_c = fig.add_subplot(grid[1, 1])
add_panel_label(ax_c, "C", "Fold-wise feature-selection stability")

y_c = np.arange(len(freq))
colors_c = [COLORS["blue_dark"] if flag else COLORS["gray_bar"] for flag in freq["is_full_data_mifs"]]
ax_c.barh(
    y_c,
    freq["selection_count"],
    color=colors_c,
    height=0.58,
    edgecolor="none",
)
ax_c.set_yticks(y_c)
ax_c.set_yticklabels(freq["plot_label"], fontsize=8.8)
ax_c.invert_yaxis()
ax_c.set_xlim(0, 5.25)
ax_c.set_xticks(range(0, 6))
ax_c.set_xlabel("Training folds selecting feature (out of 5)", fontsize=10)
ax_c.grid(axis="x", color="#D9DDE3", linewidth=0.7, alpha=0.8)
ax_c.set_axisbelow(True)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.spines["left"].set_visible(False)
ax_c.tick_params(axis="y", length=0)

for y_value, count in zip(y_c, freq["selection_count"]):
    ax_c.text(
        count + 0.08,
        y_value,
        f"{int(count)}/5",
        va="center",
        ha="left",
        fontsize=8.6,
        color=COLORS["ink"],
    )

ax_c.legend(
    handles=[
        Patch(facecolor=COLORS["blue_dark"], label="Original full-data MIFS feature"),
        Patch(facecolor=COLORS["gray_bar"], label="Alternative selected descriptor"),
    ],
    loc="lower right",
    frameon=False,
    fontsize=8.3,
    bbox_to_anchor=(1.01, -0.23),
    ncol=1,
)

ax_c.text(
    0.0,
    -0.31,
    (
        f"Held-out performance: ROC-AUC {stability_row['roc_auc_mean']:.3f} ± "
        f"{stability_row['roc_auc_sd']:.3f}; PR-AUC {stability_row['pr_auc_mean']:.3f} ± "
        f"{stability_row['pr_auc_sd']:.3f}."
    ),
    transform=ax_c.transAxes,
    ha="left",
    va="top",
    fontsize=8.8,
    color=COLORS["ink"],
)

fig.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.15)

pdf = OUTDIR / "Figure4_mifs_discovery_stability_v1.pdf"
png = OUTDIR / "Figure4_mifs_discovery_stability_v1.png"
svg = OUTDIR / "Figure4_mifs_discovery_stability_v1.svg"

fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=600, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")

print("Saved:")
print(pdf)
print(png)
print(svg)
