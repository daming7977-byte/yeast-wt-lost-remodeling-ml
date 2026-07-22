import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = Path(__file__).resolve().parent

ENGINEERED_FILE = ROOT / "ProteinLM_JCIM/05_results/protein_features_v2_enriched_benchmark.tsv"
ESM35_FILE = ROOT / "ProteinLM_JCIM/05_results/protein_level_esm2_35M_benchmark.tsv"
ESM650_FILE = ROOT / "ProteinLM_JCIM/05_results/protein_level_esm2_650M_benchmark.tsv"
AF_FILE = ROOT / "ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_feature_benchmark.tsv"

engineered = pd.read_csv(ENGINEERED_FILE, sep="\t")
esm35 = pd.read_csv(ESM35_FILE, sep="\t")
esm650 = pd.read_csv(ESM650_FILE, sep="\t")
af = pd.read_csv(AF_FILE, sep="\t")


def select_row(df, feature_set, model):
    row = df[(df["feature_set"] == feature_set) & (df["model"] == model)]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {feature_set}/{model}; found {len(row)}")
    return row.iloc[0]


def make_record(label, row):
    return {
        "label": label,
        "feature_set": row["feature_set"],
        "model": row["model"],
        "n_features": int(row["n_features"]),
        "roc": float(row["roc_auc_mean"]),
        "roc_sd": float(row["roc_auc_sd"]),
        "pr": float(row["pr_auc_mean"]),
        "pr_sd": float(row["pr_auc_sd"]),
    }


panel_a = pd.DataFrame([
    make_record(
        "Engineered\n46 descriptors",
        select_row(engineered, "protein_feature_matrix_v2_enriched", "RandomForest"),
    ),
    make_record(
        "ESM2-35M\n480 dimensions",
        select_row(esm35, "esm2_full_protein", "RandomForest"),
    ),
    make_record(
        "Traditional-16 +\nESM2-35M",
        select_row(esm35, "traditional_plus_esm2_full", "RandomForest"),
    ),
    make_record(
        "ESM2-650M\n1,280 dimensions",
        select_row(esm650, "esm2_650M_full_protein", "RandomForest"),
    ),
    make_record(
        "Traditional-16 +\nESM2-650M",
        select_row(esm650, "traditional_plus_esm2_650M", "RandomForest"),
    ),
])


panel_b = pd.DataFrame([
    make_record(
        "MIFS\n4 descriptors",
        select_row(af, "MIFS_top4", "LogisticRegression"),
    ),
    make_record(
        "AlphaFold\nglobal",
        select_row(af, "AlphaFold_basic", "LogisticRegression"),
    ),
    make_record(
        "MIFS +\nAlphaFold",
        select_row(af, "MIFS_plus_AlphaFold", "LogisticRegression"),
    ),
])


def draw_panel(ax, df, panel_letter, title, subtitle):
    x = list(range(len(df)))
    ax.bar(
        x,
        df["roc"],
        yerr=df["roc_sd"],
        capsize=4,
        width=0.62,
        color="#2C7FB8",
        edgecolor="#2C7FB8",
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#222222"},
    )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_ylim(0.45, 0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=9.5)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)

    ax.text(
        -0.08, 1.13, panel_letter, transform=ax.transAxes,
        fontsize=15, fontweight="bold", va="top"
    )
    ax.text(
        0.00, 1.13, title, transform=ax.transAxes,
        fontsize=13, fontweight="bold", va="top"
    )
    ax.text(
        0.00, 1.065, subtitle, transform=ax.transAxes,
        fontsize=9.5, va="top"
    )

    for i, row in df.iterrows():
        ax.text(
            i, row["roc"] + row["roc_sd"] + 0.008, f'{row["roc"]:.3f}',
            ha="center", va="bottom", fontsize=9
        )
        ax.text(
            i, 0.462, f'PR-AUC\n{row["pr"]:.3f}',
            ha="center", va="bottom", fontsize=8, color="#111111"
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plt.rcParams.update({
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 2, figsize=(13.8, 5.3),
    gridspec_kw={"width_ratios": [1.60, 1.0]},
)

draw_panel(
    axes[0], panel_a, "A", "Protein-level representation benchmark",
    "Random Forest; 997 proteins, including 74 WT-lost positive proteins",
)
draw_panel(
    axes[1], panel_b, "B", "Global AlphaFold-derived descriptors",
    "Logistic Regression; matched subset of 996 proteins, including 74 positives",
)

plt.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.79, wspace=0.28)

for extension, kwargs in {
    "pdf": {},
    "png": {"dpi": 600},
    "svg": {},
}.items():
    fig.savefig(
        OUTDIR / f"Figure2_protein_level_benchmark_v3.{extension}",
        bbox_inches="tight",
        **kwargs,
    )

panel_a.assign(panel="A").to_csv(
    OUTDIR / "Figure2_panel_A_source_data.tsv", sep="\t", index=False
)
panel_b.assign(panel="B").to_csv(
    OUTDIR / "Figure2_panel_B_source_data.tsv", sep="\t", index=False
)

print(f"Saved Figure 2 v3 to: {OUTDIR}")
