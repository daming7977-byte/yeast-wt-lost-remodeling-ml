import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTDIR = Path(__file__).resolve().parent

ESM_FILE = ROOT / "ProteinLM_JCIM/05_results/esm2_full_vs_tmd_GroupKFold_benchmark.tsv"
AF_FILE = ROOT / "ProteinLM_JCIM/11_TMD_AlphaFold/03_results/tmd_local_plddt_StratifiedGroupKFold_benchmark.tsv"

esm = pd.read_csv(ESM_FILE, sep="\t")
af = pd.read_csv(AF_FILE, sep="\t")


def select_row(df, feature_set, model):
    row = df[(df["feature_set"] == feature_set) & (df["model"] == model)]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {feature_set}/{model}; found {len(row)}")
    return row.iloc[0]


panel_a_specs = [
    ("Traditional-18\ndescriptors", "traditional", "RandomForest"),
    ("TMD-centered\nESM2", "esm2_tmd", "RandomForest"),
    ("Full-protein\nESM2", "esm2_full_protein", "RandomForest"),
    ("Traditional-18 +\nfull + TMD ESM2", "traditional_plus_full_plus_tmd", "RandomForest"),
]

panel_b_specs = [
    ("Traditional-15\nTMD descriptors", "traditional_TMD", "XGBoost"),
    ("Local AlphaFold\npLDDT", "local_AF_plddt", "XGBoost"),
    ("Traditional-15 +\nlocal AlphaFold", "traditional_plus_local_AF", "XGBoost"),
]


def build_panel(df, specs):
    rows = []
    for label, feature_set, model in specs:
        source = select_row(df, feature_set, model)
        rows.append({
            "label": label,
            "feature_set": feature_set,
            "model": model,
            "n_features": int(source["n_features"]),
            "roc": float(source["roc_auc_mean"]),
            "roc_sd": float(source["roc_auc_sd"]),
            "pr": float(source["pr_auc_mean"]),
            "pr_sd": float(source["pr_auc_sd"]),
        })
    return pd.DataFrame(rows)


panel_a = build_panel(esm, panel_a_specs)
panel_b = build_panel(af, panel_b_specs)


def draw_panel(ax, df, panel_letter, title, subtitle, color):
    x = list(range(len(df)))
    ax.bar(
        x, df["roc"], yerr=df["roc_sd"], capsize=4, width=0.62,
        color=color, edgecolor=color,
        error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "#222222"},
    )
    ax.axhline(0.5, color="#6B7280", linestyle="--", linewidth=1)
    ax.set_ylim(0.43, 0.70)
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
        fontsize=9.3, va="top"
    )

    for i, row in df.iterrows():
        ax.text(
            i, row["roc"] + row["roc_sd"] + 0.007, f'{row["roc"]:.3f}',
            ha="center", va="bottom", fontsize=9
        )
        ax.text(
            i, 0.442, f'PR-AUC\n{row["pr"]:.3f}',
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
    1, 2, figsize=(13.3, 5.3),
    gridspec_kw={"width_ratios": [1.2, 1.0]},
)

draw_panel(
    axes[0], panel_a, "A", "Sequence representation scale",
    "Random Forest; 5,028 TMDs from 997 genes; GroupKFold by gene",
    "#2C7FB8",
)
draw_panel(
    axes[1], panel_b, "B", "Local AlphaFold confidence profiles",
    "XGBoost; 5,027 TMDs from 996 genes; StratifiedGroupKFold by gene",
    "#D97706",
)

plt.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.79, wspace=0.28)

for extension, kwargs in {
    "pdf": {},
    "png": {"dpi": 600},
    "svg": {},
}.items():
    fig.savefig(
        OUTDIR / f"Figure3_tmd_representation_scale_v1.{extension}",
        bbox_inches="tight",
        **kwargs,
    )

panel_a.assign(panel="A").to_csv(
    OUTDIR / "Figure3_panel_A_source_data.tsv", sep="\t", index=False
)
panel_b.assign(panel="B").to_csv(
    OUTDIR / "Figure3_panel_B_source_data.tsv", sep="\t", index=False
)

print(f"Saved Figure 3 v1 to: {OUTDIR}")
