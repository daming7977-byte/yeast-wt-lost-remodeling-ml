import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib.pyplot as plt

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
OUTDIR = Path("ProteinLM_JCIM/09_feature_selection/results/top4_feature_analysis")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(FEATURES, sep="\t")

label_col = "has_WT_lost"

features = [
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd",
    "loc_er"
]

rows = []

for f in features:
    pos = df[df[label_col] == 1][f]
    neg = df[df[label_col] == 0][f]

    if set(df[f].dropna().unique()).issubset({0, 1}):
        a = int(((df[label_col] == 1) & (df[f] == 1)).sum())
        b = int(((df[label_col] == 1) & (df[f] == 0)).sum())
        c = int(((df[label_col] == 0) & (df[f] == 1)).sum())
        d = int(((df[label_col] == 0) & (df[f] == 0)).sum())
        OR, p = fisher_exact([[a, b], [c, d]])
        test = "Fisher_exact"
    else:
        stat, p = mannwhitneyu(pos, neg, alternative="two-sided")
        OR = np.nan
        test = "Mann_Whitney_U"

    rows.append({
        "feature": f,
        "positive_mean": pos.mean(),
        "positive_median": pos.median(),
        "background_mean": neg.mean(),
        "background_median": neg.median(),
        "test": test,
        "odds_ratio_if_binary": OR,
        "p_value": p
    })

stats = pd.DataFrame(rows)
stats.to_csv(OUTDIR / "top4_feature_statistics.tsv", sep="\t", index=False)

print(stats)

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

for f in features:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    pos = df[df[label_col] == 1][f]
    neg = df[df[label_col] == 0][f]

    if set(df[f].dropna().unique()).issubset({0, 1}):
        values = [
            pos.mean() * 100,
            neg.mean() * 100
        ]
        ax.bar(["WT-lost", "Background"], values)
        ax.set_ylabel("Percentage of proteins (%)")
    else:
        ax.boxplot(
            [pos, neg],
            labels=["WT-lost", "Background"],
            showfliers=False
        )
        ax.set_ylabel(f)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    safe = f.replace("/", "_")
    plt.savefig(OUTDIR / f"{safe}.pdf", dpi=600, bbox_inches="tight")
    plt.savefig(OUTDIR / f"{safe}.png", dpi=600, bbox_inches="tight")
    plt.close()

print()
print("Saved:")
print(OUTDIR / "top4_feature_statistics.tsv")
print("Individual PDF/PNG plots for each Top4 feature")
