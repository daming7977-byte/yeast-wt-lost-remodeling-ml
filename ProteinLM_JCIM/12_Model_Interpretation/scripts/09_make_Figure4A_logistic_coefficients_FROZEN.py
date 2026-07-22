import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

INFILE = Path("ProteinLM_JCIM/12_Model_Interpretation/01_results/mifs_logistic_coefficients.tsv")
OUTDIR = Path("ProteinLM_JCIM/12_Model_Interpretation/02_figures/Figure4A_FROZEN")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INFILE, sep="\t")

labels = {
    "n_tmd": "TMD count",
    "tmd_negative_fraction_std": "Variability of TMD negative charge",
    "mean_tmd_gap": "Mean inter-TMD distance",
    "loc_er": "ER-localized protein"
}

order = [
    "loc_er",
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd"
]

df["label"] = df["feature"].map(labels)
df["plot_order"] = df["feature"].apply(lambda x: order.index(x))
df = df.sort_values("plot_order")

colors = [
    "#3B6FB6" if c > 0 else "#C95A5A"
    for c in df["standardized_coefficient"]
]

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

fig, ax = plt.subplots(figsize=(5.35, 3.20))

ax.barh(
    df["label"],
    df["standardized_coefficient"],
    color=colors,
    edgecolor="none",
    height=0.78
)

ax.axvline(0, color="#777777", linewidth=0.8, zorder=0)

ax.set_xlabel("Standardized coefficient (β)", labelpad=5)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(1.0)
ax.spines["bottom"].set_linewidth(1.0)

ax.tick_params(axis="both", width=1, length=4)

# Symmetric x-axis for centered zero line
ax.set_xlim(-0.40, 0.40)

ax.text(
    -0.065,
    1.025,
    "A",
    transform=ax.transAxes,
    fontsize=9,
    fontweight="bold",
    ha="left",
    va="bottom"
)

plt.tight_layout()

plt.savefig(OUTDIR / "Figure4A_MIFS_logistic_coefficients_FROZEN.pdf", dpi=600, bbox_inches="tight")
plt.savefig(OUTDIR / "Figure4A_MIFS_logistic_coefficients_FROZEN.png", dpi=600, bbox_inches="tight")

plt.close()

print("Saved:")
print(OUTDIR / "Figure4A_MIFS_logistic_coefficients_FROZEN.pdf")
print(OUTDIR / "Figure4A_MIFS_logistic_coefficients_FROZEN.png")
