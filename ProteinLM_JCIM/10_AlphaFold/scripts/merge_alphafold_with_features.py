import pandas as pd
from pathlib import Path

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
AF = Path("ProteinLM_JCIM/10_AlphaFold/02_features/alphafold_basic_features.tsv")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/02_features/protein_feature_matrix_v2_plus_alphafold.tsv")

features = pd.read_csv(FEATURES, sep="\t")
af = pd.read_csv(AF, sep="\t")

out = features.merge(af, on="gene", how="left")

out.to_csv(OUT, sep="\t", index=False)

print("Original feature matrix:", features.shape)
print("AlphaFold features:", af.shape)
print("Merged matrix:", out.shape)
print("AlphaFold available:", int(out["alphafold_available"].sum()))
print("Missing AlphaFold:", int((out["alphafold_available"] == 0).sum()))
print("Saved:", OUT)
