import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

INFILE = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
OUT = Path("ProteinLM_JCIM/12_Model_Interpretation/01_results/mifs_logistic_coefficients.tsv")

df = pd.read_csv(INFILE, sep="\t")

features = [
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd",
    "loc_er"
]

X = df[features]
y = df["has_WT_lost"].astype(int)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

model.fit(X, y)

coef = model.named_steps["clf"].coef_[0]

out = pd.DataFrame({
    "feature": features,
    "standardized_coefficient": coef,
    "direction": ["positive" if c > 0 else "negative" for c in coef],
    "exp_coefficient": np.exp(coef)
}).sort_values("standardized_coefficient", ascending=False)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, sep="\t", index=False)

print(out)
print()
print("Saved:", OUT)
