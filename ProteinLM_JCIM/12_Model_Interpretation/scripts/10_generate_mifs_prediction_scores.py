import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

IN = Path("ProteinLM_JCIM/12_Model_Interpretation/01_results/mifs_model_input.tsv")
OUT = Path("ProteinLM_JCIM/12_Model_Interpretation/01_results/mifs_prediction_scores.tsv")

df = pd.read_csv(IN, sep="\t")

features = [
    "n_tmd",
    "tmd_negative_fraction_std",
    "mean_tmd_gap",
    "loc_er",
]

y = df["has_WT_lost"].astype(int).values
X = df[features].replace([np.inf, -np.inf], np.nan)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear"
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

prob = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba",
    n_jobs=-1
)[:, 1]

out = df.copy()
out["mifs_prediction_probability"] = prob
out["mifs_prediction_rank"] = out["mifs_prediction_probability"].rank(
    ascending=False,
    method="first"
).astype(int)

out = out.sort_values("mifs_prediction_probability", ascending=False)
out.to_csv(OUT, sep="\t", index=False)

print("Saved:", OUT)
print("Rows:", len(out))
print("Positive:", int(out["has_WT_lost"].sum()))
print("Background:", int((out["has_WT_lost"] == 0).sum()))
print()
print("ROC-AUC:", roc_auc_score(y, prob))
print("PR-AUC:", average_precision_score(y, prob))
print()
print("Top 10 prediction scores:")
print(out.head(10).to_string(index=False))
