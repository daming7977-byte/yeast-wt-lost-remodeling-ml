import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

INFILE = Path("ProteinLM_JCIM/10_AlphaFold/02_features/protein_feature_matrix_v2_plus_alphafold.tsv")
OUT = Path("ProteinLM_JCIM/10_AlphaFold/04_results/alphafold_feature_benchmark.tsv")

df = pd.read_csv(INFILE, sep="\t")

# Use only proteins with AlphaFold structures for structure-aware comparison
df = df[df["alphafold_available"] == 1].copy()

y = df["has_WT_lost"].astype(int)

mifs = [
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd",
    "loc_er"
]

af_features = [
    c for c in df.columns
    if c.startswith("af_") and c not in ["af_n_residues"]
]

feature_sets = {
    "MIFS_top4": mifs,
    "AlphaFold_basic": af_features,
    "MIFS_plus_AlphaFold": mifs + af_features
}

pos = int(y.sum())
neg = int((y == 0).sum())
scale_pos_weight = neg / pos

models = {
    "LogisticRegression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ]),
    "RandomForest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []

for fs_name, features in feature_sets.items():
    X = df[features].replace([np.inf, -np.inf], np.nan)

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "roc_auc": "roc_auc",
                "pr_auc": "average_precision"
            },
            n_jobs=-1
        )

        rows.append({
            "feature_set": fs_name,
            "model": model_name,
            "n_features": len(features),
            "n_proteins": len(df),
            "n_positive": int(y.sum()),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_sd": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_sd": scores["test_pr_auc"].std()
        })

out = pd.DataFrame(rows).sort_values(
    ["roc_auc_mean", "pr_auc_mean"],
    ascending=False
)

out.to_csv(OUT, sep="\t", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

print("Proteins with AlphaFold:", len(df))
print("Positive:", int(y.sum()))
print("AlphaFold features:", len(af_features))
print()
print(out)
print()
print("Saved:", OUT)
