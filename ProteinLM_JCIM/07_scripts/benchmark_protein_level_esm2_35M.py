import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

PROTEIN = Path("ProteinLM_JCIM/00_data/protein_master_table.tsv")
ESM_FULL = Path("ProteinLM_JCIM/01_embeddings/esm2_t12_35M_full_protein_embeddings.tsv")
OUT = Path("ProteinLM_JCIM/05_results/protein_level_esm2_35M_benchmark.tsv")

protein = pd.read_csv(PROTEIN, sep="\t")
esm = pd.read_csv(ESM_FULL, sep="\t")

df = protein.merge(esm, on="gene", how="inner", suffixes=("", "_esm"))

print("Merged proteins:", len(df))
print("Positive proteins:", int(df["has_WT_lost"].sum()))
print("Background proteins:", int((df["has_WT_lost"] == 0).sum()))

y = df["has_WT_lost"].astype(int).values

traditional_features = [
    "protein_length",
    "n_tmd",
    "n_any_eS7_event",
    "n_4KR_specific",
    "n_shared",
    "single_pass",
    "multi_pass",
    "loc_er",
    "loc_golgi",
    "loc_cell_membrane",
    "loc_mitochondrion",
    "loc_vacuole",
    "loc_nucleus",
    "is_transporter_like",
    "is_transferase_like",
    "mean_tmd_length_aa",
    "mean_relative_tmd_position",
    "min_relative_tmd_position",
    "max_relative_tmd_position",
]

# Avoid leakage from related eS7 labels in the main baseline.
safe_traditional_features = [
    c for c in traditional_features
    if c in df.columns and c not in [
        "n_any_eS7_event",
        "n_4KR_specific",
        "n_shared"
    ]
]

esm_features = [c for c in df.columns if c.startswith("esm2_full_")]

feature_sets = {
    "traditional_safe": safe_traditional_features,
    "esm2_full_protein": esm_features,
    "traditional_plus_esm2_full": safe_traditional_features + esm_features,
}

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
            n_estimators=500,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

rows = []

for fs_name, features in feature_sets.items():
    X = df[features].replace([np.inf, -np.inf], np.nan).values

    for model_name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        rows.append({
            "analysis_level": "protein",
            "feature_set": fs_name,
            "model": model_name,
            "n_features": len(features),
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
pd.set_option("display.width", 200)

print(out)
print("Saved:", OUT)
