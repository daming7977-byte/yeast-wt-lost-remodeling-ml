import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

MASTER = Path("master_feature_table_v2.tsv")
ESM_TMD = Path("ProteinLM/features/esm2_t12_35M_tmd_embeddings.tsv")
ESM_FULL = Path("ProteinLM_JCIM/01_embeddings/esm2_t12_35M_full_protein_embeddings.tsv")
OUT = Path("ProteinLM_JCIM/05_results/esm2_full_vs_tmd_benchmark.tsv")

master = pd.read_csv(MASTER, sep="\t")
esm_tmd = pd.read_csv(ESM_TMD, sep="\t")
esm_full = pd.read_csv(ESM_FULL, sep="\t")

df = master.merge(
    esm_tmd.drop(columns=["classification", "label_WT_lost"]),
    on=["gene", "key", "tmd_index"],
    how="inner"
)

df = df.merge(
    esm_full,
    on="gene",
    how="inner",
    suffixes=("", "_fullprotein")
)

print("Merged rows:", len(df))
print("Positive labels:", int(df["label_WT_lost"].sum()))

y = df["label_WT_lost"].astype(int).values

traditional_features = [
    "protein_length",
    "tmd_index",
    "tmd_start_aa",
    "tmd_end_aa",
    "tmd_length_aa",
    "tmd_length_nt",
    "relative_tmd_position",
    "tmd_count_per_gene",
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
]

traditional_features = [c for c in traditional_features if c in df.columns]
esm_tmd_features = [c for c in df.columns if c.startswith("esm2_tmd_")]
esm_full_features = [c for c in df.columns if c.startswith("esm2_full_")]

feature_sets = {
    "traditional": traditional_features,
    "esm2_tmd": esm_tmd_features,
    "esm2_full_protein": esm_full_features,
    "traditional_plus_esm2_full": traditional_features + esm_full_features,
    "traditional_plus_esm2_tmd": traditional_features + esm_tmd_features,
    "traditional_plus_full_plus_tmd": traditional_features + esm_full_features + esm_tmd_features,
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
            "feature_set": fs_name,
            "model": model_name,
            "n_features": len(features),
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_sd": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_sd": scores["test_pr_auc"].std()
        })

out = pd.DataFrame(rows).sort_values(["roc_auc_mean", "pr_auc_mean"], ascending=False)
out.to_csv(OUT, sep="\t", index=False)

print(out)
print("Saved:", OUT)
