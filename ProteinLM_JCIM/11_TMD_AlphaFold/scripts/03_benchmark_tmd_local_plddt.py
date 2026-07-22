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

MASTER = Path("master_feature_table_v2.tsv")
LOCAL = Path("ProteinLM_JCIM/11_TMD_AlphaFold/02_features/tmd_local_plddt_features.tsv")
OUT = Path("ProteinLM_JCIM/11_TMD_AlphaFold/03_results/tmd_local_plddt_benchmark.tsv")

master = pd.read_csv(MASTER, sep="\t")
local = pd.read_csv(LOCAL, sep="\t")

df = master.merge(
    local.drop(columns=["gene", "tmd_index", "tmd_start_aa", "tmd_end_aa", "classification", "label_WT_lost"], errors="ignore"),
    on="key",
    how="inner"
)

df = df[df["local_plddt_available"] == 1].copy()

y = df["label_WT_lost"].astype(int)

traditional_features = [
    "protein_length",
    "tmd_index",
    "tmd_length_aa",
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

local_features = [
    c for c in df.columns
    if (
        c.startswith("tmd_") or
        c.startswith("up20_") or
        c.startswith("down20_") or
        c.startswith("window20_")
    )
    and ("plddt" in c or "n_residues" in c)
]

delta_features = [
    "tmd_vs_window20_plddt_delta",
    "tmd_vs_up20_plddt_delta",
    "tmd_vs_down20_plddt_delta"
]

delta_features = [c for c in delta_features if c in df.columns]

local_features = sorted(set(local_features + delta_features))

feature_sets = {
    "traditional_TMD": traditional_features,
    "local_AF_plddt": local_features,
    "traditional_plus_local_AF": traditional_features + local_features,
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
            "analysis_level": "TMD",
            "feature_set": fs_name,
            "model": model_name,
            "n_features": len(features),
            "n_TMDs": len(df),
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

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, sep="\t", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print("TMDs:", len(df))
print("Positive:", int(y.sum()))
print("Traditional features:", len(traditional_features))
print("Local AF pLDDT features:", len(local_features))
print()
print(out)
print()
print("Saved:", OUT)
