import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
SEL_DIR = Path("ProteinLM_JCIM/09_feature_selection/results")
OUT = SEL_DIR / "topN_feature_retraining_results.tsv"

df = pd.read_csv(FEATURES, sep="\t")
y = df["has_WT_lost"].astype(int)

pos = int(y.sum())
neg = int((y == 0).sum())
scale_pos_weight = neg / pos

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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

rows = []

for n in [30, 20, 15, 10, 5]:
    feature_file = SEL_DIR / f"top{n}_features.txt"
    features = [
        line.strip()
        for line in open(feature_file)
        if line.strip()
    ]

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
            "feature_set": f"top{n}",
            "n_features": n,
            "model": model_name,
            "roc_auc_mean": scores["test_roc_auc"].mean(),
            "roc_auc_sd": scores["test_roc_auc"].std(),
            "pr_auc_mean": scores["test_pr_auc"].mean(),
            "pr_auc_sd": scores["test_pr_auc"].std(),
            "features": ",".join(features)
        })

out = pd.DataFrame(rows).sort_values(
    ["roc_auc_mean", "pr_auc_mean"],
    ascending=False
)

out.to_csv(OUT, sep="\t", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print(out[[
    "feature_set",
    "model",
    "n_features",
    "roc_auc_mean",
    "roc_auc_sd",
    "pr_auc_mean",
    "pr_auc_sd"
]])

print()
print("Saved:", OUT)
