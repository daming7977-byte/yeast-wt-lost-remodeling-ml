import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
OUT = Path("ProteinLM_JCIM/05_results/protein_features_v2_enriched_benchmark.tsv")

df = pd.read_csv(FEATURES, sep="\t")

print("Proteins:", len(df))
print("Positive proteins:", int(df["has_WT_lost"].sum()))
print("Background proteins:", int((df["has_WT_lost"] == 0).sum()))

y = df["has_WT_lost"].astype(int).values

exclude = {
    "gene",
    "protein_name",
    "entry_name",
    "subcellular_location",
    "n_WT_lost",
    "has_WT_lost",
    "n_any_eS7_event",
    "has_any_eS7_event",
    "n_4KR_specific",
    "has_4KR_specific",
    "n_shared",
    "has_shared",
}

feature_cols = [
    c for c in df.columns
    if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
]

print("Feature count:", len(feature_cols))

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
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

rows = []

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).values

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
        "feature_set": "protein_feature_matrix_v2_enriched",
        "model": model_name,
        "n_features": len(feature_cols),
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

print()
print("Features used:")
for c in feature_cols:
    print(c)
