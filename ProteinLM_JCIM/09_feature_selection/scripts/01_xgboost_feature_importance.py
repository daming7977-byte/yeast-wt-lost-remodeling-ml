import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")

OUTDIR = Path("ProteinLM_JCIM/09_feature_selection/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

IMPORTANCE_OUT = OUTDIR / "xgboost_feature_importance.tsv"
CV_OUT = OUTDIR / "xgboost_cv_performance.tsv"

df = pd.read_csv(FEATURES, sep="\t")

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

X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(df[feature_cols].median())
y = df["has_WT_lost"].astype(int)

print("Proteins:", len(df))
print("Positive proteins:", int(y.sum()))
print("Background proteins:", int((y == 0).sum()))
print("Feature count:", len(feature_cols))

pos = int(y.sum())
neg = int((y == 0).sum())
scale_pos_weight = neg / pos

model = XGBClassifier(
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

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring={
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    },
    n_jobs=-1,
    return_train_score=False
)

cv_summary = pd.DataFrame([{
    "model": "XGBoost",
    "feature_set": "protein_feature_matrix_v2_enriched",
    "n_features": len(feature_cols),
    "roc_auc_mean": scores["test_roc_auc"].mean(),
    "roc_auc_sd": scores["test_roc_auc"].std(),
    "pr_auc_mean": scores["test_pr_auc"].mean(),
    "pr_auc_sd": scores["test_pr_auc"].std()
}])

cv_summary.to_csv(CV_OUT, sep="\t", index=False)

model.fit(X, y)

importance = pd.DataFrame({
    "feature": feature_cols,
    "importance_gain": model.feature_importances_
}).sort_values("importance_gain", ascending=False)

importance.to_csv(IMPORTANCE_OUT, sep="\t", index=False)

for n in [30, 20, 15, 10, 5]:
    out = OUTDIR / f"top{n}_features.txt"
    importance.head(n)["feature"].to_csv(out, index=False, header=False)

print()
print("CV performance:")
print(cv_summary)

print()
print("Top 20 features:")
print(importance.head(20))

print()
print("Saved:")
print(CV_OUT)
print(IMPORTANCE_OUT)
print("top30/top20/top15/top10/top5 feature lists")
