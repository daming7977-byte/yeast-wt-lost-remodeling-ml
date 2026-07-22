import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")
OUTDIR = Path("ProteinLM_JCIM/09_feature_selection/results_nested_mifs")
OUTDIR.mkdir(parents=True, exist_ok=True)

PER_FOLD_OUT = OUTDIR / "nested_mifs_per_fold_results.tsv"
FREQ_OUT = OUTDIR / "nested_mifs_feature_selection_frequency.tsv"
SUMMARY_OUT = OUTDIR / "nested_mifs_summary.tsv"

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

X_all = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y_all = df["has_WT_lost"].astype(int).values

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
all_selected = []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_all, y_all), start=1):
    X_train_raw = X_all.iloc[train_idx].copy()
    X_test_raw = X_all.iloc[test_idx].copy()
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    train_medians = X_train_raw.median()
    X_train_xgb = X_train_raw.fillna(train_medians)
    X_test_xgb = X_test_raw.fillna(train_medians)

    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / pos

    ranker = XGBClassifier(
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

    ranker.fit(X_train_xgb, y_train)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": ranker.feature_importances_
    }).sort_values("importance_gain", ascending=False)

    top5 = importance.head(5)["feature"].tolist()

    mifs_fold = [f for f in top5 if f != "is_transporter_like"]

    # If transporter is not in top5, keep top4 among top5 for comparable compact model.
    if len(mifs_fold) > 4:
        mifs_fold = mifs_fold[:4]

    # If removing transporter leaves fewer than 4 features, add next-ranked non-transporter features.
    if len(mifs_fold) < 4:
        for f in importance["feature"].tolist():
            if f != "is_transporter_like" and f not in mifs_fold:
                mifs_fold.append(f)
            if len(mifs_fold) == 4:
                break

    all_selected.extend(mifs_fold)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    model.fit(X_train_raw[mifs_fold], y_train)
    y_score = model.predict_proba(X_test_raw[mifs_fold])[:, 1]

    roc = roc_auc_score(y_test, y_score)
    pr = average_precision_score(y_test, y_score)

    rows.append({
        "fold": fold,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "n_positive_train": int(y_train.sum()),
        "n_positive_test": int(y_test.sum()),
        "top5_from_training_fold": ",".join(top5),
        "mifs_like_features_after_transporter_removal": ",".join(mifs_fold),
        "roc_auc": roc,
        "pr_auc": pr
    })

per_fold = pd.DataFrame(rows)
per_fold.to_csv(PER_FOLD_OUT, sep="\t", index=False)

freq = (
    pd.DataFrame(Counter(all_selected).items(), columns=["feature", "selection_count"])
    .sort_values(["selection_count", "feature"], ascending=[False, True])
)
freq["selection_frequency"] = freq["selection_count"] / outer_cv.get_n_splits()
freq.to_csv(FREQ_OUT, sep="\t", index=False)

summary = pd.DataFrame([{
    "analysis": "nested_foldwise_xgboost_ranking_then_logistic_MIFS_like_top4",
    "n_folds": outer_cv.get_n_splits(),
    "roc_auc_mean": per_fold["roc_auc"].mean(),
    "roc_auc_sd": per_fold["roc_auc"].std(ddof=0),
    "pr_auc_mean": per_fold["pr_auc"].mean(),
    "pr_auc_sd": per_fold["pr_auc"].std(ddof=0)
}])
summary.to_csv(SUMMARY_OUT, sep="\t", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

print("Nested fold-wise MIFS validation")
print("=" * 80)
print()
print("Per-fold results:")
print(per_fold)
print()
print("Feature selection frequency:")
print(freq)
print()
print("Summary:")
print(summary)
print()
print("Saved:")
print(PER_FOLD_OUT)
print(FREQ_OUT)
print(SUMMARY_OUT)
