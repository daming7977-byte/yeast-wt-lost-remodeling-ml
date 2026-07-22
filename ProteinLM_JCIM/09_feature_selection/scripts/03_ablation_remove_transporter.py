import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

FEATURES = Path("ProteinLM_JCIM/03_features/protein_feature_matrix_v2_enriched.tsv")

df = pd.read_csv(FEATURES, sep="\t")

y = df["has_WT_lost"].astype(int)

features = [
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd",
    "loc_er"
]

X = df[features]

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

scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring={
        "roc_auc":"roc_auc",
        "pr_auc":"average_precision"
    },
    n_jobs=-1
)

print("="*60)
print("Ablation study")
print("="*60)

print()

print("Features used:")

for f in features:
    print("-",f)

print()

print("ROC-AUC : %.3f ± %.3f"%(scores["test_roc_auc"].mean(),
                               scores["test_roc_auc"].std()))

print("PR-AUC  : %.3f ± %.3f"%(scores["test_pr_auc"].mean(),
                               scores["test_pr_auc"].std()))
