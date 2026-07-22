#!/usr/bin/env python3
"""Repeated, split-matched validation of the primary protein-level analyses.

This script is intentionally independent of the earlier benchmark scripts. It
creates a persistent split manifest, applies all preprocessing within training
folds, repeats XGBoost feature ranking inside each training fold for the
adaptive compact model, and saves sample-level out-of-fold probabilities.
"""

from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ROOT = Path("ProteinLM_JCIM")
MASTER_FILE = ROOT / "00_data/protein_master_table.tsv"
ENGINEERED_FILE = ROOT / "03_features/protein_feature_matrix_v2_enriched.tsv"
ESM35_FILE = ROOT / "01_embeddings/esm2_t12_35M_full_protein_embeddings.tsv"
ESM650_FILE = ROOT / "01_embeddings/esm2_650M_full_protein_embeddings.tsv"
ALPHAFOLD_FILE = (
    ROOT / "10_AlphaFold/02_features/protein_feature_matrix_v2_plus_alphafold.tsv"
)
OUT_ROOT = ROOT / "13_Robustness_Validation/results"

LABEL = "has_WT_lost"
MODEL_RANDOM_STATE = 42

EXCLUDE_ENGINEERED = {
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

TRADITIONAL16 = [
    "protein_length",
    "n_tmd",
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

MIFS4 = [
    "mean_tmd_gap",
    "tmd_negative_fraction_std",
    "n_tmd",
    "loc_er",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--tag", default="repeated20x5")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def load_inputs():
    master = pd.read_csv(MASTER_FILE, sep="\t")
    engineered = pd.read_csv(ENGINEERED_FILE, sep="\t")
    esm35 = pd.read_csv(ESM35_FILE, sep="\t")
    esm650 = pd.read_csv(ESM650_FILE, sep="\t")
    alphafold = pd.read_csv(ALPHAFOLD_FILE, sep="\t")

    for name, frame in {
        "master": master,
        "engineered": engineered,
        "esm35": esm35,
        "esm650": esm650,
        "alphafold": alphafold,
    }.items():
        require_columns(frame, ["gene"], name)
        if frame["gene"].duplicated().any():
            raise ValueError(f"Duplicate genes in {name}")

    require_columns(master, [LABEL], "master")
    require_columns(engineered, [LABEL], "engineered")
    require_columns(alphafold, [LABEL, "alphafold_available"], "alphafold")

    base = engineered.copy()
    base = base.merge(
        esm35.drop(columns=["protein_length"], errors="ignore"),
        on="gene",
        how="inner",
        validate="one_to_one",
    )
    base = base.merge(
        esm650.drop(columns=["protein_length_from_fasta"], errors="ignore"),
        on="gene",
        how="inner",
        validate="one_to_one",
    )
    base = base.sort_values("gene").reset_index(drop=True)

    if len(base) != 997:
        raise ValueError(f"Expected 997 complete protein rows, found {len(base)}")
    if int(base[LABEL].sum()) != 74:
        raise ValueError("Expected 74 positive proteins")

    master_labels = master.set_index("gene")[LABEL].astype(int)
    base_labels = base.set_index("gene")[LABEL].astype(int)
    if not master_labels.sort_index().equals(base_labels.sort_index()):
        raise ValueError("Label mismatch between master and merged representation table")

    af = alphafold.loc[alphafold["alphafold_available"] == 1].copy()
    af = af.sort_values("gene").reset_index(drop=True)
    if len(af) != 996 or int(af[LABEL].sum()) != 74:
        raise ValueError("Expected 996 AlphaFold-matched proteins with 74 positives")

    engineered_features = [
        column
        for column in engineered.columns
        if column not in EXCLUDE_ENGINEERED
        and pd.api.types.is_numeric_dtype(engineered[column])
    ]
    esm35_features = [column for column in base.columns if column.startswith("esm2_full_")]
    esm650_features = [column for column in base.columns if column.startswith("esm2_650M_")]
    af_features = [
        column
        for column in af.columns
        if column.startswith("af_") and column != "af_n_residues"
    ]

    require_columns(base, engineered_features + TRADITIONAL16 + MIFS4, "base")
    require_columns(af, af_features + MIFS4, "alphafold matched")

    expected_counts = {
        "engineered": 46,
        "traditional": 16,
        "esm35": 480,
        "esm650": 1280,
        "alphafold": 13,
    }
    observed_counts = {
        "engineered": len(engineered_features),
        "traditional": len(TRADITIONAL16),
        "esm35": len(esm35_features),
        "esm650": len(esm650_features),
        "alphafold": len(af_features),
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected feature counts: observed={observed_counts}; expected={expected_counts}"
        )

    feature_sets = {
        "engineered46_rf": engineered_features,
        "esm2_35m_rf": esm35_features,
        "traditional16_plus_esm2_35m_rf": TRADITIONAL16 + esm35_features,
        "esm2_650m_rf": esm650_features,
        "traditional16_plus_esm2_650m_rf": TRADITIONAL16 + esm650_features,
        "mifs4_fixed_lr": MIFS4,
    }
    af_feature_sets = {
        "mifs4_lr_afmatched": MIFS4,
        "alphafold13_lr": af_features,
        "mifs_plus_alphafold17_lr": MIFS4 + af_features,
    }
    return base, af, engineered_features, feature_sets, af_feature_sets


def make_rf(n_trees: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_trees,
                    class_weight="balanced",
                    random_state=MODEL_RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_lr() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=MODEL_RANDOM_STATE,
                ),
            ),
        ]
    )


def score_predictions(y_true, y_score):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def fit_predict(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def feature_family(feature: str) -> str:
    if feature.startswith("loc_"):
        return "localization"
    if feature.startswith("is_"):
        return "functional_annotation"
    if any(
        token in feature
        for token in [
            "hydrophobic",
            "aromatic",
            "positive",
            "negative",
            "charge",
            "_KR_",
            "_DE_",
            "KR_fraction",
            "DE_fraction",
        ]
    ):
        return "physicochemical_composition"
    if any(token in feature for token in ["gap", "position", "tmd_length"]):
        return "tmd_organization_position"
    if feature in {"n_tmd", "single_pass", "multi_pass", "tmd_density"}:
        return "membrane_topology"
    if feature.startswith("protein_"):
        return "whole_protein_property"
    return "other_architecture"


def summarize_outputs(
    output_dir: Path,
    fold_metrics: pd.DataFrame,
    oof: pd.DataFrame,
    selections: pd.DataFrame,
    repeats: int,
    folds: int,
):
    repeat_metrics = []
    for (repeat, model), group in oof.groupby(["repeat", "model"], sort=True):
        metrics = score_predictions(group["y_true"], group["y_score"])
        repeat_metrics.append(
            {
                "repeat": repeat,
                "model": model,
                "n_predictions": len(group),
                "n_positive": int(group["y_true"].sum()),
                **metrics,
            }
        )
    repeat_metrics = pd.DataFrame(repeat_metrics)

    summaries = []
    for model, group in repeat_metrics.groupby("model", sort=True):
        sample_group = oof.loc[oof["model"] == model]
        summaries.append(
            {
                "model": model,
                "n_repeats": repeats,
                "n_folds": folds,
                "n_unique_proteins": sample_group["gene"].nunique(),
                "positive_prevalence": sample_group.drop_duplicates("gene")[
                    "y_true"
                ].mean(),
                "roc_auc_mean_across_repeats": group["roc_auc"].mean(),
                "roc_auc_sd_across_repeats": group["roc_auc"].std(ddof=1),
                "roc_auc_q025": group["roc_auc"].quantile(0.025),
                "roc_auc_q975": group["roc_auc"].quantile(0.975),
                "pr_auc_mean_across_repeats": group["pr_auc"].mean(),
                "pr_auc_sd_across_repeats": group["pr_auc"].std(ddof=1),
                "pr_auc_q025": group["pr_auc"].quantile(0.025),
                "pr_auc_q975": group["pr_auc"].quantile(0.975),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(
        "roc_auc_mean_across_repeats", ascending=False
    )

    averaged_oof = (
        oof.groupby(["gene", "y_true", "model"], as_index=False)["y_score"]
        .mean()
        .rename(columns={"y_score": "mean_repeated_oof_score"})
    )

    selected_only = selections.loc[selections["selected_compact"] == 1].copy()
    selection_summary = (
        selected_only.groupby(["feature", "family"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    selection_summary["selection_frequency"] = selection_summary[
        "selection_count"
    ] / (repeats * folds)
    selection_summary = selection_summary.sort_values(
        ["selection_count", "feature"], ascending=[False, True]
    )

    family_summary = (
        selected_only.groupby("family", as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    family_summary["mean_features_per_fold"] = family_summary[
        "selection_count"
    ] / (repeats * folds)
    family_summary["share_of_all_selected_slots"] = family_summary[
        "selection_count"
    ] / (repeats * folds * 4)
    family_summary = family_summary.sort_values("selection_count", ascending=False)

    sets = [
        set(group["feature"])
        for _, group in selected_only.groupby(["repeat", "fold"], sort=True)
    ]
    jaccard = [len(a & b) / len(a | b) for a, b in combinations(sets, 2)]
    jaccard_summary = pd.DataFrame(
        [
            {
                "n_compact_models": len(sets),
                "n_pairwise_comparisons": len(jaccard),
                "jaccard_mean": float(np.mean(jaccard)),
                "jaccard_median": float(np.median(jaccard)),
                "jaccard_q025": float(np.quantile(jaccard, 0.025)),
                "jaccard_q975": float(np.quantile(jaccard, 0.975)),
            }
        ]
    )

    fold_metrics.to_csv(output_dir / "repeated_cv_per_fold_metrics.tsv", sep="\t", index=False)
    repeat_metrics.to_csv(
        output_dir / "repeated_cv_per_repeat_metrics.tsv", sep="\t", index=False
    )
    summary.to_csv(output_dir / "repeated_cv_summary.tsv", sep="\t", index=False)
    oof.to_csv(output_dir / "repeated_cv_oof_predictions.tsv", sep="\t", index=False)
    averaged_oof.to_csv(
        output_dir / "repeated_cv_mean_oof_predictions.tsv", sep="\t", index=False
    )
    selections.to_csv(
        output_dir / "repeated_foldwise_feature_rankings.tsv", sep="\t", index=False
    )
    selection_summary.to_csv(
        output_dir / "repeated_feature_selection_frequency.tsv", sep="\t", index=False
    )
    family_summary.to_csv(
        output_dir / "repeated_feature_family_frequency.tsv", sep="\t", index=False
    )
    jaccard_summary.to_csv(
        output_dir / "repeated_feature_jaccard_summary.tsv", sep="\t", index=False
    )
    return summary, selection_summary, family_summary, jaccard_summary


def main():
    args = parse_args()
    if args.repeats < 1 or args.folds < 2:
        raise ValueError("repeats must be >=1 and folds must be >=2")

    output_dir = OUT_ROOT / args.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    base, af, engineered_features, feature_sets, af_feature_sets = load_inputs()
    base_index = {gene: index for index, gene in enumerate(base["gene"])}
    af_index = {gene: index for index, gene in enumerate(af["gene"])}

    y = base[LABEL].astype(int).to_numpy()
    genes = base["gene"].astype(str).to_numpy()

    split_rows = []
    prediction_rows = []
    metric_rows = []
    selection_rows = []

    print("=" * 88)
    print("Repeated protein-level validation")
    print(f"Proteins: {len(base)}; positives: {int(y.sum())}; prevalence: {y.mean():.4f}")
    print(f"Repeated CV: {args.repeats} x {args.folds}-fold")
    print(f"RF trees: {args.rf_trees}")
    print(f"Output: {output_dir}")
    print("=" * 88, flush=True)

    for repeat in range(1, args.repeats + 1):
        split_seed = args.base_seed + repeat - 1
        cv = StratifiedKFold(
            n_splits=args.folds, shuffle=True, random_state=split_seed
        )

        for fold, (train_idx, test_idx) in enumerate(cv.split(base, y), start=1):
            y_train = y[train_idx]
            y_test = y[test_idx]
            train_genes = genes[train_idx]
            test_genes = genes[test_idx]

            for index in test_idx:
                split_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "gene": genes[index],
                        "y_true": int(y[index]),
                    }
                )

            # Primary 997-protein representation benchmarks and fixed MIFS reference.
            for model_name, columns in feature_sets.items():
                model = make_rf(args.rf_trees) if model_name.endswith("_rf") else make_lr()
                x_train = base.iloc[train_idx][columns].replace(
                    [np.inf, -np.inf], np.nan
                )
                x_test = base.iloc[test_idx][columns].replace(
                    [np.inf, -np.inf], np.nan
                )
                scores = fit_predict(model, x_train, y_train, x_test)
                metrics = score_predictions(y_test, scores)
                metric_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "model": model_name,
                        "n_features": len(columns),
                        "n_test": len(test_idx),
                        "n_positive_test": int(y_test.sum()),
                        **metrics,
                    }
                )
                for gene, truth, score in zip(test_genes, y_test, scores):
                    prediction_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "split_seed": split_seed,
                            "gene": gene,
                            "y_true": int(truth),
                            "model": model_name,
                            "y_score": float(score),
                        }
                    )

            # Training-fold-only XGBoost ranking followed by a compact LR model.
            x_rank_train_raw = base.iloc[train_idx][engineered_features].replace(
                [np.inf, -np.inf], np.nan
            )
            train_medians = x_rank_train_raw.median()
            x_rank_train = x_rank_train_raw.fillna(train_medians)
            pos = int(y_train.sum())
            neg = int((y_train == 0).sum())
            ranker = XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=neg / pos,
                random_state=MODEL_RANDOM_STATE,
                n_jobs=-1,
            )
            ranker.fit(x_rank_train, y_train)
            ranking = pd.DataFrame(
                {
                    "feature": engineered_features,
                    "importance": ranker.feature_importances_,
                }
            ).sort_values(["importance", "feature"], ascending=[False, True])
            ranking["rank"] = np.arange(1, len(ranking) + 1)
            top5 = ranking.head(5)["feature"].tolist()
            compact = [feature for feature in top5 if feature != "is_transporter_like"]
            for feature in ranking["feature"]:
                if len(compact) >= 4:
                    break
                if feature != "is_transporter_like" and feature not in compact:
                    compact.append(feature)
            compact = compact[:4]
            if len(compact) != 4:
                raise RuntimeError("Could not construct four-feature compact set")

            for row in ranking.itertuples(index=False):
                selection_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "rank": int(row.rank),
                        "feature": row.feature,
                        "family": feature_family(row.feature),
                        "importance": float(row.importance),
                        "in_training_fold_top5": int(row.feature in top5),
                        "selected_compact": int(row.feature in compact),
                    }
                )

            compact_model = make_lr()
            x_train = base.iloc[train_idx][compact].replace(
                [np.inf, -np.inf], np.nan
            )
            x_test = base.iloc[test_idx][compact].replace(
                [np.inf, -np.inf], np.nan
            )
            scores = fit_predict(compact_model, x_train, y_train, x_test)
            metrics = score_predictions(y_test, scores)
            compact_name = "foldwise_compact4_lr"
            metric_rows.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "split_seed": split_seed,
                    "model": compact_name,
                    "n_features": 4,
                    "n_test": len(test_idx),
                    "n_positive_test": int(y_test.sum()),
                    **metrics,
                }
            )
            for gene, truth, score in zip(test_genes, y_test, scores):
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "gene": gene,
                        "y_true": int(truth),
                        "model": compact_name,
                        "y_score": float(score),
                    }
                )

            # AlphaFold matched analysis inherits the global protein fold assignment.
            af_train_genes = [gene for gene in train_genes if gene in af_index]
            af_test_genes = [gene for gene in test_genes if gene in af_index]
            af_train_idx = [af_index[gene] for gene in af_train_genes]
            af_test_idx = [af_index[gene] for gene in af_test_genes]
            af_y_train = af.iloc[af_train_idx][LABEL].astype(int).to_numpy()
            af_y_test = af.iloc[af_test_idx][LABEL].astype(int).to_numpy()
            if len(np.unique(af_y_test)) != 2:
                raise RuntimeError("AlphaFold matched test fold lacks one class")

            for model_name, columns in af_feature_sets.items():
                x_train = af.iloc[af_train_idx][columns].replace(
                    [np.inf, -np.inf], np.nan
                )
                x_test = af.iloc[af_test_idx][columns].replace(
                    [np.inf, -np.inf], np.nan
                )
                scores = fit_predict(make_lr(), x_train, af_y_train, x_test)
                metrics = score_predictions(af_y_test, scores)
                metric_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "split_seed": split_seed,
                        "model": model_name,
                        "n_features": len(columns),
                        "n_test": len(af_test_idx),
                        "n_positive_test": int(af_y_test.sum()),
                        **metrics,
                    }
                )
                for gene, truth, score in zip(af_test_genes, af_y_test, scores):
                    prediction_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "split_seed": split_seed,
                            "gene": gene,
                            "y_true": int(truth),
                            "model": model_name,
                            "y_score": float(score),
                        }
                    )

        # Checkpoint after every complete repeat.
        pd.DataFrame(split_rows).to_csv(
            output_dir / "repeated_cv_split_manifest.tsv", sep="\t", index=False
        )
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "repeated_cv_per_fold_metrics.checkpoint.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame(prediction_rows).to_csv(
            output_dir / "repeated_cv_oof_predictions.checkpoint.tsv",
            sep="\t",
            index=False,
        )
        pd.DataFrame(selection_rows).to_csv(
            output_dir / "repeated_foldwise_feature_rankings.checkpoint.tsv",
            sep="\t",
            index=False,
        )
        print(f"Completed repeat {repeat}/{args.repeats} (seed={split_seed})", flush=True)

    split_manifest = pd.DataFrame(split_rows)
    fold_metrics = pd.DataFrame(metric_rows)
    oof = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)

    expected_full_predictions = args.repeats * len(base)
    expected_af_predictions = args.repeats * len(af)
    for model, group in oof.groupby("model"):
        expected = expected_af_predictions if "afmatched" in model or model.startswith("alphafold") or model.startswith("mifs_plus_alphafold") else expected_full_predictions
        if len(group) != expected:
            raise RuntimeError(
                f"Incomplete OOF predictions for {model}: {len(group)} != {expected}"
            )
        counts = group.groupby(["repeat", "gene"]).size()
        if not (counts == 1).all():
            raise RuntimeError(f"Duplicate or missing repeat-level OOF predictions for {model}")

    split_manifest.to_csv(
        output_dir / "repeated_cv_split_manifest.tsv", sep="\t", index=False
    )
    summary, selection_summary, family_summary, jaccard_summary = summarize_outputs(
        output_dir,
        fold_metrics,
        oof,
        selections,
        args.repeats,
        args.folds,
    )

    metadata = {
        "analysis": "repeated_split_matched_protein_validation",
        "repeats": args.repeats,
        "folds": args.folds,
        "base_seed": args.base_seed,
        "split_seeds": list(range(args.base_seed, args.base_seed + args.repeats)),
        "rf_trees": args.rf_trees,
        "rf_random_state": MODEL_RANDOM_STATE,
        "n_proteins": len(base),
        "n_positive": int(y.sum()),
        "n_alphafold_matched": len(af),
        "n_alphafold_positive": int(af[LABEL].sum()),
        "engineered_feature_count": len(engineered_features),
        "traditional_feature_count": len(TRADITIONAL16),
        "mifs_fixed_features": MIFS4,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "notes": [
            "All primary Random Forest representations use identical 500-tree settings by default.",
            "The fixed MIFS model remains an exploratory full-data-derived reference.",
            "The foldwise compact model repeats XGBoost ranking using training data only.",
            "AlphaFold matched analyses inherit the global protein fold assignment after excluding the one unavailable protein.",
        ],
    }
    with open(output_dir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print("Repeated-CV summary (metrics calculated from one complete OOF vector per repeat):")
    print(summary.to_string(index=False))
    print()
    print("Most frequently selected fold-wise compact descriptors:")
    print(selection_summary.head(20).to_string(index=False))
    print()
    print("Feature-family summary:")
    print(family_summary.to_string(index=False))
    print()
    print("Pairwise Jaccard stability:")
    print(jaccard_summary.to_string(index=False))
    print()
    print(f"Saved all outputs to: {output_dir}")


if __name__ == "__main__":
    main()
