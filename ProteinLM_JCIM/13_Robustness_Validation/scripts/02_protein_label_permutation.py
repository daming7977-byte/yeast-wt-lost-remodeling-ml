#!/usr/bin/env python3
"""Full-pipeline protein-level label-permutation negative controls."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATION_SCRIPT = SCRIPT_DIR / "01_repeated_protein_validation.py"
spec = importlib.util.spec_from_file_location("repeated_validation", VALIDATION_SCRIPT)
validation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(validation)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=20260721)
    parser.add_argument("--rf-trees", type=int, default=500)
    parser.add_argument("--tag", default="permutation_seed42_b200")
    parser.add_argument(
        "--observed-oof",
        default=(
            "ProteinLM_JCIM/13_Robustness_Validation/results/repeated20x5/"
            "repeated_cv_oof_predictions.tsv"
        ),
    )
    return parser.parse_args()


def metrics(y_true, y_score):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def main():
    args = parse_args()
    outdir = validation.OUT_ROOT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    base, af, engineered_features, feature_sets, af_feature_sets = validation.load_inputs()
    genes = base["gene"].astype(str).to_numpy()
    y = base[validation.LABEL].astype(int).to_numpy()
    af_y = af[validation.LABEL].astype(int).to_numpy()
    af_index = {gene: i for i, gene in enumerate(af["gene"].astype(str))}

    cv = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.split_seed
    )
    splits = list(cv.split(base, y))

    observed_oof = pd.read_csv(args.observed_oof, sep="\t")
    observed_oof = observed_oof.loc[observed_oof["repeat"] == 1].copy()
    target_models = ["esm2_35m_rf", "foldwise_compact4_lr", "alphafold13_lr"]
    observed_rows = []
    for model_name in target_models:
        group = observed_oof.loc[observed_oof["model"] == model_name]
        result = metrics(group["y_true"], group["y_score"])
        observed_rows.append(
            {
                "model": model_name,
                "split_seed": args.split_seed,
                "n": len(group),
                "n_positive": int(group["y_true"].sum()),
                **result,
            }
        )
    observed = pd.DataFrame(observed_rows)
    observed.to_csv(outdir / "permutation_observed_metrics.tsv", sep="\t", index=False)

    null_rows = []
    for permutation in range(1, args.permutations + 1):
        rng = np.random.default_rng(args.permutation_seed + permutation)
        y_perm = rng.permutation(y)
        af_y_perm = rng.permutation(af_y)

        esm_scores = np.full(len(base), np.nan)
        compact_scores = np.full(len(base), np.nan)
        af_scores = np.full(len(af), np.nan)

        for train_idx, test_idx in splits:
            y_train = y_perm[train_idx]
            if len(np.unique(y_perm[test_idx])) != 2:
                raise RuntimeError("A permuted protein test fold lacks one class")

            # ESM2-35M Random Forest null pipeline.
            esm_columns = feature_sets["esm2_35m_rf"]
            esm_model = validation.make_rf(args.rf_trees)
            esm_scores[test_idx] = validation.fit_predict(
                esm_model,
                base.iloc[train_idx][esm_columns],
                y_train,
                base.iloc[test_idx][esm_columns],
            )

            # Training-fold-only ranking and compact Logistic Regression null pipeline.
            x_rank_raw = base.iloc[train_idx][engineered_features].replace(
                [np.inf, -np.inf], np.nan
            )
            x_rank = x_rank_raw.fillna(x_rank_raw.median())
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
                random_state=validation.MODEL_RANDOM_STATE,
                n_jobs=-1,
            )
            ranker.fit(x_rank, y_train)
            ranking = pd.DataFrame(
                {
                    "feature": engineered_features,
                    "importance": ranker.feature_importances_,
                }
            ).sort_values(["importance", "feature"], ascending=[False, True])
            top5 = ranking.head(5)["feature"].tolist()
            compact = [feature for feature in top5 if feature != "is_transporter_like"]
            for feature in ranking["feature"]:
                if len(compact) >= 4:
                    break
                if feature != "is_transporter_like" and feature not in compact:
                    compact.append(feature)
            compact = compact[:4]
            compact_scores[test_idx] = validation.fit_predict(
                validation.make_lr(),
                base.iloc[train_idx][compact],
                y_train,
                base.iloc[test_idx][compact],
            )

            # AlphaFold matched null pipeline using the inherited global fold assignment.
            train_genes = genes[train_idx]
            test_genes = genes[test_idx]
            af_train_idx = [af_index[gene] for gene in train_genes if gene in af_index]
            af_test_idx = [af_index[gene] for gene in test_genes if gene in af_index]
            if len(np.unique(af_y_perm[af_test_idx])) != 2:
                raise RuntimeError("A permuted AlphaFold test fold lacks one class")
            af_columns = af_feature_sets["alphafold13_lr"]
            af_scores[af_test_idx] = validation.fit_predict(
                validation.make_lr(),
                af.iloc[af_train_idx][af_columns],
                af_y_perm[af_train_idx],
                af.iloc[af_test_idx][af_columns],
            )

        for model_name, truth, score in [
            ("esm2_35m_rf", y_perm, esm_scores),
            ("foldwise_compact4_lr", y_perm, compact_scores),
            ("alphafold13_lr", af_y_perm, af_scores),
        ]:
            if np.isnan(score).any():
                raise RuntimeError(f"Missing permuted OOF predictions for {model_name}")
            result = metrics(truth, score)
            null_rows.append(
                {
                    "permutation": permutation,
                    "permutation_seed": args.permutation_seed + permutation,
                    "model": model_name,
                    "n": len(truth),
                    "n_positive": int(truth.sum()),
                    **result,
                }
            )

        if permutation % 10 == 0 or permutation == args.permutations:
            pd.DataFrame(null_rows).to_csv(
                outdir / "permutation_null_metrics.checkpoint.tsv",
                sep="\t",
                index=False,
            )
            print(
                f"Completed permutation {permutation}/{args.permutations}", flush=True
            )

    null = pd.DataFrame(null_rows)
    null.to_csv(outdir / "permutation_null_metrics.tsv", sep="\t", index=False)

    summary_rows = []
    for observed_row in observed.itertuples(index=False):
        group = null.loc[null["model"] == observed_row.model]
        for metric_name in ["roc_auc", "pr_auc"]:
            value = getattr(observed_row, metric_name)
            null_values = group[metric_name]
            exceedances = int((null_values >= value).sum())
            summary_rows.append(
                {
                    "model": observed_row.model,
                    "metric": metric_name,
                    "observed": value,
                    "null_mean": null_values.mean(),
                    "null_sd": null_values.std(ddof=1),
                    "null_q025": null_values.quantile(0.025),
                    "null_q975": null_values.quantile(0.975),
                    "n_permutations": args.permutations,
                    "n_null_at_or_above_observed": exceedances,
                    "empirical_p": (1 + exceedances) / (1 + args.permutations),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "permutation_empirical_pvalues.tsv", sep="\t", index=False)

    metadata = {
        "analysis": "full_pipeline_protein_label_permutation",
        "permutations": args.permutations,
        "folds": args.folds,
        "split_seed": args.split_seed,
        "permutation_seed": args.permutation_seed,
        "rf_trees": args.rf_trees,
        "models": target_models,
        "notes": [
            "Protein labels were permuted while preserving class counts.",
            "The fold assignment was fixed to the observed seed-42 split.",
            "The foldwise compact pipeline repeated XGBoost ranking inside each permuted training fold.",
            "AlphaFold labels were permuted within the 996-protein matched subset.",
        ],
    }
    with open(outdir / "analysis_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print()
    print(summary.to_string(index=False))
    print()
    print(f"Saved all outputs to: {outdir}")


if __name__ == "__main__":
    main()
