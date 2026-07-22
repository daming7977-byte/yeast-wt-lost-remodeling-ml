#!/usr/bin/env python3
"""Paired stratified bootstrap of mean repeated out-of-fold predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_OOF = Path(
    "ProteinLM_JCIM/13_Robustness_Validation/results/repeated20x5/"
    "repeated_cv_mean_oof_predictions.tsv"
)
OUT_ROOT = Path("ProteinLM_JCIM/13_Robustness_Validation/results")

COMPARISONS = [
    ("esm2_35m_minus_engineered46", "esm2_35m_rf", "engineered46_rf"),
    ("esm2_650m_minus_esm2_35m", "esm2_650m_rf", "esm2_35m_rf"),
    (
        "traditional16_plus_esm2_35m_minus_esm2_35m",
        "traditional16_plus_esm2_35m_rf",
        "esm2_35m_rf",
    ),
    (
        "traditional16_plus_esm2_650m_minus_esm2_650m",
        "traditional16_plus_esm2_650m_rf",
        "esm2_650m_rf",
    ),
    (
        "foldwise_compact4_minus_esm2_35m",
        "foldwise_compact4_lr",
        "esm2_35m_rf",
    ),
    (
        "fixed_mifs4_minus_foldwise_compact4",
        "mifs4_fixed_lr",
        "foldwise_compact4_lr",
    ),
    (
        "mifs4_minus_alphafold13_matched",
        "mifs4_lr_afmatched",
        "alphafold13_lr",
    ),
    (
        "mifs_plus_alphafold_minus_mifs4_matched",
        "mifs_plus_alphafold17_lr",
        "mifs4_lr_afmatched",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--tag", default="paired_bootstrap_b5000")
    return parser.parse_args()


def metric_values(y, scores):
    return {
        "roc_auc": roc_auc_score(y, scores),
        "pr_auc": average_precision_score(y, scores),
    }


def paired_frame(oof: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    subset = oof.loc[oof["model"].isin([model_a, model_b])].copy()
    wide = subset.pivot(index=["gene", "y_true"], columns="model", values="mean_repeated_oof_score").reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=[model_a, model_b]).reset_index(drop=True)
    if wide["gene"].duplicated().any():
        raise ValueError("Duplicate genes in paired OOF matrix")
    if wide["y_true"].nunique() != 2:
        raise ValueError("Paired comparison lacks one outcome class")
    return wide


def main():
    args = parse_args()
    outdir = OUT_ROOT / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_csv(args.oof, sep="\t")
    required = {"gene", "y_true", "model", "mean_repeated_oof_score"}
    if not required.issubset(oof.columns):
        raise ValueError(f"OOF file is missing: {sorted(required - set(oof.columns))}")

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = []
    summary_rows = []

    for comparison, model_a, model_b in COMPARISONS:
        wide = paired_frame(oof, model_a, model_b)
        y = wide["y_true"].astype(int).to_numpy()
        score_a = wide[model_a].to_numpy()
        score_b = wide[model_b].to_numpy()
        positive_idx = np.flatnonzero(y == 1)
        negative_idx = np.flatnonzero(y == 0)

        observed_a = metric_values(y, score_a)
        observed_b = metric_values(y, score_b)

        comparison_bootstraps = {"roc_auc": [], "pr_auc": []}
        for bootstrap in range(1, args.bootstraps + 1):
            sampled_positive = rng.choice(
                positive_idx, size=len(positive_idx), replace=True
            )
            sampled_negative = rng.choice(
                negative_idx, size=len(negative_idx), replace=True
            )
            sampled = np.concatenate([sampled_positive, sampled_negative])
            rng.shuffle(sampled)

            boot_a = metric_values(y[sampled], score_a[sampled])
            boot_b = metric_values(y[sampled], score_b[sampled])
            for metric in ["roc_auc", "pr_auc"]:
                delta = boot_a[metric] - boot_b[metric]
                comparison_bootstraps[metric].append(delta)
                bootstrap_rows.append(
                    {
                        "comparison": comparison,
                        "model_a": model_a,
                        "model_b": model_b,
                        "bootstrap": bootstrap,
                        "metric": metric,
                        "delta_a_minus_b": delta,
                    }
                )

        for metric in ["roc_auc", "pr_auc"]:
            values = np.asarray(comparison_bootstraps[metric])
            proportion_le_zero = (np.sum(values <= 0) + 1) / (len(values) + 1)
            proportion_ge_zero = (np.sum(values >= 0) + 1) / (len(values) + 1)
            summary_rows.append(
                {
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired_proteins": len(wide),
                    "n_positive": int(y.sum()),
                    "metric": metric,
                    "model_a_observed": observed_a[metric],
                    "model_b_observed": observed_b[metric],
                    "observed_delta_a_minus_b": observed_a[metric]
                    - observed_b[metric],
                    "bootstrap_delta_mean": values.mean(),
                    "ci95_lower": np.quantile(values, 0.025),
                    "ci95_upper": np.quantile(values, 0.975),
                    "proportion_delta_gt_zero": np.mean(values > 0),
                    "two_sided_bootstrap_p": min(
                        1.0, 2 * min(proportion_le_zero, proportion_ge_zero)
                    ),
                    "n_bootstraps": args.bootstraps,
                }
            )
        print(f"Completed {comparison}", flush=True)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    summary = pd.DataFrame(summary_rows)
    bootstrap_df.to_csv(
        outdir / "paired_bootstrap_differences.tsv", sep="\t", index=False
    )
    summary.to_csv(outdir / "paired_bootstrap_summary.tsv", sep="\t", index=False)

    metadata = {
        "analysis": "paired_stratified_bootstrap_of_mean_repeated_oof_predictions",
        "oof_file": str(args.oof),
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "comparisons": [
            {"name": name, "model_a": model_a, "model_b": model_b}
            for name, model_a, model_b in COMPARISONS
        ],
        "notes": [
            "Each protein score is the mean of its 20 out-of-fold predictions.",
            "Positive and background proteins were resampled separately, preserving prevalence.",
            "Identical bootstrap indices were used for both models in each comparison.",
            "AlphaFold comparisons use only the matched 996-protein subset.",
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
