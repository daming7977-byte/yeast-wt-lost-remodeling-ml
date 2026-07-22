# Paper2 / ProteinLM_JCIM

Reproducibility package for the manuscript **Benchmarking Protein Representations for Translational Pause Remodeling Susceptibility in Yeast Membrane Proteins**.

## Contents

The repository contains the processed protein- and TMD-level input tables, ESM2 embeddings used for benchmarking, AlphaFold-derived descriptor tables, analysis scripts, canonical result tables, robustness-validation outputs, and figure-generation scripts.

This is the curated submission/reproducibility package: it includes the scripts needed to regenerate the reported analyses and final figures, but omits superseded exploratory scripts and temporary smoke-test outputs.

Raw AlphaFold PDB files are intentionally not included. The repository therefore documents and preserves the processed descriptor tables used for the reported analyses without redistributing the raw structure archive.

## Recommended environment

The original analysis environment is recorded in `ProteinLM_JCIM/09_environment.txt`. A minimal benchmark environment can be created with:

```bash
conda create -n proteinlm_jcim python=3.11 -y
conda activate proteinlm_jcim
pip install pandas numpy scipy scikit-learn xgboost matplotlib
```

The scripts were developed with pandas, NumPy, scikit-learn, XGBoost, matplotlib, and PyTorch/ESM2 for embedding extraction. Existing embeddings are included, so embedding extraction is not required to reproduce the reported benchmark tables.

## Reproducing the reported results

Run commands from the repository root (the directory containing `ProteinLM_JCIM/`).

Protein-level benchmark scripts are in `ProteinLM_JCIM/07_scripts/`.

Feature discovery and MIFS interpretation are in:

```text
ProteinLM_JCIM/09_feature_selection/scripts/
ProteinLM_JCIM/12_Model_Interpretation/scripts/
```

AlphaFold global and TMD-local analyses are in:

```text
ProteinLM_JCIM/10_AlphaFold/scripts/
ProteinLM_JCIM/11_TMD_AlphaFold/scripts/
```

The three canonical robustness analyses are:

```text
ProteinLM_JCIM/13_Robustness_Validation/scripts/01_repeated_protein_validation.py
ProteinLM_JCIM/13_Robustness_Validation/scripts/02_protein_label_permutation.py
ProteinLM_JCIM/13_Robustness_Validation/scripts/03_paired_oof_bootstrap.py
```

Their authoritative outputs are under:

```text
ProteinLM_JCIM/13_Robustness_Validation/results/
```

See `ROBUSTNESS_VALIDATION_SUMMARY_v1.md` for the exact reported values and interpretation.

## Figures

The final figure scripts are in `ProteinLM_JCIM/06_figures/`. They read the processed result tables from the repository and write vector and raster figure files next to the scripts.

## Data conventions

- Protein-level task: 997 proteins, 74 WT-lost-positive proteins, 923 background proteins.
- TMD-level source landscape: 5,028 annotated TMDs, including 121 WT-lost events.
- AlphaFold matched subset: 996 proteins with available processed structural descriptors.
- TMD-level analyses use gene-grouped cross-validation to prevent TMDs from the same protein entering both training and test partitions.

## Citation and versioning

This package corresponds to the v13 robustness-updated manuscript candidate. Before public release, add the final repository DOI/URL to the manuscript and replace any local/private paths in scripts or metadata if the repository is moved.
