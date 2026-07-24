# Paper2 / ProteinLM_JCIM

Reproducibility package for **Benchmarking Protein Representations for Translational Pause Remodeling Susceptibility in Yeast Membrane Proteins**.

Repository: https://github.com/daming7977-byte/yeast-wt-lost-remodeling-ml

This is the curated submission package corresponding to the manuscript version submitted to the *Journal of Chemical Information and Modeling*. It contains the processed inputs, canonical result tables, analysis scripts, robustness-validation outputs, and final figure-generation scripts used for the reported analyses.

## Scope and reproducibility

The package preserves the complete analysis chain needed to reproduce the reported protein-level and gene-grouped TMD-level results. Superseded exploratory analyses, the non-grouped TMD benchmark, and temporary smoke-test outputs are intentionally omitted from this release so that the repository contains only the canonical analyses used in the manuscript and Supporting Information.

Raw AlphaFold PDB files and raw sequencing reads are not included. The repository contains the processed descriptor tables and public-data identifiers needed for the reported analyses. Existing ESM2 embeddings are included where practical; extraction scripts are provided for regeneration when needed.

## Recommended environment

The original analysis environment is recorded in `ProteinLM_JCIM/09_environment.txt`. A minimal benchmark environment can be created with:

```bash
conda create -n proteinlm_jcim python=3.11 -y
conda activate proteinlm_jcim
pip install pandas numpy scipy scikit-learn xgboost matplotlib
```

Embedding extraction additionally requires PyTorch and the relevant ESM2 model package. Existing embeddings are sufficient to reproduce the reported benchmark tables without repeating embedding extraction.

## Reproducing the reported results

Run commands from the repository root, the directory containing `ProteinLM_JCIM/`.

Protein-level representation benchmarks:

```text
ProteinLM_JCIM/07_scripts/
```

Feature discovery, MIFS interpretation, and fold-wise validation:

```text
ProteinLM_JCIM/09_feature_selection/scripts/
ProteinLM_JCIM/12_Model_Interpretation/scripts/
ProteinLM_JCIM/13_Robustness_Validation/scripts/
```

The canonical robustness analyses are:

```text
ProteinLM_JCIM/13_Robustness_Validation/scripts/01_repeated_protein_validation.py
ProteinLM_JCIM/13_Robustness_Validation/scripts/02_protein_label_permutation.py
ProteinLM_JCIM/13_Robustness_Validation/scripts/03_paired_oof_bootstrap.py
```

Their authoritative outputs are under `ProteinLM_JCIM/13_Robustness_Validation/results/`; see `ROBUSTNESS_VALIDATION_SUMMARY_v1.md` for the reported values and interpretation.

Global AlphaFold and local TMD-centered pLDDT analyses are under:

```text
ProteinLM_JCIM/10_AlphaFold/
ProteinLM_JCIM/11_TMD_AlphaFold/
```

The local AlphaFold benchmark uses `StratifiedGroupKFold` grouped by gene. All TMDs from the same protein are kept in the same partition.

## Figures

The final figure scripts are in `ProteinLM_JCIM/06_figures/`. They read the canonical processed result tables and generate the main manuscript figures.

## Data conventions

- Protein-level task: 997 proteins, 74 WT-lost-positive proteins, and 923 background proteins.
- TMD-level source landscape: 5,028 annotated TMDs, including 121 WT-lost events.
- AlphaFold matched protein-level subset: 996 proteins.
- TMD-local AlphaFold analysis: 5,027 TMDs from 996 proteins with mapped structural features.
- TMD-level analyses use gene-grouped cross-validation to prevent protein-level overlap between training and test partitions.

## Related manuscript

The WT-lost labels originate from a related TMD-level study submitted to *NAR Genomics and Bioinformatics* (manuscript ID NARGAB-2026-201). The present package supports the distinct protein-level representation-benchmarking study described in this repository.

## Versioning and citation

This repository is the public reproducibility package for the current submission. No DOI has been assigned to the repository at the time of submission. If the repository is moved or renamed, update the URL in the manuscript, Supporting Information, and cover letter.
