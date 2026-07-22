# Reproduction manifest

This package is organized around the analyses reported in the JCIM manuscript.

## Main protein-level and representation analyses

```bash
bash ProteinLM_JCIM/run_reproduction.sh main
```

This runs the engineered-descriptor benchmark, ESM2-35M and ESM2-650M benchmarks when the corresponding embedding is present, feature discovery, transporter ablation, fold-wise feature-selection validation, global AlphaFold benchmarking, and gene-grouped TMD-local AlphaFold benchmarking.

## Robustness analyses

```bash
bash ProteinLM_JCIM/run_reproduction.sh robustness
```

This runs repeated five-fold validation, full-pipeline label permutation, and paired out-of-fold bootstrap comparisons. The canonical previously generated outputs are retained under `ProteinLM_JCIM/13_Robustness_Validation/results/`.

## Figure regeneration

```bash
bash ProteinLM_JCIM/run_reproduction.sh figures
```

The scripts write vector and raster figures next to the corresponding figure scripts.

## Important reproducibility notes

- The primary protein-level task contains 997 proteins, including 74 WT-lost-positive proteins and 923 background proteins.
- TMD-level analyses use gene-grouped cross-validation to prevent TMDs from the same protein entering both training and test partitions.
- The 650M embedding is large and may be absent from a GitHub web-upload checkout. If absent, the wrapper skips only that benchmark and prints a warning; the extraction workflow is retained in the package.
- Raw AlphaFold PDB files are not redistributed. The processed AlphaFold descriptor tables used for the reported analyses are included.
