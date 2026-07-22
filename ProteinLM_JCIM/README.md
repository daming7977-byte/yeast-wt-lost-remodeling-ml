# Paper2 / ProteinLM_JCIM

Reproducibility package for **Benchmarking Protein Representations for Translational Pause Remodeling Susceptibility in Yeast Membrane Proteins**.

## What is included

The package contains processed protein- and TMD-level input tables, ESM2 embeddings used for benchmarking, AlphaFold-derived descriptor tables, analysis scripts, canonical benchmark outputs, robustness-validation outputs, and figure-generation scripts. Raw AlphaFold PDB files are not redistributed; the processed descriptor tables used in the reported analyses are included.

The large ESM2-650M embedding may be omitted from a GitHub web-upload checkout because of the web file-size limit. When present locally, it is used by the 650M benchmark; when absent, the reproduction wrapper skips only that benchmark. The extraction workflow remains in `ProteinLM_JCIM/07_scripts/`.

## Environment

The recorded analysis environment is in `ProteinLM_JCIM/09_environment.txt`. A minimal benchmark environment is:

```bash
conda create -n proteinlm_jcim python=3.11 -y
conda activate proteinlm_jcim
pip install -r ProteinLM_JCIM/requirements.txt
```

The original embedding-extraction workflow additionally requires PyTorch and ESM2/transformers support.

## Reproduction entry point

Run from the repository root:

```bash
bash ProteinLM_JCIM/run_reproduction.sh main
bash ProteinLM_JCIM/run_reproduction.sh robustness
bash ProteinLM_JCIM/run_reproduction.sh figures
```

Use `bash ProteinLM_JCIM/run_reproduction.sh all` to run all three stages. See `ProteinLM_JCIM/REPRODUCTION_MANIFEST.md` for the analysis map and interpretation notes.

## Dataset conventions

- Protein-level task: 997 proteins, 74 WT-lost-positive proteins, and 923 background proteins.
- TMD-level source landscape: 5,028 annotated TMDs, including 121 WT-lost events.
- AlphaFold matched subset: 996 proteins with available processed structural descriptors.
- TMD-level analyses use gene-grouped cross-validation to prevent protein-level overlap between training and test partitions.

## Version and citation

Public repository: https://github.com/daming7977-byte/yeast-wt-lost-remodeling-ml

This package corresponds to the robustness-updated manuscript candidate. If the repository is moved or archived, update the manuscript Data and Software Availability statement with the final repository URL or DOI.
