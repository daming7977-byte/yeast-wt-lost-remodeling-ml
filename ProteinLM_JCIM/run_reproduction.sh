#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "$SCRIPT_DIR/07_scripts" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$SCRIPT_DIR"
  PROJECT_ROOT="$SCRIPT_DIR/ProteinLM_JCIM"
fi
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
MODE="${1:-help}"

run_main() {
  "$PYTHON" "$PROJECT_ROOT/07_scripts/benchmark_protein_features_v2_enriched.py"
  "$PYTHON" "$PROJECT_ROOT/07_scripts/benchmark_protein_level_esm2_35M.py"
  if [[ -f "$PROJECT_ROOT/01_embeddings/esm2_650M_full_protein_embeddings.tsv" ]]; then
    "$PYTHON" "$PROJECT_ROOT/07_scripts/benchmark_protein_level_esm2_650M.py"
  else
    echo "SKIP: ESM2-650M embedding file is not present in this checkout."
  fi
  "$PYTHON" "$PROJECT_ROOT/09_feature_selection/scripts/01_xgboost_feature_importance.py"
  "$PYTHON" "$PROJECT_ROOT/09_feature_selection/scripts/02_topN_feature_retraining.py"
  "$PYTHON" "$PROJECT_ROOT/09_feature_selection/scripts/03_ablation_remove_transporter.py"
  "$PYTHON" "$PROJECT_ROOT/09_feature_selection/scripts/05_nested_mifs_validation.py"
  "$PYTHON" "$PROJECT_ROOT/10_AlphaFold/scripts/benchmark_alphafold_features.py"
  "$PYTHON" "$PROJECT_ROOT/11_TMD_AlphaFold/scripts/04_benchmark_tmd_local_plddt_StratifiedGroupKFold.py"
}

run_robustness() {
  "$PYTHON" "$PROJECT_ROOT/13_Robustness_Validation/scripts/01_repeated_protein_validation.py"
  "$PYTHON" "$PROJECT_ROOT/13_Robustness_Validation/scripts/02_protein_label_permutation.py"
  "$PYTHON" "$PROJECT_ROOT/13_Robustness_Validation/scripts/03_paired_oof_bootstrap.py"
}

run_figures() {
  "$PYTHON" "$PROJECT_ROOT/06_figures/make_Figure1_integrated_framework_v1.py"
  "$PYTHON" "$PROJECT_ROOT/06_figures/make_Figure2_protein_level_benchmark_v3.py"
  "$PYTHON" "$PROJECT_ROOT/06_figures/make_Figure3_tmd_representation_scale_v1.py"
  "$PYTHON" "$PROJECT_ROOT/06_figures/make_Figure4_mifs_discovery_stability_v1.py"
  "$PYTHON" "$PROJECT_ROOT/06_figures/make_Figure5_conceptual_mifs_v1.py"
}

case "$MODE" in
  main) run_main ;;
  robustness) run_robustness ;;
  figures) run_figures ;;
  all) run_main; run_robustness; run_figures ;;
  help|-h|--help)
    echo "Usage: ./run_reproduction.sh {main|robustness|figures|all}"
    echo "Set PYTHON=/path/to/python if the environment is not on PATH."
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
