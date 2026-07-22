# Paper2 robustness validation — canonical analysis summary

This summary identifies the canonical robustness outputs for manuscript use.
The canonical analyses are the protocol-matched results under `results/`:

- `results/repeated20x5/`
- `results/permutation_seed42_b200/`
- `results/paired_bootstrap_b5000/`

These analyses use 20 repeats of five-fold stratified cross-validation, fixed
500-tree Random Forest settings for the matched primary representation
comparison, training-fold-only ranking for the adaptive compact model, and
paired bootstrap comparisons of mean repeated out-of-fold predictions.

## Canonical repeated validation

| Model | ROC-AUC mean ± SD | PR-AUC mean ± SD |
|---|---:|---:|
| Fixed MIFS, protein-level | 0.654 ± 0.009 | 0.123 ± 0.004 |
| Fixed MIFS, AlphaFold-matched subset | 0.654 ± 0.009 | 0.123 ± 0.004 |
| Traditional-16 + ESM2-35M | 0.627 ± 0.019 | 0.107 ± 0.007 |
| ESM2-35M | 0.627 ± 0.018 | 0.107 ± 0.007 |
| MIFS + AlphaFold | 0.624 ± 0.016 | 0.120 ± 0.006 |
| ESM2-650M | 0.617 ± 0.017 | 0.104 ± 0.007 |
| Engineered-46 | 0.615 ± 0.018 | 0.115 ± 0.010 |
| Traditional-16 + ESM2-650M | 0.614 ± 0.016 | 0.104 ± 0.006 |
| Fold-wise adaptive compact-4 | 0.603 ± 0.021 | 0.100 ± 0.009 |
| Global AlphaFold-13 | 0.597 ± 0.016 | 0.110 ± 0.008 |

The fixed MIFS values remain an exploratory full-data-derived reference. The
adaptive compact model is the more conservative estimate because feature
ranking is repeated inside each training fold.

## Canonical permutation test

The full-pipeline protein-level permutation analysis used 200 permutations.
Empirical p-values were:

- ESM2-35M ROC-AUC: 0.00498; PR-AUC: 0.0647.
- Fold-wise adaptive compact ROC-AUC: 0.00498; PR-AUC: 0.0199.
- Global AlphaFold ROC-AUC: 0.0249; PR-AUC: 0.0746.

ROC-AUC signal is consistently above the permuted-label null. PR-AUC is more
variable because of the 7.42% protein-level positive prevalence and should be
reported cautiously.

## Canonical paired bootstrap

The paired bootstrap uses 5,000 prevalence-preserving resamples of mean
repeated out-of-fold predictions. Key comparisons:

- ESM2-35M vs Engineered-46: ΔROC-AUC = +0.0135, 95% CI −0.0502 to +0.0757.
- ESM2-650M vs ESM2-35M: ΔROC-AUC = −0.0147, 95% CI −0.0623 to +0.0339.
- Traditional-16 + ESM2-35M vs ESM2-35M: ΔROC-AUC = −0.0007, 95% CI −0.0086 to +0.0064.
- MIFS vs adaptive compact-4: ΔROC-AUC = +0.0236, 95% CI −0.0044 to +0.0529; ΔPR-AUC = +0.0160, 95% CI +0.0038 to +0.0378.
- MIFS vs global AlphaFold on the matched subset: ΔROC-AUC = +0.0427, 95% CI −0.0253 to +0.1088.
- MIFS + AlphaFold vs MIFS on the matched subset: ΔROC-AUC = −0.0152, 95% CI −0.0543 to +0.0251.

These results support the wording “did not consistently improve” rather than
claims that one representation significantly outperformed another.
