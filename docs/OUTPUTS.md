# Output-to-manuscript map

All paths below are under `artifacts/` and are generated locally.

| Output | Contents | Manuscript use |
|---|---|---|
| `manifests/manifest_summary.csv` | Cohort/acquisition availability | Cohort description |
| `results/bspc_extension_split_audit.csv` | Fold overlap counts/rates | Quantitative split audit |
| `results/bspc_extension_seed_metrics.csv` | ANN/SNN seed metrics | Private/public ANN-SNN tables |
| `results/bspc_extension_seed_summary.csv` | ANN/SNN mean and seed SD | Reference summaries |
| `results/bspc_extension_leakage_statistics.csv` | Ensemble subject-level split effects | Leakage inference table |
| `results/bspc_model_suite_seed_metrics.csv` | Five-model seed metrics | Five-model split figures |
| `results/bspc_model_suite_seed_summary.csv` | Five-model means/SDs | Strict architecture table |
| `results/bspc_model_suite_leakage_statistics.csv` | Five-model leakage tests | Split-effect claims |
| `results/bspc_model_suite_architecture_statistics.csv` | Candidate-vs-ANN tests | Architecture claims |
| `results/bspc_model_suite_complexity.csv` | Parameters/MAC equivalents | Efficiency section |
| `results/variance_decomposition_icc.csv` | Eta-squared and ICC intervals | Identity mechanism |
| `results/identity_classifier_summary.csv` | Accuracy/balanced/null tests | Identity shortcut |
| `results/strict_agreement_statistics.csv` | Bias, limits, slopes, R-squared | Agreement figure/text |

Expected final figure stems:

- `cohort_bp_distributions_nature`
- `split_leakage_effect_nature`
- `public_luh_split_effect_nature`
- `strict_model_suite_nature`
- `strict_agreement_nature`

Do not commit result CSVs or figures produced from private records. Before a
public release of aggregate tables or graphics, obtain approval under the study's
ethics and data-governance terms and perform a disclosure review.
