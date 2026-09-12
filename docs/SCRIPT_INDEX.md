# Script index

## Data and preprocessing

| Script | Purpose | Main prerequisite |
|---|---|---|
| `src/env_check.py` | Record package, CUDA, and device information | Conda environment |
| `src/build_manifest.py` | Match private modalities/labels and freeze strict folds | Authorized private data and `BP_ID_SALT` |
| `src/extract_signals.py` | Detect faces and export ROI RGB traces | Private manifest |
| `src/prepare_dataset.py` | POS/CHROM selection, filtering, windows, descriptors | Extracted signals |
| `src/validate_artifacts.py` | Check integrity and subject leakage | Prepared private artifacts |
| `src/inspect_public_luh.py` | Inspect HDF5 keys, shapes, and ranges | Public HDF5 |

## Models and training

| Script | Purpose | Status |
|---|---|---|
| `src/models.py` | ANN, SNN, ResNet-1D, TCN, Transformer definitions | Core |
| `src/train_deep.py` | Shared fold-local training and masked multi-output loss | Core utility |
| `src/run_bspc_extension_experiments.py` | Three-seed ANN/SNN split audit | Primary |
| `src/run_model_suite_extension.py` | Three additional neural architectures | Primary |
| `src/train_classical.py` | Median/ridge/SVR/tree baselines | Secondary |
| `src/train_leakage_audit.py` | Earlier single-seed private leakage runner | Legacy diagnostic |
| `src/train_public_luh.py` | Earlier public ANN/SNN baseline | Legacy diagnostic |
| `src/train_snn_fusion.py` | Quality/distillation experiment | Exploratory |
| `src/train_snn_fusion_nested.py` | Nested classical fusion experiment | Exploratory |
| `src/train_public_luh_optimized.py` | Public SNN-32/KD variants | Exploratory |
| `src/train_public_luh_v2.py` | Enhanced public SNN variant | Exploratory |

The primary publication numbers come from the two `run_*extension*.py` scripts,
not from the legacy single-seed runners.

## Analysis and reporting

| Script | Purpose |
|---|---|
| `src/revision_audit.py` | Split overlap, ICC, identity permutation, agreement |
| `src/audit_model_suite_outputs.py` | Seed/fold/model completeness |
| `src/profile_models.py` | Parameter and dense MAC-equivalent counts |
| `src/summarize_results.py` | Baseline/private summary utilities |
| `src/summarize_leakage.py` | Earlier leakage summaries |
| `src/summarize_public_luh.py` | Earlier public baseline summaries |
| `src/summarize_public_luh_optimization.py` | Exploratory public SNN summaries |
| `src/cohort_figures.py` | Earlier cohort distribution plots |
| `src/signal_quality_report.py` | Signal-quality summaries |

## Final journal figures

| Script | Outputs |
|---|---|
| `figures/make_five_model_split_figures.py` | Five-model private three-split and public two-split figures |
| `figures/make_journal_figures.py` | Cohort distributions, strict model suite, Bland-Altman/scatter figure |

All generated figure files are written to `artifacts/figures/` and remain
untracked by default.
