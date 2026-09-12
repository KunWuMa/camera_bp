# Reproduction guide

## 1. Scope

There are three reproducibility levels:

1. **Code/preflight**: checks imports, model construction, configuration, and the
   absence of restricted files. No participant data are needed.
2. **Public-only**: reproduces rPPG-BP-UKL random-window and subject-disjoint
   experiments after downloading the original public HDF5 file.
3. **Full manuscript**: reproduces the two private cohorts plus the public cohort.
   Authorized private data are required.

Run every command from the repository root.

## 2. Environment

```bash
conda env create -f environment.yml
conda activate camera_bp_bspc
python src/env_check.py
python scripts/check_release.py
```

The release check is intended for a clean export tree before upload. It will
deliberately fail if a locally populated `data/` or `artifacts/` tree is present;
run it on a clean clone/export or move governed data outside the repository via
`config/local.json`.

PyTorch CUDA 12.1 is pinned in `environment.yml`. For a CPU-only installation,
replace the PyTorch/CUDA packages with the official CPU build while retaining the
listed Python package versions. CPU execution is supported but the five-model
cross-validation run is slow.

## 3. Configure data

The default configuration uses only relative paths:

- private root: `data/private/`
- public HDF5: `data/public/rppg_bp_ukl/rPPG-BP-UKL_rppg_7s.h5`
- generated outputs: `artifacts/`

If governed storage must remain elsewhere, copy `config/default.json` to
`config/local.json`, change the paths, and keep that file untracked. The Git ignore
rules already exclude `config/local*.json`.

Before any private manifest is generated, set a secret salt of at least 16
characters:

```powershell
$env:BP_ID_SALT = "replace-with-a-long-random-secret"
```

Use the same salt only when an authorized analysis must reproduce the same folds.
Store it in an approved secret manager, not in source control.

## 4. Inspect commands without running them

```bash
python scripts/run_pipeline.py --stage all --dry-run
```

## 5. Private preprocessing

```bash
python scripts/run_pipeline.py --stage private-preprocessing
```

This executes, in order:

```bash
python src/build_manifest.py --config config/default.json
python src/extract_signals.py --config config/default.json --source all --workers 4
python src/prepare_dataset.py --config config/default.json
python src/validate_artifacts.py
```

Use `--overwrite` on the two processing scripts only when the authorized source
snapshot or extraction settings have changed.

## 6. Primary BSPC experiments

```bash
python scripts/run_pipeline.py --stage primary
```

The first runner produces the three-seed ANN/SNN split audit. The second appends
ResNet-1D, TCN, and Transformer using the same folds and training conventions.
Their dependency order is mandatory:

```bash
python src/run_bspc_extension_experiments.py \
  --seeds 20260907,20260908,20260909 --bootstrap-repeats 5000
python src/run_model_suite_extension.py \
  --seeds 20260907,20260908,20260909 --bootstrap-repeats 5000
```

Both runners can resume from their own existing output/checkpoint mechanism, but
do not merge outputs produced from different data snapshots or configurations.

## 7. Statistical analysis

```bash
python scripts/run_pipeline.py --stage analysis
```

This runs the quantitative overlap audit, between/within-subject variance
decomposition, ICC and bootstrap intervals, acquisition-disjoint identity
classification and label-permutation test, architecture/output completeness
checks, and parameter/MAC-equivalent profiling.

## 8. Figures

```bash
python scripts/run_pipeline.py --stage figures
```

Figures are saved as vector PDF plus PNG under `artifacts/figures/`. The
five-model split figures use `bspc_model_suite_seed_metrics.csv` and plot the three
seed-level subject-macro MAEs rather than reconstructing points from summary
statistics.

## 9. Exploratory experiments

The fingertip-teacher, quality-weighted SNN, and public SNN variants were post hoc
analyses and must not be presented as confirmatory comparisons.

```bash
python scripts/run_pipeline.py --stage exploratory
```

## 10. Final checks

```bash
python src/audit_model_suite_outputs.py
python scripts/check_release.py
```

Archive the configuration, code commit, hardware/software report, and secret-salt
identifier in the governed analysis record. Do not archive the salt in the public
repository.
