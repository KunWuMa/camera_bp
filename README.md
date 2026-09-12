# Camera-based blood-pressure evaluation split audit

This repository contains the data-processing, model-training, statistical-analysis,
and figure-generation code for the BSPC manuscript **“Evaluation split dominates
apparent model performance in camera-based blood pressure estimation: A
multi-cohort audit across five neural architectures.”**

The central experiment compares random-window, random-session, and
subject-disjoint evaluation for camera-derived rPPG blood-pressure regression.
Five neural architectures are evaluated: ANN, SNN, ResNet-1D, TCN, and
Transformer. Subject-macro MAE is the primary metric.

This is research software, not a clinically validated blood-pressure device.

## Data are intentionally absent

No private or public participant data are distributed here. In particular, this
repository contains no video, fingertip PPG, BP/HR table, HDF5 file, NumPy array,
model checkpoint, raw identifier, hashed identifier, or per-participant result.

- Hospital and laboratory data require approval from the corresponding author
  and the relevant data custodians. See `data/private/README.md`.
- The public rPPG-BP-UKL dataset must be downloaded from its original release.
  See `data/public/rppg_bp_ukl/README.md`.

Run `python scripts/check_release.py` immediately before every GitHub push.

## Repository map

```text
.
|-- config/                 Relative-path experiment configurations
|-- data/                   README files only; all real data are ignored
|-- docs/                   Full reproduction and protocol documentation
|-- figures/                Journal figure-generation scripts
|-- scripts/                Pipeline launcher and release privacy audit
|-- src/                    Processing, models, training, and statistics
|-- artifacts/README.md     Description of generated outputs
|-- environment.yml         Pinned Conda environment
`-- .gitignore              Raw/derived data and checkpoint exclusions
```

## Quick start

```bash
conda env create -f environment.yml
conda activate camera_bp_bspc
python scripts/check_release.py
python scripts/run_pipeline.py --stage preflight
```

For private-data processing, create a secret salt in the current shell before
building the manifest. Do not save the value in this repository.

Windows PowerShell:

```powershell
$env:BP_ID_SALT = -join ((1..48) | ForEach-Object { [char](Get-Random -Min 33 -Max 126) })
python scripts/run_pipeline.py --stage private-preprocessing
```

The full confirmatory run is:

```bash
python scripts/run_pipeline.py --stage primary
python scripts/run_pipeline.py --stage analysis
python scripts/run_pipeline.py --stage figures
```

The five-model run is computationally expensive. Use `--dry-run` to inspect the
exact command order without starting training.

## Documentation

- `docs/REPRODUCTION.md`: start-to-finish commands and dependency order.
- `docs/DATA_PREPARATION.md`: input schemas, rPPG extraction, and generated data.
- `docs/EXPERIMENTS.md`: splits, models, losses, metrics, and statistical tests.
- `docs/SCRIPT_INDEX.md`: purpose and prerequisites of every included script.
- `docs/OUTPUTS.md`: mapping from result files to manuscript tables and figures.
- `docs/PRIVACY_AND_RELEASE.md`: pre-upload safety and release checklist.
- `README_zh-CN.md`: concise Chinese guide.

## Reproducibility boundary

The configurations freeze the reported seed set, folds, preprocessing, and
training defaults. Reproduction of private-cohort numbers additionally requires
authorized access to the exact governed data snapshot. Public-only experiments
can be run after obtaining the original rPPG-BP-UKL file.

## Citation and license

Add the final article citation, repository DOI, author list, and a project license
after the submission metadata have been approved by all authors. Third-party data
remain governed by their original licenses and terms.
