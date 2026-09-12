# Troubleshooting

## `BP_ID_SALT` error

The private manifest deliberately refuses to run without a secret salt of at
least 16 characters. Set it in the current terminal and rerun. Do not write it to
the configuration file.

## Public HDF5 not found

Place `rPPG-BP-UKL_rppg_7s.h5` at the path configured by `public_h5`. The default
is `data/public/rppg_bp_ukl/`. Run `python src/inspect_public_luh.py` before
training.

## CUDA out of memory

Create ignored `config/local.json`, lower `batch_size`, and pass it to scripts
that expose `--config`. Record the change because it alters the training setup.
The architecture runners write checkpoints that can support resumption.

## Missing OpenCV face detector

Confirm the Conda `opencv` package is installed and that
`cv2.data.haarcascades/haarcascade_frontalface_default.xml` exists.

## Existing cache is reused

`prepare_dataset.py` reuses `windows.h5` unless `--overwrite` is passed.
`extract_signals.py` behaves similarly. Use overwrite only after verifying the
authorized input snapshot and configuration.

## Model-suite runner cannot find baseline predictions

Run `run_bspc_extension_experiments.py` first. The model-suite runner intentionally
requires `artifacts/results/bspc_extension_predictions.csv` so all architectures
share the same audit structure.

## Results differ slightly across hardware

Seeds and deterministic CuDNN settings are fixed, but floating-point kernels,
GPU models, PyTorch builds, and driver versions can still cause small differences.
Retain the environment report and report all three seeds rather than a selected
run.
