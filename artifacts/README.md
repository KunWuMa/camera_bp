# Generated artifacts

Pipeline outputs are written here at runtime and are excluded from Git.

Expected subdirectories:

- `manifests/`: pseudonymized acquisition manifest and split assignments.
- `signals/`: per-acquisition extracted RGB/rPPG arrays.
- `features/`: window HDF5 files and acquisition descriptors.
- `models/`: fold checkpoints.
- `results/`: predictions, subject-macro summaries, and statistical tests.
- `figures/`: generated vector PDF and PNG figures.
- `logs/`: environment and validation reports.

Even pseudonymized manifests and predictions are not part of the public code
release because linkage and membership risks remain. Share only under the
approved data-use process.
