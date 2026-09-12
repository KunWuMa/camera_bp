# Public rPPG-BP-UKL data

The public replication uses the rPPG-BP-UKL release associated with Fabian
Schrumpf and colleagues. This repository does not re-host that dataset.

Original project and data directory:

- https://github.com/Fabian-Sc85/non-invasive-bp-estimation-using-deep-learning
- https://github.com/Fabian-Sc85/non-invasive-bp-estimation-using-deep-learning/tree/main/data

Associated publications:

- https://doi.org/10.1109/CVPRW53098.2021.00424
- https://doi.org/10.3390/s21186022

Download the dataset under the original authors' terms and place the required
file at:

```text
data/public/rppg_bp_ukl/rPPG-BP-UKL_rppg_7s.h5
```

Then run:

```bash
python src/inspect_public_luh.py
python scripts/run_pipeline.py --stage primary
```

The released code expects HDF5 datasets named `rppg`, `label`, and
`subject_idx`. Do not rename or re-index subjects. Subject identifiers are used
to construct the subject-disjoint folds.
