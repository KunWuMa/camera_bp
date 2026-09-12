from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd

from common import ensure_dirs, load_config, public_dataset_path


def main():
    dataset = public_dataset_path(load_config())
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    with h5py.File(dataset, "r") as handle:
        shapes = {name: list(handle[name].shape) for name in ("rppg", "label", "subject_idx")}
    root = ensure_dirs(load_config())
    audit = pd.read_csv(root / "results" / "public_luh_split_audit.csv")
    predictions = pd.read_csv(root / "results" / "public_luh_predictions.csv")
    optimized = pd.read_csv(root / "results" / "public_luh_optimized_predictions.csv")
    v2 = pd.read_csv(root / "results" / "public_luh_v2_predictions.csv")
    strict = audit[audit.protocol == "Subject-disjoint"]
    checks = {
        "sha256": digest,
        "sha256_matches": digest == "0ed453691e12d723085c9665ff9e557261df0dd3a9f7b50371c35703d3dc5e1c",
        "shapes": shapes,
        "prediction_rows": len(predictions),
        "expected_prediction_rows": 7851 * 2 * 3 * 2,
        "strict_folds": len(strict),
        "strict_overlap_all_zero": bool((strict.train_test_subject_overlap == 0).all()),
        "finite_predictions": bool(predictions[["actual", "prediction"]].notna().all().all()),
        "optimized_prediction_rows": len(optimized),
        "expected_optimized_prediction_rows": 7851 * 2 * 2,
        "v2_prediction_rows": len(v2),
        "expected_v2_prediction_rows": 7851 * 2,
        "finite_optimized_predictions": bool(optimized[["actual", "prediction"]].notna().all().all()),
        "finite_v2_predictions": bool(v2[["actual", "prediction"]].notna().all().all()),
    }
    print(json.dumps(checks, indent=2))
    if not all((checks["sha256_matches"], checks["prediction_rows"] == checks["expected_prediction_rows"],
                checks["strict_folds"] == 4, checks["strict_overlap_all_zero"], checks["finite_predictions"],
                checks["optimized_prediction_rows"] == checks["expected_optimized_prediction_rows"],
                checks["v2_prediction_rows"] == checks["expected_v2_prediction_rows"],
                checks["finite_optimized_predictions"], checks["finite_v2_predictions"])):
        raise SystemExit("public LUH validation failed")


if __name__ == "__main__":
    main()
