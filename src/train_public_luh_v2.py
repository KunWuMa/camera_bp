from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ensure_dirs, load_config, public_dataset_path, seed_everything
from models import EnhancedSpikingRegressor, parameter_count
from train_public_luh import LuhDataset, infer, load_luh, make_weights, split_indices, subject_folds, train_model


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    data_path = public_dataset_path(cfg)
    x, labels, subjects = load_luh(data_path)
    assignment = subject_folds(subjects)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, histories, runs = [], [], []
    for fold in range(4):
        train_idx, val_idx, test_idx = split_indices("Subject-disjoint", assignment, subjects, fold, cfg["seed"])
        center = np.median(labels[train_idx], axis=0).astype(np.float32)
        scale = np.std(labels[train_idx], axis=0).astype(np.float32)
        scale[scale < 1e-5] = 1.0
        scaled = (labels - center) / scale
        weights = make_weights(subjects, train_idx)
        datasets = [LuhDataset(x, scaled, subjects, idx, weights, augment=(part == 0))
                    for part, idx in enumerate((train_idx, val_idx, test_idx))]
        loaders = [DataLoader(ds, cfg["batch_size"] * (1 if part == 0 else 2), shuffle=(part == 0),
                              num_workers=0, pin_memory=torch.cuda.is_available())
                   for part, ds in enumerate(datasets)]
        seed_everything(cfg["seed"] + fold * 101 + 30021)
        started = time.time()
        model, history, best_val = train_model(
            EnhancedSpikingRegressor(steps=32, outputs=2), loaders[0], loaders[1], device, center, scale,
            cfg["epochs"], cfg["patience"], cfg["learning_rate"], cfg["weight_decay"],
        )
        for row in history:
            histories.append({"fold": fold, "model": "Face-SNN-V2", **row})
        for local, actual, prediction, spike_rate in infer(model, loaders[2], device, center, scale):
            global_idx = datasets[2].indices[local]
            for target_idx, target in enumerate(("sbp", "dbp")):
                predictions.append({"fold": fold, "model": "Face-SNN-V2", "target": target,
                                    "subject_id": int(subjects[global_idx]), "window_id": int(global_idx),
                                    "actual": float(actual[target_idx]), "prediction": float(prediction[target_idx]),
                                    "spike_rate": spike_rate})
        runs.append({"fold": fold, "model": "Face-SNN-V2", "parameters": parameter_count(model),
                     "steps": 32, "epochs": len(history), "best_val_mae": best_val,
                     "seconds": time.time() - started, "beta": float(torch.sigmoid(model.beta_logit).detach().cpu())})
        print(json.dumps(runs[-1]), flush=True)
    results = root / "results"
    pred = pd.DataFrame(predictions)
    pred["abs_error"] = (pred.prediction - pred.actual).abs()
    folds = pred.groupby(["fold", "model", "target"], as_index=False).agg(
        mae=("abs_error", "mean"), spike_rate=("spike_rate", "mean"), windows=("window_id", "size"),
        subjects=("subject_id", "nunique"))
    per_subject = pred.groupby(["fold", "model", "target", "subject_id"], as_index=False).agg(
        subject_mae=("abs_error", "mean"))
    macro = per_subject.groupby(["model", "target"], as_index=False).agg(
        macro_subject_mae=("subject_mae", "mean"), subject_mae_sd=("subject_mae", "std"))
    summary = folds.groupby(["model", "target"], as_index=False).agg(
        mae=("mae", "mean"), fold_sd=("mae", "std"), spike_rate=("spike_rate", "mean"))
    summary = summary.merge(macro, on=["model", "target"])
    pred.to_csv(results / "public_luh_v2_predictions.csv", index=False, encoding="utf-8-sig")
    folds.to_csv(results / "public_luh_v2_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "public_luh_v2_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(results / "public_luh_v2_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runs).to_csv(results / "public_luh_v2_runs.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
