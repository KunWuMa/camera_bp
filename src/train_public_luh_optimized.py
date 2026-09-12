from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ensure_dirs, load_config, public_dataset_path, seed_everything
from models import ANNRegressor, SpikingRegressor, parameter_count
from train_public_luh import LuhDataset, infer, load_luh, make_weights, split_indices, subject_folds, train_model


def train_distilled(student, teacher, train_loader, val_loader, device, center, scale, cfg):
    student, teacher = student.to(device), teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    best_state, best_score, stale, history = None, float("inf"), 0, []
    for epoch in range(cfg["epochs"]):
        student.train()
        losses, rates = [], []
        for x, y, weights, _ in train_loader:
            x, y, weights = x.to(device), y.to(device), weights.to(device)
            with torch.no_grad():
                teacher_prediction, _, _ = teacher(x)
            prediction, _, spike_rate = student(x)
            supervised = torch.nn.functional.smooth_l1_loss(prediction, y, reduction="none")
            distilled = torch.nn.functional.smooth_l1_loss(prediction, teacher_prediction, reduction="none")
            element = supervised + 0.25 * distilled
            loss = (element * weights[:, None]).sum() / (weights.sum() * y.shape[1]) + 1e-3 * spike_rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            rates.append(float(spike_rate.detach()))
        scheduler.step()
        val_rows = infer(student, val_loader, device, center, scale)
        actual = np.stack([row[1] for row in val_rows])
        prediction = np.stack([row[2] for row in val_rows])
        score = float(np.abs(actual - prediction).mean())
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mae": score,
                        "spike_rate": float(np.mean(rates))})
        if score < best_score - 1e-4:
            best_state, best_score, stale = copy.deepcopy(student.state_dict()), score, 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    student.load_state_dict(best_state)
    return student, history, best_score


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

        seed_everything(cfg["seed"] + fold * 101)
        teacher, _, _ = train_model(
            ANNRegressor(outputs=2), loaders[0], loaders[1], device, center, scale,
            cfg["epochs"], cfg["patience"], cfg["learning_rate"], cfg["weight_decay"],
        )
        variants = (
            ("Face-SNN-32", False),
            ("Face-SNN-32-KD", True),
        )
        for variant_no, (name, use_kd) in enumerate(variants):
            seed_everything(cfg["seed"] + fold * 101 + (variant_no + 1) * 10007)
            started = time.time()
            student = SpikingRegressor(steps=32, outputs=2)
            if use_kd:
                student, history, best_val = train_distilled(
                    student, teacher, loaders[0], loaders[1], device, center, scale, cfg)
            else:
                student, history, best_val = train_model(
                    student, loaders[0], loaders[1], device, center, scale,
                    cfg["epochs"], cfg["patience"], cfg["learning_rate"], cfg["weight_decay"],
                )
            for row in history:
                histories.append({"fold": fold, "model": name, **row})
            for local, actual, prediction, spike_rate in infer(student, loaders[2], device, center, scale):
                global_idx = datasets[2].indices[local]
                for target_idx, target in enumerate(("sbp", "dbp")):
                    predictions.append({"fold": fold, "model": name, "target": target,
                                        "subject_id": int(subjects[global_idx]), "window_id": int(global_idx),
                                        "actual": float(actual[target_idx]), "prediction": float(prediction[target_idx]),
                                        "spike_rate": spike_rate})
            runs.append({"fold": fold, "model": name, "parameters": parameter_count(student),
                         "steps": 32, "epochs": len(history), "best_val_mae": best_val,
                         "seconds": time.time() - started})
            print(json.dumps(runs[-1]), flush=True)

    results = root / "results"
    pred = pd.DataFrame(predictions)
    pred["abs_error"] = (pred.prediction - pred.actual).abs()
    fold_metrics = pred.groupby(["fold", "model", "target"], as_index=False).agg(
        mae=("abs_error", "mean"), spike_rate=("spike_rate", "mean"),
        windows=("window_id", "size"), subjects=("subject_id", "nunique"))
    subject_metrics = pred.groupby(["fold", "model", "target", "subject_id"], as_index=False).agg(
        subject_mae=("abs_error", "mean"))
    macro = subject_metrics.groupby(["model", "target"], as_index=False).agg(
        macro_subject_mae=("subject_mae", "mean"), subject_mae_sd=("subject_mae", "std"))
    summary = fold_metrics.groupby(["model", "target"], as_index=False).agg(
        mae=("mae", "mean"), fold_sd=("mae", "std"), spike_rate=("spike_rate", "mean"))
    summary = summary.merge(macro, on=["model", "target"])
    pred.to_csv(results / "public_luh_optimized_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(results / "public_luh_optimized_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "public_luh_optimized_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(results / "public_luh_optimized_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runs).to_csv(results / "public_luh_optimized_runs.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
