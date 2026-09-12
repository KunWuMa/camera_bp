from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from common import ensure_dirs, load_config, public_dataset_path, seed_everything
from models import ANNRegressor, SpikingRegressor, parameter_count


class LuhDataset(Dataset):
    def __init__(self, x, y, subjects, indices, weights, augment=False):
        self.x = x[indices]
        self.y = y[indices]
        self.subjects = subjects[indices]
        self.indices = np.asarray(indices)
        self.weights = weights[indices]
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = torch.from_numpy(self.x[index])
        if self.augment:
            x = torch.roll(x, int(torch.randint(-20, 21, ()).item()))
            x = x + 0.015 * torch.randn_like(x)
        return x, torch.from_numpy(self.y[index]), torch.tensor(self.weights[index]), index


def load_luh(path: Path):
    with h5py.File(path, "r") as handle:
        x = np.asarray(handle["rppg"], dtype=np.float32).T
        y = np.asarray(handle["label"], dtype=np.float32).T
        subjects = np.asarray(handle["subject_idx"]).reshape(-1).astype(int)
    # The released signals are already standardized, but clip extreme extraction artifacts.
    x = np.clip(x, -8.0, 8.0)
    return x, y, subjects


def subject_folds(subjects, n_splits=4):
    assignment = np.full(len(subjects), -1, dtype=int)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (_, test_idx) in enumerate(splitter.split(np.zeros(len(subjects)), groups=subjects)):
        assignment[test_idx] = fold
    return assignment


def random_folds(n, seed, n_splits=4):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    assignment = np.empty(n, dtype=int)
    assignment[order] = np.arange(n) % n_splits
    return assignment


def split_indices(protocol, assignment, subjects, fold, seed):
    test_idx = np.where(assignment == fold)[0]
    outer = np.where(assignment != fold)[0]
    rng = np.random.default_rng(seed + fold * 1009)
    if protocol == "Subject-disjoint":
        candidates = np.unique(subjects[outer])
        n_val = max(1, int(np.ceil(0.2 * len(candidates))))
        val_subjects = rng.choice(candidates, size=n_val, replace=False)
        val_mask = np.isin(subjects[outer], val_subjects)
        return outer[~val_mask], outer[val_mask], test_idx
    val_fold = (fold + 1) % 4
    val_idx = np.where(assignment == val_fold)[0]
    train_idx = np.where((assignment != fold) & (assignment != val_fold))[0]
    return train_idx, val_idx, test_idx


def make_weights(subjects, train_idx):
    weights = np.ones(len(subjects), dtype=np.float32)
    unique, counts = np.unique(subjects[train_idx], return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    weights[train_idx] = np.asarray([1.0 / count_map[int(s)] for s in subjects[train_idx]], dtype=np.float32)
    weights[train_idx] /= weights[train_idx].mean()
    return weights


@torch.no_grad()
def infer(model, loader, device, center, scale):
    model.eval()
    rows = []
    for x, y, _, local in loader:
        prediction, _, spike_rate = model(x.to(device))
        prediction = prediction.cpu().numpy() * scale + center
        actual = y.numpy() * scale + center
        for j in range(len(x)):
            rows.append((int(local[j]), actual[j], prediction[j], float(spike_rate)))
    return rows


def train_model(model, train_loader, val_loader, device, center, scale, epochs, patience, lr, weight_decay):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_state, best_score, stale, history = None, float("inf"), 0, []
    for epoch in range(epochs):
        model.train()
        losses, rates = [], []
        for x, y, weights, _ in train_loader:
            x, y, weights = x.to(device), y.to(device), weights.to(device)
            prediction, _, spike_rate = model(x)
            element = torch.nn.functional.smooth_l1_loss(prediction, y, reduction="none")
            loss = (element * weights[:, None]).sum() / (weights.sum() * y.shape[1])
            if hasattr(model, "threshold"):
                loss = loss + 1e-3 * spike_rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            rates.append(float(spike_rate.detach()))
        scheduler.step()
        val_rows = infer(model, val_loader, device, center, scale)
        actual = np.stack([row[1] for row in val_rows])
        prediction = np.stack([row[2] for row in val_rows])
        score = float(np.abs(actual - prediction).mean())
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_mae": score,
                        "spike_rate": float(np.mean(rates))})
        if score < best_score - 1e-4:
            best_state, best_score, stale = copy.deepcopy(model.state_dict()), score, 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, history, best_score


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    data_path = public_dataset_path(cfg)
    x, labels, subjects = load_luh(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocols = {
        "Subject-disjoint": subject_folds(subjects),
        "Random-window": random_folds(len(subjects), cfg["seed"]),
    }
    predictions, histories, runs, split_audit = [], [], [], []
    model_factories = {
        "Face-ANN": lambda: ANNRegressor(outputs=2),
        "Face-SNN": lambda: SpikingRegressor(steps=cfg["snn_steps"], outputs=2),
    }
    for protocol, assignment in protocols.items():
        for fold in range(4):
            train_idx, val_idx, test_idx = split_indices(protocol, assignment, subjects, fold, cfg["seed"])
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
            split_audit.append({
                "protocol": protocol, "fold": fold,
                "train_windows": len(train_idx), "val_windows": len(val_idx), "test_windows": len(test_idx),
                "train_subjects": len(np.unique(subjects[train_idx])),
                "val_subjects": len(np.unique(subjects[val_idx])),
                "test_subjects": len(np.unique(subjects[test_idx])),
                "train_test_subject_overlap": len(set(subjects[train_idx]) & set(subjects[test_idx])),
            })
            baseline = np.tile(center, (len(test_idx), 1))
            for local, global_idx in enumerate(test_idx):
                for target_idx, target in enumerate(("sbp", "dbp")):
                    predictions.append({"protocol": protocol, "fold": fold, "model": "Train-median",
                                        "target": target, "subject_id": int(subjects[global_idx]),
                                        "window_id": int(global_idx), "actual": float(labels[global_idx, target_idx]),
                                        "prediction": float(baseline[local, target_idx]), "spike_rate": 0.0})
            for model_no, (model_name, factory) in enumerate(model_factories.items()):
                seed_everything(cfg["seed"] + fold * 101 + model_no * 10007 + (0 if protocol == "Subject-disjoint" else 50000))
                started = time.time()
                model, history, best_val = train_model(
                    factory(), loaders[0], loaders[1], device, center, scale,
                    cfg["epochs"], cfg["patience"], cfg["learning_rate"], cfg["weight_decay"],
                )
                for row in history:
                    histories.append({"protocol": protocol, "fold": fold, "model": model_name, **row})
                for local, actual, prediction, spike_rate in infer(model, loaders[2], device, center, scale):
                    global_idx = datasets[2].indices[local]
                    for target_idx, target in enumerate(("sbp", "dbp")):
                        predictions.append({"protocol": protocol, "fold": fold, "model": model_name,
                                            "target": target, "subject_id": int(subjects[global_idx]),
                                            "window_id": int(global_idx), "actual": float(actual[target_idx]),
                                            "prediction": float(prediction[target_idx]), "spike_rate": spike_rate})
                runs.append({"protocol": protocol, "fold": fold, "model": model_name,
                             "parameters": parameter_count(model), "epochs": len(history),
                             "best_val_mae": best_val, "seconds": time.time() - started})
                print(json.dumps(runs[-1]), flush=True)
    results_dir = root / "results"
    pred = pd.DataFrame(predictions)
    pred["abs_error"] = (pred["prediction"] - pred["actual"]).abs()
    fold_metrics = pred.groupby(["protocol", "fold", "model", "target"], as_index=False).agg(
        mae=("abs_error", "mean"), windows=("window_id", "size"), subjects=("subject_id", "nunique"))
    subject_metrics = pred.groupby(["protocol", "fold", "model", "target", "subject_id"], as_index=False).agg(
        subject_mae=("abs_error", "mean"))
    macro_metrics = subject_metrics.groupby(["protocol", "model", "target"], as_index=False).agg(
        macro_subject_mae=("subject_mae", "mean"), subject_mae_sd=("subject_mae", "std"), subjects=("subject_id", "nunique"))
    summary = fold_metrics.groupby(["protocol", "model", "target"], as_index=False).agg(
        mae=("mae", "mean"), fold_sd=("mae", "std"), folds=("fold", "nunique"))
    summary = summary.merge(macro_metrics, on=["protocol", "model", "target"], how="left")
    pred.to_csv(results_dir / "public_luh_predictions.csv", index=False, encoding="utf-8-sig")
    fold_metrics.to_csv(results_dir / "public_luh_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results_dir / "public_luh_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(histories).to_csv(results_dir / "public_luh_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(runs).to_csv(results_dir / "public_luh_runs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(split_audit).to_csv(results_dir / "public_luh_split_audit.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"device": str(device), "windows": len(x), "subjects": len(np.unique(subjects))}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
