from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common import ensure_dirs, load_config, seed_everything
from train_deep import ArrayDataset, load_arrays, predict, train_one


def main():
    cfg = load_config()
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    data = load_arrays(root / "features" / "windows.h5")
    base = np.where(data["source"] == "lab")[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    protocols = {}
    rng = np.random.default_rng(cfg["seed"])
    window_order = rng.permutation(base)
    window_fold = np.empty(len(data["labels"]), dtype=int); window_fold.fill(-1)
    window_fold[window_order] = np.arange(len(window_order)) % 5
    protocols["Random-window"] = window_fold
    sessions = np.unique(data["session"][base]); rng.shuffle(sessions)
    session_map = {sid: i % 5 for i, sid in enumerate(sessions)}
    session_fold = np.full(len(data["labels"]), -1, dtype=int)
    for i in base: session_fold[i] = session_map[data["session"][i]]
    protocols["Random-session"] = session_fold
    output, audit = [], []
    for protocol, assignment in protocols.items():
        for fold in range(5):
            test_idx = base[assignment[base] == fold]
            val_idx = base[assignment[base] == ((fold + 1) % 5)]
            train_idx = base[(assignment[base] != fold) & (assignment[base] != ((fold + 1) % 5))]
            center = np.nanmedian(data["labels"][train_idx], axis=0).astype(np.float32)
            scale = np.nanstd(data["labels"][train_idx], axis=0).astype(np.float32); scale[scale < 1e-5] = 1
            y = (data["labels"] - center) / scale
            counts = Counter(data["session"][train_idx]); weights = np.ones(len(y), np.float32)
            for i in train_idx: weights[i] = 1 / counts[data["session"][i]]
            weights[train_idx] /= weights[train_idx].mean()
            datasets = [ArrayDataset(data["face"], data["finger"], y, data["quality"], data["subject"],
                                     data["session"], idx, weights, augment=(k == 0))
                        for k, idx in enumerate((train_idx, val_idx, test_idx))]
            loaders = (DataLoader(datasets[0], cfg["batch_size"], shuffle=True),
                       DataLoader(datasets[1], cfg["batch_size"] * 2),
                       DataLoader(datasets[2], cfg["batch_size"] * 2))
            audit.append({"protocol": protocol, "fold": fold,
                "train_test_subject_overlap": len(set(data["subject"][train_idx]) & set(data["subject"][test_idx])),
                "train_test_session_overlap": len(set(data["session"][train_idx]) & set(data["session"][test_idx]))})
            for name, display in (("ann", "Face-ANN"), ("snn", "Face-SNN")):
                model, history, _ = train_one(name, loaders[0], loaders[1], device, cfg, center, scale)
                for local, actual, pred, rate in predict(model, loaders[2], device, center, scale):
                    for j, target in enumerate(("sbp", "dbp", "hr")):
                        output.append({"protocol": protocol, "fold": fold, "model": display, "target": target,
                            "subject_id": datasets[2].subjects[local], "session_id": datasets[2].sessions[local],
                            "actual": actual[j], "prediction": pred[j]})
                print(json.dumps({"protocol": protocol, "fold": fold, "model": name, "epochs": len(history)}), flush=True)
    pred = pd.DataFrame(output)
    pred.to_csv(root / "results" / "leakage_window_predictions.csv", index=False, encoding="utf-8-sig")
    session = pred.groupby(["protocol", "fold", "model", "target", "subject_id", "session_id"], as_index=False).agg(
        actual=("actual", "first"), prediction=("prediction", "median"))
    metric = session.assign(abs_error=lambda d: abs(d.prediction - d.actual)).groupby(
        ["protocol", "model", "target"], as_index=False).agg(mae=("abs_error", "mean"), sessions=("session_id", "size"))
    strict = pd.read_csv(root / "results" / "main_metrics.csv")
    strict = strict[(strict.source == "lab") & strict.model.isin(["Face-ANN", "Face-SNN"])][["model", "target", "mae", "n_sessions"]]
    strict.insert(0, "protocol", "Subject-disjoint")
    strict = strict.rename(columns={"n_sessions": "sessions"})
    metric = pd.concat([metric, strict], ignore_index=True)
    metric.to_csv(root / "results" / "leakage_protocol_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(root / "results" / "leakage_split_audit.csv", index=False, encoding="utf-8-sig")
    print(metric.to_string(index=False))


if __name__ == "__main__": main()
