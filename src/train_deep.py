from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import ensure_dirs, load_config, seed_everything
from models import build_model, parameter_count


class ArrayDataset(Dataset):
    def __init__(self, x, teacher_x, y, quality, subjects, sessions, indices, weights, augment=False):
        self.x, self.y, self.quality = x[indices], y[indices], quality[indices]
        self.teacher_x = teacher_x[indices]
        self.subjects = subjects[indices]
        self.sessions = sessions[indices]
        self.weights = weights[indices]
        self.augment = augment

    def __len__(self): return len(self.x)

    def __getitem__(self, i):
        x = torch.from_numpy(self.x[i])
        if self.augment:
            x = torch.roll(x, int(torch.randint(-15, 16, ()).item())) + .02 * torch.randn_like(x)
        return (x, torch.from_numpy(self.teacher_x[i]), torch.from_numpy(self.y[i]),
                torch.tensor(self.quality[i], dtype=torch.float32),
                torch.tensor(self.weights[i], dtype=torch.float32), i)


def masked_huber(pred, target, sample_weight, target_weight=None):
    mask = torch.isfinite(target)
    safe_target = torch.where(mask, target, pred.detach())
    loss = torch.nn.functional.smooth_l1_loss(pred, safe_target, reduction="none")
    target_weight = pred.new_ones(pred.shape[1]) if target_weight is None else target_weight
    combined = mask * sample_weight[:, None] * target_weight[None, :]
    return (loss * combined).sum() / combined.sum().clamp_min(1)


def relational_loss(student_feature, teacher_feature):
    s = torch.nn.functional.normalize(student_feature, dim=1)
    t = torch.nn.functional.normalize(teacher_feature, dim=1)
    return torch.nn.functional.smooth_l1_loss(s @ s.T, t @ t.T)


@torch.no_grad()
def predict(model, loader, device, center, scale):
    model.eval()
    rows = []
    for x, teacher_x, y, q, w, local_idx in loader:
        pred, _, spike_rate = model(x.to(device))
        pred = pred.cpu().numpy() * scale + center
        y = y.numpy() * scale + center
        for k in range(len(x)):
            rows.append((int(local_idx[k]), y[k], pred[k], float(spike_rate)))
    return rows


def train_one(name, train_loader, val_loader, device, cfg, center, scale, teacher=None):
    model = build_model(name, cfg["snn_steps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    best, best_score, stale, history = None, float("inf"), 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        losses, rates = [], []
        for x, teacher_x, y, q, session_weight, _ in train_loader:
            x, y = x.to(device), y.to(device)
            # Smoothly downweight very noisy face traces without deleting difficult samples.
            q = q.to(device)
            reliability = torch.sigmoid((q[:, 0] + 3.0) / 4.0) * (.5 + .5 * q[:, 2].clamp(0, 1))
            use_quality = name in {"snn_q", "snn_kd"}
            sample_weight = session_weight.to(device) * (reliability if use_quality else 1.0)
            pred, feature, spike_rate = model(x)
            target_weight = pred.new_tensor([1.0, 1.0, cfg["aux_hr_weight"]])
            loss = masked_huber(pred, y, sample_weight, target_weight)
            if name in {"snn_kd", "snn_kd_noq"} and teacher is not None:
                teacher.eval()
                with torch.no_grad():
                    teacher_pred, teacher_feature, _ = teacher(teacher_x.to(device))
                paired = q[:, 1] > -29.0
                kd_output = masked_huber(pred, teacher_pred, paired.to(pred.dtype), target_weight)
                kd_relation = relational_loss(feature[paired], teacher_feature[paired]) if paired.sum() > 1 else pred.new_zeros(())
                loss = loss + cfg["distill_weight"] * (kd_output + .25 * kd_relation)
            if name.startswith("snn"):
                loss = loss + 1e-3 * spike_rate
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            rates.append(float(spike_rate.detach()))
        scheduler.step()
        val_rows = predict(model, val_loader, device, center, scale)
        actual = np.stack([r[1] for r in val_rows])[:, :2]
        estimated = np.stack([r[2] for r in val_rows])[:, :2]
        score = float(np.nanmean(np.abs(actual - estimated)))
        history.append({"epoch": epoch + 1, "train_loss": np.mean(losses), "val_mae_scaled": score,
                        "spike_rate": np.mean(rates) if rates else 0})
        if score < best_score - 1e-4:
            best_score, best, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    model.load_state_dict(best)
    return model, history, best_score


def load_arrays(path):
    with h5py.File(path, "r") as h5:
        decode = lambda a: np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in a])
        return {"face": h5["face"][:], "finger": h5["finger"][:], "labels": h5["labels"][:],
                "quality": h5["quality"][:], "subject": decode(h5["subject_id"][:]),
                "session": decode(h5["session_id"][:]), "source": decode(h5["source"][:]),
                "fold": h5["fold"][:]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--sources", default="hospital,lab")
    parser.add_argument("--models", default="teacher,ann,snn,snn_q,snn_kd_noq,snn_kd")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    data = load_arrays(root / "features" / "windows.h5")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_models = args.models.split(",")
    all_predictions, all_history, run_summary = [], [], []
    for source in args.sources.split(","):
        source_mask = data["source"] == source
        source_folds = sorted(np.unique(data["fold"][source_mask]))
        for raw_test_fold in source_folds:
            test_fold = int(raw_test_fold)
            test_idx = np.where(source_mask & (data["fold"] == test_fold))[0]
            outer_train_idx = np.where(source_mask & (data["fold"] != test_fold))[0]
            candidate_subjects = np.unique(data["subject"][outer_train_idx])
            rng = np.random.default_rng(cfg["seed"] + 1009 * test_fold + (0 if source == "hospital" else 100000))
            val_subjects = rng.choice(candidate_subjects, size=max(1, int(np.ceil(.1 * len(candidate_subjects)))), replace=False)
            is_val = np.isin(data["subject"][outer_train_idx], val_subjects)
            val_idx, train_idx = outer_train_idx[is_val], outer_train_idx[~is_val]
            center = np.nanmedian(data["labels"][train_idx], axis=0).astype(np.float32)
            scale = np.nanstd(data["labels"][train_idx], axis=0).astype(np.float32)
            center[~np.isfinite(center)] = 0; scale[~np.isfinite(scale) | (scale < 1e-5)] = 1
            y_scaled = (data["labels"] - center) / scale
            counts = Counter(data["session"][train_idx])
            weights = np.ones(len(data["labels"]), np.float32)
            for i in train_idx: weights[i] = 1.0 / counts[data["session"][i]]
            weights[train_idx] /= weights[train_idx].mean()

            def loaders(signal_name):
                split_indices = [train_idx, val_idx, test_idx]
                if signal_name == "finger":
                    split_indices = [idx[data["quality"][idx, 1] > -29.0] for idx in split_indices]
                datasets = [ArrayDataset(data[signal_name], data["finger"], y_scaled, data["quality"], data["subject"],
                                         data["session"], idx, weights, augment=(split_no == 0 and signal_name == "face"))
                            for split_no, idx in enumerate(split_indices)]
                return (DataLoader(datasets[0], cfg["batch_size"], shuffle=True, num_workers=0),
                        DataLoader(datasets[1], cfg["batch_size"] * 2, shuffle=False, num_workers=0),
                        DataLoader(datasets[2], cfg["batch_size"] * 2, shuffle=False, num_workers=0), datasets[2])

            teacher_model = None
            # Teacher is trained on finger PPG. Its predictions are also a performance upper reference.
            if "teacher" in requested_models or "snn_kd" in requested_models:
                tr, va, te, test_ds = loaders("finger")
                teacher_model, history, score = train_one("teacher", tr, va, device, cfg, center, scale)
                torch.save(teacher_model.state_dict(), root / "models" / f"{source}_fold{test_fold}_teacher.pt")
                if "teacher" in requested_models:
                    rows = predict(teacher_model, te, device, center, scale)
                    for local, actual, pred, rate in rows:
                        for j, target in enumerate(("sbp", "dbp", "hr")):
                            if np.isfinite(actual[j]):
                                all_predictions.append({"source": source, "fold": test_fold, "model": "PPG-Teacher",
                                    "target": target, "subject_id": test_ds.subjects[local], "session_id": test_ds.sessions[local],
                                    "actual": actual[j], "prediction": pred[j], "spike_rate": rate})
                all_history += [{"source": source, "fold": test_fold, "model": "teacher", **h} for h in history]

            for name in [m for m in requested_models if m != "teacher"]:
                tr, va, te, test_ds = loaders("face")
                start = time.time()
                model, history, score = train_one(name, tr, va, device, cfg, center, scale,
                                                   teacher=teacher_model if name in {"snn_kd", "snn_kd_noq"} else None)
                torch.save(model.state_dict(), root / "models" / f"{source}_fold{test_fold}_{name}.pt")
                rows = predict(model, te, device, center, scale)
                display = {"ann": "Face-ANN", "snn": "Face-SNN", "snn_q": "Face-SNN+Quality",
                           "snn_kd_noq": "Face-SNN+Distill", "snn_kd": "SpikeBP-Distill"}[name]
                for local, actual, pred, rate in rows:
                    for j, target in enumerate(("sbp", "dbp", "hr")):
                        if np.isfinite(actual[j]):
                            all_predictions.append({"source": source, "fold": test_fold, "model": display,
                                "target": target, "subject_id": test_ds.subjects[local], "session_id": test_ds.sessions[local],
                                "actual": actual[j], "prediction": pred[j], "spike_rate": rate})
                all_history += [{"source": source, "fold": test_fold, "model": name, **h} for h in history]
                run_summary.append({"source": source, "fold": test_fold, "model": name,
                                    "parameters": parameter_count(model), "epochs": len(history),
                                    "best_val_mae_scaled": score, "seconds": time.time() - start,
                                    "test_windows": len(test_idx)})
                print(json.dumps(run_summary[-1]), flush=True)

    pd.DataFrame(all_predictions).to_csv(root / "results" / "deep_window_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_history).to_csv(root / "results" / "deep_history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(run_summary).to_csv(root / "results" / "deep_run_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"device": str(device), "predictions": len(all_predictions), "runs": len(run_summary)}, indent=2))


if __name__ == "__main__":
    main()
