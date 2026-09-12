from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import ensure_dirs, load_config, seed_everything
from models import build_model
from train_deep import load_arrays
from train_snn_fusion import extract_features


def main():
    cfg = load_config(); seed_everything(cfg["seed"])
    root = ensure_dirs(cfg); data = load_arrays(root / "features" / "windows.h5")
    sessions = pd.read_csv(root / "features" / "session_features.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, selections = [], []
    grid = {"svr__C": [.1, 1, 10, 100], "svr__epsilon": [.1, 1, 2], "svr__gamma": ["scale", .01, .1]}
    for source in ("hospital", "lab"):
        idx = np.where(data["source"] == source)[0]
        for fold in sorted(np.unique(data["fold"][idx])):
            fold = int(fold)
            snn = build_model("snn_q", cfg["snn_steps"]).to(device)
            snn.load_state_dict(torch.load(root / "models" / f"{source}_fold{fold}_snn_q.pt", map_location=device))
            z = extract_features(snn, data["face"][idx], device)
            embedded = pd.DataFrame(z, columns=[f"spike_{i}" for i in range(z.shape[1])])
            embedded["session_id"] = data["session"][idx]
            embedded = embedded.groupby("session_id", as_index=False).median()
            table = sessions[sessions.source == source].copy().merge(embedded, on="session_id")
            feature_cols = [c for c in table if (c.startswith("face_") and pd.api.types.is_numeric_dtype(table[c])) or c.startswith("spike_")]
            train, test = table[table.fold != fold], table[table.fold == fold]
            for target in (["sbp", "dbp", "hr"] if source == "lab" else ["sbp", "dbp"]):
                estimator = make_pipeline(SimpleImputer(), StandardScaler(), SVR())
                search = GridSearchCV(estimator, grid, scoring="neg_mean_absolute_error",
                                      cv=GroupKFold(n_splits=3), n_jobs=-1, refit=True)
                search.fit(train[feature_cols], train[target], groups=train.subject_id)
                pred = search.predict(test[feature_cols])
                for (_, row), estimate in zip(test.iterrows(), pred):
                    predictions.append({"source": source, "fold": fold, "model": "MorphSpike-Nested", "target": target,
                                        "subject_id": row.subject_id, "session_id": row.session_id,
                                        "actual": row[target], "prediction": estimate})
                selections.append({"source": source, "fold": fold, "target": target,
                                   "inner_mae": -search.best_score_, **search.best_params_})
                print(json.dumps(selections[-1]), flush=True)
    pred = pd.DataFrame(predictions)
    pred.to_csv(root / "results" / "fusion_nested_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(root / "results" / "fusion_nested_selections.csv", index=False, encoding="utf-8-sig")
    metric = pred.assign(abs_error=lambda d: abs(d.prediction - d.actual)).groupby(
        ["source", "model", "target"], as_index=False).agg(mae=("abs_error", "mean"), sessions=("session_id", "size"))
    metric.to_csv(root / "results" / "fusion_nested_metrics.csv", index=False, encoding="utf-8-sig")
    print(metric.to_string(index=False))


if __name__ == "__main__": main()
