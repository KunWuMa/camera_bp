from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import ensure_dirs, load_config, seed_everything
from models import build_model
from train_deep import load_arrays


@torch.no_grad()
def extract_features(model, windows, device, batch_size=256):
    model.eval(); output = []
    for start in range(0, len(windows), batch_size):
        x = torch.from_numpy(windows[start:start + batch_size]).to(device)
        _, feature, _ = model(x)
        output.append(feature.cpu().numpy())
    return np.concatenate(output)


def main():
    cfg = load_config(); seed_everything(cfg["seed"])
    root = ensure_dirs(cfg); data = load_arrays(root / "features" / "windows.h5")
    session_df = pd.read_csv(root / "features" / "session_features.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions, summaries = [], []
    for source in ("hospital", "lab"):
        idx = np.where(data["source"] == source)[0]
        local_windows = data["face"][idx]
        for fold in sorted(np.unique(data["fold"][idx])):
            fold = int(fold)
            model = build_model("snn_q", cfg["snn_steps"]).to(device)
            state = torch.load(root / "models" / f"{source}_fold{fold}_snn_q.pt", map_location=device)
            model.load_state_dict(state)
            z = extract_features(model, local_windows, device)
            embedded = pd.DataFrame(z, columns=[f"spike_{i}" for i in range(z.shape[1])])
            embedded["session_id"] = data["session"][idx]
            embedded = embedded.groupby("session_id", as_index=False).median()
            source_sessions = session_df[session_df.source == source].copy().merge(embedded, on="session_id", how="inner")
            rppg_cols = [c for c in source_sessions.columns if c.startswith("face_") and pd.api.types.is_numeric_dtype(source_sessions[c])]
            feature_cols = rppg_cols + [c for c in source_sessions if c.startswith("spike_")]
            train, test = source_sessions[source_sessions.fold != fold], source_sessions[source_sessions.fold == fold]
            targets = ["sbp", "dbp"] + (["hr"] if source == "lab" else [])
            regressor = MultiOutputRegressor(make_pipeline(SimpleImputer(), StandardScaler(), SVR(C=10, epsilon=.1)))
            regressor.fit(train[feature_cols], train[targets])
            pred = regressor.predict(test[feature_cols])
            for j, target in enumerate(targets):
                for (_, row), estimate in zip(test.iterrows(), pred[:, j]):
                    predictions.append({"source": source, "fold": fold, "model": "MorphSpike-Fusion", "target": target,
                                        "subject_id": row.subject_id, "session_id": row.session_id,
                                        "actual": row[target], "prediction": estimate})
            summaries.append({"source": source, "fold": fold, "train_sessions": len(train), "test_sessions": len(test),
                              "features": len(feature_cols), "spike_features": z.shape[1]})
            print(json.dumps(summaries[-1]), flush=True)
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(root / "results" / "fusion_predictions.csv", index=False, encoding="utf-8-sig")
    metric = pred_df.assign(abs_error=lambda d: abs(d.prediction - d.actual)).groupby(
        ["source", "model", "target"], as_index=False).agg(mae=("abs_error", "mean"), sessions=("session_id", "size"))
    metric.to_csv(root / "results" / "fusion_metrics.csv", index=False, encoding="utf-8-sig")
    print(metric.to_string(index=False))


if __name__ == "__main__": main()
