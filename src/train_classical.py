from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import ensure_dirs, load_config, seed_everything


def models(seed: int):
    return {
        "Ridge-rPPG": make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0)),
        "SVR-rPPG": MultiOutputRegressor(make_pipeline(SimpleImputer(), StandardScaler(), SVR(C=10, epsilon=.1))),
        "ExtraTrees-rPPG": make_pipeline(SimpleImputer(), ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=4, max_features=.8, random_state=seed, n_jobs=-1)),
        "HistGB-rPPG": MultiOutputRegressor(make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(
            max_iter=300, learning_rate=.04, l2_regularization=2, random_state=seed))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    df = pd.read_csv(root / "features" / "session_features.csv")
    feature_cols = [c for c in df if c.startswith("face_") and c not in {"face_method", "face_spectral_hr"}]
    output = []
    fold_metrics = []
    for source, sdf in df.groupby("source"):
        targets = ["sbp", "dbp"] + (["hr"] if sdf["hr"].notna().all() else [])
        for fold in sorted(sdf["fold"].unique()):
            train = sdf[sdf.fold != fold].copy()
            test = sdf[sdf.fold == fold].copy()
            xtr, xte = train[feature_cols], test[feature_cols]
            ytr, yte = train[targets].to_numpy(float), test[targets].to_numpy(float)
            candidates = models(cfg["seed"] + int(fold))
            for name, model in candidates.items():
                model.fit(xtr, ytr)
                pred = np.asarray(model.predict(xte)).reshape(len(test), len(targets))
                for i, target in enumerate(targets):
                    for (_, row), actual, estimate in zip(test.iterrows(), yte[:, i], pred[:, i]):
                        output.append({"source": source, "fold": fold, "model": name, "target": target,
                                       "subject_id": row.subject_id, "session_id": row.session_id,
                                       "actual": actual, "prediction": estimate})
                    fold_metrics.append({"source": source, "fold": fold, "model": name, "target": target,
                                         "mae": mean_absolute_error(yte[:, i], pred[:, i]),
                                         "rmse": mean_squared_error(yte[:, i], pred[:, i], squared=False)})
            if source == "hospital":
                demo_cols = ["age", "height_cm", "weight_kg"]
                demo_train = train[demo_cols].copy()
                demo_test = test[demo_cols].copy()
                demo_train["sex"] = train.sex.astype(str).map({"男": 1.0, "女": 0.0})
                demo_test["sex"] = test.sex.astype(str).map({"男": 1.0, "女": 0.0})
                demo_model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=10.0))
                demo_model.fit(demo_train, ytr)
                pred = np.asarray(demo_model.predict(demo_test)).reshape(len(test), len(targets))
                for i, target in enumerate(targets):
                    for (_, row), actual, estimate in zip(test.iterrows(), yte[:, i], pred[:, i]):
                        output.append({"source": source, "fold": fold, "model": "Ridge-Demographics", "target": target,
                                       "subject_id": row.subject_id, "session_id": row.session_id,
                                       "actual": actual, "prediction": estimate})
            # Honest no-feature baseline: training-fold target median.
            baseline = np.tile(np.nanmedian(ytr, axis=0), (len(test), 1))
            for i, target in enumerate(targets):
                for (_, row), actual, estimate in zip(test.iterrows(), yte[:, i], baseline[:, i]):
                    output.append({"source": source, "fold": fold, "model": "TrainMedian", "target": target,
                                   "subject_id": row.subject_id, "session_id": row.session_id,
                                   "actual": actual, "prediction": estimate})
    pred_path = root / "results" / "classical_predictions.csv"
    pd.DataFrame(output).to_csv(pred_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_metrics).to_csv(root / "results" / "classical_fold_metrics.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"predictions": len(output), "path": str(pred_path), "features": len(feature_cols)}, indent=2))


if __name__ == "__main__":
    main()
