from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from common import ensure_dirs, load_config, public_dataset_path, seed_everything
from models import ANNRegressor, SpikingRegressor, parameter_count
from train_deep import ArrayDataset, load_arrays, predict, train_one
from train_public_luh import (
    LuhDataset,
    infer,
    load_luh,
    make_weights,
    random_folds,
    split_indices,
    subject_folds,
    train_model,
)


TARGETS = (("sbp", 0), ("dbp", 1))


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("At least three training seeds are required for the BSPC stability study")
    return seeds


def private_protocols(data: dict, source: str, base: np.ndarray, base_seed: int) -> dict[str, np.ndarray]:
    n_splits = 5
    rng = np.random.default_rng(base_seed + (0 if source == "hospital" else 100_000))

    window_assignment = np.full(len(data["labels"]), -1, dtype=int)
    window_order = rng.permutation(base)
    window_assignment[window_order] = np.arange(len(window_order)) % n_splits

    session_assignment = np.full(len(data["labels"]), -1, dtype=int)
    sessions = np.unique(data["session"][base]).copy()
    rng.shuffle(sessions)
    session_map = {session_id: index % n_splits for index, session_id in enumerate(sessions)}
    for index in base:
        session_assignment[index] = session_map[data["session"][index]]

    return {
        "Random-window": window_assignment,
        "Random-session": session_assignment,
        "Subject-disjoint": data["fold"].astype(int),
    }


def private_split(
    data: dict,
    base: np.ndarray,
    source: str,
    protocol: str,
    assignment: np.ndarray,
    fold: int,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_idx = base[assignment[base] == fold]
    outer = base[assignment[base] != fold]
    if protocol != "Subject-disjoint":
        validation_fold = (fold + 1) % 5
        val_idx = base[assignment[base] == validation_fold]
        train_idx = base[(assignment[base] != fold) & (assignment[base] != validation_fold)]
        return train_idx, val_idx, test_idx

    candidates = np.unique(data["subject"][outer])
    rng = np.random.default_rng(base_seed + fold * 1009 + (0 if source == "hospital" else 100_000))
    n_val = max(1, int(np.ceil(0.1 * len(candidates))))
    val_subjects = rng.choice(candidates, size=n_val, replace=False)
    is_val = np.isin(data["subject"][outer], val_subjects)
    return outer[~is_val], outer[is_val], test_idx


def private_loaders(data: dict, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, cfg: dict):
    center = np.nanmedian(data["labels"][train_idx], axis=0).astype(np.float32)
    scale = np.nanstd(data["labels"][train_idx], axis=0).astype(np.float32)
    center[~np.isfinite(center)] = 0
    scale[~np.isfinite(scale) | (scale < 1e-5)] = 1
    scaled = (data["labels"] - center) / scale

    counts = Counter(data["session"][train_idx])
    weights = np.ones(len(data["labels"]), dtype=np.float32)
    for index in train_idx:
        weights[index] = 1.0 / counts[data["session"][index]]
    weights[train_idx] /= weights[train_idx].mean()

    datasets = [
        ArrayDataset(
            data["face"], data["finger"], scaled, data["quality"], data["subject"],
            data["session"], indices, weights, augment=(part == 0),
        )
        for part, indices in enumerate((train_idx, val_idx, test_idx))
    ]
    loaders = [
        DataLoader(
            dataset,
            batch_size=cfg["batch_size"] * (1 if part == 0 else 2),
            shuffle=(part == 0),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        for part, dataset in enumerate(datasets)
    ]
    return center, scale, datasets, loaders


def run_private(cfg: dict, seeds: list[int], data: dict, device: torch.device):
    rows: list[dict] = []
    audit: list[dict] = []
    run_count = 0
    for source in ("hospital", "lab"):
        base = np.where(data["source"] == source)[0]
        protocols = private_protocols(data, source, base, cfg["seed"])
        for protocol, assignment in protocols.items():
            folds = sorted(np.unique(assignment[base]).astype(int).tolist())
            for fold in folds:
                train_idx, val_idx, test_idx = private_split(
                    data, base, source, protocol, assignment, fold, cfg["seed"]
                )
                audit.append({
                    "cohort": source,
                    "protocol": protocol,
                    "fold": fold,
                    "train_windows": len(train_idx),
                    "validation_windows": len(val_idx),
                    "test_windows": len(test_idx),
                    "train_subjects": len(np.unique(data["subject"][train_idx])),
                    "validation_subjects": len(np.unique(data["subject"][val_idx])),
                    "test_subjects": len(np.unique(data["subject"][test_idx])),
                    "train_test_subject_overlap": len(
                        set(data["subject"][train_idx]) & set(data["subject"][test_idx])
                    ),
                    "train_test_session_overlap": len(
                        set(data["session"][train_idx]) & set(data["session"][test_idx])
                    ),
                })
                center, scale, datasets, loaders = private_loaders(data, train_idx, val_idx, test_idx, cfg)

                for seed_index, training_seed in enumerate(seeds):
                    median_prediction = np.tile(center, (len(test_idx), 1))
                    for local, global_index in enumerate(test_idx):
                        for target, target_index in TARGETS:
                            rows.append({
                                "cohort": source,
                                "protocol": protocol,
                                "seed": training_seed,
                                "fold": fold,
                                "model": "Train-median",
                                "target": target,
                                "subject_id": datasets[2].subjects[local],
                                "unit_id": datasets[2].sessions[local],
                                "unit_type": "acquisition",
                                "actual": float(data["labels"][global_index, target_index]),
                                "prediction": float(median_prediction[local, target_index]),
                                "spike_rate": 0.0,
                            })

                    for model_index, (internal_name, display_name) in enumerate(
                        (("ann", "Face-ANN"), ("snn", "Face-SNN"))
                    ):
                        run_seed = training_seed + fold * 101 + model_index * 10_007 + (
                            0 if protocol == "Subject-disjoint" else 50_000 if protocol == "Random-window" else 70_000
                        )
                        seed_everything(run_seed)
                        started = time.time()
                        model, history, best_val = train_one(
                            internal_name, loaders[0], loaders[1], device, cfg, center, scale
                        )
                        predictions = predict(model, loaders[2], device, center, scale)
                        for local, actual, estimate, spike_rate in predictions:
                            for target, target_index in TARGETS:
                                rows.append({
                                    "cohort": source,
                                    "protocol": protocol,
                                    "seed": training_seed,
                                    "fold": fold,
                                    "model": display_name,
                                    "target": target,
                                    "subject_id": datasets[2].subjects[local],
                                    "unit_id": datasets[2].sessions[local],
                                    "unit_type": "acquisition",
                                    "actual": float(actual[target_index]),
                                    "prediction": float(estimate[target_index]),
                                    "spike_rate": float(spike_rate),
                                })
                        run_count += 1
                        print(json.dumps({
                            "cohort": source,
                            "protocol": protocol,
                            "fold": fold,
                            "seed": training_seed,
                            "model": display_name,
                            "epochs": len(history),
                            "best_val_mae": best_val,
                            "seconds": time.time() - started,
                            "run": run_count,
                        }), flush=True)

    window = pd.DataFrame(rows)
    # Each acquisition contributes exactly once. Under random-window splitting, its windows may
    # be scored by several fold models, so fold is deliberately omitted before aggregation.
    session = window.groupby(
        ["cohort", "protocol", "seed", "model", "target", "subject_id", "unit_id", "unit_type"],
        as_index=False,
    ).agg(
        actual=("actual", "first"),
        prediction=("prediction", "median"),
        spike_rate=("spike_rate", "mean"),
    )
    session["fold"] = -1
    return session, pd.DataFrame(audit), run_count


def run_public(cfg: dict, seeds: list[int], device: torch.device, start_run: int):
    data_path = public_dataset_path(cfg)
    signals, labels, subjects = load_luh(data_path)
    protocols = {
        "Random-window": random_folds(len(subjects), cfg["seed"]),
        "Subject-disjoint": subject_folds(subjects),
    }
    rows: list[dict] = []
    audit: list[dict] = []
    run_count = start_run
    for protocol, assignment in protocols.items():
        for fold in range(4):
            train_idx, val_idx, test_idx = split_indices(
                protocol, assignment, subjects, fold, cfg["seed"]
            )
            audit.append({
                "cohort": "public_ukl",
                "protocol": protocol,
                "fold": fold,
                "train_windows": len(train_idx),
                "validation_windows": len(val_idx),
                "test_windows": len(test_idx),
                "train_subjects": len(np.unique(subjects[train_idx])),
                "validation_subjects": len(np.unique(subjects[val_idx])),
                "test_subjects": len(np.unique(subjects[test_idx])),
                "train_test_subject_overlap": len(set(subjects[train_idx]) & set(subjects[test_idx])),
                "train_test_session_overlap": np.nan,
            })
            center = np.median(labels[train_idx], axis=0).astype(np.float32)
            scale = np.std(labels[train_idx], axis=0).astype(np.float32)
            scale[scale < 1e-5] = 1
            scaled = (labels - center) / scale
            weights = make_weights(subjects, train_idx)
            datasets = [
                LuhDataset(signals, scaled, subjects, indices, weights, augment=(part == 0))
                for part, indices in enumerate((train_idx, val_idx, test_idx))
            ]
            loaders = [
                DataLoader(
                    dataset,
                    batch_size=cfg["batch_size"] * (1 if part == 0 else 2),
                    shuffle=(part == 0),
                    num_workers=0,
                    pin_memory=torch.cuda.is_available(),
                )
                for part, dataset in enumerate(datasets)
            ]
            factories = {
                "Face-ANN": lambda: ANNRegressor(outputs=2),
                "Face-SNN": lambda: SpikingRegressor(steps=cfg["snn_steps"], outputs=2),
            }
            for training_seed in seeds:
                baseline = np.tile(center, (len(test_idx), 1))
                for local, global_index in enumerate(test_idx):
                    for target, target_index in TARGETS:
                        rows.append({
                            "cohort": "public_ukl",
                            "protocol": protocol,
                            "seed": training_seed,
                            "fold": fold,
                            "model": "Train-median",
                            "target": target,
                            "subject_id": int(subjects[global_index]),
                            "unit_id": int(global_index),
                            "unit_type": "window",
                            "actual": float(labels[global_index, target_index]),
                            "prediction": float(baseline[local, target_index]),
                            "spike_rate": 0.0,
                        })
                for model_index, (model_name, factory) in enumerate(factories.items()):
                    run_seed = training_seed + fold * 101 + model_index * 10_007 + (
                        0 if protocol == "Subject-disjoint" else 50_000
                    )
                    seed_everything(run_seed)
                    started = time.time()
                    model, history, best_val = train_model(
                        factory(), loaders[0], loaders[1], device, center, scale,
                        cfg["epochs"], cfg["patience"], cfg["learning_rate"], cfg["weight_decay"],
                    )
                    for local, actual, estimate, spike_rate in infer(model, loaders[2], device, center, scale):
                        global_index = datasets[2].indices[local]
                        for target, target_index in TARGETS:
                            rows.append({
                                "cohort": "public_ukl",
                                "protocol": protocol,
                                "seed": training_seed,
                                "fold": fold,
                                "model": model_name,
                                "target": target,
                                "subject_id": int(subjects[global_index]),
                                "unit_id": int(global_index),
                                "unit_type": "window",
                                "actual": float(actual[target_index]),
                                "prediction": float(estimate[target_index]),
                                "spike_rate": float(spike_rate),
                            })
                    run_count += 1
                    print(json.dumps({
                        "cohort": "public_ukl",
                        "protocol": protocol,
                        "fold": fold,
                        "seed": training_seed,
                        "model": model_name,
                        "parameters": parameter_count(model),
                        "epochs": len(history),
                        "best_val_mae": best_val,
                        "seconds": time.time() - started,
                        "run": run_count,
                    }), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(audit), run_count


def seed_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["abs_error"] = (frame["prediction"] - frame["actual"]).abs()
    macro = frame.groupby(
        ["cohort", "protocol", "seed", "model", "target", "subject_id"], as_index=False
    ).agg(subject_mae=("abs_error", "mean"))
    macro = macro.groupby(
        ["cohort", "protocol", "seed", "model", "target"], as_index=False
    ).agg(macro_subject_mae=("subject_mae", "mean"), subjects=("subject_id", "nunique"))
    metric = frame.groupby(
        ["cohort", "protocol", "seed", "model", "target"], as_index=False
    ).agg(
        mae=("abs_error", "mean"),
        units=("unit_id", "nunique"),
        spike_rate=("spike_rate", "mean"),
    )
    return metric.merge(macro, on=["cohort", "protocol", "seed", "model", "target"], how="left")


def summarize_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics.groupby(["cohort", "protocol", "model", "target"], as_index=False).agg(
        mean_mae=("mae", "mean"),
        seed_sd_mae=("mae", "std"),
        mean_macro_subject_mae=("macro_subject_mae", "mean"),
        seed_sd_macro_subject_mae=("macro_subject_mae", "std"),
        mean_spike_rate=("spike_rate", "mean"),
        seeds=("seed", "nunique"),
        units=("units", "max"),
        subjects=("subjects", "max"),
    )


def paired_leakage_statistics(predictions: pd.DataFrame, base_seed: int, repeats: int = 5000) -> pd.DataFrame:
    # Average predictions across training seeds before inference. This avoids treating seeds as
    # independent participants and leaves subjects as the statistical unit.
    ensemble = predictions.groupby(
        ["cohort", "protocol", "model", "target", "subject_id", "unit_id", "unit_type"],
        as_index=False,
    ).agg(actual=("actual", "first"), prediction=("prediction", "mean"))
    ensemble["abs_error"] = (ensemble["prediction"] - ensemble["actual"]).abs()
    subject = ensemble.groupby(
        ["cohort", "protocol", "model", "target", "subject_id"], as_index=False
    ).agg(subject_mae=("abs_error", "mean"))

    rows: list[dict] = []
    rng = np.random.default_rng(base_seed + 909)
    for (cohort, model, target), group in subject.groupby(["cohort", "model", "target"]):
        comparators = ["Random-window"]
        if cohort in {"hospital", "lab"}:
            comparators.append("Random-session")
        strict = group[group["protocol"] == "Subject-disjoint"][["subject_id", "subject_mae"]]
        for comparator in comparators:
            loose = group[group["protocol"] == comparator][["subject_id", "subject_mae"]]
            paired = strict.merge(loose, on="subject_id", suffixes=("_strict", "_comparator"))
            if paired.empty:
                continue
            delta = paired["subject_mae_strict"].to_numpy() - paired["subject_mae_comparator"].to_numpy()
            choices = rng.integers(0, len(delta), size=(repeats, len(delta)))
            bootstrap = delta[choices].mean(axis=1)
            try:
                wilcoxon_p = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
            except ValueError:
                wilcoxon_p = 1.0
            strict_mae = float(paired["subject_mae_strict"].mean())
            comparator_mae = float(paired["subject_mae_comparator"].mean())
            rows.append({
                "cohort": cohort,
                "comparison": f"{comparator}_vs_Subject-disjoint",
                "model": model,
                "target": target,
                "subjects": len(paired),
                "comparator_macro_mae": comparator_mae,
                "strict_macro_mae": strict_mae,
                "delta_strict_minus_comparator": float(delta.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "optimism_percent": float(100 * delta.mean() / strict_mae) if strict_mae else np.nan,
                "wilcoxon_p": wilcoxon_p,
            })
    result = pd.DataFrame(rows)
    order = np.argsort(result["wilcoxon_p"].to_numpy())
    adjusted = np.empty(len(result), dtype=float)
    running = 0.0
    total = len(result)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(result.iloc[index]["wilcoxon_p"]))
        running = max(running, candidate)
        adjusted[index] = running
    result["holm_adjusted_p"] = adjusted
    result["holm_significant_0_05"] = result["holm_adjusted_p"] < 0.05
    return result.sort_values(["cohort", "comparison", "model", "target"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Run the additional multi-seed BSPC leakage experiments")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default="20260907,20260908,20260909")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_arrays(root / "features" / "windows.h5")

    private_predictions, private_audit, run_count = run_private(cfg, seeds, data, device)
    public_predictions, public_audit, run_count = run_public(cfg, seeds, device, run_count)
    predictions = pd.concat([private_predictions, public_predictions], ignore_index=True)
    metrics = seed_metrics(predictions)
    summary = summarize_seeds(metrics)
    leakage = paired_leakage_statistics(predictions, cfg["seed"], args.bootstrap_repeats)
    audit = pd.concat([private_audit, public_audit], ignore_index=True)

    results = root / "results"
    predictions.to_csv(results / "bspc_extension_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results / "bspc_extension_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "bspc_extension_seed_summary.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(results / "bspc_extension_leakage_statistics.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(results / "bspc_extension_split_audit.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({
        "device": str(device),
        "training_seeds": seeds,
        "training_runs": run_count,
        "prediction_rows": len(predictions),
        "bootstrap_repeats": args.bootstrap_repeats,
    }, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)
    print(leakage.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
