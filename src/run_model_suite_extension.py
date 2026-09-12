from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from common import ensure_dirs, load_config, public_dataset_path, seed_everything
from models import (
    ResNet1DRegressor,
    TCNRegressor,
    TransformerRegressor,
    build_model,
    parameter_count,
)
from run_bspc_extension_experiments import (
    TARGETS,
    paired_leakage_statistics,
    parse_seeds,
    private_loaders,
    private_protocols,
    private_split,
    seed_metrics,
    summarize_seeds,
)
from train_deep import load_arrays, predict, train_one
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


MODEL_SPECS = (
    ("resnet1d", "ResNet-1D", ResNet1DRegressor),
    ("tcn", "TCN", TCNRegressor),
    ("transformer", "Transformer", TransformerRegressor),
)


def checkpoint_rows(path: Path) -> tuple[list[dict], set[tuple]]:
    if not path.exists():
        return [], set()
    frame = pd.read_pickle(path)
    keys = set(
        frame[["cohort", "protocol", "fold", "seed", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    return frame.to_dict("records"), keys


def run_private_models(
    cfg: dict, seeds: list[int], data: dict, device: torch.device, checkpoint_path: Path
):
    rows, completed = checkpoint_rows(checkpoint_path)
    run_count = len(completed)
    for source in ("hospital", "lab"):
        base = np.where(data["source"] == source)[0]
        protocols = private_protocols(data, source, base, cfg["seed"])
        for protocol, assignment in protocols.items():
            folds = sorted(np.unique(assignment[base]).astype(int).tolist())
            for fold in folds:
                train_idx, val_idx, test_idx = private_split(
                    data, base, source, protocol, assignment, fold, cfg["seed"]
                )
                center, scale, datasets, loaders = private_loaders(
                    data, train_idx, val_idx, test_idx, cfg
                )
                for training_seed in seeds:
                    for model_index, (internal_name, display_name, _) in enumerate(MODEL_SPECS, start=2):
                        key = (source, protocol, fold, training_seed, display_name)
                        if key in completed:
                            continue
                        run_seed = training_seed + fold * 101 + model_index * 10_007 + (
                            0
                            if protocol == "Subject-disjoint"
                            else 50_000
                            if protocol == "Random-window"
                            else 70_000
                        )
                        seed_everything(run_seed)
                        started = time.time()
                        model, history, best_val = train_one(
                            internal_name, loaders[0], loaders[1], device, cfg, center, scale
                        )
                        predictions = predict(model, loaders[2], device, center, scale)
                        for local, actual, estimate, spike_rate in predictions:
                            for target, target_index in TARGETS:
                                rows.append(
                                    {
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
                                    }
                                )
                        run_count += 1
                        completed.add(key)
                        pd.DataFrame(rows).to_pickle(checkpoint_path)
                        print(
                            json.dumps(
                                {
                                    "cohort": source,
                                    "protocol": protocol,
                                    "fold": fold,
                                    "seed": training_seed,
                                    "model": display_name,
                                    "parameters": parameter_count(model),
                                    "epochs": len(history),
                                    "best_val_mae": best_val,
                                    "seconds": time.time() - started,
                                    "run": run_count,
                                }
                            ),
                            flush=True,
                        )

    window = pd.DataFrame(rows)
    session = window.groupby(
        ["cohort", "protocol", "seed", "model", "target", "subject_id", "unit_id", "unit_type"],
        as_index=False,
    ).agg(
        actual=("actual", "first"),
        prediction=("prediction", "median"),
        spike_rate=("spike_rate", "mean"),
    )
    session["fold"] = -1
    return session, run_count


def run_public_models(
    cfg: dict, seeds: list[int], device: torch.device, start_run: int, checkpoint_path: Path
):
    data_path = public_dataset_path(cfg)
    signals, labels, subjects = load_luh(data_path)
    protocols = {
        "Random-window": random_folds(len(subjects), cfg["seed"]),
        "Subject-disjoint": subject_folds(subjects),
    }
    rows, completed = checkpoint_rows(checkpoint_path)
    run_count = start_run + len(completed)
    for protocol, assignment in protocols.items():
        for fold in range(4):
            train_idx, val_idx, test_idx = split_indices(
                protocol, assignment, subjects, fold, cfg["seed"]
            )
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
            for training_seed in seeds:
                for model_index, (internal_name, display_name, factory) in enumerate(MODEL_SPECS, start=2):
                    key = ("public_ukl", protocol, fold, training_seed, display_name)
                    if key in completed:
                        continue
                    run_seed = training_seed + fold * 101 + model_index * 10_007 + (
                        0 if protocol == "Subject-disjoint" else 50_000
                    )
                    seed_everything(run_seed)
                    started = time.time()
                    model, history, best_val = train_model(
                        factory(outputs=2),
                        loaders[0],
                        loaders[1],
                        device,
                        center,
                        scale,
                        cfg["epochs"],
                        cfg["patience"],
                        cfg["learning_rate"],
                        cfg["weight_decay"],
                    )
                    for local, actual, estimate, spike_rate in infer(
                        model, loaders[2], device, center, scale
                    ):
                        global_index = datasets[2].indices[local]
                        for target, target_index in TARGETS:
                            rows.append(
                                {
                                    "cohort": "public_ukl",
                                    "protocol": protocol,
                                    "seed": training_seed,
                                    "fold": fold,
                                    "model": display_name,
                                    "target": target,
                                    "subject_id": int(subjects[global_index]),
                                    "unit_id": int(global_index),
                                    "unit_type": "window",
                                    "actual": float(actual[target_index]),
                                    "prediction": float(estimate[target_index]),
                                    "spike_rate": float(spike_rate),
                                }
                            )
                    run_count += 1
                    completed.add(key)
                    pd.DataFrame(rows).to_pickle(checkpoint_path)
                    print(
                        json.dumps(
                            {
                                "cohort": "public_ukl",
                                "protocol": protocol,
                                "fold": fold,
                                "seed": training_seed,
                                "model": display_name,
                                "parameters": parameter_count(model),
                                "epochs": len(history),
                                "best_val_mae": best_val,
                                "seconds": time.time() - started,
                                "run": run_count,
                            }
                        ),
                        flush=True,
                    )
    return pd.DataFrame(rows), run_count


def paired_architecture_statistics(
    predictions: pd.DataFrame, base_seed: int, repeats: int = 5000
) -> pd.DataFrame:
    ensemble = predictions.groupby(
        ["cohort", "protocol", "model", "target", "subject_id", "unit_id", "unit_type"],
        as_index=False,
    ).agg(actual=("actual", "first"), prediction=("prediction", "mean"))
    ensemble["abs_error"] = (ensemble["prediction"] - ensemble["actual"]).abs()
    subject = ensemble.groupby(
        ["cohort", "protocol", "model", "target", "subject_id"], as_index=False
    ).agg(subject_mae=("abs_error", "mean"))
    subject = subject[subject["protocol"].eq("Subject-disjoint")]

    rows: list[dict] = []
    rng = np.random.default_rng(base_seed + 1919)
    candidate_models = ["Face-SNN", "ResNet-1D", "TCN", "Transformer"]
    for cohort in sorted(subject["cohort"].unique()):
        for target in ("sbp", "dbp"):
            ann = subject[
                subject["cohort"].eq(cohort)
                & subject["target"].eq(target)
                & subject["model"].eq("Face-ANN")
            ][["subject_id", "subject_mae"]]
            for model in candidate_models:
                candidate = subject[
                    subject["cohort"].eq(cohort)
                    & subject["target"].eq(target)
                    & subject["model"].eq(model)
                ][["subject_id", "subject_mae"]]
                paired = candidate.merge(ann, on="subject_id", suffixes=("_model", "_ann"))
                delta = paired["subject_mae_model"].to_numpy() - paired["subject_mae_ann"].to_numpy()
                choices = rng.integers(0, len(delta), size=(repeats, len(delta)))
                bootstrap = delta[choices].mean(axis=1)
                try:
                    p_value = float(stats.wilcoxon(delta, alternative="two-sided").pvalue)
                except ValueError:
                    p_value = 1.0
                rows.append(
                    {
                        "cohort": cohort,
                        "protocol": "Subject-disjoint",
                        "model": model,
                        "reference": "Face-ANN",
                        "target": target,
                        "subjects": len(paired),
                        "model_macro_mae": float(paired["subject_mae_model"].mean()),
                        "ann_macro_mae": float(paired["subject_mae_ann"].mean()),
                        "delta_model_minus_ann": float(delta.mean()),
                        "ci_low": float(np.quantile(bootstrap, .025)),
                        "ci_high": float(np.quantile(bootstrap, .975)),
                        "wilcoxon_p": p_value,
                    }
                )
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
    result["holm_significant_0_05"] = result["holm_adjusted_p"] < .05
    return result.sort_values(["cohort", "model", "target"]).reset_index(drop=True)


def complexity_table() -> pd.DataFrame:
    rows = []
    for internal_name, display_name in (
        ("ann", "Face-ANN"),
        ("snn", "Face-SNN"),
        ("resnet1d", "ResNet-1D"),
        ("tcn", "TCN"),
        ("transformer", "Transformer"),
    ):
        model = build_model(internal_name)
        rows.append(
            {
                "model": display_name,
                "trainable_parameters_outputs3": parameter_count(model),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Add ResNet-1D, TCN and Transformer to the BSPC audit")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default="20260907,20260908,20260909")
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = parse_seeds(args.seeds)
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke_test:
        public_path = public_dataset_path(cfg)
        public_signals, _, _ = load_luh(public_path)
        for internal_name, display_name, _ in MODEL_SPECS:
            model = build_model(internal_name).to(device)
            private_output, _, _ = model(torch.randn(4, 300, device=device))
            output, feature, spike_rate = model(
                torch.from_numpy(public_signals[:4]).to(device)
            )
            print(
                json.dumps(
                    {
                        "model": display_name,
                        "parameters": parameter_count(model),
                        "private_output_shape": list(private_output.shape),
                        "public_input_length": int(public_signals.shape[1]),
                        "output_shape": list(output.shape),
                        "feature_shape": list(feature.shape),
                        "spike_rate": float(spike_rate),
                    }
                )
            )
        return

    data = load_arrays(root / "features" / "windows.h5")
    results = root / "results"
    private_checkpoint = results / ".bspc_model_suite_private_checkpoint.pkl"
    public_checkpoint = results / ".bspc_model_suite_public_checkpoint.pkl"
    private_predictions, run_count = run_private_models(
        cfg, seeds, data, device, private_checkpoint
    )
    public_predictions, run_count = run_public_models(
        cfg, seeds, device, run_count, public_checkpoint
    )
    added = pd.concat([private_predictions, public_predictions], ignore_index=True)

    existing_path = root / "results" / "bspc_extension_predictions.csv"
    existing = pd.read_csv(existing_path, encoding="utf-8-sig")
    predictions = pd.concat([existing, added], ignore_index=True)
    metrics = seed_metrics(predictions)
    summary = summarize_seeds(metrics)
    leakage = paired_leakage_statistics(predictions, cfg["seed"], args.bootstrap_repeats)
    architecture = paired_architecture_statistics(predictions, cfg["seed"], args.bootstrap_repeats)
    complexity = complexity_table()

    added.to_csv(results / "bspc_added_deep_predictions.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(results / "bspc_model_suite_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(results / "bspc_model_suite_seed_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(results / "bspc_model_suite_seed_summary.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(results / "bspc_model_suite_leakage_statistics.csv", index=False, encoding="utf-8-sig")
    architecture.to_csv(results / "bspc_model_suite_architecture_statistics.csv", index=False, encoding="utf-8-sig")
    complexity.to_csv(results / "bspc_model_suite_complexity.csv", index=False, encoding="utf-8-sig")
    private_checkpoint.unlink(missing_ok=True)
    public_checkpoint.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "device": str(device),
                "training_seeds": seeds,
                "new_training_runs": run_count,
                "added_prediction_rows": len(added),
                "suite_prediction_rows": len(predictions),
                "bootstrap_repeats": args.bootstrap_repeats,
            },
            indent=2,
        ),
        flush=True,
    )
    print(summary.to_string(index=False), flush=True)
    print(architecture.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
