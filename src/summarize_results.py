from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import ensure_dirs, load_config, seed_everything


def metrics(group):
    actual = group.actual.to_numpy(float)
    pred = group.prediction.to_numpy(float)
    err = pred - actual
    subject_macro_mae = group.assign(abs_error=np.abs(err)).groupby("subject_id").abs_error.mean().mean()
    return pd.Series({"n_sessions": len(group), "mae": np.mean(np.abs(err)),
                      "subject_macro_mae": subject_macro_mae,
                      "rmse": np.sqrt(np.mean(err ** 2)), "bias": np.mean(err),
                      "error_sd": np.std(err, ddof=1),
                      "pearson_r": stats.pearsonr(actual, pred).statistic if len(group) > 2 and np.std(pred) > 0 else np.nan,
                      "r2": 1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2) if len(group) > 2 else np.nan})


def subject_bootstrap(group, seed, repeats=2000):
    rng = np.random.default_rng(seed)
    by_subject = group.assign(abs_error=np.abs(group.prediction - group.actual)).groupby("subject_id").abs_error.agg(["sum", "count"])
    choices = rng.integers(0, len(by_subject), size=(repeats, len(by_subject)))
    sums = by_subject["sum"].to_numpy()[choices].sum(axis=1)
    counts = by_subject["count"].to_numpy()[choices].sum(axis=1)
    values = sums / counts
    return np.quantile(values, [.025, .975])


def plot_bland_altman(group, title, path):
    actual, pred = group.actual.to_numpy(float), group.prediction.to_numpy(float)
    mean, diff = (actual + pred) / 2, pred - actual
    bias, sd = diff.mean(), diff.std(ddof=1)
    fig, ax = plt.subplots(figsize=(4.2, 3.3), constrained_layout=True)
    ax.scatter(mean, diff, s=15, alpha=.6, edgecolors="none")
    ax.axhline(bias, color="#C43C39", label=f"bias={bias:.1f}")
    ax.axhline(bias + 1.96 * sd, color="#4C72B0", ls="--")
    ax.axhline(bias - 1.96 * sd, color="#4C72B0", ls="--")
    ax.set(title=title, xlabel="Mean reference and estimate (mmHg)", ylabel="Estimate - reference (mmHg)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def paired_comparison(predictions, source, target, model_a, model_b, seed):
    a = predictions[(predictions.source == source) & (predictions.target == target) & (predictions.model == model_a)]
    b = predictions[(predictions.source == source) & (predictions.target == target) & (predictions.model == model_b)]
    paired = a.merge(b, on=["source", "target", "subject_id", "session_id"], suffixes=("_a", "_b"))
    if paired.empty: return None
    paired["delta_abs_error"] = np.abs(paired.prediction_a - paired.actual_a) - np.abs(paired.prediction_b - paired.actual_b)
    by_subject = paired.groupby("subject_id").delta_abs_error.agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    choices = rng.integers(0, len(by_subject), size=(2000, len(by_subject)))
    delta = by_subject["sum"].to_numpy()[choices].sum(1) / by_subject["count"].to_numpy()[choices].sum(1)
    try: pvalue = stats.wilcoxon(paired.delta_abs_error).pvalue
    except ValueError: pvalue = np.nan
    return {"source": source, "target": target, "model_a": model_a, "model_b": model_b,
            "n_common_sessions": len(paired), "mae_delta_a_minus_b": paired.delta_abs_error.mean(),
            "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975), "wilcoxon_p": pvalue}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    root = ensure_dirs(cfg)
    parts = []
    deep_path = root / "results" / "deep_window_predictions.csv"
    classical_path = root / "results" / "classical_predictions.csv"
    if deep_path.exists():
        deep = pd.read_csv(deep_path)
        # One estimate per acquisition; prevents longer videos from dominating metrics.
        deep = deep.groupby(["source", "fold", "model", "target", "subject_id", "session_id"], as_index=False).agg(
            actual=("actual", "first"), prediction=("prediction", "median"), spike_rate=("spike_rate", "mean"))
        parts.append(deep)
    if classical_path.exists():
        parts.append(pd.read_csv(classical_path))
    if not parts:
        raise FileNotFoundError("No prediction files found")
    predictions = pd.concat(parts, ignore_index=True)
    ensemble_rows = []
    for deep_model, ensemble_name in (("Face-SNN+Quality", "Hybrid-SNN-SVR"),
                                      ("SpikeBP-Distill", "Hybrid-SpikeBP-SVR")):
        a = predictions[predictions.model == deep_model]
        b = predictions[predictions.model == "SVR-rPPG"]
        paired = a.merge(b, on=["source", "fold", "target", "subject_id", "session_id"], suffixes=("_a", "_b"))
        if len(paired):
            ensemble_rows.append(pd.DataFrame({"source": paired.source, "fold": paired.fold,
                "model": ensemble_name, "target": paired.target, "subject_id": paired.subject_id,
                "session_id": paired.session_id, "actual": paired.actual_a,
                "prediction": .5 * (paired.prediction_a + paired.prediction_b),
                "spike_rate": paired.spike_rate_a if "spike_rate_a" in paired else np.nan}))
    if ensemble_rows:
        predictions = pd.concat([predictions, *ensemble_rows], ignore_index=True)
    predictions.to_csv(root / "results" / "session_predictions.csv", index=False, encoding="utf-8-sig")
    fold_rows = []
    for key, group in predictions.groupby(["source", "fold", "model", "target"]):
        fold_rows.append({"source": key[0], "fold": key[1], "model": key[2], "target": key[3], **metrics(group).to_dict()})
    pd.DataFrame(fold_rows).to_csv(root / "results" / "all_fold_metrics.csv", index=False, encoding="utf-8-sig")
    rows = []
    for key, group in predictions.groupby(["source", "model", "target"]):
        row = {"source": key[0], "model": key[1], "target": key[2], **metrics(group).to_dict()}
        lo, hi = subject_bootstrap(group, cfg["seed"])
        row.update(mae_ci_low=lo, mae_ci_high=hi)
        if "spike_rate" in group:
            row["spike_rate"] = group.spike_rate.mean()
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["source", "target", "mae"])
    table.to_csv(root / "results" / "main_metrics.csv", index=False, encoding="utf-8-sig")
    table.to_latex(root / "results" / "main_metrics.tex", index=False, float_format="%.2f")

    selected = table[table.model.isin(["TrainMedian", "Face-ANN", "Face-SNN", "Face-SNN+Quality",
                                        "Face-SNN+Distill", "SpikeBP-Distill"])]
    for source in selected.source.unique():
        panel = selected[(selected.source == source) & selected.target.isin(["sbp", "dbp"])]
        pivot = panel.pivot(index="model", columns="target", values="mae")
        order = [m for m in ["TrainMedian", "Face-ANN", "Face-SNN", "Face-SNN+Quality",
                              "Face-SNN+Distill", "SpikeBP-Distill"] if m in pivot.index]
        pivot = pivot.loc[order]
        ax = pivot.rename(columns={"sbp": "SBP", "dbp": "DBP"}).plot.bar(figsize=(6.4, 3.2), color=["#4C72B0", "#DD8452"])
        ax.set(ylabel="MAE (mmHg)", xlabel="", title=f"{source.title()} subject-disjoint evaluation")
        ax.tick_params(axis="x", rotation=25, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        ax.figure.savefig(root / "figures" / f"model_mae_{source}.png", dpi=300)
        plt.close(ax.figure)

    comparisons = []
    for source in predictions.source.unique():
        for target in ("sbp", "dbp"):
            for baseline in ("Face-SNN", "Face-SNN+Quality", "Face-SNN+Distill", "Face-ANN", "TrainMedian"):
                row = paired_comparison(predictions, source, target, "SpikeBP-Distill", baseline, cfg["seed"])
                if row: comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(root / "results" / "paired_comparisons.csv", index=False, encoding="utf-8-sig")

    primary = predictions[(predictions.model == "SpikeBP-Distill") & predictions.target.isin(["sbp", "dbp"])]
    for (source, target), group in primary.groupby(["source", "target"]):
        plot_bland_altman(group, f"{source.title()} {target.upper()}", root / "figures" / f"bland_altman_{source}_{target}.png")
    print(json.dumps({"session_predictions": len(predictions), "metric_rows": len(table),
                      "best": table.groupby(["source", "target"]).first().reset_index()[["source", "target", "model", "mae"]].to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
