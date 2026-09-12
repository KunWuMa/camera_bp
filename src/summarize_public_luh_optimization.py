from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from common import ensure_dirs, load_config


def main():
    root = ensure_dirs(load_config())
    results = root / "results"
    base = pd.read_csv(results / "public_luh_predictions.csv")
    base = base[(base.protocol == "Subject-disjoint") & base.model.isin(["Face-ANN", "Face-SNN"])]
    base = base[["fold", "model", "target", "subject_id", "window_id", "actual", "prediction", "spike_rate"]]
    optimized = pd.read_csv(results / "public_luh_optimized_predictions.csv")
    v2 = pd.read_csv(results / "public_luh_v2_predictions.csv")
    pred = pd.concat([base, optimized, v2], ignore_index=True)
    pred["abs_error"] = (pred.prediction - pred.actual).abs()
    fold = pred.groupby(["fold", "model", "target"], as_index=False).agg(mae=("abs_error", "mean"))
    per_subject = pred.groupby(["model", "target", "subject_id"], as_index=False).agg(
        subject_mae=("abs_error", "mean"))
    summary = fold.groupby(["model", "target"], as_index=False).agg(
        mae=("mae", "mean"), fold_sd=("mae", "std"))
    macro = per_subject.groupby(["model", "target"], as_index=False).agg(
        macro_subject_mae=("subject_mae", "mean"), subject_mae_sd=("subject_mae", "std"))
    summary = summary.merge(macro, on=["model", "target"])
    comparisons = []
    for target in ("sbp", "dbp"):
        pivot = per_subject[per_subject.target == target].pivot(index="subject_id", columns="model", values="subject_mae")
        for model in [name for name in pivot.columns if name != "Face-ANN"]:
            statistic, pvalue = wilcoxon(pivot[model], pivot["Face-ANN"], alternative="two-sided")
            comparisons.append({"target": target, "model": model,
                                "mean_subject_mae_delta_vs_ann": float((pivot[model] - pivot["Face-ANN"]).mean()),
                                "wilcoxon_statistic": float(statistic), "wilcoxon_p": float(pvalue),
                                "subjects": len(pivot)})
    comparison = pd.DataFrame(comparisons)
    summary.to_csv(results / "public_luh_model_comparison.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(results / "public_luh_model_comparison_tests.csv", index=False, encoding="utf-8-sig")
    wide = summary.pivot(index="model", columns="target", values="mae")
    lines = [
        "# Public rPPG-LUH SNN optimization audit",
        "",
        "All variants use the same four outer subject-disjoint folds. These are exploratory ablations; the public outer folds had already been observed before optimization.",
        "",
        "| Model | SBP MAE | DBP MAE |",
        "|---|---:|---:|",
    ]
    for model in ["Face-ANN", "Face-SNN", "Face-SNN-32", "Face-SNN-32-KD", "Face-SNN-V2"]:
        lines.append(f"| {model} | {wide.loc[model, 'sbp']:.2f} | {wide.loc[model, 'dbp']:.2f} |")
    lines += [
        "",
        "The 32-step SNN improves DBP but slightly worsens SBP; ANN-to-SNN distillation does not improve outer-fold accuracy. The enhanced state-statistics readout also fails to surpass ANN. No variant supports an SNN-accuracy superiority claim.",
        "",
        "| Target | Model | Subject-MAE delta vs ANN | Wilcoxon p |",
        "|---|---|---:|---:|",
        *[f"| {row.target.upper()} | {row.model} | {row.mean_subject_mae_delta_vs_ann:.2f} | {row.wilcoxon_p:.4f} |"
          for row in comparison.itertuples()],
        "",
    ]
    (results / "public_luh_optimization_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
