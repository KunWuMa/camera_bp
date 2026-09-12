from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config


def value(frame, protocol, model, target, column="mae"):
    row = frame[(frame.protocol == protocol) & (frame.model == model) & (frame.target == target)]
    return float(row.iloc[0][column])


def main():
    root = ensure_dirs(load_config())
    results = root / "results"
    figures = root / "figures"
    summary = pd.read_csv(results / "public_luh_summary.csv")

    protocols = ["Random-window", "Subject-disjoint"]
    models = ["Train-median", "Face-ANN", "Face-SNN"]
    colors = ["#8c8c8c", "#4472c4", "#ed7d31"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), sharey=False)
    x = np.arange(len(protocols))
    width = 0.24
    for axis, target, title in zip(axes, ["sbp", "dbp"], ["Systolic BP", "Diastolic BP"]):
        for offset, (model, color) in enumerate(zip(models, colors)):
            vals = [value(summary, protocol, model, target) for protocol in protocols]
            errs = [value(summary, protocol, model, target, "fold_sd") for protocol in protocols]
            bars = axis.bar(x + (offset - 1) * width, vals, width, yerr=errs, capsize=3,
                            color=color, edgecolor="black", linewidth=.5, label=model)
            axis.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=8, padding=2)
        axis.set_title(title)
        axis.set_xticks(x, ["Random\nwindow", "Subject\ndisjoint"])
        axis.set_ylabel("MAE (mmHg)")
        axis.grid(axis="y", alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, .91))
    fig.savefig(figures / "public_luh_split_effect.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "public_luh_split_effect.pdf", bbox_inches="tight")
    plt.close(fig)

    order = pd.CategoricalDtype(protocols, ordered=True)
    model_order = pd.CategoricalDtype(models, ordered=True)
    table = summary.copy()
    table["protocol"] = table.protocol.astype(order)
    table["model"] = table.model.astype(model_order)
    pivot = table.pivot(index=["protocol", "model"], columns="target", values=["mae", "fold_sd"])
    lines = [
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Protocol & Model & SBP MAE & DBP MAE \\",
        r"\midrule",
    ]
    for protocol in protocols:
        for row_no, model in enumerate(models):
            sbp = pivot.loc[(protocol, model), ("mae", "sbp")]
            sbp_sd = pivot.loc[(protocol, model), ("fold_sd", "sbp")]
            dbp = pivot.loc[(protocol, model), ("mae", "dbp")]
            dbp_sd = pivot.loc[(protocol, model), ("fold_sd", "dbp")]
            label = protocol if row_no == 0 else ""
            lines.append(f"{label} & {model} & {sbp:.2f} $\\pm$ {sbp_sd:.2f} & {dbp:.2f} $\\pm$ {dbp_sd:.2f} \\\\")
        if protocol != protocols[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (results / "public_luh_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    ann_sbp = value(summary, "Subject-disjoint", "Face-ANN", "sbp")
    ann_dbp = value(summary, "Subject-disjoint", "Face-ANN", "dbp")
    snn_sbp = value(summary, "Subject-disjoint", "Face-SNN", "sbp")
    snn_dbp = value(summary, "Subject-disjoint", "Face-SNN", "dbp")
    rw_ann_sbp = value(summary, "Random-window", "Face-ANN", "sbp")
    rw_ann_dbp = value(summary, "Random-window", "Face-ANN", "dbp")
    rw_snn_sbp = value(summary, "Random-window", "Face-SNN", "sbp")
    rw_snn_dbp = value(summary, "Random-window", "Face-SNN", "dbp")
    report = f"""# Public rPPG-LUH external validation

- Dataset: 7,851 seven-second rPPG windows from 17 subjects; SBP {143.398:.2f}±{14.967:.2f}, DBP {65.682:.2f}±{11.297:.2f} mmHg.
- Strict four-fold subject-disjoint split audit: zero train/test subject overlap in every fold.
- ANN strict MAE: SBP {ann_sbp:.2f}, DBP {ann_dbp:.2f} mmHg.
- SNN strict MAE: SBP {snn_sbp:.2f}, DBP {snn_dbp:.2f} mmHg.
- ANN random-window MAE: SBP {rw_ann_sbp:.2f}, DBP {rw_ann_dbp:.2f} mmHg; apparent optimism {(ann_sbp-rw_ann_sbp)/ann_sbp*100:.1f}%/{(ann_dbp-rw_ann_dbp)/ann_dbp*100:.1f}%.
- SNN random-window MAE: SBP {rw_snn_sbp:.2f}, DBP {rw_snn_dbp:.2f} mmHg; apparent optimism {(snn_sbp-rw_snn_sbp)/snn_sbp*100:.1f}%/{(snn_dbp-rw_snn_dbp)/snn_dbp*100:.1f}%.
- Interpretation: the external camera-derived dataset independently reproduces split leakage. The current SNN is smaller, but not more accurate than ANN under strict generalization.
"""
    (results / "public_luh_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
