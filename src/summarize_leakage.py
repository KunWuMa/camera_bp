from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from common import ensure_dirs, load_config


def main():
    root = ensure_dirs(load_config())
    pred = pd.read_csv(root / "results" / "leakage_window_predictions.csv")
    # Random-window predictions come from different fold models but are combined once per acquisition.
    session = pred.groupby(["protocol", "model", "target", "subject_id", "session_id"], as_index=False).agg(
        actual=("actual", "first"), prediction=("prediction", "median"))
    metric = session.assign(abs_error=lambda d: abs(d.prediction - d.actual)).groupby(
        ["protocol", "model", "target"], as_index=False).agg(mae=("abs_error", "mean"), sessions=("session_id", "size"))
    strict = pd.read_csv(root / "results" / "main_metrics.csv")
    strict = strict[(strict.source == "lab") & strict.model.isin(["Face-ANN", "Face-SNN"])][["model", "target", "mae", "n_sessions"]]
    strict.insert(0, "protocol", "Subject-disjoint")
    strict = strict.rename(columns={"n_sessions": "sessions"})
    metric = pd.concat([metric, strict], ignore_index=True)
    metric.to_csv(root / "results" / "leakage_protocol_metrics.csv", index=False, encoding="utf-8-sig")
    pivot = metric[metric.target.isin(["sbp", "dbp"])].pivot_table(index=["model", "target"], columns="protocol", values="mae").reset_index()
    if {"Random-window", "Subject-disjoint"}.issubset(pivot):
        pivot["optimism_percent"] = 100 * (pivot["Subject-disjoint"] - pivot["Random-window"]) / pivot["Subject-disjoint"]
    pivot.to_csv(root / "results" / "leakage_optimism.csv", index=False, encoding="utf-8-sig")
    panel = metric[(metric.model == "Face-SNN") & metric.target.isin(["sbp", "dbp"])]
    chart = panel.pivot(index="protocol", columns="target", values="mae").loc[
        ["Random-window", "Random-session", "Subject-disjoint"]]
    ax = chart.rename(columns={"sbp": "SBP", "dbp": "DBP"}).plot.bar(figsize=(5.6, 3.1), color=["#4C72B0", "#DD8452"])
    ax.set(xlabel="", ylabel="MAE (mmHg)", title="Split protocol changes the reported SNN error")
    ax.tick_params(axis="x", rotation=15); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    ax.figure.tight_layout(); ax.figure.savefig(root / "figures" / "split_leakage_effect.png", dpi=300); plt.close(ax.figure)
    print(metric.to_string(index=False))
    print(json.dumps(pivot.to_dict("records"), indent=2))


if __name__ == "__main__": main()
