from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config


def main():
    cfg = load_config()
    root = ensure_dirs(cfg)
    df = pd.read_csv(root / "manifests" / "session_manifest.csv")
    colors = {"hospital": "#4C72B0", "lab": "#DD8452"}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), constrained_layout=True)
    for source, group in df.groupby("source"):
        axes[0].hist(group.sbp, bins=np.arange(70, 201, 10), alpha=.55, color=colors[source], label=source.title())
        axes[1].hist(group.dbp, bins=np.arange(45, 121, 5), alpha=.55, color=colors[source], label=source.title())
    for ax, name in zip(axes, ("SBP", "DBP")):
        ax.set(xlabel=f"{name} (mmHg)", ylabel="Acquisitions")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.savefig(root / "figures" / "cohort_bp_distributions.png", dpi=300)
    plt.close(fig)

    summary = []
    for source, group in df.groupby("source"):
        summary.append({"source": source, "sessions": len(group), "subjects": group.subject_id.nunique(),
                        "repeat_sessions": len(group) - group.subject_id.nunique(),
                        "sbp_mean": group.sbp.mean(), "sbp_sd": group.sbp.std(),
                        "dbp_mean": group.dbp.mean(), "dbp_sd": group.dbp.std(),
                        "finger_available": int(group.has_finger.sum()),
                        "age_missing": int((group.age.isna() | (group.age < 0)).sum()),
                        "hr_missing": int(group.hr.isna().sum())})
    pd.DataFrame(summary).to_csv(root / "results" / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    df.groupby(["source", "fold"]).agg(sessions=("session_id", "size"), subjects=("subject_id", "nunique"),
        sbp_mean=("sbp", "mean"), sbp_sd=("sbp", "std"), dbp_mean=("dbp", "mean"), dbp_sd=("dbp", "std")).reset_index().to_csv(
            root / "results" / "fold_summary.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
