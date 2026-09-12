from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config


def main():
    root = ensure_dirs(load_config())
    df = pd.read_csv(root / "features" / "session_features.csv")
    report = df.groupby("source").agg(
        sessions=("session_id", "size"), face_snr_median=("face_snr_db", "median"),
        face_snr_q25=("face_snr_db", lambda x: x.quantile(.25)),
        face_snr_q75=("face_snr_db", lambda x: x.quantile(.75)),
        finger_snr_median=("finger_snr_db", "median"),
        detection_rate_median=("face_detection_rate", "median"),
        low_detection=("face_detection_rate", lambda x: int((x < .5).sum())),
    ).reset_index()
    report.to_csv(root / "results" / "signal_quality_summary.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for source, group in df.groupby("source"):
        ax.hist(group.face_snr_db.clip(-30, 20), bins=np.arange(-30, 22, 2), alpha=.55, label=source.title())
    ax.set(xlabel="Selected facial pulse SNR (dB)", ylabel="Acquisitions")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(root / "figures" / "signal_quality_distribution.png", dpi=300)
    plt.close(fig)
    print(json.dumps(report.to_dict("records"), indent=2))


if __name__ == "__main__": main()
