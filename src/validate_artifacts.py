from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config


def main():
    cfg = load_config()
    root = ensure_dirs(cfg)
    manifest = pd.read_csv(root / "manifests" / "session_manifest.csv")
    records = []
    log_path = root / "logs" / "signal_extraction.jsonl"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
    latest = pd.DataFrame(records).drop_duplicates("session_id", keep="last")
    latest.to_csv(root / "logs" / "signal_extraction_latest.csv", index=False, encoding="utf-8-sig")
    missing_signal = []
    for row in manifest.itertuples():
        if not (root / "signals" / row.source / f"{row.session_id}.npz").exists(): missing_signal.append(row.session_id)
    report = {"manifest_sessions": len(manifest), "latest_log_sessions": len(latest),
              "latest_status": latest.status.value_counts().to_dict(), "missing_signal_files": missing_signal}
    h5_path = root / "features" / "windows.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as h5:
            sources = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h5["source"][:]])
            subjects = np.asarray([x.decode() if isinstance(x, bytes) else str(x) for x in h5["subject_id"][:]])
            folds = h5["fold"][:]
            leakage = []
            for source in np.unique(sources):
                for fold in np.unique(folds[sources == source]):
                    test = set(subjects[(sources == source) & (folds == fold)])
                    train = set(subjects[(sources == source) & (folds != fold)])
                    if test & train: leakage.append({"source": source, "fold": int(fold), "overlap": len(test & train)})
            report.update(windows=len(sources), window_sessions=len(set(
                x.decode() if isinstance(x, bytes) else str(x) for x in h5["session_id"][:])), leakage=leakage,
                finite_face=bool(np.isfinite(h5["face"][:]).all()), finite_labels=bool(np.isfinite(h5["labels"][:, :2]).all()))
    (root / "logs" / "artifact_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing_signal or report.get("leakage") or not report.get("finite_face", True) or not report.get("finite_labels", True):
        raise SystemExit(2)


if __name__ == "__main__": main()
