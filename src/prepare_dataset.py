from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats

from common import ensure_dirs, load_config


EPS = 1e-8


def bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    if len(x) < 30:
        return x.astype(np.float32)
    nyq = fs / 2
    hi = min(high, nyq * .95)
    sos = signal.butter(3, [low / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, signal.detrend(x)).astype(np.float32)


def pos_trace(rgb: np.ndarray, fs: float) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    n = len(rgb)
    length = max(16, int(round(1.6 * fs)))
    out = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    projection = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)
    for end in range(length, n + 1):
        start = end - length
        c = rgb[start:end].T
        cn = c / (c.mean(axis=1, keepdims=True) + EPS)
        s = projection @ cn
        h = s[0] + (np.std(s[0]) / (np.std(s[1]) + EPS)) * s[1]
        h -= h.mean()
        out[start:end] += h
        weight[start:end] += 1
    return (out / np.maximum(weight, 1)).astype(np.float32)


def chrom_trace(rgb: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb, dtype=np.float64)
    c = c / (c.mean(axis=0, keepdims=True) + EPS) - 1
    x = 3 * c[:, 0] - 2 * c[:, 1]
    y = 1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2]
    alpha = np.std(x) / (np.std(y) + EPS)
    return (x - alpha * y).astype(np.float32)


def pulse_snr(x: np.ndarray, fs: float) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    if len(x) < max(30, fs * 4) or np.std(x) < EPS:
        return -30.0, np.nan
    freqs, power = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 10)))
    band = (freqs >= .7) & (freqs <= 4.0)
    if not band.any() or power[band].sum() <= 0:
        return -30.0, np.nan
    idx = np.where(band)[0][np.argmax(power[band])]
    peak = freqs[idx]
    signal_band = ((np.abs(freqs - peak) <= .1) | (np.abs(freqs - 2 * peak) <= .1)) & band
    sig = power[signal_band].sum()
    noise = power[band & ~signal_band].sum() + EPS
    return float(10 * np.log10((sig + EPS) / noise)), float(peak * 60)


def choose_face_signal(face_rgb: np.ndarray, fs: float, low: float, high: float) -> tuple[np.ndarray, dict]:
    candidates = []
    methods = []
    for roi in range(face_rgb.shape[1] // 3):
        rgb = face_rgb[:, roi * 3:(roi + 1) * 3]
        for method, raw in (("pos", pos_trace(rgb, fs)), ("chrom", chrom_trace(rgb))):
            filtered = bandpass(raw, fs, low, high)
            snr, hr = pulse_snr(filtered, fs)
            candidates.append((snr, filtered, hr))
            methods.append(f"roi{roi}_{method}")
    best = int(np.argmax([x[0] for x in candidates]))
    snr, trace, hr = candidates[best]
    return trace, {"method": methods[best], "snr_db": snr, "spectral_hr": hr}


def choose_finger_signal(rgb: np.ndarray, text: np.ndarray, fs: float, low: float, high: float, target_len: int) -> tuple[np.ndarray, dict]:
    candidates = []
    names = []
    if text.size:
        raw = signal.resample(text.astype(np.float64), target_len)
        filtered = bandpass(raw, fs, low, high)
        candidates.append((pulse_snr(filtered, fs), filtered))
        names.append("text")
    if rgb.size:
        for channel, name in enumerate(("r", "g", "b")):
            raw = signal.resample(rgb[:, channel].astype(np.float64), target_len)
            filtered = bandpass(raw, fs, low, high)
            candidates.append((pulse_snr(filtered, fs), filtered))
            names.append(name)
    if not candidates:
        return np.zeros(target_len, dtype=np.float32), {"method": "missing", "snr_db": -30.0, "spectral_hr": np.nan}
    best = int(np.argmax([x[0][0] for x in candidates]))
    (snr, hr), trace = candidates[best]
    return trace.astype(np.float32), {"method": names[best], "snr_db": snr, "spectral_hr": hr}


def zscore_window(x: np.ndarray) -> np.ndarray:
    return ((x - np.mean(x)) / (np.std(x) + EPS)).astype(np.float32)


def handcrafted(x: np.ndarray, fs: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    freqs, power = signal.welch(x, fs=fs, nperseg=min(len(x), int(fs * 10)))
    band = (freqs >= .7) & (freqs <= 4.0)
    bp = power[band]
    bf = freqs[band]
    peak = bf[np.argmax(bp)] if len(bp) else 0
    spectral_entropy = -np.sum((bp / (bp.sum() + EPS)) * np.log(bp / (bp.sum() + EPS) + EPS)) if len(bp) else 0
    peaks, _ = signal.find_peaks(x, distance=max(1, int(fs * .3)), prominence=max(EPS, .15 * np.std(x)))
    ibi = np.diff(peaks) / fs if len(peaks) >= 2 else np.array([])
    snr, spectral_hr = pulse_snr(x, fs)
    return np.asarray([
        np.mean(x), np.std(x), np.min(x), np.max(x), np.ptp(x),
        np.quantile(x, .05), np.quantile(x, .25), np.median(x), np.quantile(x, .75), np.quantile(x, .95),
        stats.skew(x), stats.kurtosis(x), np.mean(np.abs(dx)), np.std(dx), np.sqrt(np.mean(dx ** 2)),
        peak, spectral_hr, snr, spectral_entropy, len(peaks),
        np.mean(ibi) if len(ibi) else 0, np.std(ibi) if len(ibi) else 0,
        np.sqrt(np.mean(np.diff(ibi) ** 2)) if len(ibi) >= 2 else 0,
    ], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only: process first N available sessions")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = ensure_dirs(cfg)
    manifest = pd.read_csv(root / "manifests" / "session_manifest.csv")
    if args.limit:
        available = manifest.apply(lambda r: (root / "signals" / r["source"] / f"{r['session_id']}.npz").exists(), axis=1)
        manifest = manifest[available].head(args.limit).reset_index(drop=True)
    fs = float(cfg["signal_rate_hz"])
    window = int(cfg["window_seconds"] * fs)
    stride = int(cfg["window_stride_seconds"] * fs)
    h5_path = root / "features" / "windows.h5"
    if h5_path.exists() and not args.overwrite:
        print(f"cached={h5_path}")
        return

    face_windows, finger_windows, labels = [], [], []
    quality, demographics = [], []
    subjects, sessions, sources, folds = [], [], [], []
    session_features = []
    session_rows = []
    failures = []

    for row_num, row in manifest.iterrows():
        path = root / "signals" / row["source"] / f"{row['session_id']}.npz"
        if not path.exists():
            failures.append({"session_id": row["session_id"], "reason": "missing_npz"})
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                face_rgb = z["face_rgb"]
                finger_rgb = z["finger_rgb"]
                finger_text = z["finger_text"]
                face_meta = json.loads(str(z["face_meta"]))
            input_fps = float(face_meta.get("fps") or fs)
            target_len = max(window, int(round(len(face_rgb) * fs / input_fps)))
            face_rgb = signal.resample(face_rgb, target_len, axis=0).astype(np.float32)
            face, face_q = choose_face_signal(face_rgb, fs, cfg["bandpass_low_hz"], cfg["bandpass_high_hz"])
            finger, finger_q = choose_finger_signal(finger_rgb, finger_text, fs, cfg["bandpass_low_hz"], cfg["bandpass_high_hz"], target_len)
            usable_len = min(len(face), len(finger))
            face, finger = face[:usable_len], finger[:usable_len]
            session_features.append(handcrafted(face, fs))
            session_rows.append({
                "source": row["source"], "subject_id": row["subject_id"], "session_id": row["session_id"],
                "fold": int(row["fold"]), "sbp": row["sbp"], "dbp": row["dbp"], "hr": row["hr"],
                "age": row["age"], "sex": row["sex"], "height_cm": row["height_cm"], "weight_kg": row["weight_kg"],
                "face_snr_db": face_q["snr_db"], "finger_snr_db": finger_q["snr_db"],
                "face_detection_rate": float(face_meta.get("detection_rate", 0.0)),
                "face_method": face_q["method"], "finger_method": finger_q["method"],
                "face_spectral_hr": face_q["spectral_hr"], "finger_spectral_hr": finger_q["spectral_hr"],
                "samples": usable_len,
            })
            starts = list(range(0, usable_len - window + 1, stride))
            if not starts:
                starts = [0]
                face = signal.resample(face, window)
                finger = signal.resample(finger, window)
            for start in starts:
                fw = zscore_window(face[start:start + window])
                pw = zscore_window(finger[start:start + window])
                face_windows.append(fw)
                finger_windows.append(pw)
                labels.append([row["sbp"], row["dbp"], row["hr"]])
                quality.append([face_q["snr_db"], finger_q["snr_db"], float(face_meta.get("detection_rate", 0.0))])
                sex = 1.0 if str(row["sex"]) == "男" else 0.0 if str(row["sex"]) == "女" else np.nan
                demographics.append([row["age"], sex, row["height_cm"], row["weight_kg"]])
                subjects.append(row["subject_id"])
                sessions.append(row["session_id"])
                sources.append(row["source"])
                folds.append(int(row["fold"]))
        except Exception as exc:
            failures.append({"session_id": row["session_id"], "reason": repr(exc)})
        if (row_num + 1) % 25 == 0:
            print(json.dumps({"prepared": row_num + 1, "total": len(manifest), "windows": len(face_windows), "failures": len(failures)}), flush=True)

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("face", data=np.stack(face_windows), compression="gzip", compression_opts=4)
        h5.create_dataset("finger", data=np.stack(finger_windows), compression="gzip", compression_opts=4)
        h5.create_dataset("labels", data=np.asarray(labels, dtype=np.float32))
        h5.create_dataset("quality", data=np.asarray(quality, dtype=np.float32))
        h5.create_dataset("demographics", data=np.asarray(demographics, dtype=np.float32))
        string_type = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("subject_id", data=np.asarray(subjects, dtype=object), dtype=string_type)
        h5.create_dataset("session_id", data=np.asarray(sessions, dtype=object), dtype=string_type)
        h5.create_dataset("source", data=np.asarray(sources, dtype=object), dtype=string_type)
        h5.create_dataset("fold", data=np.asarray(folds, dtype=np.int16))
        h5.attrs["sample_rate_hz"] = fs
        h5.attrs["window_seconds"] = cfg["window_seconds"]
        h5.attrs["stride_seconds"] = cfg["window_stride_seconds"]

    feature_names = [
        "mean", "std", "min", "max", "range", "q05", "q25", "median", "q75", "q95",
        "skew", "kurtosis", "mean_abs_diff", "diff_std", "diff_rms", "peak_hz", "spectral_hr",
        "snr_db", "spectral_entropy", "peak_count", "ibi_mean", "ibi_std", "rmssd",
    ]
    sf = pd.DataFrame(np.stack(session_features), columns=[f"face_{x}" for x in feature_names])
    session_df = pd.concat([pd.DataFrame(session_rows).reset_index(drop=True), sf], axis=1)
    session_df.to_csv(root / "features" / "session_features.csv", index=False, encoding="utf-8-sig")
    (root / "logs" / "prepare_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(h5_path), "windows": len(face_windows), "sessions": len(session_rows),
        "subjects": len(set(subjects)), "failures": len(failures),
        "source_windows": dict(Counter(sources)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
