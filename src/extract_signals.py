from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import ensure_dirs, load_config


def masked_rgb_mean(rgb: np.ndarray) -> np.ndarray:
    if rgb.size == 0:
        return np.full(3, np.nan, dtype=np.float32)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mask = (y > 25) & (cr > 125) & (cr < 180) & (cb > 70) & (cb < 140)
    pixels = rgb[mask]
    if len(pixels) < max(50, rgb.shape[0] * rgb.shape[1] * 0.05):
        pixels = rgb.reshape(-1, 3)
    lo = np.quantile(pixels, 0.02, axis=0)
    hi = np.quantile(pixels, 0.98, axis=0)
    keep = np.all((pixels >= lo) & (pixels <= hi), axis=1)
    pixels = pixels[keep] if keep.any() else pixels
    return pixels.mean(axis=0).astype(np.float32)


def face_rois(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> list[np.ndarray]:
    x, y, w, h = box
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def crop(rx1, ry1, rx2, ry2):
        x1, x2 = int(x + rx1 * w), int(x + rx2 * w)
        y1, y2 = int(y + ry1 * h), int(y + ry2 * h)
        return rgb[max(0, y1):min(rgb.shape[0], y2), max(0, x1):min(rgb.shape[1], x2)]

    return [
        crop(.22, .10, .78, .30),
        crop(.10, .48, .42, .75),
        crop(.58, .48, .90, .75),
        crop(.10, .08, .90, .82),
    ]


def detect_largest(detector, frame_bgr: np.ndarray, previous=None):
    small = cv2.resize(frame_bgr, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(80, 80))
    if len(faces):
        x, y, w, h = max(faces, key=lambda q: q[2] * q[3])
        return (int(2*x), int(2*y), int(2*w), int(2*h)), True
    if previous is not None:
        return previous, False
    height, width = frame_bgr.shape[:2]
    return (int(.2 * width), int(.05 * height), int(.6 * width), int(.9 * height)), False


def extract_face(path: Path, detect_interval: int = 60) -> tuple[np.ndarray, dict]:
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open face video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    traces = []
    box = None
    detections = 0
    detection_attempts = 0
    index = 0
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        if index % detect_interval == 0 or box is None:
            detection_attempts += 1
            box, found = detect_largest(detector, frame, box)
            detections += int(found)
        means = np.concatenate([masked_rgb_mean(roi) for roi in face_rois(frame, box)])
        traces.append(means)
        index += 1
    cap.release()
    if len(traces) < 30:
        raise RuntimeError(f"Too few decoded face frames ({len(traces)}): {path}")
    return np.stack(traces).astype(np.float32), {
        "fps": fps, "frames": len(traces), "declared_frames": declared_frames,
        "width": width, "height": height, "detection_attempts": detection_attempts,
        "detections": detections, "detection_rate": detections / max(1, detection_attempts),
    }


def extract_finger_video(path: Path) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open finger video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    traces = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        y1, y2 = int(.15 * height), int(.85 * height)
        x1, x2 = int(.15 * width), int(.85 * width)
        rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        traces.append(rgb.reshape(-1, 3).mean(axis=0))
    cap.release()
    if len(traces) < 30:
        raise RuntimeError(f"Too few decoded finger frames ({len(traces)}): {path}")
    return np.asarray(traces, dtype=np.float32), {
        "fps": fps, "frames": len(traces), "declared_frames": declared_frames,
        "width": width, "height": height,
    }


def process_row(payload: dict) -> dict:
    output_path = Path(payload.pop("output_path"))
    overwrite = payload.pop("overwrite")
    if output_path.exists() and not overwrite:
        return {"session_id": payload["session_id"], "status": "cached", "path": str(output_path)}
    started = time.time()
    face_trace, face_meta = extract_face(Path(payload["face_path"]))
    finger_trace = np.empty((0, 3), dtype=np.float32)
    finger_text = np.empty(0, dtype=np.float32)
    finger_meta = {}
    finger_path = payload.get("finger_path")
    finger_txt_path = payload.get("finger_txt_path")
    if isinstance(finger_path, str) and finger_path:
        finger_trace, finger_meta = extract_finger_video(Path(finger_path))
    elif isinstance(finger_txt_path, str) and finger_txt_path:
        path = Path(finger_txt_path)
        if path.exists() and path.stat().st_size:
            finger_text = np.loadtxt(path, dtype=np.float32)
            finger_meta = {"samples": int(finger_text.size)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        face_rgb=face_trace,
        finger_rgb=finger_trace,
        finger_text=finger_text,
        labels=np.asarray([payload["sbp"], payload["dbp"], payload["hr"]], dtype=np.float32),
        demographics=np.asarray([payload["age"], payload["height_cm"], payload["weight_kg"]], dtype=np.float32),
        face_meta=json.dumps(face_meta),
        finger_meta=json.dumps(finger_meta),
    )
    return {
        "session_id": payload["session_id"], "source": payload["source"], "status": "ok",
        "path": str(output_path), "seconds": time.time() - started,
        "face_frames": len(face_trace), "finger_samples": len(finger_trace) or len(finger_text),
        "face_detection_rate": face_meta["detection_rate"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--source", choices=["hospital", "lab", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dirs(cfg)
    manifest_path = out / "manifests" / "session_manifest.csv"
    df = pd.read_csv(manifest_path)
    df = df[df["has_face"]].copy()
    if args.source != "all":
        df = df[df["source"] == args.source]
    if args.limit:
        df = df.head(args.limit)
    tasks = []
    for row in df.to_dict(orient="records"):
        row["output_path"] = str(out / "signals" / row["source"] / f"{row['session_id']}.npz")
        row["overwrite"] = args.overwrite
        tasks.append(row)

    log_path = out / "logs" / "signal_extraction.jsonl"
    counts = {"ok": 0, "cached": 0, "error": 0}
    with log_path.open("a", encoding="utf-8") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_row, task): task["session_id"] for task in tasks}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"session_id": futures[future], "status": "error", "error": repr(exc)}
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
            if done % 10 == 0 or result["status"] == "error" or done == len(tasks):
                print(json.dumps({"done": done, "total": len(tasks), "counts": counts, "last": result}, ensure_ascii=False), flush=True)
    print(json.dumps({"complete": True, "counts": counts, "log": str(log_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
