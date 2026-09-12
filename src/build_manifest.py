from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from common import ensure_dirs, load_config, subject_hash


def norm_id(value) -> str | None:
    if pd.isna(value):
        return None
    digits = re.sub(r"\D", "", str(value).split(".")[0])
    return digits.lstrip("0") or "0" if digits else None


def norm_date(value) -> str | None:
    if pd.isna(value):
        return None
    try:
        return pd.to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        match = re.search(r"(20\d{2})[-./](\d{1,2})[-./](\d{1,2})", str(value))
        if not match:
            return None
        return f"{match.group(1)}-{int(match.group(2)):02d}-{int(match.group(3)):02d}"


def parse_bp(value) -> tuple[float, float]:
    vals = re.findall(r"\d+(?:\.\d+)?", str(value))
    return (float(vals[0]), float(vals[1])) if len(vals) >= 2 else (np.nan, np.nan)


def first_num(value) -> float:
    if pd.isna(value):
        return np.nan
    vals = re.findall(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(vals[0]) if vals else np.nan


def classify_variant(path: Path) -> str:
    text = str(path)
    if "trim_无损" in text:
        return "trim_lossless"
    if "trim_有损" in text:
        return "trim_lossy"
    if "trim" in text.lower():
        return "trim_other"
    return "raw"


def canonical_video(paths: list[Path]) -> Path | None:
    if not paths:
        return None
    priority = {"trim_lossless": 0, "raw": 1, "trim_lossy": 2, "trim_other": 3}
    return sorted(paths, key=lambda p: (priority[classify_variant(p)], -p.stat().st_size, str(p)))[0]


def hospital_manifest(raw_root: Path, layout: dict) -> pd.DataFrame:
    hospital = raw_root / layout["hospital_dir"]
    book = raw_root / layout["hospital_metadata"]
    df = pd.read_excel(book, sheet_name="Sheet1", dtype=str)

    videos = defaultdict(lambda: defaultdict(list))
    rx = re.compile(r"^(?P<id>\d+)_.*?(?P<date>20\d{2}-\d{1,2}-\d{1,2})$", re.I)
    for path in hospital.rglob("*.mp4"):
        match = rx.match(path.stem)
        if not match:
            continue
        key = (norm_id(match.group("id")), norm_date(match.group("date")))
        low = path.stem.lower()
        modality = "face" if "face" in low else "finger" if "finger" in low else None
        if modality:
            videos[key][modality].append(path)

    rows = []
    for _, row in df.iterrows():
        raw_id = norm_id(row.iloc[3])
        date = norm_date(row.iloc[6])
        key = (raw_id, date)
        sbp, dbp = parse_bp(row.iloc[4])
        face = canonical_video(videos[key].get("face", []))
        finger = canonical_video(videos[key].get("finger", []))
        rows.append({
            "source": "hospital",
            "subject_id": subject_hash("hospital", raw_id),
            "session_id": subject_hash("hospital-session", f"{raw_id}|{date}"),
            "date": date,
            "face_path": str(face) if face else "",
            "finger_path": str(finger) if finger else "",
            "finger_txt_path": "",
            "face_variant": classify_variant(face) if face else "",
            "finger_variant": classify_variant(finger) if finger else "",
            "sbp": sbp,
            "dbp": dbp,
            "hr": first_num(row.iloc[5]),
            "age": first_num(row.iloc[0]),
            "sex": str(row.iloc[1]) if not pd.isna(row.iloc[1]) else "",
            "height_cm": first_num(row.iloc[7]),
            "weight_kg": first_num(row.iloc[8]),
        })
    return pd.DataFrame(rows)


def lab_manifest(raw_root: Path, layout: dict) -> pd.DataFrame:
    lab = raw_root / layout["laboratory_dir"]
    face_dir = lab / layout["laboratory_face_dir"]
    finger_dir = lab / layout["laboratory_finger_dir"]
    label_csv = raw_root / layout["laboratory_labels"]
    labels = pd.read_csv(label_csv)
    rows = []
    for session in labels.columns:
        match = re.match(r"^([^_]+)_(\d{4}-\d{1,2}-\d{1,2})_", session)
        if not match:
            continue
        raw_id, date = match.group(1), norm_date(match.group(2))
        face = face_dir / f"{session}.mp4"
        finger_txt = finger_dir / f"{session}.txt"
        values = labels[session].astype(float).to_numpy()
        rows.append({
            "source": "lab",
            "subject_id": subject_hash("lab", raw_id),
            "session_id": subject_hash("lab-session", session),
            "date": date,
            "face_path": str(face) if face.exists() else "",
            "finger_path": "",
            "finger_txt_path": str(finger_txt) if finger_txt.exists() else "",
            "face_variant": "raw",
            "finger_variant": "txt",
            "sbp": values[0], "dbp": values[1], "hr": values[2],
            "age": np.nan, "sex": "", "height_cm": np.nan, "weight_kg": np.nan,
        })
    return pd.DataFrame(rows)


def assign_folds(df: pd.DataFrame, n_splits: int, seed: int) -> pd.Series:
    unique_groups = df["subject_id"].nunique()
    splits = min(n_splits, unique_groups)
    fold = np.full(len(df), -1, dtype=int)
    # GroupKFold is deterministic. Sorting makes assignment stable across filesystem order.
    order = np.argsort(df["session_id"].to_numpy())
    ordered = df.iloc[order].reset_index()
    if splits < unique_groups:
        # Balance the broad SBP spectrum without ever separating repeated acquisitions.
        strata = pd.qcut(ordered["sbp"].rank(method="first"), q=min(5, splits), labels=False)
        splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
        iterator = splitter.split(ordered, y=strata, groups=ordered["subject_id"])
    else:
        splitter = GroupKFold(n_splits=splits)
        iterator = splitter.split(ordered, groups=ordered["subject_id"])
    for fold_id, (_, test_idx) in enumerate(iterator):
        fold[ordered.loc[test_idx, "index"].to_numpy()] = fold_id
    return pd.Series(fold, index=df.index)


def validate(df: pd.DataFrame) -> None:
    assert df["session_id"].is_unique, "Session IDs must be unique"
    assert df[["sbp", "dbp"]].notna().all().all(), "Missing BP labels"
    assert (df["sbp"] > df["dbp"]).all(), "SBP must exceed DBP"
    for source, part in df.groupby("source"):
        for fold, test in part.groupby("fold"):
            train = part[part["fold"] != fold]
            assert not (set(train["subject_id"]) & set(test["subject_id"])), f"Leakage in {source} fold {fold}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = ensure_dirs(cfg)
    raw_root = Path(cfg["raw_root"])
    layout = cfg["private_layout"]
    hospital = hospital_manifest(raw_root, layout)
    lab = lab_manifest(raw_root, layout)
    hospital["fold"] = assign_folds(hospital, cfg["hospital_folds"], cfg["seed"])
    lab["fold"] = assign_folds(lab, cfg["lab_folds"], cfg["seed"])
    manifest = pd.concat([hospital, lab], ignore_index=True)
    manifest["has_face"] = manifest["face_path"].map(lambda x: bool(x) and Path(x).exists())
    manifest["has_finger"] = manifest.apply(
        lambda r: (bool(r["finger_path"]) and Path(r["finger_path"]).exists())
        or (bool(r["finger_txt_path"]) and Path(r["finger_txt_path"]).exists()), axis=1
    )
    validate(manifest)
    path = out / "manifests" / "session_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    summary = manifest.groupby("source").agg(
        sessions=("session_id", "size"), subjects=("subject_id", "nunique"),
        face=("has_face", "sum"), finger=("has_finger", "sum"),
        sbp_mean=("sbp", "mean"), dbp_mean=("dbp", "mean")
    ).reset_index()
    summary.to_csv(out / "manifests" / "manifest_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"manifest={path}")


if __name__ == "__main__":
    main()
