from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    selected = path or os.environ.get("BP_CONFIG")
    path = Path(selected) if selected else PROJECT_ROOT / "config" / "default.json"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    for key in ("raw_root", "output_root", "public_h5"):
        if key not in cfg:
            continue
        value = Path(cfg[key]).expanduser()
        if not value.is_absolute():
            value = PROJECT_ROOT / value
        cfg[key] = str(value.resolve())
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def subject_hash(source: str, raw_id: str) -> str:
    salt = os.environ.get("BP_ID_SALT")
    if not salt or len(salt) < 16:
        raise RuntimeError(
            "Set BP_ID_SALT to a private random value of at least 16 characters. "
            "Never commit the salt or any identifier lookup table."
        )
    value = f"{salt}|{source}|{raw_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def ensure_dirs(cfg: dict) -> Path:
    root = Path(cfg["output_root"])
    for name in ("manifests", "signals", "features", "models", "results", "figures", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def public_dataset_path(cfg: dict) -> Path:
    path = Path(cfg["public_h5"])
    if not path.exists():
        raise FileNotFoundError(
            f"Public rPPG-BP-UKL file not found: {path}. "
            "See data/public/rppg_bp_ukl/README.md."
        )
    return path
