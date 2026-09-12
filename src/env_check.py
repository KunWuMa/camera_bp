from __future__ import annotations

import json
import platform

import cv2
import h5py
import numpy as np
import pandas as pd
import sklearn
import torch

from common import ensure_dirs, load_config


info = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "opencv": cv2.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
    "h5py": h5py.__version__,
}
print(json.dumps(info, ensure_ascii=False, indent=2))
root = ensure_dirs(load_config())
(root / "logs" / "environment.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
