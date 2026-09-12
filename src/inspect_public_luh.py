from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from common import load_config, public_dataset_path


def describe(node: h5py.Dataset) -> dict:
    out = {"shape": list(node.shape), "dtype": str(node.dtype)}
    if np.issubdtype(node.dtype, np.number):
        values = np.asarray(node)
        finite = values[np.isfinite(values)]
        out.update(
            finite=int(finite.size),
            minimum=float(finite.min()) if finite.size else None,
            maximum=float(finite.max()) if finite.size else None,
            mean=float(finite.mean()) if finite.size else None,
            std=float(finite.std()) if finite.size else None,
        )
    return out


def main() -> None:
    path = public_dataset_path(load_config())
    report = {"path": str(path), "size_bytes": path.stat().st_size, "datasets": {}}
    with h5py.File(path, "r") as handle:
        def visit(name: str, obj: h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                report["datasets"][name] = describe(obj)

        handle.visititems(visit)
        labels = np.asarray(handle["label"])
        subjects = np.asarray(handle["subject_idx"]).reshape(-1).astype(int)
        report["label_rows"] = [
            {
                "row": row,
                "minimum": float(labels[row].min()),
                "maximum": float(labels[row].max()),
                "mean": float(labels[row].mean()),
                "std": float(labels[row].std()),
            }
            for row in range(labels.shape[0])
        ]
        unique, counts = np.unique(subjects, return_counts=True)
        report["subjects"] = {"count": int(unique.size), "samples": dict(zip(unique.tolist(), counts.tolist()))}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
