"""Data-free forward-pass smoke test for the five reported architectures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models import build_model, parameter_count  # noqa: E402


def main() -> None:
    x = torch.zeros(2, 300)
    for name in ("ann", "snn", "resnet1d", "tcn", "transformer"):
        model = build_model(name, steps=8).eval()
        with torch.no_grad():
            prediction, feature, spike_rate = model(x)
        assert prediction.shape == (2, 3)
        assert feature.shape[0] == 2
        assert torch.isfinite(prediction).all()
        print(
            f"{name:12s} parameters={parameter_count(model):6d} "
            f"output={tuple(prediction.shape)} spike_rate={float(spike_rate):.4f}"
        )


if __name__ == "__main__":
    main()
