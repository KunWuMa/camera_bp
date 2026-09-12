from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

from common import ensure_dirs, load_config
from models import ANNRegressor, SpikingRegressor, parameter_count


def analog_macs(model, x):
    total = 0
    hooks = []
    def hook(module, inputs, output):
        nonlocal total
        if isinstance(module, nn.Conv1d):
            total += output.numel() * module.kernel_size[0] * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features
        elif isinstance(module, nn.GRU):
            batch, steps, _ = inputs[0].shape
            directions = 2 if module.bidirectional else 1
            total += batch * steps * directions * 3 * (
                module.input_size * module.hidden_size + module.hidden_size ** 2 + module.hidden_size)
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.GRU)): hooks.append(module.register_forward_hook(hook))
    with torch.no_grad(): model(x)
    for h in hooks: h.remove()
    return total // x.shape[0]


def main():
    cfg = load_config()
    root = ensure_dirs(cfg)
    x = torch.randn(32, int(cfg["signal_rate_hz"] * cfg["window_seconds"]))
    rows = []
    for name, model in (("Face-ANN", ANNRegressor()), ("Face-SNN", SpikingRegressor(steps=cfg["snn_steps"]))):
        with torch.no_grad(): _, _, rate = model(x)
        rows.append({"model": name, "parameters": parameter_count(model), "dense_mac_equivalents": analog_macs(model, x),
                     "untrained_spike_rate": float(rate), "input_samples": x.shape[1]})
    (root / "results" / "model_complexity.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
