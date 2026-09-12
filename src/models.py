from __future__ import annotations

import torch
from torch import nn


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, channels, 7, stride=2, padding=3), nn.BatchNorm1d(channels), nn.GELU(),
            nn.Conv1d(channels, channels, 5, stride=2, padding=2), nn.BatchNorm1d(channels), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(1))


class ANNRegressor(nn.Module):
    def __init__(self, hidden: int = 64, outputs: int = 3):
        super().__init__()
        self.encoder = TemporalEncoder(hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(.2), nn.Linear(hidden, outputs))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x).transpose(1, 2)
        z, _ = self.gru(z)
        feature = z.mean(1)
        return self.head(feature), feature, x.new_zeros(())


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class ResNet1DRegressor(nn.Module):
    """Compact 1-D residual CNN baseline for waveform regression."""
    def __init__(self, outputs: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 64),
            ResidualBlock1D(64, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, outputs)
        )

    def forward(self, x: torch.Tensor):
        feature = self.blocks(self.stem(x.unsqueeze(1))).mean(-1)
        return self.head(feature), feature, x.new_zeros(())


class Chomp1D(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.amount].contiguous() if self.amount else x


class TemporalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = 2 * dilation
        self.body = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            Chomp1D(padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(.1),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
            Chomp1D(padding),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class TCNRegressor(nn.Module):
    """Dilated causal temporal convolutional network baseline."""
    def __init__(self, outputs: int = 3):
        super().__init__()
        self.stem = TemporalEncoder(64)
        self.tcn = nn.Sequential(*(TemporalConvBlock(64, dilation) for dilation in (1, 2, 4, 8)))
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, outputs)
        )

    def forward(self, x: torch.Tensor):
        feature = self.tcn(self.stem(x)).mean(-1)
        return self.head(feature), feature, x.new_zeros(())


class TransformerRegressor(nn.Module):
    """Lightweight convolutional-token Transformer encoder baseline."""
    def __init__(self, outputs: int = 3, hidden: int = 64):
        super().__init__()
        self.encoder = TemporalEncoder(hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(hidden))
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(.2), nn.Linear(64, outputs)
        )
        nn.init.normal_(self.cls_token, std=.02)

    @staticmethod
    def sinusoidal_position(length: int, hidden: int, device: torch.device, dtype: torch.dtype):
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden, 2, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10_000.0, device=device, dtype=dtype)) / hidden)
        )
        encoding = torch.zeros(length, hidden, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        tokens = self.encoder(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.sinusoidal_position(
            tokens.shape[1], tokens.shape[2], tokens.device, tokens.dtype
        )
        feature = self.transformer(tokens)[:, 0]
        return self.head(feature), feature, x.new_zeros(())


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        ctx.save_for_backward(x)
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        return grad_output / (1.0 + 4.0 * x.abs()).pow(2)


spike = SurrogateSpike.apply


class SpikingRegressor(nn.Module):
    """Rate-coded LIF regressor with an explicit sparse-spike energy proxy."""
    def __init__(self, hidden: int = 96, steps: int = 8, outputs: int = 3, beta: float = .9):
        super().__init__()
        self.steps = steps
        self.beta = beta
        self.encoder = TemporalEncoder(64)
        self.input = nn.Linear(64, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.readout = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, outputs))
        self.threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        temporal = self.encoder(x)
        temporal = torch.nn.functional.adaptive_avg_pool1d(temporal, self.steps).transpose(1, 2)
        membrane = x.new_zeros((x.shape[0], self.input.out_features))
        spikes = x.new_zeros(membrane.shape)
        readout_state = x.new_zeros(membrane.shape)
        all_spikes = []
        threshold = self.threshold.clamp(.5, 2.0)
        for t in range(self.steps):
            current = self.input(temporal[:, t]) + self.recurrent(spikes)
            membrane = self.beta * membrane + self.norm(current)
            spikes = spike(membrane - threshold)
            membrane = membrane - spikes * threshold.detach()
            readout_state = readout_state + membrane + spikes
            all_spikes.append(spikes)
        feature = readout_state / self.steps
        spike_rate = torch.stack(all_spikes).mean()
        return self.readout(feature), feature, spike_rate


class EnhancedSpikingRegressor(nn.Module):
    """LIF regressor with learnable decay and separated temporal state statistics."""
    def __init__(self, hidden: int = 96, steps: int = 32, outputs: int = 3):
        super().__init__()
        self.steps = steps
        self.encoder = TemporalEncoder(64)
        self.input = nn.Linear(64, hidden)
        self.recurrent = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.threshold = nn.Parameter(torch.tensor(1.0))
        self.beta_logit = nn.Parameter(torch.tensor(2.1972246))  # sigmoid(beta_logit) = 0.9
        self.time_logits = nn.Parameter(torch.zeros(steps))
        self.feature_norm = nn.LayerNorm(hidden * 3)
        self.readout = nn.Sequential(
            nn.Linear(hidden * 3, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, outputs)
        )

    def forward(self, x: torch.Tensor):
        temporal = self.encoder(x)
        temporal = torch.nn.functional.adaptive_avg_pool1d(temporal, self.steps).transpose(1, 2)
        membrane = x.new_zeros((x.shape[0], self.input.out_features))
        spikes = x.new_zeros(membrane.shape)
        membrane_trace, spike_trace = [], []
        threshold = self.threshold.clamp(.5, 2.0)
        beta = torch.sigmoid(self.beta_logit).clamp(.75, .99)
        for t in range(self.steps):
            current = self.input(temporal[:, t]) + self.recurrent(spikes)
            membrane = beta * membrane + self.norm(current)
            spikes = spike(membrane - threshold)
            membrane = membrane - spikes * threshold.detach()
            membrane_trace.append(membrane)
            spike_trace.append(spikes)
        membranes = torch.stack(membrane_trace, dim=1)
        spike_states = torch.stack(spike_trace, dim=1)
        attention = torch.softmax(self.time_logits, dim=0)
        weighted_membrane = (membranes * attention[None, :, None]).sum(1)
        feature = self.feature_norm(torch.cat((weighted_membrane, membranes[:, -1], spike_states.mean(1)), dim=1))
        spike_rate = spike_states.mean()
        return self.readout(feature), feature, spike_rate


def build_model(name: str, steps: int = 8) -> nn.Module:
    if name in {"teacher", "ann"}:
        return ANNRegressor()
    if name == "resnet1d":
        return ResNet1DRegressor()
    if name == "tcn":
        return TCNRegressor()
    if name == "transformer":
        return TransformerRegressor()
    if name.startswith("snn"):
        return SpikingRegressor(steps=steps)
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
