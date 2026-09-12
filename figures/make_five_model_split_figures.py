"""Create the five-model split-comparison figures used by the BSPC paper.

The source table contains one row per cohort, split, seed, model, and target.
Only subject-macro MAE is plotted; the training-median control is excluded.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts" / "results"
OUT = ROOT / "artifacts" / "figures"

MODELS = ["Face-ANN", "Face-SNN", "ResNet-1D", "TCN", "Transformer"]
LABELS = {
    "Face-ANN": "ANN",
    "Face-SNN": "SNN",
    "ResNet-1D": "ResNet-1D",
    "TCN": "TCN",
    "Transformer": "Transformer",
}
COLORS = {
    "Face-ANN": "#2878B5",
    "Face-SNN": "#E07A1F",
    "ResNet-1D": "#2A9D8F",
    "TCN": "#C23B61",
    "Transformer": "#7656A5",
}
MARKERS = {
    "Face-ANN": "o",
    "Face-SNN": "s",
    "ResNet-1D": "D",
    "TCN": "^",
    "Transformer": "P",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_metrics() -> pd.DataFrame:
    path = RESULTS / "bspc_model_suite_seed_metrics.csv"
    data = pd.read_csv(path, encoding="utf-8-sig")
    data = data[data["model"].isin(MODELS)].copy()
    data["target"] = data["target"].str.lower()
    expected = {"cohort", "protocol", "seed", "model", "target", "macro_subject_mae"}
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    duplicates = data.duplicated(["cohort", "protocol", "seed", "model", "target"])
    if duplicates.any():
        raise ValueError("Duplicate seed-metric rows detected")
    return data


def tidy_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.055,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def model_series(
    ax: plt.Axes,
    frame: pd.DataFrame,
    protocols: list[str],
    model: str,
    offset: float,
    connect_at_offset: bool,
) -> tuple[np.ndarray, np.ndarray]:
    means, sds = [], []
    for i, protocol in enumerate(protocols):
        values = (
            frame[(frame["protocol"] == protocol) & (frame["model"] == model)]
            .sort_values("seed")["macro_subject_mae"]
            .to_numpy(float)
        )
        if len(values) != 3:
            raise ValueError(f"Expected 3 seeds for {model}, {protocol}; found {len(values)}")
        mean = values.mean()
        sd = values.std(ddof=1)
        means.append(mean)
        sds.append(sd)
        seed_jitter = np.array([-0.018, 0.0, 0.018])
        ax.scatter(
            i + offset + seed_jitter,
            values,
            s=12,
            color=COLORS[model],
            alpha=0.27,
            linewidths=0,
            zorder=2,
        )

    x = np.arange(len(protocols), dtype=float)
    line_x = x + offset if connect_at_offset else x
    ax.plot(
        line_x,
        means,
        color=COLORS[model],
        linewidth=1.35 if model != "TCN" else 1.7,
        alpha=0.96,
        zorder=3,
    )
    ax.errorbar(
        x + offset,
        means,
        yerr=sds,
        fmt=MARKERS[model],
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor=COLORS[model],
        markeredgewidth=1.15,
        ecolor=COLORS[model],
        elinewidth=0.85,
        capsize=2.0,
        capthick=0.85,
        zorder=4,
        label=LABELS[model],
    )
    return np.asarray(means), np.asarray(sds)


def make_private_figure(data: pd.DataFrame) -> None:
    protocols = ["Random-window", "Random-session", "Subject-disjoint"]
    xlabels = ["Random\nwindow", "Random\nsession", "Subject\ndisjoint"]
    panels = [
        ("hospital", "sbp", "Hospital: systolic BP"),
        ("hospital", "dbp", "Hospital: diastolic BP"),
        ("lab", "sbp", "Laboratory: systolic BP"),
        ("lab", "dbp", "Laboratory: diastolic BP"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), constrained_layout=False)
    offsets = np.linspace(-0.22, 0.22, len(MODELS))

    for panel_label, ax, (cohort, target, title) in zip("abcd", axes.flat, panels):
        frame = data[(data["cohort"] == cohort) & (data["target"] == target)]
        for model, offset in zip(MODELS, offsets):
            model_series(ax, frame, protocols, model, offset, connect_at_offset=True)
        ax.set_xticks(range(3), xlabels)
        ax.set_xlim(-0.42, 2.42)
        ax.set_ylabel("Subject-macro MAE (mmHg)")
        ax.set_title(title, pad=7)
        tidy_axis(ax)
        add_panel_label(ax, panel_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.995),
        ncol=5,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.10, top=0.90, wspace=0.31, hspace=0.43)
    fig.savefig(OUT / "split_leakage_effect_nature.pdf", bbox_inches="tight")
    fig.savefig(OUT / "split_leakage_effect_nature.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def make_public_figure(data: pd.DataFrame) -> None:
    protocols = ["Random-window", "Subject-disjoint"]
    xlabels = ["Random window", "Subject disjoint"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.05), constrained_layout=False)

    for panel_label, ax, target, title in zip(
        "ab", axes, ["sbp", "dbp"], ["Systolic BP", "Diastolic BP"]
    ):
        frame = data[(data["cohort"] == "public_ukl") & (data["target"] == target)]
        tcn_means = None
        for model in MODELS:
            means, _ = model_series(ax, frame, protocols, model, 0.0, connect_at_offset=False)
            if model == "TCN":
                tcn_means = means
        assert tcn_means is not None
        ax.set_xticks(range(2), xlabels)
        ax.set_xlim(-0.18, 1.25)
        ax.set_ylabel("Subject-macro MAE (mmHg)")
        ax.set_title(title, pad=7)
        tidy_axis(ax)
        add_panel_label(ax, panel_label)
        ax.annotate(
            f"TCN: {tcn_means[0]:.2f} to {tcn_means[1]:.2f}",
            xy=(1.0, tcn_means[1]),
            xytext=(0.34, 0.94),
            textcoords="axes fraction",
            color=COLORS["TCN"],
            fontsize=7.6,
            fontweight="bold",
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["TCN"],
                "lw": 0.9,
                "shrinkA": 2,
                "shrinkB": 4,
            },
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=5,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.84, wspace=0.30)
    fig.savefig(OUT / "public_luh_split_effect_nature.pdf", bbox_inches="tight")
    fig.savefig(OUT / "public_luh_split_effect_nature.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    make_private_figure(data)
    make_public_figure(data)


if __name__ == "__main__":
    main()
