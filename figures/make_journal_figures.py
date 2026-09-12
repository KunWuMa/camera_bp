from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IC = ROOT / "artifacts"
OUT = ROOT / "artifacts" / "figures"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GRAY = "#7A7A7A"
LIGHT_GRAY = "#D9D9D9"
MODEL_COLORS = {
    "Face-ANN": "#0072B2",
    "Face-SNN": "#D55E00",
    "ResNet-1D": "#009E73",
    "TCN": "#CC79A7",
    "Transformer": "#E69F00",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def cohort_distributions() -> None:
    df = pd.read_csv(IC / "manifests" / "session_manifest.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.35))
    specifications = (("sbp", "Systolic BP (mmHg)"), ("dbp", "Diastolic BP (mmHg)"))
    cohorts = (("hospital", "Hospital", BLUE), ("lab", "Laboratory", VERMILLION))
    rng = np.random.default_rng(20260908)
    for label, (ax, (column, xlabel)) in enumerate(zip(axes, specifications)):
        for position, (key, name, color) in zip((1, 0), cohorts):
            values = df.loc[df["source"].eq(key), column].dropna().to_numpy()
            violin = ax.violinplot([values], positions=[position], vert=False, widths=0.72,
                                   showmeans=False, showmedians=False, showextrema=False,
                                   points=150, bw_method=0.25)
            body = violin["bodies"][0]
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_linewidth(0.8)
            body.set_alpha(0.56)
            jitter = rng.uniform(-0.22, 0.22, size=len(values))
            ax.scatter(values, position + jitter, s=4.0, color=color, alpha=0.20,
                       linewidths=0, rasterized=True, zorder=2)
            quartiles = np.percentile(values, [25, 50, 75])
            whiskers = np.percentile(values, [5, 95])
            ax.plot(whiskers, [position, position], color="#242424", lw=0.7, zorder=4)
            ax.plot(quartiles[[0, 2]], [position, position], color="#242424", lw=4.0,
                    solid_capstyle="round", zorder=5)
            ax.scatter(quartiles[1], position, s=22, facecolor="white", edgecolor="#242424",
                       linewidth=0.8, zorder=6)
        ax.set_xlabel(xlabel)
        ax.set_yticks((1, 0), (f"Hospital\n$n$={sum(df['source'].eq('hospital'))}",
                              f"Laboratory\n$n$={sum(df['source'].eq('lab'))}"))
        ax.set_ylim(-0.62, 1.62)
        ax.tick_params(axis="y", length=0, pad=7)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + label))
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.22, top=0.93, wspace=0.38)
    save(fig, "cohort_bp_distributions_nature")


def leakage_effect() -> None:
    summary = pd.read_csv(IC / "results" / "bspc_extension_seed_summary.csv", encoding="utf-8-sig")
    seeds = pd.read_csv(IC / "results" / "bspc_extension_seed_metrics.csv", encoding="utf-8-sig")
    order = ["Random-window", "Random-session", "Subject-disjoint"]
    display = ["Random\nwindow", "Random\nsession", "Subject\ndisjoint"]
    x = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.35), sharex=True)
    panels = [
        ("hospital", "sbp", "Hospital: systolic BP"),
        ("hospital", "dbp", "Hospital: diastolic BP"),
        ("lab", "sbp", "Laboratory: systolic BP"),
        ("lab", "dbp", "Laboratory: diastolic BP"),
    ]
    models = (("Face-ANN", "ANN", BLUE, "o"), ("Face-SNN", "SNN", VERMILLION, "s"))
    for index, (ax, (cohort, target, title)) in enumerate(zip(axes.flat, panels)):
        for model, label, color, marker in models:
            part = summary[(summary["cohort"].eq(cohort)) &
                           (summary["target"].eq(target)) &
                           (summary["model"].eq(model))].set_index("protocol").loc[order]
            means = part["mean_macro_subject_mae"].to_numpy()
            errors = part["seed_sd_macro_subject_mae"].to_numpy()
            raw = seeds[(seeds["cohort"].eq(cohort)) &
                        (seeds["target"].eq(target)) &
                        (seeds["model"].eq(model))]
            for protocol_index, protocol in enumerate(order):
                values = raw.loc[raw["protocol"].eq(protocol), "macro_subject_mae"].to_numpy()
                ax.scatter(np.full(len(values), x[protocol_index]) + np.linspace(-0.035, 0.035, len(values)),
                           values, s=9, color=color, alpha=0.26, linewidths=0, zorder=2)
            ax.errorbar(x, means, yerr=errors, color=color, marker=marker, ms=4.8,
                        markerfacecolor="white", markeredgewidth=1.0, lw=1.15,
                        capsize=2.3, capthick=0.8, label=label, zorder=4)
        ax.set_title(title, fontsize=8, pad=5)
        ax.set_ylabel("Subject-macro MAE (mmHg)")
        ax.set_xticks(x, display)
        ax.margins(x=0.12, y=0.18)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    axes[0, 0].legend(loc="best", ncol=2, handletextpad=0.4, columnspacing=0.8)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.13, top=0.94, wspace=0.25, hspace=0.42)
    save(fig, "split_leakage_effect_nature")


def public_split_effect() -> None:
    summary = pd.read_csv(IC / "results" / "bspc_extension_seed_summary.csv", encoding="utf-8-sig")
    seeds = pd.read_csv(IC / "results" / "bspc_extension_seed_metrics.csv", encoding="utf-8-sig")
    summary = summary[summary["cohort"].eq("public_ukl")]
    seeds = seeds[seeds["cohort"].eq("public_ukl")]
    protocols = ["Random-window", "Subject-disjoint"]
    x = np.arange(2, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.60))
    models = (
        ("Train-median", "Training median", GRAY, "^", -0.18),
        ("Face-ANN", "ANN", BLUE, "o", 0.00),
        ("Face-SNN", "SNN", VERMILLION, "s", 0.18),
    )
    for index, (ax, target, title) in enumerate(zip(axes, ("sbp", "dbp"), ("Systolic BP", "Diastolic BP"))):
        for model, label, color, marker, offset in models:
            part = summary[(summary["target"].eq(target)) & (summary["model"].eq(model))].set_index("protocol").loc[protocols]
            means = part["mean_macro_subject_mae"].to_numpy()
            errors = part["seed_sd_macro_subject_mae"].to_numpy()
            for protocol_index, protocol in enumerate(protocols):
                raw = seeds[(seeds["target"].eq(target)) & (seeds["model"].eq(model)) &
                            (seeds["protocol"].eq(protocol))]["macro_subject_mae"].to_numpy()
                jitter = np.linspace(-0.035, 0.035, len(raw))
                ax.scatter(np.full(len(raw), x[protocol_index] + offset) + jitter, raw,
                           s=10, color=color, alpha=0.42, linewidths=0, zorder=2)
            ax.errorbar(x + offset, means, yerr=errors, fmt=marker, color=color, ms=5.0,
                        capsize=2.5, capthick=0.9, elinewidth=0.9, markerfacecolor="white",
                        markeredgewidth=1.0, linestyle="none", label=label, zorder=4)
        ax.set_xticks(x, ["Random\nwindow", "Subject\ndisjoint"])
        ax.set_ylabel("Subject-macro MAE (mmHg)")
        ax.set_title(title, fontsize=8, pad=5)
        ax.margins(x=0.20, y=0.12)
        clean_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    axes[0].legend(loc="upper left", handletextpad=0.4)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.22, top=0.88, wspace=0.32)
    save(fig, "public_luh_split_effect_nature")


def strict_model_suite() -> None:
    summary = pd.read_csv(IC / "results" / "bspc_model_suite_seed_summary.csv", encoding="utf-8-sig")
    seeds = pd.read_csv(IC / "results" / "bspc_model_suite_seed_metrics.csv", encoding="utf-8-sig")
    models = ["Face-ANN", "Face-SNN", "ResNet-1D", "TCN", "Transformer"]
    labels = ["ANN", "SNN", "ResNet", "TCN", "Transf."]
    cohorts = [("hospital", "Hospital"), ("lab", "Laboratory"), ("public_ukl", "Public UKL")]
    x = np.arange(len(models), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(7.08, 4.20), sharex=True)
    for row, (target, target_name) in enumerate((("sbp", "Systolic BP"), ("dbp", "Diastolic BP"))):
        for column, (cohort, cohort_name) in enumerate(cohorts):
            ax = axes[row, column]
            part = summary[
                summary["cohort"].eq(cohort)
                & summary["protocol"].eq("Subject-disjoint")
                & summary["target"].eq(target)
                & summary["model"].isin(models)
            ].set_index("model").loc[models]
            for position, model in enumerate(models):
                color = MODEL_COLORS[model]
                raw = seeds[
                    seeds["cohort"].eq(cohort)
                    & seeds["protocol"].eq("Subject-disjoint")
                    & seeds["target"].eq(target)
                    & seeds["model"].eq(model)
                ]["macro_subject_mae"].to_numpy()
                ax.scatter(
                    np.full(len(raw), position) + np.linspace(-0.07, 0.07, len(raw)),
                    raw,
                    s=10,
                    color=color,
                    alpha=0.30,
                    linewidths=0,
                    zorder=2,
                )
                ax.errorbar(
                    position,
                    part.loc[model, "mean_macro_subject_mae"],
                    yerr=part.loc[model, "seed_sd_macro_subject_mae"],
                    fmt="o",
                    color=color,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    markersize=4.8,
                    elinewidth=0.9,
                    capsize=2.3,
                    zorder=4,
                )
            ax.axhline(
                part.loc["Face-ANN", "mean_macro_subject_mae"],
                color=MODEL_COLORS["Face-ANN"],
                lw=0.7,
                ls=(0, (2, 2)),
                alpha=0.45,
                zorder=1,
            )
            ax.set_title(f"{cohort_name}: {target_name}", fontsize=8, pad=5)
            ax.set_ylabel("Subject-macro MAE (mmHg)" if column == 0 else "")
            ax.set_xticks(x, labels, rotation=28, ha="right")
            ax.margins(x=0.10, y=0.18)
            clean_axis(ax)
            panel_label(ax, chr(ord("a") + row * 3 + column))
    fig.text(
        0.995,
        0.015,
        "Points: training seeds; bars: mean ± SD",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.94, wspace=0.30, hspace=0.44)
    save(fig, "strict_model_suite_nature")


def strict_agreement() -> None:
    predictions = pd.read_csv(
        IC / "results" / "bspc_extension_predictions.csv", low_memory=False
    )
    predictions = predictions[
        predictions["protocol"].eq("Subject-disjoint")
        & predictions["model"].eq("Face-ANN")
    ]
    ensemble = predictions.groupby(
        ["cohort", "target", "subject_id", "unit_id", "unit_type"], as_index=False
    ).agg(actual=("actual", "first"), prediction=("prediction", "mean"))
    statistics = pd.read_csv(IC / "results" / "strict_agreement_statistics.csv")
    cohorts = [("hospital", "Hospital"), ("lab", "Laboratory"), ("public_ukl", "Public UKL")]
    targets = [("sbp", "SBP", BLUE), ("dbp", "DBP", VERMILLION)]
    fig, axes = plt.subplots(4, 3, figsize=(7.08, 7.15))
    for target_index, (target, target_label, color) in enumerate(targets):
        scatter_row = target_index * 2
        bland_row = scatter_row + 1
        for column, (cohort, cohort_label) in enumerate(cohorts):
            part = ensemble[
                ensemble["cohort"].eq(cohort) & ensemble["target"].eq(target)
            ]
            actual = part["actual"].to_numpy(float)
            predicted = part["prediction"].to_numpy(float)
            difference = predicted - actual
            mean_pair = (predicted + actual) / 2
            stat = statistics[
                statistics["cohort"].eq(cohort)
                & statistics["target"].eq(target.upper())
            ].iloc[0]

            ax = axes[scatter_row, column]
            ax.scatter(actual, predicted, s=4, color=color, alpha=0.16, linewidths=0,
                       rasterized=True)
            lower = min(actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())
            ax.plot([lower, upper], [lower, upper], color="#333333", lw=0.8,
                    ls=(0, (3, 2)))
            fitted = np.polyfit(actual, predicted, 1)
            ax.plot([lower, upper], np.polyval(fitted, [lower, upper]), color=color, lw=1.0)
            ax.text(0.04, 0.93, f"$R^2$={stat['pooled_r_squared']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=6.8)
            ax.set_title(cohort_label, fontsize=8, pad=5)
            ax.set_xlabel(f"Reference {target_label} (mmHg)")
            ax.set_ylabel(f"Predicted {target_label} (mmHg)" if column == 0 else "")
            clean_axis(ax)
            panel_label(ax, chr(ord("a") + scatter_row * 3 + column))

            ax = axes[bland_row, column]
            ax.scatter(mean_pair, difference, s=4, color=color, alpha=0.16, linewidths=0,
                       rasterized=True)
            bias = float(stat["subject_balanced_bias"])
            lower_loa = float(stat["loa_lower"])
            upper_loa = float(stat["loa_upper"])
            ax.axhline(bias, color=color, lw=1.0)
            ax.axhline(lower_loa, color="#555555", lw=0.75, ls=(0, (3, 2)))
            ax.axhline(upper_loa, color="#555555", lw=0.75, ls=(0, (3, 2)))
            ax.text(0.04, 0.93, f"bias={bias:.1f}; LoA [{lower_loa:.1f}, {upper_loa:.1f}]",
                    transform=ax.transAxes, va="top", fontsize=6.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.6})
            ax.set_xlabel(f"Mean {target_label} (mmHg)")
            ax.set_ylabel("Prediction - reference (mmHg)" if column == 0 else "")
            clean_axis(ax)
            panel_label(ax, chr(ord("a") + bland_row * 3 + column))
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.075, top=0.965,
                        wspace=0.28, hspace=0.62)
    save(fig, "strict_agreement_nature")


def main() -> None:
    set_style()
    cohort_distributions()
    strict_model_suite()
    strict_agreement()
    print(f"saved distribution, strict-model, and agreement figures to {OUT}")


if __name__ == "__main__":
    main()
