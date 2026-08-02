#!/usr/bin/env python3
"""Plot raw-matched Time-warp purity changes from the mechanism benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALGORITHMS = ("flowmop", "flowcut")
ALGORITHM_LABELS = {"flowmop": "FlowMOP", "flowcut": "FlowCut"}
VARIANTS = ("source_timewarp", "random_timewarp")
VARIANT_LABELS = {
    "source_timewarp": "Source-linked Time warp",
    "random_timewarp": "Random Time warp",
}
VARIANT_COLORS = {
    "source_timewarp": "#C44E52",
    "random_timewarp": "#4C72B0",
}
SUBSETS = (
    ("all", "All inputs"),
    ("segment", "Segment inputs"),
    ("bimix", "Bimix inputs"),
    ("trimix", "Trimix inputs"),
)
METRICS = (
    ("delta_sensitivity_percent", "Change in retained-target purity (%)", (-56, 45)),
    ("delta_specificity_percent", "Change in removed-non-target purity (%)", (-66, 75)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "benchmark_results/rate_density_mechanism/"
            "mixed_segment_timewarp_500k_strong_30files/results_with_raw_delta.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figs_data/revision_timewarp_mechanism.svg"),
    )
    parser.add_argument("--png", type=Path, default=Path("figs_data/revision_timewarp_mechanism.png"))
    return parser.parse_args()


def load_plot_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["status"].eq("ok")].copy()
    data = data[data["algorithm"].isin(ALGORITHMS) & data["variant"].isin(VARIANTS)]
    if data.empty:
        raise ValueError(f"no plottable rows found in {path}")
    data["algorithm_label"] = data["algorithm"].map(ALGORITHM_LABELS)
    data["variant_label"] = data["variant"].map(VARIANT_LABELS)
    data["delta_sensitivity_percent"] = 100 * data["delta_sensitivity_vs_raw"]
    data["delta_specificity_percent"] = 100 * data["delta_specificity_vs_raw"]
    return data


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if len(values) == 1:
        return mean, 0.0
    ci = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return mean, ci


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    seed: int,
    ylim: tuple[float, float] | None = None,
    show_ylabel: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    offsets = {"source_timewarp": -0.18, "random_timewarp": 0.18}

    for alg_index, algorithm in enumerate(ALGORITHMS):
        for variant in VARIANTS:
            subset = data[(data["algorithm"].eq(algorithm)) & (data["variant"].eq(variant))]
            values = subset[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            x = alg_index + offsets[variant]
            jitter = rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                s=22,
                color=VARIANT_COLORS[variant],
                alpha=0.45,
                linewidth=0,
                zorder=2,
            )
            mean, ci = mean_ci(values)
            ax.errorbar(
                x,
                mean,
                yerr=ci,
                fmt="o",
                markersize=6,
                color=VARIANT_COLORS[variant],
                markeredgecolor="black",
                markeredgewidth=0.6,
                capsize=4,
                linewidth=1.5,
                zorder=3,
            )

    ax.axhline(0, color="#333333", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=7)
    ax.set_xticks(range(len(ALGORITHMS)), [ALGORITHM_LABELS[item] for item in ALGORITHMS])
    ax.set_ylabel(ylabel if show_ylabel else "")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)


def add_legend(fig: plt.Figure) -> None:
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=VARIANT_COLORS[variant],
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label=VARIANT_LABELS[variant],
        )
        for variant in VARIANTS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.015),
        ncol=2,
        frameon=False,
        fontsize=9,
    )


def main() -> int:
    args = parse_args()
    data = load_plot_data(args.input)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=(12.8, 5.55), constrained_layout=False, sharey="row")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.95, wspace=0.22, hspace=0.46)

    seed = 1
    for col_index, (subset_key, subset_label) in enumerate(SUBSETS):
        subset = data if subset_key == "all" else data[data["mix_method"].eq(subset_key)]
        n_files = subset["base_file"].nunique()
        for row_index, (metric, ylabel, ylim) in enumerate(METRICS):
            letter = "ABCDEFGH"[row_index * len(SUBSETS) + col_index]
            draw_panel(
                axes[row_index, col_index],
                subset,
                metric,
                f"{letter}  {subset_label} (n={n_files})",
                ylabel,
                seed=seed,
                ylim=ylim,
                show_ylabel=col_index == 0,
            )
            seed += 1

    add_legend(fig)
    fig.text(
        0.5,
        0.095,
        "Negative values indicate reduced purity relative to the matched raw input.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    if args.png is not None:
        print(f"Wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
