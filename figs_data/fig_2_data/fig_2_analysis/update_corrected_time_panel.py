#!/usr/bin/env python3
"""Regenerate the Figure 2 time-gating panel with corrected competitors.

FlowMOP values are reconstructed from the original FlowJo exports used for the
manuscript. PeacoQC and FlowCut values are read from the leakage-corrected
benchmark analysis and matched by input filename.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import seaborn as sns
except ModuleNotFoundError:
    sys.path.append("/tmp/flowmop_figure2_deps")
    import seaborn as sns


METRICS = ("retained_target_purity", "removed_nontarget_purity")
METHODS = ("flowmop", "peacoqc", "flowcut")
MIX_METHODS = ("Segment", "Bimix", "Trimix")
METHOD_LABELS = ("FlowMOP", "PeacoQC", "FlowCut")
BIG_FONTS = {
    "base": 20,
    "label": 22,
    "tick": 18,
    "legend": 20,
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-primary-results", type=Path, required=True)
    parser.add_argument(
        "--output-data",
        type=Path,
        default=here / "fig_2_time_corrected_data.csv",
    )
    parser.add_argument(
        "--output-tests",
        type=Path,
        default=here / "fig_2_time_corrected_paired_tests.csv",
    )
    parser.add_argument(
        "--output-svg",
        type=Path,
        default=here / "svg_exports" / "fig_2_time_panel.svg",
    )
    return parser.parse_args()


def proportions_from_name(input_name: str) -> list[int]:
    token = Path(input_name).stem.split("_")[-2]
    return [int(value) for value in re.findall(r"(\d{2})", token)]


def reconstruct_flowmop(path: Path, dataset: str, bin_size: int) -> pd.DataFrame:
    source = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for record in source.itertuples(index=False, name=None):
        raw_name = str(record[0])
        if "Mean" in raw_name or "SD" in raw_name:
            continue
        input_name = raw_name.removeprefix("flowmop_")
        stem_parts = Path(input_name).stem.split("_")
        mix_method = stem_parts[-1].capitalize()
        proportions = proportions_from_name(input_name)
        if not proportions:
            continue
        exclude_primary = (
            len(proportions) == 2 and proportions[0] == proportions[1]
        )

        total = float(str(record[1]).replace("E?", "E-"))
        retained_fraction = float(record[2]) / 100.0
        retained_total = total * retained_fraction
        retained_composition = [
            float(record[3 + index]) / 100.0
            for index in range(len(proportions))
        ]
        maximum = max(proportions)
        target = [value == maximum for value in proportions]

        retained_target_purity = sum(
            value for value, is_target in zip(retained_composition, target)
            if is_target
        )
        original_counts = [total * value / 100.0 for value in proportions]
        retained_counts = [
            retained_total * value for value in retained_composition
        ]
        removed_total = total - retained_total
        removed_nontarget = sum(
            original - retained
            for original, retained, is_target in zip(
                original_counts, retained_counts, target
            )
            if not is_target
        )
        removed_nontarget_purity = (
            removed_nontarget / removed_total if removed_total else math.nan
        )
        rows.append(
            {
                "dataset": dataset,
                "synthetic_bin_size": bin_size,
                "input_name": input_name,
                "mix_method": mix_method,
                "algorithm": "flowmop",
                "exclude_primary_5050": exclude_primary,
                "retained_target_purity": retained_target_purity,
                "removed_nontarget_purity": removed_nontarget_purity,
            }
        )
    return pd.DataFrame(rows)


def load_combined_data(corrected_results: Path, here: Path) -> pd.DataFrame:
    flowmop = pd.concat(
        [
            reconstruct_flowmop(
                here / "flowmop_timegates_combos.csv", "largecut", 5000
            ),
            reconstruct_flowmop(
                here / "flowmop_timegates_smallcut.csv", "smallcut", 2000
            ),
        ],
        ignore_index=True,
    )
    corrected = pd.read_csv(corrected_results)
    corrected = corrected[
        corrected["algorithm"].isin(("peacoqc", "flowcut"))
    ].copy()
    corrected["mix_method"] = corrected["mix_method"].str.capitalize()
    corrected = corrected[
        [
            "dataset",
            "synthetic_bin_size",
            "input_name",
            "mix_method",
            "algorithm",
            "exclude_primary_5050",
            *METRICS,
        ]
    ]
    combined = pd.concat([flowmop, corrected], ignore_index=True)
    combined = combined[~combined["exclude_primary_5050"].astype(bool)].copy()
    combined["algorithm"] = pd.Categorical(
        combined["algorithm"], categories=METHODS, ordered=True
    )
    combined["mix_method"] = pd.Categorical(
        combined["mix_method"], categories=MIX_METHODS, ordered=True
    )
    return combined.sort_values(
        ["synthetic_bin_size", "mix_method", "input_name", "algorithm"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def paired_tests(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, bin_size, mix_method), group in data.groupby(
        ["dataset", "synthetic_bin_size", "mix_method"], observed=True
    ):
        for metric in METRICS:
            pivot = group.pivot(
                index="input_name", columns="algorithm", values=metric
            )
            for method_a, method_b in itertools.combinations(METHODS, 2):
                paired = pivot[[method_a, method_b]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                statistic, raw_p = stats.ttest_rel(
                    paired[method_a], paired[method_b]
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "synthetic_bin_size": int(bin_size),
                        "mix_method": str(mix_method),
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "n_pairs": len(paired),
                        "mean_a": paired[method_a].mean(),
                        "mean_b": paired[method_b].mean(),
                        "mean_difference": (
                            paired[method_a] - paired[method_b]
                        ).mean(),
                        "t_statistic": statistic,
                        "p_value_raw": raw_p,
                        "p_value_bonferroni": min(float(raw_p) * 3.0, 1.0),
                    }
                )
    return pd.DataFrame(rows)


def plot_panel(data: pd.DataFrame, output_svg: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": BIG_FONTS["base"],
            "axes.titlesize": BIG_FONTS["label"],
            "axes.labelsize": BIG_FONTS["label"],
            "xtick.labelsize": BIG_FONTS["tick"],
            "ytick.labelsize": BIG_FONTS["tick"],
            "legend.fontsize": BIG_FONTS["legend"],
            "legend.title_fontsize": BIG_FONTS["legend"],
            "svg.fonttype": "path",
        }
    )
    sns.set_theme(style="ticks")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
    titles = [
        [
            "Sensitivity\nBin Size 5000",
            "Specificity\nBin Size 5000",
        ],
        ["Bin Size 2000", "Bin Size 2000"],
    ]
    handles = labels = None
    for row, bin_size in enumerate((5000, 2000)):
        subset = data[data["synthetic_bin_size"] == bin_size]
        for column, metric in enumerate(METRICS):
            axis = axes[row, column]
            sns.violinplot(
                data=subset,
                x="mix_method",
                y=metric,
                hue="algorithm",
                order=MIX_METHODS,
                hue_order=METHODS,
                ax=axis,
                cut=0,
                inner="quartile",
                linewidth=1,
                dodge=True,
                palette="deep",
            )
            axis.set_title(
                titles[row][column],
                loc="center",
                fontsize=BIG_FONTS["label"],
                weight="bold",
                pad=28,
            )
            axis.set_ylabel("")
            axis.set_xlabel(
                "Mix Method" if row == 1 else "",
                labelpad=15,
                fontsize=BIG_FONTS["label"],
            )
            axis.tick_params(
                axis="both", which="major", labelsize=BIG_FONTS["tick"]
            )
            if row == 0 and column == 0:
                handles, labels = axis.get_legend_handles_labels()
            if axis.get_legend() is not None:
                axis.get_legend().remove()
            _, ymax = axis.get_ylim()
            axis.set_ylim(bottom=0, top=ymax)

    legend = figure.legend(
        handles,
        METHOD_LABELS,
        title="Cleaning Method\n",
        bbox_to_anchor=(0.5, 0),
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=BIG_FONTS["legend"],
        title_fontsize=BIG_FONTS["legend"],
    )
    plt.setp(legend.get_title(), multialignment="center")
    sns.despine(figure)
    plt.tight_layout(rect=[0, 0.12, 1, 1])
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve().parent
    data = load_combined_data(args.corrected_primary_results, here)
    tests = paired_tests(data)
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_data, index=False)
    tests.to_csv(args.output_tests, index=False)
    plot_panel(data, args.output_svg)
    print(f"Wrote {args.output_data}")
    print(f"Wrote {args.output_tests}")
    print(f"Wrote {args.output_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
