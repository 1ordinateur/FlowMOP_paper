#!/usr/bin/env python3
"""Full Segment benchmark for PeacoQC bin resolution and removal localisation."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_DATASETS = (
    ("smallcut", 2000, Path("/mnt/d/github_remotes/flowmop_data/synthetic_combos_smallcut")),
    ("largecut", 5000, Path("/mnt/d/github_remotes/flowmop_data/synthetic_combos_largecut")),
)
DEFAULT_SETTINGS = ("auto", "1000", "2500", "5000")
DEFAULT_OUT_DIR = Path("benchmark_results/peacoqc_segment_localization/full_segment")
FLOWMOP_OUTPUT_ROOTS = {
    "smallcut": Path(
        "benchmark_results/mad_smoothing_smallcut_unclamped_010_090/runs/0p1_0p9"
    ),
    "largecut": Path("benchmark_results/mad_smoothing_largecut_full/runs/0p1_0p9"),
}


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    dataset: str
    synthetic_bin_size: int
    proportions: tuple[int, int]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    requested_setting: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--rscript",
        type=Path,
        default=Path("/tmp/flowmop-r/bin/Rscript"),
    )
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Reuse completed R outputs and rebuild summaries and figures only.",
    )
    return parser.parse_args()


def proportions_from_path(path: Path) -> tuple[int, int]:
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[-1].lower() != "segment":
        raise ValueError(f"not a Segment benchmark filename: {path.name}")
    token = parts[-2]
    if not token.isdigit() or len(token) != 4:
        raise ValueError(f"invalid Segment proportion token in {path.name}")
    return int(token[:2]), int(token[2:])


def discover_datasets(limit: int | None = None) -> tuple[list[DatasetSpec], list[str]]:
    datasets: list[DatasetSpec] = []
    excluded: list[str] = []
    for dataset, synthetic_bin_size, directory in DEFAULT_DATASETS:
        for path in sorted(directory.glob("*_segment.fcs")):
            proportions = proportions_from_path(path)
            maximum = max(proportions)
            target_ids = tuple(
                index + 1 for index, value in enumerate(proportions) if value == maximum
            )
            if len(target_ids) != 1:
                excluded.append(path.name)
                continue
            datasets.append(
                DatasetSpec(
                    path=path,
                    dataset=dataset,
                    synthetic_bin_size=synthetic_bin_size,
                    proportions=proportions,
                    target_ids=target_ids,
                )
            )
    datasets.sort(key=lambda item: (item.dataset, item.path.name))
    if limit is not None:
        datasets = datasets[:limit]
    if not datasets:
        raise FileNotFoundError("no evaluable Segment FCS files found")
    return datasets, excluded


def normalize_settings(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        setting = str(value).lower()
        if setting != "auto":
            numeric = int(setting)
            if numeric < 1:
                raise ValueError("fixed events-per-bin settings must be positive")
            setting = str(numeric)
        if setting not in normalized:
            normalized.append(setting)
    return normalized


def build_runs(
    datasets: list[DatasetSpec], settings: list[str], out_dir: Path
) -> list[RunSpec]:
    return [
        RunSpec(
            dataset=dataset,
            requested_setting=setting,
            output_dir=(
                out_dir
                / "runs"
                / dataset.dataset
                / dataset.path.stem
                / f"epb_{setting}"
            ),
        )
        for dataset in datasets
        for setting in settings
    ]


def run_one(run: RunSpec, args: argparse.Namespace, runner: Path) -> RunSpec:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(args.rscript),
        str(runner),
        str(run.dataset.path),
        run.requested_setting,
        ",".join(str(value) for value in run.dataset.target_ids),
        str(run.output_dir),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PeacoQC failed for {run.dataset.path.name}, setting={run.requested_setting}:\n"
            f"{completed.stderr[-4000:]}"
        )
    return run


def collect_peacoqc(runs: list[RunSpec]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in runs:
        path = run.output_dir / "run_summary.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"missing PeacoQC output for {run.dataset.path.name}, "
                f"setting={run.requested_setting}"
            )
        frame = pd.read_csv(path)
        frame.insert(1, "dataset", run.dataset.dataset)
        frame.insert(2, "synthetic_bin_size", run.dataset.synthetic_bin_size)
        frame.insert(3, "proportions", ",".join(map(str, run.dataset.proportions)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def transition_distance(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transitions = np.flatnonzero(labels[1:] != labels[:-1])
    if len(transitions) == 0:
        return transitions, np.full(len(labels), np.nan)
    indices = np.arange(len(labels))
    distance = np.full(len(labels), np.inf)
    for transition in transitions:
        distance = np.minimum(
            distance,
            np.minimum(np.abs(indices - transition), np.abs(indices - (transition + 1))),
        )
    return transitions, distance


def boundary_spillover(
    removed: np.ndarray, target: np.ndarray, transitions: np.ndarray
) -> tuple[float, float]:
    widths: list[int] = []
    for transition in transitions:
        left = int(transition)
        if target[left]:
            count = 0
            while left >= 0 and target[left] and removed[left]:
                count += 1
                left -= 1
            widths.append(count)
        right = int(transition + 1)
        if right < len(target) and target[right]:
            count = 0
            while right < len(target) and target[right] and removed[right]:
                count += 1
                right += 1
            widths.append(count)
    if not widths:
        return math.nan, math.nan
    return float(max(widths)), float(sum(widths))


def score_mask(
    retained: np.ndarray,
    labels: np.ndarray,
    target_ids: tuple[int, ...],
) -> dict[str, object]:
    target = np.isin(labels, target_ids)
    removed = ~retained
    nontarget = ~target
    retained_count = int(retained.sum())
    removed_count = int(removed.sum())
    target_count = int(target.sum())
    nontarget_count = int(nontarget.sum())
    retained_target = int((retained & target).sum())
    removed_target = int((removed & target).sum())
    removed_nontarget = int((removed & nontarget).sum())
    transitions, distances = transition_distance(labels)
    removed_target_distances = distances[removed & target]
    spillover_max, spillover_total = boundary_spillover(removed, target, transitions)
    return {
        "events": len(labels),
        "source_transition_count": len(transitions),
        "retained_count": retained_count,
        "removed_count": removed_count,
        "retained_fraction": retained_count / len(labels),
        "removed_fraction": removed_count / len(labels),
        "retained_target_purity": retained_target / retained_count
        if retained_count
        else math.nan,
        "removed_nontarget_purity": removed_nontarget / removed_count
        if removed_count
        else math.nan,
        "target_retention": retained_target / target_count if target_count else math.nan,
        "target_removal_fraction": removed_target / target_count if target_count else math.nan,
        "nontarget_recall": removed_nontarget / nontarget_count
        if nontarget_count
        else math.nan,
        "removed_target_count": removed_target,
        "removed_nontarget_count": removed_nontarget,
        "removed_target_distance_median": float(np.median(removed_target_distances))
        if len(removed_target_distances)
        else math.nan,
        "removed_target_distance_p95": float(np.quantile(removed_target_distances, 0.95))
        if len(removed_target_distances)
        else math.nan,
        "removed_target_distance_max": float(np.max(removed_target_distances))
        if len(removed_target_distances)
        else math.nan,
        "boundary_target_spillover_max": spillover_max,
        "boundary_target_spillover_total": spillover_total,
    }


def flowmop_lookup(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in root.glob("**/flowmop_*_segment.fcs"):
        input_name = re.sub(r"^flowmop_", "", path.name)
        if input_name in found:
            raise ValueError(f"duplicate FlowMOP output for {input_name}")
        found[input_name] = path
    return found


def collect_flowmop(datasets: list[DatasetSpec]) -> pd.DataFrame:
    from fcsparser import parse

    lookups = {dataset: flowmop_lookup(root) for dataset, root in FLOWMOP_OUTPUT_ROOTS.items()}
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        output = lookups[dataset.dataset].get(dataset.path.name)
        if output is None:
            raise FileNotFoundError(f"missing fixed-setting FlowMOP output for {dataset.path.name}")
        _, frame = parse(str(output), reformat_meta=True)
        if "passed_time" not in frame or "SampleIDInt" not in frame:
            raise ValueError(f"required FlowMOP columns missing from {output}")
        retained = frame["passed_time"].to_numpy(dtype=float) > 0.5
        labels = frame["SampleIDInt"].round().to_numpy(dtype=int)
        row = {
            "input_fcs": dataset.path.name,
            "dataset": dataset.dataset,
            "synthetic_bin_size": dataset.synthetic_bin_size,
            "proportions": ",".join(map(str, dataset.proportions)),
            "target_source_ids": ",".join(map(str, dataset.target_ids)),
            "algorithm": "flowmop",
            "requested_setting": "0.1,0.9",
            "output_fcs": str(output),
            **score_mask(retained, labels, dataset.target_ids),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def mean_ci(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(numeric) == 0:
        return math.nan, math.nan
    mean = float(np.mean(numeric))
    if len(numeric) == 1:
        return mean, math.nan
    ci = 1.96 * float(np.std(numeric, ddof=1)) / math.sqrt(len(numeric))
    return mean, ci


def holm_adjust(values: list[float]) -> list[float]:
    output = [math.nan] * len(values)
    finite = sorted(
        ((index, value) for index, value in enumerate(values) if np.isfinite(value)),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, value) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - rank) * value))
        output[index] = running
    return output


def paired_test(delta: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(delta, errors="coerce").dropna().to_numpy(dtype=float)
    mean, ci = mean_ci(pd.Series(values))
    if len(values) > 1 and np.any(np.abs(values) > 0):
        wilcoxon = stats.wilcoxon(values, alternative="two-sided")
        paired_t = stats.ttest_1samp(values, popmean=0)
        return {
            "pairs": len(values),
            "mean_change": mean,
            "ci95_change": ci,
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_p_value": float(wilcoxon.pvalue),
            "paired_t_statistic": float(paired_t.statistic),
            "paired_t_p_value": float(paired_t.pvalue),
        }
    return {
        "pairs": len(values),
        "mean_change": mean,
        "ci95_change": ci,
        "wilcoxon_statistic": math.nan,
        "wilcoxon_p_value": math.nan,
        "paired_t_statistic": math.nan,
        "paired_t_p_value": math.nan,
    }


def write_analysis(
    peacoqc: pd.DataFrame, flowmop: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "retained_target_purity",
        "removed_nontarget_purity",
        "target_retention",
        "target_removal_fraction",
        "nontarget_recall",
        "removed_fraction",
        "removed_target_distance_median",
        "removed_target_distance_p95",
        "boundary_target_spillover_max",
    ]
    summary_rows: list[dict[str, object]] = []
    for keys, group in peacoqc.groupby(
        ["requested_setting", "dataset", "synthetic_bin_size"], sort=False
    ):
        setting, dataset, synthetic_bin_size = keys
        row: dict[str, object] = {
            "algorithm": "peacoqc",
            "requested_setting": setting,
            "dataset": dataset,
            "synthetic_bin_size": synthetic_bin_size,
            "files": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        summary_rows.append(row)
    for setting, group in peacoqc.groupby("requested_setting", sort=False):
        row = {
            "algorithm": "peacoqc",
            "requested_setting": setting,
            "dataset": "all",
            "synthetic_bin_size": "all",
            "files": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        summary_rows.append(row)
    for dataset, group in flowmop.groupby("dataset", sort=False):
        row = {
            "algorithm": "flowmop",
            "requested_setting": "0.1,0.9",
            "dataset": dataset,
            "synthetic_bin_size": int(group["synthetic_bin_size"].iloc[0]),
            "files": len(group),
        }
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        summary_rows.append(row)
    row = {
        "algorithm": "flowmop",
        "requested_setting": "0.1,0.9",
        "dataset": "all",
        "synthetic_bin_size": "all",
        "files": len(flowmop),
    }
    for metric in metrics:
        row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(flowmop[metric])
    summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    contrast_rows: list[dict[str, object]] = []
    alternatives = [value for value in peacoqc["requested_setting"].unique() if value != "auto"]
    for metric in metrics:
        metric_rows: list[dict[str, object]] = []
        wide = peacoqc.pivot(
            index=["dataset", "input_fcs"],
            columns="requested_setting",
            values=metric,
        )
        for setting in alternatives:
            pair = wide[["auto", setting]].dropna()
            metric_rows.append(
                {
                    "metric": metric,
                    "reference_setting": "auto",
                    "comparison_setting": setting,
                    **paired_test(pair[setting] - pair["auto"]),
                }
            )
        adjusted = holm_adjust([float(row["wilcoxon_p_value"]) for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["wilcoxon_holm_p_value"] = adjusted_p
        contrast_rows.extend(metric_rows)
    setting_contrasts = pd.DataFrame(contrast_rows)
    setting_contrasts.to_csv(out_dir / "peacoqc_setting_contrasts.csv", index=False)

    flowmop_rows: list[dict[str, object]] = []
    for metric in metrics:
        metric_rows = []
        for setting, group in peacoqc.groupby("requested_setting", sort=False):
            pair = group[["dataset", "input_fcs", metric]].merge(
                flowmop[["dataset", "input_fcs", metric]],
                on=["dataset", "input_fcs"],
                suffixes=("_peacoqc", "_flowmop"),
                validate="one_to_one",
            )
            metric_rows.append(
                {
                    "metric": metric,
                    "peacoqc_setting": setting,
                    **paired_test(pair[f"{metric}_peacoqc"] - pair[f"{metric}_flowmop"]),
                }
            )
        adjusted = holm_adjust([float(row["wilcoxon_p_value"]) for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["wilcoxon_holm_p_value"] = adjusted_p
        flowmop_rows.extend(metric_rows)
    flowmop_contrasts = pd.DataFrame(flowmop_rows)
    flowmop_contrasts.to_csv(out_dir / "flowmop_contrasts.csv", index=False)
    return summary, setting_contrasts, flowmop_contrasts


def plot_results(peacoqc: pd.DataFrame, flowmop: pd.DataFrame, output: Path) -> None:
    metrics = (
        ("removed_nontarget_purity", "Removed-non-target purity"),
        ("target_removal_fraction", "Target-source events removed"),
        ("removed_target_distance_p95", "95th percentile distance of\nremoved target events from transition"),
    )
    order = [value for value in DEFAULT_SETTINGS if value in set(peacoqc["requested_setting"])]
    positions = np.arange(len(order))
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), constrained_layout=True)
    for axis, (metric, label) in zip(axes, metrics):
        for _, group in peacoqc.groupby(["dataset", "input_fcs"]):
            values = group.set_index("requested_setting")[metric].reindex(order)
            axis.plot(positions, values, color="#4C72B0", alpha=0.12, linewidth=0.7)
        means = peacoqc.groupby("requested_setting")[metric].mean().reindex(order)
        axis.plot(positions, means, color="#4C72B0", marker="o", linewidth=2.2, label="PeacoQC")
        flowmop_mean = flowmop[metric].mean()
        axis.axhline(flowmop_mean, color="#C44E52", linestyle="--", linewidth=1.8, label="FlowMOP rerun")
        axis.set_xticks(positions, ["Automatic" if value == "auto" else f"{int(value):,}" for value in order])
        axis.set_xlabel("PeacoQC events per bin")
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#E0E0E0", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if not args.rscript.exists():
        raise FileNotFoundError(f"Rscript not found: {args.rscript}")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    settings = normalize_settings(args.settings)
    datasets, excluded = discover_datasets(args.limit_files)
    runs = build_runs(datasets, settings, args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_peacoqc_segment_localization.R")
    metadata = {
        "datasets": [
            {
                "input_fcs": str(dataset.path),
                "dataset": dataset.dataset,
                "synthetic_bin_size": dataset.synthetic_bin_size,
                "proportions": dataset.proportions,
                "target_ids": dataset.target_ids,
            }
            for dataset in datasets
        ],
        "excluded_equal_mixtures": excluded,
        "settings": settings,
        "jobs": args.jobs,
        "rscript": str(args.rscript),
        "runner": str(runner),
        "flowmop_reference": {
            key: str(value) for key, value in FLOWMOP_OUTPUT_ROOTS.items()
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if not args.skip_runs:
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_one, run, args, runner): run for run in runs}
            completed_count = 0
            for future in as_completed(futures):
                run = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
                    print(
                        f"FAILED {run.dataset.path.name} setting={run.requested_setting}: {exc}",
                        flush=True,
                    )
                else:
                    completed_count += 1
                    print(
                        f"Completed {completed_count}/{len(runs)}: {run.dataset.path.name} "
                        f"setting={run.requested_setting}",
                        flush=True,
                    )
        if failures:
            raise RuntimeError("\n\n".join(failures))

    peacoqc = collect_peacoqc(runs)
    peacoqc.insert(4, "algorithm", "peacoqc")
    flowmop = collect_flowmop(datasets)
    peacoqc.to_csv(args.out_dir / "peacoqc_results.csv", index=False)
    flowmop.to_csv(args.out_dir / "flowmop_results.csv", index=False)
    write_analysis(peacoqc, flowmop, args.out_dir)
    plot_results(peacoqc, flowmop, args.out_dir / "segment_localization.svg")
    print(f"Wrote full Segment analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
