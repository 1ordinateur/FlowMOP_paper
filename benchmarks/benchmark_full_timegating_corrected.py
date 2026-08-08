#!/usr/bin/env python3
"""Run and analyse the leakage-corrected full Figure 2 time-gating benchmark."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ALGORITHMS = ("flowmop", "peacoqc", "flowcut")
METRICS = (
    "retained_target_purity",
    "removed_nontarget_purity",
    "target_retention",
    "nontarget_recall",
    "removed_fraction",
)
EXPECTED_COUNTS = {"smallcut": 87, "largecut": 92}


@dataclass(frozen=True)
class InputSpec:
    dataset: str
    synthetic_bin_size: int
    path: Path
    mix_method: str
    proportions: tuple[int, ...]
    target_ids: tuple[int, ...]

    @property
    def exclude_primary_5050(self) -> bool:
        return len(self.proportions) == 2 and self.proportions == (50, 50)


def proportions_from_token(token: str) -> tuple[int, ...]:
    if not token.isdigit() or len(token) % 2:
        raise ValueError(f"invalid proportion token {token!r}")
    return tuple(int(token[index : index + 2]) for index in range(0, len(token), 2))


def parse_input(path: Path, dataset: str, synthetic_bin_size: int) -> InputSpec:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot parse synthetic benchmark filename {path.name}")
    mix_method = parts[-1].lower()
    if mix_method not in {"segment", "bimix", "trimix"}:
        raise ValueError(f"unknown mixture type in {path.name}")
    proportions = proportions_from_token(parts[-2])
    expected_sources = 3 if mix_method == "trimix" else 2
    if len(proportions) != expected_sources:
        raise ValueError(
            f"{path.name} has {len(proportions)} proportions; expected {expected_sources}"
        )
    maximum = max(proportions)
    target_ids = tuple(
        index + 1 for index, value in enumerate(proportions) if value == maximum
    )
    return InputSpec(
        dataset=dataset,
        synthetic_bin_size=synthetic_bin_size,
        path=path.resolve(),
        mix_method=mix_method,
        proportions=proportions,
        target_ids=target_ids,
    )


def discover_inputs(
    smallcut_dir: Path, largecut_dir: Path, require_expected_counts: bool = True
) -> list[InputSpec]:
    inputs: list[InputSpec] = []
    for dataset, bin_size, directory in (
        ("smallcut", 2000, smallcut_dir),
        ("largecut", 5000, largecut_dir),
    ):
        paths = sorted(directory.glob("*.fcs"))
        if require_expected_counts and len(paths) != EXPECTED_COUNTS[dataset]:
            raise ValueError(
                f"{dataset} contains {len(paths)} FCS files; expected {EXPECTED_COUNTS[dataset]}"
            )
        inputs.extend(parse_input(path, dataset, bin_size) for path in paths)
    inputs.sort(key=lambda item: (item.dataset, item.path.name))
    return inputs


def preflight_subset(inputs: Sequence[InputSpec]) -> list[InputSpec]:
    selected: list[InputSpec] = []
    for dataset in ("smallcut", "largecut"):
        for mix_method in ("segment", "bimix", "trimix"):
            matches = [
                item
                for item in inputs
                if item.dataset == dataset
                and item.mix_method == mix_method
                and not item.exclude_primary_5050
            ]
            if not matches:
                raise ValueError(f"no non-tied {dataset}/{mix_method} preflight input")
            selected.append(matches[0])
    return selected


def write_manifest(inputs: Sequence[InputSpec], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "array_index",
                "dataset",
                "synthetic_bin_size",
                "input_fcs",
                "input_name",
                "mix_method",
                "proportions",
                "target_source_ids",
                "exclude_primary_5050",
            ),
        )
        writer.writeheader()
        for index, item in enumerate(inputs, start=1):
            writer.writerow(
                {
                    "array_index": index,
                    "dataset": item.dataset,
                    "synthetic_bin_size": item.synthetic_bin_size,
                    "input_fcs": item.path,
                    "input_name": item.path.name,
                    "mix_method": item.mix_method,
                    "proportions": ",".join(map(str, item.proportions)),
                    "target_source_ids": ",".join(map(str, item.target_ids)),
                    "exclude_primary_5050": str(item.exclude_primary_5050).lower(),
                }
            )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    expected = list(range(1, len(rows) + 1))
    observed = [int(row["array_index"]) for row in rows]
    if observed != expected:
        raise ValueError("manifest array indices must be contiguous and one-based")
    return rows


def run_logged(
    command: Sequence[str], log_path: Path, timeout: float | None, env: dict[str, str]
) -> None:
    started = time.time()
    completed = subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=env,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + f"\n\nWALL_SECONDS\n{time.time() - started:.6f}"
        + f"\n\nEXIT_CODE\n{completed.returncode}"
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )


def valid_result(path: Path, algorithm: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return (
            len(rows) == 1
            and rows[0].get("algorithm") == algorithm
            and rows[0].get("sample_channel_used_for_qc", "").upper() == "FALSE"
        )
    except Exception:
        return False


def run_index(args: argparse.Namespace) -> None:
    rows = read_manifest(args.manifest)
    if args.index < 1 or args.index > len(rows):
        raise IndexError(f"array index {args.index} outside 1..{len(rows)}")
    row = rows[args.index - 1]
    source = Path(row["input_fcs"])
    if not source.exists():
        raise FileNotFoundError(source)

    result_root = args.out_dir / "runs" / row["dataset"] / source.stem
    result_root.mkdir(parents=True, exist_ok=True)
    target_ids = row["target_source_ids"]
    runner = args.repo_root / "benchmarks" / "run_corrected_timegating_method.R"
    flowmop_exec = args.flowmop_root / "flowmop_exec.py"
    if not runner.exists() or not flowmop_exec.exists():
        raise FileNotFoundError(f"missing runner ({runner}) or FlowMOP executable ({flowmop_exec})")

    scratch_parent = args.scratch_dir
    scratch_parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(scratch_parent / "matplotlib"))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("DASK_NUM_WORKERS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    with tempfile.TemporaryDirectory(prefix=f"flowmop_fig2_{args.index}_", dir=scratch_parent) as tmp:
        scratch = Path(tmp)
        staged_input = scratch / source.name
        shutil.copy2(source, staged_input)

        algorithms = tuple(args.algorithms)
        for algorithm in algorithms:
            output_csv = result_root / f"{algorithm}.csv"
            if valid_result(output_csv, algorithm) and not args.force:
                continue
            log_path = result_root / f"{algorithm}.log"
            if algorithm == "flowmop":
                flowmop_dir = scratch / "flowmop_output"
                command = [
                    str(args.python_bin),
                    str(flowmop_exec),
                    str(staged_input),
                    "--output-dir",
                    str(flowmop_dir),
                    "--fluor-mode",
                    "positive_geomeans",
                    "--mad-smoothing",
                    *(str(value) for value in args.mad_smoothing),
                    "--mad-factor",
                    str(args.mad_factor),
                    "--skip-debris",
                    "--skip-doublets",
                    "--disable-dask",
                ]
                run_logged(command, log_path, args.timeout, env)
                annotated = flowmop_dir / f"flowmop_{staged_input.stem}.fcs"
                if not annotated.exists():
                    raise FileNotFoundError(f"FlowMOP did not create {annotated}")
                score_command = [
                    str(args.rscript),
                    str(runner),
                    "flowmop",
                    str(annotated),
                    target_ids,
                    str(output_csv),
                ]
                run_logged(
                    score_command,
                    result_root / "flowmop_scoring.log",
                    args.timeout,
                    env,
                )
            else:
                command = [
                    str(args.rscript),
                    str(runner),
                    algorithm,
                    str(staged_input),
                    target_ids,
                    str(output_csv),
                ]
                run_logged(command, log_path, args.timeout, env)
            if not valid_result(output_csv, algorithm):
                raise RuntimeError(f"invalid or leakage-positive result: {output_csv}")

    completion = {
        "array_index": args.index,
        "dataset": row["dataset"],
        "input_name": source.name,
        "algorithms": list(algorithms),
        "completed_at_epoch": time.time(),
        "pbs_jobid": os.environ.get("PBS_JOBID", "manual"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    (result_root / "complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    total = len(indexed)
    for rank, (index, value) in enumerate(indexed):
        candidate = min(1.0, value * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def plot_primary_results(primary: "object", out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    method_order = list(ALGORITHMS)
    method_labels = {"flowmop": "FlowMOP", "peacoqc": "PeacoQC", "flowcut": "FlowCut"}
    mix_order = ["segment", "bimix", "trimix"]
    mix_labels = ["Segment", "Bimix", "Trimix"]
    dataset_order = ["largecut", "smallcut"]
    metric_order = ["retained_target_purity", "removed_nontarget_purity"]
    metric_labels = ["Retained-target purity", "Removed-nontarget purity"]

    colors = {"flowmop": "#4C72B0", "peacoqc": "#DD8452", "flowcut": "#55A868"}
    offsets = {"flowmop": -0.25, "peacoqc": 0.0, "flowcut": 0.25}
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15})
    figure, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    for row_index, dataset in enumerate(dataset_order):
        subset = primary[primary["dataset"] == dataset].copy()
        for column_index, (metric, metric_label) in enumerate(zip(metric_order, metric_labels)):
            axis = axes[row_index, column_index]
            if subset.empty:
                axis.set_visible(False)
                continue
            for mix_index, mix_method in enumerate(mix_order):
                for algorithm in method_order:
                    values = subset.loc[
                        (subset["mix_method"] == mix_method)
                        & (subset["algorithm"] == algorithm),
                        metric,
                    ].dropna().to_numpy(dtype=float)
                    if len(values) == 0:
                        continue
                    position = mix_index + offsets[algorithm]
                    if len(np.unique(values)) > 1:
                        violin = axis.violinplot(
                            values,
                            positions=[position],
                            widths=0.22,
                            showmeans=False,
                            showmedians=False,
                            showextrema=False,
                        )
                        for body in violin["bodies"]:
                            body.set_facecolor(colors[algorithm])
                            body.set_edgecolor("black")
                            body.set_alpha(0.65)
                    else:
                        axis.scatter(
                            [position], [values[0]], s=60, color=colors[algorithm], edgecolor="black"
                        )
                    quartiles = np.percentile(values, [25, 50, 75])
                    axis.vlines(position, quartiles[0], quartiles[2], color="black", linewidth=2)
                    axis.scatter([position], [quartiles[1]], color="white", edgecolor="black", s=32, zorder=4)
                    jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) > 1 else np.array([0.0])
                    axis.scatter(
                        position + jitter,
                        values,
                        s=14,
                        color=colors[algorithm],
                        edgecolor="none",
                        alpha=0.45,
                        zorder=3,
                    )
            axis.set_xticks(range(len(mix_order)), mix_labels)
            axis.set_ylim(0, 1.03)
            axis.set_xlabel("Synthetic mixture" if row_index == 1 else "")
            axis.set_ylabel(metric_label if column_index == 0 else "")
            bin_size = int(subset["synthetic_bin_size"].iloc[0])
            axis.set_title(f"{metric_label}\n{bin_size:,}-event intervals")
            axis.spines[["top", "right"]].set_visible(False)
    figure.legend(
        [Patch(facecolor=colors[method], edgecolor="black", label=method_labels[method]) for method in method_order],
        [method_labels[method] for method in method_order],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(out_dir / "corrected_figure2_time_panel.svg", bbox_inches="tight")
    figure.savefig(out_dir / "corrected_figure2_time_panel.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyse(args: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    from scipy import stats

    manifest = pd.read_csv(args.manifest)
    result_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for row in manifest.itertuples(index=False):
        stem = Path(row.input_name).stem
        root = args.results_dir / "runs" / row.dataset / stem
        for algorithm in ALGORITHMS:
            path = root / f"{algorithm}.csv"
            if not valid_result(path, algorithm):
                missing.append(f"{row.dataset}/{stem}/{algorithm}")
                continue
            frame = pd.read_csv(path)
            frame["dataset"] = row.dataset
            frame["synthetic_bin_size"] = int(row.synthetic_bin_size)
            frame["input_name"] = row.input_name
            frame["mix_method"] = row.mix_method
            frame["proportions"] = row.proportions
            frame["exclude_primary_5050"] = bool(row.exclude_primary_5050)
            result_frames.append(frame)
    if missing and not args.allow_partial:
        raise RuntimeError(f"missing {len(missing)} results; first entries: {missing[:12]}")
    if not result_frames:
        raise RuntimeError("no completed benchmark results")

    results = pd.concat(result_frames, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "all_results.csv", index=False)
    primary = results[~results["exclude_primary_5050"]].copy()
    primary.to_csv(args.out_dir / "primary_results_excluding_ties.csv", index=False)
    if primary["algorithm"].nunique() == len(ALGORITHMS):
        plot_primary_results(primary, args.out_dir)

    summary = (
        primary.groupby(
            ["dataset", "synthetic_bin_size", "mix_method", "algorithm"],
            dropna=False,
        )[list(METRICS)]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    summary.to_csv(args.out_dir / "summary.csv", index=False)

    test_rows: list[dict[str, object]] = []
    method_pairs = list(itertools.combinations(ALGORITHMS, 2))
    grouped = primary.groupby(["dataset", "synthetic_bin_size", "mix_method"])
    for (dataset, bin_size, mix_method), group in grouped:
        for metric in ("retained_target_purity", "removed_nontarget_purity"):
            group_rows: list[dict[str, object]] = []
            raw_p_values: list[float] = []
            for method_a, method_b in method_pairs:
                pivot = group.pivot(index="input_name", columns="algorithm", values=metric)
                paired = pivot[[method_a, method_b]].dropna()
                if len(paired) < 2:
                    t_statistic = t_p_value = wilcoxon_statistic = wilcoxon_p_value = math.nan
                    mean_difference = math.nan
                else:
                    differences = paired[method_a] - paired[method_b]
                    t_statistic, t_p_value = stats.ttest_rel(
                        paired[method_a], paired[method_b]
                    )
                    try:
                        wilcoxon_statistic, wilcoxon_p_value = stats.wilcoxon(differences)
                    except ValueError:
                        wilcoxon_statistic, wilcoxon_p_value = 0.0, 1.0
                    mean_difference = float(differences.mean())
                raw_p_values.append(float(t_p_value))
                group_rows.append(
                    {
                        "dataset": dataset,
                        "synthetic_bin_size": int(bin_size),
                        "mix_method": mix_method,
                        "metric": metric,
                        "method_a": method_a,
                        "method_b": method_b,
                        "pairs": len(paired),
                        "mean_a_minus_b": mean_difference,
                        "paired_t_statistic": t_statistic,
                        "paired_t_p_value": t_p_value,
                        "paired_t_bonferroni_p_value": min(1.0, float(t_p_value) * 3)
                        if math.isfinite(float(t_p_value))
                        else math.nan,
                        "wilcoxon_statistic": wilcoxon_statistic,
                        "wilcoxon_p_value": wilcoxon_p_value,
                    }
                )
            finite_wilcoxon = [
                float(row["wilcoxon_p_value"])
                if math.isfinite(float(row["wilcoxon_p_value"]))
                else 1.0
                for row in group_rows
            ]
            for row, adjusted in zip(group_rows, holm_adjust(finite_wilcoxon)):
                row["wilcoxon_holm_p_value"] = adjusted
            test_rows.extend(group_rows)
    pd.DataFrame(test_rows).to_csv(args.out_dir / "paired_tests.csv", index=False)

    completion = {
        "manifest_rows": int(len(manifest)),
        "expected_result_rows": int(len(manifest) * len(ALGORITHMS)),
        "completed_result_rows": int(len(results)),
        "primary_input_files": int(primary["input_name"].nunique()),
        "excluded_tied_input_files": sorted(
            results.loc[results["exclude_primary_5050"], "input_name"].unique().tolist()
        ),
        "missing": missing,
        "all_source_labels_excluded_from_qc": bool(
            (~results["sample_channel_used_for_qc"].astype(bool)).all()
        ),
    }
    (args.out_dir / "completion_report.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("build-manifest")
    manifest_parser.add_argument("--smallcut-dir", type=Path, required=True)
    manifest_parser.add_argument("--largecut-dir", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--preflight", action="store_true")
    manifest_parser.add_argument("--allow-unexpected-counts", action="store_true")

    run_parser = subparsers.add_parser("run-index")
    run_parser.add_argument("--manifest", type=Path, required=True)
    run_parser.add_argument("--index", type=int, required=True)
    run_parser.add_argument("--out-dir", type=Path, required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--flowmop-root", type=Path, required=True)
    run_parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    run_parser.add_argument("--rscript", type=Path, default=Path("Rscript"))
    run_parser.add_argument("--scratch-dir", type=Path, required=True)
    run_parser.add_argument("--timeout", type=float, default=21600)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
    )
    run_parser.add_argument(
        "--mad-smoothing",
        nargs=2,
        type=float,
        default=(0.01, 0.05),
        metavar=("SHORT", "LONG"),
    )
    run_parser.add_argument("--mad-factor", type=int, default=5)

    analyse_parser = subparsers.add_parser("analyse")
    analyse_parser.add_argument("--manifest", type=Path, required=True)
    analyse_parser.add_argument("--results-dir", type=Path, required=True)
    analyse_parser.add_argument("--out-dir", type=Path, required=True)
    analyse_parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-manifest":
        inputs = discover_inputs(
            args.smallcut_dir,
            args.largecut_dir,
            require_expected_counts=not args.allow_unexpected_counts,
        )
        if args.preflight:
            inputs = preflight_subset(inputs)
        write_manifest(inputs, args.output)
        print(f"Wrote {len(inputs)} inputs to {args.output}")
    elif args.command == "run-index":
        run_index(args)
    else:
        analyse(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
