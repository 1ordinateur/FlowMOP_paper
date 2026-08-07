#!/usr/bin/env python3
"""Matched FlowMOP automatic-versus-1,000-event Segment benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmark_peacoqc_segment_localization import (
    DatasetSpec,
    discover_datasets,
    mean_ci,
    score_mask,
)


DEFAULT_OUT_DIR = Path("benchmark_results/flowmop_segment_bin_resolution")
DEFAULT_SETTINGS = ("auto", "1000")


@dataclass(frozen=True)
class RunSpec:
    dataset: DatasetSpec
    setting: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def output_fcs(run: RunSpec) -> Path:
    return run.output_dir / f"flowmop_{run.dataset.path.name}"


def run_one(run: RunSpec, runner: Path, timeout: float, rerun: bool) -> dict[str, object]:
    output = output_fcs(run)
    if output.exists() and not rerun:
        return {"run": run, "status": "reused", "wall_time_s": math.nan}
    run.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(runner),
        str(run.dataset.path),
        str(run.output_dir),
        run.setting,
        "--mad-smoothing",
        "0.1",
        "0.9",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    wall_time = time.perf_counter() - started
    (run.output_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (run.output_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"FlowMOP failed for {run.dataset.path.name}, setting={run.setting}:\n"
            f"{completed.stderr[-4000:]}"
        )
    if not output.exists():
        raise FileNotFoundError(f"FlowMOP did not create {output}")
    match = re.search(r"FLOWMOP_EVENTS_PER_BIN=(\d+)", completed.stdout)
    if match is None:
        raise ValueError(f"events-per-bin record missing for {run.dataset.path.name}")
    return {
        "run": run,
        "status": "ok",
        "wall_time_s": wall_time,
        "events_per_bin_used": int(match.group(1)),
    }


def collect(runs: list[RunSpec], runtime: dict[tuple[str, str], dict[str, object]]) -> pd.DataFrame:
    from fcsparser import parse

    rows: list[dict[str, object]] = []
    for run in runs:
        output = output_fcs(run)
        _, frame = parse(str(output), reformat_meta=True)
        if "passed_time" not in frame or "SampleIDInt" not in frame:
            raise ValueError(f"required columns missing from {output}")
        retained = frame["passed_time"].to_numpy(dtype=float) > 0.5
        labels = frame["SampleIDInt"].round().to_numpy(dtype=int)
        key = (run.dataset.path.name, run.setting)
        info = runtime.get(key, {})
        events_per_bin = info.get("events_per_bin_used")
        if events_per_bin is None:
            log = (run.output_dir / "stdout.log").read_text(encoding="utf-8")
            match = re.search(r"FLOWMOP_EVENTS_PER_BIN=(\d+)", log)
            if match is None:
                raise ValueError(f"events-per-bin record missing from {run.output_dir}")
            events_per_bin = int(match.group(1))
        rows.append(
            {
                "input_fcs": run.dataset.path.name,
                "dataset": run.dataset.dataset,
                "synthetic_bin_size": run.dataset.synthetic_bin_size,
                "proportions": ",".join(map(str, run.dataset.proportions)),
                "target_source_ids": ",".join(map(str, run.dataset.target_ids)),
                "algorithm": "flowmop",
                "requested_setting": run.setting,
                "events_per_bin_used": events_per_bin,
                "mad_smoothing": "0.1,0.9",
                "status": info.get("status", "reused"),
                "wall_time_s": info.get("wall_time_s", math.nan),
                "output_fcs": str(output),
                **score_mask(retained, labels, run.dataset.target_ids),
            }
        )
    return pd.DataFrame(rows)


def paired_test(values: pd.Series) -> dict[str, float]:
    delta = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    mean, ci = mean_ci(pd.Series(delta))
    result: dict[str, float] = {
        "pairs": len(delta),
        "mean_change": mean,
        "ci95_change": ci,
    }
    if len(delta) > 1 and np.any(np.abs(delta) > 0):
        wilcoxon = stats.wilcoxon(delta, alternative="two-sided")
        paired_t = stats.ttest_1samp(delta, 0)
        result.update(
            wilcoxon_statistic=float(wilcoxon.statistic),
            wilcoxon_p_value=float(wilcoxon.pvalue),
            paired_t_statistic=float(paired_t.statistic),
            paired_t_p_value=float(paired_t.pvalue),
        )
    return result


def holm_adjust(values: list[float]) -> list[float]:
    adjusted = [math.nan] * len(values)
    finite = sorted(
        ((index, value) for index, value in enumerate(values) if np.isfinite(value)),
        key=lambda item: item[1],
    )
    running = 0.0
    for rank, (index, value) in enumerate(finite):
        running = max(running, min(1.0, (len(finite) - rank) * value))
        adjusted[index] = running
    return adjusted


def analyse(results: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "retained_target_purity",
        "removed_nontarget_purity",
        "target_retention",
        "nontarget_recall",
        "removed_fraction",
        "target_removal_fraction",
        "removed_target_distance_p95",
    ]
    summary_rows: list[dict[str, object]] = []
    for setting, group in results.groupby("requested_setting", sort=False):
        row: dict[str, object] = {
            "algorithm": "flowmop",
            "requested_setting": setting,
            "files": len(group),
            "events_per_bin_min": int(group["events_per_bin_used"].min()),
            "events_per_bin_median": float(group["events_per_bin_used"].median()),
            "events_per_bin_max": int(group["events_per_bin_used"].max()),
        }
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(group[metric])
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    wide = results.pivot(index=["dataset", "input_fcs"], columns="requested_setting")
    contrast_rows: list[dict[str, object]] = []
    for metric in metrics:
        contrast_rows.append(
            {
                "metric": metric,
                "reference_setting": "auto",
                "comparison_setting": "1000",
                **paired_test(wide[metric]["1000"] - wide[metric]["auto"]),
            }
        )
    adjusted = holm_adjust(
        [float(row.get("wilcoxon_p_value", math.nan)) for row in contrast_rows]
    )
    for row, adjusted_p in zip(contrast_rows, adjusted):
        row["wilcoxon_holm_p_value"] = adjusted_p
    pd.DataFrame(contrast_rows).to_csv(out_dir / "flowmop_setting_contrasts.csv", index=False)

    peacoqc_path = Path(
        "benchmark_results/peacoqc_segment_localization/full_segment/peacoqc_results.csv"
    )
    if peacoqc_path.exists():
        peacoqc = pd.read_csv(peacoqc_path)
        peacoqc["requested_setting"] = peacoqc["requested_setting"].astype(str)
        comparison_rows: list[dict[str, object]] = []
        for setting in DEFAULT_SETTINGS:
            peacoqc_setting = peacoqc[peacoqc["requested_setting"] == setting]
            flowmop_setting = results[results["requested_setting"] == setting]
            paired = flowmop_setting.merge(
                peacoqc_setting,
                on=["dataset", "input_fcs"],
                suffixes=("_flowmop", "_peacoqc"),
                validate="one_to_one",
            )
            setting_rows: list[dict[str, object]] = []
            for metric in metrics:
                setting_rows.append(
                    {
                        "requested_setting": setting,
                        "metric": metric,
                        "reference_algorithm": "flowmop",
                        "comparison_algorithm": "peacoqc",
                        **paired_test(
                            paired[f"{metric}_peacoqc"]
                            - paired[f"{metric}_flowmop"]
                        ),
                    }
                )
            adjusted = holm_adjust(
                [float(row.get("wilcoxon_p_value", math.nan)) for row in setting_rows]
            )
            for row, adjusted_p in zip(setting_rows, adjusted):
                row["wilcoxon_holm_p_value"] = adjusted_p
            comparison_rows.extend(setting_rows)
        pd.DataFrame(comparison_rows).to_csv(
            out_dir / "flowmop_vs_peacoqc_matched_bins.csv", index=False
        )


def main() -> int:
    args = parse_args()
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    settings = [str(value).lower() for value in args.settings]
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError("this focused benchmark requires settings: auto 1000")
    datasets, excluded = discover_datasets(args.limit_files)
    runs = [
        RunSpec(
            dataset=dataset,
            setting=setting,
            output_dir=args.out_dir / "runs" / dataset.dataset / dataset.path.stem / f"epb_{setting}",
        )
        for dataset in datasets
        for setting in settings
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_flowmop_segment_bin_setting.py")
    flowmop_commit = subprocess.run(
        ["git", "-C", "FlowMOP", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    metadata = {
        "design": "Pre-specified automatic versus fixed 1,000-event bins; all other settings fixed.",
        "files": len(datasets),
        "excluded_equal_mixtures": excluded,
        "settings": settings,
        "mad_smoothing": [0.1, 0.9],
        "min_cells": 1000,
        "max_bins_auto": 600,
        "step_auto": 200,
        "mad_factor": 4,
        "overlap_fraction": 0.5,
        "sample_id_use": "Scoring only; FlowMOP excludes channel names containing 'sample'.",
        "flowmop_git_commit": flowmop_commit,
        "jobs": args.jobs,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    runtime: dict[tuple[str, str], dict[str, object]] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(run_one, run, runner, args.timeout, args.rerun): run for run in runs
        }
        completed_count = 0
        for future in as_completed(futures):
            run = futures[future]
            try:
                info = future.result()
                runtime[(run.dataset.path.name, run.setting)] = info
            except Exception as exc:
                failures.append(str(exc))
                print(f"FAILED {run.dataset.path.name} setting={run.setting}: {exc}", flush=True)
            else:
                completed_count += 1
                print(
                    f"Completed {completed_count}/{len(runs)}: "
                    f"{run.dataset.path.name} setting={run.setting}",
                    flush=True,
                )
    if failures:
        raise RuntimeError("\n\n".join(failures))

    results = collect(runs, runtime)
    results.to_csv(args.out_dir / "flowmop_results.csv", index=False)
    analyse(results, args.out_dir)
    print(results.groupby("requested_setting").size().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
