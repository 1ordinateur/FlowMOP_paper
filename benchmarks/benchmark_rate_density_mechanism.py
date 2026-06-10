#!/usr/bin/env python3
"""Benchmark FlowMOP, FlowCut, and PeacoQC on source-linked Time warping.

This mechanism benchmark uses the existing labeled synthetic-combo FCS files
directly. Fluorescence and scatter values are left unchanged. Only the Time
channel is rewritten for the two perturbed variants so that source-linked and
source-independent acquisition-rate changes can be isolated.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fcswrite
import numpy as np
import pandas as pd
from fcsparser import parse


DEFAULT_BASE_FILES = [
    "B1B3_1585_bimix.fcs",
    "C3C1_8020_bimix.fcs",
    "C1C3_1090_bimix.fcs",
    "A05A3_1585_bimix.fcs",
    "B3B1_5050_bimix.fcs",
    "B1B3B05_504010_trimix.fcs",
    "A1A05A3_501535_trimix.fcs",
    "A3A1A05_305020_trimix.fcs",
    "A3A1A05_502525_trimix.fcs",
    "C3C1C05_107020_trimix.fcs",
    "B3B1_8020_segment.fcs",
    "B05B3_1090_segment.fcs",
    "B3B05_9010_segment.fcs",
    "C1C3_2575_segment.fcs",
    "C05C3_1585_segment.fcs",
]
VARIANTS = ("raw", "source_timewarp", "random_timewarp")
KNOWN_MIX_METHODS = {"segment", "bimix", "trimix"}


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    mix_method: str
    proportions: tuple[int, ...]

    @property
    def target_source_ids(self) -> tuple[int, ...]:
        max_value = max(self.proportions)
        return tuple(index + 1 for index, value in enumerate(self.proportions) if value == max_value)


@dataclass(frozen=True)
class TimeDiagnostics:
    time_start: float | None
    time_end: float | None
    time_range: float | None
    time_delta_median: float | None
    time_delta_p99: float | None
    time_delta_max: float | None
    source_transition_count: int
    source_counts: str
    source_multipliers: str
    event_start_index: int
    event_end_index: int


@dataclass(frozen=True)
class RunResult:
    algorithm: str
    variant: str
    base_file: str
    input_fcs: str
    status: str
    mix_method: str = ""
    proportions: str = ""
    target_source_ids: str = ""
    retained_count: int | None = None
    removed_count: int | None = None
    retained_fraction: float | None = None
    removed_fraction: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    balanced_score: float | None = None
    retained_target_count: int | None = None
    removed_nontarget_count: int | None = None
    time_start: float | None = None
    time_end: float | None = None
    time_range: float | None = None
    time_delta_median: float | None = None
    time_delta_p99: float | None = None
    time_delta_max: float | None = None
    source_transition_count: int | None = None
    source_counts: str = ""
    source_multipliers: str = ""
    event_start_index: int | None = None
    event_end_index: int | None = None
    exit_code: int | None = None
    stderr_tail: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/mnt/d/github_remotes/flowmop_data/synthetic_combos_smallcut"),
    )
    parser.add_argument("--base-files", nargs="+", default=DEFAULT_BASE_FILES)
    parser.add_argument("--events", type=int, default=500_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("benchmark_results/rate_density_mechanism/mixed_segment_timewarp_500k"),
    )
    parser.add_argument("--timeout", type=float, default=None, help="Subprocess timeout in seconds. Default: no timeout.")
    parser.add_argument(
        "--rscript",
        type=Path,
        help="Optional Rscript path, e.g. /tmp/flowmop-r/bin/Rscript. Defaults to Rscript on PATH.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=("flowmop", "flowcut", "peacoqc"),
        default=["flowmop", "flowcut", "peacoqc"],
    )
    parser.add_argument("--random-seed", type=int, default=13)
    parser.add_argument("--random-chunk-size", type=int, default=2_000)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_flowmop_exec(root: Path) -> Path:
    for candidate in (root / "flowmop_exec.py", root / "FlowMOP" / "flowmop_exec.py"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError("flowmop_exec.py not found")


def proportions_from_token(token: str) -> tuple[int, ...]:
    if not token.isdigit() or len(token) % 2 != 0:
        raise ValueError(f"invalid proportion token {token!r}")
    return tuple(int(token[index : index + 2]) for index in range(0, len(token), 2))


def parse_dataset_info(path: Path) -> DatasetInfo:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot parse benchmark filename {path.name!r}")
    mix_method = parts[-1].lower()
    if mix_method not in KNOWN_MIX_METHODS:
        raise ValueError(f"unknown mix method in benchmark filename {path.name!r}")
    proportions = proportions_from_token(parts[-2])
    if mix_method in {"segment", "bimix"} and len(proportions) != 2:
        raise ValueError(f"{mix_method} file should have two source proportions: {path.name!r}")
    if mix_method == "trimix" and len(proportions) != 3:
        raise ValueError(f"trimix file should have three source proportions: {path.name!r}")
    return DatasetInfo(path=path, mix_method=mix_method, proportions=proportions)


def load_fcs(path: Path) -> pd.DataFrame:
    _, df = parse(str(path), reformat_meta=True)
    return df


def write_fcs(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fcswrite.write_fcs(
        str(path),
        list(df.columns),
        df.to_numpy(dtype=np.float32, copy=True),
        compat_chn_names=False,
    )


def normalized_column_key(column: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def find_sample_col(columns: Sequence[object]) -> object:
    normalized = {normalized_column_key(column): column for column in columns}
    sample_col = normalized.get("sampleidint")
    if sample_col is not None:
        return sample_col
    candidates = [column for key, column in normalized.items() if "sample" in key and "id" in key]
    if candidates:
        return candidates[0]
    raise ValueError("SampleIDInt/source label channel not found")


def source_ids_from_df(df: pd.DataFrame) -> np.ndarray:
    sample_col = find_sample_col(df.columns)
    return df[sample_col].round().astype(int).to_numpy()


def select_event_window(df: pd.DataFrame, n_events: int, info: DatasetInfo) -> tuple[pd.DataFrame, int, int]:
    if n_events < 1:
        raise ValueError("--events must be positive")
    if len(df) < n_events:
        raise ValueError(f"not enough events: {len(df)} < {n_events}")
    start = 0
    if info.mix_method == "segment":
        labels = source_ids_from_df(df)
        transitions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        if len(transitions):
            center = int(transitions[len(transitions) // 2])
            start = max(0, min(center - n_events // 2, len(df) - n_events))
    end = start + n_events
    return df.iloc[start:end].copy(), start, end


def source_multiplier_map(source_ids: np.ndarray) -> dict[int, float]:
    unique = sorted(int(value) for value in np.unique(source_ids))
    values = np.linspace(1.0, 2.0, num=len(unique), dtype=float)
    return {source: float(multiplier) for source, multiplier in zip(unique, values)}


def format_source_multipliers(multipliers: dict[int, float]) -> str:
    return ";".join(f"{source}:{value:g}" for source, value in sorted(multipliers.items()))


def original_time_values(df: pd.DataFrame) -> np.ndarray:
    if "Time" not in df.columns:
        return np.arange(len(df), dtype=float)
    values = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        finite = values[np.isfinite(values)]
        fallback = float(finite[0]) if len(finite) else 0.0
        values = np.where(np.isfinite(values), values, fallback)
    return values


def positive_time_deltas(time_values: np.ndarray) -> tuple[np.ndarray, float, float]:
    if len(time_values) < 2:
        return np.array([], dtype=float), float(time_values[0]) if len(time_values) else 0.0, 0.0
    raw_deltas = np.diff(time_values).astype(float, copy=False)
    positive = raw_deltas[np.isfinite(raw_deltas) & (raw_deltas > 0)]
    fallback_delta = float(np.median(positive)) if len(positive) else 1.0
    deltas = np.where(np.isfinite(raw_deltas) & (raw_deltas > 0), raw_deltas, fallback_delta)
    raw_duration = float(time_values[-1] - time_values[0])
    if not np.isfinite(raw_duration) or raw_duration <= 0:
        raw_duration = float(deltas.sum())
    return deltas, float(time_values[0]), raw_duration


def rescale_deltas(deltas: np.ndarray, raw_duration: float) -> np.ndarray:
    total = float(deltas.sum())
    if total <= 0 or not np.isfinite(total):
        return np.full_like(deltas, raw_duration / len(deltas) if len(deltas) else 0.0)
    return deltas * (raw_duration / total)


def time_from_deltas(start: float, deltas: np.ndarray, n_events: int) -> np.ndarray:
    if n_events < 1:
        return np.array([], dtype=np.float32)
    new_time = np.empty(n_events, dtype=float)
    new_time[0] = start
    if n_events > 1:
        new_time[1:] = start + np.cumsum(deltas)
    return new_time.astype(np.float32)


def apply_source_timewarp(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    warped = df.copy()
    labels = source_ids_from_df(warped)
    multipliers = source_multiplier_map(labels)
    time_values = original_time_values(warped)
    deltas, start, raw_duration = positive_time_deltas(time_values)
    if len(deltas):
        transition_multipliers = np.array([multipliers[int(value)] for value in labels[1:]], dtype=float)
        deltas = rescale_deltas(deltas * transition_multipliers, raw_duration)
    warped["Time"] = time_from_deltas(start, deltas, len(warped))
    return warped, multipliers


def apply_random_timewarp(df: pd.DataFrame, base_name: str, seed: int, chunk_size: int) -> tuple[pd.DataFrame, dict[int, float]]:
    if chunk_size < 1:
        raise ValueError("--random-chunk-size must be positive")
    warped = df.copy()
    labels = source_ids_from_df(warped)
    multipliers = source_multiplier_map(labels)
    multiplier_values = np.array(list(multipliers.values()), dtype=float)
    time_values = original_time_values(warped)
    deltas, start, raw_duration = positive_time_deltas(time_values)
    if len(deltas):
        file_seed = int(seed + zlib.crc32(base_name.encode("utf-8")))
        rng = np.random.default_rng(file_seed)
        n_chunks = int(math.ceil(len(deltas) / chunk_size))
        chunk_multipliers = rng.choice(multiplier_values, size=n_chunks, replace=True)
        transition_multipliers = chunk_multipliers[np.arange(len(deltas)) // chunk_size]
        deltas = rescale_deltas(deltas * transition_multipliers, raw_duration)
    warped["Time"] = time_from_deltas(start, deltas, len(warped))
    return warped, multipliers


def make_variant(df: pd.DataFrame, variant: str, base_name: str, seed: int, chunk_size: int) -> tuple[pd.DataFrame, dict[int, float]]:
    labels = source_ids_from_df(df)
    multipliers = source_multiplier_map(labels)
    if variant == "raw":
        return df.copy(), multipliers
    if variant == "source_timewarp":
        return apply_source_timewarp(df)
    if variant == "random_timewarp":
        return apply_random_timewarp(df, base_name=base_name, seed=seed, chunk_size=chunk_size)
    raise ValueError(f"unknown variant {variant!r}")


def time_diagnostics(df: pd.DataFrame, multipliers: dict[int, float], event_start: int, event_end: int) -> TimeDiagnostics:
    labels = source_ids_from_df(df)
    counts = pd.Series(labels).value_counts().sort_index()
    source_counts = ";".join(f"{int(source)}:{int(count)}" for source, count in counts.items())
    transitions = int(np.sum(labels[1:] != labels[:-1])) if len(labels) > 1 else 0
    time_values = original_time_values(df)
    deltas = np.diff(time_values) if len(time_values) > 1 else np.array([], dtype=float)
    positive = deltas[np.isfinite(deltas) & (deltas > 0)]
    return TimeDiagnostics(
        time_start=float(time_values[0]) if len(time_values) else None,
        time_end=float(time_values[-1]) if len(time_values) else None,
        time_range=float(time_values[-1] - time_values[0]) if len(time_values) > 1 else None,
        time_delta_median=float(np.median(positive)) if len(positive) else None,
        time_delta_p99=float(np.quantile(positive, 0.99)) if len(positive) else None,
        time_delta_max=float(np.max(positive)) if len(positive) else None,
        source_transition_count=transitions,
        source_counts=source_counts,
        source_multipliers=format_source_multipliers(multipliers),
        event_start_index=event_start,
        event_end_index=event_end,
    )


def write_flowcut_runner(path: Path) -> None:
    path.write_text(
        r'''
library(flowCore)
library(flowCut)
args <- commandArgs(trailingOnly = TRUE)
input_fcs <- args[[1]]
output_fcs <- args[[2]]
ff <- read.FCS(input_fcs, transformation = FALSE)
channels <- which(!grepl("FSC|SSC|Time|Sample", colnames(ff), ignore.case = TRUE))
res <- flowCut(
  ff,
  Channels = channels,
  Plot = "None",
  AllowFlaggedRerun = FALSE,
  Verbose = FALSE,
  PrintToConsole = FALSE
)
write.FCS(res$frame, filename = output_fcs)
'''.strip()
        + "\n",
        encoding="utf-8",
    )


def write_peacoqc_runner(path: Path) -> None:
    path.write_text(
        r'''
library(flowCore)
library(PeacoQC)

args <- commandArgs(trailingOnly = TRUE)
input_fcs <- args[[1]]
output_csv <- args[[2]]

sample_col_index <- function(names) {
  normalized <- tolower(gsub("[^A-Za-z0-9]", "", names))
  exact <- which(normalized == "sampleidint")
  if (length(exact) > 0) {
    return(exact[[1]])
  }
  candidates <- which(grepl("sample", normalized) & grepl("id", normalized))
  if (length(candidates) > 0) {
    return(candidates[[1]])
  }
  stop("SampleIDInt/source label channel not found")
}

extract_good_cells <- function(result) {
  candidates <- list(
    result$GoodCells,
    result$goodCells,
    result$good_cells,
    result$goodcells,
    result$PeacoQC_result$GoodCells,
    result$PeacoQC_result$goodCells
  )
  for (candidate in candidates) {
    if (!is.null(candidate)) {
      return(as.logical(candidate))
    }
  }
  return(NULL)
}

extract_final_frame <- function(result) {
  candidates <- list(
    result$FinalFF,
    result$finalFF,
    result$Final_FF,
    result$ff_final
  )
  for (candidate in candidates) {
    if (!is.null(candidate)) {
      return(candidate)
    }
  }
  return(NULL)
}

ff <- read.FCS(input_fcs, transformation = FALSE)
channel_names <- colnames(ff)
sample_index <- sample_col_index(channel_names)
source_ids <- as.integer(round(exprs(ff)[, sample_index]))
channels <- which(!grepl("FSC|SSC|Time|Sample", channel_names, ignore.case = TRUE))

result <- PeacoQC::PeacoQC(
  ff,
  channels = channels,
  determine_good_cells = "all",
  plot = FALSE,
  save_fcs = FALSE,
  report = FALSE,
  output_directory = tempdir()
)

good_cells <- extract_good_cells(result)
if (!is.null(good_cells) && length(good_cells) == length(source_ids)) {
  write.csv(
    data.frame(mode = "mask", retained = as.integer(good_cells), SampleIDInt = source_ids),
    output_csv,
    row.names = FALSE
  )
} else {
  final_frame <- extract_final_frame(result)
  if (is.null(final_frame)) {
    stop("PeacoQC did not return GoodCells or a final flowFrame")
  }
  final_source <- as.integer(round(exprs(final_frame)[, sample_index]))
  write.csv(
    data.frame(mode = "retained", retained = 1L, SampleIDInt = final_source),
    output_csv,
    row.names = FALSE
  )
}
'''.strip()
        + "\n",
        encoding="utf-8",
    )


def subprocess_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def run_flowmop(flowmop_exec: Path, input_fcs: Path, output_dir: Path, timeout: float | None) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(flowmop_exec),
        str(input_fcs),
        "--output-dir",
        str(output_dir),
        "--fluor-mode",
        "positive_geomeans",
        "--mad-smoothing",
        "0.01",
        "0.05",
        "--skip-debris",
        "--skip-doublets",
        "--disable-remove-zeros",
    ]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        env=subprocess_env(output_dir),
    )
    return proc.returncode, proc.stderr[-2000:]


def run_rscript(
    rscript: Path | None,
    runner: Path,
    input_path: Path,
    output_path: Path,
    timeout: float | None,
) -> tuple[int, str]:
    executable = str(rscript) if rscript is not None else shutil.which("Rscript")
    if executable is None:
        return 127, "Rscript not found on PATH"
    if rscript is not None and not rscript.exists():
        return 127, f"Rscript not found at {rscript}"
    cmd = [executable, str(runner), str(input_path), str(output_path)]
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.returncode, proc.stderr[-2000:]


def read_retained_labels(path: Path) -> np.ndarray:
    _, df = parse(str(path), reformat_meta=True)
    col = find_sample_col(df.columns)
    return df[col].round().astype(int).to_numpy()


def read_flowmop_passed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    _, df = parse(str(path), reformat_meta=True)
    passed_col = next((c for c in df.columns if normalized_column_key(c) == "passedtime"), None)
    if passed_col is None:
        raise ValueError(f"passed_time not found in {path}")
    labels = source_ids_from_df(df)
    return df[passed_col].astype(float).to_numpy() > 0.5, labels


def score_from_counts(
    algorithm: str,
    variant: str,
    info: DatasetInfo,
    input_fcs: Path,
    diagnostics: TimeDiagnostics,
    original_labels: np.ndarray,
    retained_labels: np.ndarray,
) -> RunResult:
    target_set = set(info.target_source_ids)
    original_counts = pd.Series(original_labels).value_counts().to_dict()
    retained_counts = pd.Series(retained_labels).value_counts().to_dict()

    retained_count = int(len(retained_labels))
    removed_count = int(len(original_labels) - retained_count)
    retained_target = int(sum(retained_counts.get(source, 0) for source in target_set))
    original_nontarget = int(sum(count for source, count in original_counts.items() if int(source) not in target_set))
    retained_nontarget = int(sum(count for source, count in retained_counts.items() if int(source) not in target_set))
    removed_nontarget = int(original_nontarget - retained_nontarget)
    sensitivity = retained_target / retained_count if retained_count else math.nan
    specificity = removed_nontarget / removed_count if removed_count else math.nan
    balanced_score = float(np.nanmean([sensitivity, specificity]))

    return ok_result(
        algorithm=algorithm,
        variant=variant,
        info=info,
        input_fcs=input_fcs,
        diagnostics=diagnostics,
        retained_count=retained_count,
        removed_count=removed_count,
        retained_fraction=retained_count / len(original_labels),
        removed_fraction=removed_count / len(original_labels),
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_score=balanced_score,
        retained_target_count=retained_target,
        removed_nontarget_count=removed_nontarget,
    )


def score_mask(
    algorithm: str,
    variant: str,
    info: DatasetInfo,
    input_fcs: Path,
    diagnostics: TimeDiagnostics,
    retained_mask: np.ndarray,
    labels: np.ndarray,
) -> RunResult:
    if len(retained_mask) != len(labels):
        raise ValueError(f"retained mask length {len(retained_mask)} does not match labels length {len(labels)}")
    target_set = set(info.target_source_ids)
    target_mask = np.isin(labels, list(target_set))
    removed_mask = ~retained_mask

    retained_count = int(retained_mask.sum())
    removed_count = int(removed_mask.sum())
    retained_target = int((retained_mask & target_mask).sum())
    removed_nontarget = int((removed_mask & ~target_mask).sum())
    sensitivity = retained_target / retained_count if retained_count else math.nan
    specificity = removed_nontarget / removed_count if removed_count else math.nan
    balanced_score = float(np.nanmean([sensitivity, specificity]))

    return ok_result(
        algorithm=algorithm,
        variant=variant,
        info=info,
        input_fcs=input_fcs,
        diagnostics=diagnostics,
        retained_count=retained_count,
        removed_count=removed_count,
        retained_fraction=retained_count / len(labels),
        removed_fraction=removed_count / len(labels),
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_score=balanced_score,
        retained_target_count=retained_target,
        removed_nontarget_count=removed_nontarget,
    )


def ok_result(
    algorithm: str,
    variant: str,
    info: DatasetInfo,
    input_fcs: Path,
    diagnostics: TimeDiagnostics,
    retained_count: int,
    removed_count: int,
    retained_fraction: float,
    removed_fraction: float,
    sensitivity: float,
    specificity: float,
    balanced_score: float,
    retained_target_count: int,
    removed_nontarget_count: int,
) -> RunResult:
    return RunResult(
        algorithm=algorithm,
        variant=variant,
        base_file=info.path.name,
        input_fcs=str(input_fcs),
        status="ok",
        mix_method=info.mix_method,
        proportions=",".join(str(value) for value in info.proportions),
        target_source_ids=",".join(str(value) for value in info.target_source_ids),
        retained_count=retained_count,
        removed_count=removed_count,
        retained_fraction=retained_fraction,
        removed_fraction=removed_fraction,
        sensitivity=sensitivity,
        specificity=specificity,
        balanced_score=balanced_score,
        retained_target_count=retained_target_count,
        removed_nontarget_count=removed_nontarget_count,
        time_start=diagnostics.time_start,
        time_end=diagnostics.time_end,
        time_range=diagnostics.time_range,
        time_delta_median=diagnostics.time_delta_median,
        time_delta_p99=diagnostics.time_delta_p99,
        time_delta_max=diagnostics.time_delta_max,
        source_transition_count=diagnostics.source_transition_count,
        source_counts=diagnostics.source_counts,
        source_multipliers=diagnostics.source_multipliers,
        event_start_index=diagnostics.event_start_index,
        event_end_index=diagnostics.event_end_index,
    )


def error_result(
    algorithm: str,
    variant: str,
    info: DatasetInfo,
    input_fcs: Path,
    diagnostics: TimeDiagnostics,
    status: str,
    exit_code: int | None = None,
    stderr_tail: str = "",
) -> RunResult:
    return RunResult(
        algorithm=algorithm,
        variant=variant,
        base_file=info.path.name,
        input_fcs=str(input_fcs),
        status=status,
        mix_method=info.mix_method,
        proportions=",".join(str(value) for value in info.proportions),
        target_source_ids=",".join(str(value) for value in info.target_source_ids),
        time_start=diagnostics.time_start,
        time_end=diagnostics.time_end,
        time_range=diagnostics.time_range,
        time_delta_median=diagnostics.time_delta_median,
        time_delta_p99=diagnostics.time_delta_p99,
        time_delta_max=diagnostics.time_delta_max,
        source_transition_count=diagnostics.source_transition_count,
        source_counts=diagnostics.source_counts,
        source_multipliers=diagnostics.source_multipliers,
        event_start_index=diagnostics.event_start_index,
        event_end_index=diagnostics.event_end_index,
        exit_code=exit_code,
        stderr_tail=stderr_tail,
    )


def score_peacoqc(
    variant: str,
    info: DatasetInfo,
    input_fcs: Path,
    diagnostics: TimeDiagnostics,
    output_csv: Path,
    original_labels: np.ndarray,
) -> RunResult:
    df = pd.read_csv(output_csv)
    if df.empty:
        raise ValueError(f"empty PeacoQC output {output_csv}")
    mode = str(df["mode"].iloc[0])
    labels = df["SampleIDInt"].round().astype(int).to_numpy()
    if mode == "mask":
        retained_mask = df["retained"].astype(bool).to_numpy()
        return score_mask("peacoqc", variant, info, input_fcs, diagnostics, retained_mask, labels)
    if mode == "retained":
        return score_from_counts("peacoqc", variant, info, input_fcs, diagnostics, original_labels, labels)
    raise ValueError(f"unknown PeacoQC output mode {mode!r}")


def row_dict(result: RunResult) -> dict[str, object]:
    return {
        "algorithm": result.algorithm,
        "variant": result.variant,
        "base_file": result.base_file,
        "input_fcs": result.input_fcs,
        "status": result.status,
        "mix_method": result.mix_method,
        "proportions": result.proportions,
        "target_source_ids": result.target_source_ids,
        "retained_count": result.retained_count,
        "removed_count": result.removed_count,
        "retained_fraction": result.retained_fraction,
        "removed_fraction": result.removed_fraction,
        "sensitivity": result.sensitivity,
        "specificity": result.specificity,
        "balanced_score": result.balanced_score,
        "retained_target_count": result.retained_target_count,
        "removed_nontarget_count": result.removed_nontarget_count,
        "time_start": result.time_start,
        "time_end": result.time_end,
        "time_range": result.time_range,
        "time_delta_median": result.time_delta_median,
        "time_delta_p99": result.time_delta_p99,
        "time_delta_max": result.time_delta_max,
        "source_transition_count": result.source_transition_count,
        "source_counts": result.source_counts,
        "source_multipliers": result.source_multipliers,
        "event_start_index": result.event_start_index,
        "event_end_index": result.event_end_index,
        "exit_code": result.exit_code,
        "stderr_tail": result.stderr_tail,
    }


def write_outputs(results: list[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(row_dict(RunResult("", "", "", "", "")).keys())
    rows = [row_dict(result) for result in results]
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    if ok_rows:
        ok = pd.DataFrame(ok_rows)
        summary = (
            ok.groupby(["algorithm", "variant"], dropna=False)
            .agg(
                runs=("status", "size"),
                retained_fraction_mean=("retained_fraction", "mean"),
                removed_fraction_mean=("removed_fraction", "mean"),
                sensitivity_mean=("sensitivity", "mean"),
                specificity_mean=("specificity", "mean"),
                balanced_score_mean=("balanced_score", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            columns=[
                "algorithm",
                "variant",
                "runs",
                "retained_fraction_mean",
                "removed_fraction_mean",
                "sensitivity_mean",
                "specificity_mean",
                "balanced_score_mean",
            ]
        )
    summary.to_csv(out_dir / "summary.csv", index=False)

    baseline_rows = {
        (row["algorithm"], row["base_file"]): row
        for row in rows
        if row["status"] == "ok" and row["variant"] == "raw"
    }
    metrics = ["retained_fraction", "removed_fraction", "sensitivity", "specificity", "balanced_score"]
    delta_rows = []
    for row in rows:
        baseline = baseline_rows.get((row["algorithm"], row["base_file"]))
        row_with_delta = dict(row)
        for metric in metrics:
            baseline_value = baseline.get(metric) if baseline is not None else None
            row_value = row.get(metric)
            row_with_delta[f"raw_{metric}"] = baseline_value
            row_with_delta[f"delta_{metric}_vs_raw"] = (
                row_value - baseline_value
                if row_value is not None and baseline_value is not None
                else None
            )
        delta_rows.append(row_with_delta)

    delta_fields = [*fields]
    for metric in metrics:
        delta_fields.extend([f"raw_{metric}", f"delta_{metric}_vs_raw"])
    with (out_dir / "results_with_raw_delta.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=delta_fields)
        writer.writeheader()
        writer.writerows(delta_rows)


def prepare_output_dirs(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    input_dir = out_dir / "inputs"
    flowmop_dir = out_dir / "flowmop_outputs"
    flowcut_dir = out_dir / "flowcut_outputs"
    peacoqc_dir = out_dir / "peacoqc_outputs"
    for directory in (input_dir, flowmop_dir, flowcut_dir, peacoqc_dir):
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
    return input_dir, flowmop_dir, flowcut_dir, peacoqc_dir


def main() -> int:
    args = parse_args()
    root = repo_root()
    flowmop_exec = find_flowmop_exec(root)
    input_dir, flowmop_dir, flowcut_dir, peacoqc_dir = prepare_output_dirs(args.out_dir)

    flowcut_runner = args.out_dir / "run_flowcut.R"
    peacoqc_runner = args.out_dir / "run_peacoqc.R"
    write_flowcut_runner(flowcut_runner)
    write_peacoqc_runner(peacoqc_runner)

    results: list[RunResult] = []
    for base_name in args.base_files:
        base_path = args.dataset_dir / base_name
        info = parse_dataset_info(base_path)
        print(f"Preparing {base_name}")
        base, event_start, event_end = select_event_window(load_fcs(base_path), args.events, info)
        original_labels = source_ids_from_df(base)

        for variant in VARIANTS:
            variant_df, multipliers = make_variant(base, variant, base_name, args.random_seed, args.random_chunk_size)
            diagnostics = time_diagnostics(variant_df, multipliers, event_start, event_end)
            input_fcs = input_dir / f"{Path(base_name).stem}_{variant}.fcs"
            write_fcs(input_fcs, variant_df)
            variant_labels = source_ids_from_df(variant_df)

            if "flowmop" in args.algorithms:
                print(f"Running FlowMOP {input_fcs.name}")
                output_dir = flowmop_dir / input_fcs.stem
                code, stderr_tail = run_flowmop(flowmop_exec, input_fcs, output_dir, args.timeout)
                output_fcs = output_dir / f"flowmop_{input_fcs.name}"
                if code == 0 and output_fcs.exists():
                    try:
                        passed, out_labels = read_flowmop_passed(output_fcs)
                        results.append(score_mask("flowmop", variant, info, input_fcs, diagnostics, passed, out_labels))
                    except Exception as exc:
                        results.append(error_result("flowmop", variant, info, input_fcs, diagnostics, "score_error", code, str(exc)))
                else:
                    results.append(error_result("flowmop", variant, info, input_fcs, diagnostics, "run_error", code, stderr_tail))

            if "flowcut" in args.algorithms:
                print(f"Running FlowCut {input_fcs.name}")
                output_fcs = flowcut_dir / f"{input_fcs.stem}_flowcut.fcs"
                code, stderr_tail = run_rscript(args.rscript, flowcut_runner, input_fcs, output_fcs, args.timeout)
                if code == 0 and output_fcs.exists():
                    try:
                        retained_labels = read_retained_labels(output_fcs)
                        results.append(
                            score_from_counts("flowcut", variant, info, input_fcs, diagnostics, variant_labels, retained_labels)
                        )
                    except Exception as exc:
                        results.append(error_result("flowcut", variant, info, input_fcs, diagnostics, "score_error", code, str(exc)))
                else:
                    results.append(error_result("flowcut", variant, info, input_fcs, diagnostics, "run_error", code, stderr_tail))

            if "peacoqc" in args.algorithms:
                print(f"Running PeacoQC {input_fcs.name}")
                output_csv = peacoqc_dir / f"{input_fcs.stem}_peacoqc.csv"
                code, stderr_tail = run_rscript(args.rscript, peacoqc_runner, input_fcs, output_csv, args.timeout)
                if code == 0 and output_csv.exists():
                    try:
                        results.append(score_peacoqc(variant, info, input_fcs, diagnostics, output_csv, variant_labels))
                    except Exception as exc:
                        results.append(error_result("peacoqc", variant, info, input_fcs, diagnostics, "score_error", code, str(exc)))
                else:
                    results.append(error_result("peacoqc", variant, info, input_fcs, diagnostics, "run_error", code, stderr_tail))

            write_outputs(results, args.out_dir)

    write_outputs(results, args.out_dir)
    print(f"Wrote mechanism benchmark outputs to {args.out_dir.resolve()}")
    return 0 if all(result.status == "ok" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
