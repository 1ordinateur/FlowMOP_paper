#!/usr/bin/env python3
"""Summarise the full-dataset FlowMOP smoothing grid and identify trade-offs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


SETTING_VALUES = {
    "0_0": (0.0, 0.0),
    "001_002": (0.01, 0.02),
    "001_005": (0.01, 0.05),
    "001_009": (0.01, 0.09),
    "001_02": (0.01, 0.20),
    "002_005": (0.02, 0.05),
    "002_009": (0.02, 0.09),
    "002_02": (0.02, 0.20),
    "002_034": (0.02, 0.34),
    "005_009": (0.05, 0.09),
    "005_02": (0.05, 0.20),
    "005_034": (0.05, 0.34),
    "01_02": (0.10, 0.20),
    "01_034": (0.10, 0.34),
    "01_05": (0.10, 0.50),
    "01_09": (0.10, 0.90),
    "01_10": (0.10, 1.00),
    "02_09": (0.20, 0.90),
    "04_09": (0.40, 0.90),
}


def read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest["exclude_primary_5050"] = (
        manifest["exclude_primary_5050"].astype(str).str.lower() == "true"
    )
    return manifest


def is_pareto_frontier(frame: pd.DataFrame) -> np.ndarray:
    sensitivity = frame["sensitivity_macro"].to_numpy(float)
    specificity = frame["specificity_macro"].to_numpy(float)
    frontier = np.ones(len(frame), dtype=bool)
    for index in range(len(frame)):
        dominated = (
            (sensitivity >= sensitivity[index])
            & (specificity >= specificity[index])
            & ((sensitivity > sensitivity[index]) | (specificity > specificity[index]))
        )
        dominated[index] = False
        frontier[index] = not dominated.any()
    return frontier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mad-factor", type=int, default=5)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    records: list[dict[str, object]] = []
    missing: list[str] = []
    for slug, (short, long) in SETTING_VALUES.items():
        for item in manifest.itertuples(index=False):
            stem = Path(item.input_name).stem
            result_path = (
                args.results_root / slug / "runs" / item.dataset / stem / "flowmop.csv"
            )
            if not result_path.exists():
                missing.append(f"{slug}/{item.dataset}/{stem}")
                continue
            with result_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 1:
                raise ValueError(f"Expected one result row in {result_path}")
            row = rows[0]
            if str(row["sample_channel_used_for_qc"]).upper() != "FALSE":
                raise ValueError(f"Source-label leakage guard failed in {result_path}")
            records.append(
                {
                    "setting": f"{short:g},{long:g}",
                    "slug": slug,
                    "short": short,
                    "long": long,
                    "dataset": item.dataset,
                    "synthetic_bin_size": int(item.synthetic_bin_size),
                    "mix_method": item.mix_method,
                    "input_name": item.input_name,
                    "exclude_primary_5050": bool(item.exclude_primary_5050),
                    "sensitivity": float(row["retained_target_purity"]),
                    "specificity": float(row["removed_nontarget_purity"]),
                }
            )

    if missing:
        raise RuntimeError(f"Missing {len(missing)} results; first entries: {missing[:12]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = pd.DataFrame(records)
    all_results.to_csv(args.output_dir / "all_results.csv", index=False)
    primary = all_results.loc[~all_results["exclude_primary_5050"]].copy()
    primary.to_csv(args.output_dir / "primary_results_excluding_ties.csv", index=False)

    by_group = (
        primary.groupby(
            ["setting", "slug", "short", "long", "dataset", "synthetic_bin_size", "mix_method"],
            as_index=False,
        )[["sensitivity", "specificity"]]
        .mean()
    )
    by_group.to_csv(args.output_dir / "means_by_benchmark_group.csv", index=False)

    macro = (
        by_group.groupby(["setting", "slug", "short", "long"], as_index=False)[
            ["sensitivity", "specificity"]
        ]
        .mean()
        .rename(
            columns={
                "sensitivity": "sensitivity_macro",
                "specificity": "specificity_macro",
            }
        )
    )
    micro = (
        primary.groupby(["setting", "slug", "short", "long"], as_index=False)[
            ["sensitivity", "specificity"]
        ]
        .mean()
        .rename(
            columns={
                "sensitivity": "sensitivity_file_weighted",
                "specificity": "specificity_file_weighted",
            }
        )
    )
    summary = macro.merge(micro, on=["setting", "slug", "short", "long"])
    sensitivity = summary["sensitivity_macro"]
    specificity = summary["specificity_macro"]
    summary["balanced_mean"] = (sensitivity + specificity) / 2
    summary["harmonic_mean"] = 2 * sensitivity * specificity / (sensitivity + specificity)
    summary["distance_to_ideal"] = np.sqrt((1 - sensitivity) ** 2 + (1 - specificity) ** 2)
    summary["pareto_frontier"] = is_pareto_frontier(summary)
    summary = summary.sort_values(
        ["balanced_mean", "distance_to_ideal"], ascending=[False, True]
    ).reset_index(drop=True)
    summary["balanced_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(args.output_dir / "tradeoff_summary.csv", index=False)

    best = summary.iloc[0]
    completion = {
        "settings": len(SETTING_VALUES),
        "manifest_inputs_per_setting": int(len(manifest)),
        "primary_inputs_per_setting": int((~manifest["exclude_primary_5050"]).sum()),
        "result_rows": int(len(all_results)),
        "mad_factor": int(args.mad_factor),
        "all_source_labels_excluded_from_qc": True,
        "primary_selection_rule": "Highest equal-weight macro-average of sensitivity and specificity across the six benchmark groups; Pareto status and secondary harmonic/distance criteria are also reported.",
        "best_equal_weight_setting": best["setting"],
        "best_equal_weight_score": float(best["balanced_mean"]),
        "best_is_pareto_frontier": bool(best["pareto_frontier"]),
    }
    (args.output_dir / "completion_report.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(completion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
