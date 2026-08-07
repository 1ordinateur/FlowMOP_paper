#!/usr/bin/env python3
"""Run one Segment file with FlowMOP automatic or fixed acquisition bins."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_fcs", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("setting", help="'auto' or a fixed events-per-bin integer")
    parser.add_argument("--mad-smoothing", type=float, nargs=2, default=(0.1, 0.9))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    flowmop_root = repo_root / "FlowMOP"
    sys.path.insert(0, str(flowmop_root))

    from functions.time_gating import MADTimeGate
    from flowmop_exec import process_file

    original_find = MADTimeGate._find_events_per_bin
    requested = args.setting.lower()
    fixed_events_per_bin = None if requested == "auto" else int(requested)
    if fixed_events_per_bin is not None and fixed_events_per_bin < 1:
        raise ValueError("fixed events per bin must be positive")

    def recorded_find(self: MADTimeGate, arr):
        if fixed_events_per_bin is None:
            value = original_find(self, arr)
        else:
            value = fixed_events_per_bin
        print(f"FLOWMOP_EVENTS_PER_BIN={value}", flush=True)
        return value

    # Benchmark-only override: the production default and source tree are unchanged.
    MADTimeGate._find_events_per_bin = recorded_find
    args.output_dir.mkdir(parents=True, exist_ok=True)
    process_file(
        str(args.input_fcs),
        output_dir=str(args.output_dir),
        fluor_mode="positive_geomeans",
        mad_smoothing=list(args.mad_smoothing),
        skip_debris=True,
        skip_doublets=True,
        min_cells=1000,
        max_bins=600,
        step_val=200,
        mad_factor=4,
        enable_dask=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
