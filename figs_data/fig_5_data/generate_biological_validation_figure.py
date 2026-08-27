#!/usr/bin/env python3
"""Generate PBMC biological-validation Figures 5 and 6 and their audit tables.

The analysis uses one prespecified technical repeat from each of eight
independent sample groups.
Expert-defined gates are imported from the FlowJo workspace with FlowKit. Comparator
and manually cleaned FCS files are mapped back to the row-complete FlowMOP FCS
files using exact, unique, monotonic event matching.

Required packages: flowkit, flowio, numpy, pandas, matplotlib, scipy, and
cairosvg.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import tempfile
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cairosvg
import flowio
import flowkit as fk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp, ttest_rel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SOURCE = REPO.parent / "flowmop_data/pbmc_biological_validation"

WORKSPACE_NAME = "FLOWMOP_CLEANUP_COMPARISON_15-Aug-2026_v2.wsp"
RAW_REL = Path("clean_data_03062026_NR_mad5_metadata_preserved")
METHODS_B = ("Raw", "Expert Manual", "FlowMOP", "PeacoQC", "FlowCut")
METHOD_COLOURS = {
    "Raw": "#5A5A5A",
    "Expert Manual": "#D88700",
    "FlowMOP": "#0072B2",
    "PeacoQC": "#009E73",
    "FlowCut": "#CC79A7",
}
GREY = "#B8B8B8"
FLOWJO_PSEUDOCOLOR = mpl.colors.LinearSegmentedColormap.from_list(
    "flowjo_pseudocolor",
    ("#000066", "#00FFFF", "#00FF00", "#FFFF00", "#FF0000"),
)
SUPPLIED_PDF_SAMPLES = frozenset({"11A", "19A", "22A"})
ANALYSIS_SAMPLES = ("1B", "5B", "10A", "11B", "16A", "19A", "20A", "22A")
TIME_REPRESENTATIVE = "19A"
DEBRIS_REPRESENTATIVE = "19A"
EXPECTED_INPUT_FILES = 36
EXPECTED_ANALYSIS_SAMPLES = 8
EXPECTED_NKT_SAMPLES = 8
ENDPOINTS = ("live_cd45", "b_cells", "t_cells", "nkt_cells")
DISPLAY_ENDPOINTS = ("b_cells", "t_cells", "nkt_cells")
METRICS = ("count", "frequency")
POPULATION_LABELS = {
    "live_cd45": "Live CD45+",
    "b_cells": "B-cell",
    "t_cells": "T-cell",
    "nkt_cells": "NKT-cell",
}
ENDPOINT_LABELS = {endpoint: f"{label} count" for endpoint, label in POPULATION_LABELS.items()}
FREQUENCY_PARENTS = {
    "live_cd45": "live_cells",
    "b_cells": "live_cells",
    "t_cells": "live_cells",
    "nkt_cells": "live_cells",
}

MANUAL_BASE = ("Time gate", "Single Cells", "Single Cells", "Cells")
NONTIME_BASE = ("Single Cells", "Single Cells", "Cells")
BIO_PATHS = {
    "live_cells": ("Live cells",),
    "live_cd45": ("Live cells", "CD45+"),
    "q1_b": ("Live cells", "CD45+", "Q1: CD3- , CD19+"),
    "q3_t": ("Live cells", "CD45+", "Q3: CD3+ , CD19-"),
    "cd19_negative": ("Live cells", "CD45+", "CD19-"),
    "nkt_cells": ("Live cells", "CD45+", "CD19-", "NKT cells"),
}
VALIDATED_GATE_PATHS = {
    "singlet_fsc": ("Single Cells",),
    "singlet_ssc": ("Single Cells", "Single Cells"),
    "debris": ("Single Cells", "Single Cells", "Cells"),
    "live": ("Single Cells", "Single Cells", "Cells", "Live cells"),
    "cd45": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+"),
    "q1_b": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "Q1: CD3- , CD19+"),
    "q2": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "Q2: CD3+ , CD19+"),
    "q3_t": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "Q3: CD3+ , CD19-"),
    "q4": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "Q4: CD3- , CD19-"),
    "cd19_negative": ("Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "CD19-"),
    "nkt_cells": (
        "Single Cells", "Single Cells", "Cells", "Live cells", "CD45+", "CD19-", "NKT cells"
    ),
}


@dataclass
class FCSData:
    path: Path
    events: np.ndarray
    labels: tuple[str, ...]
    text: dict[str, str]

    @property
    def index(self) -> dict[str, int]:
        return {label: i for i, label in enumerate(self.labels)}


@dataclass
class SampleMasks:
    sample: str
    raw: FCSData
    manual_export: np.ndarray
    flowmop_time: np.ndarray
    flowmop_debris: np.ndarray
    flowmop_doublet: np.ndarray
    flowmop_final: np.ndarray
    peacoqc_time: np.ndarray
    flowcut_time: np.ndarray
    manual_time: np.ndarray
    manual_doublet: np.ndarray
    manual_debris: np.ndarray
    manual_non_time: np.ndarray
    manual_all_flowkit: np.ndarray
    biology: dict[str, np.ndarray]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 23,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def natural_sample_key(sample: str) -> tuple[int, str]:
    number = "".join(c for c in sample if c.isdigit())
    letters = "".join(c for c in sample if not c.isdigit())
    return int(number), letters


def read_fcs(path: Path) -> FCSData:
    fd = flowio.FlowData(str(path))
    channel_items = sorted(fd.channels.items(), key=lambda item: int(item[0]))
    labels = tuple(str(info["pnn"]) for _, info in channel_items)
    events = np.asarray(fd.events, dtype=np.float32).reshape(fd.event_count, fd.channel_count)
    return FCSData(path=path, events=events, labels=labels, text=dict(fd.text))


def fcs_event_count(path: Path) -> int:
    return int(flowio.FlowData(str(path), only_text=True).event_count)


def extract_sample_id(path: Path, kind: str) -> str:
    name = path.stem
    if kind == "raw":
        return name.removeprefix("flowmop_")
    if kind in {"flowcut", "peacoqc"}:
        return name.removesuffix(f"_{kind}")
    if kind == "manual":
        return name.removeprefix("export_").split("_", 1)[0]
    return name


def inventory_inputs(source: Path) -> tuple[list[str], dict[str, dict[str, Path]]]:
    globs = {
        "raw": RAW_REL / "flowmop_*.fcs",
        "timepass": RAW_REL / "timepass/*.fcs",
        "debrispass": RAW_REL / "debrispass/*.fcs",
        "doubletpass": RAW_REL / "doubletpass/*.fcs",
        "passfiltered": RAW_REL / "passfiltered/*.fcs",
        "flowcut": Path("flowcut_peacoqc/flowcut/*_flowcut.fcs"),
        "peacoqc": Path("flowcut_peacoqc/peacoqc/*_peacoqc.fcs"),
        "manual": Path("Manually cleaned samples/*.fcs"),
    }
    files: dict[str, dict[str, Path]] = {}
    for kind, pattern in globs.items():
        mapping: dict[str, Path] = {}
        id_kind = kind if kind in {"raw", "flowcut", "peacoqc", "manual"} else "plain"
        for path in source.glob(str(pattern)):
            sample = extract_sample_id(path, id_kind)
            if sample in mapping:
                raise AssertionError(f"Duplicate {kind} file for sample {sample}")
            mapping[sample] = path
        files[kind] = mapping

    raw_ids = set(files["raw"])
    if len(raw_ids) != EXPECTED_INPUT_FILES:
        raise AssertionError(
            f"Expected exactly {EXPECTED_INPUT_FILES} raw input files, found {len(raw_ids)}"
        )
    for kind, mapping in files.items():
        if set(mapping) != raw_ids:
            missing = sorted(raw_ids - set(mapping), key=natural_sample_key)
            extra = sorted(set(mapping) - raw_ids, key=natural_sample_key)
            raise AssertionError(f"{kind}: IDs differ from raw set; missing={missing}, extra={extra}")
    analysis_ids = set(ANALYSIS_SAMPLES)
    if not analysis_ids.issubset(raw_ids):
        missing = sorted(analysis_ids - raw_ids, key=natural_sample_key)
        raise AssertionError(f"Missing selected biological-validation repeats: {missing}")
    if len(analysis_ids) != EXPECTED_ANALYSIS_SAMPLES:
        raise AssertionError(
            f"Expected exactly {EXPECTED_ANALYSIS_SAMPLES} selected sample groups, "
            f"found {len(analysis_ids)}"
        )
    # Retain the complete file inventory for representative rendering and
    # validation. Only ANALYSIS_SAMPLES enter the inferential analysis below.
    return sorted(analysis_ids, key=natural_sample_key), files


def workspace_records(
    workspace: Path,
) -> tuple[ET.ElementTree, dict[str, dict[str, ET.Element]], dict[str, str]]:
    tree = ET.parse(workspace)
    root = tree.getroot()
    sample_list = next(node for node in root.iter() if local_name(node.tag) == "SampleList")
    records: dict[str, dict[str, ET.Element]] = {"flowmop": {}, "flowcut": {}, "peacoqc": {}}
    numeric_ids: dict[str, str] = {}
    for sample_element in sample_list:
        if local_name(sample_element.tag) != "Sample":
            continue
        dataset = next(node for node in sample_element.iter() if local_name(node.tag) == "DataSet")
        sample_node = next(node for node in sample_element.iter() if local_name(node.tag) == "SampleNode")
        uri = dataset.attrib["uri"].lower()
        branch = "peacoqc" if "peacoqc" in uri else "flowcut" if "flowcut" in uri else "flowmop"
        sample = sample_node.attrib["name"].removesuffix(".fcs")
        if sample in records[branch]:
            raise AssertionError(f"Duplicate workspace {branch} branch for {sample}")
        records[branch][sample] = sample_element
        if branch == "flowmop":
            numeric_ids[sample] = dataset.attrib["sampleID"]
    return tree, records, numeric_ids


def direct_population(parent: ET.Element, name: str) -> ET.Element:
    queue = list(parent)
    while queue:
        node = queue.pop(0)
        if local_name(node.tag) == "Population":
            if node.attrib.get("name") == name:
                return node
            continue
        queue.extend(list(node))
    raise KeyError(f"Direct population {name!r} not found below {parent.attrib.get('name', 'sample')!r}")


def population_at_path(sample_element: ET.Element, path: Sequence[str]) -> ET.Element:
    node = next(node for node in sample_element.iter() if local_name(node.tag) == "SampleNode")
    for name in path:
        node = direct_population(node, name)
    return node


def population_count(sample_element: ET.Element, path: Sequence[str]) -> int:
    return int(population_at_path(sample_element, path).attrib["count"])


def gate_signature(population: ET.Element) -> tuple[str, str]:
    gate = next(node for node in population if local_name(node.tag) == "Gate")
    geometry_tags = {
        "RectangleGate", "PolygonGate", "dimension", "fcs-dimension", "vertex", "coordinate"
    }
    useful_attributes = {"name", "value", "min", "max", "eventsInside"}
    pieces: list[object] = []
    for node in gate.iter():
        name = local_name(node.tag)
        if name not in geometry_tags:
            continue
        attrs = tuple(
            sorted(
                (local_name(key), value)
                for key, value in node.attrib.items()
                if local_name(key) in useful_attributes
            )
        )
        pieces.append((name, attrs))
    canonical = json.dumps(pieces, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest(), canonical


def validate_gate_coordinates(
    samples: Sequence[str], records: dict[str, dict[str, ET.Element]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for sample in samples:
        for label, path in VALIDATED_GATE_PATHS.items():
            signatures: dict[str, str] = {}
            canonical: dict[str, str] = {}
            for branch in ("flowmop", "flowcut", "peacoqc"):
                signature, geometry = gate_signature(population_at_path(records[branch][sample], path))
                signatures[branch] = signature
                canonical[branch] = geometry
            identical = len(set(signatures.values())) == 1
            rows.append(
                {
                    "sample": sample,
                    "gate": label,
                    "path": " / ".join(path),
                    "flowmop_sha256": signatures["flowmop"],
                    "flowcut_sha256": signatures["flowcut"],
                    "peacoqc_sha256": signatures["peacoqc"],
                    "coordinate_identical": identical,
                    "canonical_geometry": canonical["flowmop"],
                }
            )
            if not identical:
                failures.append(f"{sample}:{label}")
    if failures:
        raise AssertionError("Non-identical gates across branches: " + ", ".join(failures))
    return rows


def isolated_workspace(
    original_root: ET.Element, sample: str, flowmop_numeric_id: str, destination: Path
) -> None:
    root = copy.deepcopy(original_root)
    sample_list = next(node for node in root.iter() if local_name(node.tag) == "SampleList")
    retained = None
    for sample_element in list(sample_list):
        if local_name(sample_element.tag) != "Sample":
            continue
        dataset = next(node for node in sample_element.iter() if local_name(node.tag) == "DataSet")
        sample_node = next(node for node in sample_element.iter() if local_name(node.tag) == "SampleNode")
        uri = dataset.attrib.get("uri", "").lower()
        if (
            sample_node.attrib.get("name") == f"{sample}.fcs"
            and "flowmop_" in uri
            and dataset.attrib.get("sampleID") == flowmop_numeric_id
        ):
            retained = sample_element
        else:
            sample_list.remove(sample_element)
    if retained is None:
        raise AssertionError(f"Could not isolate FlowMOP workspace branch for {sample}")
    for parent in root.iter():
        for child in list(parent):
            if local_name(child.tag) == "SampleRef" and child.attrib.get("sampleID") != flowmop_numeric_id:
                parent.remove(child)
    ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)


def exact_subsequence_mask(raw: FCSData, filtered: FCSData, label: str) -> np.ndarray:
    # FlowJo can rewrite Time and compensated fluorescence values on export.
    # Pulse-geometry channels remain byte-identical and jointly provide a
    # unique event fingerprint in these files.
    shared = [
        channel
        for channel in ("FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W")
        if channel in raw.index and channel in filtered.index
    ]
    if len(shared) < 3:
        raise AssertionError(f"{label}: fewer than three shared matching channels")
    raw_values = np.ascontiguousarray(raw.events[:, [raw.index[c] for c in shared]], dtype=np.float32)
    filtered_values = np.ascontiguousarray(
        filtered.events[:, [filtered.index[c] for c in shared]], dtype=np.float32
    )
    void_type = np.dtype((np.void, raw_values.dtype.itemsize * raw_values.shape[1]))
    raw_keys = raw_values.view(void_type).ravel()
    filtered_keys = filtered_values.view(void_type).ravel()
    unique_keys, first, counts = np.unique(raw_keys, return_index=True, return_counts=True)
    positions = np.searchsorted(unique_keys, filtered_keys)
    in_range = positions < len(unique_keys)
    equal = np.zeros(len(filtered_keys), dtype=bool)
    equal[in_range] = unique_keys[positions[in_range]] == filtered_keys[in_range]
    if not np.all(equal):
        raise AssertionError(f"{label}: {int((~equal).sum())} filtered events have no exact raw match")
    if not np.all(counts[positions] == 1):
        raise AssertionError(f"{label}: matching key is not unique for every retained event")
    indices = first[positions]
    if np.any(np.diff(indices) <= 0):
        raise AssertionError(f"{label}: retained-event mapping is not strictly monotonic")
    mask = np.zeros(len(raw.events), dtype=bool)
    mask[indices] = True
    if int(mask.sum()) != len(filtered.events):
        raise AssertionError(f"{label}: mapped event count changed")
    return mask


def channel_values(raw: FCSData, channel: str) -> np.ndarray:
    return raw.events[:, raw.index[channel]]


def gate_membership(ws: fk.Workspace, sid: str, name: str, path: Sequence[str]) -> np.ndarray:
    return np.asarray(ws.get_gate_membership(sid, name, gate_path=tuple(path)), dtype=bool)


def processed_column(df: pd.DataFrame, channel: str) -> str:
    matches = [column for column in df.columns if column == channel or column.startswith(channel + " ")]
    if len(matches) != 1:
        raise KeyError(f"Expected one processed column for {channel!r}, found {matches}")
    return matches[0]


def geometric_gate_membership(gate: object, processed: pd.DataFrame) -> np.ndarray:
    """Apply stored gate coordinates without inheriting the workspace parent mask."""
    frame = pd.DataFrame(
        {
            dim.id: processed[processed_column(processed, dim.id)].to_numpy()
            for dim in gate.dimensions
        }
    )
    return np.asarray(gate.apply(frame), dtype=bool)


def write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1.0, adjusted_sorted)
    return adjusted


def p_text(p_value: float) -> str:
    if not np.isfinite(p_value) or p_value >= 0.05:
        return "ns"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def bracket_p_text(p_value: float) -> str:
    if p_value < 0.001:
        return "p < .001"
    return f"p = {p_value:.3f}".replace("0.", ".")


def add_bracket(
    ax: plt.Axes, x1: float, x2: float, y: float, height: float, label: str,
    *, fontsize: float = 10,
) -> None:
    ax.plot((x1, x1, x2, x2), (y, y + height, y + height, y),
            color="#333333", lw=1.25, clip_on=False)
    ax.text((x1 + x2) / 2, y + height * 1.12, label, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", clip_on=False)


def clean_svg(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    # Matplotlib can occasionally leave NUL padding between an embedded PNG
    # payload and the following SVG attribute. NUL is invalid XML and causes
    # CairoSVG to reject an otherwise complete figure.
    path.write_text(
        "\n".join(line.replace("\x00", "").rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def expected_workspace_endpoint_counts(
    sample: str, records: dict[str, dict[str, ET.Element]]
) -> dict[tuple[str, str], int]:
    """Return endpoint counts stored directly in the revised FlowJo hierarchy."""
    result: dict[tuple[str, str], int] = {}
    # Only Live CD45+ retains a one-to-one stored-workspace endpoint. The
    # lineage endpoints below deliberately reuse gate coordinates without the
    # CD45+ parent and therefore cannot be compared with stored hierarchy counts.
    endpoint_paths = {"live_cd45": BIO_PATHS["live_cd45"]}
    for endpoint, bio_path in endpoint_paths.items():
        result[("Expert Manual", endpoint)] = population_count(
            records["flowmop"][sample], MANUAL_BASE + bio_path
        )
        result[("FlowCut", endpoint)] = population_count(
            records["flowcut"][sample], NONTIME_BASE + bio_path
        )
        result[("PeacoQC", endpoint)] = population_count(
            records["peacoqc"][sample], NONTIME_BASE + bio_path
        )
        result[("FlowMOP all stored", endpoint)] = population_count(
            records["flowmop"][sample], ("passed_final subset",) + bio_path
        )
    return result


def workspace_path_count_rows(
    sample: str,
    records: dict[str, dict[str, ET.Element]],
    ws: fk.Workspace,
    sid: str,
) -> list[dict[str, object]]:
    """Audit FlowKit counts against counts stored by FlowJo.

    FlowJo and FlowKit use slightly different polygon boundary and biexponential
    implementations. The stored count remains the reference count, while the
    event-level FlowKit count and its difference are retained explicitly.
    """
    paths = {
        "manual_time": ("Time gate",),
        "manual_singlets": ("Single Cells", "Single Cells"),
        "manual_all_pre_live": MANUAL_BASE,
        **{f"manual_{endpoint}": MANUAL_BASE + path for endpoint, path in BIO_PATHS.items()},
        "flowmop_time_flag": ("FlowMop_passed_time",),
        "flowmop_debris_flag": ("passed_debris subset",),
        "flowmop_doublet_flag": ("passed_doublet subset",),
        "flowmop_final_flag": ("passed_final subset",),
    }
    rows: list[dict[str, object]] = []
    for label, path in paths.items():
        parent_path = ("root",) + tuple(path[:-1])
        observed = int(gate_membership(ws, sid, path[-1], parent_path).sum())
        expected = population_count(records["flowmop"][sample], path)
        rows.append(
            {
                "sample": sample,
                "validation_type": "flowjo_population_count",
                "item": label,
                "path": " / ".join(path),
                "expected_count": expected,
                "observed_count": observed,
                "difference": observed - expected,
                "exact_match": observed == expected,
                "status": "pass" if observed == expected else "recorded_flowkit_flowjo_boundary_difference",
            }
        )
    return rows


def extract_sample_masks(
    sample: str,
    files: dict[str, dict[str, Path]],
    original_root: ET.Element,
    numeric_id: str,
    records: dict[str, dict[str, ET.Element]],
    temp_dir: Path,
) -> tuple[SampleMasks, list[dict[str, object]]]:
    raw = read_fcs(files["raw"][sample])
    isolated = temp_dir / f"{sample}.wsp"
    isolated_workspace(original_root, sample, numeric_id, isolated)
    ws = fk.Workspace(str(isolated), fcs_samples=[str(raw.path)])
    ws.analyze_samples(use_mp=False)
    sid = f"{sample}.fcs"
    if ws.get_sample_ids() != [sid]:
        raise AssertionError(f"{sample}: isolated workspace loaded IDs {ws.get_sample_ids()}")

    flags = {
        "flowmop_time": channel_values(raw, "passed_time") == 1,
        "flowmop_debris": channel_values(raw, "passed_debris") == 1,
        "flowmop_doublet": channel_values(raw, "passed_doublet") == 1,
        "flowmop_final": channel_values(raw, "passed_final") == 1,
    }
    explicit_final = flags["flowmop_time"] & flags["flowmop_debris"] & flags["flowmop_doublet"]
    if not np.array_equal(explicit_final, flags["flowmop_final"]):
        mismatch = int(np.logical_xor(explicit_final, flags["flowmop_final"]).sum())
        raise AssertionError(f"{sample}: explicit FlowMOP intersection differs from passed_final at {mismatch} events")

    validation = workspace_path_count_rows(sample, records, ws, sid)
    module_map = {
        "flowmop_time": "timepass",
        "flowmop_debris": "debrispass",
        "flowmop_doublet": "doubletpass",
        "flowmop_final": "passfiltered",
    }
    metadata_keys = {
        "flowmop_time": "flowmop_time_passed",
        "flowmop_debris": "flowmop_debris_passed",
        "flowmop_doublet": "flowmop_doublet_passed",
        "flowmop_final": "flowmop_final_passed",
    }
    for flag_name, file_kind in module_map.items():
        flag_count = int(flags[flag_name].sum())
        output_count = fcs_event_count(files[file_kind][sample])
        metadata_count = int(raw.text[metadata_keys[flag_name]])
        exact = flag_count == output_count == metadata_count
        validation.append(
            {
                "sample": sample,
                "validation_type": "flowmop_flag_count",
                "item": flag_name,
                "path": str(files[file_kind][sample].relative_to(files[file_kind][sample].parents[2])),
                "expected_count": output_count,
                "observed_count": flag_count,
                "metadata_count": metadata_count,
                "difference": flag_count - output_count,
                "exact_match": exact,
                "status": "pass" if exact else "fail",
            }
        )
        if not exact:
            raise AssertionError(f"{sample}: {flag_name} count does not match FCS output/metadata")

    manual_filtered = read_fcs(files["manual"][sample])
    peacoqc_filtered = read_fcs(files["peacoqc"][sample])
    flowcut_filtered = read_fcs(files["flowcut"][sample])
    manual_export = exact_subsequence_mask(raw, manual_filtered, f"{sample} manual export")
    peacoqc_time = exact_subsequence_mask(raw, peacoqc_filtered, f"{sample} PeacoQC")
    flowcut_time = exact_subsequence_mask(raw, flowcut_filtered, f"{sample} FlowCut")

    expected_manual_pre_live = population_count(records["flowmop"][sample], MANUAL_BASE)
    manual_exact = int(manual_export.sum()) == expected_manual_pre_live == len(manual_filtered.events)
    validation.append(
        {
            "sample": sample,
            "validation_type": "manual_export_count",
            "item": "manual_all_pre_live",
            "path": str(files["manual"][sample].name),
            "expected_count": expected_manual_pre_live,
            "observed_count": int(manual_export.sum()),
            "fcs_count": len(manual_filtered.events),
            "difference": int(manual_export.sum()) - expected_manual_pre_live,
            "exact_match": manual_exact,
            "status": "pass" if manual_exact else "fail",
        }
    )
    if not manual_exact:
        raise AssertionError(f"{sample}: manual export count does not reproduce FlowJo manual all-step count")

    summary_csv = files["flowcut"][sample].parents[1] / "default_qc_summary.csv"
    summary = pd.read_csv(summary_csv)
    id_column = next(c for c in summary.columns if c.lower() in {"sample", "sample_id", "file", "filename"})
    sample_summary = summary[
        summary[id_column].astype(str).str.replace(".fcs", "", regex=False).str.replace("flowmop_", "", regex=False)
        == sample
    ]
    for method, filtered, mask in (
        ("flowcut", flowcut_filtered, flowcut_time),
        ("peacoqc", peacoqc_filtered, peacoqc_time),
    ):
        summary_row = sample_summary[sample_summary["method"].str.lower() == method]
        workspace_root = int(
            next(
                node for node in records[method][sample].iter() if local_name(node.tag) == "SampleNode"
            ).attrib["count"]
        )
        summary_count = int(summary_row.iloc[0]["retained_events"]) if len(summary_row) == 1 else -1
        exact = int(mask.sum()) == len(filtered.events) == workspace_root == summary_count
        validation.append(
            {
                "sample": sample,
                "validation_type": "comparator_root_count",
                "item": method,
                "path": str(filtered.path.name),
                "expected_count": workspace_root,
                "observed_count": int(mask.sum()),
                "fcs_count": len(filtered.events),
                "summary_count": summary_count,
                "difference": int(mask.sum()) - workspace_root,
                "exact_match": exact,
                "status": "pass" if exact else "fail",
                "summary_row_found": len(summary_row) == 1,
            }
        )
        if not exact or len(summary_row) != 1:
            raise AssertionError(f"{sample}: {method} count/summary validation failed")

    manual_time = gate_membership(ws, sid, "Time gate", ("root",))
    manual_doublet = gate_membership(ws, sid, "Single Cells", ("root", "Single Cells"))
    manual_non_time = gate_membership(ws, sid, "Cells", ("root", "Single Cells", "Single Cells"))
    manual_all_flowkit = gate_membership(
        ws, sid, "Cells", ("root", "Time gate", "Single Cells", "Single Cells")
    )
    processed = ws.get_gate_events(sid, source=None).drop(columns=["sample_id"])
    debris_gate = ws.get_gate(sid, "Cells", gate_path=("root", "Single Cells", "Single Cells"))
    debris_frame = pd.DataFrame(
        {
            dim.id: processed[processed_column(processed, dim.id)].to_numpy()
            for dim in debris_gate.dimensions
        }
    )
    manual_debris = np.asarray(debris_gate.apply(debris_frame), dtype=bool)

    biology = {
        label: gate_membership(ws, sid, path[-1], ("root",) + path[:-1])
        for label, path in BIO_PATHS.items()
    }
    # CD45+ is retained as a standalone reference endpoint. The lineage gates
    # reuse the expert coordinates but are evaluated within Live cells without
    # inheriting the CD45+ parent, so preprocessing effects are not hidden by a
    # downstream CD45+ intersection.
    q1_gate = ws.get_gate(
        sid, "Q1: CD3- , CD19+", gate_path=("root", "Live cells", "CD45+")
    )
    q3_gate = ws.get_gate(
        sid, "Q3: CD3+ , CD19-", gate_path=("root", "Live cells", "CD45+")
    )
    cd19_negative_gate = ws.get_gate(
        sid, "CD19-", gate_path=("root", "Live cells", "CD45+")
    )
    nkt_gate = ws.get_gate(
        sid, "NKT cells", gate_path=("root", "Live cells", "CD45+", "CD19-")
    )
    biology["b_cells"] = biology["live_cells"] & geometric_gate_membership(q1_gate, processed)
    biology["t_cells"] = biology["live_cells"] & geometric_gate_membership(q3_gate, processed)
    biology["live_cd19_negative"] = (
        biology["live_cells"] & geometric_gate_membership(cd19_negative_gate, processed)
    )
    biology["nkt_cells"] = (
        biology["live_cd19_negative"] & geometric_gate_membership(nkt_gate, processed)
    )

    validation.append(
        {
            "sample": sample,
            "validation_type": "explicit_final_intersection",
            "item": "passed_time & passed_debris & passed_doublet",
            "path": "raw flag channels",
            "expected_count": int(flags["flowmop_final"].sum()),
            "observed_count": int(explicit_final.sum()),
            "difference": 0,
            "exact_match": True,
            "status": "pass",
        }
    )

    masks = SampleMasks(
        sample=sample,
        raw=raw,
        manual_export=manual_export,
        peacoqc_time=peacoqc_time,
        flowcut_time=flowcut_time,
        manual_time=manual_time,
        manual_doublet=manual_doublet,
        manual_debris=manual_debris,
        manual_non_time=manual_non_time,
        manual_all_flowkit=manual_all_flowkit,
        biology=biology,
        **flags,
    )
    return masks, validation


def calculate_counts(masks: SampleMasks) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def endpoint_row(
        panel: str,
        comparison: str,
        method: str,
        cleaning: np.ndarray,
        endpoint: str,
        raw_count: int,
    ) -> dict[str, object]:
        count = int(np.sum(cleaning & masks.biology[endpoint]))
        frequency_parent = FREQUENCY_PARENTS[endpoint]
        frequency_denominator = int(np.sum(cleaning & masks.biology[frequency_parent]))
        frequency_percent = (
            100.0 * count / frequency_denominator
            if frequency_denominator > 0 else math.nan
        )
        return {
            "panel": panel,
            "sample": masks.sample,
            "endpoint": endpoint,
            "endpoint_label": POPULATION_LABELS[endpoint],
            "comparison": comparison,
            "method": method,
            "count": count,
            "raw_denominator_count": raw_count,
            "frequency_parent": frequency_parent,
            "frequency_denominator_count": frequency_denominator,
            "frequency_percent": frequency_percent,
        }

    explicit_final = masks.flowmop_time & masks.flowmop_debris & masks.flowmop_doublet
    time_masks = {
        "Raw": masks.manual_non_time,
        "Expert Manual": masks.manual_time & masks.manual_non_time,
        "FlowMOP": masks.flowmop_time & masks.manual_non_time,
        "PeacoQC": masks.peacoqc_time & masks.manual_non_time,
        "FlowCut": masks.flowcut_time & masks.manual_non_time,
    }
    for endpoint in ENDPOINTS:
        raw_count = int(np.sum(time_masks["Raw"] & masks.biology[endpoint]))
        for method, cleaning in time_masks.items():
            rows.append(endpoint_row("B", "time", method, cleaning, endpoint, raw_count))

    controlled = {
        "debris": {
            "Raw": masks.manual_time & masks.manual_doublet,
            "Expert Manual": masks.manual_time & masks.manual_doublet & masks.manual_debris,
            "FlowMOP": masks.manual_time & masks.manual_doublet & masks.flowmop_debris,
        },
        "doublet": {
            "Raw": masks.manual_time & masks.manual_debris,
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": masks.manual_time & masks.manual_debris & masks.flowmop_doublet,
        },
        "all steps": {
            "Raw": np.ones(len(masks.raw.events), dtype=bool),
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": explicit_final,
        },
    }
    for endpoint in ENDPOINTS:
        for comparison, method_masks in controlled.items():
            raw_count = int(np.sum(method_masks["Raw"] & masks.biology[endpoint]))
            for method, cleaning in method_masks.items():
                rows.append(
                    endpoint_row("D", comparison, method, cleaning, endpoint, raw_count)
                )
    return rows


def calculate_cleaning_retention(masks: SampleMasks) -> list[dict[str, object]]:
    """Return event-level retention before applying any biological endpoint gate."""
    rows: list[dict[str, object]] = []
    groups = {
        "time": {
            "Raw": masks.manual_non_time,
            "Expert Manual": masks.manual_time & masks.manual_non_time,
            "FlowMOP": masks.flowmop_time & masks.manual_non_time,
            "PeacoQC": masks.peacoqc_time & masks.manual_non_time,
            "FlowCut": masks.flowcut_time & masks.manual_non_time,
        },
        "debris": {
            "Raw": masks.manual_time & masks.manual_doublet,
            "Expert Manual": masks.manual_time & masks.manual_doublet & masks.manual_debris,
            "FlowMOP": masks.manual_time & masks.manual_doublet & masks.flowmop_debris,
        },
        "doublet": {
            "Raw": masks.manual_time & masks.manual_debris,
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": masks.manual_time & masks.manual_debris & masks.flowmop_doublet,
        },
        "all steps": {
            "Raw": np.ones(len(masks.raw.events), dtype=bool),
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": masks.flowmop_time & masks.flowmop_debris & masks.flowmop_doublet,
        },
    }
    for comparison, methods in groups.items():
        raw_count = int(methods["Raw"].sum())
        if raw_count <= 0:
            raise AssertionError(f"{masks.sample}: empty matched Raw mask for {comparison}")
        for method, retained in methods.items():
            count = int(retained.sum())
            rows.append(
                {
                    "sample": masks.sample,
                    "comparison": comparison,
                    "method": method,
                    "retained_event_count": count,
                    "matched_raw_event_count": raw_count,
                    "raw_normalized_event_retention": 1.0 if method == "Raw" else count / raw_count,
                    "removed_event_fraction": 0.0 if method == "Raw" else 1.0 - count / raw_count,
                }
            )
    return rows


def normalize_counts(count_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    keys = sorted({(r["panel"], r["sample"], r["endpoint"], r["comparison"]) for r in count_rows})
    for panel, sample, endpoint, comparison in keys:
        subset = [
            r for r in count_rows
            if (r["panel"], r["sample"], r["endpoint"], r["comparison"])
            == (panel, sample, endpoint, comparison)
        ]
        raw_count = next(int(r["count"]) for r in subset if r["method"] == "Raw")
        if any(int(r["raw_denominator_count"]) != raw_count for r in subset):
            raise AssertionError(f"Inconsistent raw denominator for {(panel, sample, endpoint, comparison)}")
        raw_frequency = next(float(r["frequency_percent"]) for r in subset if r["method"] == "Raw")
        prespecified_nkt_exclusion = sample == "15A" and endpoint == "nkt_cells"
        if not prespecified_nkt_exclusion and (raw_count <= 0 or not np.isfinite(raw_frequency)):
            raise AssertionError(
                f"Invalid raw count or frequency for {(panel, sample, endpoint, comparison)}"
            )
        for row in subset:
            frequency = float(row["frequency_percent"])
            included = (
                not prespecified_nkt_exclusion
                and raw_count > 0
                and np.isfinite(raw_frequency)
                and np.isfinite(frequency)
            )
            ratio = (
                1.0 if row["method"] == "Raw" else
                float(row["count"]) / raw_count if raw_count > 0 else math.nan
            )
            frequency_ratio = (
                1.0 if row["method"] == "Raw" else
                frequency / raw_frequency if raw_frequency > 0 else math.nan
            )
            result.append(
                {
                    **row,
                    "raw_normalized_ratio": ratio,
                    "raw_frequency_percent": raw_frequency,
                    "raw_normalized_frequency_ratio": frequency_ratio,
                    "included": included,
                    "exclusion_reason": (
                        "" if included else
                        "prespecified sample 15A exclusion from all NKT summaries and tests"
                    ),
                }
            )
    return result


def calculate_tests(ratio_rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    tests: list[dict[str, object]] = []

    def values(
        panel: str, endpoint: str, metric: str, comparison: str, method: str
    ) -> tuple[np.ndarray, list[str]]:
        selected = [
            row for row in ratio_rows
            if row["panel"] == panel and row["endpoint"] == endpoint
            and row["comparison"] == comparison and row["method"] == method
            and bool(row["included"])
        ]
        selected.sort(key=lambda row: natural_sample_key(str(row["sample"])))
        value_field = (
            "raw_normalized_ratio" if metric == "count"
            else "raw_normalized_frequency_ratio"
        )
        return (
            np.asarray([float(row[value_field]) for row in selected], dtype=float),
            [str(row["sample"]) for row in selected],
        )

    def test_row(
        panel: str, endpoint: str, metric: str, comparison: str, method: str, reference: str,
        adjustment: str,
    ) -> dict[str, object]:
        observed, observed_ids = values(panel, endpoint, metric, comparison, method)
        reference_values, reference_ids = values(panel, endpoint, metric, comparison, reference)
        if observed_ids != reference_ids:
            raise AssertionError(
                f"Pairing differs for {panel}, {endpoint}, {metric}, {comparison}, {method}"
            )
        if reference == "Raw":
            test = ttest_1samp(observed, popmean=1.0, nan_policy="raise")
        else:
            test = ttest_rel(observed, reference_values, nan_policy="raise")
        return {
            "panel": panel,
            "endpoint": endpoint,
            "endpoint_label": f"{POPULATION_LABELS[endpoint]} {metric}",
            "metric": metric,
            "value_scale": (
                "count / matched ungated input"
                if metric == "count" else "frequency / matched ungated-input frequency"
            ),
            "comparison": comparison,
            "method": method,
            "reference": reference,
            "contrast": f"{method} vs {reference}",
            "n_pairs": len(observed),
            "mean_method_value": float(np.mean(observed)),
            "sd_method_value": float(np.std(observed, ddof=1)),
            "mean_reference_value": float(np.mean(reference_values)),
            "sd_reference_value": float(np.std(reference_values, ddof=1)),
            "mean_paired_difference": float(np.mean(observed - reference_values)),
            "t_statistic": float(test.statistic),
            "p_value_raw": float(test.pvalue),
            "adjustment": adjustment,
        }

    for metric in METRICS:
        for endpoint in ENDPOINTS:
            family = []
            for reference_index, reference in enumerate(METHODS_B):
                for method in METHODS_B[reference_index + 1 :]:
                    family.append(
                        test_row(
                            "B", endpoint, metric, "time", method, reference,
                            "Holm across all ten pairwise tests within endpoint and metric",
                        )
                    )
            adjusted = holm_adjust([float(row["p_value_raw"]) for row in family])
            for row, p_adjusted in zip(family, adjusted):
                row["holm_family"] = f"B|{endpoint}|{metric}|10 pairwise contrasts"
                row["p_value_holm"] = float(p_adjusted)
                row["display_label"] = p_text(float(p_adjusted))
                row["interpretation"] = "difference detected" if p_adjusted < 0.05 else "no difference detected"
                tests.append(row)

    for metric in METRICS:
        for endpoint in ENDPOINTS:
            for comparison in ("debris", "doublet", "all steps"):
                adjustment = "Holm across three tests within endpoint, metric, and gate group"
                family = [
                    test_row("D", endpoint, metric, comparison, "Expert Manual", "Raw", adjustment),
                    test_row("D", endpoint, metric, comparison, "FlowMOP", "Raw", adjustment),
                    test_row("D", endpoint, metric, comparison, "FlowMOP", "Expert Manual", adjustment),
                ]
                adjusted = holm_adjust([float(row["p_value_raw"]) for row in family])
                for row, p_adjusted in zip(family, adjusted):
                    row["holm_family"] = f"D|{endpoint}|{metric}|{comparison}|3 contrasts"
                    row["p_value_holm"] = float(p_adjusted)
                    row["display_label"] = p_text(float(p_adjusted))
                    row["interpretation"] = (
                        "difference detected" if p_adjusted < 0.05 else "no difference detected"
                    )
                    tests.append(row)
    return tests


def select_representative(
    samples: Sequence[str], count_rows: Sequence[dict[str, object]], ratio_rows: Sequence[dict[str, object]]
) -> tuple[str, str, list[dict[str, object]]]:
    vector_keys: list[tuple[str, str, str, str, str]] = []
    # Representative selection remains based on the prespecified raw-normalized
    # count vector; frequencies are displayed and tested as complementary
    # biological-composition outcomes rather than changing the representative.
    for metric in ("count",):
        for endpoint in ENDPOINTS:
            for method in ("Expert Manual", "FlowMOP", "PeacoQC", "FlowCut"):
                vector_keys.append(("B", endpoint, metric, "time", method))
            for comparison in ("debris", "doublet", "all steps"):
                for method in ("Expert Manual", "FlowMOP"):
                    vector_keys.append(("D", endpoint, metric, comparison, method))

    matrix = np.full((len(samples), len(vector_keys)), np.nan, dtype=float)
    eligibility: list[bool] = []
    minimum_live: list[int] = []
    for i, sample in enumerate(samples):
        sample_live = [
            int(row["count"])
            for row in count_rows
            if row["sample"] == sample and row["endpoint"] == "live_cd45"
        ]
        minimum_live.append(min(sample_live))
        for j, (panel, endpoint, metric, comparison, method) in enumerate(vector_keys):
            matches = [
                row for row in ratio_rows
                if row["sample"] == sample
                and row["panel"] == panel
                and row["endpoint"] == endpoint
                and row["comparison"] == comparison
                and row["method"] == method
            ]
            if len(matches) != 1:
                raise AssertionError(f"Representative vector lookup failed for {sample}, {vector_keys[j]}")
            matrix[i, j] = (
                float(
                    matches[0]["raw_normalized_ratio"]
                    if metric == "count" else matches[0]["raw_normalized_frequency_ratio"]
                ) if bool(matches[0]["included"])
                else math.nan
            )
        eligibility.append(minimum_live[-1] >= 5000 and bool(np.isfinite(matrix[i]).all()))

    eligible_matrix = matrix[np.asarray(eligibility)]
    if len(eligible_matrix) == 0:
        raise AssertionError("No sample has complete data and at least 5,000 live CD45+ events in every workflow")
    median = np.median(eligible_matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(eligible_matrix - median), axis=0)
    fallback = np.std(eligible_matrix, axis=0, ddof=1)
    scale = np.where(mad > 0, mad, np.where(fallback > 0, fallback, 1.0))
    distance = np.full(len(samples), np.nan)
    distance[np.asarray(eligibility)] = np.sqrt(
        np.mean(((eligible_matrix - median) / scale) ** 2, axis=1)
    )
    pdf_candidates = np.array(
        [eligible and sample in SUPPLIED_PDF_SAMPLES for sample, eligible in zip(samples, eligibility)],
        dtype=bool,
    )
    if not np.any(pdf_candidates):
        raise AssertionError("No eligible sample is represented in the supplied FlowJo debris PDF")
    candidate_distance = np.where(pdf_candidates, distance, np.nan)
    representative_index = int(np.nanargmin(candidate_distance))
    representative = samples[representative_index]
    debris_direction_match: list[bool] = []
    for sample, eligible in zip(samples, eligibility):
        manual = next(
            float(row["raw_normalized_ratio"])
            for row in ratio_rows
            if row["sample"] == sample and row["panel"] == "D"
            and row["endpoint"] == "live_cd45" and row["comparison"] == "debris"
            and row["method"] == "Expert Manual"
        )
        flowmop = next(
            float(row["raw_normalized_ratio"])
            for row in ratio_rows
            if row["sample"] == sample and row["panel"] == "D"
            and row["endpoint"] == "live_cd45" and row["comparison"] == "debris"
            and row["method"] == "FlowMOP"
        )
        debris_direction_match.append(eligible and np.isfinite(flowmop) and flowmop < manual)
    debris_candidate_distance = np.where(np.asarray(debris_direction_match), distance, np.nan)
    if not np.any(np.isfinite(debris_candidate_distance)):
        debris_representative_index = representative_index
    else:
        debris_representative_index = int(np.nanargmin(debris_candidate_distance))
    debris_representative = samples[debris_representative_index]
    eligible_distances = sorted(float(x) for x in distance[np.isfinite(distance)])
    rows: list[dict[str, object]] = []
    labels = ["|".join(key) for key in vector_keys]
    for i, sample in enumerate(samples):
        rows.append(
            {
                "sample": sample,
                "eligible": eligibility[i],
                "minimum_live_cd45_across_displayed_workflows": minimum_live[i],
                "complete_endpoint_vector": bool(np.isfinite(matrix[i]).all()),
                "available_in_debris_pdf": sample in SUPPLIED_PDF_SAMPLES,
                "debris_direction_match": debris_direction_match[i],
                "robust_standardized_rms_distance": distance[i],
                "eligible_rank": (
                    eligible_distances.index(float(distance[i])) + 1 if np.isfinite(distance[i]) else ""
                ),
                "selected": sample == representative,
                "selected_debris": sample == debris_representative,
                "selection_rule": (
                    "minimum robust-standardized RMS distance from cohort-median endpoint vector among "
                    "eligible samples supplied in the FlowJo debris PDF; scale=1.4826*MAD with SD fallback"
                ),
                "debris_selection_rule": (
                    "minimum full-vector robust-standardized RMS distance among eligible samples "
                    "with FlowMOP debris-controlled Live CD45+ retention below Expert Manual"
                ),
                "vector_labels_json": json.dumps(labels, separators=(",", ":")),
                "endpoint_vector_json": json.dumps(matrix[i].tolist(), separators=(",", ":")),
            }
        )
    return representative, debris_representative, rows


def method_mask_for_time(masks: SampleMasks, method: str) -> np.ndarray:
    return {
        "Expert Manual": masks.manual_time,
        "FlowMOP": masks.flowmop_time,
        "PeacoQC": masks.peacoqc_time,
        "FlowCut": masks.flowcut_time,
    }[method]


def deterministic_subset(mask: np.ndarray, limit: int, seed: int) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def style_flow_axis(
    ax: plt.Axes,
    title: str,
    colour: str = "#222222",
    *,
    show_axis_arrow: bool = False,
    axis_label_fontsize: float = 18,
) -> None:
    """Match Figure 6's square, framelike representative-flow panels."""
    x_label = ax.get_xlabel()
    y_label = ax.get_ylabel()
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, color="#111111", fontweight="bold", pad=5, fontsize=18)
    ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8A8A8A")
        spine.set_linewidth(0.75)
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
        length=0,
    )
    if show_axis_arrow:
        arrow_origin = (-0.07, -0.10)
        for end in ((0.48, -0.10), (-0.07, 0.45)):
            arrow = FancyArrowPatch(
                arrow_origin,
                end,
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#111111",
                clip_on=False,
                zorder=10,
            )
            ax.add_artist(arrow)
        ax.text(
            0.205, -0.19, x_label, transform=ax.transAxes,
            ha="center", va="top", fontsize=axis_label_fontsize,
            fontweight="bold", clip_on=False,
        )
        ax.text(
            -0.16, 0.175, y_label, transform=ax.transAxes,
            ha="center", va="center", rotation=90, fontsize=axis_label_fontsize,
            fontweight="bold", clip_on=False,
        )
    ax.grid(False)


def scatter_masked(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    retained: np.ndarray,
    colour: str,
) -> None:
    retained_idx = indices[retained[indices]]
    if len(retained_idx) == 0:
        return
    xv = x[retained_idx]
    yv = y[retained_idx]
    finite = np.isfinite(xv) & np.isfinite(yv)
    retained_idx = retained_idx[finite]
    xv = xv[finite]
    yv = yv[finite]
    if len(retained_idx) == 0:
        return
    xlo, xhi = np.quantile(xv, [0.001, 0.999])
    ylo, yhi = np.quantile(yv, [0.001, 0.999])
    if xhi <= xlo or yhi <= ylo:
        return
    x_edges = np.linspace(float(xlo), float(xhi), 257)
    y_edges = np.linspace(float(ylo), float(yhi), 257)
    density, _, _ = np.histogram2d(xv, yv, bins=(x_edges, y_edges))
    density = gaussian_filter(density, sigma=0.65, mode="nearest")
    xi = np.clip(np.searchsorted(x_edges, xv, side="right") - 1, 0, density.shape[0] - 1)
    yi = np.clip(np.searchsorted(y_edges, yv, side="right") - 1, 0, density.shape[1] - 1)
    local_density = np.log1p(density[xi, yi])
    positive = local_density[local_density > 0]
    if len(positive):
        low = float(np.quantile(positive, 0.01))
        high = float(np.quantile(positive, 0.997))
        span = max(high - low, 1e-12)
        # Keep sparse events in the deep-blue portion of the FlowJo transfer;
        # lower exponents over-promote them into the visually pale cyan band.
        local_density = np.clip((local_density - low) / span, 0, 1) ** 1.20
    colours = FLOWJO_PSEUDOCOLOR(local_density)
    order = np.argsort(local_density)
    ax.scatter(
        xv[order],
        yv[order],
        s=0.85,
        c=colours[order],
        alpha=1.0,
        linewidths=0,
        rasterized=True,
    )


def scatter_pseudocolor(
    ax: plt.Axes, x: np.ndarray, y: np.ndarray, indices: np.ndarray
) -> None:
    retained = np.zeros(len(x), dtype=bool)
    retained[indices] = True
    scatter_masked(ax, x, y, indices, retained, "#000000")


def rectangle_bounds(gate: object) -> dict[str, tuple[float, float]]:
    return {dim.id: (float(dim.min), float(dim.max)) for dim in gate.dimensions}


def overlay_rectangle(
    ax: plt.Axes, gate: object, x_channel: str, y_channel: str, colour: str = "#222222"
) -> None:
    bounds = rectangle_bounds(gate)
    xmin, xmax = bounds[x_channel]
    ymin, ymax = bounds[y_channel]
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, ec=colour, lw=1.25))


def robust_limits(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[float, float, float, float]:
    xv = x[mask]
    yv = y[mask]
    if len(xv) == 0:
        raise AssertionError("Cannot derive representative axes from an empty mask")
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[finite]
    yv = yv[finite]
    if len(xv) == 0:
        raise AssertionError("Cannot derive representative axes from non-finite values")
    xlo, xhi = np.quantile(xv, [0.002, 0.998])
    ylo, yhi = np.quantile(yv, [0.002, 0.998])
    xpad = max((xhi - xlo) * 0.04, 1e-9)
    ypad = max((yhi - ylo) * 0.04, 1e-9)
    return float(xlo - xpad), float(xhi + xpad), float(ylo - ypad), float(yhi + ypad)


def apply_limits(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])


def endpoint_ratio_array(
    ratio_rows: Sequence[dict[str, object]], panel: str, endpoint: str, metric: str,
    comparison: str, method: str
) -> tuple[np.ndarray, list[str]]:
    rows = [
        row for row in ratio_rows
        if row["panel"] == panel
        and row["endpoint"] == endpoint
        and row["comparison"] == comparison
        and row["method"] == method
        and bool(row["included"])
    ]
    rows.sort(key=lambda row: natural_sample_key(str(row["sample"])))
    value_field = (
        "raw_normalized_ratio" if metric == "count"
        else "raw_normalized_frequency_ratio"
    )
    return np.asarray([float(row[value_field]) for row in rows]), [str(row["sample"]) for row in rows]


def plot_panel_b_axis(
    ax: plt.Axes,
    endpoint: str,
    metric: str,
    ratio_rows: Sequence[dict[str, object]],
    tests: Sequence[dict[str, object]],
) -> None:
    methods = METHODS_B
    values: dict[str, np.ndarray] = {}
    sample_ids: dict[str, list[str]] = {}
    for method in methods:
        values[method], sample_ids[method] = endpoint_ratio_array(
            ratio_rows, "B", endpoint, metric, "time", method
        )
    common = sample_ids[methods[0]]
    if not all(ids == common for ids in sample_ids.values()):
        raise AssertionError(f"Panel B pairing differs among methods for {endpoint}")
    matrix = np.column_stack([values[method] for method in methods])
    x = np.arange(len(methods), dtype=float)
    for row in matrix:
        ax.plot(x, row, color=GREY, lw=1.1, alpha=0.55, zorder=1)
    for j, method in enumerate(methods):
        ax.scatter(x[j] + np.linspace(-0.025, 0.025, len(matrix)), matrix[:, j], s=18,
                   c=METHOD_COLOURS[method], edgecolors="white", linewidths=0.25, alpha=0.82, zorder=2)
        mean = float(np.mean(matrix[:, j]))
        sd = float(np.std(matrix[:, j], ddof=1))
        ax.errorbar(x[j], mean, yerr=sd, fmt="none", ecolor="#111111", lw=1.6, capsize=4, zorder=3)
        ax.scatter(x[j], mean, marker="D", s=52, c=METHOD_COLOURS[method], edgecolors="#111111",
                   linewidths=0.6, zorder=4)
    ax.axhline(1, color="#666666", lw=0.8, ls="--")
    ymax = max(1.08, float(np.max(matrix)))
    ymin = min(0.92, float(np.min(matrix)))
    span = max(0.08, ymax - ymin)
    significant: list[tuple[int, int, float]] = []
    method_index = {method: index for index, method in enumerate(methods)}
    for row in tests:
        if row["panel"] != "B" or row["endpoint"] != endpoint or row["metric"] != metric:
            continue
        p_value = float(row["p_value_holm"])
        if p_value < 0.05 and "Raw" not in (str(row["reference"]), str(row["method"])):
            x1 = method_index[str(row["reference"])]
            x2 = method_index[str(row["method"])]
            significant.append((x1, x2, p_value))
    significant.sort(key=lambda item: (item[1] - item[0], item[0]))
    bracket_start = ymax + 0.09 * span
    bracket_gap = 0.135 * span
    bracket_height = 0.03 * span
    for level, (x1, x2, p_value) in enumerate(significant):
        add_bracket(ax, x1, x2, bracket_start + level * bracket_gap,
                    bracket_height, bracket_p_text(p_value), fontsize=18)
    upper = (
        bracket_start + (len(significant) - 1) * bracket_gap + 0.13 * span
        if significant else ymax + 0.15 * span
    )
    ax.set_ylim(ymin - 0.08 * span, upper)
    ax.set_xlim(-0.35, 4.35)
    ax.set_xticks(x, ["Raw", "Expert Manual", "FlowMOP", "PeacoQC", "FlowCut"])
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(42)
        tick_label.set_ha("right")
        tick_label.set_fontsize(20)
    ax.set_title(
        f"{POPULATION_LABELS[endpoint]} {metric}",
        fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.55)


def plot_panel_d_axis(
    ax: plt.Axes,
    endpoint: str,
    metric: str,
    ratio_rows: Sequence[dict[str, object]],
    tests: Sequence[dict[str, object]],
) -> None:
    """Plot Debris, Doublet, and Combined outcomes with all pairwise brackets."""
    comparisons = ("debris", "doublet", "all steps")
    x_groups = (
        (0.0, 0.7, 1.4),
        (2.7, 3.4, 4.1),
        (5.4, 6.1, 6.8),
    )
    all_values: list[float] = []
    test_lookup = {
        (str(row["comparison"]), str(row["method"]), str(row["reference"])): float(row["p_value_holm"])
        for row in tests
        if row["panel"] == "D" and row["endpoint"] == endpoint and row["metric"] == metric
    }
    for comparison, xs in zip(comparisons, x_groups):
        raw, raw_ids = endpoint_ratio_array(ratio_rows, "D", endpoint, metric, comparison, "Raw")
        manual, manual_ids = endpoint_ratio_array(
            ratio_rows, "D", endpoint, metric, comparison, "Expert Manual"
        )
        flowmop, flowmop_ids = endpoint_ratio_array(
            ratio_rows, "D", endpoint, metric, comparison, "FlowMOP"
        )
        if raw_ids != manual_ids or raw_ids != flowmop_ids:
            raise AssertionError(f"Panel D pairing differs for {endpoint}, {comparison}")
        matrix = np.column_stack((raw, manual, flowmop))
        all_values.extend(matrix.ravel().tolist())
        for row in matrix:
            ax.plot(xs, row, color=GREY, lw=1.0, alpha=0.55, zorder=1)
        for x, vals, method in zip(xs, (raw, manual, flowmop), ("Raw", "Expert Manual", "FlowMOP")):
            ax.scatter(x + np.linspace(-0.025, 0.025, len(vals)), vals, s=17, c=METHOD_COLOURS[method],
                       edgecolors="white", linewidths=0.25, alpha=0.82, zorder=2)
            mean = float(np.mean(vals)); sd = float(np.std(vals, ddof=1))
            ax.errorbar(x, mean, yerr=sd, fmt="none", ecolor="#111111", lw=1.5, capsize=4, zorder=3)
            ax.scatter(x, mean, marker="D", s=48, c=METHOD_COLOURS[method], edgecolors="#111111",
                       linewidths=0.6, zorder=4)
    ymin = min(0.92, min(all_values)); ymax = max(1.08, max(all_values))
    span = max(0.08, ymax - ymin)
    bracket_start = ymax + 0.10 * span
    bracket_gap = 0.16 * span
    bracket_height = 0.035 * span
    max_levels = 0
    for comparison, xs in zip(comparisons, x_groups):
        local = (
            (xs[0], xs[1], test_lookup[(comparison, "Expert Manual", "Raw")]),
            (xs[1], xs[2], test_lookup[(comparison, "FlowMOP", "Expert Manual")]),
            (xs[0], xs[2], test_lookup[(comparison, "FlowMOP", "Raw")]),
        )
        significant = [(x1, x2, p) for x1, x2, p in local if p < 0.05]
        max_levels = max(max_levels, len(significant))
        for level, (x1, x2, p_value) in enumerate(significant):
            add_bracket(ax, x1, x2, bracket_start + level * bracket_gap,
                        bracket_height, bracket_p_text(p_value), fontsize=18)
    ax.axhline(1, color="#666666", lw=0.8, ls="--")
    upper = (
        bracket_start + (max_levels - 1) * bracket_gap + 0.16 * span
        if max_levels else ymax + 0.16 * span
    )
    ax.set_ylim(ymin - 0.08 * span, upper)
    # Leave extra room before the first Raw–Manual bracket so its enlarged
    # p-value label is not clipped at the left edge of the subplot.
    ax.set_xlim(-0.95, 7.15)
    ticks = [x for group in x_groups for x in group]
    labels = ["Raw", "Expert Manual", "FlowMOP"] * 3
    ax.set_xticks(ticks, labels)
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(55)
        tick_label.set_ha("right")
        tick_label.set_fontsize(19)
    for centre, label in zip((0.7, 3.4, 6.1), ("Debris", "Doublet", "Combined")):
        ax.text(centre, -0.39, label, transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=20, fontweight="bold")
    ax.set_title(f"{POPULATION_LABELS[endpoint]} {metric}", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.55)


def plot_combined_axis(
    ax: plt.Axes,
    endpoint: str,
    metric: str,
    ratio_rows: Sequence[dict[str, object]],
    tests: Sequence[dict[str, object]],
) -> None:
    """Plot the matched Raw, expert, and FlowMOP combined-cleaning outcomes."""
    methods = ("Raw", "Expert Manual", "FlowMOP")
    arrays: list[np.ndarray] = []
    identifiers: list[list[str]] = []
    for method in methods:
        values, sample_ids = endpoint_ratio_array(
            ratio_rows, "D", endpoint, metric, "all steps", method
        )
        arrays.append(values)
        identifiers.append(sample_ids)
    if not all(ids == identifiers[0] for ids in identifiers):
        raise AssertionError(f"Combined-cleaning pairing differs for {endpoint}")

    matrix = np.column_stack(arrays)
    x = np.arange(3, dtype=float)
    for row in matrix:
        ax.plot(x, row, color=GREY, lw=1.1, alpha=0.55, zorder=1)
    for j, (method, values) in enumerate(zip(methods, arrays)):
        ax.scatter(
            x[j] + np.linspace(-0.025, 0.025, len(values)), values, s=22,
            c=METHOD_COLOURS[method], edgecolors="white", linewidths=0.25,
            alpha=0.85, zorder=2,
        )
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        ax.errorbar(x[j], mean, yerr=sd, fmt="none", ecolor="#111111", lw=1.7,
                    capsize=4, zorder=3)
        ax.scatter(x[j], mean, marker="D", s=58, c=METHOD_COLOURS[method],
                   edgecolors="#111111", linewidths=0.6, zorder=4)

    p_value = next(
        float(row["p_value_holm"])
        for row in tests
        if row["panel"] == "D"
        and row["comparison"] == "all steps"
        and row["endpoint"] == endpoint
        and row["metric"] == metric
        and row["reference"] == "Expert Manual"
        and row["method"] == "FlowMOP"
    )
    ymin = min(0.92, float(np.min(matrix)))
    ymax = max(1.08, float(np.max(matrix)))
    span = max(0.08, ymax - ymin)
    if p_value < 0.05:
        add_bracket(ax, 1, 2, ymax + 0.10 * span, 0.035 * span,
                    bracket_p_text(p_value), fontsize=18)
        upper = ymax + 0.30 * span
    else:
        upper = ymax + 0.16 * span
    ax.axhline(1, color="#666666", lw=0.8, ls="--")
    ax.set_ylim(ymin - 0.08 * span, upper)
    ax.set_xlim(-0.35, 2.35)
    ax.set_xticks(x, ("Raw", "Expert Manual", "FlowMOP"))
    for tick_label in ax.get_xticklabels():
        tick_label.set_rotation(35)
        tick_label.set_ha("right")
        tick_label.set_fontsize(20)
    ax.set_title(
        f"{POPULATION_LABELS[endpoint]} {metric}",
        fontweight="bold", pad=12,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEEEEE", lw=0.55)


def controlled_cleaning_masks(masks: SampleMasks) -> dict[str, dict[str, np.ndarray]]:
    return {
        "Debris": {
            "Raw": masks.manual_time & masks.manual_doublet,
            "Expert Manual": masks.manual_time & masks.manual_doublet & masks.manual_debris,
            "FlowMOP": masks.manual_time & masks.manual_doublet & masks.flowmop_debris,
        },
        "Doublet": {
            "Raw": masks.manual_time & masks.manual_debris,
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": masks.manual_time & masks.manual_debris & masks.flowmop_doublet,
        },
        "All steps": {
            "Raw": np.ones(len(masks.raw.events), dtype=bool),
            "Expert Manual": masks.manual_time & masks.manual_debris & masks.manual_doublet,
            "FlowMOP": masks.flowmop_time & masks.flowmop_debris & masks.flowmop_doublet,
        },
    }


def overlay_gate_geometry(
    ax: plt.Axes, gate: object, x_channel: str, y_channel: str, colour: str = "#222222"
) -> None:
    if hasattr(gate, "vertices"):
        ax.add_patch(Polygon(np.asarray(gate.vertices), closed=True, fill=False, ec=colour, lw=1.15))
    else:
        overlay_rectangle(ax, gate, x_channel, y_channel, colour)


def biological_plot_spec(
    ws: fk.Workspace, sid: str, processed: pd.DataFrame, masks: SampleMasks, kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, str, str]:
    if kind == "live":
        x_channel, y_channel = "cFluor B548-A", "SSC-A"
        upstream = masks.biology["live_cells"]
        gate = ws.get_gate(sid, "CD45+", gate_path=("root", "Live cells"))
        labels = ("CD45", "SSC-A")
    elif kind == "bt":
        x_channel, y_channel = "cFluor R685-A", "cFluor BYG710-A"
        upstream = masks.biology["live_cells"]
        gate = ws.get_gate(
            sid, "Q1: CD3- , CD19+", gate_path=("root", "Live cells", "CD45+")
        )
        labels = ("CD3", "CD19")
    elif kind == "nkt":
        x_channel, y_channel = "cFluor R685-A", "cFluor BYG750-A"
        upstream = masks.biology["live_cd19_negative"]
        gate = ws.get_gate(
            sid, "NKT cells", gate_path=("root", "Live cells", "CD45+", "CD19-")
        )
        labels = ("CD3", "CD56")
    else:
        raise ValueError(kind)
    x = processed[processed_column(processed, x_channel)].to_numpy()
    y = processed[processed_column(processed, y_channel)].to_numpy()
    return x, y, upstream, gate, labels[0], labels[1]


def overlay_bt_quadrants(ax: plt.Axes, gate: object) -> None:
    dimensions = {dim.id: dim for dim in gate.dimensions}
    cd3_threshold = float(dimensions["cFluor R685-A"].max)
    cd19_threshold = float(dimensions["cFluor BYG710-A"].min)
    ax.axvline(cd3_threshold, color="#222222", lw=1.15)
    ax.axhline(cd19_threshold, color="#222222", lw=1.15)


def draw_biological_plot(
    ax: plt.Axes,
    ws: fk.Workspace,
    sid: str,
    processed: pd.DataFrame,
    masks: SampleMasks,
    cleaning: np.ndarray,
    kind: str,
    title: str,
    seed: int,
    limits: tuple[float, float, float, float],
    show_axis_arrow: bool = False,
    axis_label_fontsize: float = 18,
) -> None:
    """Draw retained events only, using axes fixed by the matched raw input."""
    x, y, upstream, gate, x_label, y_label = biological_plot_spec(
        ws, sid, processed, masks, kind
    )
    base = cleaning & upstream
    indices = deterministic_subset(base, 80000, seed=seed)
    scatter_pseudocolor(ax, x, y, indices)
    if kind == "bt":
        overlay_bt_quadrants(ax, gate)
    else:
        x_channel = "cFluor B548-A" if kind == "live" else "cFluor R685-A"
        y_channel = "SSC-A" if kind == "live" else "cFluor BYG750-A"
        overlay_gate_geometry(ax, gate, x_channel, y_channel)
    apply_limits(ax, limits)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    style_flow_axis(
        ax, title, show_axis_arrow=show_axis_arrow,
        axis_label_fontsize=axis_label_fontsize,
    )
    denominator = int(np.sum(cleaning & masks.biology["live_cells"]))
    if kind == "live":
        upstream_count = int(np.sum(cleaning & masks.biology["live_cells"]))
        live_cd45_count = int(np.sum(cleaning & masks.biology["live_cd45"]))
        percentage = 100 * live_cd45_count / upstream_count if upstream_count else math.nan
        label = f"Live CD45+\n{percentage:.2f}%"
    elif kind == "bt":
        b_pct = 100 * int(np.sum(cleaning & masks.biology["b_cells"])) / denominator if denominator else math.nan
        t_pct = 100 * int(np.sum(cleaning & masks.biology["t_cells"])) / denominator if denominator else math.nan
        label = ""
    else:
        percentage = 100 * int(np.sum(cleaning & masks.biology["nkt_cells"])) / denominator if denominator else math.nan
        label = f"NKT\n{percentage:.2f}%"
    annotation_box = dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.72)
    if kind == "bt":
        ax.text(0.03, 0.97, f"B (Q1)\n{b_pct:.2f}%", transform=ax.transAxes,
                ha="left", va="top", fontsize=13.5, fontweight="bold", linespacing=0.9,
                bbox=annotation_box, zorder=8)
        ax.text(0.97, 0.03, f"T (Q3)\n{t_pct:.2f}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=13.5, fontweight="bold", linespacing=0.9,
                bbox=annotation_box, zorder=8)
    else:
        ax.text(0.97, 0.97, label, transform=ax.transAxes, ha="right", va="top",
                fontsize=13.5, fontweight="bold", linespacing=0.9,
                bbox=annotation_box, zorder=8)


def exact_fsc_threshold(values: np.ndarray, retained: np.ndarray) -> float | None:
    accepted = values[retained]
    rejected = values[~retained]
    if len(accepted) == 0 or len(rejected) == 0:
        return None
    low_pass = float(np.min(accepted))
    high_fail = float(np.max(rejected))
    if high_fail >= low_pass:
        return None
    threshold = (high_fail + low_pass) / 2
    return threshold if np.array_equal(values > threshold, retained) else None


def make_figure(
    time_output_svg: Path,
    time_output_png: Path,
    cleanup_output_svg: Path,
    cleanup_output_png: Path,
    supplement_output_svg: Path,
    supplement_output_png: Path,
    time_representative: str,
    time_masks: SampleMasks,
    time_ws: fk.Workspace,
    cleanup_representative: str,
    cleanup_masks: SampleMasks,
    cleanup_ws: fk.Workspace,
    debris_representative: str,
    debris_masks: SampleMasks,
    debris_ws: fk.Workspace,
    ratio_rows: Sequence[dict[str, object]],
    tests: Sequence[dict[str, object]],
) -> None:
    configure_style()
    sid = f"{time_representative}.fcs"
    masks = time_masks
    ws = time_ws
    processed = ws.get_gate_events(sid, source=None).drop(columns=["sample_id"])
    debris_sid = f"{debris_representative}.fcs"
    debris_processed = debris_ws.get_gate_events(debris_sid, source=None).drop(columns=["sample_id"])
    # Figure 5: time representatives and time-controlled endpoint statistics.
    fig = plt.figure(figsize=(24, 42), facecolor="white", constrained_layout=False)
    outer = fig.add_gridspec(
        2, 1, height_ratios=(22.0, 18.0), hspace=0.20,
        left=0.075, right=0.99, top=0.975, bottom=0.045,
    )
    panel_a_axes: list[plt.Axes] = []
    panel_b_axes: list[plt.Axes] = []
    fixed_axis_groups: dict[str, list[plt.Axes]] = {}

    # Panel A: four controlled time workflows with fixed expert downstream gates.
    grid_a = outer[0].subgridspec(4, 5, hspace=0.30, wspace=0.12)
    time_x = processed[processed_column(processed, "Time")].to_numpy()
    time_y = processed[processed_column(processed, "APC-Fire 810-A")].to_numpy()
    completely_raw = np.ones(len(masks.raw.events), dtype=bool)
    time_pregate = masks.manual_non_time
    time_limits = robust_limits(time_x, time_y, completely_raw)
    time_cleaning: dict[str, np.ndarray] = {}
    time_cleaning["Raw"] = completely_raw
    for method in METHODS_B[1:]:
        time_cleaning[method] = time_pregate & method_mask_for_time(masks, method)
    for j, method in enumerate(METHODS_B):
        ax = fig.add_subplot(grid_a[0, j])
        panel_a_axes.append(ax)
        fixed_axis_groups.setdefault("A|time", []).append(ax)
        cleaning = time_cleaning[method]
        indices = deterministic_subset(cleaning, 100000, seed=501 + j)
        scatter_pseudocolor(ax, time_x, time_y, indices)
        if method == "Expert Manual":
            time_gate = ws.get_gate(sid, "Time gate", gate_path=("root",))
            overlay_rectangle(ax, time_gate, "Time", "APC-Fire 810-A", colour="#111111")
        apply_limits(ax, time_limits)
        denominator = len(cleaning) if method == "Raw" else int(time_pregate.sum())
        pct = 100 * int(cleaning.sum()) / denominator
        ax.text(0.97, 0.97, f"Retained\n{pct:.1f}%", transform=ax.transAxes, ha="right", va="top",
                fontsize=13.5, fontweight="bold", color="#111111", linespacing=0.9,
                bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.72), zorder=8)
        ax.set_xlabel("Time")
        ax.set_ylabel("CD123")
        display_method = method
        style_flow_axis(
            ax,
            display_method,
            show_axis_arrow=(j == 0),
            axis_label_fontsize=18,
        )
    for row_index, kind in enumerate(("live", "bt", "nkt"), start=1):
        x, y, upstream, _gate, _x_label, _y_label = biological_plot_spec(
            ws, sid, processed, masks, kind
        )
        row_limits = robust_limits(x, y, completely_raw & upstream)
        for j, method in enumerate(METHODS_B):
            ax = fig.add_subplot(grid_a[row_index, j])
            panel_a_axes.append(ax)
            fixed_axis_groups.setdefault(f"A|{kind}", []).append(ax)
            draw_biological_plot(
                ax, ws, sid, processed, masks, time_cleaning[method], kind, "",
                seed=600 + row_index * 20 + j, limits=row_limits,
                show_axis_arrow=(j == 0),
                axis_label_fontsize=18,
            )

    # Panels B and C: paired frequency and count endpoints after time gating.
    grid_b = outer[1].subgridspec(2, 3, hspace=0.48, wspace=0.28)
    for row_index, metric in enumerate(("frequency", "count")):
        for j, endpoint in enumerate(DISPLAY_ENDPOINTS):
            ax = fig.add_subplot(grid_b[row_index, j])
            panel_b_axes.append(ax)
            plot_panel_b_axis(ax, endpoint, metric, ratio_rows, tests)
            if j == 0:
                ax.set_ylabel(
                    "Count / Raw Count" if metric == "count"
                    else "Freq / Raw Freq (%)",
                    fontsize=23, fontweight="bold", labelpad=16,
                )
            position = ax.get_position()
            compressed_height = position.height * 0.72
            compressed_y = (
                position.y0
                if row_index == 0
                else position.y1 - compressed_height
            )
            ax.set_position(
                [
                    position.x0,
                    compressed_y,
                    position.width,
                    compressed_height,
                ]
            )

    for label, axes in fixed_axis_groups.items():
        reference_limits = (*axes[0].get_xlim(), *axes[0].get_ylim())
        for axis in axes[1:]:
            observed_limits = (*axis.get_xlim(), *axis.get_ylim())
            if not np.allclose(observed_limits, reference_limits, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"Representative axes differ within {label}: {reference_limits} vs {observed_limits}"
                )

    fig.canvas.draw()
    first_column = [panel_a_axes[i] for i in (0, 5, 10, 15)]
    for axis, label in zip(first_column, ("Time", "Live CD45+ reference", "CD19 × CD3 B/T", "NKT")):
        position = axis.get_position()
        fig.text(position.x0 - 0.048, position.y0 + position.height / 2, label,
                 ha="center", va="center", rotation=90, fontsize=24, fontweight="bold")
    for letter, axes in zip("ABC", (panel_a_axes, panel_b_axes[:3], panel_b_axes[3:])):
        title_y = min(0.995, max(axis.get_position().y1 for axis in axes) + 0.030)
        fig.text(0.012, title_y, f"{letter})", fontsize=23, fontweight="bold", va="top")
    fig.savefig(time_output_svg)
    clean_svg(time_output_svg)
    fig.savefig(time_output_png, dpi=300)
    plt.close(fig)

    # Figure 6 uses the cohort representative; Supplementary Figure S8 uses a
    # separate debris example that visibly exercises the debris-removal gate.
    sid = f"{cleanup_representative}.fcs"
    masks = cleanup_masks
    ws = cleanup_ws
    processed = ws.get_gate_events(sid, source=None).drop(columns=["sample_id"])

    # Supplementary Figure S8: representative debris and doublet gates only.
    fig = plt.figure(figsize=(24, 30), facecolor="white", constrained_layout=False)
    outer = fig.add_gridspec(
        2, 1, height_ratios=(16.0, 12.0), hspace=0.18,
        left=0.075, right=0.99, top=0.99, bottom=0.04,
    )
    panel_c_axes: list[plt.Axes] = []
    panel_c_row_labels: list[tuple[plt.Axes, str]] = []
    fixed_axis_groups = {}
    fsc = processed[processed_column(processed, "FSC-A")].to_numpy()
    ssc = processed[processed_column(processed, "SSC-A")].to_numpy()
    cleaning_masks = controlled_cleaning_masks(masks)
    debris_cleaning_masks = controlled_cleaning_masks(debris_masks)
    debris_fsc = debris_processed[processed_column(debris_processed, "FSC-A")].to_numpy()
    debris_ssc = debris_processed[processed_column(debris_processed, "SSC-A")].to_numpy()

    debris_start = len(panel_c_axes)
    debris_grid = outer[0].subgridspec(
        4, 4, height_ratios=(0.27, 1, 1, 1), hspace=0.12, wspace=0.12
    )
    debris_header = fig.add_subplot(debris_grid[0, :])
    debris_header.axis("off")
    debris_header_title = debris_header.text(
        0.5, 0.72, "Debris", fontsize=23, fontweight="bold",
        ha="center", va="center", transform=debris_header.transAxes,
    )
    panel_c_axes.append(debris_header)
    debris_limits: dict[str, tuple[float, float, float, float]] = {}
    for kind in ("live", "bt", "nkt"):
        bx, by, upstream, _gate, _xl, _yl = biological_plot_spec(
            debris_ws, debris_sid, debris_processed, debris_masks, kind
        )
        debris_limits[kind] = robust_limits(
            bx, by, debris_cleaning_masks["Debris"]["Raw"] & upstream
        )
    debris_scatter_limits = robust_limits(
        debris_fsc, debris_ssc, debris_cleaning_masks["Debris"]["Raw"]
    )
    debris_gate = debris_ws.get_gate(
        debris_sid, "Cells", gate_path=("root", "Single Cells", "Single Cells")
    )
    flowmop_threshold = exact_fsc_threshold(
        channel_values(debris_masks.raw, "FSC-A"), debris_masks.flowmop_debris
    )
    for row_index, method in enumerate(("Raw", "Expert Manual", "FlowMOP"), start=1):
        cleaning = debris_cleaning_masks["Debris"][method]
        ax = fig.add_subplot(debris_grid[row_index, 0])
        panel_c_axes.append(ax)
        fixed_axis_groups.setdefault("C|debris|decision", []).append(ax)
        row_label = "Ungated input" if method == "Raw" else method
        panel_c_row_labels.append((ax, row_label))
        indices = deterministic_subset(cleaning, 100000, seed=980 + row_index)
        scatter_pseudocolor(ax, debris_fsc, debris_ssc, indices)
        if method == "Expert Manual":
            ax.add_patch(Polygon(np.asarray(debris_gate.vertices), closed=True, fill=False,
                                 ec="#111111", lw=1.4))
        if method == "FlowMOP" and flowmop_threshold is not None:
            ax.axvline(flowmop_threshold, color="#111111", lw=1.4)
        apply_limits(ax, debris_scatter_limits)
        ax.set_xlabel("FSC-A")
        ax.set_ylabel("SSC-A")
        retained_pct = 100 * int(cleaning.sum()) / int(debris_cleaning_masks["Debris"]["Raw"].sum())
        ax.text(0.97, 0.97, f"Retained\n{retained_pct:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=13.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.72), zorder=8)
        style_flow_axis(
            ax, "FSC-A × SSC-A" if row_index == 1 else "",
            show_axis_arrow=(row_index == 3),
        )
        for column, kind in enumerate(("live", "bt", "nkt"), start=1):
            target_ax = fig.add_subplot(debris_grid[row_index, column])
            panel_c_axes.append(target_ax)
            fixed_axis_groups.setdefault(f"C|debris|{kind}", []).append(target_ax)
            draw_biological_plot(
                target_ax, debris_ws, debris_sid, debris_processed, debris_masks, cleaning, kind,
                {"live": "Live CD45+ reference", "bt": "CD19 × CD3 quadrants", "nkt": "NKT gate"}[kind]
                if row_index == 1 else "",
                seed=1000 + row_index * 10 + column,
                limits=debris_limits[kind],
                show_axis_arrow=(row_index == 3),
            )

    debris_block_axes = panel_c_axes[debris_start:].copy()
    doublet_start = len(panel_c_axes)
    doublet_grid = outer[1].subgridspec(
        4, 5, height_ratios=(0.27, 1, 1, 1), hspace=0.12, wspace=0.11
    )
    doublet_header = fig.add_subplot(doublet_grid[0, :])
    doublet_header.axis("off")
    doublet_header_title = doublet_header.text(
        0.5, 0.72, "Doublet", fontsize=23, fontweight="bold",
        ha="center", va="center", transform=doublet_header.transAxes,
    )
    panel_c_axes.append(doublet_header)
    doublet_pregate = masks.manual_time & masks.manual_debris
    doublet_specs = (
        ("FSC-H", "FSC-W", ("root",), "FSC-H × FSC-W"),
        ("SSC-H", "SSC-W", ("root", "Single Cells"), "SSC-H × SSC-W"),
    )
    doublet_axis_limits: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for x_channel, y_channel, _gate_path, _title in doublet_specs:
        dx = processed[processed_column(processed, x_channel)].to_numpy()
        dy = processed[processed_column(processed, y_channel)].to_numpy()
        doublet_axis_limits[(x_channel, y_channel)] = robust_limits(dx, dy, doublet_pregate)
    doublet_bio_limits: dict[str, tuple[float, float, float, float]] = {}
    for kind in ("live", "bt", "nkt"):
        bx, by, upstream, _gate, _xl, _yl = biological_plot_spec(ws, sid, processed, masks, kind)
        doublet_bio_limits[kind] = robust_limits(bx, by, doublet_pregate & upstream)
    for row_index, method in enumerate(("Raw", "Expert Manual", "FlowMOP"), start=1):
        cleaning = cleaning_masks["Doublet"][method]
        for panel_index, (x_channel, y_channel, gate_path, title) in enumerate(doublet_specs):
            ax = fig.add_subplot(doublet_grid[row_index, panel_index])
            panel_c_axes.append(ax)
            fixed_axis_groups.setdefault(f"C|doublet|{x_channel}|{y_channel}", []).append(ax)
            if panel_index == 0:
                row_label = "Ungated input" if method == "Raw" else method
                panel_c_row_labels.append((ax, row_label))
            x = processed[processed_column(processed, x_channel)].to_numpy()
            y = processed[processed_column(processed, y_channel)].to_numpy()
            indices = deterministic_subset(cleaning, 100000, seed=1100 + row_index * 10 + panel_index)
            scatter_pseudocolor(ax, x, y, indices)
            # These rectangles are the expert's manual doublet gates. Do not
            # draw them on the ungated or FlowMOP projections, where those
            # gates were not applied.
            if method == "Expert Manual":
                gate = ws.get_gate(sid, "Single Cells", gate_path=gate_path)
                overlay_rectangle(ax, gate, x_channel, y_channel, "#111111")
            apply_limits(ax, doublet_axis_limits[(x_channel, y_channel)])
            ax.set_xlabel(x_channel)
            ax.set_ylabel(y_channel)
            if panel_index == 0:
                retained_pct = 100 * int(cleaning.sum()) / int(cleaning_masks["Doublet"]["Raw"].sum())
                ax.text(0.97, 0.97, f"Retained\n{retained_pct:.1f}%", transform=ax.transAxes,
                        ha="right", va="top", fontsize=13.5, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.72), zorder=8)
            style_flow_axis(
                ax, title if row_index == 1 else "",
                show_axis_arrow=(row_index == 3),
            )
        for column, kind in enumerate(("live", "bt", "nkt"), start=2):
            target_ax = fig.add_subplot(doublet_grid[row_index, column])
            panel_c_axes.append(target_ax)
            fixed_axis_groups.setdefault(f"C|doublet|{kind}", []).append(target_ax)
            draw_biological_plot(
                target_ax, ws, sid, processed, masks, cleaning_masks["Doublet"][method], kind,
                {"live": "Live CD45+ reference", "bt": "CD19 × CD3 quadrants", "nkt": "NKT gate"}[kind]
                if row_index == 1 else "",
                seed=1100 + row_index * 10 + column,
                limits=doublet_bio_limits[kind],
                show_axis_arrow=(row_index == 3),
            )

    doublet_block_axes = panel_c_axes[doublet_start:].copy()

    for label, axes in fixed_axis_groups.items():
        reference_limits = (*axes[0].get_xlim(), *axes[0].get_ylim())
        for axis in axes[1:]:
            observed_limits = (*axis.get_xlim(), *axis.get_ylim())
            if not np.allclose(observed_limits, reference_limits, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"Representative axes differ within {label}: {reference_limits} vs {observed_limits}"
                )

    fig.canvas.draw()
    for axis, label in panel_c_row_labels:
        position = axis.get_position()
        fig.text(
            position.x0 - 0.045, position.y0 + position.height / 2, label,
            ha="center", va="center", rotation=90, fontsize=15, fontweight="bold",
            color="#222222",
        )
    renderer = fig.canvas.get_renderer()
    header_titles = (debris_header_title, doublet_header_title)
    for letter, title_artist in zip("AB", header_titles):
        title_bbox = title_artist.get_window_extent(renderer).transformed(fig.transFigure.inverted())
        title_y = (title_bbox.y0 + title_bbox.y1) / 2
        fig.text(
            0.012, title_y, f"{letter})", fontsize=23, fontweight="bold",
            va="center", ha="left",
        )
    fig.savefig(supplement_output_svg)
    clean_svg(supplement_output_svg)
    fig.savefig(supplement_output_png, dpi=300)
    plt.close(fig)

    # Main Figure 6: the complete automated pathway and its downstream outcomes.
    fig = plt.figure(figsize=(24, 38), facecolor="white", constrained_layout=False)
    outer = fig.add_gridspec(
        2, 1, height_ratios=(13.0, 21.0), hspace=0.17,
        left=0.075, right=0.99, top=0.98, bottom=0.055,
    )
    combined_axes: list[plt.Axes] = []
    combined_row_labels: list[tuple[plt.Axes, str]] = []
    combined_fixed_axes: dict[str, list[plt.Axes]] = {}
    combined_grid = outer[0].subgridspec(3, 6, hspace=0.04, wspace=0.28)
    combined_time_x = processed[processed_column(processed, "Time")].to_numpy()
    combined_time_y = processed[processed_column(processed, "APC-Fire 810-A")].to_numpy()
    combined_time_limits = robust_limits(
        combined_time_x, combined_time_y, cleaning_masks["All steps"]["Raw"]
    )
    combined_scatter_limits = robust_limits(fsc, ssc, cleaning_masks["All steps"]["Raw"])
    combined_doublet_x = processed[processed_column(processed, "FSC-H")].to_numpy()
    combined_doublet_y = processed[processed_column(processed, "FSC-W")].to_numpy()
    combined_doublet_limits = robust_limits(
        combined_doublet_x, combined_doublet_y, cleaning_masks["All steps"]["Raw"]
    )
    combined_bio_limits: dict[str, tuple[float, float, float, float]] = {}
    for kind in ("live", "bt", "nkt"):
        bx, by, upstream, _gate, _xl, _yl = biological_plot_spec(
            ws, sid, processed, masks, kind
        )
        combined_bio_limits[kind] = robust_limits(
            bx, by, cleaning_masks["All steps"]["Raw"] & upstream
        )
    debris_gate = ws.get_gate(sid, "Cells", gate_path=("root", "Single Cells", "Single Cells"))
    time_gate = ws.get_gate(sid, "Time gate", gate_path=("root",))
    doublet_gate = ws.get_gate(sid, "Single Cells", gate_path=("root",))
    for row_index, method in enumerate(("Raw", "Expert Manual", "FlowMOP")):
        cleaning = cleaning_masks["All steps"][method]
        ax = fig.add_subplot(combined_grid[row_index, 0])
        combined_axes.append(ax)
        combined_fixed_axes.setdefault("time", []).append(ax)
        row_label = "Ungated input" if method == "Raw" else method
        combined_row_labels.append((ax, row_label))
        indices = deterministic_subset(cleaning, 100000, seed=1500 + row_index)
        scatter_pseudocolor(ax, combined_time_x, combined_time_y, indices)
        if method == "Expert Manual":
            overlay_rectangle(ax, time_gate, "Time", "APC-Fire 810-A", colour="#111111")
        apply_limits(ax, combined_time_limits)
        ax.set_xlabel("Time")
        ax.set_ylabel("CD123")
        retained_pct = 100 * int(cleaning.sum()) / len(cleaning)
        ax.text(0.97, 0.97, f"Retained\n{retained_pct:.1f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=13.5, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.72),
                zorder=8)
        style_flow_axis(
            ax, "Time" if row_index == 0 else "",
            show_axis_arrow=(row_index == 2), axis_label_fontsize=18,
        )

        debris_ax = fig.add_subplot(combined_grid[row_index, 1])
        combined_axes.append(debris_ax)
        combined_fixed_axes.setdefault("debris", []).append(debris_ax)
        scatter_pseudocolor(debris_ax, fsc, ssc, indices)
        if method == "Expert Manual":
            debris_ax.add_patch(Polygon(np.asarray(debris_gate.vertices), closed=True, fill=False,
                                        ec="#111111", lw=1.2))
        apply_limits(debris_ax, combined_scatter_limits)
        debris_ax.set_xlabel("FSC-A")
        debris_ax.set_ylabel("SSC-A")
        style_flow_axis(
            debris_ax, "Debris" if row_index == 0 else "",
            show_axis_arrow=(row_index == 2), axis_label_fontsize=18,
        )

        doublet_ax = fig.add_subplot(combined_grid[row_index, 2])
        combined_axes.append(doublet_ax)
        combined_fixed_axes.setdefault("doublet", []).append(doublet_ax)
        scatter_pseudocolor(doublet_ax, combined_doublet_x, combined_doublet_y, indices)
        if method == "Expert Manual":
            overlay_rectangle(
                doublet_ax, doublet_gate, "FSC-H", "FSC-W", colour="#111111"
            )
        apply_limits(doublet_ax, combined_doublet_limits)
        doublet_ax.set_xlabel("FSC-H")
        doublet_ax.set_ylabel("FSC-W")
        style_flow_axis(
            doublet_ax, "Doublet" if row_index == 0 else "",
            show_axis_arrow=(row_index == 2), axis_label_fontsize=18,
        )

        for column, kind in enumerate(("live", "bt", "nkt"), start=3):
            target_ax = fig.add_subplot(combined_grid[row_index, column])
            combined_axes.append(target_ax)
            combined_fixed_axes.setdefault(kind, []).append(target_ax)
            draw_biological_plot(
                target_ax, ws, sid, processed, masks, cleaning, kind,
                {"live": "Live CD45+ reference", "bt": "CD19 × CD3 quadrants", "nkt": "NKT gate"}[kind]
                if row_index == 0 else "",
                seed=1520 + row_index * 10 + column,
                limits=combined_bio_limits[kind],
                show_axis_arrow=(row_index == 2),
                axis_label_fontsize=18,
            )

    stats_axes: list[plt.Axes] = []
    stats_grid = outer[1].subgridspec(2, 4, hspace=0.62, wspace=0.24)
    for row_index, metric in enumerate(("frequency", "count")):
        for j, endpoint in enumerate(ENDPOINTS):
            ax = fig.add_subplot(stats_grid[row_index, j])
            stats_axes.append(ax)
            plot_panel_d_axis(ax, endpoint, metric, ratio_rows, tests)
            if j == 0:
                ax.set_ylabel(
                    "Count / Raw Count" if metric == "count"
                    else "Freq / Raw Freq (%)",
                    fontsize=23, fontweight="bold", labelpad=16,
                )
            position = ax.get_position()
            compressed_height = position.height * 0.72
            compressed_y = (
                position.y0
                if row_index == 0
                else position.y1 - compressed_height
            )
            ax.set_position(
                [
                    position.x0,
                    compressed_y,
                    position.width,
                    compressed_height,
                ]
            )

    for label, axes in combined_fixed_axes.items():
        reference_limits = (*axes[0].get_xlim(), *axes[0].get_ylim())
        for axis in axes[1:]:
            observed_limits = (*axis.get_xlim(), *axis.get_ylim())
            if not np.allclose(observed_limits, reference_limits, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"Combined representative axes differ within {label}: "
                    f"{reference_limits} vs {observed_limits}"
                )

    fig.canvas.draw()
    for axis, label in combined_row_labels:
        position = axis.get_position()
        fig.text(position.x0 - 0.045, position.y0 + position.height / 2, label,
                 ha="center", va="center", rotation=90, fontsize=28,
                 fontweight="bold", color="#222222")
    for letter, axes in zip("ABC", (combined_axes, stats_axes[:4], stats_axes[4:])):
        title_y = min(0.995, max(axis.get_position().y1 for axis in axes) + 0.030)
        fig.text(0.012, title_y, f"{letter})", fontsize=23, fontweight="bold", va="top")
    fig.savefig(cleanup_output_svg)
    clean_svg(cleanup_output_svg)
    fig.savefig(cleanup_output_png, dpi=300)
    plt.close(fig)


def validate_nkt_exclusion(ratio_rows: Sequence[dict[str, object]]) -> None:
    excluded = [row for row in ratio_rows if not bool(row["included"])]
    if excluded:
        raise AssertionError(f"Unexpected endpoint exclusions: {excluded[:5]}")
    for panel in ("B", "D"):
        for endpoint in ENDPOINTS:
            n = len(
                {
                    str(row["sample"])
                    for row in ratio_rows
                    if row["panel"] == panel and row["endpoint"] == endpoint and bool(row["included"])
                }
            )
            if n != EXPECTED_ANALYSIS_SAMPLES:
                raise AssertionError(
                    f"Panel {panel} {endpoint}: expected n={EXPECTED_ANALYSIS_SAMPLES}, observed n={n}"
                )


def validate_normalization(
    count_rows: Sequence[dict[str, object]], ratio_rows: Sequence[dict[str, object]]
) -> None:
    count_lookup = {
        (row["panel"], row["sample"], row["endpoint"], row["comparison"], row["method"]): int(row["count"])
        for row in count_rows
    }
    for row in ratio_rows:
        key = (row["panel"], row["sample"], row["endpoint"], row["comparison"], row["method"])
        count = count_lookup[key]
        raw_count = int(row["raw_denominator_count"])
        frequency_denominator = int(row["frequency_denominator_count"])
        if raw_count > 0:
            expected = count / raw_count
            if not math.isclose(float(row["raw_normalized_ratio"]), expected, rel_tol=0, abs_tol=1e-14):
                raise AssertionError(f"Normalization mismatch for {key}")
        if frequency_denominator > 0:
            expected_frequency = 100.0 * count / frequency_denominator
            if not math.isclose(
                float(row["frequency_percent"]), expected_frequency, rel_tol=0, abs_tol=1e-12
            ):
                raise AssertionError(f"Frequency mismatch for {key}")
        raw_frequency = float(row["raw_frequency_percent"])
        frequency = float(row["frequency_percent"])
        if raw_frequency > 0:
            expected_frequency_ratio = frequency / raw_frequency
            if not math.isclose(
                float(row["raw_normalized_frequency_ratio"]),
                expected_frequency_ratio,
                rel_tol=0,
                abs_tol=1e-14,
            ):
                raise AssertionError(f"Frequency normalization mismatch for {key}")
        if row["method"] == "Raw" and raw_count > 0 and float(row["raw_normalized_ratio"]) != 1.0:
            raise AssertionError(f"Raw is not exactly one for {key}")
        if (
            row["method"] == "Raw"
            and raw_frequency > 0
            and float(row["raw_normalized_frequency_ratio"]) != 1.0
        ):
            raise AssertionError(f"Raw frequency is not exactly one for {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=REPO / "figs_data")
    parser.add_argument("--data-dir", type=Path, default=HERE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    workspace = source / WORKSPACE_NAME
    if not workspace.exists():
        raise FileNotFoundError(workspace)

    samples, files = inventory_inputs(source)
    tree, records, numeric_ids = workspace_records(workspace)
    for branch in records:
        if not set(samples).issubset(records[branch]):
            missing = sorted(set(samples) - set(records[branch]), key=natural_sample_key)
            raise AssertionError(
                f"Workspace {branch} branch is missing independent samples: {missing}"
            )

    coordinate_rows = validate_gate_coordinates(samples, records)
    validation_rows: list[dict[str, object]] = [
        {"validation_type": "gate_coordinate_identity", **row, "status": "pass"}
        for row in coordinate_rows
    ]
    count_rows: list[dict[str, object]] = []
    cleaning_retention_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="flowmop_fig5_") as temp_name:
        temp_dir = Path(temp_name)
        for index, sample in enumerate(samples, start=1):
            print(f"[{index:02d}/{len(samples):02d}] analysing {sample}", flush=True)
            masks, sample_validation = extract_sample_masks(
                sample, files, tree.getroot(), numeric_ids[sample], records, temp_dir
            )
            sample_counts = calculate_counts(masks)
            cleaning_retention_rows.extend(calculate_cleaning_retention(masks))
            references = expected_workspace_endpoint_counts(sample, records)
            for row in sample_counts:
                reference_key: tuple[str, str] | None = None
                if row["panel"] == "B" and row["method"] in {"Expert Manual", "FlowCut", "PeacoQC"}:
                    reference_key = (str(row["method"]), str(row["endpoint"]))
                elif (
                    row["panel"] == "D"
                    and row["method"] == "FlowMOP"
                    and row["comparison"] == "all steps"
                ):
                    reference_key = ("FlowMOP all stored", str(row["endpoint"]))
                if reference_key is not None and reference_key in references:
                    reference = references[reference_key]
                    row["workspace_reference_count"] = reference
                    row["flowkit_minus_workspace"] = int(row["count"]) - reference
                else:
                    row["workspace_reference_count"] = ""
                    row["flowkit_minus_workspace"] = ""
            count_rows.extend(sample_counts)
            validation_rows.extend(sample_validation)
            del masks

        ratio_rows = normalize_counts(count_rows)
        validate_normalization(count_rows, ratio_rows)
        validate_nkt_exclusion(ratio_rows)
        tests = calculate_tests(ratio_rows)
        recalculated_tests = calculate_tests(ratio_rows)
        for first, second in zip(tests, recalculated_tests):
            if not math.isclose(float(first["p_value_raw"]), float(second["p_value_raw"]), rel_tol=0, abs_tol=1e-15):
                raise AssertionError("Paired-test recalculation changed a raw p-value")
            if not math.isclose(float(first["p_value_holm"]), float(second["p_value_holm"]), rel_tol=0, abs_tol=1e-15):
                raise AssertionError("Paired-test recalculation changed a Holm-adjusted p-value")

        cleanup_representative, _debris_candidate, selection_rows = select_representative(
            samples, count_rows, ratio_rows
        )
        time_representative = TIME_REPRESENTATIVE
        debris_representative = DEBRIS_REPRESENTATIVE
        if debris_representative not in files["raw"]:
            raise AssertionError(f"Missing debris representative {debris_representative}")
        for row in selection_rows:
            row["selected_debris"] = row["sample"] == debris_representative
            row["debris_selection_rule"] = (
                "prespecified illustrative debris input"
            )
            if row["sample"] == time_representative:
                row["selection"] = "figure_5_representative"
                row["included_in_inferential_analysis"] = True
                row["reason"] = (
                    "prespecified included technical repeat"
                )
        print(
            f"Representative samples: time={time_representative}, "
            f"cleanup={cleanup_representative}, debris={debris_representative}",
            flush=True,
        )
        time_rep_masks, time_rep_validation = extract_sample_masks(
            time_representative, files, tree.getroot(), numeric_ids[time_representative],
            records, temp_dir,
        )
        validation_rows.extend(
            {**row, "validation_repeat": "time_representative_render"}
            for row in time_rep_validation
        )
        time_rep_wsp = temp_dir / f"{time_representative}_time_render.wsp"
        isolated_workspace(
            tree.getroot(), time_representative, numeric_ids[time_representative], time_rep_wsp
        )
        time_rep_ws = fk.Workspace(
            str(time_rep_wsp), fcs_samples=[str(files["raw"][time_representative])]
        )
        time_rep_ws.analyze_samples(use_mp=False)
        cleanup_rep_masks, cleanup_rep_validation = extract_sample_masks(
            cleanup_representative, files, tree.getroot(), numeric_ids[cleanup_representative],
            records, temp_dir,
        )
        validation_rows.extend(
            {**row, "validation_repeat": "cleanup_representative_render"}
            for row in cleanup_rep_validation
        )
        cleanup_rep_wsp = temp_dir / f"{cleanup_representative}_cleanup_render.wsp"
        isolated_workspace(
            tree.getroot(), cleanup_representative, numeric_ids[cleanup_representative],
            cleanup_rep_wsp,
        )
        cleanup_rep_ws = fk.Workspace(
            str(cleanup_rep_wsp), fcs_samples=[str(files["raw"][cleanup_representative])]
        )
        cleanup_rep_ws.analyze_samples(use_mp=False)
        debris_rep_masks, debris_rep_validation = extract_sample_masks(
            debris_representative, files, tree.getroot(), numeric_ids[debris_representative],
            records, temp_dir,
        )
        validation_rows.extend(
            {**row, "validation_repeat": "debris_representative_render"}
            for row in debris_rep_validation
        )
        debris_rep_wsp = temp_dir / f"{debris_representative}_debris_render.wsp"
        isolated_workspace(
            tree.getroot(), debris_representative, numeric_ids[debris_representative],
            debris_rep_wsp,
        )
        debris_rep_ws = fk.Workspace(
            str(debris_rep_wsp), fcs_samples=[str(files["raw"][debris_representative])]
        )
        debris_rep_ws.analyze_samples(use_mp=False)

        make_figure(
            output_dir / "figure_5.svg",
            output_dir / "figure_5.png",
            output_dir / "figure_6.svg",
            output_dir / "figure_6.png",
            output_dir / "Supp_fig_8.svg",
            output_dir / "Supp_fig_8.png",
            time_representative,
            time_rep_masks,
            time_rep_ws,
            cleanup_representative,
            cleanup_rep_masks,
            cleanup_rep_ws,
            debris_representative,
            debris_rep_masks,
            debris_rep_ws,
            ratio_rows,
            tests,
        )

    write_rows(data_dir / "biological_validation_endpoint_counts.csv", count_rows)
    write_rows(data_dir / "biological_validation_raw_normalized_ratios.csv", ratio_rows)
    write_rows(data_dir / "biological_validation_paired_tests.csv", tests)
    write_rows(data_dir / "biological_validation_cleaning_retention.csv", cleaning_retention_rows)
    write_rows(data_dir / "biological_validation_gate_validation.csv", validation_rows)
    write_rows(data_dir / "representative_sample_selection.csv", selection_rows)
    run_metadata = {
        "source_dir": str(source),
        "workspace": WORKSPACE_NAME,
        "sample_count": len(samples),
        "samples": samples,
        "sample_selection": (
            "one prespecified complete PBMC sample from each of eight independent donors: "
            "1B, 5B, 10A, 11B, 16A, 19A, 20A, and 22A"
        ),
        "time_representative_sample": time_representative,
        "cleanup_representative_sample": cleanup_representative,
        "debris_representative_sample": debris_representative,
        "flowkit_version": fk.__version__,
        "explicit_final_mask": "passed_time & passed_debris & passed_doublet",
        "normalization": (
            "each endpoint count and frequency divided by its corresponding matched "
            "ungated-input value; Raw = 1 for both metrics"
        ),
        "frequency_definitions": {
            "live_cd45": "percentage of Live cells before matched-Raw normalization",
            "b_cells": "percentage of Live cells before matched-Raw normalization",
            "t_cells": "percentage of Live cells before matched-Raw normalization",
            "nkt_cells": "percentage of Live cells before matched-Raw normalization",
        },
        "statistical_metrics": ["raw-normalized count", "raw-normalized frequency"],
        "matched_ungated_inputs": {
            "time": "Expert singlet + debris masks, without a time mask",
            "debris": "Expert time + doublet masks, without a debris mask",
            "doublet": "Expert time + debris masks, without a doublet mask",
            "all steps": "no time, debris, or doublet preprocessing mask",
        },
        "nkt_exclusion": "none",
        "time_figure_svg": str(output_dir / "figure_5.svg"),
        "time_figure_png": str(output_dir / "figure_5.png"),
        "cleanup_figure_svg": str(output_dir / "figure_6.svg"),
        "cleanup_figure_png": str(output_dir / "figure_6.png"),
        "module_supplement_svg": str(output_dir / "Supp_fig_8.svg"),
        "module_supplement_png": str(output_dir / "Supp_fig_8.png"),
    }
    (data_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir / 'figure_5.svg'}")
    print(f"Wrote {output_dir / 'figure_5.png'}")
    print(f"Wrote {output_dir / 'figure_6.svg'}")
    print(f"Wrote {output_dir / 'figure_6.png'}")


if __name__ == "__main__":
    main()
