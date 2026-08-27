#!/usr/bin/env python3
"""Generate tumour Figure 7 and its manual-versus-FlowMOP supplement.

The source PDFs and the authoritative population counts live in
``../flowmop_data/tumour_data/Shared_tumour_FlowMOP.zip``.  PyMuPDF is
used only to rasterise the supplied FlowJo PDF plots; statistical panels
and workflow annotations remain vector elements in the SVG outputs.

Required packages: matplotlib, numpy, Pillow, PyMuPDF, scipy.
"""

from __future__ import annotations

import csv
import base64
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import ttest_rel

try:
    import pymupdf
except ImportError:  # PyMuPDF versions before 1.24 exposed only ``fitz``.
    import fitz as pymupdf


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE_ARCHIVE = REPO.parent / "flowmop_data/tumour_data/Shared_tumour_FlowMOP.zip"
FIGURE_SVG = REPO / "figs_data/figure_7.svg"
FIGURE_PNG = REPO / "figs_data/figure_7.png"
PANELS_SVG = HERE / "tumour_validation_panels.svg"
PANELS_PNG = HERE / "tumour_validation_panels.png"
WORKFLOW_SVG = HERE / "tumour_preprocessing_workflow.svg"
WORKFLOW_PNG = HERE / "tumour_preprocessing_workflow.png"
VALUES_CSV = HERE / "tumour_endpoint_values.csv"
TESTS_CSV = HERE / "tumour_paired_t_tests.csv"

SAMPLES = ("Sample 1", "Sample 2", "Sample 3")
METHODS = ("Raw", "Manual", "FlowMOP")
METHOD_COLOURS = {"Raw": "#6f6f6f", "Manual": "#D88700", "FlowMOP": "#0072B2"}

ENDPOINT_LABELS = {
    "live_cd45_count": "Live CD45+ cells",
    "t_cell_frequency_pct_total": "T-cell recovery",
    "b_cell_frequency_pct_total": "B-cell recovery",
}

# Crop rectangles are in FlowJo PDF points (origin at top left after rendering).
# They start and end at the plot frames, excluding FlowJo's numeric axes, tick
# labels, channel-name axes, and sample/population metadata. Protein directions
# are shown once for the complete Panel A grid rather than repeated per plot.
TB_PLOT_RECTS = (
    (110, 243, 205, 338),
    (254, 243, 350, 338),
    # FlowMOP's exported plot has a wider numeric-axis gutter.
    (382, 243, 478, 338),
)


@dataclass(frozen=True)
class SampleCounts:
    raw_total: int
    live: dict[str, int]
    t_cells: dict[str, int]
    b_cells: dict[str, int]
    quadrants: dict[str, dict[str, int]]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_sample_node(root: ET.Element, sample_file: str) -> ET.Element:
    for node in root.iter():
        if local_name(node.tag) == "SampleNode" and node.attrib.get("name") == sample_file:
            return node
    raise KeyError(f"SampleNode not found for {sample_file}")


def direct_population(parent: ET.Element, name: str) -> ET.Element:
    """Find a named Population below a parent without crossing another Population."""
    queue = list(parent)
    while queue:
        node = queue.pop(0)
        if local_name(node.tag) == "Population":
            if node.attrib.get("name") == name:
                return node
            continue
        queue.extend(list(node))
    raise KeyError(f"Direct population {name!r} not found")


def direct_population_prefix(parent: ET.Element, prefix: str) -> ET.Element:
    """Find a direct Population whose name starts with ``prefix``."""
    queue = list(parent)
    while queue:
        node = queue.pop(0)
        if local_name(node.tag) == "Population":
            if node.attrib.get("name", "").startswith(prefix):
                return node
            continue
        queue.extend(list(node))
    raise KeyError(f"Direct population beginning {prefix!r} not found")


def population_path(parent: ET.Element, names: tuple[str, ...]) -> ET.Element:
    node = parent
    for name in names:
        node = direct_population(node, name)
    return node


def population_count(node: ET.Element) -> int:
    return int(node.attrib["count"])


def quadrant_count(live_node: ET.Element, prefix: str) -> int:
    for child in live_node.iter():
        if local_name(child.tag) == "Population" and child.attrib.get("name", "").startswith(prefix):
            return population_count(child)
    raise KeyError(f"Quadrant {prefix!r} not found below {live_node.attrib.get('name')}")


def read_workspace_counts(workspace_bytes: bytes) -> dict[str, SampleCounts]:
    root = ET.fromstring(workspace_bytes)
    result: dict[str, SampleCounts] = {}

    source_files = sorted(
        {
            node.attrib["name"]
            for node in root.iter()
            if local_name(node.tag) == "SampleNode"
            and node.attrib.get("name", "").startswith("trim_")
            and node.attrib.get("name", "").endswith("_T.fcs")
        }
    )
    if len(source_files) != len(SAMPLES):
        raise ValueError(
            f"Expected {len(SAMPLES)} tumour samples in the workspace; "
            f"found {len(source_files)}"
        )
    sample_files = dict(zip(SAMPLES, source_files))

    for sample in SAMPLES:
        sample_node = find_sample_node(root, sample_files[sample])
        raw_total = population_count(sample_node)

        raw_live = direct_population(sample_node, "live CD45+ cells")
        manual_live = population_path(
            sample_node,
            ("Time, SSC-A subset", "cells", "Single Cells", "live CD45+ cells"),
        )
        flowmop_live = population_path(
            sample_node,
            ("passed_debris", "passed_doublet", "passed_time", "live CD45+ cells"),
        )
        live_nodes = {"Raw": raw_live, "Manual": manual_live, "FlowMOP": flowmop_live}
        quadrants = {
            "Raw": {
                quadrant: population_count(direct_population_prefix(sample_node, f"{quadrant}:"))
                for quadrant in ("Q1", "Q2", "Q3", "Q4")
            },
            "Manual": {
                quadrant: quadrant_count(manual_live, f"{quadrant}:")
                for quadrant in ("Q1", "Q2", "Q3", "Q4")
            },
            "FlowMOP": {
                quadrant: quadrant_count(flowmop_live, f"{quadrant}:")
                for quadrant in ("Q1", "Q2", "Q3", "Q4")
            },
        }

        # Workspace controls identify BUV395-A as CD3 and BV510-A as CD19.
        # Thus Q1 is CD3+CD19- (T) and Q3 is CD3-CD19+ (B).
        result[sample] = SampleCounts(
            raw_total=raw_total,
            live={method: population_count(node) for method, node in live_nodes.items()},
            t_cells={
                "Raw": population_count(direct_population_prefix(sample_node, "Q1:")),
                "Manual": quadrant_count(manual_live, "Q1:"),
                "FlowMOP": quadrant_count(flowmop_live, "Q1:"),
            },
            b_cells={
                "Raw": population_count(direct_population_prefix(sample_node, "Q3:")),
                "Manual": quadrant_count(manual_live, "Q3:"),
                "FlowMOP": quadrant_count(flowmop_live, "Q3:"),
            },
            quadrants=quadrants,
        )

    return result


def calculate_values(counts: dict[str, SampleCounts]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    raw_values: dict[tuple[str, str], float] = {}

    for sample in SAMPLES:
        c = counts[sample]
        for method in METHODS:
            values = {
                "live_cd45_count": float(c.live[method]),
                "t_cell_frequency_pct_total": 100.0 * c.t_cells[method] / c.raw_total,
                "b_cell_frequency_pct_total": 100.0 * c.b_cells[method] / c.raw_total,
            }
            if method == "Raw":
                for endpoint, value in values.items():
                    raw_values[(sample, endpoint)] = value
            for endpoint, value in values.items():
                rows.append(
                    {
                        "sample": sample,
                        "method": method,
                        "endpoint": endpoint,
                        "endpoint_label": ENDPOINT_LABELS[endpoint].replace("\n", " "),
                        "value": value,
                        "raw_total_events": c.raw_total,
                        "live_cd45_count": c.live[method],
                        "t_cell_count_q1": c.t_cells[method],
                        "b_cell_count_q3": c.b_cells[method],
                    }
                )

    for row in rows:
        raw = raw_values[(str(row["sample"]), str(row["endpoint"]))]
        row["raw_normalized_pct"] = 100.0 * float(row["value"]) / raw
    return rows


def calculate_tests(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tests: list[dict[str, object]] = []
    comparisons = (("Manual", "Raw"), ("FlowMOP", "Raw"), ("Manual", "FlowMOP"))
    for endpoint in ENDPOINT_LABELS:
        by_method = {
            method: np.array(
                [
                    next(
                        float(row["raw_normalized_pct"])
                        for row in rows
                        if row["sample"] == sample
                        and row["method"] == method
                        and row["endpoint"] == endpoint
                    )
                    for sample in SAMPLES
                ],
                dtype=float,
            )
            for method in METHODS
        }
        for method_a, method_b in comparisons:
            result = ttest_rel(by_method[method_a], by_method[method_b])
            tests.append(
                {
                    "endpoint": endpoint,
                    "endpoint_label": ENDPOINT_LABELS[endpoint].replace("\n", " "),
                    "comparison": f"{method_a} vs {method_b}",
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_pairs": len(SAMPLES),
                    "t_statistic": float(result.statistic),
                    "p_value": float(result.pvalue),
                    "p_adjustment": "none",
                }
            )
    return tests


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_page(pdf_bytes: bytes, page_index: int, scale: float = 4.0) -> Image.Image:
    document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pixmap = document[page_index].get_pixmap(
        matrix=pymupdf.Matrix(scale, scale),
        alpha=False,
    )
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    document.close()
    return image


def crop_points(page: Image.Image, rect: tuple[float, float, float, float], scale: float) -> Image.Image:
    return page.crop(tuple(round(value * scale) for value in rect))


def remove_raster_quadrant_annotations(image: Image.Image) -> Image.Image:
    """Mask FlowJo's corner annotations so they can be redrawn as vector text."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    inset = max(3, round(min(width, height) * 0.012))
    box_width = round(width * 0.20)
    top_right_box_width = round(width * 0.27)
    box_height = round(height * 0.15)
    boxes = (
        (inset, inset, box_width, box_height),
        (width - top_right_box_width, inset, width - inset, box_height),
        (inset, height - box_height, box_width, height - inset),
        (width - box_width, height - box_height, width - inset, height - inset),
    )
    for box in boxes:
        draw.rectangle(box, fill="white")
    return image


def format_quadrant_percentage(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def add_vector_quadrant_annotations(
    ax: plt.Axes,
    quadrant_counts: dict[str, int],
) -> None:
    total = sum(quadrant_counts.values())
    positions = {
        "Q1": (0.022, 0.978, "left", "top"),
        "Q2": (0.978, 0.978, "right", "top"),
        "Q3": (0.978, 0.022, "right", "bottom"),
        "Q4": (0.022, 0.022, "left", "bottom"),
    }
    for quadrant, (x, y, horizontal, vertical) in positions.items():
        percentage = 100.0 * quadrant_counts[quadrant] / total
        ax.text(
            x,
            y,
            f"{quadrant}\n{format_quadrant_percentage(percentage)}",
            transform=ax.transAxes,
            ha=horizontal,
            va=vertical,
            fontsize=12,
            fontweight="bold",
            linespacing=0.95,
            color="#111111",
            zorder=5,
        )


def _rotated_text(
    image: Image.Image,
    centre: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
) -> None:
    probe = ImageDraw.Draw(image)
    box = probe.textbbox((0, 0), text, font=font)
    width = box[2] - box[0] + 8
    height = box[3] - box[1] + 8
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    draw.text((4 - box[0], 4 - box[1]), text, fill="#111111", font=font)
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    x = round(centre[0] - label.width / 2)
    y = round(centre[1] - label.height / 2)
    image.paste(label, (x, y), label)


def relabel_subset_axes(image: Image.Image, gate_type: str, scale: float) -> Image.Image:
    """Replace FlowJo fluorophore axis text with biological marker labels."""
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font_path = Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), round(8.5 * scale))

    # Coordinates are relative to the strip crop in PDF points.
    x_centres_points = (69, 201, 317)
    # The first source panel has a narrower left gutter than the two gated
    # panels. Mask and replace at the exact FlowJo label positions so no
    # fluorophore text remains beside the biological protein labels.
    y_label_centres_points = (9, 161, 277)
    height_points = image.height / scale
    bottom_band = (height_points - 38, height_points)

    # Remove the original fluorophore names, retaining ticks and gate geometry.
    draw.rectangle(
        (0, round(bottom_band[0] * scale), image.width, image.height),
        fill="white",
    )
    for centre in y_label_centres_points:
        draw.rectangle(
            (
                round((centre - 14) * scale),
                round(20 * scale),
                round((centre + 14) * scale),
                round((height_points - 38) * scale),
            ),
            fill="white",
        )

    x_label = "CD45" if gate_type == "live" else "CD19"
    y_label = "Live/Dead" if gate_type == "live" else "CD3"
    for centre in x_centres_points:
        box = draw.textbbox((0, 0), x_label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (round(centre * scale - text_width / 2), round((height_points - 17) * scale)),
            x_label,
            fill="#111111",
            font=font,
        )
    for centre in y_label_centres_points:
        _rotated_text(
            image,
            (round(centre * scale), round((height_points / 2) * scale)),
            y_label,
            font,
        )
    return image


def endpoint_array(
    rows: list[dict[str, object]], endpoint: str, method: str
) -> np.ndarray:
    return np.array(
        [
            next(
                float(row["raw_normalized_pct"])
                for row in rows
                if row["sample"] == sample
                and row["method"] == method
                and row["endpoint"] == endpoint
            )
            for sample in SAMPLES
        ]
    )


def p_label(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#333333", lw=1.5)
    ax.text(
        (x1 + x2) / 2,
        y + height * 1.15,
        label,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "svg.fonttype": "none",
            "svg.image_inline": True,
            "savefig.facecolor": "white",
        }
    )


def clean_svg_whitespace(path: Path) -> None:
    """Remove Matplotlib's trailing spaces while preserving valid SVG text."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def make_figure_7(
    subset_pdf: bytes,
    counts: dict[str, SampleCounts],
    rows: list[dict[str, object]],
    tests: list[dict[str, object]],
) -> None:
    scale = 4.0
    pages = [render_page(subset_pdf, index, scale) for index in range(3)]

    fig = plt.figure(figsize=(18, 18), facecolor="white", constrained_layout=False)
    # Panel A uses its own compact grid so the square flow plots determine the
    # layout instead of inheriting the much wider statistical-panel columns.
    gating_grid = fig.add_gridspec(
        3,
        3,
        left=0.15,
        right=0.91,
        bottom=0.40,
        top=0.95,
        wspace=0.02,
        hspace=0.025,
    )
    gating_axes: list[plt.Axes] = []

    for row_index, (_sample, page) in enumerate(zip(SAMPLES, pages)):
        for column, rect in enumerate(TB_PLOT_RECTS):
            crop = remove_raster_quadrant_annotations(crop_points(page, rect, scale))
            ax = fig.add_subplot(gating_grid[row_index, column])
            ax.imshow(crop)
            ax.set_axis_off()
            ax.add_patch(
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    transform=ax.transAxes,
                    fill=False,
                    edgecolor="#888888",
                    linewidth=0.8,
                    zorder=4,
                )
            )
            add_vector_quadrant_annotations(ax, counts[_sample].quadrants[METHODS[column]])
            gating_axes.append(ax)

    grid_left = min(ax.get_position().x0 for ax in gating_axes)
    grid_right = max(ax.get_position().x1 for ax in gating_axes)
    grid_bottom = min(ax.get_position().y0 for ax in gating_axes)
    grid_top = max(ax.get_position().y1 for ax in gating_axes)
    fig.text(0.018, 0.965, "B)", fontsize=23, fontweight="bold", va="top")
    for ax, method in zip(gating_axes[:3], METHODS):
        pos = ax.get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            grid_top + 0.018,
            method,
            fontsize=19,
            fontweight="bold",
            ha="center",
        )
    for row_index, sample in enumerate(SAMPLES):
        row_axes = gating_axes[row_index * 3 : (row_index + 1) * 3]
        row_centre = sum(ax.get_position().y0 + ax.get_position().height / 2 for ax in row_axes) / 3
        fig.text(
            grid_left - 0.090,
            row_centre,
            sample,
            fontsize=19,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
    # Anchor the shared key to the bottom-left plot and use the same arrow
    # geometry, stroke, arrowheads, and typography as panel A.
    add_arrowed_plot_axes(gating_axes[-3], "CD19", "CD3", fontsize=20)

    stats_grid = fig.add_gridspec(
        1,
        3,
        left=0.095,
        right=0.975,
        bottom=0.045,
        top=0.255,
        wspace=0.34,
    )
    endpoint_order = (
        "live_cd45_count",
        "b_cell_frequency_pct_total",
        "t_cell_frequency_pct_total",
    )
    test_lookup = {(row["endpoint"], row["comparison"]): float(row["p_value"]) for row in tests}
    x = np.arange(len(METHODS), dtype=float)

    for panel_index, endpoint in enumerate(endpoint_order):
        ax = fig.add_subplot(stats_grid[panel_index])
        values = {method: endpoint_array(rows, endpoint, method) for method in METHODS}
        matrix = np.column_stack([values[method] for method in METHODS])

        for sample_index, _sample in enumerate(SAMPLES):
            ax.plot(
                x,
                matrix[sample_index],
                color="#b7b7b7",
                lw=1.8,
                alpha=0.9,
                zorder=1,
            )
            for method_index, method in enumerate(METHODS):
                ax.scatter(
                    method_index,
                    matrix[sample_index, method_index],
                    s=70,
                    facecolor=METHOD_COLOURS[method],
                    edgecolor="white",
                    linewidth=0.9,
                    zorder=3,
                )

        for method_index, method in enumerate(METHODS):
            mean = float(np.mean(values[method]))
            sd = float(np.std(values[method], ddof=1))
            ax.errorbar(
                method_index,
                mean,
                yerr=sd,
                fmt="none",
                color="#111111",
                capsize=5,
                capthick=1.8,
                lw=2.0,
                zorder=2,
            )
            ax.scatter(
                method_index,
                mean,
                marker="D",
                s=64,
                facecolor=METHOD_COLOURS[method],
                edgecolor="#111111",
                linewidth=1.0,
                zorder=2.5,
            )

        ax.axhline(100, color="#777777", lw=1.2, ls="--", zorder=0)
        ax.set_title(ENDPOINT_LABELS[endpoint], fontsize=17, fontweight="bold", pad=13)
        ax.set_xticks(x, METHODS)
        ax.tick_params(axis="x", labelsize=17, width=1.5)
        ax.tick_params(axis="y", labelsize=17, width=1.3)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight("bold")
        ax.set_ylabel(
            "Recovery relative\nto Raw (%)" if panel_index == 0 else "",
            fontsize=24,
            fontweight="bold",
            labelpad=32 if panel_index == 0 else 0,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e7e7e7", lw=0.7)
        ax.set_xlim(-0.35, 2.35)

        data_max = max(110.0, float(np.nanmax(matrix)))
        data_min = min(0.0, float(np.nanmin(matrix)))
        span = data_max - data_min
        bracket_height = max(3.0, span * 0.025)
        bracket_start = data_max + span * 0.08
        bracket_gap = max(10.0, span * 0.13)
        comparisons = (
            (0, 1, "Manual vs Raw"),
            (0, 2, "FlowMOP vs Raw"),
            (1, 2, "Manual vs FlowMOP"),
        )
        significant = [
            (x1, x2, comparison)
            for x1, x2, comparison in comparisons
            if test_lookup[(endpoint, comparison)] < 0.05
        ]
        for level, (x1, x2, comparison) in enumerate(significant):
            add_bracket(
                ax,
                x1,
                x2,
                bracket_start + level * bracket_gap,
                bracket_height,
                p_label(test_lookup[(endpoint, comparison)]),
            )
        if significant:
            ax.set_ylim(data_min, bracket_start + (len(significant) - 0.25) * bracket_gap)
        else:
            ax.set_ylim(data_min, data_max + max(8.0, span * 0.10))

    stats_top = max(ax.get_position().y1 for ax in fig.axes[-3:])
    fig.text(0.018, stats_top + 0.055, "C)", fontsize=23, fontweight="bold", va="top")

    fig.savefig(PANELS_SVG)
    clean_svg_whitespace(PANELS_SVG)
    fig.savefig(PANELS_PNG, dpi=300)
    plt.close(fig)


def figure_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3,rad=0",
    arrowstyle: str = "-|>",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle=arrowstyle,
        mutation_scale=16,
        linewidth=1.5,
        color="#444444",
        connectionstyle=connectionstyle,
        zorder=10,
    )
    fig.add_artist(arrow)


def add_image_axis(
    fig: plt.Figure,
    bounds: tuple[float, float, float, float],
    image: Image.Image,
    title: str,
) -> plt.Axes:
    ax = fig.add_axes(bounds)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, fontsize=13, fontweight="bold", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)
    ax.add_patch(
        Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor="#999999",
            linewidth=0.8,
            zorder=5,
        )
    )
    return ax


def add_arrowed_plot_axes(
    ax: plt.Axes,
    x_label: str,
    y_label: str,
    *,
    fontsize: float = 20,
) -> None:
    """Add the same balanced arrow-axis geometry used in Figure 6."""
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
        0.205, -0.19, x_label,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=fontsize, fontweight="bold",
        clip_on=False,
    )
    ax.text(
        -0.16, 0.175, y_label,
        transform=ax.transAxes,
        ha="center", va="center", rotation=90,
        fontsize=fontsize, fontweight="bold",
        clip_on=False,
    )


def crop_relative(
    image: Image.Image,
    bounds: tuple[float, float, float, float],
) -> Image.Image:
    """Crop using fractional bounds, primarily to remove embedded PDF axes."""
    left, top, right, bottom = bounds
    width, height = image.size
    return image.crop(
        (
            round(left * width),
            round(top * height),
            round(right * width),
            round(bottom * height),
        )
    )


def make_workflow_panel(manual_pdf: bytes, qc_pdf: bytes) -> None:
    scale = 4.0
    manual_page = render_page(manual_pdf, 0, scale)
    qc_page = render_page(qc_pdf, 0, scale)

    manual_rects = (
        (310, 20, 485, 238),
        (310, 265, 485, 460),
        (310, 500, 485, 690),
    )
    flowmop_rects = (
        # Use identically sized crops around each plotting frame. The source
        # PDF rows are separated by different amounts of metadata, so the
        # previous row-specific bounds left the time plot lower and clipped
        # the top of the doublet plot after the crops were resized.
        (250, 79, 330, 154),
        (250, 177, 330, 252),
        (250, 273, 330, 348),
    )
    manual_images = [crop_points(manual_page, rect, scale) for rect in manual_rects]
    flowmop_images = [crop_points(qc_page, rect, scale) for rect in flowmop_rects]

    # The first manual crop contains a small FlowJo method tag above the plot;
    # the simplified panel identifies each row directly.
    ImageDraw.Draw(manual_images[0]).rectangle(
        (0, 0, manual_images[0].width, round(25 * scale)),
        fill="white",
    )

    # Retain the plot frames and data while removing the FlowJo/FlowMOP tick
    # labels and embedded channel labels. Clean vector arrow axes are added
    # below the FlowMOP row, matching Figure 6.
    manual_frame_bounds = (
        (0.089, 0.108, 0.926, 0.780),
        (0.086, 0.064, 0.924, 0.818),
        (0.097, 0.057, 0.936, 0.830),
    )
    flowmop_frame_bounds = (
        (0.134, 0.043, 0.875, 0.837),
        (0.119, 0.033, 0.856, 0.830),
        (0.131, 0.033, 0.875, 0.830),
    )
    manual_images = [
        crop_relative(image, bounds)
        for image, bounds in zip(manual_images, manual_frame_bounds)
    ]
    flowmop_images = [
        crop_relative(image, bounds)
        for image, bounds in zip(flowmop_images, flowmop_frame_bounds)
    ]

    # Use the same square presentation for both rows so the representative
    # preprocessing plots occupy the same visual footprint as the plots in
    # panel B.
    workflow_square_size = round(110 * scale)
    manual_images = [
        image.resize(
            (workflow_square_size, workflow_square_size),
            Image.Resampling.LANCZOS,
        )
        for image in manual_images
    ]
    flowmop_images = [
        image.resize(
            (workflow_square_size, workflow_square_size),
            Image.Resampling.LANCZOS,
        )
        for image in flowmop_images
    ]

    fig = plt.figure(figsize=(18, 9), facecolor="white")
    fig.text(0.012, 0.98, "A)", fontsize=23, fontweight="bold", va="top")

    x_positions = (0.175, 0.425, 0.675)
    plot_width = 0.21
    plot_height = 0.40
    manual_y = 0.54
    flowmop_y = 0.12
    column_titles = ("Time", "Debris", "Doublet")

    for x, title in zip(x_positions, column_titles):
        fig.text(
            x + plot_width / 2, 0.965, title,
            ha="center", va="center", fontsize=19, fontweight="bold",
        )

    for x, image in zip(x_positions, manual_images):
        add_image_axis(fig, (x, manual_y, plot_width, plot_height), image, "")
    flowmop_axes = [
        add_image_axis(fig, (x, flowmop_y, plot_width, plot_height), image, "")
        for x, image in zip(x_positions, flowmop_images)
    ]

    for ax, x_label, y_label in zip(
        flowmop_axes,
        ("Time", "FSC-A", "FSC-A"),
        ("SSC-A", "SSC-A", "FSC-H"),
    ):
        add_arrowed_plot_axes(ax, x_label, y_label, fontsize=20)

    fig.text(
        0.095, manual_y + plot_height / 2, "Manual",
        fontsize=22, fontweight="bold", rotation=90, ha="center", va="center",
    )
    fig.text(
        0.095, flowmop_y + plot_height / 2, "FlowMOP",
        fontsize=22, fontweight="bold", rotation=90, ha="center", va="center",
    )

    fig.savefig(WORKFLOW_SVG)
    clean_svg_whitespace(WORKFLOW_SVG)
    fig.savefig(WORKFLOW_PNG, dpi=300)
    plt.close(fig)


def compose_figure_7() -> None:
    """Stack the preprocessing plots above the tumour plots and statistics."""
    workflow = base64.b64encode(WORKFLOW_SVG.read_bytes()).decode("ascii")
    panels = base64.b64encode(PANELS_SVG.read_bytes()).decode("ascii")
    width = 1296
    workflow_height = 648
    panels_height = 1296
    total_height = workflow_height + panels_height
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
 width="18in" height="27in" viewBox="0 0 {width} {total_height}">
 <image x="0" y="0" width="{width}" height="{workflow_height}"
  href="data:image/svg+xml;base64,{workflow}"/>
 <image x="0" y="{workflow_height}" width="{width}" height="{panels_height}"
  href="data:image/svg+xml;base64,{panels}"/>
</svg>\n'''
    FIGURE_SVG.write_text(svg, encoding="utf-8")
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("cairosvg is required to rasterise the composed Figure 7") from exc
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(FIGURE_PNG),
                     output_width=2700, output_height=4050)


def validate_expected_tests(tests: list[dict[str, object]]) -> None:
    expected = {
        ("live_cd45_count", "Manual vs Raw"): 0.0690529497646822,
        ("live_cd45_count", "FlowMOP vs Raw"): 0.02413878756883129,
        ("live_cd45_count", "Manual vs FlowMOP"): 0.5451172596060034,
        ("t_cell_frequency_pct_total", "Manual vs Raw"): 0.022539919204771903,
        ("t_cell_frequency_pct_total", "FlowMOP vs Raw"): 0.006243241891017893,
        ("t_cell_frequency_pct_total", "Manual vs FlowMOP"): 0.9850278614301512,
        ("b_cell_frequency_pct_total", "Manual vs Raw"): 0.02497001334909625,
        ("b_cell_frequency_pct_total", "FlowMOP vs Raw"): 0.020416317359503784,
        ("b_cell_frequency_pct_total", "Manual vs FlowMOP"): 0.8552235077516148,
    }
    observed = {(str(row["endpoint"]), str(row["comparison"])): float(row["p_value"]) for row in tests}
    for key, expected_value in expected.items():
        if not math.isclose(observed[key], expected_value, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f"Unexpected paired-test result for {key}: {observed[key]}")


def main() -> None:
    configure_style()
    if not SOURCE_ARCHIVE.exists():
        raise FileNotFoundError(f"Tumour source archive not found: {SOURCE_ARCHIVE}")

    with ZipFile(SOURCE_ARCHIVE) as archive:
        workspace = archive.read("20260812_gating.wsp")
        subset_pdf = archive.read("20260812_gating/Subset gating.pdf")
        manual_pdf = archive.read("20260812_gating/Manual gating.pdf")
        qc_pdf = archive.read("20260812_gating/QC gating.pdf")

    counts = read_workspace_counts(workspace)
    rows = calculate_values(counts)
    tests = calculate_tests(rows)
    validate_expected_tests(tests)
    write_csv(VALUES_CSV, rows)
    write_csv(TESTS_CSV, tests)
    make_figure_7(subset_pdf, counts, rows, tests)
    make_workflow_panel(manual_pdf, qc_pdf)
    compose_figure_7()

    for path in (
        FIGURE_SVG, FIGURE_PNG, PANELS_SVG, PANELS_PNG,
        WORKFLOW_SVG, WORKFLOW_PNG, VALUES_CSV, TESTS_CSV,
    ):
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
