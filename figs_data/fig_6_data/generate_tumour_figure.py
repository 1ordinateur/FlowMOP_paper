#!/usr/bin/env python3
"""Generate tumour Figure 6 and its manual-versus-FlowMOP supplement.

The source PDFs and the authoritative population counts live in
``../flowmop_data/tumour_data/Shared_tumour_FlowMOP.zip``.  PyMuPDF is
used only to rasterise the supplied FlowJo PDF plots; statistical panels
and workflow annotations remain vector elements in the SVG outputs.

Required packages: matplotlib, numpy, Pillow, PyMuPDF, scipy.
"""

from __future__ import annotations

import csv
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
FIGURE_SVG = REPO / "figs_data/figure_6.svg"
FIGURE_PNG = REPO / "figs_data/figure_6.png"
SUPP_SVG = REPO / "figs_data/Supp_fig_8.svg"
SUPP_PNG = REPO / "figs_data/Supp_fig_8.png"
VALUES_CSV = HERE / "tumour_endpoint_values.csv"
TESTS_CSV = HERE / "tumour_paired_t_tests.csv"

SAMPLES = ("LB202", "LB236", "LB262")
SAMPLE_FILES = {sample: f"trim_{sample}_T.fcs" for sample in SAMPLES}
DISPLAY_SAMPLE_LABELS = {
    "LB202": "Sample 1",
    "LB236": "Sample 2",
    "LB262": "Sample 3",
}
METHODS = ("Raw", "Manual", "FlowMOP")
METHOD_COLOURS = {"Raw": "#6f6f6f", "Manual": "#D88700", "FlowMOP": "#0072B2"}

ENDPOINT_LABELS = {
    "live_cd45_count": "Live CD45+ cells",
    "t_cell_frequency_pct_total": "T-cell frequency\n(% original total)",
    "b_cell_frequency_pct_total": "B-cell frequency\n(% original total)",
    "t_b_ratio": "T:B-cell ratio",
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

    for sample in SAMPLES:
        sample_node = find_sample_node(root, SAMPLE_FILES[sample])
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
                "t_b_ratio": c.t_cells[method] / c.b_cells[method],
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
        fontsize=13,
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


def make_figure_6(
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
        bottom=0.37,
        top=0.92,
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
    fig.text(0.018, 0.965, "A)", fontsize=23, fontweight="bold", va="top")
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
            DISPLAY_SAMPLE_LABELS[sample],
            fontsize=19,
            fontweight="bold",
            rotation=90,
            ha="center",
            va="center",
        )
    axis_origin_x = grid_left - 0.030
    axis_origin_y = grid_bottom - 0.020
    axis_arrow_length = 0.065
    vertical_arrow = FancyArrowPatch(
        (axis_origin_x, axis_origin_y),
        (axis_origin_x, axis_origin_y + axis_arrow_length),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#111111",
        zorder=10,
    )
    fig.add_artist(vertical_arrow)
    fig.text(
        axis_origin_x - 0.014,
        axis_origin_y + axis_arrow_length / 2,
        "CD3",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        rotation=90,
    )

    horizontal_arrow = FancyArrowPatch(
        (axis_origin_x, axis_origin_y),
        (axis_origin_x + axis_arrow_length, axis_origin_y),
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=1.8,
        color="#111111",
        zorder=10,
    )
    fig.add_artist(horizontal_arrow)
    fig.text(
        axis_origin_x + axis_arrow_length / 2,
        axis_origin_y - 0.012,
        "CD19",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="top",
    )

    stats_grid = fig.add_gridspec(
        1,
        4,
        left=0.075,
        right=0.985,
        bottom=0.045,
        top=0.255,
        wspace=0.32,
    )
    endpoint_order = (
        "t_b_ratio",
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
        ax.tick_params(axis="x", labelsize=15, width=1.5)
        ax.tick_params(axis="y", labelsize=13, width=1.3)
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight("bold")
        ax.set_ylabel(
            "Relative to matched Raw (%)" if panel_index == 0 else "",
            fontsize=16,
            fontweight="bold",
            labelpad=28 if panel_index == 0 else 0,
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

    stats_top = max(ax.get_position().y1 for ax in fig.axes[-4:])
    fig.text(0.018, stats_top + 0.055, "B)", fontsize=23, fontweight="bold", va="top")

    fig.savefig(FIGURE_SVG)
    clean_svg_whitespace(FIGURE_SVG)
    fig.savefig(FIGURE_PNG, dpi=300)
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
    return ax


def make_supplement(manual_pdf: bytes, qc_pdf: bytes) -> None:
    scale = 4.0
    manual_page = render_page(manual_pdf, 0, scale)
    qc_page = render_page(qc_pdf, 0, scale)

    manual_rects = (
        (310, 20, 485, 238),
        (310, 265, 485, 460),
        (310, 500, 485, 690),
    )
    flowmop_rects = (
        (250, 65, 330, 161),
        (250, 178, 330, 260),
        (250, 295, 330, 352),
    )
    manual_images = [crop_points(manual_page, rect, scale) for rect in manual_rects]
    flowmop_images = [crop_points(qc_page, rect, scale) for rect in flowmop_rects]

    # The first source crops contain small FlowJo method tags above the plots.
    # The figure already identifies each workflow with the large row labels.
    ImageDraw.Draw(manual_images[0]).rectangle(
        (0, 0, manual_images[0].width, round(25 * scale)),
        fill="white",
    )
    ImageDraw.Draw(flowmop_images[0]).rectangle(
        (0, 0, flowmop_images[0].width, round(15 * scale)),
        fill="white",
    )

    flowmop_square_size = round(90 * scale)
    flowmop_images = [
        image.resize(
            (flowmop_square_size, flowmop_square_size),
            Image.Resampling.LANCZOS,
        )
        for image in flowmop_images
    ]

    fig = plt.figure(figsize=(18, 9.5), facecolor="white")
    fig.text(
        0.02,
        0.96,
        f"Representative tumour {DISPLAY_SAMPLE_LABELS[SAMPLES[0]]}",
        fontsize=17,
        fontweight="bold",
    )

    manual_x_positions = (0.16, 0.42, 0.68)
    manual_width = 0.18
    manual_y, manual_h = 0.57, 0.31
    flow_x_positions = (0.17, 0.43, 0.69)
    flow_width = 0.15
    flow_h = flow_width * 18 / 9.5
    flow_y = 0.13
    manual_titles = ("Time gate", "Cells / debris gate", "Single-cell gate")
    flow_titles = ("Time exclusion", "Debris exclusion", "Doublet exclusion")

    manual_axes = [
        add_image_axis(fig, (x, manual_y, manual_width, manual_h), image, title)
        for x, image, title in zip(manual_x_positions, manual_images, manual_titles)
    ]
    flow_axes = [
        add_image_axis(fig, (x, flow_y, flow_width, flow_h), image, "")
        for x, image in zip(flow_x_positions, flowmop_images)
    ]
    flow_top = flow_y + flow_h
    split_y = flow_top + 0.03
    for x, title in zip(flow_x_positions, flow_titles):
        fig.text(
            x + flow_width / 2,
            split_y + 0.035,
            title,
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    fig.text(0.025, 0.75, "Manual", fontsize=16, fontweight="bold", color=METHOD_COLOURS["Manual"])
    fig.text(0.025, 0.29, "FlowMOP", fontsize=16, fontweight="bold", color=METHOD_COLOURS["FlowMOP"])
    fig.text(0.025, 0.70, "Sequential", fontsize=12, color="#444444")
    fig.text(0.025, 0.24, "Parallel", fontsize=12, color="#444444")

    # Manual gates are applied successively, left to right.
    for left_ax, right_ax in zip(manual_axes[:-1], manual_axes[1:]):
        left = left_ax.get_position()
        right = right_ax.get_position()
        figure_arrow(fig, (left.x1 + 0.01, left.y0 + left.height / 2), (right.x0 - 0.01, right.y0 + right.height / 2))

    result_box_x = 0.925
    result_box_left = 0.875
    last_manual = manual_axes[-1].get_position()
    manual_result_y = last_manual.y0 + last_manual.height / 2
    fig.text(
        result_box_x,
        manual_result_y,
        "Manual retained\npopulation",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FFF3D9", ec=METHOD_COLOURS["Manual"], lw=1.5),
    )
    figure_arrow(
        fig,
        (last_manual.x1 + 0.01, manual_result_y),
        (result_box_left, manual_result_y),
    )

    # FlowMOP supplies the same Raw input to three independent branches. A
    # split rail above and convergence rail below keep the arrows out of the
    # source plots while making the parallel structure explicit.
    fig.text(
        0.085,
        split_y,
        "Raw events",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.45", fc="#f3f3f3", ec="#666666", lw=1.4),
    )
    input_rail_start = 0.14
    input_rail_end = flow_axes[-1].get_position().x1 + 0.025
    fig.add_artist(
        mpl.lines.Line2D(
            [input_rail_start, input_rail_end],
            [split_y, split_y],
            transform=fig.transFigure,
            color="#444444",
            lw=1.5,
        )
    )
    figure_arrow(fig, (0.12, split_y), (input_rail_start, split_y))
    for ax in flow_axes:
        pos = ax.get_position()
        figure_arrow(
            fig,
            (pos.x0 + pos.width / 2, split_y),
            (pos.x0 + pos.width / 2, pos.y1 - 0.002),
        )

    # The three independent branches converge at a bracket representing the
    # intersection of events retained by every FlowMOP gate.
    merge_y = 0.06
    branch_centres = [
        ax.get_position().x0 + ax.get_position().width / 2
        for ax in flow_axes
    ]
    join_offset = 0.035
    for ax, branch_x in zip(flow_axes, branch_centres):
        pos = ax.get_position()
        figure_arrow(
            fig,
            (branch_x, pos.y0 - 0.004),
            (branch_x + join_offset, merge_y),
            connectionstyle="arc3,rad=0.35",
            arrowstyle="-",
        )
    # Draw the final arrow after the branch connectors so its baseline remains
    # visually continuous through each merge point.
    figure_arrow(
        fig,
        (branch_centres[0] + join_offset, merge_y),
        (result_box_left, merge_y),
    )

    fig.text(
        result_box_x,
        merge_y,
        "Intersection of\nretained events",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="#DCEEF8", ec=METHOD_COLOURS["FlowMOP"], lw=1.5),
    )

    fig.savefig(SUPP_SVG)
    clean_svg_whitespace(SUPP_SVG)
    fig.savefig(SUPP_PNG, dpi=300)
    plt.close(fig)


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
        ("t_b_ratio", "Manual vs Raw"): 0.7153138768719286,
        ("t_b_ratio", "FlowMOP vs Raw"): 0.906148056655327,
        ("t_b_ratio", "Manual vs FlowMOP"): 0.6357144095973232,
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
    make_figure_6(subset_pdf, counts, rows, tests)
    make_supplement(manual_pdf, qc_pdf)

    for path in (FIGURE_SVG, FIGURE_PNG, SUPP_SVG, SUPP_PNG, VALUES_CSV, TESTS_CSV):
        print(path.relative_to(REPO))


if __name__ == "__main__":
    main()
