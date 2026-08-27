#!/usr/bin/env python3
"""Create submission-sized vector panels for Figures 2 and 3.

The original composites remain the source of the flow-density panels.  This
script crops them into page-sized SVGs, normalises their typography, and
redraws the small orientation axes with the same arrowed, bold
style used by the biological-validation figures.  Figure 3C/D are regenerated
from the repository CSVs so their labels and statistics remain reproducible.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import ttest_rel


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE / "fig_2_data" / "fig_2_analysis"
SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

METHOD_COLOURS = {
    "FlowMOP": "#0072B2",
    "PeacoQC": "#009E73",
    "FlowCut": "#CC79A7",
}
FIGURE2_METHOD_COLOURS = {
    "flowmop": "#5875A4",
    "peacoqc": "#CC8963",
    "flowcut": "#5F9E6E",
}
EXPERT_COLOURS = {
    "Expert 1": "#009E73",
    "Expert 2": "#D55E00",
    "Expert 3": "#7B61A8",
    "Expert 4": "#E69F00",
}


def qname(name: str) -> str:
    return f"{{{SVG_NS}}}{name}"


def normalise_svg(root: ET.Element, figure: int) -> None:
    """Apply the shared sans-serif hierarchy and stable method palette."""
    colour_map = {}
    size_map: dict[str, str]
    if figure == 2:
        # Preserve the established Figure 2 palette.  Only its typography and
        # composition are harmonised with the later manuscript figures.
        size_map = {
            "26.6667": "17",
            "18.6667": "15",
            "16": "13",
            "14": "11",
            "13.3333": "11.5",
            "10.6667": "9.5",
            "7.11111": "8",
        }
    else:
        colour_map = {"#5875A4": METHOD_COLOURS["FlowMOP"]}
        size_map = {
            "26.6667": "14",
            "24": "13",
            "14.6667": "10",
            "13.3333": "9",
            "12": "8.5",
            "10.6667": "8",
        }

    for element in root.iter():
        for attribute, value in tuple(element.attrib.items()):
            for old, new in colour_map.items():
                value = re.sub(re.escape(old), new, value, flags=re.IGNORECASE)
            element.set(attribute, value)
        family = element.attrib.get("font-family")
        if family:
            element.set("font-family", "DejaVu Sans")
        style = element.attrib.get("style")
        if style:
            style = re.sub(
                r"font-family:[^;]+", "font-family:DejaVu Sans", style
            )
            for old, new in size_map.items():
                style = re.sub(
                    rf"font-size:\s*{re.escape(old)}(?=px|pt|;)",
                    f"font-size:{new}",
                    style,
                )
            element.set("style", style)

        size = element.attrib.get("font-size")
        if size in size_map:
            element.set("font-size", size_map[size])

        for attribute in ("fill", "stroke"):
            value = element.attrib.get(attribute)
            if value in colour_map:
                element.set(attribute, colour_map[value])

        text = (element.text or "").strip()
        if text == "BiMix":
            element.text = "Bimix"
            text = "Bimix"
        if figure == 3 and text == "Unprocessed":
            element.text = ""
            text = ""
        elif figure == 3 and text == "/ Ungated":
            element.text = "Ungated"
            element.set("font-weight", "700")
            element.set(
                "transform",
                "matrix(-1.83697e-16 -1 1 -1.83697e-16 32.0513 130)",
            )
            text = "Ungated"
        if text.startswith("p "):
            element.set("font-family", "DejaVu Sans")
            element.set("font-weight", "700")
            element.set("font-size", "11.5" if figure == 2 else "9")
            text = re.sub(r"(?<=\D)0\.(?=\d)", ".", text)
            match = re.fullmatch(r"p = \.(\d{1,3})", text)
            if match:
                text = f"p = .{match.group(1).ljust(3, '0')}"
            element.text = text
        if text in {
            "Unprocessed", "FlowMOP", "PeacoQC", "FlowCut",
            "Sensitivity", "Specificity", "Combined", "High Debris",
            "Low Debris", "Expert",
        }:
            element.set("font-family", "DejaVu Sans")
            element.set("font-weight", "700")


def add_arrow_marker(defs: ET.Element, marker_id: str) -> None:
    marker = ET.SubElement(
        defs,
        qname("marker"),
        {
            "id": marker_id,
            "markerWidth": "5",
            "markerHeight": "5",
            "refX": "4.6",
            "refY": "2.5",
            "orient": "auto",
            "markerUnits": "strokeWidth",
        },
    )
    ET.SubElement(
        marker,
        qname("path"),
        {"d": "M 0 0 L 5 2.5 L 0 5 z", "fill": "#333333"},
    )


def add_axis_key(
    root: ET.Element,
    *,
    origin: tuple[float, float],
    x_label: str,
    y_label: str,
    marker_id: str,
    length: float,
    fontsize: float,
    line_width: float = 0.8,
) -> None:
    defs = next((child for child in root if child.tag == qname("defs")), None)
    if defs is None:
        defs = ET.Element(qname("defs"))
        root.insert(0, defs)
    add_arrow_marker(defs, marker_id)

    x0, y0 = origin
    common = {
        "stroke": "#333333",
        "stroke-width": str(line_width),
        "stroke-linecap": "round",
        "marker-end": f"url(#{marker_id})",
    }
    ET.SubElement(
        root, qname("line"),
        {**common, "x1": str(x0), "y1": str(y0), "x2": str(x0 + length), "y2": str(y0)},
    )
    ET.SubElement(
        root, qname("line"),
        {**common, "x1": str(x0), "y1": str(y0), "x2": str(x0), "y2": str(y0 - length)},
    )

    text_common = {
        "font-family": "DejaVu Sans",
        "font-size": str(fontsize),
        "font-weight": "700",
        "fill": "#222222",
    }
    x_text = ET.SubElement(
        root,
        qname("text"),
        {
            **text_common,
            "x": str(x0 + length / 2),
            "y": str(y0 + fontsize + 3),
            "text-anchor": "middle",
        },
    )
    x_text.text = x_label
    y_text = ET.SubElement(
        root,
        qname("text"),
        {
            **text_common,
            "x": str(x0 - fontsize - 3),
            "y": str(y0 - length / 2),
            "text-anchor": "middle",
            "dominant-baseline": "middle",
            "transform": f"rotate(-90 {x0 - fontsize - 3} {y0 - length / 2})",
        },
    )
    y_text.text = y_label


def crop_panel(
    source: Path,
    output: Path,
    *,
    figure: int,
    crop: tuple[float, float, float, float],
    covers: tuple[tuple[float, float, float, float], ...] = (),
    vertical_spines: tuple[tuple[float, float, float], ...] = (),
    axis: tuple[tuple[float, float], str, str, float, float] | tuple[tuple[float, float], str, str, float, float, float] | None = None,
    text_y_shifts: dict[str, float] | None = None,
    slices: tuple[tuple[float, float, float], ...] = (),
    labels: tuple[tuple[float, float, str, float], ...] = (),
) -> None:
    source_root = ET.parse(source).getroot()
    normalise_svg(source_root, figure)
    if text_y_shifts:
        translate_pattern = re.compile(
            r"translate\(\s*([-+0-9.eE]+)[ ,]+([-+0-9.eE]+)\s*\)"
        )
        for element in source_root.iter():
            shift = text_y_shifts.get((element.text or "").strip())
            transform = element.attrib.get("transform")
            if shift is None or not transform:
                continue
            match = translate_pattern.fullmatch(transform)
            if match:
                element.set(
                    "transform",
                    f"translate({match.group(1)} {float(match.group(2)) + shift:g})",
                )
    x, y, width, height = crop

    root = ET.Element(
        qname("svg"),
        {
            "width": str(width),
            "height": str(height),
            "viewBox": f"0 0 {width} {height}",
            "version": "1.1",
        },
    )
    for child in list(source_root):
        if child.tag == qname("defs"):
            root.append(copy.deepcopy(child))
    ET.SubElement(
        root, qname("rect"),
        {"x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "#FFFFFF"},
    )
    if slices:
        defs = next(child for child in root if child.tag == qname("defs"))
        for slice_index, (source_y1, source_y2, destination_y) in enumerate(slices):
            clip_id = f"panel-slice-{figure}-{output.stem}-{slice_index}"
            clip = ET.SubElement(
                defs,
                qname("clipPath"),
                {"id": clip_id, "clipPathUnits": "userSpaceOnUse"},
            )
            ET.SubElement(
                clip,
                qname("rect"),
                {
                    "x": "0",
                    "y": str(destination_y),
                    "width": str(width),
                    "height": str(source_y2 - source_y1),
                },
            )
            clipped = ET.SubElement(
                root, qname("g"), {"clip-path": f"url(#{clip_id})"}
            )
            content = ET.SubElement(
                clipped,
                qname("g"),
                {"transform": f"translate({-x} {destination_y - source_y1})"},
            )
            for child in list(source_root):
                if child.tag != qname("defs"):
                    content.append(copy.deepcopy(child))
    else:
        content = ET.SubElement(root, qname("g"), {"transform": f"translate({-x} {-y})"})
        for child in list(source_root):
            if child.tag != qname("defs"):
                content.append(copy.deepcopy(child))

    for cx, cy, cw, ch in covers:
        ET.SubElement(
            root, qname("rect"),
            {"x": str(cx), "y": str(cy), "width": str(cw), "height": str(ch), "fill": "#FFFFFF"},
        )
    for spine_x, spine_y1, spine_y2 in vertical_spines:
        ET.SubElement(
            root,
            qname("line"),
            {
                "x1": str(spine_x),
                "y1": str(spine_y1),
                "x2": str(spine_x),
                "y2": str(spine_y2),
                "stroke": "#777777",
                "stroke-width": "0.6",
            },
        )
    for label_x, label_y, label_text, label_size in labels:
        text_element = ET.SubElement(
            root,
            qname("text"),
            {
                "x": str(label_x),
                "y": str(label_y),
                "text-anchor": "middle",
                "font-family": "DejaVu Sans",
                "font-size": str(label_size),
                "font-weight": "400",
                "fill": "#222222",
            },
        )
        text_element.text = label_text
    if axis:
        origin, x_label, y_label, length, fontsize = axis[:5]
        line_width = axis[5] if len(axis) == 6 else 0.8
        add_axis_key(
            root,
            origin=origin,
            x_label=x_label,
            y_label=y_label,
            marker_id=f"harmonized-arrow-{figure}-{output.stem}",
            length=length,
            fontsize=fontsize,
            line_width=line_width,
        )

    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)


def parse_figure3_data() -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], float]:
    flowmop = pd.read_csv(ANALYSIS / "flowmop_debrisgates.csv").iloc[:3].copy()
    flowmop["sample"] = flowmop.iloc[:, 0].str.extract(r"_(\d+)\.fcs").astype(int)
    flowmop = flowmop.sort_values("sample")
    flowmop_values = (
        100 - flowmop["passeddebris subset/sample1 | Freq. of Parent (%)"]
    ).to_numpy(float)

    human = pd.read_csv(ANALYSIS / "human_debris.csv")
    human = human[~human.iloc[:, 0].isin(("Mean", "SD"))].copy()
    human["sample"] = human.iloc[:, 0].str.extract(r"debris_(\d+)").astype(int)
    identifiers = {
        "Expert 1": "E1",
        "Expert 2": "Debris_E2",
        "Expert 3": "E3",
        "Expert 4": "null_debris",
    }
    strategies: dict[str, dict[str, np.ndarray]] = {}
    samplewise = {"FlowMOP": flowmop_values}
    for expert, identifier in identifiers.items():
        strategies[expert] = {}
        for source_name, display_name in (("groupwise", "Groupwise"), ("samplewise", "Individual")):
            rows = human[
                human.iloc[:, 0].str.contains(f"_{identifier}_{source_name}", regex=False)
            ].sort_values("sample")
            values = (100 - rows["1 | Freq. of Parent (%)"]).to_numpy(float)
            if len(values) != 3:
                raise RuntimeError(f"expected three {source_name} values for {expert}")
            strategies[expert][display_name] = values
        samplewise[expert] = strategies[expert]["Individual"]

    raw_p = ttest_rel(samplewise["FlowMOP"], samplewise["Expert 4"]).pvalue
    adjusted_p = min(1.0, raw_p * 4)
    return samplewise, strategies, adjusted_p


def bracket_label(value: float) -> str:
    if value < 0.001:
        return "p < .001"
    return f"p = {value:.3f}".replace("0.", ".")


def figure2_bracket_label(value: float) -> str:
    """Match the rounding used in the existing Figure 2 annotations."""
    if value < 0.001:
        return "p < .001"
    if value < 0.01:
        return f"p = {value:.3f}".replace("0.", ".")
    return f"p = {round(value, 2):.3f}".replace("0.", ".")


def generate_figure2_statistics(output: Path) -> None:
    """Recompose Figure 2B as a readable two-by-two panel."""
    data = pd.read_csv(ANALYSIS / "fig_2_time_corrected_data.csv")
    tests = pd.read_csv(ANALYSIS / "fig_2_time_corrected_paired_tests.csv")
    methods = ("flowmop", "peacoqc", "flowcut")
    method_labels = ("FlowMOP", "PeacoQC", "FlowCut")
    mixes = ("Segment", "Bimix", "Trimix")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "svg.fonttype": "none",
            "svg.hashsalt": "flowmop-figure-2b",
            "savefig.facecolor": "white",
        }
    )
    # The taller canvas provides enough physical headroom for the two stacked
    # significance brackets without letting their labels collide with the
    # two-line panel titles after the SVG is scaled in the manuscript.
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.8), sharey=True)
    figure.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.22,
        top=0.90,
        hspace=0.52,
        wspace=0.16,
    )
    specifications = (
        (5000, "retained_target_purity", "Sensitivity"),
        (5000, "removed_nontarget_purity", "Specificity"),
        (2000, "retained_target_purity", "Sensitivity"),
        (2000, "removed_nontarget_purity", "Specificity"),
    )
    offsets = {"flowmop": -0.27, "peacoqc": 0.0, "flowcut": 0.27}
    for axis_index, (axis, (bin_size, metric, title)) in enumerate(
        zip(axes.ravel(), specifications)
    ):
        subset = data[data["synthetic_bin_size"] == bin_size]
        for mix_index, mix in enumerate(mixes):
            for method in methods:
                values = subset[
                    (subset["mix_method"] == mix)
                    & (subset["algorithm"] == method)
                ][metric].dropna().to_numpy(float)
                position = mix_index + offsets[method]
                violins = axis.violinplot(
                    [values],
                    positions=[position],
                    widths=0.24,
                    showmeans=False,
                    showmedians=False,
                    showextrema=False,
                    points=100,
                )
                body = violins["bodies"][0]
                body.set_facecolor(FIGURE2_METHOD_COLOURS[method])
                body.set_edgecolor("#555555")
                body.set_linewidth(0.7)
                body.set_alpha(1)
                quartiles = np.quantile(values, (0.25, 0.5, 0.75))
                axis.hlines(
                    quartiles,
                    position - 0.09,
                    position + 0.09,
                    colors="#555555",
                    linewidths=(0.55, 0.8, 0.55),
                    linestyles=(":", "-", ":"),
                )
        axis.set_title(
            f"{title}\nBin size {bin_size}", fontweight="bold", pad=5
        )
        axis.set_xlabel("")
        axis.set_xticks(range(len(mixes)), mixes)
        axis.set_ylabel(
            "Proportion" if axis_index % 2 == 0 else "", fontweight="bold"
        )
        axis.set_ylim(0, 1.30)
        axis.set_yticks((0, 0.25, 0.50, 0.75, 1.00))
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(width=0.7, length=3)

        significant = tests[
            (tests["synthetic_bin_size"] == bin_size)
            & (tests["metric"] == metric)
            & (tests["p_value_bonferroni"] < 0.05)
        ]
        for mix_index, mix in enumerate(mixes):
            comparisons = significant[significant["mix_method"] == mix]
            for level, comparison in enumerate(
                comparisons.itertuples(index=False)
            ):
                x1 = mix_index + offsets[comparison.method_a]
                x2 = mix_index + offsets[comparison.method_b]
                y = 1.015 + 0.09 * level
                axis.plot(
                    (x1, x1, x2, x2),
                    (y, y + 0.025, y + 0.025, y),
                    color="#222222",
                    lw=0.8,
                    clip_on=False,
                )
                axis.text(
                    (x1 + x2) / 2,
                    y + 0.035,
                    figure2_bracket_label(
                        float(comparison.p_value_bonferroni)
                    ),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    clip_on=False,
                )

    handles = [
        Patch(
            facecolor=FIGURE2_METHOD_COLOURS[name], edgecolor="#555555"
        )
        for name in methods
    ]
    figure.legend(
        handles,
        method_labels,
        title="Cleaning method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        frameon=False,
        title_fontsize=14,
    )
    for axis in axes[0]:
        axis.tick_params(labelbottom=False)
    for axis in axes[1]:
        axis.set_xlabel("Mix method", fontweight="bold", labelpad=8)
    figure.text(0.008, 0.975, "B)", fontsize=16, va="top")
    figure.savefig(output, format="svg", metadata={"Date": None})
    plt.close(figure)


def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, value: float) -> None:
    height = 0.8
    ax.plot((x1, x1, x2, x2), (y, y + height, y + height, y), color="#333333", lw=1.25, clip_on=False)
    ax.text(
        (x1 + x2) / 2,
        y + height + 0.25,
        bracket_label(value),
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        clip_on=False,
    )


def generate_figure3_statistics(output: Path) -> None:
    samplewise, strategies, adjusted_p = parse_figure3_data()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "svg.fonttype": "none",
            "svg.hashsalt": "flowmop-figure-3",
            "savefig.facecolor": "white",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    # Use more of the fixed-height canvas for the actual C/D plotting areas.
    # This keeps the combined panel page-efficient while giving the charts the
    # same visual weight as the statistical panels in Figure 4.
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.28, top=0.85, wspace=0.30)

    methods = tuple(samplewise)
    x = np.arange(len(methods), dtype=float)
    means = [samplewise[method].mean() for method in methods]
    errors = [samplewise[method].std(ddof=1) for method in methods]
    colours = [METHOD_COLOURS["FlowMOP"]] + [EXPERT_COLOURS[name] for name in methods[1:]]
    axes[0].bar(x, means, yerr=errors, capsize=4, width=0.72, color=colours, edgecolor="#333333", linewidth=0.7)
    jitter = np.array((-0.07, 0.0, 0.07))
    for index, method in enumerate(methods):
        axes[0].scatter(
            index + jitter,
            samplewise[method],
            s=28,
            facecolors="white",
            edgecolors="#222222",
            linewidths=0.8,
            zorder=3,
        )
    add_bracket(axes[0], 0, 4, 82.0, adjusted_p)
    axes[0].set_ylim(0, 88)
    axes[0].set_xticks(x, methods, rotation=45, ha="right")
    axes[0].set_ylabel("Low-debris proportion (%)", fontweight="bold")
    axes[0].set_title("Cleanup-method comparison", fontweight="bold", pad=10)

    experts = tuple(strategies)
    x = np.arange(len(experts), dtype=float)
    width = 0.34
    for offset, strategy, hatch in ((-width / 2, "Groupwise", None), (width / 2, "Individual", "////")):
        values = [strategies[expert][strategy] for expert in experts]
        axes[1].bar(
            x + offset,
            [value.mean() for value in values],
            yerr=[value.std(ddof=1) for value in values],
            capsize=3,
            width=width,
            color=[EXPERT_COLOURS[expert] for expert in experts],
            edgecolor="#333333",
            linewidth=0.7,
            hatch=hatch,
        )
        for index, value in enumerate(values):
            axes[1].scatter(
                index + offset + jitter * 0.55,
                value,
                s=25,
                facecolors="white" if strategy == "Groupwise" else "#222222",
                edgecolors="#222222",
                linewidths=0.7,
                zorder=3,
            )
    axes[1].set_ylim(0, 82)
    axes[1].set_xticks(x, experts, rotation=38, ha="right")
    axes[1].set_ylabel("Low-debris proportion (%)", fontweight="bold")
    axes[1].set_title("Gating-strategy comparison", fontweight="bold", pad=10)
    axes[1].legend(
        handles=(
            Patch(facecolor="white", edgecolor="#333333", label="Groupwise"),
            Patch(facecolor="white", edgecolor="#333333", hatch="////", label="Individual sample"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=13,
    )

    fig.text(0.012, 0.945, "C)", fontsize=18, fontweight="bold", va="top")
    fig.text(0.512, 0.945, "D)", fontsize=18, fontweight="bold", va="top")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(0.8)
        axis.tick_params(width=0.8)
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.7)
        axis.set_axisbelow(True)

    fig.savefig(output, format="svg", metadata={"Date": None})
    plt.close(fig)


def main() -> int:
    crop_panel(
        HERE / "figure_2.svg",
        HERE / "figure_2_panel_a.svg",
        figure=2,
        crop=(0, 0, 720, 430),
        covers=(
            (0, 318, 80, 76),
            (42, 214, 24, 146),
            (82, 360, 130, 28),
            (216, 45, 15, 205),
            (376, 45, 15, 205),
            (537, 45, 15, 205),
            (216, 214, 15, 146),
            (376, 214, 15, 146),
            (537, 214, 15, 146),
        ),
        vertical_spines=(
            (231, 32, 171),
            (391, 32, 171),
            (552, 35, 174),
            (231, 217, 356),
            (391, 219, 358),
            (552, 219, 358),
        ),
        axis=((28, 369), "Time", "CD3", 30, 10.5),
    )
    generate_figure2_statistics(HERE / "figure_2_panel_b.svg")
    crop_panel(
        HERE / "figure_3.svg",
        HERE / "figure_3_panel_a.svg",
        figure=3,
        crop=(0, 0, 500, 220),
        covers=(
            (0, 158, 70, 87),
            (40, 35, 32, 125),
            (65, 164, 157, 50),
        ),
        axis=((55, 176), "FE2-A", "SE2-A", 48, 10.5, 1.2),
        text_y_shifts={"100% of sample": 1000, "50% of sample": 1000},
        labels=(
            (136, 205, "100% of sample", 11.5),
            (288, 205, "50% of sample", 11.5),
            (439, 205, "50% of sample", 11.5),
        ),
    )
    crop_panel(
        HERE / "figure_3.svg",
        HERE / "figure_3_panel_b.svg",
        figure=3,
        crop=(0, 315, 500, 415),
        covers=(
            (0, 322, 70, 93),
            (40, 222, 32, 102),
            (65, 354, 148, 31),
            (65, 385, 25, 25),
        ),
        axis=((55, 362), "FE2-A", "SE2-A", 48, 10.5, 1.2),
        text_y_shifts={
            "38% retained": 1000,
            "29% of 38%": 1000,
            "71% of 38%": 1000,
            "37% retained": 1000,
            "27% of 37%": 1000,
            "73% of 37%": 1000,
        },
        slices=((315, 540, 0), (540, 750, 205)),
        labels=(
            (136, 197, "38% retained", 11.5),
            (288, 197, "29% of 38%", 11.5),
            (439, 197, "71% of 38%", 11.5),
            (136, 397, "37% retained", 11.5),
            (288, 397, "27% of 37%", 11.5),
            (439, 397, "73% of 37%", 11.5),
        ),
    )
    generate_figure3_statistics(HERE / "figure_3_panel_cd.svg")
    print("Generated harmonized Figure 2 and Figure 3 SVG panels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
