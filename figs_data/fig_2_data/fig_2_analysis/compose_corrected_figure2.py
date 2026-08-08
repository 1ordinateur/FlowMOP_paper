#!/usr/bin/env python3
"""Overlay corrected benchmark violins and annotations onto Figure 2.

Panel A and every non-benchmark element of the existing composite are retained
byte-for-byte. The script appends a white-backed replacement for only the four
violin interiors, the significance brackets, and the Sensitivity/Specificity
headings in panel B.
"""

from __future__ import annotations

import argparse
import copy
import html
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE)

TARGET_AXES = {
    "axes_1": (73.0, 554.0, 300.0, 172.0),
    "axes_2": (413.0, 554.0, 301.0, 172.0),
    "axes_3": (73.0, 783.0, 300.0, 170.0),
    "axes_4": (413.0, 783.0, 301.0, 170.0),
}

# Positions reproduce the existing manually composed Figure 2 layout.
ANNOTATION_POSITIONS = {
    (5000, "retained_target_purity", "Segment", "flowmop", "flowcut"):
        (98, 150, 548, 555, 543),
    (5000, "retained_target_purity", "Bimix", "flowmop", "flowcut"):
        (198, 251, 549, 555, 544),
    (5000, "retained_target_purity", "Bimix", "peacoqc", "flowcut"):
        (225, 249, 524, 530, 519),
    (5000, "removed_nontarget_purity", "Bimix", "flowmop", "peacoqc"):
        (536, 563, 551, 557, 545),
    (5000, "removed_nontarget_purity", "Bimix", "flowmop", "flowcut"):
        (536, 591, 526, 532, 520),
    (5000, "removed_nontarget_purity", "Trimix", "flowmop", "peacoqc"):
        (638, 665, 556, 561, 551),
    (5000, "removed_nontarget_purity", "Trimix", "flowmop", "flowcut"):
        (638, 692, 533, 538, 526),
    (2000, "retained_target_purity", "Segment", "flowmop", "flowcut"):
        (96, 149, 778, 784, 775),
    (2000, "retained_target_purity", "Segment", "peacoqc", "flowcut"):
        (124, 148, 758, 763, 753),
    (2000, "retained_target_purity", "Bimix", "peacoqc", "flowcut"):
        (225, 252, 779, 785, 773),
    (2000, "removed_nontarget_purity", "Bimix", "flowmop", "peacoqc"):
        (536, 563, 804, 810, 798),
    (2000, "removed_nontarget_purity", "Bimix", "peacoqc", "flowcut"):
        (567, 591, 781, 786, 774),
    (2000, "removed_nontarget_purity", "Trimix", "flowmop", "peacoqc"):
        (636, 663, 778, 784, 773),
    (2000, "removed_nontarget_purity", "Trimix", "peacoqc", "flowcut"):
        (665, 690, 801, 807, 796),
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-svg", type=Path, default=repo_root / "figs_data" / "figure_2.svg"
    )
    parser.add_argument(
        "--panel-svg", type=Path, default=here / "svg_exports" / "fig_2_time_panel.svg"
    )
    parser.add_argument(
        "--tests", type=Path, default=here / "fig_2_time_corrected_paired_tests.csv"
    )
    parser.add_argument(
        "--output-svg", type=Path, default=repo_root / "figs_data" / "figure_2.svg"
    )
    return parser.parse_args()


def find_by_id(root: ET.Element, element_id: str) -> ET.Element:
    for element in root.iter():
        if element.attrib.get("id") == element_id:
            return element
    raise KeyError(element_id)


def source_axis_rectangle(root: ET.Element, axis: ET.Element) -> tuple[float, ...]:
    collection = next(
        element
        for element in list(axis)
        if element.attrib.get("id", "").startswith("FillBetweenPolyCollection_")
    )
    clipped = next(
        element for element in collection.iter() if "clip-path" in element.attrib
    )
    match = re.fullmatch(r"url\(#(.+)\)", clipped.attrib["clip-path"])
    clip = find_by_id(root, match.group(1))
    rectangle = next(clip.iter(f"{{{SVG_NAMESPACE}}}rect"))
    return tuple(float(rectangle.attrib[name]) for name in ("x", "y", "width", "height"))


def clean_plot_element(element: ET.Element) -> ET.Element:
    cleaned = copy.deepcopy(element)
    for child in cleaned.iter():
        child.attrib.pop("clip-path", None)
    return cleaned


def plot_overlay(panel_svg: Path) -> str:
    root = ET.parse(panel_svg).getroot()
    pieces: list[str] = []
    for axis_id, target in TARGET_AXES.items():
        axis = find_by_id(root, axis_id)
        source = source_axis_rectangle(root, axis)
        sx = target[2] / source[2]
        sy = target[3] / source[3]
        tx = target[0] - sx * source[0]
        ty = target[1] - sy * source[1]
        children = [
            clean_plot_element(element)
            for element in list(axis)
            if element.attrib.get("id", "").startswith(
                ("FillBetweenPolyCollection_", "line2d_")
            )
        ]
        content = "".join(
            ET.tostring(element, encoding="unicode") for element in children
        )
        pieces.append(
            f'<g transform="matrix({sx:.12g} 0 0 {sy:.12g} '
            f'{tx:.12g} {ty:.12g})">{content}</g>'
        )
    return "".join(pieces)


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "p < 0.001"
    if value < 0.01:
        return f"p = {value:.3f}"
    return f"p = {value:.2f}"


def annotations(tests_path: Path) -> str:
    tests = pd.read_csv(tests_path)
    significant = tests[tests["p_value_bonferroni"] < 0.05].copy()
    if len(significant) != len(ANNOTATION_POSITIONS):
        raise RuntimeError(
            f"expected {len(ANNOTATION_POSITIONS)} significant tests, "
            f"found {len(significant)}"
        )
    pieces: list[str] = []
    for row in significant.itertuples(index=False):
        key = (
            int(row.synthetic_bin_size),
            row.metric,
            row.mix_method,
            row.method_a,
            row.method_b,
        )
        if key not in ANNOTATION_POSITIONS:
            raise RuntimeError(f"no layout position for significant test {key}")
        x1, x2, y1, y2, text_y = ANNOTATION_POSITIONS[key]
        label = html.escape(format_p_value(float(row.p_value_bonferroni)))
        pieces.append(
            f'<path d="M{x1} {y2} L{x1} {y1} L{x2} {y1} L{x2} {y2}" '
            'stroke="#000000" stroke-width="2" stroke-miterlimit="8" '
            'fill="none" fill-rule="evenodd"/>'
            f'<text x="{(x1 + x2) / 2:g}" y="{text_y}" text-anchor="middle" '
            'font-family="Aptos,Aptos_MSFontService,sans-serif" '
            f'font-weight="400" font-size="13.3333">{label}</text>'
        )
    return "".join(pieces)


def replacement_group(panel_svg: Path, tests_path: Path) -> str:
    whiteouts = "".join(
        [
            '<rect x="65" y="447" width="655" height="31" fill="#FFFFFF"/>',
            '<rect x="65" y="500" width="650" height="55" fill="#FFFFFF"/>',
            '<rect x="65" y="730" width="650" height="82" fill="#FFFFFF"/>',
            '<rect x="74" y="555" width="298" height="169" fill="#FFFFFF"/>',
            '<rect x="414" y="555" width="299" height="169" fill="#FFFFFF"/>',
            '<rect x="74" y="784" width="298" height="168" fill="#FFFFFF"/>',
            '<rect x="414" y="784" width="299" height="168" fill="#FFFFFF"/>',
        ]
    )
    headings = (
        '<text x="223" y="470" text-anchor="middle" '
        'font-family="Aptos,Aptos_MSFontService,sans-serif" '
        'font-weight="700" font-size="18.6667">Sensitivity</text>'
        '<text x="563" y="470" text-anchor="middle" '
        'font-family="Aptos,Aptos_MSFontService,sans-serif" '
        'font-weight="700" font-size="18.6667">Specificity</text>'
        '<text x="180.615" y="747" '
        'font-family="Aptos,Aptos_MSFontService,sans-serif" '
        'font-weight="700" font-size="16">Bin Size 2000</text>'
        '<text x="521.387" y="747" '
        'font-family="Aptos,Aptos_MSFontService,sans-serif" '
        'font-weight="700" font-size="16">Bin Size 2000</text>'
    )
    # The lower heading whiteout crosses the top of both lower plot areas.
    # Restore the original full y-axis spines and their 1.00 tick marks so the
    # axes remain visibly anchored to the upper limit.
    lower_y_axes = (
        '<path d="M72.9414 460.588 72.9414 289.708" '
        'stroke="#262626" stroke-width="1.04351" stroke-linecap="square" '
        'fill="none" transform="matrix(1 0 0 1.00016 0.0921262 492)"/>'
        '<path d="M0 0-5.00884 0" stroke="#262626" '
        'stroke-width="1.04351" stroke-linejoin="round" fill="none" '
        'transform="matrix(1 0 0 1.00016 73.0336 787.118)"/>'
        '<path d="M413.191 460.588 413.191 289.708" '
        'stroke="#262626" stroke-width="1.04351" stroke-linecap="square" '
        'fill="none" transform="matrix(1 0 0 1.00016 0.0921262 492)"/>'
        '<path d="M0 0-5.00884 0" stroke="#262626" '
        'stroke-width="1.04351" stroke-linejoin="round" fill="none" '
        'transform="matrix(1 0 0 1.00016 413.284 784.92)"/>'
    )
    return (
        '<!-- corrected-time-benchmark-overlay:start -->'
        '<g id="corrected-time-benchmark-overlay">'
        + whiteouts
        + headings
        + plot_overlay(panel_svg)
        + lower_y_axes
        + annotations(tests_path)
        + "</g>"
        '<!-- corrected-time-benchmark-overlay:end -->'
    )


def main() -> int:
    args = parse_args()
    source = args.base_svg.read_text(encoding="utf-8")
    source = re.sub(
        r'<!-- corrected-time-benchmark-overlay:start -->.*?'
        r'<!-- corrected-time-benchmark-overlay:end -->\s*',
        "",
        source,
        flags=re.DOTALL,
    )
    overlay = replacement_group(args.panel_svg, args.tests)
    if "</svg>" not in source:
        raise RuntimeError(f"not an SVG document: {args.base_svg}")
    result = source.replace("</svg>", overlay + "</svg>")
    args.output_svg.write_text(result, encoding="utf-8")
    print(f"Wrote {args.output_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
