#!/usr/bin/env python3
"""Generate the revised expert-ranking figures directly as editable SVGs."""

from __future__ import annotations

import html
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cairosvg


FIGS_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
WORKBOOK = Path(__file__).with_name("ranking_analysis.xlsx")
MEDIA_DIR = REPO_ROOT / "FlowMOP_submission_media"
EXPORT_DIR = Path(__file__).with_name("svg_exports")

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
NS_PKG_REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"

METHOD_ORDER = [1, 2, 3, 4, 5, 6, 7]
DATASET_LABELS = {
    "Mouse DRG": "Mouse DRG",
    "Mouse Skin": "Mouse Skin",
    "Human T cell diff": "Human Cultured T Cells",
    "Human T Diff": "Human Cultured T Cells",
    "Mouse Bonemarrow": "Mouse Bone Marrow",
    "Mouse Spleen": "Mouse Spleen",
    "Mouse blood": "Mouse Blood",
    "Mouse Blood": "Mouse Blood",
    "Mouse Brain": "Mouse Brain",
    "Mouse CNS": "Mouse CNS",
    "Human Liver": "Human Liver",
}

FIGURES = {
    "Time": {
        "svg": FIGS_DIR / "figure_5.svg",
        "export": EXPORT_DIR / "time_ranking_panel.svg",
        "png": MEDIA_DIR / "image5_revised.png",
    },
    "Debris": {
        "svg": FIGS_DIR / "figure_6.svg",
        "export": EXPORT_DIR / "debris_ranking_panel.svg",
        "png": MEDIA_DIR / "image6_revised.png",
    },
    "Doublets": {
        "svg": FIGS_DIR / "figure_7.svg",
        "export": EXPORT_DIR / "doublets_ranking_panel.svg",
        "png": MEDIA_DIR / "image7_revised.png",
    },
}

RANK_COLOURS = [
    "#FECB33",  # 1, best
    "#F98E09",
    "#ED5A2A",
    "#C43757",
    "#8C2981",
    "#4A0C6B",
    "#140B34",  # 7, worst
]


def column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def read_workbook() -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(WORKBOOK) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(NS_MAIN + "t"))
            for item in shared_root.findall(NS_MAIN + "si")
        ]

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships.findall(NS_PKG_REL + "Relationship")
        }

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets: dict[str, list[list[str]]] = {}
        for sheet in workbook.findall(f"{NS_MAIN}sheets/{NS_MAIN}sheet"):
            name = sheet.attrib["name"]
            target = rel_targets[sheet.attrib[NS_REL + "id"]]
            path = target if target.startswith("xl/") else "xl/" + target
            root = ET.fromstring(archive.read(path))
            rows: list[list[str]] = []
            for row in root.iter(NS_MAIN + "row"):
                cells: dict[int, str] = {}
                for cell in row.findall(NS_MAIN + "c"):
                    value_node = cell.find(NS_MAIN + "v")
                    value = "" if value_node is None else value_node.text or ""
                    if cell.attrib.get("t") == "s" and value:
                        value = shared[int(value)]
                    cells[column_index(cell.attrib["r"])] = value
                if cells:
                    rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
            sheets[name] = rows
    return sheets


def parse_rankings(rows: list[list[str]]) -> tuple[list[str], dict[int, list[float | None]]]:
    datasets = rows[0][1:]
    max_rank = len(rows) - 1
    method_ids = sorted(
        {int(value) for row in rows[1:] for value in row[1:] if value != ""}
    )
    ranks = {method_id: [None] * len(datasets) for method_id in method_ids}

    for position, row in enumerate(rows[1:], start=1):
        for dataset_index, value in enumerate(row[1:]):
            if value:
                method_id = int(value)
                if ranks[method_id][dataset_index] is not None:
                    raise ValueError(f"Duplicate method {method_id} in {datasets[dataset_index]}")
                ranks[method_id][dataset_index] = float(position)

    for dataset_index, dataset in enumerate(datasets):
        observed = sorted(
            int(values[dataset_index])
            for values in ranks.values()
            if values[dataset_index] is not None
        )
        expected = list(range(1, len(observed) + 1))
        if observed != expected:
            raise ValueError(f"Invalid ranking positions for {dataset}: {observed}")

    if max(
        int(rank)
        for values in ranks.values()
        for rank in values
        if rank is not None
    ) != max_rank:
        raise ValueError("Maximum rank did not match the worksheet row count")
    return datasets, ranks


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def text_element(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 20,
    weight: int = 400,
    anchor: str = "middle",
    fill: str = "#222222",
    transform: str | None = None,
) -> str:
    transform_attr = f' transform="{transform}"' if transform else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}"{transform_attr}>{esc(value)}</text>'
    )


def generate_svg(
    analysis_name: str,
    datasets: list[str],
    ranks: dict[int, list[float | None]],
    method_names: dict[int, str],
) -> str:
    method_ids = [method_id for method_id in METHOD_ORDER if method_id in ranks]
    max_rank = max(int(rank) for values in ranks.values() for rank in values if rank is not None)

    width = 1500
    left = 240
    top = 210
    cell_width = 110
    cell_height = 58
    x_gap = 6
    y_gap = 7
    average_gap = 24
    average_width = 145
    grid_width = len(datasets) * (cell_width + x_gap) - x_gap
    average_x = left + grid_width + average_gap
    height = top + len(method_ids) * (cell_height + y_gap) + 120

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text_element(32, top + (len(method_ids) * (cell_height + y_gap)) / 2,
                     "Gate Provided By", size=24, weight=600,
                     transform=f"rotate(-90 32 {top + (len(method_ids) * (cell_height + y_gap)) / 2})"),
    ]

    for dataset_index, dataset in enumerate(datasets):
        x = left + dataset_index * (cell_width + x_gap) + cell_width / 2
        label = DATASET_LABELS.get(dataset, dataset)
        parts.append(
            text_element(
                x - 6,
                top - 24,
                label,
                size=19,
                anchor="start",
                transform=f"rotate(-43 {x - 6} {top - 24})",
            )
        )

    parts.extend([
        text_element(average_x + average_width / 2, top - 92, "Average", size=20, weight=600),
        text_element(average_x + average_width / 2, top - 66, "Score", size=20, weight=600),
        text_element(average_x + average_width / 2, top - 38, "(1 = best)", size=15, fill="#555555"),
    ])

    average_scores: dict[int, float] = {}
    for method_id in method_ids:
        observed = [rank for rank in ranks[method_id] if rank is not None]
        average_scores[method_id] = sum(observed) / len(observed)

    for row_index, method_id in enumerate(method_ids):
        y = top + row_index * (cell_height + y_gap)
        parts.append(
            text_element(left - 22, y + cell_height / 2 + 7, method_names[method_id],
                         size=21, anchor="end", weight=600 if method_id == 5 else 400)
        )

        for dataset_index, rank in enumerate(ranks[method_id]):
            x = left + dataset_index * (cell_width + x_gap)
            if rank is None:
                fill = "#ffffff"
                stroke = "#c8c8c8"
                label = "N/A"
                label_fill = "#888888"
            else:
                rank_int = int(rank)
                fill = RANK_COLOURS[rank_int - 1]
                stroke = "#ffffff"
                label = str(rank_int)
                label_fill = "#111111" if rank_int <= 3 else "#ffffff"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
            )
            parts.append(text_element(x + cell_width / 2, y + 38, label, size=23,
                                      weight=700, fill=label_fill))

        average = average_scores[method_id]
        parts.append(
            f'<rect x="{average_x}" y="{y}" width="{average_width}" height="{cell_height}" '
            'rx="2" fill="#e8edf2" stroke="#ffffff" stroke-width="3"/>'
        )
        parts.append(text_element(average_x + average_width / 2, y + 38,
                                  f"{average:.2f}", size=22, weight=700))

        if method_id == 5:
            border_x = left - 7
            border_width = average_x + average_width - border_x + 7
            parts.append(
                f'<rect x="{border_x}" y="{y - 5}" width="{border_width}" '
                f'height="{cell_height + 10}" fill="none" stroke="#111111" stroke-width="4"/>'
            )

    legend_y = top + len(method_ids) * (cell_height + y_gap) + 30
    legend_cell = 25
    legend_gap = 48
    legend_width = max_rank * legend_gap + 165
    legend_x = (width - legend_width) / 2
    parts.append(text_element(legend_x, legend_y + 20, "Rank", size=18, weight=600, anchor="start"))
    cursor = legend_x + 62
    for rank in range(1, max_rank + 1):
        parts.append(
            f'<rect x="{cursor}" y="{legend_y}" width="{legend_cell}" height="{legend_cell}" '
            f'fill="{RANK_COLOURS[rank - 1]}"/>'
        )
        label_fill = "#111111" if rank <= 3 else "#ffffff"
        parts.append(text_element(cursor + legend_cell / 2, legend_y + 19, str(rank),
                                  size=14, weight=700, fill=label_fill))
        cursor += legend_gap
    parts.append(text_element(legend_x + 62, legend_y + 50, "Best", size=16, anchor="start"))
    parts.append(text_element(cursor - legend_gap + legend_cell, legend_y + 50,
                              "Worst", size=16, anchor="end"))
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    sheets = read_workbook()
    method_names = {
        int(row[1]): row[0]
        for row in sheets["Cleaning Experts + Algo IDs"][1:]
        if len(row) > 1 and row[1]
    }
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    for analysis_name, destinations in FIGURES.items():
        datasets, ranks = parse_rankings(sheets[analysis_name])
        svg = generate_svg(analysis_name, datasets, ranks, method_names)
        destinations["svg"].write_text(svg, encoding="utf-8")
        shutil.copyfile(destinations["svg"], destinations["export"])
        cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            write_to=str(destinations["png"]),
            output_width=3000,
        )
        print(f"Generated {analysis_name}: {destinations['svg']} and {destinations['png']}")


if __name__ == "__main__":
    main()
