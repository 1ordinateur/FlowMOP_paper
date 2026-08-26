#!/usr/bin/env python3
"""Harmonize Figure 4 typography with Figures 5 and 6.

The original Figure 4 SVG remains the source of the density plots and chart
geometry. This script changes only presentation: panel labels, headings,
external arrow-axis labels, and the two chart y-axis titles.
"""

from pathlib import Path
import base64
import copy
import io
import xml.etree.ElementTree as ET

from PIL import Image


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)

FIGURES_DIR = Path(__file__).resolve().parents[1]
SOURCE = FIGURES_DIR / "figure_4.svg"
OUTPUT = FIGURES_DIR / "figure_4_harmonized.svg"
PANEL_A_OUTPUT = FIGURES_DIR / "figure_4_panel_a.svg"
PANEL_BC_OUTPUT = FIGURES_DIR / "figure_4_panel_bc.svg"
FONT = "DejaVu Sans"


def set_text_style(element: ET.Element, size: float, weight: int) -> None:
    element.set("font-family", f"{FONT},sans-serif")
    element.set("font-size", f"{size:g}")
    element.set("font-weight", str(weight))


tree = ET.parse(SOURCE)
root = tree.getroot()
texts = [element for element in root.iter() if element.tag.endswith("text")]
density_hrefs: dict[str, str] = {}

# The first nested group is panel B. Compress it vertically about its top edge
# so the category labels remain within the one-page figure without reducing
# the size of panels A or C.
main_group = next(element for element in root if element.tag.endswith("g"))
panel_b_group = list(main_group)[1]
panel_b_group.set("transform", "translate(0 594) scale(1 0.78) translate(0 -594)")

# The plot exports include their original ticked axes within each PNG tile.
# Crop to the interior plotting rectangle, then stretch that interior back to
# the existing tile box. This removes only axes/ticks and keeps the density
# layer and quadrant annotations at their intended panel size.
for image in [element for element in root.iter() if element.tag.endswith("image")]:
    image_id = image.get("id", "")
    if not image_id.startswith("img") or image_id in {"img52", "img54", "img56", "img58", "img60"}:
        continue
    href = image.get(f"{{{XLINK_NS}}}href", "")
    if not href.startswith("data:image/png;base64,"):
        continue
    payload = base64.b64decode(href.split(",", 1)[1])
    source_image = Image.open(io.BytesIO(payload)).convert("RGBA")
    width, height = source_image.size
    cropped = source_image.crop((round(width * 0.20), 0, width, round(height * 0.82)))
    if image_id in {"img2", "img5", "img8", "img11", "img14"}:
        # Quadrant values are redrawn below as SVG text. Remove only their
        # neutral-colour glyph pixels from the corner regions, preserving the
        # coloured density pixels beneath them.
        pixels = cropped.load()
        cropped_width, cropped_height = cropped.size
        for pixel_y in range(cropped_height):
            for pixel_x in range(cropped_width):
                in_corner = (
                    (pixel_x < 34 or pixel_x >= cropped_width - 34)
                    and (pixel_y < 15 or pixel_y >= cropped_height - 18)
                )
                if not in_corner:
                    continue
                red, green, blue, alpha = pixels[pixel_x, pixel_y]
                if alpha > 0 and max(red, green, blue) - min(red, green, blue) < 22 and max(red, green, blue) < 235:
                    pixels[pixel_x, pixel_y] = (255, 255, 255, 255)
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG", optimize=True)
    image.set(
        f"{{{XLINK_NS}}}href",
        "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
    )
    density_hrefs[image_id] = image.get(f"{{{XLINK_NS}}}href", "")

# Replace the PowerPoint-exported bitmap hatch tiles with native SVG patterns.
# The five colours correspond to the four expert bars and the legend swatch.
hatch_colours = {
    "pattern53": "#55a868",
    "pattern55": "#c44e52",
    "pattern57": "#8172b2",
    "pattern59": "#e69500",
    "pattern61": "#b8b8b8",
}
for pattern in [element for element in root.iter() if element.tag.endswith("pattern")]:
    pattern_id = pattern.get("id")
    if pattern_id not in hatch_colours:
        continue
    pattern.clear()
    pattern.attrib.update(
        {
            "id": pattern_id,
            "width": "8",
            "height": "8",
            "patternUnits": "userSpaceOnUse",
        }
    )
    ET.SubElement(
        pattern,
        f"{{{SVG_NS}}}rect",
        {"x": "0", "y": "0", "width": "8", "height": "8", "fill": hatch_colours[pattern_id]},
    )
    ET.SubElement(
        pattern,
        f"{{{SVG_NS}}}path",
        {
            "d": "M-2 2L2-2M0 8L8 0M6 10L10 6",
            "fill": "none",
            "stroke": "#202020",
            "stroke-width": "0.8",
        },
    )

parent_map = {child: parent for parent in root.iter() for child in parent}
for image in [element for element in root.iter() if element.tag.endswith("image")]:
    if image.get("id") in {"img52", "img54", "img56", "img58", "img60"}:
        parent_map[image].remove(image)

for index, element in enumerate(texts):
    label = "".join(element.itertext()).strip()

    if label == "A)":
        set_text_style(element, 10, 700)
    elif label in {"B)", "C)"}:
        # Panels B--C are displayed at 62% of the width used for panel A.
        set_text_style(element, 16.8, 700)
        if label == "B)":
            # Panel A and panel B are placed from the same manuscript inset.
            # Give their labels the same source-canvas x coordinate as well.
            element.set("transform", "translate(6 598)")
    elif label in {"CTV", "CFSE", "Mixed", "Expert", "FlowMOP"} and index <= 11:
        set_text_style(element, 10, 700)
    elif index in range(82, 96):
        # Arrow-axis labels for the three representative-plot rows.
        set_text_style(element, 10, 700)

title_replacements = {
    # Match the terminology used in Figure 3C and 3D.
    "Doublet Gating": ("Cleanup-method comparison", 165, 597),
    "Gating Grouping": ("Gating-strategy comparison", 535, 597),
}
for element in texts:
    label = "".join(element.itertext()).strip()
    if label in title_replacements:
        replacement, x, y = title_replacements[label]
        for child in list(element):
            element.remove(child)
        element.text = replacement
        set_text_style(element, 11.5, 700)
        element.set("text-anchor", "middle")
        element.set("transform", f"translate({x} {y})")
    elif label in {"Method Comparison", "Strategy Comparison"}:
        element.set("display", "none")
    elif label == "p = 0.009":
        set_text_style(element, 9, 400)
        element.set("transform", "translate(620.63 619)")

for element in root.iter():
    if not element.tag.endswith("path"):
        continue
    path_data = element.get("d", "")
    if path_data.startswith("M630 554C630 551.239 630 549"):
        element.set(
            "d",
            "M630 629C630 626.239 630 624 630 624L664 624"
            "C664 624 664 626.239 664 629",
        )

# The original chart y-axis titles are outlined glyphs rather than SVG text.
# Mask only those narrow title strips, leaving the tick labels and axes intact,
# then redraw the titles using the manuscript figure typography.
for x, y, width, height in ((15, 650, 39, 270), (260, 625, 70, 215)):
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "fill": "#ffffff"},
    )

axis_titles = (
    (32, 756, "Doublets Removed (%)", 14),
    (276, 731, "Remaining Doublets (%)", 17),
)
for x, y, label, font_size in axis_titles:
    element = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": str(y),
            "transform": f"rotate(-90 {x} {y})",
            "text-anchor": "middle",
            "font-family": f"{FONT},sans-serif",
            "font-size": str(font_size),
            "font-weight": "700",
        },
    )
    element.text = label

for y, label in ((638, "0.5"), (677, "0.4"), (715, "0.3"), (754, "0.2"), (792, "0.1"), (831, "0.0")):
    tick = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": "326",
            "y": str(y),
            "text-anchor": "end",
            "font-family": f"{FONT},sans-serif",
            "font-size": "10.5",
            "font-weight": "400",
        },
    )
    tick.text = label

# Harmonize the panel C x-axis title while retaining its legend immediately
# below. The narrow mask does not touch the category tick labels.
ET.SubElement(
    root,
    f"{{{SVG_NS}}}rect",
    {"x": "440", "y": "900", "width": "140", "height": "31", "fill": "#ffffff"},
)
method_title = ET.SubElement(
    root,
    f"{{{SVG_NS}}}text",
    {
        "x": "510",
        "y": "920",
        "text-anchor": "middle",
        "font-family": f"{FONT},sans-serif",
        "font-size": "15.2",
        "font-weight": "700",
    },
)
method_title.text = "Method"

# Replace the original panel-A presentation shell with the cropped density
# tiles. This clean overlay removes PowerPoint's native axes and tick labels in
# one operation and avoids masking any plot content.
def white_rect(x: float, y: float, width: float, height: float) -> None:
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "fill": "#ffffff"},
    )


# The PowerPoint chart exports panel B's category labels as slanted outline
# paths. Replace them with regular DejaVu Sans text, matching the category
# labels used in the other manuscript figures.
# Rebuild panel B's axes as native SVG. The earlier PowerPoint paths were
# partially lost when the category-label strip was cleaned, leaving both axes
# looking clipped. The replacement spans the complete 0--100 range and keeps
# all typography as selectable vector text.
white_rect(0, 610, 94, 244)
white_rect(45, 838, 225, 122)

b_left, b_right, b_top, b_bottom = 75.0, 253.0, 623.0, 837.0
for path_data in (
    f"M{b_left:g} {b_top:g}V{b_bottom:g}",
    f"M{b_left:g} {b_bottom:g}H{b_right:g}",
):
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": path_data,
            "fill": "none",
            "stroke": "#262626",
            "stroke-width": "1.1",
            "stroke-linecap": "square",
        },
    )

for value in range(0, 101, 20):
    tick_y = b_bottom - (value / 100) * (b_bottom - b_top)
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": f"M{b_left - 3.5:g} {tick_y:g}H{b_left:g}",
            "fill": "none",
            "stroke": "#262626",
            "stroke-width": "0.8",
        },
    )
    tick_label = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(b_left - 7),
            "y": str(tick_y + 3.2),
            "text-anchor": "end",
            "font-family": f"{FONT},sans-serif",
            "font-size": "10.5",
            "font-style": "normal",
            "font-weight": "400",
        },
    )
    tick_label.text = str(value)

b_y_title = ET.SubElement(
    root,
    f"{{{SVG_NS}}}text",
    {
        "x": "32",
        "y": str((b_top + b_bottom) / 2),
        "transform": f"rotate(-90 32 {(b_top + b_bottom) / 2:g})",
        "text-anchor": "middle",
        "font-family": f"{FONT},sans-serif",
        "font-size": "11.5",
        "font-style": "normal",
        "font-weight": "700",
    },
)
b_y_title.text = "Doublets Removed (%)"

# Move panel C's y-axis title towards its own plotting area. Rebuilding this
# axis also prevents the old outlined tick labels from being partially masked.
white_rect(258, 610, 77, 244)
# Clear the complete original x-axis/category-label strip, including its
# baseline, ticks, outlined-label fragments, and antialiased end-cap. The
# intended baseline and labels are redrawn below; no x ticks are used, matching
# panel B and avoiding ambiguous dot/dash remnants beneath the bars.
white_rect(317, 827, 413, 84)
# The exported chart's true zero line is at y=827.  Using the lower edge of
# the surrounding chart box (y=837) created a second baseline beneath the
# bars, making the panel appear to float.  Rebuild the axis at the data
# baseline and extend it left to meet the y-axis.
c_left, c_right, c_top, c_bottom = 318.0, 705.0, 623.0, 827.0
for path_data in (
    f"M{c_left:g} {c_top:g}V{c_bottom:g}",
    f"M{c_left:g} {c_bottom:g}H{c_right:g}",
):
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": path_data,
            "fill": "none",
            "stroke": "#262626",
            "stroke-width": "1.1",
            "stroke-linecap": "square",
        },
    )

for step in range(6):
    value = step / 10
    tick_y = c_bottom - (value / 0.5) * (c_bottom - c_top)
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": f"M{c_left - 3.5:g} {tick_y:g}H{c_left:g}",
            "fill": "none",
            "stroke": "#262626",
            "stroke-width": "0.8",
        },
    )
    tick_label = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(c_left - 7),
            "y": str(tick_y + 3.2),
            "text-anchor": "end",
            "font-family": f"{FONT},sans-serif",
            "font-size": "10.5",
            "font-style": "normal",
            "font-weight": "400",
        },
    )
    tick_label.text = f"{value:.1f}"

c_y_title = ET.SubElement(
    root,
    f"{{{SVG_NS}}}text",
    {
        "x": "276",
        "y": "725",
        "transform": "rotate(-90 276 725)",
        "text-anchor": "middle",
        "font-family": f"{FONT},sans-serif",
        "font-size": "11.5",
        "font-style": "normal",
        "font-weight": "700",
    },
)
c_y_title.text = "Remaining Doublets (%)"

# The exported bar centres are 109, 141, 173, 205, and 237.  Anchor each
# category label to its corresponding centre rather than the left-shifted
# coordinates inherited from the earlier composition.
for x, label in zip((109, 141, 173, 205, 237), ("FlowMOP", "Expert 1", "Expert 2", "Expert 3", "Expert 4")):
    category = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": "858",
            "transform": f"rotate(-45 {x} 858)",
            "text-anchor": "end",
            "font-family": f"{FONT},sans-serif",
            "font-size": "10.5",
            "font-style": "normal",
            "font-weight": "400",
        },
    )
    category.text = label

# Redraw panel C's category labels with exactly the same SVG styling and
# baseline used for panel B.
for x, label in zip((406, 494, 581, 669), ("Expert 1", "Expert 2", "Expert 3", "Expert 4")):
    category = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": "858",
            "transform": f"rotate(-45 {x} 858)",
            "text-anchor": "end",
            "font-family": f"{FONT},sans-serif",
            "font-size": "10.5",
            "font-style": "normal",
            "font-weight": "400",
        },
    )
    category.text = label

# Draw the chart spines last so none of the category-label masks can interrupt
# the baselines. This keeps both lower panels visually grounded and aligned.
for path_data in (
    f"M{b_left:g} {b_top:g}V{b_bottom:g}H{b_right:g}",
    f"M{c_left:g} {c_top:g}V{c_bottom:g}H{c_right:g}",
):
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": path_data,
            "fill": "none",
            "stroke": "#262626",
            "stroke-width": "1.1",
            "stroke-linecap": "square",
            "stroke-linejoin": "miter",
        },
    )

# Replace the inherited outlined legend with compact native SVG elements.
# This prevents its glyphs from interacting with the crop boundary and keeps
# its typography consistent with the later biological-validation figures.
white_rect(400, 890, 230, 72)
legend_title = ET.SubElement(
    root,
    f"{{{SVG_NS}}}text",
    {
        "x": "510",
        "y": "909",
        "text-anchor": "middle",
        "font-family": f"{FONT},sans-serif",
        "font-size": "11.5",
        "font-weight": "700",
    },
)
legend_title.text = "Method"

for x, fill, label in (
    (444, "#ffffff", "Group"),
    (520, "url(#pattern61)", "Sample"),
):
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}rect",
        {
            "x": str(x),
            "y": "920",
            "width": "24",
            "height": "10",
            "fill": fill,
            "stroke": "#262626",
            "stroke-width": "0.8",
        },
    )
    legend_label = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x + 31),
            "y": "929",
            "font-family": f"{FONT},sans-serif",
            "font-size": "9.5",
            "font-weight": "400",
        },
    )
    legend_label.text = label


white_rect(0, 0, 720, 575)

panel_a = ET.SubElement(
    root,
    f"{{{SVG_NS}}}text",
    {
        "x": "6",
        "y": "31",
        "font-family": f"{FONT},sans-serif",
        "font-size": "15.5",
        "font-weight": "700",
    },
)
panel_a.text = "A)"

for x, title in zip((107, 239, 371, 503, 635), ("CTV", "CFSE", "Mixed", "Expert", "FlowMOP")):
    heading = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(x),
            "y": "30",
            "text-anchor": "middle",
            "font-family": f"{FONT},sans-serif",
            "font-size": "9.5",
            "font-weight": "700",
        },
    )
    heading.text = title

plot_ids = (
    ("img2", "img5", "img8", "img11", "img14"),
    ("img17", "img20", "img23", "img26", "img29"),
    ("img32", "img35", "img38", "img41", "img44"),
)
quadrant_values = {
    "img2": ("0", "0.032", "1.85", "98.1"),
    "img5": ("95.4", "6.00e-3", "4.63", "0"),
    "img8": ("44.4", "6.80", "3.48", "45.3"),
    "img11": ("47.8", "0.19", "3.58", "48.5"),
    "img14": ("47.4", "0.26", "3.90", "48.4"),
}
for y, row_ids in zip((39, 185, 331), plot_ids):
    for x, image_id in zip((47, 179, 311, 443, 575), row_ids):
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}image",
            {
                "x": str(x),
                "y": str(y),
                "width": "120",
                "height": "120",
                "preserveAspectRatio": "none",
                f"{{{XLINK_NS}}}href": density_hrefs[image_id],
            },
        )
        if image_id in quadrant_values:
            top_left, top_right, bottom_left, bottom_right = quadrant_values[image_id]
            for text_x, text_y, anchor, value in (
                (x + 3, y + 8, "start", top_left),
                (x + 117, y + 8, "end", top_right),
                (x + 3, y + 116, "start", bottom_left),
                (x + 117, y + 116, "end", bottom_right),
            ):
                value_text = ET.SubElement(
                    root,
                    f"{{{SVG_NS}}}text",
                    {
                        "x": str(text_x),
                        "y": str(text_y),
                        "text-anchor": anchor,
                        "font-family": f"{FONT},sans-serif",
                        "font-size": "5.5",
                        "font-weight": "400",
                    },
                )
                value_text.text = value
        ET.SubElement(
            root,
            f"{{{SVG_NS}}}rect",
            {
                "x": str(x),
                "y": str(y),
                "width": "120",
                "height": "120",
                "fill": "none",
                "stroke": "#777777",
                "stroke-width": "0.45",
            },
        )

# Redraw only the shared directional axes used by Figures 5 and 6. The
# geometry below is the same relative geometry used by those figure scripts:
# the origin sits 7% left and 10% below the plotting rectangle.
def directional_axis(plot_y: float, x_label: str, y_label: str) -> None:
    colour = "#111111"
    origin_x = 47 - 0.07 * 120
    origin_y = plot_y + 1.10 * 120
    horizontal_end = 47 + 0.48 * 120
    vertical_end = plot_y + (1 - 0.45) * 120
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": f"M{origin_x:g} {origin_y:g}H{horizontal_end:g}",
            "stroke": colour,
            "stroke-width": "1.4",
            "stroke-linecap": "round",
            "fill": "none",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": (
                f"M{horizontal_end - 5.2:g} {origin_y - 2.6:g}"
                f"L{horizontal_end:g} {origin_y:g}"
                f"L{horizontal_end - 5.2:g} {origin_y + 2.6:g}z"
            ),
            "fill": colour,
            "stroke": colour,
            "stroke-width": "1.4",
            "stroke-linecap": "round",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": f"M{origin_x:g} {origin_y:g}V{vertical_end:g}",
            "stroke": colour,
            "stroke-width": "1.4",
            "stroke-linecap": "round",
            "fill": "none",
        },
    )
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}path",
        {
            "d": (
                f"M{origin_x - 2.6:g} {vertical_end + 5.2:g}"
                f"L{origin_x:g} {vertical_end:g}"
                f"L{origin_x + 2.6:g} {vertical_end + 5.2:g}z"
            ),
            "fill": colour,
            "stroke": colour,
            "stroke-width": "1.4",
            "stroke-linecap": "round",
        },
    )
    horizontal = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(47 + 0.205 * 120),
            "y": str(plot_y + 1.19 * 120),
            "text-anchor": "middle",
            "font-family": f"{FONT},sans-serif",
            "font-size": "9.5",
            "font-weight": "700",
        },
    )
    horizontal.text = x_label
    vertical = ET.SubElement(
        root,
        f"{{{SVG_NS}}}text",
        {
            "x": str(47 - 0.16 * 120),
            "y": str(plot_y + (1 - 0.175) * 120),
            "transform": (
                f"rotate(-90 {47 - 0.16 * 120:g} "
                f"{plot_y + (1 - 0.175) * 120:g})"
            ),
            "text-anchor": "middle",
            "font-family": f"{FONT},sans-serif",
            "font-size": "9.5",
            "font-weight": "700",
        },
    )
    vertical.text = y_label


directional_axis(39, "CTV", "CFSE")
directional_axis(185, "FSC-A", "FSC-H")
directional_axis(331, "SSC-A", "SSC-H")

# Guarantee a clean lower crop edge. Some PDF viewers otherwise expose a
# sub-pixel remnant from content outside panel A's view box.
white_rect(0, 480, 720, 10)

tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)

# Export genuinely tight page-ready assets. LaTeX should not trim the master
# canvas a second time: that changes the apparent scale and can clip axes or
# labels. Panel A and the aligned B--C row are therefore written with their
# own exact view boxes and natural aspect ratios.
for output_path, view_box, width, height in (
    (PANEL_A_OUTPUT, "0 0 720 485", 720, 485),
    # Retain a full lower margin beneath the legend so its descenders and the
    # following manuscript caption remain visually separated at low zoom.
    (PANEL_BC_OUTPUT, "0 570 720 400", 720, 400),
):
    panel_root = copy.deepcopy(root)
    panel_root.set("width", str(width))
    panel_root.set("height", str(height))
    panel_root.set("viewBox", view_box)
    panel_root.set("overflow", "hidden")
    ET.ElementTree(panel_root).write(output_path, encoding="utf-8", xml_declaration=True)
