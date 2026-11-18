#!/usr/bin/env python3
"""Give Figure 1 the shared bold subpanel labels.

The editable source of Figure 1 is not available; only the raster
FlowMOP_submission_media/image1.png exists. This script embeds that image
unchanged, covers its original regular-weight labels with white boxes, and
draws bold labels that render at the shared 9.2 pt size in the manuscript.
"""

from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "FlowMOP_submission_media" / "image1.png"
OUTPUT = HERE / "figure_1_labelled.svg"

PANEL_LABEL_PT = 9.2
# Figure 1 is height-limited in the manuscript: 1123 px wide image placed at
# 456.1 pt wide (\linewidth and 0.82\textheight, keepaspectratio).
PLACED_WIDTH_PT = 456.1
# Original label ink boxes in image pixels: (label, left, top, right, bottom).
LABELS = (
    ("A)", 16, 14, 35, 32),
    ("B)", 29, 433, 46, 451),
    ("C)", 17, 1085, 37, 1103),
)
COVER_MARGIN = 3


def main() -> int:
    width, height = Image.open(SOURCE).size
    font_size = round(PANEL_LABEL_PT * width / PLACED_WIDTH_PT, 2)
    encoded = base64.b64encode(SOURCE.read_bytes()).decode("ascii")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" version="1.1">',
        f'<image x="0" y="0" width="{width}" height="{height}" '
        f'xlink:href="data:image/png;base64,{encoded}"/>',
    ]
    for text, left, top, right, bottom in LABELS:
        parts.append(
            f'<rect x="{left - COVER_MARGIN}" y="{top - COVER_MARGIN}" '
            f'width="{right - left + 2 * COVER_MARGIN}" '
            f'height="{bottom - top + 2 * COVER_MARGIN}" fill="#FFFFFF"/>'
        )
        parts.append(
            f'<text x="{left}" y="{bottom + 0.5}" font-family="DejaVu Sans" '
            f'font-size="{font_size}" font-weight="700" fill="#000000">{text}</text>'
        )
    parts.append("</svg>\n")
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUTPUT.name} (label font-size {font_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
