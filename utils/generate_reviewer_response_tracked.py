#!/usr/bin/env python3
"""Generate a readable word-level tracked response from a Git baseline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BLUE_OPEN = '<span style="color:#0066cc">'
RED_OPEN = '<span style="color:#c00000"><s>'
RED_CLOSE = '</s></span>'
SPAN_CLOSE = '</span>'


def tracked_markdown(base: str, source: Path) -> str:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--word-diff=porcelain",
            "--unified=100000",
            base,
            "--",
            str(source),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output: list[str] = []
    in_hunk = False
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line == "~":
            output.append("\n")
        elif line.startswith(" "):
            output.append(line[1:])
        elif line.startswith("-"):
            output.extend((RED_OPEN, line[1:], RED_CLOSE))
        elif line.startswith("+"):
            output.extend((BLUE_OPEN, line[1:], SPAN_CLOSE))
        elif line.startswith("\\ No newline at end of file"):
            continue
        else:
            raise RuntimeError(f"Unexpected word-diff record: {line!r}")
    if not output:
        raise RuntimeError("Git produced no full-file word diff")
    return "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Git revision used as the old version")
    parser.add_argument(
        "--source", default="reviewer_response_point_by_point.md", type=Path
    )
    parser.add_argument(
        "--output", default="reviewer_response_point_by_point_tracked.md", type=Path
    )
    parser.add_argument(
        "--baseline-label",
        default=(
            "repository state on 24 August 2026; latest commit then was "
            "7cb7dab from 19 August 2026"
        ),
    )
    args = parser.parse_args()

    content = tracked_markdown(args.base, args.source)
    first_newline = content.find("\n")
    if first_newline < 0:
        raise RuntimeError("Tracked document has no title line")
    key = (
        f"\n\n**Tracked against `{args.base}` ({args.baseline_label}).** "
        f"{BLUE_OPEN}Blue text was added.{SPAN_CLOSE} "
        f"{RED_OPEN}Red struck-through text was removed.{RED_CLOSE}\n"
    )
    content = content[:first_newline] + key + content[first_newline:]
    args.output.write_text(content, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
