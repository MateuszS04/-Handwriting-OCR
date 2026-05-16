"""Render a debug overlay of all segmented line polygons on each page.

Useful for *seeing* what the line segmenter is doing before deciding how to
post-process. Each polygon is drawn with a thin colored outline and the line
index is printed near its top-left corner; the baseline is drawn as a dashed
line if present.

Output: one PNG per page in ``--out-dir`` named ``<page_stem>_overlay.png``.

Reads the sidecar JSONs (``<page>.lines.json``) produced by
``src.segment.segment_lines``; the segmenter does not need to be re-run.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.utils.io import setup_logging

log = logging.getLogger(__name__)

# Cycle through a small palette so neighbors are visually distinct.
PALETTE = [
    (220, 20, 60),    # crimson
    (30, 144, 255),   # dodger blue
    (34, 139, 34),    # forest green
    (255, 140, 0),    # dark orange
    (148, 0, 211),    # dark violet
    (0, 139, 139),    # dark cyan
    (255, 20, 147),   # deep pink
    (105, 105, 105),  # dim gray
]


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        return ImageFont.load_default()


def render_overlay(page_path: Path, sidecar: Path, out_path: Path) -> None:
    im = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    entries = json.loads(sidecar.read_text())

    px_per_in = max(im.size) / 11.0          # rough scale for label size
    label_size = max(18, int(px_per_in * 0.25))
    font = _font(label_size)

    for i, ent in enumerate(entries):
        color = PALETTE[i % len(PALETTE)]
        poly = [tuple(p) for p in ent["polygon"]]
        if len(poly) >= 3:
            fill = (*color, 40)
            draw.polygon(poly, outline=color, fill=fill, width=3)
        baseline = ent.get("baseline") or []
        if len(baseline) >= 2:
            draw.line([tuple(p) for p in baseline], fill=color, width=2)

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if xs and ys:
            anchor = (min(xs) + 4, min(ys) + 4)
            label = str(i + 1)
            tw = draw.textlength(label, font=font)
            th = label_size + 4
            draw.rectangle(
                [anchor, (anchor[0] + tw + 8, anchor[1] + th)],
                fill=(255, 255, 255, 220),
            )
            draw.text((anchor[0] + 4, anchor[1] + 2), label,
                      fill=color, font=font)

    im.save(out_path, "PNG")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True, type=Path,
                    help="folder with the source page PNGs")
    ap.add_argument("--lines-dir", required=True, type=Path,
                    help="folder with the *.lines.json sidecars")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0,
                    help="render only the first N pages (0 = all)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Recurse: supports both flat (data/lines/<page>.lines.json) and
    # per-page-folder (data/lines_merged/<page>/<page>.lines.json) layouts.
    sidecars = sorted(args.lines_dir.rglob("*.lines.json"))
    if args.limit:
        sidecars = sidecars[: args.limit]

    for sc in sidecars:
        page_stem = sc.stem.replace(".lines", "")
        page_path = args.pages_dir / f"{page_stem}.png"
        if not page_path.exists():
            log.warning("missing page image for %s", page_stem)
            continue
        out = args.out_dir / f"{page_stem}_overlay.png"
        render_overlay(page_path, sc, out)
        log.info("wrote %s", out)


if __name__ == "__main__":
    main()
