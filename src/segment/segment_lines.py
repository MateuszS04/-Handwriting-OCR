"""Line segmentation via Kraken's baseline segmenter.

Two backends are supported:

  * ``kraken`` (default, recommended) — uses the pretrained baseline model
    shipped with ``kraken``. Polygon line crops are extracted and saved as
    individual PNGs.

  * ``projection`` (fallback) — naive horizontal projection profile. Useful
    only as a smoke test if Kraken cannot be installed.

Output filename convention:
    <page_stem>_l001.png, <page_stem>_l002.png, ...

The line ordering is top-to-bottom on the page. A sidecar JSON
(``<page_stem>.lines.json``) is written alongside, listing each crop's
bounding polygon for later reference.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from src.utils.io import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kraken backend
# ---------------------------------------------------------------------------
def segment_with_kraken(image_path: Path) -> List[dict]:
    """Return list of {'polygon': [(x,y), ...], 'baseline': [...]} dicts.

    Compatible with Kraken >=4.x where ``blla.segment`` returns a
    ``Segmentation`` dataclass (not a dict). Each ``line`` is a
    ``BaselineLine`` with ``.boundary`` and ``.baseline`` attributes.
    """
    try:
        from kraken import blla  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "kraken is not installed. `pip install kraken` or use "
            "--backend projection."
        ) from e

    im = Image.open(image_path).convert("RGB")
    seg = blla.segment(im)

    raw_lines = getattr(seg, "lines", None)
    if raw_lines is None and isinstance(seg, dict):
        raw_lines = seg.get("lines", [])
    raw_lines = raw_lines or []

    lines: List[dict] = []
    for line in raw_lines:
        boundary = getattr(line, "boundary", None) \
            if not isinstance(line, dict) else line.get("boundary")
        baseline = getattr(line, "baseline", None) \
            if not isinstance(line, dict) else line.get("baseline")

        if not boundary:
            if baseline and len(baseline) >= 2:
                xs = [p[0] for p in baseline]
                ys = [p[1] for p in baseline]
                pad_y = 40
                boundary = [
                    (min(xs), max(0, min(ys) - pad_y)),
                    (max(xs), max(0, min(ys) - pad_y)),
                    (max(xs), max(ys) + pad_y),
                    (min(xs), max(ys) + pad_y),
                ]
            else:
                continue

        lines.append({
            "polygon": [tuple(map(int, p)) for p in boundary],
            "baseline": [tuple(map(int, p)) for p in (baseline or [])],
        })
    return lines


def crop_polygon(im: Image.Image, polygon: List[Tuple[int, int]]) -> Image.Image:
    """Mask the polygon area on white background, then crop to its bbox."""
    from PIL import ImageDraw

    arr = np.array(im.convert("RGB"))
    h, w = arr.shape[:2]
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).polygon(polygon, outline=255, fill=255)
    mask_arr = np.array(mask) > 0

    white = np.full_like(arr, 255)
    out = np.where(mask_arr[..., None], arr, white)

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))
    return Image.fromarray(out[y0:y1, x0:x1])


# ---------------------------------------------------------------------------
# Naive projection backend (fallback only)
# ---------------------------------------------------------------------------
def segment_with_projection(image_path: Path, min_height: int = 30) -> List[dict]:
    im = Image.open(image_path).convert("L")
    arr = 255 - np.array(im)
    row_ink = (arr > 40).sum(axis=1)
    threshold = max(5, row_ink.mean() * 0.5)
    in_line = row_ink > threshold

    bands = []
    start = None
    for y, on in enumerate(in_line):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_height:
                bands.append((start, y))
            start = None
    if start is not None:
        bands.append((start, len(in_line)))

    w = im.size[0]
    return [
        {
            "polygon": [(0, y0), (w, y0), (w, y1), (0, y1)],
            "baseline": [(0, (y0 + y1) // 2), (w, (y0 + y1) // 2)],
        }
        for y0, y1 in bands
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def process_page(
    image_path: Path,
    out_dir: Path,
    backend: str,
) -> int:
    if backend == "kraken":
        lines = segment_with_kraken(image_path)
    elif backend == "projection":
        lines = segment_with_projection(image_path)
    else:
        raise ValueError(f"unknown backend: {backend}")

    if not lines:
        log.warning("no lines on %s", image_path.name)
        return 0

    im = Image.open(image_path).convert("RGB")
    sidecar = []
    for i, line in enumerate(lines, start=1):
        crop = crop_polygon(im, line["polygon"])
        out = out_dir / f"{image_path.stem}_l{i:03d}.png"
        crop.save(out, "PNG")
        sidecar.append({
            "file": out.name,
            "polygon": line["polygon"],
            "baseline": line["baseline"],
        })

    side_path = out_dir / f"{image_path.stem}.lines.json"
    side_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2))
    return len(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", choices=["kraken", "projection"], default="kraken")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    setup_logging()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(args.in_dir.glob("*.png"))
    total = 0
    for p in pages:
        already = list(args.out_dir.glob(f"{p.stem}_l*.png"))
        if already and not args.overwrite:
            log.info("skip (exists): %s (%d lines)", p.name, len(already))
            continue
        n = process_page(p, args.out_dir, args.backend)
        log.info("%s -> %d line crops", p.name, n)
        total += n

    log.info("done. line crops written: %d", total)


if __name__ == "__main__":
    main()
