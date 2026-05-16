"""Post-process line segmentation: merge baseline-overlapping fragments.

Kraken occasionally splits a single visual handwritten line into 2-3 pieces.
The fragments are easy to spot: their baseline y-coordinates are *much*
closer to each other than the typical spacing between real consecutive
lines on the page. This script clusters lines by baseline-y proximity,
unions their polygons, and emits one clean crop per cluster.

The threshold is per-page and adaptive:

    cluster_gap_threshold = max(min_abs_px,
                                frac * median_consecutive_baseline_gap)

Two consecutive (sorted by y) lines are merged when their baseline-y
difference is below this threshold. Tiny clusters (area below
``--min-area-frac`` × median cluster area) are dropped as noise.

Inputs:
    --pages-dir   data/pages_clean    (the source page PNGs)
    --lines-dir   data/lines          (the existing crops + *.lines.json)
    --out-dir     data/lines_merged   (cleaned output)

Output:
    one ``<page>_l###.png`` per merged line
    one ``<page>.lines.json`` sidecar (same schema as the upstream segmenter)
    one summary line per page in the log: ``before -> after``
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw

from src.utils.io import setup_logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class RawLine:
    polygon: List[tuple]
    baseline: List[tuple]
    bl_y: float
    bbox: tuple   # (x0, y0, x1, y1)


def _load_lines(sidecar: Path) -> List[RawLine]:
    out = []
    for ent in json.loads(sidecar.read_text()):
        poly = [tuple(p) for p in ent["polygon"]]
        bl = [tuple(p) for p in (ent.get("baseline") or [])]
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        bl_y = (statistics.median(p[1] for p in bl)
                if bl else (bbox[1] + bbox[3]) / 2)
        out.append(RawLine(polygon=poly, baseline=bl, bl_y=float(bl_y), bbox=bbox))
    return out


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def _fit_baseline(line: RawLine) -> tuple[float, float]:
    """Return (slope, intercept) of the line's baseline in image coords.

    Falls back to the bbox vertical midline if the baseline has fewer than
    two distinct x-coordinates.
    """
    bl = line.baseline if line.baseline else []
    xs = np.array([p[0] for p in bl], dtype=float)
    ys = np.array([p[1] for p in bl], dtype=float)
    if len(xs) < 2 or (xs.max() - xs.min()) < 5:
        cx_mid = (line.bbox[1] + line.bbox[3]) / 2.0
        return 0.0, float(cx_mid)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)


def _median_baseline_gap(lines: List[RawLine]) -> float:
    """Median of consecutive baseline-y gaps, ignoring near-zero overlaps."""
    if len(lines) < 2:
        return 0.0
    ys = sorted(r.bl_y for r in lines)
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    nonzero = [g for g in gaps if g > 1.0] or gaps
    return float(statistics.median(nonzero)) if nonzero else 0.0


def _cluster_baseline_aware(
    lines: List[RawLine],
    *,
    frac: float,
    min_abs_px: int,
    x_overlap_frac: float,
) -> List[List[RawLine]]:
    """Group fragments that lie on the same (possibly slanted) baseline.

    Algorithm (per page):
      1. Fit each fragment's baseline to a line ``y = a·x + b``.
      2. For every pair (i, j):
            * skip if they overlap horizontally by more than ``x_overlap_frac``
              of the narrower fragment's width — those are stacked, not the
              same line;
            * otherwise compute a "meeting x" (gap midpoint when disjoint;
              overlap midpoint when slightly overlapping) and extrapolate
              both baselines to it;
            * if the two extrapolated y-values agree within
              ``frac × median_baseline_gap`` → mark i, j as the same line
              via union-find.
      3. Collect connected components; sort top-to-bottom.

    This preserves slope information, so a slanted line "\" no longer gets
    its right tail attached to the flatter line above it.
    """
    n = len(lines)
    if n == 0:
        return []
    if n == 1:
        return [lines]

    fits = [_fit_baseline(r) for r in lines]
    median_gap = _median_baseline_gap(lines)
    y_thresh = max(float(min_abs_px), frac * median_gap)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        xi0, _, xi1, _ = lines[i].bbox
        wi = max(1, xi1 - xi0)
        for j in range(i + 1, n):
            xj0, _, xj1, _ = lines[j].bbox
            wj = max(1, xj1 - xj0)

            ovl = max(0, min(xi1, xj1) - max(xi0, xj0))
            if ovl / min(wi, wj) > x_overlap_frac:
                continue   # stacked → different lines

            if xi1 <= xj0:
                meet_x = (xi1 + xj0) / 2.0
            elif xj1 <= xi0:
                meet_x = (xj1 + xi0) / 2.0
            else:
                meet_x = (max(xi0, xj0) + min(xi1, xj1)) / 2.0

            yi = fits[i][0] * meet_x + fits[i][1]
            yj = fits[j][0] * meet_x + fits[j][1]

            if abs(yi - yj) < y_thresh:
                union(i, j)

    groups: dict[int, list[RawLine]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    clusters = sorted(groups.values(), key=lambda cl: min(r.bl_y for r in cl))

    log.debug("median_gap=%.0f  y_thresh=%.0f  -> %d clusters (from %d frags)",
              median_gap, y_thresh, len(clusters), n)
    return clusters


def _cluster(
    lines: List[RawLine],
    *,
    frac: float,
    min_abs_px: int,
    x_overlap_frac: float = 0.30,
) -> List[List[RawLine]]:
    """Public entry; baseline-aware clustering (handles slanted lines)."""
    return _cluster_baseline_aware(
        lines, frac=frac, min_abs_px=min_abs_px, x_overlap_frac=x_overlap_frac,
    )


# ---------------------------------------------------------------------------
# rendering one merged crop from a cluster of polygons
# ---------------------------------------------------------------------------
def _compute_pad(cluster: List[RawLine], pad_px: int, pad_frac: float) -> int:
    """Per-cluster padding: scales with the line's own height.

    Returns ``max(pad_px, pad_frac * cluster_bbox_height)``. The fractional
    term ensures tall cursive (h~270 px) gets ~30 px while compact lines
    (h~80 px) still get the absolute floor.
    """
    if not cluster:
        return int(pad_px)
    y0 = min(r.bbox[1] for r in cluster)
    y1 = max(r.bbox[3] for r in cluster)
    return int(max(pad_px, round(pad_frac * (y1 - y0))))


def _render_cluster(
    page_im: Image.Image,
    cluster: List[RawLine],
    *,
    pad: int,
    use_mask: bool = True,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Mask the union of all cluster polygons (dilated by `pad`) on white
    background, then crop to the padded bbox.

    Returns ``(crop_image, (x0, y0, x1, y1))`` where the bbox is the
    image-clamped padded bbox actually used.
    """
    w, h = page_im.size
    arr = np.array(page_im.convert("RGB"))

    if use_mask:
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        for r in cluster:
            if len(r.polygon) >= 3:
                md.polygon(r.polygon, outline=255, fill=255)
        mask_arr = np.array(mask)
        if pad > 0:
            try:
                import cv2
                k = 2 * pad + 1
                kernel = np.ones((k, k), np.uint8)
                mask_arr = cv2.dilate(mask_arr, kernel, iterations=1)
            except ImportError:
                from PIL import ImageFilter
                mask_arr = np.array(
                    Image.fromarray(mask_arr).filter(
                        ImageFilter.MaxFilter(min(99, 2 * pad + 1))
                    )
                )
        keep = mask_arr > 0
        white = np.full_like(arr, 255)
        out_arr = np.where(keep[..., None], arr, white)
    else:
        out_arr = arr

    x0 = min(r.bbox[0] for r in cluster) - pad
    y0 = min(r.bbox[1] for r in cluster) - pad
    x1 = max(r.bbox[2] for r in cluster) + pad
    y1 = max(r.bbox[3] for r in cluster) + pad
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    return Image.fromarray(out_arr[y0:y1, x0:x1]), (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# per-page driver
# ---------------------------------------------------------------------------
def _process_page(
    page_path: Path,
    sidecar: Path,
    out_dir: Path,
    *,
    frac: float,
    min_abs_px: int,
    x_overlap_frac: float,
    min_w: int,
    min_h: int,
    min_area_frac: float,
    pad_px: int,
    pad_frac: float,
    use_mask: bool,
) -> tuple[int, int]:
    raw = _load_lines(sidecar)
    if not raw:
        return 0, 0

    clusters = _cluster(
        raw, frac=frac, min_abs_px=min_abs_px, x_overlap_frac=x_overlap_frac,
    )

    raw_bboxes = []
    raw_areas = []
    for cl in clusters:
        x0 = min(r.bbox[0] for r in cl)
        y0 = min(r.bbox[1] for r in cl)
        x1 = max(r.bbox[2] for r in cl)
        y1 = max(r.bbox[3] for r in cl)
        raw_bboxes.append((x0, y0, x1, y1))
        raw_areas.append((x1 - x0) * (y1 - y0))
    median_area = statistics.median(raw_areas) if raw_areas else 0
    area_thresh = median_area * min_area_frac

    page_dir = out_dir / page_path.stem
    page_dir.mkdir(parents=True, exist_ok=True)

    page_im = Image.open(page_path).convert("RGB")
    sidecar_out = []
    kept = 0
    for cl, bbox, area in zip(clusters, raw_bboxes, raw_areas):
        x0, y0, x1, y1 = bbox
        if (x1 - x0) < min_w or (y1 - y0) < min_h or area < area_thresh:
            log.debug("drop tiny cluster on %s: %dx%d area=%d",
                      page_path.name, x1 - x0, y1 - y0, area)
            continue
        kept += 1
        pad = _compute_pad(cl, pad_px=pad_px, pad_frac=pad_frac)
        crop, padded = _render_cluster(page_im, cl, pad=pad, use_mask=use_mask)
        out_name = f"{page_path.stem}_l{kept:03d}.png"
        crop.save(page_dir / out_name, "PNG")
        px0, py0, px1, py1 = padded
        sidecar_out.append({
            "file": out_name,                            # name within page folder
            "rel_path": f"{page_path.stem}/{out_name}",  # path under out_dir
            "polygon": [[px0, py0], [px1, py0], [px1, py1], [px0, py1]],
            "baseline": [
                [int(min(r.baseline[0][0] for r in cl if r.baseline)
                     if any(r.baseline for r in cl) else px0),
                 int(statistics.median(r.bl_y for r in cl))],
                [int(max(r.baseline[-1][0] for r in cl if r.baseline)
                     if any(r.baseline for r in cl) else px1),
                 int(statistics.median(r.bl_y for r in cl))],
            ],
            "pad_px": pad,
            "merged_from": [r.bbox for r in cl],
        })

    (page_dir / f"{page_path.stem}.lines.json").write_text(
        json.dumps(sidecar_out, ensure_ascii=False, indent=2)
    )
    return len(raw), kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True, type=Path)
    ap.add_argument("--lines-dir", required=True, type=Path,
                    help="folder with the input *_l###.png and *.lines.json")
    ap.add_argument("--out-dir", required=True, type=Path)

    ap.add_argument("--frac", type=float, default=0.45,
                    help="merge baselines whose extrapolated y agree within "
                         "frac * median_gap")
    ap.add_argument("--min-abs-px", type=int, default=12,
                    help="absolute floor for the merge threshold")
    ap.add_argument("--x-overlap-frac", type=float, default=0.30,
                    help="if two fragments horizontally overlap by more than "
                         "this fraction of the narrower one's width, they are "
                         "treated as DIFFERENT lines (stacked, not the same).")
    ap.add_argument("--min-width", type=int, default=120,
                    help="drop crops narrower than this (px)")
    ap.add_argument("--min-height", type=int, default=35,
                    help="drop crops shorter than this (px)")
    ap.add_argument("--min-area-frac", type=float, default=0.08,
                    help="drop crops smaller than frac * median_cluster_area")

    ap.add_argument("--pad-px", type=int, default=15,
                    help="absolute floor for crop padding on all 4 sides (px). "
                         "Protects ascenders/descenders from being clipped.")
    ap.add_argument("--pad-frac", type=float, default=0.12,
                    help="additional padding as fraction of the cluster's "
                         "bbox height; effective pad = max(pad-px, "
                         "pad-frac * height). Default ~12%% works for both "
                         "tall and compact handwriting.")
    ap.add_argument("--no-mask", action="store_true",
                    help="skip polygon masking entirely (faster, but neighbor-"
                         "line strokes from above/below may bleed into the "
                         "crop). Useful when polygons are too tight to begin "
                         "with.")

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sidecars = sorted(args.lines_dir.glob("*.lines.json"))
    if not sidecars:
        raise SystemExit(f"no *.lines.json found in {args.lines_dir}")

    total_before, total_after = 0, 0
    for sc in sidecars:
        page_stem = sc.stem.replace(".lines", "")
        page_path = args.pages_dir / f"{page_stem}.png"
        if not page_path.exists():
            log.warning("missing page image %s", page_path.name)
            continue
        existing = list((args.out_dir / page_stem).glob(f"{page_stem}_l*.png"))
        if existing and not args.overwrite:
            log.info("skip (exists): %s (%d lines)", page_stem, len(existing))
            continue
        before, after = _process_page(
            page_path, sc, args.out_dir,
            frac=args.frac, min_abs_px=args.min_abs_px,
            x_overlap_frac=args.x_overlap_frac,
            min_w=args.min_width, min_h=args.min_height,
            min_area_frac=args.min_area_frac,
            pad_px=args.pad_px, pad_frac=args.pad_frac,
            use_mask=not args.no_mask,
        )
        total_before += before
        total_after += after
        log.info("%s: %d -> %d lines", page_stem, before, after)

    log.info("done. %d -> %d (%+d)",
             total_before, total_after, total_after - total_before)


if __name__ == "__main__":
    main()
