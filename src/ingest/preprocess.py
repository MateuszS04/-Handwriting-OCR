"""Page-level preprocessing: deskew + mild denoise.

Intentionally conservative — heavy binarization or aggressive denoising tends
to hurt downstream TrOCR (which was pretrained on grayscale-ish RGB photos of
text). We keep the page in RGB and only:
  1. estimate skew via Hough transform on edge pixels and rotate to correct it,
  2. apply mild bilateral filtering to suppress paper texture,
  3. clip to 8-bit RGB and save.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.utils.io import setup_logging

log = logging.getLogger(__name__)


def estimate_skew_deg(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 720, threshold=300)
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = (theta * 180.0 / math.pi) - 90.0
        if -20.0 < deg < 20.0:
            angles.append(deg)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.05:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_page(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_deg(gray)
    rotated = rotate(img_bgr, angle)
    denoised = cv2.bilateralFilter(rotated, d=5, sigmaColor=35, sigmaSpace=35)
    return denoised


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    setup_logging()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pages = sorted(args.in_dir.glob("*.png"))
    for p in pages:
        out = args.out_dir / p.name
        if out.exists() and not args.overwrite:
            continue
        img = cv2.imread(p.as_posix(), cv2.IMREAD_COLOR)
        if img is None:
            log.warning("could not read %s", p)
            continue
        proc = preprocess_page(img)
        Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)).save(out, "PNG")
        log.info("preprocessed %s", p.name)


if __name__ == "__main__":
    main()
