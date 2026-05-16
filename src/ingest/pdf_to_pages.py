"""Rasterize PDFs (and copy JPGs) from `Listy/` into single-page PNGs.

Output filename convention:
    <source_stem>_p0001.png, <source_stem>_p0002.png, ...

This `_p####` suffix is parsed downstream by `utils.io.page_id_from_filename`
to keep all crops from one document on the same side of the train/val/test
split.
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import fitz  # pymupdf
from PIL import Image

from src.utils.io import setup_logging

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def rasterize_pdf(pdf_path: Path, out_dir: Path, dpi: int) -> int:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    n = 0
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        out = out_dir / f"{pdf_path.stem}_p{i:04d}.png"
        pix.save(out.as_posix())
        n += 1
    doc.close()
    return n


def copy_image(img_path: Path, out_dir: Path) -> int:
    out = out_dir / f"{img_path.stem}_p0001.png"
    if img_path.suffix.lower() == ".png":
        shutil.copy2(img_path, out)
    else:
        with Image.open(img_path) as im:
            im.convert("RGB").save(out, "PNG")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    setup_logging()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(p for p in args.in_dir.iterdir() if p.is_file())
    total_pages = 0
    for src in sources:
        ext = src.suffix.lower()
        if ext == ".pdf":
            existing = list(args.out_dir.glob(f"{src.stem}_p*.png"))
            if existing and not args.overwrite:
                log.info("skip (exists): %s", src.name)
                continue
            n = rasterize_pdf(src, args.out_dir, args.dpi)
            log.info("pdf %s -> %d page(s)", src.name, n)
            total_pages += n
        elif ext in IMAGE_EXTS:
            target = args.out_dir / f"{src.stem}_p0001.png"
            if target.exists() and not args.overwrite:
                log.info("skip (exists): %s", src.name)
                continue
            n = copy_image(src, args.out_dir)
            log.info("img %s -> %d page(s)", src.name, n)
            total_pages += n
        else:
            log.debug("skip (unsupported): %s", src.name)

    log.info("done. pages written: %d", total_pages)


if __name__ == "__main__":
    main()
