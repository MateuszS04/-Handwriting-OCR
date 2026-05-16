# Projekt_ML — OCR Dataset Pipeline For Polish Handwritten Letters

This repository prepares a line-level OCR dataset from scanned handwritten
Polish letters in `Listy/`. The current pipeline focuses on:

1. rasterizing PDF/JPG scans into page images,
2. preprocessing pages,
3. segmenting pages into line crops,
4. merging bad line fragments,
5. transcribing line crops with Gemini using whole-page context,
6. connecting transcriptions with image paths for TrOCR training.

The TrOCR training code is not currently part of the active source tree. The
output of this repo is a clean `train.jsonl` / `val.jsonl` / `test.jsonl`
dataset that can be consumed by a future TrOCR fine-tuning script.

Reference PDFs in the repo root:

- `2508.11499v1.pdf` — historical handwriting recognition with TrOCR.
- `2602.14524v1.pdf` — error patterns in TrOCR vs vision-language models.

## Current Pipeline

```text
Listy PDFs/JPGs
  -> data/pages
  -> data/pages_clean
  -> data/lines
  -> data/lines_merged/<page_id>/
  -> data/gt_raw/gemini_lines.json
  -> data/splits/all.jsonl
  -> data/splits/train.jsonl / val.jsonl / test.jsonl
```

## Repository Layout

```text
Projekt_ML/
├── Listy/                         # raw scanned letters
├── data/
│   ├── pages/                     # rasterized page PNGs
│   ├── pages_clean/               # preprocessed page PNGs
│   ├── lines/                     # raw Kraken line crops + sidecars
│   ├── lines_merged/              # final line crops, one folder per page
│   ├── debug/                     # overlay visualizations
│   ├── gt_raw/                    # Gemini transcription JSON
│   └── splits/                    # training manifests
├── src/
│   ├── ingest/
│   │   ├── pdf_to_pages.py
│   │   └── preprocess.py
│   ├── segment/
│   │   ├── segment_lines.py
│   │   ├── merge_fragments.py
│   │   └── visualize.py
│   ├── geminilabel/
│   │   ├── gemini_context_label.py
│   │   └── connect_transcripts.py
│   └── utils/
│       └── io.py
├── requirements.txt
└── README.md
```

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kraken may need to be installed separately:

```bash
pip install "kraken[pdf]" --no-deps
```

If you use the Gemini step, set your key:

```bash
export GEMINI_API_KEY="your_key_here"
```

or put it in `.env`:

```text
GEMINI_API_KEY=your_key_here
```

## 1. Rasterize Raw Scans

Convert PDFs and image files from `Listy/` into one PNG per page.

```bash
python -m src.ingest.pdf_to_pages \
  --in-dir Listy \
  --out-dir data/pages \
  --dpi 300
```

Output naming:

```text
data/pages/<source_stem>_p0001.png
data/pages/<source_stem>_p0002.png
```

## 2. Preprocess Pages

Deskew and lightly denoise pages.

```bash
python -m src.ingest.preprocess \
  --in-dir data/pages \
  --out-dir data/pages_clean
```

The preprocessing stays conservative. It does not binarize images by default,
because TrOCR-style models usually benefit from grayscale/RGB stroke gradients.

## 3. Segment Pages Into Raw Lines

Use Kraken to produce initial line crops and sidecar metadata.

```bash
python -m src.segment.segment_lines \
  --in-dir data/pages_clean \
  --out-dir data/lines \
  --backend kraken
```

There is also a fallback projection backend:

```bash
python -m src.segment.segment_lines \
  --in-dir data/pages_clean \
  --out-dir data/lines \
  --backend projection
```

Kraken output can over-segment slanted handwriting. The next step fixes many
of those cases.

## 4. Merge Line Fragments

Post-process Kraken output into the final line-crop folder structure.

```bash
rm -rf data/lines_merged

python -m src.segment.merge_fragments \
  --pages-dir data/pages_clean \
  --lines-dir data/lines \
  --out-dir data/lines_merged
```

Output layout:

```text
data/lines_merged/
├── 20251126183542690_0001_p0001/
│   ├── 20251126183542690_0001_p0001.lines.json
│   ├── 20251126183542690_0001_p0001_l001.png
│   ├── 20251126183542690_0001_p0001_l002.png
│   └── ...
└── ...
```

Useful tuning flags:

```bash
python -m src.segment.merge_fragments \
  --pages-dir data/pages_clean \
  --lines-dir data/lines \
  --out-dir data/lines_merged \
  --frac 0.45 \
  --x-overlap-frac 0.30 \
  --pad-px 15 \
  --pad-frac 0.12
```

Key flags:

- `--frac`: how tolerant merging is for slanted baseline fragments.
- `--x-overlap-frac`: helps prevent stacked neighboring lines from merging.
- `--pad-px`: minimum crop padding on all sides.
- `--pad-frac`: extra crop padding as a fraction of line height.
- `--no-mask`: crop rectangular page regions without polygon masking.

## 5. Visualize Segmentation Quality

Render overlays on top of full pages.

```bash
python -m src.segment.visualize \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out-dir data/debug/lines_merged_overlay
```

Open images in:

```text
data/debug/lines_merged_overlay/
```

Use these overlays to check whether:

- each visual line has one polygon,
- slanted lines are not split into fragments,
- ascenders/descenders are not clipped,
- neighboring lines are not merged.

## 6. Transcribe Lines With Gemini + Page Context

This step sends each full page to Gemini first, saves the rough page
transcription, then sends each line crop with that page transcription as
context.

Small test run:

```bash
python -m src.geminilabel.gemini_context_label \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out data/gt_raw/gemini_lines.json \
  --limit-pages 1 \
  --limit-lines 5
```

Full run:

```bash
python -m src.geminilabel.gemini_context_label \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out data/gt_raw/gemini_lines.json
```

Default model:

```text
gemini-2.5-flash
```

You can override it:

```bash
python -m src.geminilabel.gemini_context_label \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out data/gt_raw/gemini_lines.json \
  --model gemini-2.5-pro
```

Outputs:

```text
data/gt_raw/gemini_lines.json        # one record per line crop
data/gt_raw/gemini_lines.pages.json  # one rough transcript per full page
```

Line record example:

```json
{
  "file": "20251126183542690_0001_p0001/20251126183542690_0001_p0001_l001.png",
  "page_id": "20251126183542690_0001_p0001",
  "line": 1,
  "text": "Kochana Stefciu ...",
  "confidence": 0.91,
  "model": "gemini-2.5-flash",
  "prompt_version": "page-context-v1"
}
```

The script is resumable. It saves after every line and skips already-labeled
`file` entries unless you pass `--overwrite`.

## 7. Create Training Manifests

Connect the segmented line images with Gemini transcriptions.

```bash
python -m src.geminilabel.connect_transcripts \
  --lines-dir data/lines_merged \
  --transcripts data/gt_raw/gemini_lines.json \
  --out data/splits/all.jsonl \
  --splits-dir data/splits
```

Outputs:

```text
data/splits/all.jsonl
data/splits/train.jsonl
data/splits/val.jsonl
data/splits/test.jsonl
```

The split is page-stratified, so lines from the same page stay in the same
split. This prevents train/test leakage from the same handwriting and scan.

Training JSONL record example:

```json
{
  "file": "20251126183542690_0001_p0001/20251126183542690_0001_p0001_l001.png",
  "image_path": "data/lines_merged/20251126183542690_0001_p0001/20251126183542690_0001_p0001_l001.png",
  "text": "Kochana Stefciu ...",
  "page_id": "20251126183542690_0001_p0001",
  "line_no": 1
}
```

Optional confidence filter:

```bash
python -m src.geminilabel.connect_transcripts \
  --lines-dir data/lines_merged \
  --transcripts data/gt_raw/gemini_lines.json \
  --out data/splits/all.jsonl \
  --splits-dir data/splits \
  --min-confidence 0.75
```

## Full Command Sequence

```bash
python -m src.ingest.pdf_to_pages \
  --in-dir Listy \
  --out-dir data/pages \
  --dpi 300

python -m src.ingest.preprocess \
  --in-dir data/pages \
  --out-dir data/pages_clean

python -m src.segment.segment_lines \
  --in-dir data/pages_clean \
  --out-dir data/lines \
  --backend kraken

rm -rf data/lines_merged

python -m src.segment.merge_fragments \
  --pages-dir data/pages_clean \
  --lines-dir data/lines \
  --out-dir data/lines_merged

python -m src.segment.visualize \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out-dir data/debug/lines_merged_overlay

python -m src.geminilabel.gemini_context_label \
  --pages-dir data/pages_clean \
  --lines-dir data/lines_merged \
  --out data/gt_raw/gemini_lines.json

python -m src.geminilabel.connect_transcripts \
  --lines-dir data/lines_merged \
  --transcripts data/gt_raw/gemini_lines.json \
  --out data/splits/all.jsonl \
  --splits-dir data/splits
```

## Notes

- Gemini labels are bootstrap ground truth. Review a subset manually before
  treating them as final labels.
- Gemini may silently normalize spelling or guess uncertain words. This is why
  the page-context prompt asks it to preserve original spelling and use the line
  image as the source of truth.
- For TrOCR, keep line images RGB/grayscale-like. Avoid hard binarization unless
  you run an experiment proving it helps your data.
- If segmentation looks wrong, inspect overlay images before tuning parameters.

