# Projekt_ML — TrOCR fine-tuning on handwritten Polish letters

End-to-end OCR pipeline for the `Listy/` dataset (handwritten Polish letters,
~300 multi-page documents). The system fine-tunes Microsoft's
[TrOCR](https://huggingface.co/microsoft/trocr-large-handwritten) on
line-level crops, using **Gemini** as a bootstrap labeler to produce ground
truth that is then partially human-reviewed.

Reference reading included in the repo root:
- `2508.11499v1.pdf` — Meoded, *HTR of historical manuscripts with TrOCR*
  (motivates the augmentation + ensemble strategy used here).
- `2602.14524v1.pdf` — Vesalainen et al., *Error patterns in historical OCR:
  TrOCR vs VLM* (motivates the human-review pass over Gemini labels).

---

## Pipeline

```
PDF/JPG  ──▶ rasterize ──▶ preprocess ──▶ line segmentation
                                                │
                                                ▼
                                       Gemini labeling
                                                │
                                                ▼
                                       (optional) human review
                                                │
                                                ▼
                                       HF Dataset (image, text)
                                                │
                                                ▼
                                       TrOCR fine-tuning
                                                │
                                                ▼
                                       Inference + CER/WER eval
```

## Repo layout

```
Projekt_ML/
├── data/
│   ├── pages/              # rasterized pages   (.png, gitignored)
│   ├── lines/              # line crops         (.png, gitignored)
│   ├── gt_reviewed/        # human-corrected    (.jsonl, gitignored)
│   └── splits/             # train/val/test     (.jsonl, gitignored)
├── src/
│   ├── ingest/             # pdf_to_pages.py, preprocess.py
│   ├── segment/            # segment_lines.py
│   ├── label/              # gemini_label.py
│   ├── infer/              # predict.py
│   └── utils/              # logging, io helpers
├── configs/
│   └── trocr_pl.yaml
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# (line segmentation only) pip install kraken

# 3. Rasterize PDFs and JPGs to pages
python -m src.ingest.pdf_to_pages \
    --in-dir  Listy \
    --out-dir data/pages \
    --dpi 300

# 4. Preprocess pages (deskew, denoise)
python -m src.ingest.preprocess \
    --in-dir  data/pages \
    --out-dir data/pages_clean

# 5. Segment lines (Kraken)
python -m src.segment.segment_lines \
    --in-dir  data/pages_clean \
    --out-dir data/lines



# 7. Build train/val/test splits (page-stratified to avoid leakage)
python -m src.train.dataset build_splits \
    --gt   data/gt_raw/gemini.jsonl \
    --out  data/splits





