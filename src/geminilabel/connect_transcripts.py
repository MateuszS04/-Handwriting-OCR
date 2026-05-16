"""Connect segmented line crops with Gemini transcripts for TrOCR training.

The line segmentation step writes one sidecar per page:

    data/lines_merged/<page>/<page>.lines.json

Each sidecar entry describes one crop and usually contains:

    {
      "file": "<page>_l001.png",
      "rel_path": "<page>/<page>_l001.png",
      "polygon": [[x0, y0], ...],
      "baseline": [[x0, y], [x1, y]]
    }

This module joins those line records with transcript JSON produced by Gemini
and writes a training JSONL manifest:

    {"file": "...", "image_path": "...", "text": "...", "page_id": "...", ...}

Supported transcript JSON shapes:

1. List of line records:
   [
     {"file": "page/page_l001.png", "text": "..."},
     {"file": "page_l002.png", "transcription": "..."}
   ]

2. Dict containing a list under "lines", "items", "data", or "transcripts":
   {"page_id": "page", "lines": [{"line": 1, "text": "..."}, ...]}

3. Mapping from file/path to text:
   {"page/page_l001.png": "transcription", "page/page_l002.png": "..."}

When a transcript does not contain a filename, but it is inside a page-level
JSON with an ordered "lines" list, matching falls back to line number/order.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from src.utils.io import setup_logging, write_jsonl

log = logging.getLogger(__name__)

TEXT_FIELDS = ("text", "transcription", "transcript", "ground_truth", "label")
FILE_FIELDS = ("file", "rel_path", "image", "image_path", "filename", "path")
LIST_FIELDS = ("lines", "items", "data", "transcripts", "records")
CONF_FIELDS = ("confidence", "score", "probability")


@dataclass(frozen=True)
class LineRecord:
    """One segmented crop from a sidecar JSON."""

    page_id: str
    line_no: int
    file: str
    rel_path: str
    image_path: Path
    polygon: list | None
    baseline: list | None
    pad_px: int | None


@dataclass(frozen=True)
class TranscriptRecord:
    """One normalized transcript entry loaded from Gemini JSON."""

    text: str
    file: str | None = None
    page_id: str | None = None
    line_no: int | None = None
    confidence: float | None = None
    source_json: str | None = None


def _line_no_from_name(name: str) -> int | None:
    match = re.search(r"_l(\d+)", Path(name).stem)
    return int(match.group(1)) if match else None


def _page_id_from_line_name(name: str) -> str | None:
    stem = Path(name).stem
    match = re.match(r"(.+)_l\d+$", stem)
    return match.group(1) if match else None


def _text_from_record(record: dict) -> str | None:
    for field in TEXT_FIELDS:
        value = record.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
    return None


def _file_from_record(record: dict) -> str | None:
    for field in FILE_FIELDS:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _confidence_from_record(record: dict) -> float | None:
    for field in CONF_FIELDS:
        value = record.get(field)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def _line_no_from_record(record: dict, fallback_index: int | None = None) -> int | None:
    for field in ("line_no", "line_number", "line", "index", "idx", "number"):
        value = record.get(field)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    file_value = _file_from_record(record)
    if file_value:
        from_file = _line_no_from_name(file_value)
        if from_file is not None:
            return from_file
    return fallback_index


def _iter_json_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
    else:
        for p in sorted(path.rglob("*.json")):
            # Sidecars are line metadata, not transcript files.
            if not p.name.endswith(".lines.json"):
                yield p


def load_lines(lines_dir: Path, *, require_images: bool = True) -> list[LineRecord]:
    """Load line metadata from all ``*.lines.json`` files under ``lines_dir``."""

    line_records: list[LineRecord] = []
    for sidecar in sorted(lines_dir.rglob("*.lines.json")):
        page_id = sidecar.stem.replace(".lines", "")
        entries = json.loads(sidecar.read_text(encoding="utf-8"))
        for index, entry in enumerate(entries, start=1):
            file_name = entry["file"]
            rel_path = entry.get("rel_path")
            if not rel_path:
                candidate = sidecar.parent / file_name
                try:
                    rel_path = candidate.relative_to(lines_dir).as_posix()
                except ValueError:
                    rel_path = file_name

            image_path = lines_dir / rel_path
            if require_images and not image_path.exists():
                log.warning("missing image for sidecar entry: %s", image_path)
                continue

            line_no = _line_no_from_name(file_name) or index
            line_records.append(
                LineRecord(
                    page_id=page_id,
                    line_no=line_no,
                    file=file_name,
                    rel_path=rel_path,
                    image_path=image_path,
                    polygon=entry.get("polygon"),
                    baseline=entry.get("baseline"),
                    pad_px=entry.get("pad_px"),
                )
            )
    return line_records


def _normalize_line_dicts(
    records: Iterable[dict],
    *,
    page_id: str | None,
    source_json: Path,
) -> Iterator[TranscriptRecord]:
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            continue
        text = _text_from_record(record)
        if text is None:
            continue

        file_value = _file_from_record(record)
        line_no = _line_no_from_record(record, fallback_index=index)
        record_page_id = record.get("page_id") or record.get("page") or page_id
        if file_value and not record_page_id:
            record_page_id = _page_id_from_line_name(file_value)

        yield TranscriptRecord(
            text=text,
            file=file_value,
            page_id=str(record_page_id) if record_page_id else None,
            line_no=line_no,
            confidence=_confidence_from_record(record),
            source_json=source_json.as_posix(),
        )


def load_transcripts(transcripts_path: Path) -> list[TranscriptRecord]:
    """Load transcript records from a JSON file or a directory of JSON files."""

    transcripts: list[TranscriptRecord] = []
    for path in _iter_json_files(transcripts_path):
        data = json.loads(path.read_text(encoding="utf-8"))
        page_id = path.stem

        if isinstance(data, list):
            transcripts.extend(
                _normalize_line_dicts(data, page_id=page_id, source_json=path)
            )
            continue

        if isinstance(data, dict):
            # Mapping from file path to raw string transcript.
            if all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
                for file_value, text in data.items():
                    transcripts.append(
                        TranscriptRecord(
                            text=text.strip(),
                            file=file_value,
                            page_id=_page_id_from_line_name(file_value),
                            line_no=_line_no_from_name(file_value),
                            source_json=path.as_posix(),
                        )
                    )
                continue

            # Page-level dict containing an ordered list of line objects.
            for field in LIST_FIELDS:
                if isinstance(data.get(field), list):
                    record_page_id = data.get("page_id") or data.get("page") or page_id
                    transcripts.extend(
                        _normalize_line_dicts(
                            data[field], page_id=str(record_page_id), source_json=path
                        )
                    )
                    break
            else:
                # Single line object.
                transcripts.extend(
                    _normalize_line_dicts([data], page_id=page_id, source_json=path)
                )

    return transcripts


def _transcript_keys(t: TranscriptRecord) -> list[str]:
    keys: list[str] = []
    if t.file:
        p = Path(t.file)
        keys.extend([t.file, p.name, p.stem])
    if t.page_id and t.line_no is not None:
        keys.append(f"{t.page_id}#{t.line_no:03d}")
        keys.append(f"{t.page_id}#{t.line_no}")
    return keys


def _line_keys(line: LineRecord) -> list[str]:
    return [
        line.rel_path,
        line.file,
        Path(line.file).stem,
        f"{line.page_id}#{line.line_no:03d}",
        f"{line.page_id}#{line.line_no}",
    ]


def build_training_manifest(
    *,
    lines_dir: Path,
    transcripts_path: Path,
    out_jsonl: Path,
    require_images: bool = True,
    min_confidence: float | None = None,
) -> list[dict]:
    """Join line crops with transcript JSON and write a training JSONL file."""

    lines = load_lines(lines_dir, require_images=require_images)
    transcripts = load_transcripts(transcripts_path)

    transcript_index: dict[str, TranscriptRecord] = {}
    duplicate_keys: set[str] = set()
    for transcript in transcripts:
        if min_confidence is not None and transcript.confidence is not None:
            if transcript.confidence < min_confidence:
                continue
        for key in _transcript_keys(transcript):
            if key in transcript_index:
                duplicate_keys.add(key)
            else:
                transcript_index[key] = transcript

    if duplicate_keys:
        log.warning("duplicate transcript keys ignored after first match: %d", len(duplicate_keys))

    manifest: list[dict] = []
    unmatched_lines: list[str] = []
    for line in lines:
        transcript = None
        for key in _line_keys(line):
            transcript = transcript_index.get(key)
            if transcript:
                break

        if transcript is None:
            unmatched_lines.append(line.rel_path)
            continue

        manifest.append(
            {
                "file": line.rel_path,
                "image_path": line.image_path.as_posix(),
                "text": transcript.text,
                "page_id": line.page_id,
                "line_no": line.line_no,
                "confidence": transcript.confidence,
                "source_json": transcript.source_json,
                "polygon": line.polygon,
                "baseline": line.baseline,
                "pad_px": line.pad_px,
            }
        )

    write_jsonl(manifest, out_jsonl)

    log.info("loaded lines: %d", len(lines))
    log.info("loaded transcripts: %d", len(transcripts))
    log.info("matched training records: %d -> %s", len(manifest), out_jsonl)
    if unmatched_lines:
        log.warning("unmatched lines: %d (first: %s)", len(unmatched_lines), unmatched_lines[:5])

    return manifest


def write_page_stratified_splits(
    manifest: list[dict],
    out_dir: Path,
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Optionally split the manifest by page to avoid line-level leakage."""

    if not manifest:
        return
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0:
        raise ValueError("train_ratio + val_ratio must be <= 1.0")

    by_page: dict[str, list[dict]] = {}
    for record in manifest:
        by_page.setdefault(record["page_id"], []).append(record)

    page_ids = sorted(by_page)
    random.Random(seed).shuffle(page_ids)
    n = len(page_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    buckets = {
        "train": page_ids[:n_train],
        "val": page_ids[n_train : n_train + n_val],
        "test": page_ids[n_train + n_val :],
    }
    for name, ids in buckets.items():
        records = [record for page_id in ids for record in by_page[page_id]]
        write_jsonl(records, out_dir / f"{name}.jsonl")
        log.info("%s split: %d pages, %d records", name, len(ids), len(records))


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines-dir", required=True, type=Path)
    ap.add_argument("--transcripts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="include records even when the image file is missing on disk",
    )
    args = ap.parse_args()

    manifest = build_training_manifest(
        lines_dir=args.lines_dir,
        transcripts_path=args.transcripts,
        out_jsonl=args.out,
        require_images=not args.allow_missing_images,
        min_confidence=args.min_confidence,
    )
    if args.splits_dir:
        write_page_stratified_splits(
            manifest,
            args.splits_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()

