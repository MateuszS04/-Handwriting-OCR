"""Transcribe line crops with Gemini using whole-page context.

Workflow:

1. For each page folder in ``data/lines_merged`` load the matching full page
   image from ``data/pages_clean/<page_id>.png``.
2. Send the whole page to Gemini once and save its rough full-page transcript.
3. For every line crop from that page, send:
      * the line crop image
      * the full-page transcript from step 2 as context
   and ask Gemini to return the exact transcription of only that line.
4. Save one local JSON record per line. No training happens here.

The output JSON is directly compatible with:

    python -m src.geminilabel.connect_transcripts \
      --lines-dir data/lines_merged \
      --transcripts data/gt_raw/gemini_lines.json \
      --out data/splits/all.jsonl \
      --splits-dir data/splits
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.geminilabel.connect_transcripts import LineRecord, load_lines
from src.utils.io import setup_logging

log = logging.getLogger(__name__)

PROMPT_VERSION = "page-context-v1"

PAGE_PROMPT = """You are an OCR system for handwritten Polish letters.

Transcribe the entire page as best as possible.

Rules:
- Preserve original spelling, punctuation, capitalization, and Polish diacritics.
- Do NOT modernize, normalize, translate, or correct grammar.
- Keep the natural reading order from top to bottom.
- If a word is uncertain, keep the most likely reading.
- If a character is illegible, write '?'.

Return STRICT JSON:
{"text": "<full page transcription>", "notes": "<optional short note>"}"""

LINE_PROMPT_TEMPLATE = """You are an OCR system for handwritten Polish letters.

Below is a rough full-page transcription from the same page. Use it only as
context to resolve ambiguous words in the line image. The line image is the
source of truth.

FULL PAGE CONTEXT:
{page_text}

Task:
Transcribe ONLY the provided line image.

Rules:
- Output exactly the text visible in this one line crop.
- Preserve original spelling, abbreviations, punctuation, capitalization, and
  Polish diacritics (ą ć ę ł ń ó ś ź ż and capitals).
- Do NOT copy extra words from the page context if they are not visible in the
  line image.
- Do NOT modernize, normalize, translate, or correct grammar.
- If a word is crossed out, wrap it in <strike>...</strike>.
- If a character is illegible, write '?'.
- If the crop is blank or contains no readable text, return an empty text.

Return STRICT JSON:
{{"text": "<line transcription>", "confidence": <float 0..1>, "notes": "<optional short note>"}}"""


class TransientGeminiError(Exception):
    """Retryable Gemini/API error."""


def _json_or_fallback(raw: str, *, fallback_text: str = "") -> dict:
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {"text": fallback_text, "notes": raw[:300]}
    except json.JSONDecodeError:
        return {"text": fallback_text, "notes": f"non-JSON response: {raw[:300]}"}


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type(TransientGeminiError),
)
def _generate_json(client, *, model: str, image_path: Path, prompt: str) -> dict:
    from google.genai import types

    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        msg = str(e).lower()
        if any(k in msg for k in ("rate", "quota", "503", "504", "timeout", "deadline")):
            raise TransientGeminiError(str(e)) from e
        raise

    return _json_or_fallback(resp.text or "{}")


def _read_json_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} must contain a JSON list")


def _read_page_contexts(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    raise ValueError(f"{path} must contain a JSON object")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _group_lines_by_page(lines: list[LineRecord]) -> dict[str, list[LineRecord]]:
    grouped: dict[str, list[LineRecord]] = {}
    for line in lines:
        grouped.setdefault(line.page_id, []).append(line)
    for page_lines in grouped.values():
        page_lines.sort(key=lambda r: r.line_no)
    return dict(sorted(grouped.items()))


def label_with_page_context(
    *,
    pages_dir: Path,
    lines_dir: Path,
    out: Path,
    page_context_out: Path,
    model: str,
    page_model: str | None = None,
    sleep: float = 0.3,
    limit_pages: int = 0,
    limit_lines: int = 0,
    overwrite: bool = False,
) -> list[dict]:
    """Run page-context Gemini labeling and write local JSON outputs."""

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Add it to .env or your shell env.")

    from google import genai

    client = genai.Client(api_key=api_key)
    page_model = page_model or model

    existing_records = [] if overwrite else _read_json_list(out)
    done_files = {r.get("file") for r in existing_records if isinstance(r.get("file"), str)}
    page_contexts = {} if overwrite else _read_page_contexts(page_context_out)

    lines = load_lines(lines_dir)
    grouped = _group_lines_by_page(lines)
    if limit_pages:
        grouped = dict(list(grouped.items())[:limit_pages])

    records = list(existing_records)
    new_lines = 0

    log.info("pages to process: %d", len(grouped))
    for page_id, page_lines in grouped.items():
        page_path = pages_dir / f"{page_id}.png"
        if not page_path.exists():
            log.warning("missing page image for %s: %s", page_id, page_path)
            continue

        page_context = page_contexts.get(page_id)
        if not page_context:
            log.info("page context: %s", page_id)
            page_resp = _generate_json(client, model=page_model, image_path=page_path, prompt=PAGE_PROMPT)
            page_context = {
                "page_id": page_id,
                "page_image": page_path.as_posix(),
                "model": page_model,
                "prompt_version": PROMPT_VERSION,
                "text": str(page_resp.get("text", "")).strip(),
                "notes": page_resp.get("notes", ""),
            }
            page_contexts[page_id] = page_context
            _write_json(page_context_out, page_contexts)
            time.sleep(sleep)

        page_text = page_context.get("text", "")

        for line in page_lines:
            if line.rel_path in done_files:
                continue
            if limit_lines and new_lines >= limit_lines:
                _write_json(out, records)
                _write_json(page_context_out, page_contexts)
                log.info("limit reached. wrote %d total line records", len(records))
                return records

            prompt = LINE_PROMPT_TEMPLATE.format(page_text=page_text)
            log.info("line: %s", line.rel_path)
            try:
                line_resp = _generate_json(
                    client,
                    model=model,
                    image_path=line.image_path,
                    prompt=prompt,
                )
                text = str(line_resp.get("text", "")).strip()
                confidence = line_resp.get("confidence", None)
            except Exception as e:
                log.exception("failed line %s", line.rel_path)
                text = ""
                confidence = 0.0
                line_resp = {"notes": f"ERROR: {e}"}

            record = {
                "file": line.rel_path,
                "page_id": page_id,
                "line": line.line_no,
                "text": text,
                "confidence": confidence,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "page_context_file": page_context_out.as_posix(),
                "page_context_text": page_text,
                "notes": line_resp.get("notes", ""),
            }
            records.append(record)
            done_files.add(line.rel_path)
            new_lines += 1

            # Persist after every line, so long jobs are resumable.
            _write_json(out, records)
            time.sleep(sleep)

    _write_json(out, records)
    _write_json(page_context_out, page_contexts)
    log.info("done. total records: %d, new records: %d", len(records), new_lines)
    return records


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages-dir", required=True, type=Path)
    ap.add_argument("--lines-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--page-context-out",
        type=Path,
        default=None,
        help="where to store whole-page Gemini transcripts; default: <out>.pages.json",
    )
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--page-model", default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    page_context_out = args.page_context_out
    if page_context_out is None:
        page_context_out = args.out.with_suffix(".pages.json")

    label_with_page_context(
        pages_dir=args.pages_dir,
        lines_dir=args.lines_dir,
        out=args.out,
        page_context_out=page_context_out,
        model=args.model,
        page_model=args.page_model,
        sleep=args.sleep,
        limit_pages=args.limit_pages,
        limit_lines=args.limit_lines,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()

