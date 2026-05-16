"""Transcribe line crops with Gemini using whole-page context.

Workflow:

1. For each page folder in ``data/lines_merged`` load the matching full page
   image from ``data/pages_clean/<page_id>.png``.
2. Send ONE Gemini request containing:
      * the whole page image
      * all line crop images for that page
      * the expected line ids / file paths
3. Save the returned full-page transcript and one local JSON record per line.
   No training happens here.

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

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
except ImportError:  # allows --dry-run before optional deps are installed
    def retry(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

    def retry_if_exception_type(*args, **kwargs):
        return None

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

from src.geminilabel.connect_transcripts import LineRecord, load_lines
from src.utils.io import setup_logging

log = logging.getLogger(__name__)

PROMPT_VERSION = "page-batch-v1"

BATCH_PROMPT_TEMPLATE = """You are an OCR system for handwritten Polish letters.

You will receive:
1. One full-page image.
2. Then {line_count} cropped line images from that same page.

The cropped line images are given in this exact order:
{line_manifest}

Tasks:
1. Transcribe the full page as best as possible.
2. Transcribe EACH line crop separately.

Rules:
- Preserve original spelling, punctuation, capitalization, and Polish diacritics
  (ą ć ę ł ń ó ś ź ż and capitals).
- Do NOT modernize, normalize, translate, or correct grammar.
- Use the full page as context, but each line transcription must contain only
  the text visible in that specific line crop.
- Do NOT copy words from nearby lines if they are not visible in the current
  crop.
- If a word is crossed out, wrap it in <strike>...</strike>.
- If a character is illegible, write '?'.
- If a crop is blank or contains no readable text, return an empty text.

Return STRICT JSON:
{{
  "page_text": "<full page transcription>",
  "notes": "<optional page-level note>",
  "lines": [
    {{
      "file": "<exact file path from the manifest>",
      "line": 1,
      "text": "<line transcription>",
      "confidence": <float 0..1>,
      "notes": "<optional short note>"
    }}
  ]
}}

The JSON array must contain exactly {line_count} line objects, one for every
line crop in the manifest, in the same order."""


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


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=30),
    retry=retry_if_exception_type(TransientGeminiError),
)
def _generate_page_batch_json(
    client,
    *,
    model: str,
    page_path: Path,
    lines: list[LineRecord],
    prompt: str,
) -> dict:
    """Send full page + all line crops for one page in a single request."""
    from google.genai import types

    contents: list[object] = [
        "FULL PAGE IMAGE:",
        types.Part.from_bytes(data=page_path.read_bytes(), mime_type="image/png"),
        prompt,
    ]
    for line in lines:
        contents.extend(
            [
                f"LINE {line.line_no:03d} FILE {line.rel_path}:",
                types.Part.from_bytes(data=line.image_path.read_bytes(), mime_type="image/png"),
            ]
        )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
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
    dry_run: bool = False,
) -> list[dict]:
    """Run page-batch Gemini labeling and write local JSON outputs."""

    client = None
    if not dry_run:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            log.info("python-dotenv not installed; reading GEMINI_API_KEY from shell env only")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is not set. Add it to .env or your shell env.")

        from google import genai

        client = genai.Client(api_key=api_key)
    if page_model and page_model != model:
        log.warning("--page-model is ignored in batch mode; using --model=%s", model)

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

        missing_lines = [line for line in page_lines if line.rel_path not in done_files]
        if not missing_lines:
            log.info("skip page (already complete): %s", page_id)
            continue
        if limit_lines:
            remaining = limit_lines - new_lines
            if remaining <= 0:
                _write_json(out, records)
                _write_json(page_context_out, page_contexts)
                log.info("limit reached. wrote %d total line records", len(records))
                return records
            missing_lines = missing_lines[:remaining]

        log.info(
            "planned request: page=%s all_lines=%d already_done=%d to_send=%d",
            page_id,
            len(page_lines),
            len(page_lines) - len(missing_lines),
            len(missing_lines),
        )
        if dry_run:
            for line in missing_lines[:10]:
                log.info("  would send line %03d: %s", line.line_no, line.rel_path)
            if len(missing_lines) > 10:
                log.info("  ... and %d more lines", len(missing_lines) - 10)
            continue

        line_manifest = "\n".join(
            f"- line={line.line_no} file={line.rel_path}" for line in missing_lines
        )
        prompt = BATCH_PROMPT_TEMPLATE.format(
            line_count=len(missing_lines),
            line_manifest=line_manifest,
        )

        log.info(
            "sending Gemini page batch: %s (1 page image + %d line images)",
            page_id,
            len(missing_lines),
        )
        try:
            batch_resp = _generate_page_batch_json(
                client,
                model=model,
                page_path=page_path,
                lines=missing_lines,
                prompt=prompt,
            )
        except Exception as e:
            log.exception("failed page batch %s", page_id)
            page_contexts[page_id] = {
                "page_id": page_id,
                "page_image": page_path.as_posix(),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "text": "",
                "notes": f"ERROR: {e}",
            }
            _write_json(page_context_out, page_contexts)
            time.sleep(sleep)
            continue

        page_text = str(batch_resp.get("page_text", "")).strip()
        page_contexts[page_id] = {
            "page_id": page_id,
            "page_image": page_path.as_posix(),
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "text": page_text,
            "notes": batch_resp.get("notes", ""),
        }
        _write_json(page_context_out, page_contexts)

        response_lines = batch_resp.get("lines", [])
        if not isinstance(response_lines, list):
            log.warning("page %s returned no JSON list under key 'lines'", page_id)
            response_lines = []
        log.info("Gemini returned %d line records for page %s", len(response_lines), page_id)

        by_file: dict[str, dict] = {}
        by_line_no: dict[int, dict] = {}
        for item in response_lines:
            if not isinstance(item, dict):
                continue
            file_value = item.get("file")
            if isinstance(file_value, str):
                by_file[file_value] = item
                by_file[Path(file_value).name] = item
            line_value = item.get("line")
            if isinstance(line_value, int):
                by_line_no[line_value] = item
            elif isinstance(line_value, str) and line_value.isdigit():
                by_line_no[int(line_value)] = item

        for line in missing_lines:
            line_resp = by_file.get(line.rel_path) or by_file.get(line.file) or by_line_no.get(line.line_no)
            if line_resp is None:
                log.warning("missing response for %s", line.rel_path)
                line_resp = {
                    "text": "",
                    "confidence": 0.0,
                    "notes": "Missing from Gemini page-batch response",
                }

            record = {
                "file": line.rel_path,
                "page_id": page_id,
                "line": line.line_no,
                "text": str(line_resp.get("text", "")).strip(),
                "confidence": line_resp.get("confidence", None),
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "page_context_file": page_context_out.as_posix(),
                "page_context_text": page_text,
                "notes": line_resp.get("notes", ""),
            }
            records.append(record)
            done_files.add(line.rel_path)
            new_lines += 1

        # Persist after every page batch, so the job is resumable while using
        # only one API request per page.
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
    ap.add_argument("--page-model", default=None, help="ignored in page-batch mode")
    ap.add_argument("--sleep", type=float, default=0.3)
    ap.add_argument("--limit-pages", type=int, default=0)
    ap.add_argument("--limit-lines", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="do not call Gemini; print which pages/lines would be sent",
    )
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
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

