"""Small I/O helpers shared across the pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Iterator


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[dict], path: str | Path, *, append: bool = False) -> None:
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open(mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def page_id_from_filename(name: str) -> str:
    """Strip page suffix (`_p0001`) and extension to get the source-document id."""
    stem = Path(name).stem
    if "_p" in stem:
        stem = stem.rsplit("_p", 1)[0]
    return stem
