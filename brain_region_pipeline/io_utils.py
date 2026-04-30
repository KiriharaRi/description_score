"""Small JSON and hashing helpers shared across the pipeline."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from disk."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk with UTF-8 and indentation."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON into a list of dicts."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write newline-delimited JSON rows."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_json_hash(payload: Any) -> str:
    """Return a short stable hash for any JSON-serializable payload."""

    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(text.encode("utf-8")).hexdigest()[:16]

