"""Output helpers for TR-aligned feature rows."""

from __future__ import annotations

from pathlib import Path

from .io_utils import write_jsonl
from .models import TRFeatureRow


def save_readable_tr_rows(output_dir: Path, tr_rows: list[TRFeatureRow]) -> None:
    """Save a compact readable view for manual TR-level inspection."""

    rows = []
    for row in tr_rows:
        rows.append(
            {
                "tr_index": row.tr_index,
                "tr_time": f"[{row.tr_start_s:.2f}, {row.tr_end_s:.2f})",
                "module_descriptions": row.module_descriptions,
                "weights": row.weights,
            },
        )
    write_jsonl(output_dir / "tr_descriptions_readable.jsonl", rows)
