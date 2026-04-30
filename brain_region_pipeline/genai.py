"""Google GenAI helpers for structured generation and annotation."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .config import GenerationConfig


def resolve_api_key() -> str:
    """Resolve Gemini API key from environment."""

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY before running.")
    return key


def create_genai_client():
    """Create a Google GenAI client, optionally honoring GEMINI_BASE_URL."""

    from google import genai
    from google.genai import types as genai_types

    base_url = os.environ.get("GEMINI_BASE_URL")
    if base_url:
        return genai.Client(
            api_key=resolve_api_key(),
            http_options=genai_types.HttpOptions(base_url=base_url),
        )
    return genai.Client(api_key=resolve_api_key())


def generate_structured_json(
    *,
    model: str,
    system_instruction: str,
    contents: list[Any],
    response_schema: dict[str, Any],
    cfg: GenerationConfig,
) -> dict[str, Any]:
    """Generate JSON with retries and schema enforcement."""

    from google.genai import types

    client = create_genai_client()
    last_error: Exception | None = None
    for attempt in range(cfg.max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=cfg.temperature,
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    system_instruction=system_instruction,
                    http_options=types.HttpOptions(
                        timeout=int(cfg.generation_timeout_s * 1000),
                    ),
                ),
            )
            text = getattr(response, "text", None)
            if text is None:
                raise RuntimeError(f"Gemini returned no text: {response!r}")
            return json.loads(text)
        except Exception as exc:
            last_error = exc
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.retry_delay_s * (2 ** attempt))

    raise RuntimeError(
        f"Gemini generation failed after {cfg.max_retries} attempts.",
    ) from last_error
