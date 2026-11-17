"""OpenRouter text and speech requests using the official OpenAI Python client."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, OpenAI

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class OpenRouterError(RuntimeError):
    """A safe, actionable error without raw provider bodies or credentials."""


def make_client(*, timeout: float = 45.0, http_client=None) -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Create a key at https://openrouter.ai/keys "
            "and set it in your local environment. Do not put it in code or chat."
        )
    if not math.isfinite(timeout) or timeout <= 0:
        raise OpenRouterError("Timeout must be a finite positive number of seconds.")
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
        timeout=timeout,
        max_retries=0,
        default_headers={"X-OpenRouter-Title": "HebrewDeepFake"},
        http_client=http_client,
    )


def _validate_input(text: str, model: str) -> None:
    if not text.strip():
        raise OpenRouterError("Input text must not be empty.")
    provider, separator, name = model.partition("/")
    if not separator or not provider or not name or any(c.isspace() for c in model):
        raise OpenRouterError("Use a model slug such as openai/gpt-4o-mini.")


def _safe_api_error(error: APIError) -> OpenRouterError:
    if isinstance(error, APITimeoutError):
        return OpenRouterError("OpenRouter timed out. Try again or increase --timeout.")
    if isinstance(error, APIConnectionError):
        return OpenRouterError("Cannot reach OpenRouter. Check the network, proxy and TLS configuration.")
    if isinstance(error, APIStatusError):
        status = error.status_code
        guidance = {
            400: "Check the model, voice, input and output format.",
            401: "Check OPENROUTER_API_KEY; it may be invalid or revoked.",
            402: "Add OpenRouter credits or check the key's spending limit.",
            403: "The key, provider or model is not permitted for this request.",
            404: "Check the exact model slug and endpoint in the current catalog.",
            408: "The upstream provider timed out; retry later.",
            422: "The selected model does not accept these request parameters.",
            429: "Rate limited; wait before making another request.",
        }.get(status, "The service failed; retry later." if status >= 500 else "The request was rejected.")
        return OpenRouterError(f"OpenRouter HTTP {status}: {guidance}")
    return OpenRouterError("OpenRouter returned an invalid API response. Check the selected model and retry.")


def request_text(
    prompt: str,
    *,
    model: str | None = None,
    max_tokens: int = 128,
    timeout: float = 45.0,
    http_client=None,
) -> dict[str, Any]:
    selected = (model if model is not None else os.environ.get("OPENROUTER_MODEL") or DEFAULT_MODEL).strip()
    _validate_input(prompt, selected)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise OpenRouterError("--max-tokens must be a positive integer.")
    try:
        with make_client(timeout=timeout, http_client=http_client) as client:
            response = client.chat.completions.create(
                model=selected,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        choices = getattr(response, "choices", None)
        if not choices or not choices[0].message or not choices[0].message.content:
            raise OpenRouterError("The model returned no text. Check its output modality or increase --max-tokens.")
        return {
            "requested_model": selected,
            "returned_model": response.model,
            "generation_id": response.id,
            "text": choices[0].message.content,
            "finish_reason": choices[0].finish_reason,
            "usage": response.usage.model_dump() if response.usage else None,
        }
    except APIError as error:
        raise _safe_api_error(error) from None


def request_speech(
    text: str,
    *,
    model: str,
    voice: str,
    output: str | Path,
    response_format: str = "mp3",
    timeout: float = 90.0,
    extra_body: dict[str, Any] | None = None,
    http_client=None,
) -> dict[str, Any]:
    _validate_input(text, model)
    if not voice.strip():
        raise OpenRouterError("Choose a voice ID from the selected model's supported_voices catalog.")
    if response_format not in {"mp3", "pcm"}:
        raise OpenRouterError("The speech example supports mp3 or raw pcm output.")
    destination = Path(output)
    if destination.suffix.lower() != "." + response_format:
        raise OpenRouterError("The output extension must match --format (pcm is raw PCM, not WAV).")
    if destination.exists() or destination.with_suffix(destination.suffix + ".json").exists():
        raise OpenRouterError("Output already exists. Choose a new --output path.")
    try:
        with make_client(timeout=timeout, http_client=http_client) as client:
            with client.audio.speech.with_streaming_response.create(
                model=model, input=text, voice=voice, response_format=response_format,
                extra_body=extra_body,
            ) as response:
                content_type = response.headers.get("content-type", "").split(";")[0].lower()
                if not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
                    raise OpenRouterError("OpenRouter returned a non-audio response; no recording was saved.")
                parts = []; size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise OpenRouterError("Audio exceeded the 20 MiB sample safety limit; no file was saved.")
                    parts.append(chunk)
                audio = b''.join(parts)
                generation_id = response.headers.get("x-generation-id")
        if not audio:
            raise OpenRouterError("OpenRouter returned empty audio; no recording was saved.")
        import hashlib
        metadata = {
            "model": model, "voice": voice, "response_format": response_format,
            "input_text": text, "generation_id": generation_id, "content_type": content_type,
            "bytes": len(audio), "sha256": hashlib.sha256(audio).hexdigest(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": BASE_URL + "/audio/speech", "voice_cloning": False,
            "request_options": extra_body,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as file:
            file.write(audio)
        destination.with_suffix(destination.suffix + ".json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {**metadata, "output": str(destination)}
    except APIError as error:
        raise _safe_api_error(error) from None

