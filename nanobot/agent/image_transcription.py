"""Convert image content blocks to validated text for non-vision models."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import MutableMapping
from typing import Any
from urllib.parse import unquote_to_bytes

from loguru import logger

from nanobot.config.schema import Config
from nanobot.providers.factory import make_provider

DEFAULT_IMAGE_TRANSCRIPTION_PROMPT = """You are an image transcription engine. Analyze the supplied image and return
exactly one valid JSON object. Do not use Markdown fences and do not output
explanations outside the JSON.

The JSON schema is:
{
  "description": string,
  "contentType": "json" | "text",
  "content": object | string
}

"description" must give a concise, factual overview of the whole image. Do not
invent details that are not visible.

Use contentType "text" only when the image is essentially a text document and
discarding spatial layout would not change its meaning. In that case, "content"
must contain the visible text in natural reading order, preserving paragraphs,
headings, lists, and line breaks where useful. Do not summarize or translate it.

For screenshots, user interfaces, diagrams, charts, tables, photographs, or
mixed visual content, use contentType "json". The "content" value must be an
object with this shape:
{
  "coordinateSystem": "normalized-1000",
  "width": 1000,
  "height": 1000,
  "elements": [
    {
      "id": string,
      "type": "text" | "icon" | "image" | "shape" | "chart" | "table" |
              "control" | "container" | "other",
      "x": integer,
      "y": integer,
      "width": integer,
      "height": integer,
      "label": string (optional),
      "text": string (optional),
      "description": string (optional),
      "elements": array (optional)
    }
  ]
}

Use nested elements only for genuine visual or semantic containment. List
elements in visual reading order. Keep exact visible text in "text", semantic
roles in "label", and concise visual details in "description".

Keep all coordinates inside the image bounds. Do not infer hidden content.
Return exactly the JSON object and nothing else."""

IMAGE_TRANSCRIPTION_FAILURE_TEXT = (
    "图片不可用。请明确告知用户：“我目前无法查看这张图片，"
    "图片转写功能似乎出现了问题。”"
)
_TRANSCRIPTION_KEYS = frozenset({"description", "contentType", "content"})


class ImageTranscriptionError(ValueError):
    """The vision model returned an invalid transcription."""


def validate_image_transcription(raw: str | None) -> dict[str, Any]:
    """Validate the JSON transcription contract."""
    try:
        value = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ImageTranscriptionError("response is not valid JSON") from exc
    if not isinstance(value, dict) or set(value) != _TRANSCRIPTION_KEYS:
        raise ImageTranscriptionError("response must contain exactly the required fields")
    if not isinstance(value["description"], str) or not value["description"].strip():
        raise ImageTranscriptionError("description must not be empty")
    content_type = value["contentType"]
    content = value["content"]
    if content_type == "text":
        if not isinstance(content, str) or not content.strip():
            raise ImageTranscriptionError("text content must be a non-empty string")
    elif content_type == "json":
        if not isinstance(content, dict) or not content:
            raise ImageTranscriptionError("json content must be a non-empty object")
    else:
        raise ImageTranscriptionError("contentType must be 'json' or 'text'")
    return value


class ImageTranscriber:
    """Try configured vision presets in order and replace image blocks in-place."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._settings = config.image_transcription
        self._providers: dict[str, Any] = {}

    async def replace_images(
        self,
        messages: list[dict[str, Any]],
        cache: MutableMapping[str, str],
    ) -> bool:
        """Replace every image block, returning whether any image was found."""
        found = False
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for index, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                found = True
                replacement = await self._replacement_for_block(block, cache)
                content[index] = {"type": "text", "text": replacement}
        return found

    async def _replacement_for_block(
        self,
        block: dict[str, Any],
        cache: MutableMapping[str, str],
    ) -> str:
        url = (block.get("image_url") or {}).get("url")
        if not isinstance(url, str) or not url:
            return IMAGE_TRANSCRIPTION_FAILURE_TEXT
        try:
            cache_key, image_size = _image_identity(url)
        except ValueError:
            return IMAGE_TRANSCRIPTION_FAILURE_TEXT
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if image_size is not None and image_size > self._settings.max_image_mb * 1024 * 1024:
            cache[cache_key] = IMAGE_TRANSCRIPTION_FAILURE_TEXT
            return cache[cache_key]

        logger.info(
            "Image transcription started ({} preset(s))",
            len(self._settings.model_presets),
        )
        result = await self._transcribe(url)
        logger.info(
            "Image transcription completed ({})",
            "success" if result is not None else "failed",
        )
        if result is None:
            replacement = IMAGE_TRANSCRIPTION_FAILURE_TEXT
        else:
            replacement = (
                "[Image transcription]\n"
                + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                + "\n[/Image transcription]"
            )
        cache[cache_key] = replacement
        return replacement

    async def _transcribe(self, url: str) -> dict[str, Any] | None:
        prompt = self._settings.system_prompt or DEFAULT_IMAGE_TRANSCRIPTION_PROMPT
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this image."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            },
        ]
        for preset_name in self._settings.model_presets:
            try:
                provider = self._providers.get(preset_name)
                if provider is None:
                    provider = make_provider(
                        self._config,
                        preset_name=preset_name,
                        enable_fallbacks=False,
                    )
                    self._providers[preset_name] = provider
                preset = self._config.model_presets[preset_name]
                response = await provider.chat_with_retry(
                    messages=messages,
                    tools=None,
                    model=preset.model,
                    temperature=preset.temperature,
                    max_tokens=preset.max_tokens,
                    reasoning_effort=preset.reasoning_effort,
                    allow_image_stripping=False,
                )
                if response.finish_reason == "error":
                    raise ImageTranscriptionError(response.content or "vision request failed")
                return validate_image_transcription(response.content)
            except Exception as exc:
                logger.warning(
                    "Image transcription preset {!r} failed: {}",
                    preset_name,
                    exc,
                )
        return None


def _image_identity(url: str) -> tuple[str, int | None]:
    if not url.startswith("data:"):
        encoded = url.encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), None
    try:
        header, payload = url.split(",", 1)
    except ValueError as exc:
        raise ValueError("invalid data URL") from exc
    try:
        raw = (
            base64.b64decode(payload, validate=True)
            if ";base64" in header.lower()
            else unquote_to_bytes(payload)
        )
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid image data URL") from exc
    return hashlib.sha256(raw).hexdigest(), len(raw)
