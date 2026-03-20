"""Resolve read_file results that reference workspace images via a vision LLM pass."""

from __future__ import annotations

import base64
from pathlib import Path

from nanobot.agent.tools.filesystem import unpack_read_file_image_payload
from nanobot.providers.base import LLMProvider

_MAX_IMAGE_BYTES = 12 * 1024 * 1024


async def expand_read_file_tool_result(
    provider: LLMProvider,
    model: str | None,
    tool_name: str,
    result: str,
) -> str:
    """If result is an internal image marker from read_file, run one vision call and return text."""
    if tool_name != "read_file":
        return result
    parsed = unpack_read_file_image_payload(result)
    if not parsed:
        return result
    mime, file_path = parsed
    p = Path(file_path)
    if not p.is_file():
        return f"Error: image path no longer exists: {file_path}"
    raw = p.read_bytes()
    if len(raw) > _MAX_IMAGE_BYTES:
        return f"Error: image too large ({len(raw)} bytes) for vision; max {_MAX_IMAGE_BYTES}."

    b64 = base64.b64encode(raw).decode("ascii")
    url = f"data:{mime};base64,{b64}"
    vision_user: list[dict] = [
        {
            "type": "text",
            "text": (
                "Briefly describe this image for another assistant answering the user. "
                "Cover: main subjects, what they appear to be doing, setting/environment, "
                "and any prominent readable text. Be factual; if uncertain say so."
            ),
        },
        {"type": "image_url", "image_url": {"url": url}, "_meta": {"path": str(p)}},
    ]
    messages = [
        {"role": "system", "content": "You write concise, neutral image descriptions only."},
        {"role": "user", "content": vision_user},
    ]
    resp = await provider.chat_with_retry(messages=messages, tools=None, model=model)
    if resp.finish_reason == "error" or not (resp.content or "").strip():
        err = (resp.content or "unknown error")[:400]
        return (
            f"[Image: {p.name}] Vision analysis failed ({err}). "
            "Use a vision-capable chat model, or send the picture as a channel attachment."
        )
    return f"[Image: {p.name} — vision summary]\n{resp.content.strip()}"
