"""Formatting helpers for user-visible media lists."""

from __future__ import annotations


def _media_filename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def format_media_list(paths: list[str]) -> str:
    """Format media paths as a numbered filename-only list."""
    return "\n".join(
        f"{index}. {_media_filename(path)}"
        for index, path in enumerate(paths, start=1)
    )
