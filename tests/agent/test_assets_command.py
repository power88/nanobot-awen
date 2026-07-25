from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent import loop as loop_module
from nanobot.bus.events import InboundMessage
from nanobot.command.builtin import (
    build_help_text,
    builtin_command_palette,
    register_builtin_commands,
)
from nanobot.command.router import CommandContext, CommandRouter


def _message(raw: str, **kwargs) -> InboundMessage:
    return InboundMessage(
        channel=kwargs.pop("channel", "websocket"),
        sender_id=kwargs.pop("sender_id", "user-1"),
        chat_id="chat-1",
        content=raw,
        metadata=kwargs.pop("metadata", {}),
        **kwargs,
    )


def _loop(loop_factory, **kwargs):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    return loop_factory(provider=provider, **kwargs)


async def _dispatch(loop, raw: str, **msg_kwargs):
    router = CommandRouter()
    register_builtin_commands(router)
    msg = _message(raw, **msg_kwargs)
    return await router.dispatch(
        CommandContext(
            msg=msg,
            session=None,
            key=loop._effective_session_key(msg),
            raw=raw,
            loop=loop,
        )
    )


async def _buffer(loop, paths: list[Path]) -> None:
    await loop._route_media_collection(
        _message("", media=[str(path) for path in paths])
    )
    await loop.bus.consume_outbound()


def test_assets_is_in_help_and_command_palette() -> None:
    item = next(
        entry for entry in builtin_command_palette()
        if entry["command"] == "/assets"
    )
    assert item["arg_hint"] == "[list|rm <numbers|all>]"
    assert item["lifecycle"] == "side_channel"
    assert item["accepts_args"] is True
    assert "/assets [list|rm <numbers|all>]" in build_help_text()


def test_asset_formatter_hides_windows_and_posix_parent_paths() -> None:
    from nanobot.utils.media import format_media_list

    assert format_media_list([
        r"C:\private\uploads\photo.png",
        "/private/uploads/clip.mp4",
    ]) == "1. photo.png\n2. clip.mp4"


@pytest.mark.asyncio
async def test_assets_help_and_empty_operations(loop_factory) -> None:
    loop = _loop(loop_factory)

    help_out = await _dispatch(loop, "/assets")
    list_out = await _dispatch(loop, "/assets list")
    remove_out = await _dispatch(loop, "/assets rm all")

    assert "/assets list" in help_out.content
    assert list_out.content == "素材缓冲区为空。"
    assert remove_out.content == "素材缓冲区为空。"


@pytest.mark.asyncio
async def test_assets_list_remove_duplicates_and_renumber(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    paths = [tmp_path / name for name in ("one.png", "two.mp4", "three.ogg")]
    for path in paths:
        path.write_bytes(b"asset")
    await _buffer(loop, paths)

    listed = await _dispatch(loop, "/assets list")
    removed = await _dispatch(loop, "/assets rm 3, 1, 3")

    assert listed.content == (
        "当前共有 3 个素材：\n"
        "1. one.png\n2. two.mp4\n3. three.ogg"
    )
    assert removed.content == (
        "已移除：\n1. one.png\n2. three.ogg\n\n"
        "剩余 1 个素材：\n1. two.mp4"
    )
    assert loop._list_pending_media("websocket:chat-1") == [str(paths[1])]
    assert all(path.exists() for path in paths)


@pytest.mark.asyncio
async def test_assets_remove_accepts_spaces_around_commas(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    paths = [tmp_path / f"{index}.png" for index in range(1, 4)]
    for path in paths:
        path.write_bytes(b"asset")
    await _buffer(loop, paths)

    removed = await _dispatch(loop, "/assets rm 1 , 2")

    assert "1. 1.png\n2. 2.png" in removed.content
    assert loop._list_pending_media("websocket:chat-1") == [str(paths[2])]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["0", "-1", "four", "4"])
async def test_assets_invalid_indices_are_rejected_atomically(
    loop_factory, tmp_path, value
) -> None:
    loop = _loop(loop_factory)
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        path.write_bytes(b"asset")
    await _buffer(loop, paths)

    result = await _dispatch(loop, f"/assets rm {value}")

    assert "当前有效编号范围为 1-2" in result.content
    assert loop._list_pending_media("websocket:chat-1") == [str(path) for path in paths]


@pytest.mark.asyncio
async def test_assets_syntax_error_is_usage_and_does_not_remove(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    path = tmp_path / "one.png"
    path.write_bytes(b"asset")
    await _buffer(loop, [path])

    result = await _dispatch(loop, "/assets rm 1,,2")

    assert result.content.startswith("素材缓冲管理：")
    assert loop._list_pending_media("websocket:chat-1") == [str(path)]


@pytest.mark.asyncio
async def test_assets_clear_does_not_delete_files(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    paths = [tmp_path / "one.png", tmp_path / "two.mp4"]
    for path in paths:
        path.write_bytes(b"asset")
    await _buffer(loop, paths)

    result = await _dispatch(loop, "/assets rm all")

    assert result.content == "已清空 2 个素材。"
    assert loop._list_pending_media("websocket:chat-1") == []
    assert all(path.exists() for path in paths)


@pytest.mark.asyncio
async def test_assets_ttl_refresh_rules(loop_factory, tmp_path, monkeypatch) -> None:
    loop = _loop(loop_factory)
    paths = [tmp_path / "one.png", tmp_path / "two.png"]
    for path in paths:
        path.write_bytes(b"asset")
    now = 1000.0
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: now)
    await _buffer(loop, paths)
    original_expiry = loop._pending_media["websocket:chat-1"].expires_at

    now += 10
    await _dispatch(loop, "/assets rm nope")
    assert loop._pending_media["websocket:chat-1"].expires_at == original_expiry

    now += 10
    await _dispatch(loop, "/assets")
    assert loop._pending_media["websocket:chat-1"].expires_at == original_expiry

    now += 10
    await _dispatch(loop, "/assets list")
    assert loop._pending_media["websocket:chat-1"].expires_at == (
        now + loop_module._PENDING_MEDIA_TTL_SECONDS
    )

    now += 10
    await _dispatch(loop, "/assets rm 1")
    assert loop._pending_media["websocket:chat-1"].expires_at == (
        now + loop_module._PENDING_MEDIA_TTL_SECONDS
    )


@pytest.mark.asyncio
async def test_pending_media_query_returns_a_copy(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    path = tmp_path / "one.png"
    path.write_bytes(b"asset")
    await _buffer(loop, [path])

    paths = loop._list_pending_media("websocket:chat-1")
    paths.clear()

    assert loop._list_pending_media("websocket:chat-1") == [str(path)]


@pytest.mark.asyncio
async def test_assets_uses_unified_session_key_across_channels(
    loop_factory, tmp_path
) -> None:
    loop = _loop(loop_factory, unified_session=True)
    path = tmp_path / "one.png"
    path.write_bytes(b"asset")
    telegram = InboundMessage(
        channel="telegram",
        sender_id="user-1",
        chat_id="telegram-chat",
        content="",
        media=[str(path)],
    )
    await loop._route_media_collection(telegram)
    await loop.bus.consume_outbound()

    result = await _dispatch(loop, "/assets list")

    assert result.content == "当前共有 1 个素材：\n1. one.png"
    assert loop._list_pending_media("unified:default") == [str(path)]


@pytest.mark.asyncio
async def test_internal_assets_command_does_not_manage_buffer(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    path = tmp_path / "one.png"
    path.write_bytes(b"asset")
    await _buffer(loop, [path])

    result = await _dispatch(
        loop,
        "/assets rm all",
        channel="system",
        sender_id="system",
    )

    assert "仅适用于用户消息" in result.content
    assert loop._list_pending_media("websocket:chat-1") == [str(path)]
