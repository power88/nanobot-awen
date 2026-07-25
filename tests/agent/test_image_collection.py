from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent import loop as loop_module
from nanobot.bus.events import InboundMessage


def _message(*, content: str = "", media: list[str] | None = None, chat_id: str = "chat-1"):
    return InboundMessage(
        channel="websocket",
        sender_id="user-1",
        chat_id=chat_id,
        content=content,
        media=media or [],
        metadata={"request_id": "req-1"},
    )


def _loop(loop_factory):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    return loop_factory(provider=provider)


@pytest.mark.asyncio
async def test_image_only_messages_accumulate_and_ack(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    first = tmp_path / "first.png"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"\xff\xd8\xff")

    assert await loop._route_image_collection(_message(media=[str(first)])) is None
    assert await loop._route_image_collection(
        _message(content=f"[image: {second}]", media=[str(second)])
    ) is None

    first_ack = await loop.bus.consume_outbound()
    second_ack = await loop.bus.consume_outbound()
    assert first_ack.content == "已添加 1 张图片，发送图片继续添加"
    assert second_ack.content == "已添加 2 张图片，发送图片继续添加"
    assert second_ack.metadata == {"request_id": "req-1"}


@pytest.mark.asyncio
async def test_text_message_merges_and_clears_pending_images(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    pending = tmp_path / "pending.png"
    attached = tmp_path / "attached.png"
    pending.write_bytes(b"\x89PNG\r\n\x1a\n")
    attached.write_bytes(b"\x89PNG\r\n\x1a\n")

    await loop._route_image_collection(_message(media=[str(pending)]))
    merged = await loop._route_image_collection(
        _message(content="比较这两张图", media=[str(attached)])
    )

    assert merged is not None
    assert merged.content == "比较这两张图"
    assert merged.media == [str(pending), str(attached)]
    assert loop._pending_images == {}


@pytest.mark.asyncio
async def test_pending_images_are_isolated_by_session(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    await loop._route_image_collection(_message(media=[str(image)], chat_id="chat-1"))
    other = await loop._route_image_collection(_message(content="hello", chat_id="chat-2"))

    assert other is not None
    assert other.media == []
    assert "websocket:chat-1" in loop._pending_images


@pytest.mark.asyncio
async def test_internal_message_does_not_consume_pending_images(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    await loop._route_image_collection(_message(media=[str(image)]))

    internal = _message(content="subagent result")
    internal.channel = "system"
    internal.session_key_override = "websocket:chat-1"
    routed = await loop._route_image_collection(internal)

    assert routed is internal
    assert loop._pending_images["websocket:chat-1"].paths == [str(image)]


@pytest.mark.asyncio
async def test_pending_images_expire(loop_factory, tmp_path, monkeypatch) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    now = 1000.0
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: now)

    await loop._route_image_collection(_message(media=[str(image)]))
    loop._expire_pending_images(now + loop_module._PENDING_IMAGE_TTL_SECONDS)

    assert loop._pending_images == {}
    text = await loop._route_image_collection(_message(content="too late"))
    assert text is not None
    assert text.media == []


@pytest.mark.asyncio
async def test_run_routes_collection_before_agent_dispatch(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    dispatched = asyncio.Event()
    captured: list[InboundMessage] = []

    async def capture_dispatch(msg: InboundMessage) -> None:
        captured.append(msg)
        dispatched.set()

    loop._dispatch = AsyncMock(side_effect=capture_dispatch)
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_message(media=[str(image)]))
        ack = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=2)
        assert ack.content == "已添加 1 张图片，发送图片继续添加"
        loop._dispatch.assert_not_awaited()

        await loop.bus.publish_inbound(_message(content="描述这张图片"))
        await asyncio.wait_for(dispatched.wait(), timeout=2)
        assert captured[0].content == "描述这张图片"
        assert captured[0].media == [str(image)]
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)
