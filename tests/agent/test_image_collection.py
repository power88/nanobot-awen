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
async def test_attachment_messages_accumulate_and_ack(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    first = tmp_path / "first.png"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"\x89PNG\r\n\x1a\n")
    second.write_bytes(b"video")

    assert await loop._route_media_collection(_message(media=[str(first)])) is None
    assert await loop._route_media_collection(
        _message(content=f"[video: {second}]", media=[str(second)])
    ) is None

    first_ack = await loop.bus.consume_outbound()
    second_ack = await loop.bus.consume_outbound()
    assert first_ack.content == (
        "已添加 1 个附件，发送附件继续添加"
        "\n当前附件列表:\n1. first.png"
        "\n使用`/assets`管理附件"
    )
    assert second_ack.content == (
        "已添加 2 个附件，发送附件继续添加"
        "\n当前附件列表:\n1. first.png\n2. second.mp4"
        "\n使用`/assets`管理附件"
    )
    assert second_ack.metadata == {"request_id": "req-1"}


@pytest.mark.asyncio
async def test_transcribed_audio_still_waits_for_user_text(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"audio")

    assert await loop._route_media_collection(
        _message(content="[transcription: 帮我整理会议记录]", media=[str(audio)])
    ) is None

    merged = await loop._route_media_collection(_message(content="提炼行动项"))
    assert merged is not None
    assert merged.content == "提炼行动项"
    assert merged.media == [str(audio)]


@pytest.mark.asyncio
async def test_text_message_merges_and_clears_pending_media(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    pending = tmp_path / "pending.png"
    attached = tmp_path / "attached.png"
    pending.write_bytes(b"\x89PNG\r\n\x1a\n")
    attached.write_bytes(b"\x89PNG\r\n\x1a\n")

    await loop._route_media_collection(_message(media=[str(pending)]))
    merged = await loop._route_media_collection(
        _message(content="比较这两张图", media=[str(attached)])
    )

    assert merged is not None
    assert merged.content == "比较这两张图"
    assert merged.media == [str(pending), str(attached)]
    assert loop._pending_media == {}


@pytest.mark.asyncio
async def test_pending_media_is_isolated_by_session(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    await loop._route_media_collection(_message(media=[str(image)], chat_id="chat-1"))
    other = await loop._route_media_collection(_message(content="hello", chat_id="chat-2"))

    assert other is not None
    assert other.media == []
    assert "websocket:chat-1" in loop._pending_media


@pytest.mark.asyncio
async def test_internal_message_does_not_consume_pending_media(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    await loop._route_media_collection(_message(media=[str(image)]))

    internal = _message(content="subagent result")
    internal.channel = "system"
    internal.session_key_override = "websocket:chat-1"
    routed = await loop._route_media_collection(internal)

    assert routed is internal
    assert loop._pending_media["websocket:chat-1"].paths == [str(image)]


@pytest.mark.asyncio
async def test_pending_media_expires(loop_factory, tmp_path, monkeypatch) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    now = 1000.0
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: now)

    await loop._route_media_collection(_message(media=[str(image)]))
    loop._expire_pending_media(now + loop_module._PENDING_MEDIA_TTL_SECONDS)

    assert loop._pending_media == {}
    text = await loop._route_media_collection(_message(content="too late"))
    assert text is not None
    assert text.media == []


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["brief.pdf", "brief.docx"])
async def test_document_is_buffered_until_user_sends_text(
    loop_factory, tmp_path, filename
) -> None:
    loop = _loop(loop_factory)
    video = tmp_path / "clip.mp4"
    document = tmp_path / filename
    video.write_bytes(b"video")
    document.write_bytes(b"document")

    await loop._route_media_collection(_message(media=[str(video)]))
    routed = await loop._route_media_collection(
        _message(content=f"[file: {document.name}]", media=[str(document)])
    )

    assert routed is None
    assert loop._pending_media["websocket:chat-1"].paths == [
        str(video),
        str(document),
    ]

    merged = await loop._route_media_collection(_message(content="一起分析"))
    assert merged is not None
    assert merged.media == [str(video), str(document)]
    assert loop._pending_media == {}


@pytest.mark.asyncio
async def test_mixed_document_and_media_buffers_all_attachments(
    loop_factory, tmp_path
) -> None:
    loop = _loop(loop_factory)
    audio = tmp_path / "recording.ogg"
    document = tmp_path / "brief.pdf"
    audio.write_bytes(b"audio")
    document.write_bytes(b"%PDF")

    routed = await loop._route_media_collection(
        _message(
            content=f"[audio: {audio}]\n[file: {document.name}]",
            media=[str(audio), str(document)],
        )
    )

    assert routed is None
    assert loop._pending_media["websocket:chat-1"].paths == [
        str(audio),
        str(document),
    ]

    merged = await loop._route_media_collection(_message(content="一起分析"))
    assert merged is not None
    assert merged.media == [str(audio), str(document)]
    assert loop._pending_media == {}


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
        assert ack.content == (
            "已添加 1 个附件，发送附件继续添加"
            "\n当前附件列表:\n1. pending.png"
            "\n使用`/assets`管理附件"
        )
        loop._dispatch.assert_not_awaited()

        await loop.bus.publish_inbound(_message(content="描述这张图片"))
        await asyncio.wait_for(dispatched.wait(), timeout=2)
        assert captured[0].content == "描述这张图片"
        assert captured[0].media == [str(image)]
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)


@pytest.mark.asyncio
async def test_registered_command_does_not_consume_pending_media(
    loop_factory, tmp_path
) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    await loop._route_media_collection(_message(media=[str(image)]))
    await loop.bus.consume_outbound()

    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_message(content="/help"))
        response = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=2)

        assert "/assets [list|rm <numbers|all>]" in response.content
        assert loop._list_pending_media("websocket:chat-1") == [str(image)]
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)


@pytest.mark.asyncio
async def test_new_clears_and_reports_pending_media(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    await loop._route_media_collection(_message(media=[str(image)]))
    await loop.bus.consume_outbound()

    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_message(content="/new"))
        response = await asyncio.wait_for(loop.bus.consume_outbound(), timeout=2)

        assert response.content == "New session started. Cleared 1 buffered asset(s)."
        assert loop._list_pending_media("websocket:chat-1") == []
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)


@pytest.mark.asyncio
async def test_unknown_slash_command_submits_pending_media(loop_factory, tmp_path) -> None:
    loop = _loop(loop_factory)
    image = tmp_path / "pending.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    await loop._route_media_collection(_message(media=[str(image)]))
    await loop.bus.consume_outbound()
    dispatched = asyncio.Event()
    captured: list[InboundMessage] = []

    async def capture_dispatch(msg: InboundMessage) -> None:
        captured.append(msg)
        dispatched.set()

    loop._dispatch = AsyncMock(side_effect=capture_dispatch)
    run_task = asyncio.create_task(loop.run())
    try:
        await loop.bus.publish_inbound(_message(content="/unknown"))
        await asyncio.wait_for(dispatched.wait(), timeout=2)

        assert captured[0].content == "/unknown"
        assert captured[0].media == [str(image)]
        assert loop._list_pending_media("websocket:chat-1") == []
    finally:
        loop.stop()
        await asyncio.wait_for(run_task, timeout=2)
