from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nanobot.agent.image_transcription import (
    IMAGE_TRANSCRIPTION_FAILURE_TEXT,
    ImageTranscriber,
    ImageTranscriptionError,
    validate_image_transcription,
)
from nanobot.agent.runner import AgentRunner, AgentRunSpec
from nanobot.config.schema import Config
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
)
from nanobot.utils.llm_runtime import LLMRuntime


def _config(**image_transcription) -> Config:
    return Config.model_validate(
        {
            "imageTranscription": {
                "enabled": True,
                "modelPresets": ["vision-a", "vision-b"],
                **image_transcription,
            },
            "agents": {"defaults": {"supportsVision": False}},
            "modelPresets": {
                "vision-a": {
                    "model": "provider/vision-a",
                    "provider": "openai",
                    "supportsVision": True,
                },
                "vision-b": {
                    "model": "provider/vision-b",
                    "provider": "openai",
                    "supportsVision": True,
                },
            },
        }
    )


def _data_url(data: bytes = b"image") -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


def _image_block(data: bytes = b"image") -> dict:
    return {"type": "image_url", "image_url": {"url": _data_url(data)}}


def _json_payload() -> str:
    return json.dumps(
        {
            "description": "A title",
            "contentType": "json",
            "content": {
                "coordinateSystem": "normalized-1000",
                "width": 1000,
                "height": 1000,
                "elements": [
                    {
                        "id": "e1",
                        "type": "text",
                        "x": 10,
                        "y": 20,
                        "width": 100,
                        "height": 50,
                        "text": "Nanobot",
                    }
                ],
            },
        }
    )


def test_config_accepts_aliases_and_preserves_vision_defaults() -> None:
    config = _config(maxImageMb=9, systemPrompt="custom")

    assert config.image_transcription.max_image_mb == 9
    assert config.image_transcription.system_prompt == "custom"
    assert config.agents.defaults.supports_vision is False
    assert Config().agents.defaults.supports_vision is True
    assert Config().image_transcription.enabled is False


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            {
                "imageTranscription": {"enabled": True, "modelPresets": []},
            },
            "must not be empty",
        ),
        (
            {
                "imageTranscription": {
                    "enabled": True,
                    "modelPresets": ["missing"],
                },
            },
            "not found",
        ),
        (
            {
                "imageTranscription": {
                    "enabled": True,
                    "modelPresets": ["vision"],
                },
                "modelPresets": {
                    "vision": {
                        "model": "x",
                        "supportsVision": False,
                    }
                },
            },
            "must support vision",
        ),
    ],
)
def test_config_rejects_invalid_transcription_presets(payload: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(payload)


def test_validate_text_and_json_transcriptions() -> None:
    text = validate_image_transcription(
        '{"description":"page","contentType":"text","content":"Title\\nBody"}'
    )
    structured = validate_image_transcription(_json_payload())

    assert text["content"] == "Title\nBody"
    assert structured["contentType"] == "json"
    assert structured["content"]["elements"][0]["text"] == "Nanobot"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        '{"description":"x","contentType":"text"}',
        '{"description":"x","contentType":"markdown","content":"x"}',
        '{"description":"x","contentType":"json","content":"not an object"}',
        '{"description":"x","contentType":"json","content":{}}',
    ],
)
def test_validate_rejects_invalid_outputs(raw: str) -> None:
    with pytest.raises(ImageTranscriptionError):
        validate_image_transcription(raw)


@pytest.mark.asyncio
async def test_presets_fall_back_and_duplicate_images_use_cache(monkeypatch) -> None:
    first = MagicMock()
    first.chat_with_retry = AsyncMock(return_value=LLMResponse(content="invalid"))
    second = MagicMock()
    second.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=_json_payload())
    )
    providers = {"vision-a": first, "vision-b": second}
    monkeypatch.setattr(
        "nanobot.agent.image_transcription.make_provider",
        lambda _config, *, preset_name, enable_fallbacks: providers[preset_name],
    )
    transcriber = ImageTranscriber(_config())
    messages = [
        {"role": "user", "content": [_image_block(), _image_block()]},
    ]

    found = await transcriber.replace_images(messages, {})

    assert found is True
    assert first.chat_with_retry.await_count == 1
    assert second.chat_with_retry.await_count == 1
    assert all(block["type"] == "text" for block in messages[0]["content"])
    assert messages[0]["content"][0] == messages[0]["content"][1]
    assert "Image transcription" in messages[0]["content"][0]["text"]
    assert second.chat_with_retry.await_args.kwargs["allow_image_stripping"] is False


@pytest.mark.asyncio
async def test_transcription_logs_start_and_completion(monkeypatch) -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content=_json_payload())
    )
    monkeypatch.setattr(
        "nanobot.agent.image_transcription.make_provider",
        lambda *_args, **_kwargs: provider,
    )
    info = MagicMock()
    monkeypatch.setattr("nanobot.agent.image_transcription.logger.info", info)

    await ImageTranscriber(_config()).replace_images(
        [{"role": "user", "content": [_image_block()]}],
        {},
    )

    assert info.call_count == 2
    assert info.call_args_list[0].args[0].startswith("Image transcription started")
    assert info.call_args_list[1].args == (
        "Image transcription completed ({})",
        "success",
    )


@pytest.mark.asyncio
async def test_all_presets_fail_and_oversized_images_become_placeholder(
    monkeypatch,
) -> None:
    provider = MagicMock()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(content="{}", finish_reason="error")
    )
    monkeypatch.setattr(
        "nanobot.agent.image_transcription.make_provider",
        lambda *_args, **_kwargs: provider,
    )
    transcriber = ImageTranscriber(_config(maxImageMb=1))
    messages = [
        {
            "role": "user",
            "content": [
                _image_block(b"small"),
                _image_block(b"x" * (1024 * 1024 + 1)),
            ],
        }
    ]

    await transcriber.replace_images(messages, {})

    assert provider.chat_with_retry.await_count == 2
    assert [block["text"] for block in messages[0]["content"]] == [
        IMAGE_TRANSCRIPTION_FAILURE_TEXT,
        IMAGE_TRANSCRIPTION_FAILURE_TEXT,
    ]


class _ScriptedProvider(LLMProvider):
    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.seen_messages: list[list[dict]] = []

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.seen_messages.append(messages)
        return self.responses.pop(0)

    def get_default_model(self) -> str:
        return "test"


class _StreamingProvider(_ScriptedProvider):
    def __init__(self, responses: list[tuple[LLMResponse, str | None]]) -> None:
        super().__init__([response for response, _delta in responses])
        self._deltas = [delta for _response, delta in responses]

    async def chat_stream(self, messages, on_content_delta=None, **_kwargs):
        self.seen_messages.append(messages)
        response = self.responses.pop(0)
        delta = self._deltas.pop(0)
        if delta and on_content_delta:
            await on_content_delta(delta)
        return response


@pytest.mark.asyncio
async def test_provider_vision_rejection_invokes_recovery_and_persists_text() -> None:
    provider = _ScriptedProvider(
        [
            LLMResponse(
                content="image input is unsupported",
                finish_reason="error",
                error_status_code=415,
            ),
            LLMResponse(content="ok"),
        ]
    )
    messages = [{"role": "user", "content": [_image_block()]}]
    recoveries = 0

    async def recover(recovery_messages, _response) -> bool:
        nonlocal recoveries
        recoveries += 1
        recovery_messages[0]["content"][0] = {
            "type": "text",
            "text": "transcribed",
        }
        return True

    response = await provider.chat_with_retry(
        messages=messages,
        image_recovery_callback=recover,
    )

    assert response.content == "ok"
    assert recoveries == 1
    assert messages[0]["content"][0] == {"type": "text", "text": "transcribed"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        LLMResponse(
            content="invalid API key",
            finish_reason="error",
            error_status_code=401,
        ),
        LLMResponse(
            content="invalid API key",
            finish_reason="error",
            error_status_code=400,
        ),
        LLMResponse(
            content="rate limit",
            finish_reason="error",
            error_status_code=429,
            error_should_retry=False,
        ),
        LLMResponse(
            content="request timed out",
            finish_reason="error",
            error_kind="timeout",
            error_should_retry=False,
        ),
    ],
)
async def test_provider_does_not_transcribe_non_vision_errors(response) -> None:
    provider = _ScriptedProvider([response])
    callback = AsyncMock(return_value=True)

    result = await provider.chat_with_retry(
        messages=[{"role": "user", "content": [_image_block()]}],
        image_recovery_callback=callback,
    )

    assert result is response
    callback.assert_not_awaited()
    assert len(provider.seen_messages) == 1


@pytest.mark.asyncio
async def test_streaming_vision_recovery_only_runs_before_output() -> None:
    rejected = LLMResponse(
        content="unsupported image",
        finish_reason="error",
        error_status_code=422,
    )
    provider = _StreamingProvider(
        [(rejected, None), (LLMResponse(content="ok"), "ok")]
    )
    messages = [{"role": "user", "content": [_image_block()]}]
    callback = AsyncMock(return_value=True)

    async def replace(recovery_messages, _response):
        recovery_messages[0]["content"][0] = {"type": "text", "text": "transcribed"}
        return await callback(recovery_messages, _response)

    deltas: list[str] = []

    async def on_delta(delta: str) -> None:
        deltas.append(delta)

    result = await provider.chat_stream_with_retry(
        messages=messages,
        on_content_delta=on_delta,
        image_recovery_callback=replace,
    )

    assert result.content == "ok"
    callback.assert_awaited_once()
    assert deltas == ["ok"]

    provider = _StreamingProvider([(rejected, "partial")])
    callback.reset_mock()
    result = await provider.chat_stream_with_retry(
        messages=[{"role": "user", "content": [_image_block()]}],
        on_content_delta=on_delta,
        image_recovery_callback=callback,
    )

    assert result is rejected
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_pretranscribes_for_non_vision_runtime() -> None:
    provider = MagicMock()
    seen: list[list[dict]] = []

    async def chat_with_retry(*, messages, **_kwargs):
        seen.append(messages)
        return LLMResponse(content="done")

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    transcriber = MagicMock()

    async def replace(messages, _cache):
        for message in messages:
            if isinstance(message.get("content"), list):
                message["content"] = [
                    {"type": "text", "text": "transcribed"}
                    if block.get("type") == "image_url"
                    else block
                    for block in message["content"]
                ]
        return True

    transcriber.replace_images = AsyncMock(side_effect=replace)
    runtime = LLMRuntime(
        provider=provider,
        model="text-only",
        generation=GenerationSettings(),
        context_window_tokens=10_000,
        supports_vision=False,
    )
    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": [_image_block()]}],
            tools=tools,
            runtime=runtime,
            max_iterations=1,
            max_tool_result_chars=1000,
            image_transcriber=transcriber,
        )
    )

    assert result.final_content == "done"
    assert transcriber.replace_images.await_count == 1
    assert not any(
        block.get("type") == "image_url"
        for message in seen[0]
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
    )


@pytest.mark.asyncio
async def test_runner_sends_images_normally_for_vision_runtime() -> None:
    provider = MagicMock()
    seen: list[list[dict]] = []

    async def chat_with_retry(*, messages, **_kwargs):
        seen.append(messages)
        return LLMResponse(content="seen")

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    transcriber = MagicMock()
    transcriber.replace_images = AsyncMock(return_value=True)
    runtime = LLMRuntime(
        provider=provider,
        model="vision",
        generation=GenerationSettings(),
        context_window_tokens=10_000,
        supports_vision=True,
    )

    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": [_image_block()]}],
            tools=tools,
            runtime=runtime,
            max_iterations=1,
            max_tool_result_chars=1000,
            image_transcriber=transcriber,
        )
    )

    assert result.final_content == "seen"
    transcriber.replace_images.assert_not_awaited()
    assert seen[0][0]["content"][0]["type"] == "image_url"


@pytest.mark.asyncio
async def test_runner_pretranscribes_image_returned_by_read_file() -> None:
    provider = MagicMock()
    seen: list[list[dict]] = []
    responses = [
        LLMResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="read-1",
                    name="read_file",
                    arguments={"path": "screen.png"},
                )
            ],
            finish_reason="tool_calls",
        ),
        LLMResponse(content="described"),
    ]

    async def chat_with_retry(*, messages, **_kwargs):
        seen.append(messages)
        return responses.pop(0)

    provider.chat_with_retry = chat_with_retry
    tools = MagicMock()
    tools.get_definitions.return_value = []
    tools.execute = AsyncMock(return_value=[_image_block()])
    transcriber = MagicMock()

    async def replace(messages, _cache):
        for message in messages:
            if isinstance(message.get("content"), list):
                for index, block in enumerate(message["content"]):
                    if block.get("type") == "image_url":
                        message["content"][index] = {
                            "type": "text",
                            "text": "read_file transcription",
                        }
        return True

    transcriber.replace_images = AsyncMock(side_effect=replace)
    runtime = LLMRuntime(
        provider=provider,
        model="text-only",
        generation=GenerationSettings(),
        context_window_tokens=10_000,
        supports_vision=False,
    )

    result = await AgentRunner().run(
        AgentRunSpec(
            initial_messages=[{"role": "user", "content": "read it"}],
            tools=tools,
            runtime=runtime,
            max_iterations=2,
            max_tool_result_chars=1000,
            image_transcriber=transcriber,
        )
    )

    assert result.final_content == "described"
    assert transcriber.replace_images.await_count == 1
    assert any(
        block.get("text") == "read_file transcription"
        for message in seen[1]
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
    )
    assert not any(
        block.get("type") == "image_url"
        for message in result.messages
        for block in (
            message.get("content")
            if isinstance(message.get("content"), list)
            else []
        )
    )
