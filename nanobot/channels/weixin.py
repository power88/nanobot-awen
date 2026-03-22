"""Personal WeChat (微信) channel using HTTP long-poll API.

Uses the ilinkai.weixin.qq.com API for personal WeChat messaging.
No WebSocket, no local WeChat client needed — just HTTP requests with a
bot token obtained via QR code login.

Protocol reverse-engineered from ``@tencent-weixin/openclaw-weixin`` v1.0.2.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import secrets
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import url2pathname

import httpx
from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.paths import get_media_dir, get_runtime_subdir
from nanobot.config.schema import Base
from nanobot.utils.helpers import split_message

# ---------------------------------------------------------------------------
# Protocol constants (from openclaw-weixin types.ts)
# ---------------------------------------------------------------------------

# MessageItemType
ITEM_TEXT = 1
ITEM_IMAGE = 2
ITEM_VOICE = 3
ITEM_FILE = 4
ITEM_VIDEO = 5

# MessageType  (1 = inbound from user, 2 = outbound from bot)
MESSAGE_TYPE_USER = 1
MESSAGE_TYPE_BOT = 2

# MessageState
MESSAGE_STATE_FINISH = 2

WEIXIN_MAX_MESSAGE_LEN = 4000
BASE_INFO: dict[str, str] = {"channel_version": "1.0.2"}

# Session-expired error code
ERRCODE_SESSION_EXPIRED = -14

# Retry constants (matching the reference plugin's monitor.ts)
MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_DELAY_S = 30
RETRY_DELAY_S = 2

# Default long-poll timeout; overridden by server via longpolling_timeout_ms.
DEFAULT_LONG_POLL_TIMEOUT_S = 35

# Upload pipeline (types.ts UploadMediaType, upload.ts, cdn-upload.ts, send-media.ts, send.ts)
UPLOAD_MEDIA_IMAGE = 1
UPLOAD_MEDIA_VIDEO = 2
UPLOAD_MEDIA_FILE = 3
CDN_UPLOAD_MAX_RETRIES = 3
WEIXIN_API_TIMEOUT_S = 60.0
CDN_UPLOAD_TIMEOUT_S = 180.0

# mime.ts — extension → MIME for outbound routing
_EXTENSION_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def _mime_from_filename(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return _EXTENSION_TO_MIME.get(ext, "application/octet-stream")


def _aes_ecb_padded_size(plaintext_size: int) -> int:
    """AES-128-ECB ciphertext size with PKCS7 (aes-ecb.ts aesEcbPaddedSize)."""
    return int(math.ceil((plaintext_size + 1) / 16) * 16)


def _extension_from_content_type_or_url(content_type: str | None, url: str) -> str:
    """Infer file extension from Content-Type or URL (upload.ts getExtensionFromContentTypeOrUrl)."""
    if content_type:
        ct = content_type.split(";")[0].strip().lower()
        ext = mimetypes.guess_extension(ct, strict=False)
        if ext == ".jpe":
            ext = ".jpg"
        if ext:
            return ext
    path = urlparse(url).path.lower()
    suf = Path(path).suffix
    if suf in _EXTENSION_TO_MIME:
        return suf
    return ".bin"


def _encrypt_aes_ecb(plaintext: bytes, key: bytes) -> bytes:
    """AES-128-ECB encrypt with PKCS7 padding (aes-ecb.ts encryptAesEcb)."""
    try:
        from Crypto.Cipher import AES
        from Crypto.Util.Padding import pad

        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.encrypt(pad(plaintext, AES.block_size))
    except ImportError:
        pass
    try:
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        padder = padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.ECB())
        encryptor = cipher.encryptor()
        return encryptor.update(padded) + encryptor.finalize()
    except ImportError as e:
        raise RuntimeError(
            "WeChat media upload requires 'pycryptodome' or 'cryptography' for AES encryption"
        ) from e


def _cdn_media_aes_key_json_field(aes_key_raw: bytes) -> str:
    """Match send.ts ``Buffer.from(uploaded.aeskey).toString('base64')`` (aeskey is hex string)."""
    return base64.b64encode(aes_key_raw.hex().encode("ascii")).decode()


def _encode_uri_component(s: str) -> str:
    """Match JS ``encodeURIComponent`` (cdn-url.ts).

    ``urllib.parse.quote`` defaults to ``safe='/'``, which leaves ``/`` unencoded.
    ``upload_param`` from ``getUploadUrl`` is often Base64 and contains ``/``; if
    unencoded, the query string breaks and CDN upload/decrypt yields corrupt bytes
    (gray/black images, wrong size metadata).
    """
    return quote(s, safe="")


def _build_cdn_upload_url(*, cdn_base_url: str, upload_param: str, filekey: str) -> str:
    """CDN POST URL (cdn-url.ts buildCdnUploadUrl)."""
    return (
        f"{cdn_base_url}/upload"
        f"?encrypted_query_param={_encode_uri_component(upload_param)}"
        f"&filekey={_encode_uri_component(filekey)}"
    )


def _strip_optional_quotes(s: str) -> str:
    """Strip one pair of surrounding quotes (models often return '\"C:\\\\path\"')."""
    t = s.strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ('"', "'"):
        return t[1:-1].strip()
    return t


def _path_from_file_uri(uri: str) -> Path:
    """Convert ``file:`` URI to a local :class:`Path`.

    ``urllib.parse.urlparse`` splits ``file://C:/Users/...`` (two slashes, common on
    Windows) into ``netloc='C:'`` and ``path='/Users/...'``. Using only
    ``url2pathname(path)`` drops the drive and yields ``\\Users\\...`` — wrong.
    ``file:///C:/...`` (three slashes) is handled by ``url2pathname`` on the path.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return Path(uri)
    path = unquote(parsed.path)
    netloc = unquote(parsed.netloc or "")

    if os.name == "nt":
        # file:///C:/Users/...  → path like /C:/Users/...
        if len(path) >= 4 and path[0] == "/" and path[2] == ":" and path[1].isalpha():
            return Path(url2pathname(path))
        # file://C:/Users/...  → netloc=C: path=/Users/...
        if netloc and len(netloc) >= 2 and netloc[1] == ":":
            combined = netloc + (path if path.startswith("/") else "/" + path)
            return Path(combined)
        # file://localhost/C:/Users/...
        if netloc.lower() == "localhost" and len(path) >= 4 and path[0] == "/" and path[2] == ":":
            return Path(url2pathname(path))
        try:
            return Path(url2pathname(path))
        except OSError:
            return Path(path)

    # POSIX: file:///home/foo
    return Path(path)


class WeixinConfig(Base):
    """Personal WeChat channel configuration."""

    enabled: bool = False
    allow_from: list[str] = Field(default_factory=list)
    base_url: str = "https://ilinkai.weixin.qq.com"
    cdn_base_url: str = "https://novac2c.cdn.weixin.qq.com/c2c"
    token: str = ""  # Manually set token, or obtained via QR login
    state_dir: str = ""  # Default: ~/.nanobot/weixin/
    poll_timeout: int = DEFAULT_LONG_POLL_TIMEOUT_S  # seconds for long-poll


class WeixinChannel(BaseChannel):
    """
    Personal WeChat channel using HTTP long-poll.

    Connects to ilinkai.weixin.qq.com API to receive and send personal
    WeChat messages. Authentication is via QR code login which produces
    a bot token.
    """

    name = "weixin"
    display_name = "WeChat"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WeixinConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = WeixinConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WeixinConfig = config

        # State
        self._client: httpx.AsyncClient | None = None
        self._get_updates_buf: str = ""
        self._context_tokens: dict[str, str] = {}  # from_user_id -> context_token
        self._processed_ids: OrderedDict[str, None] = OrderedDict()
        self._state_dir: Path | None = None
        self._token: str = ""
        self._poll_task: asyncio.Task | None = None
        self._next_poll_timeout_s: int = DEFAULT_LONG_POLL_TIMEOUT_S

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _get_state_dir(self) -> Path:
        if self._state_dir:
            return self._state_dir
        if self.config.state_dir:
            d = Path(self.config.state_dir).expanduser()
        else:
            d = get_runtime_subdir("weixin")
        d.mkdir(parents=True, exist_ok=True)
        self._state_dir = d
        return d

    def _load_state(self) -> bool:
        """Load saved account state. Returns True if a valid token was found."""
        state_file = self._get_state_dir() / "account.json"
        if not state_file.exists():
            return False
        try:
            data = json.loads(state_file.read_text())
            self._token = data.get("token", "")
            self._get_updates_buf = data.get("get_updates_buf", "")
            base_url = data.get("base_url", "")
            if base_url:
                self.config.base_url = base_url
            return bool(self._token)
        except Exception as e:
            logger.warning("Failed to load WeChat state: {}", e)
            return False

    def _save_state(self) -> None:
        state_file = self._get_state_dir() / "account.json"
        try:
            data = {
                "token": self._token,
                "get_updates_buf": self._get_updates_buf,
                "base_url": self.config.base_url,
            }
            state_file.write_text(json.dumps(data, ensure_ascii=False))
        except Exception as e:
            logger.warning("Failed to save WeChat state: {}", e)

    # ------------------------------------------------------------------
    # HTTP helpers  (matches api.ts buildHeaders / apiFetch)
    # ------------------------------------------------------------------

    @staticmethod
    def _random_wechat_uin() -> str:
        """X-WECHAT-UIN: random uint32 → decimal string → base64.

        Matches the reference plugin's ``randomWechatUin()`` in api.ts.
        Generated fresh for **every** request (same as reference).
        """
        uint32 = int.from_bytes(os.urandom(4), "big")
        return base64.b64encode(str(uint32).encode()).decode()

    def _make_headers(self, *, auth: bool = True) -> dict[str, str]:
        """Build per-request headers (new UIN each call, matching reference)."""
        headers: dict[str, str] = {
            "X-WECHAT-UIN": self._random_wechat_uin(),
            "Content-Type": "application/json",
            "AuthorizationType": "ilink_bot_token",
        }
        if auth and self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _api_get(
        self,
        endpoint: str,
        params: dict | None = None,
        *,
        auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        assert self._client is not None
        url = f"{self.config.base_url}/{endpoint}"
        hdrs = self._make_headers(auth=auth)
        if extra_headers:
            hdrs.update(extra_headers)
        resp = await self._client.get(url, params=params, headers=hdrs)
        resp.raise_for_status()
        return resp.json()

    async def _api_post(
        self,
        endpoint: str,
        body: dict | None = None,
        *,
        auth: bool = True,
        timeout: float | None = None,
    ) -> dict:
        assert self._client is not None
        url = f"{self.config.base_url}/{endpoint}"
        payload = body or {}
        if "base_info" not in payload:
            payload["base_info"] = BASE_INFO
        post_kw: dict[str, Any] = {}
        if timeout is not None:
            post_kw["timeout"] = timeout
        resp = await self._client.post(
            url,
            json=payload,
            headers=self._make_headers(auth=auth),
            **post_kw,
        )
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # QR Code Login  (matches login-qr.ts)
    # ------------------------------------------------------------------

    async def _qr_login(self) -> bool:
        """Perform QR code login flow. Returns True on success."""
        try:
            logger.info("Starting WeChat QR code login...")

            data = await self._api_get(
                "ilink/bot/get_bot_qrcode",
                params={"bot_type": "3"},
                auth=False,
            )
            qrcode_img_content = data.get("qrcode_img_content", "")
            qrcode_id = data.get("qrcode", "")

            if not qrcode_id:
                logger.error("Failed to get QR code from WeChat API: {}", data)
                return False

            scan_url = qrcode_img_content or qrcode_id
            self._print_qr_code(scan_url)

            logger.info("Waiting for QR code scan...")
            while self._running:
                try:
                    # Reference plugin sends iLink-App-ClientVersion header for
                    # QR status polling (login-qr.ts:81).
                    status_data = await self._api_get(
                        "ilink/bot/get_qrcode_status",
                        params={"qrcode": qrcode_id},
                        auth=False,
                        extra_headers={"iLink-App-ClientVersion": "1"},
                    )
                except httpx.TimeoutException:
                    continue

                status = status_data.get("status", "")
                if status == "confirmed":
                    token = status_data.get("bot_token", "")
                    bot_id = status_data.get("ilink_bot_id", "")
                    base_url = status_data.get("baseurl", "")
                    user_id = status_data.get("ilink_user_id", "")
                    if token:
                        self._token = token
                        if base_url:
                            self.config.base_url = base_url
                        self._save_state()
                        logger.info(
                            "WeChat login successful! bot_id={} user_id={}",
                            bot_id,
                            user_id,
                        )
                        return True
                    else:
                        logger.error("Login confirmed but no bot_token in response")
                        return False
                elif status == "scaned":
                    logger.info("QR code scanned, waiting for confirmation...")
                elif status == "expired":
                    logger.warning("QR code expired")
                    return False
                # status == "wait" — keep polling

                await asyncio.sleep(1)

        except Exception as e:
            logger.error("WeChat QR login failed: {}", e)

        return False

    @staticmethod
    def _print_qr_code(url: str) -> None:
        try:
            import qrcode as qr_lib

            qr = qr_lib.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        except ImportError:
            logger.info("QR code URL (install 'qrcode' for terminal display): {}", url)
            print(f"\nLogin URL: {url}\n")

    # ------------------------------------------------------------------
    # Channel lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._next_poll_timeout_s = self.config.poll_timeout
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._next_poll_timeout_s + 10, connect=30),
            follow_redirects=True,
        )

        if self.config.token:
            self._token = self.config.token
        elif not self._load_state():
            if not await self._qr_login():
                logger.error("WeChat login failed. Run 'nanobot weixin login' to authenticate.")
                self._running = False
                return

        logger.info("WeChat channel starting with long-poll...")

        consecutive_failures = 0
        while self._running:
            try:
                await self._poll_once()
                consecutive_failures = 0
            except httpx.TimeoutException:
                # Normal for long-poll, just retry
                continue
            except Exception as e:
                if not self._running:
                    break
                consecutive_failures += 1
                logger.error(
                    "WeChat poll error ({}/{}): {}",
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES,
                    e,
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                    await asyncio.sleep(BACKOFF_DELAY_S)
                else:
                    await asyncio.sleep(RETRY_DELAY_S)

    async def stop(self) -> None:
        self._running = False
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        if self._client:
            await self._client.aclose()
            self._client = None
        self._save_state()
        logger.info("WeChat channel stopped")

    # ------------------------------------------------------------------
    # Polling  (matches monitor.ts monitorWeixinProvider)
    # ------------------------------------------------------------------

    async def _poll_once(self) -> None:
        body: dict[str, Any] = {
            "get_updates_buf": self._get_updates_buf,
            "base_info": BASE_INFO,
        }

        # Adjust httpx timeout to match the current poll timeout
        assert self._client is not None
        self._client.timeout = httpx.Timeout(self._next_poll_timeout_s + 10, connect=30)

        data = await self._api_post("ilink/bot/getupdates", body)

        # Check for API-level errors (monitor.ts checks both ret and errcode)
        ret = data.get("ret", 0)
        errcode = data.get("errcode", 0)
        is_error = (ret is not None and ret != 0) or (errcode is not None and errcode != 0)

        if is_error:
            if errcode == ERRCODE_SESSION_EXPIRED or ret == ERRCODE_SESSION_EXPIRED:
                logger.warning(
                    "WeChat session expired (errcode {}). Pausing 60 min.",
                    errcode,
                )
                await asyncio.sleep(3600)
                return
            raise RuntimeError(
                f"getUpdates failed: ret={ret} errcode={errcode} errmsg={data.get('errmsg', '')}"
            )

        # Honour server-suggested poll timeout (monitor.ts:102-105)
        server_timeout_ms = data.get("longpolling_timeout_ms")
        if server_timeout_ms and server_timeout_ms > 0:
            self._next_poll_timeout_s = max(server_timeout_ms // 1000, 5)

        # Update cursor
        new_buf = data.get("get_updates_buf", "")
        if new_buf:
            self._get_updates_buf = new_buf
            self._save_state()

        # Process messages (WeixinMessage[] from types.ts)
        msgs: list[dict] = data.get("msgs", []) or []
        for msg in msgs:
            try:
                await self._process_message(msg)
            except Exception as e:
                logger.error("Error processing WeChat message: {}", e)

    # ------------------------------------------------------------------
    # Inbound message processing  (matches inbound.ts + process-message.ts)
    # ------------------------------------------------------------------

    async def _process_message(self, msg: dict) -> None:
        """Process a single WeixinMessage from getUpdates."""
        # Skip bot's own messages (message_type 2 = BOT)
        if msg.get("message_type") == MESSAGE_TYPE_BOT:
            return

        # Deduplication by message_id
        msg_id = str(msg.get("message_id", "") or msg.get("seq", ""))
        if not msg_id:
            msg_id = f"{msg.get('from_user_id', '')}_{msg.get('create_time_ms', '')}"
        if msg_id in self._processed_ids:
            return
        self._processed_ids[msg_id] = None
        while len(self._processed_ids) > 1000:
            self._processed_ids.popitem(last=False)

        from_user_id = msg.get("from_user_id", "") or ""
        if not from_user_id:
            return

        # Cache context_token (required for all replies — inbound.ts:23-27)
        ctx_token = msg.get("context_token", "")
        if ctx_token:
            self._context_tokens[from_user_id] = ctx_token

        # Parse item_list (WeixinMessage.item_list — types.ts:161)
        item_list: list[dict] = msg.get("item_list") or []
        content_parts: list[str] = []
        media_paths: list[str] = []

        for item in item_list:
            item_type = item.get("type", 0)

            if item_type == ITEM_TEXT:
                text = (item.get("text_item") or {}).get("text", "")
                if text:
                    # Handle quoted/ref messages (inbound.ts:86-98)
                    ref = item.get("ref_msg")
                    if ref:
                        ref_item = ref.get("message_item")
                        # If quoted message is media, just pass the text
                        if ref_item and ref_item.get("type", 0) in (
                            ITEM_IMAGE,
                            ITEM_VOICE,
                            ITEM_FILE,
                            ITEM_VIDEO,
                        ):
                            content_parts.append(text)
                        else:
                            parts: list[str] = []
                            if ref.get("title"):
                                parts.append(ref["title"])
                            if ref_item:
                                ref_text = (ref_item.get("text_item") or {}).get("text", "")
                                if ref_text:
                                    parts.append(ref_text)
                            if parts:
                                content_parts.append(f"[引用: {' | '.join(parts)}]\n{text}")
                            else:
                                content_parts.append(text)
                    else:
                        content_parts.append(text)

            elif item_type == ITEM_IMAGE:
                image_item = item.get("image_item") or {}
                file_path = await self._download_media_item(image_item, "image")
                if file_path:
                    content_parts.append(f"[image]\n[Image: source: {file_path}]")
                    media_paths.append(file_path)
                else:
                    content_parts.append("[image]")

            elif item_type == ITEM_VOICE:
                voice_item = item.get("voice_item") or {}
                # Voice-to-text provided by WeChat (inbound.ts:101-103)
                voice_text = voice_item.get("text", "")
                if voice_text:
                    content_parts.append(f"[voice] {voice_text}")
                else:
                    file_path = await self._download_media_item(voice_item, "voice")
                    if file_path:
                        transcription = await self.transcribe_audio(file_path)
                        if transcription:
                            content_parts.append(f"[voice] {transcription}")
                        else:
                            content_parts.append(f"[voice]\n[Audio: source: {file_path}]")
                        media_paths.append(file_path)
                    else:
                        content_parts.append("[voice]")

            elif item_type == ITEM_FILE:
                file_item = item.get("file_item") or {}
                file_name = file_item.get("file_name", "unknown")
                file_path = await self._download_media_item(
                    file_item,
                    "file",
                    file_name,
                )
                if file_path:
                    content_parts.append(f"[file: {file_name}]\n[File: source: {file_path}]")
                    media_paths.append(file_path)
                else:
                    content_parts.append(f"[file: {file_name}]")

            elif item_type == ITEM_VIDEO:
                video_item = item.get("video_item") or {}
                file_path = await self._download_media_item(video_item, "video")
                if file_path:
                    content_parts.append(f"[video]\n[Video: source: {file_path}]")
                    media_paths.append(file_path)
                else:
                    content_parts.append("[video]")

        content = "\n".join(content_parts)
        if not content:
            return

        logger.info(
            "WeChat inbound: from={} items={} bodyLen={}",
            from_user_id,
            ",".join(str(i.get("type", 0)) for i in item_list),
            len(content),
        )

        await self._handle_message(
            sender_id=from_user_id,
            chat_id=from_user_id,
            content=content,
            media=media_paths or None,
            metadata={"message_id": msg_id},
        )

    # ------------------------------------------------------------------
    # Media download  (matches media-download.ts + pic-decrypt.ts)
    # ------------------------------------------------------------------

    async def _download_media_item(
        self,
        typed_item: dict,
        media_type: str,
        filename: str | None = None,
    ) -> str | None:
        """Download + AES-decrypt a media item. Returns local path or None."""
        try:
            media = typed_item.get("media") or {}
            encrypt_query_param = media.get("encrypt_query_param", "")

            if not encrypt_query_param:
                return None

            # Resolve AES key (media-download.ts:43-45, pic-decrypt.ts:40-52)
            # image_item.aeskey is a raw hex string (16 bytes as 32 hex chars).
            # media.aes_key is always base64-encoded.
            # For images, prefer image_item.aeskey; for others use media.aes_key.
            raw_aeskey_hex = typed_item.get("aeskey", "")
            media_aes_key_b64 = media.get("aes_key", "")

            aes_key_b64: str = ""
            if raw_aeskey_hex:
                # Convert hex → raw bytes → base64 (matches media-download.ts:43-44)
                aes_key_b64 = base64.b64encode(bytes.fromhex(raw_aeskey_hex)).decode()
            elif media_aes_key_b64:
                aes_key_b64 = media_aes_key_b64

            # Build CDN download URL (cdn-url.ts buildCdnDownloadUrl — encodeURIComponent)
            cdn_url = (
                f"{self.config.cdn_base_url}/download"
                f"?encrypted_query_param={_encode_uri_component(encrypt_query_param)}"
            )

            assert self._client is not None
            resp = await self._client.get(cdn_url)
            resp.raise_for_status()
            data = resp.content

            if aes_key_b64 and data:
                data = _decrypt_aes_ecb(data, aes_key_b64)
            elif not aes_key_b64:
                logger.debug("No AES key for {} item, using raw bytes", media_type)

            if not data:
                return None

            media_dir = get_media_dir("weixin")
            ext = _ext_for_type(media_type)
            if not filename:
                ts = int(time.time())
                h = abs(hash(encrypt_query_param)) % 100000
                filename = f"{media_type}_{ts}_{h}{ext}"
            safe_name = os.path.basename(filename)
            file_path = media_dir / safe_name
            file_path.write_bytes(data)
            logger.debug("Downloaded WeChat {} to {}", media_type, file_path)
            return str(file_path)

        except Exception as e:
            logger.error("Error downloading WeChat media: {}", e)
            return None

    # ------------------------------------------------------------------
    # Outbound  (matches send.ts, send-media.ts, upload.ts, cdn-upload.ts)
    # ------------------------------------------------------------------

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client or not self._token:
            logger.warning("WeChat client not initialized or not authenticated")
            return

        content = msg.content.strip()
        media_refs = list(msg.media or [])

        if not content and not media_refs:
            return

        ctx_token = self._context_tokens.get(msg.chat_id, "")
        if not ctx_token:
            logger.warning(
                "WeChat: no context_token for chat_id={}, cannot send",
                msg.chat_id,
            )
            return

        try:
            # Official send-media.ts / send.ts: sendWeixinMediaFile passes `text` into
            # sendImageMessageWeixin → sendMediaItems, which sends TEXT (caption) then
            # media as separate sendmessage calls — repeated for each file when multiple.
            if media_refs:
                for ref in media_refs:
                    try:
                        local_path = await self._ensure_local_media_path(ref)
                        await self._send_weixin_media_file(
                            local_path,
                            to_user_id=msg.chat_id,
                            context_token=ctx_token,
                            caption=content,
                        )
                    except Exception as e:
                        logger.error("WeChat send media failed ref={}: {}", ref, e)
            elif content:
                for chunk in split_message(content, WEIXIN_MAX_MESSAGE_LEN):
                    await self._send_text(msg.chat_id, chunk, ctx_token)
        except Exception as e:
            logger.error("Error sending WeChat message: {}", e)

    def _resolve_local_media_path(self, media: str) -> Path:
        """Resolve file:// or filesystem path (channel.ts resolveLocalPath; Windows-safe)."""
        s = _strip_optional_quotes(media)
        if s.startswith("file://"):
            return _path_from_file_uri(s)
        p = Path(s).expanduser()
        if not p.is_absolute():
            p = Path.cwd() / p
        return p.resolve()

    async def _download_remote_media(self, url: str) -> Path:
        """Download remote image/video/file to outbound-temp (upload.ts downloadRemoteImageToTemp)."""
        assert self._client is not None
        dest_dir = get_runtime_subdir("weixin") / "outbound-temp"
        dest_dir.mkdir(parents=True, exist_ok=True)
        resp = await self._client.get(
            url,
            follow_redirects=True,
            timeout=WEIXIN_API_TIMEOUT_S,
        )
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        ext = _extension_from_content_type_or_url(ct, url)
        name = f"weixin-remote-{uuid.uuid4().hex[:12]}{ext}"
        path = dest_dir / name
        path.write_bytes(resp.content)
        logger.debug("WeChat: downloaded remote media to {}", path)
        return path

    async def _ensure_local_media_path(self, media: str) -> Path:
        s = media.strip()
        if s.startswith("http://") or s.startswith("https://"):
            return await self._download_remote_media(s)
        path = self._resolve_local_media_path(s)
        if not path.is_file():
            raise FileNotFoundError(f"WeChat media file not found: {path}")
        return path

    async def _upload_buffer_to_cdn(
        self,
        *,
        buf: bytes,
        upload_param: str,
        filekey: str,
        aes_key: bytes,
        label: str,
    ) -> str:
        """POST ciphertext to CDN; return x-encrypted-param (cdn-upload.ts uploadBufferToCdn)."""
        assert self._client is not None
        ciphertext = _encrypt_aes_ecb(buf, aes_key)
        cdn_url = _build_cdn_upload_url(
            cdn_base_url=self.config.cdn_base_url,
            upload_param=upload_param,
            filekey=filekey,
        )
        logger.debug(
            "{}: CDN POST ciphertext_size={}",
            label,
            len(ciphertext),
        )

        last_err: Exception | None = None
        for attempt in range(1, CDN_UPLOAD_MAX_RETRIES + 1):
            try:
                resp = await self._client.post(
                    cdn_url,
                    content=ciphertext,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=CDN_UPLOAD_TIMEOUT_S,
                )
                err_hdr = resp.headers.get("x-error-message") or resp.headers.get("X-Error-Message")
                if 400 <= resp.status_code < 500:
                    msg = err_hdr or resp.text
                    raise RuntimeError(f"CDN upload client error {resp.status_code}: {msg}")
                if resp.status_code != 200:
                    msg = err_hdr or f"status {resp.status_code}"
                    raise RuntimeError(f"CDN upload server error: {msg}")
                download_param = resp.headers.get("x-encrypted-param") or resp.headers.get(
                    "X-Encrypted-Param"
                )
                if not download_param:
                    raise RuntimeError("CDN upload response missing x-encrypted-param header")
                logger.debug("{}: CDN upload success attempt={}", label, attempt)
                return download_param
            except Exception as e:
                last_err = e
                if isinstance(e, RuntimeError) and "client error" in str(e).lower():
                    raise
                if attempt < CDN_UPLOAD_MAX_RETRIES:
                    logger.warning(
                        "{}: attempt {} failed, retrying: {}",
                        label,
                        attempt,
                        e,
                    )
                else:
                    logger.error("{}: all attempts failed: {}", label, e)

        assert last_err is not None
        raise last_err

    async def _upload_media_to_cdn(
        self,
        *,
        file_path: Path,
        to_user_id: str,
        media_type: int,
        label: str,
    ) -> dict[str, Any]:
        """Read file → MD5 → AES key → getUploadUrl → CDN upload (upload.ts uploadMediaToCdn)."""
        plaintext = file_path.read_bytes()
        rawsize = len(plaintext)
        rawfilemd5 = hashlib.md5(plaintext).hexdigest()
        filesize_cipher = _aes_ecb_padded_size(rawsize)
        filekey = secrets.token_hex(16)
        aes_key = os.urandom(16)
        aeskey_hex = aes_key.hex()

        logger.debug(
            "{}: file={} rawsize={} filesize_cipher={} md5={} filekey={}",
            label,
            file_path,
            rawsize,
            filesize_cipher,
            rawfilemd5,
            filekey,
        )

        up_body: dict[str, Any] = {
            "filekey": filekey,
            "media_type": media_type,
            "to_user_id": to_user_id,
            "rawsize": rawsize,
            "rawfilemd5": rawfilemd5,
            "filesize": filesize_cipher,
            "no_need_thumb": True,
            "aeskey": aeskey_hex,
            "base_info": BASE_INFO,
        }
        up_resp = await self._api_post(
            "ilink/bot/getuploadurl",
            up_body,
            timeout=WEIXIN_API_TIMEOUT_S,
        )
        upload_param = up_resp.get("upload_param", "")
        if not upload_param:
            logger.error(
                "{}: getUploadUrl returned no upload_param: {}",
                label,
                up_resp,
            )
            raise RuntimeError(f"{label}: getUploadUrl returned no upload_param")

        download_encrypted = await self._upload_buffer_to_cdn(
            buf=plaintext,
            upload_param=upload_param,
            filekey=filekey,
            aes_key=aes_key,
            label=f"{label}[filekey={filekey}]",
        )

        return {
            "filekey": filekey,
            "download_encrypted_query_param": download_encrypted,
            "aes_key_raw": aes_key,
            "file_size": rawsize,
            "file_size_ciphertext": filesize_cipher,
        }

    async def _send_weixin_media_file(
        self,
        file_path: Path,
        *,
        to_user_id: str,
        context_token: str,
        caption: str = "",
    ) -> None:
        """Route by MIME (send-media.ts sendWeixinMediaFile).

        Matches official flow: optional caption is sent first (split if over limit),
        then upload + send one media — same as sendMediaItems(text, mediaItem).
        """
        cap = caption.strip()
        if cap:
            for chunk in split_message(cap, WEIXIN_MAX_MESSAGE_LEN):
                await self._send_text(to_user_id, chunk, context_token)

        mime = _mime_from_filename(str(file_path))
        if mime.startswith("video/"):
            uploaded = await self._upload_media_to_cdn(
                file_path=file_path,
                to_user_id=to_user_id,
                media_type=UPLOAD_MEDIA_VIDEO,
                label="uploadVideoToWeixin",
            )
            await self._send_video_message_weixin(to_user_id, uploaded, context_token)
            return
        if mime.startswith("image/"):
            uploaded = await self._upload_media_to_cdn(
                file_path=file_path,
                to_user_id=to_user_id,
                media_type=UPLOAD_MEDIA_IMAGE,
                label="uploadFileToWeixin",
            )
            await self._send_image_message_weixin(to_user_id, uploaded, context_token)
            return

        file_name = os.path.basename(str(file_path))
        uploaded = await self._upload_media_to_cdn(
            file_path=file_path,
            to_user_id=to_user_id,
            media_type=UPLOAD_MEDIA_FILE,
            label="uploadFileAttachmentToWeixin",
        )
        await self._send_file_message_weixin(
            to_user_id,
            file_name,
            uploaded,
            context_token,
        )

    async def _send_text(
        self,
        to_user_id: str,
        text: str,
        context_token: str,
    ) -> None:
        """Send a text message matching the exact protocol from send.ts."""
        if not text:
            return
        await self._send_weixin_message(
            to_user_id,
            [{"type": ITEM_TEXT, "text_item": {"text": text}}],
            context_token,
        )

    async def _send_weixin_message(
        self,
        to_user_id: str,
        item_list: list[dict],
        context_token: str,
    ) -> None:
        """Send one message with the given item_list (send.ts sendMessage)."""
        client_id = f"nanobot-{uuid.uuid4().hex[:12]}"
        weixin_msg: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": to_user_id,
            "client_id": client_id,
            "message_type": MESSAGE_TYPE_BOT,
            "message_state": MESSAGE_STATE_FINISH,
            "item_list": item_list,
        }
        if context_token:
            weixin_msg["context_token"] = context_token
        body: dict[str, Any] = {"msg": weixin_msg, "base_info": BASE_INFO}
        data = await self._api_post("ilink/bot/sendmessage", body, timeout=WEIXIN_API_TIMEOUT_S)
        errcode = data.get("errcode", 0)
        if errcode and errcode != 0:
            logger.warning(
                "WeChat send error (code {}): {}",
                errcode,
                data.get("errmsg", ""),
            )

    async def _send_image_message_weixin(
        self,
        to_user_id: str,
        uploaded: dict[str, Any],
        context_token: str,
    ) -> None:
        """send.ts sendImageMessageWeixin (caption already sent by _send_weixin_media_file)."""
        aes_b64 = _cdn_media_aes_key_json_field(uploaded["aes_key_raw"])
        image_item: dict[str, Any] = {
            "type": ITEM_IMAGE,
            "image_item": {
                "media": {
                    "encrypt_query_param": uploaded["download_encrypted_query_param"],
                    "aes_key": aes_b64,
                    "encrypt_type": 1,
                },
                "mid_size": uploaded["file_size_ciphertext"],
            },
        }
        await self._send_weixin_message(to_user_id, [image_item], context_token)

    async def _send_video_message_weixin(
        self,
        to_user_id: str,
        uploaded: dict[str, Any],
        context_token: str,
    ) -> None:
        """send.ts sendVideoMessageWeixin."""
        aes_b64 = _cdn_media_aes_key_json_field(uploaded["aes_key_raw"])
        video_item: dict[str, Any] = {
            "type": ITEM_VIDEO,
            "video_item": {
                "media": {
                    "encrypt_query_param": uploaded["download_encrypted_query_param"],
                    "aes_key": aes_b64,
                    "encrypt_type": 1,
                },
                "video_size": uploaded["file_size_ciphertext"],
            },
        }
        await self._send_weixin_message(to_user_id, [video_item], context_token)

    async def _send_file_message_weixin(
        self,
        to_user_id: str,
        file_name: str,
        uploaded: dict[str, Any],
        context_token: str,
    ) -> None:
        """send.ts sendFileMessageWeixin."""
        aes_b64 = _cdn_media_aes_key_json_field(uploaded["aes_key_raw"])
        file_item: dict[str, Any] = {
            "type": ITEM_FILE,
            "file_item": {
                "media": {
                    "encrypt_query_param": uploaded["download_encrypted_query_param"],
                    "aes_key": aes_b64,
                    "encrypt_type": 1,
                },
                "file_name": file_name,
                "len": str(uploaded["file_size"]),
            },
        }
        await self._send_weixin_message(to_user_id, [file_item], context_token)


# ---------------------------------------------------------------------------
# AES-128-ECB decryption  (matches pic-decrypt.ts parseAesKey + aes-ecb.ts)
# ---------------------------------------------------------------------------


def _parse_aes_key(aes_key_b64: str) -> bytes:
    """Parse a base64-encoded AES key, handling both encodings seen in the wild.

    From ``pic-decrypt.ts parseAesKey``:

    * ``base64(raw 16 bytes)``            → images (media.aes_key)
    * ``base64(hex string of 16 bytes)``  → file / voice / video

    In the second case base64-decoding yields 32 ASCII hex chars which must
    then be parsed as hex to recover the actual 16-byte key.
    """
    decoded = base64.b64decode(aes_key_b64)
    if len(decoded) == 16:
        return decoded
    if len(decoded) == 32 and re.fullmatch(rb"[0-9a-fA-F]{32}", decoded):
        # hex-encoded key: base64 → hex string → raw bytes
        return bytes.fromhex(decoded.decode("ascii"))
    raise ValueError(
        f"aes_key must decode to 16 raw bytes or 32-char hex string, got {len(decoded)} bytes"
    )


def _decrypt_aes_ecb(data: bytes, aes_key_b64: str) -> bytes:
    """Decrypt AES-128-ECB media data.

    ``aes_key_b64`` is always base64-encoded (caller converts hex keys first).
    """
    try:
        key = _parse_aes_key(aes_key_b64)
    except Exception as e:
        logger.warning("Failed to parse AES key, returning raw data: {}", e)
        return data

    try:
        from Crypto.Cipher import AES

        cipher = AES.new(key, AES.MODE_ECB)
        return cipher.decrypt(data)  # pycryptodome auto-strips PKCS7 with unpad
    except ImportError:
        pass

    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher_obj = Cipher(algorithms.AES(key), modes.ECB())
        decryptor = cipher_obj.decryptor()
        return decryptor.update(data) + decryptor.finalize()
    except ImportError:
        logger.warning("Cannot decrypt media: install 'pycryptodome' or 'cryptography'")
        return data


def _ext_for_type(media_type: str) -> str:
    return {
        "image": ".jpg",
        "voice": ".silk",
        "video": ".mp4",
        "file": "",
    }.get(media_type, "")
