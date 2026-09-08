# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camb.ai realtime speech-to-speech translation as a LiveKit ``RealtimeModel``."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import weakref
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import aiohttp

from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, llm, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given

from ...log import logger
from ...models import DEFAULT_REALTIME_MODE, NUM_CHANNELS, REALTIME_SAMPLE_RATE, RealtimeMode
from ...tts import API_KEY_HEADER

REALTIME_BASE_URL = "wss://realtime.camb.ai"
_REALTIME_PATH = "/v1/realtime"

# Only a fallback: text.done and audio.done arrive together and end a response. This
# releases a response the server never completes. Measured intra-response audio gaps
# reach 2.25s, so the window sits well above that to avoid ending a turn mid-speech.
_AUDIO_IDLE_TIMEOUT = 6.0


@dataclass
class _RealtimeOptions:
    source_language: str
    target_language: str
    mode: RealtimeMode
    voice_id: int | None
    api_key: str
    base_url: str


@dataclass
class _Generation:
    """One translated utterance in flight."""

    message_id: str
    text_ch: utils.aio.Chan[str]
    audio_ch: utils.aio.Chan[rtc.AudioFrame]
    message_ch: utils.aio.Chan[llm.MessageGeneration]
    modalities: asyncio.Future[list[Literal["text", "audio"]]]
    started_at: float
    text_done: bool = False
    audio_done: bool = False
    last_audio_at: float = field(default_factory=time.monotonic)


class RealtimeModel(llm.RealtimeModel):
    def __init__(
        self,
        *,
        source_language: str,
        target_language: str,
        mode: RealtimeMode = DEFAULT_REALTIME_MODE,
        voice_id: int | None = None,
        api_key: str | None = None,
        base_url: str = REALTIME_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        """Translate speech to speech with Camb.ai.

        Args:
            source_language: BCP-47 tag of the speaker's language, e.g. ``"en-US"``.
            target_language: BCP-47 tag to translate into, e.g. ``"es-ES"``.
            mode: ``"fast"`` (default) or ``"slow"``; see ``RealtimeMode``.
            voice_id: Synthesize the translation with one of your cloned voices. When
                omitted the server picks a built-in voice for ``target_language``.
            api_key: Camb.ai API key. Falls back to the ``CAMB_API_KEY`` env var.
            base_url: Realtime endpoint. Override to reach a non-production deployment.
            http_session: Session to use instead of the shared one.
        """
        super().__init__(
            capabilities=llm.RealtimeCapabilities(
                message_truncation=False,
                # The endpoint segments utterances itself, so the session must not run its
                # own barge-in detection: a translator's speaker never stops talking.
                turn_detection=True,
                user_transcription=True,
                auto_tool_reply_generation=False,
                audio_output=True,
                manual_function_calls=False,
            )
        )

        camb_api_key = api_key or os.environ.get("CAMB_API_KEY")
        if not camb_api_key:
            raise ValueError(
                "Camb.ai API key is required, either as `api_key` or by setting the "
                "CAMB_API_KEY environment variable"
            )

        self._opts = _RealtimeOptions(
            source_language=source_language,
            target_language=target_language,
            mode=mode,
            voice_id=voice_id,
            api_key=camb_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._sessions = weakref.WeakSet[RealtimeSession]()

    @property
    def model(self) -> str:
        return f"camb-realtime-{self._opts.mode}"

    @property
    def provider(self) -> str:
        return "Camb.ai"

    def _ensure_http_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def session(self, *, turn_detection_disabled: bool = False) -> RealtimeSession:
        sess = RealtimeSession(self)
        self._sessions.add(sess)
        return sess

    async def aclose(self) -> None:
        for sess in list(self._sessions):
            await sess.aclose()


class RealtimeSession(llm.RealtimeSession[Literal["cambai_server_event_received"]]):
    def __init__(self, realtime_model: RealtimeModel) -> None:
        super().__init__(realtime_model)
        self._realtime_model: RealtimeModel = realtime_model
        self._opts = realtime_model._opts

        self._msg_ch = utils.aio.Chan[dict[str, Any]]()
        self._ready = asyncio.Event()
        self._current: _Generation | None = None
        self._input_resampler: rtc.AudioResampler | None = None
        # The server sends 400ms blobs; the room pipeline expects small, even frames.
        self._bstream = utils.audio.AudioByteStream(
            REALTIME_SAMPLE_RATE, NUM_CHANNELS, samples_per_channel=REALTIME_SAMPLE_RATE // 10
        )
        self._chat_ctx = llm.ChatContext.empty()
        self._pending_reply: asyncio.Future[llm.GenerationCreatedEvent] | None = None
        self._turn_started_at: float | None = None
        self._item_id = 0

        self._main_atask = asyncio.create_task(self._main_task(), name="cambai-realtime")

    async def _main_task(self) -> None:
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("the camb.ai realtime session ended with an error")
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=e,
                    recoverable=False,
                ),
            )
            raise

    async def _run(self) -> None:
        url = f"{self._opts.base_url}{_REALTIME_PATH}?mode={self._opts.mode}"
        session = self._realtime_model._ensure_http_session()

        try:
            ws = await session.ws_connect(url, headers={API_KEY_HEADER: self._opts.api_key})
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                "failed to connect to the Camb.ai realtime endpoint",
                status_code=e.status,
                body=e.message,
            ) from e
        except Exception as e:
            raise APIConnectionError("failed to connect to the Camb.ai realtime endpoint") from e

        session_update = {
            "type": "session.update",
            "session": {
                "mode": self._opts.mode,
                "source_language": self._opts.source_language,
                "target_language": self._opts.target_language,
                "output_modalities": ["text", "audio"],
            },
            "auth": {"api_key": self._opts.api_key},
        }
        if self._opts.voice_id is not None:
            session_update["session"]["voice"] = {  # type: ignore[index]
                "type": "cloned",
                "voice_id": self._opts.voice_id,
            }
        await ws.send_str(json.dumps(session_update))

        tasks = [
            asyncio.create_task(self._send_task(ws), name="cambai-realtime-send"),
            asyncio.create_task(self._recv_task(ws), name="cambai-realtime-recv"),
            asyncio.create_task(self._watchdog_task(), name="cambai-realtime-watchdog"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await utils.aio.cancel_and_wait(*tasks)
            await ws.close()
            self._finish_generation()

    async def _send_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        # Audio sent before the session exists is discarded by the server.
        await self._ready.wait()
        async for msg in self._msg_ch:
            await ws.send_str(json.dumps(msg))

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            msg = await ws.receive()
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            ):
                if not self._ready.is_set():
                    raise APIConnectionError(
                        "the Camb.ai realtime session closed before it became ready"
                    )
                return

            if msg.type == aiohttp.WSMsgType.BINARY:
                self._push_audio(msg.data)
                continue

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            self._handle_event(json.loads(msg.data))

    def _handle_event(self, event: dict[str, Any]) -> None:
        etype = event.get("type")

        if etype == "session.created":
            self._ready.set()
        elif etype == "conversation.item.input_audio_transcription.completed":
            self._emit_input_transcript(event.get("transcript", ""))
        elif etype == "response.text.delta":
            gen = self._ensure_generation()
            if delta := event.get("delta", ""):
                gen.text_ch.send_nowait(delta)
        elif etype == "response.text.done":
            gen = self._ensure_generation()
            gen.text_done = True
            self._finish_if_complete(gen)
        elif etype == "response.audio.delta":
            if raw := (event.get("delta") or event.get("audio")):
                self._push_audio(base64.b64decode(raw))
        elif etype == "response.audio.done":
            current = self._current
            if current is not None:
                current.audio_done = True
                self._finish_if_complete(current)
        elif etype == "error":
            message = (event.get("error") or {}).get("message") or "unknown realtime error"
            logger.error("camb.ai realtime error: %s", message)
            self.emit(
                "error",
                llm.RealtimeModelError(
                    timestamp=time.time(),
                    label=self._realtime_model.label,
                    error=APIStatusError(message, status_code=500, body=None),
                    recoverable=True,
                ),
            )

    def _finish_if_complete(self, gen: _Generation) -> None:
        """A response ends when the server has reported both its text and its audio done."""
        if gen.text_done and gen.audio_done:
            self._finish_generation()

    async def _watchdog_task(self) -> None:
        """Backstop for a response the server never reports as done."""
        while True:
            await asyncio.sleep(0.2)
            gen = self._current
            if gen and time.monotonic() - gen.last_audio_at > _AUDIO_IDLE_TIMEOUT:
                # Not conditional on text.done: the server can stream deltas for a
                # response it never reports as done, and a turn that never ends blocks
                # every utterance after it.
                self._finish_generation()

    def _ensure_generation(self) -> _Generation:
        if self._current is not None:
            return self._current

        self._item_id += 1
        message_id = f"camb-translation-{self._item_id}"
        gen = _Generation(
            message_id=message_id,
            text_ch=utils.aio.Chan[str](),
            audio_ch=utils.aio.Chan[rtc.AudioFrame](),
            message_ch=utils.aio.Chan[llm.MessageGeneration](),
            modalities=asyncio.Future[list[Literal["text", "audio"]]](),
            started_at=self._turn_started_at or time.time(),
        )
        gen.modalities.set_result(["audio", "text"])
        self._current = gen

        gen.message_ch.send_nowait(
            llm.MessageGeneration(
                message_id=message_id,
                text_stream=gen.text_ch,
                audio_stream=gen.audio_ch,
                modalities=gen.modalities,
            )
        )

        ev = llm.GenerationCreatedEvent(
            message_stream=gen.message_ch,
            function_stream=utils.aio.Chan[llm.FunctionCall](),
            user_initiated=False,
            response_id=message_id,
        )
        if self._pending_reply is not None and not self._pending_reply.done():
            self._pending_reply.set_result(ev)
            self._pending_reply = None
        self.emit("generation_created", ev)
        return gen

    def _push_audio(self, data: bytes) -> None:
        if not data:
            return
        gen = self._ensure_generation()
        gen.last_audio_at = time.monotonic()
        for frame in self._bstream.push(data):
            gen.audio_ch.send_nowait(frame)

    def _finish_generation(self) -> None:
        gen, self._current = self._current, None
        if gen is None:
            return
        for frame in self._bstream.flush():
            gen.audio_ch.send_nowait(frame)
        for ch in (gen.text_ch, gen.audio_ch, gen.message_ch):
            if not ch.closed:
                ch.close()
        self._turn_started_at = None

    def _emit_input_transcript(self, transcript: str) -> None:
        self._item_id += 1
        self.emit(
            "input_audio_transcription_completed",
            llm.InputTranscriptionCompleted(
                item_id=f"camb-source-{self._item_id}",
                transcript=transcript,
                is_final=True,
                turn_started_at=self._turn_started_at,
            ),
        )

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._turn_started_at is None:
            self._turn_started_at = time.time()
        for f in self._resample(frame):
            self._msg_ch.send_nowait(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(f.data.tobytes()).decode(),
                }
            )

    def _resample(self, frame: rtc.AudioFrame) -> Iterator[rtc.AudioFrame]:
        if self._input_resampler and frame.sample_rate != self._input_resampler._input_rate:
            self._input_resampler = None

        if self._input_resampler is None and (
            frame.sample_rate != REALTIME_SAMPLE_RATE or frame.num_channels != NUM_CHANNELS
        ):
            self._input_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=REALTIME_SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        if self._input_resampler:
            yield from self._input_resampler.push(frame)
        else:
            yield frame

    def push_video(self, frame: rtc.VideoFrame) -> None:
        pass

    def commit_audio(self) -> None:
        pass

    def clear_audio(self) -> None:
        pass

    def generate_reply(
        self,
        *,
        instructions: NotGivenOr[str] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool]] = NOT_GIVEN,
    ) -> asyncio.Future[llm.GenerationCreatedEvent]:
        """Resolve with the next translation the server produces.

        Translation is driven by the incoming speech, so this does not prompt the server;
        it hands back the generation that the next utterance creates. ``instructions`` is
        not supported and is ignored.
        """
        if is_given(instructions):
            logger.warning("camb.ai realtime translation ignores per-reply instructions")

        if self._pending_reply is not None and not self._pending_reply.done():
            return self._pending_reply

        fut = asyncio.Future[llm.GenerationCreatedEvent]()
        self._pending_reply = fut
        return fut

    def interrupt(self) -> None:
        pass

    def truncate(
        self,
        *,
        message_id: str,
        modalities: list[Literal["text", "audio"]],
        audio_end_ms: int,
        audio_transcript: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        pass

    @property
    def chat_ctx(self) -> llm.ChatContext:
        return self._chat_ctx.copy()

    @property
    def tools(self) -> llm.ToolContext:
        return llm.ToolContext.empty()

    async def update_instructions(self, instructions: str) -> None:
        # AgentSession sets instructions on startup; a translation has none to steer.
        if instructions:
            logger.warning("camb.ai realtime translation ignores instructions")

    async def update_chat_ctx(self, chat_ctx: llm.ChatContext) -> None:
        pass

    async def update_tools(self, tools: list[llm.Tool]) -> None:
        if tools:
            raise llm.RealtimeError("Camb.ai realtime translation does not support tools")

    def update_options(self, *, tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN) -> None:
        pass

    async def aclose(self) -> None:
        self._msg_ch.close()
        await utils.aio.cancel_and_wait(self._main_atask)


__all__ = ["RealtimeModel", "RealtimeSession", "REALTIME_BASE_URL"]
