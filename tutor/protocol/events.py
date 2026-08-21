"""EVENT messages: text frames on the device WebSocket.

Envelope: {"type": "EVENT", "event": "<name>", "data": {...}, "ts": <unix ms>}
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from tutor.protocol.frames import ProtocolError

# 디바이스 → 서버로 올라오는 이벤트 이름 목록(허용 목록).
DEVICE_EVENTS = {
    "hello",
    "hint_request",
    "capture_failed",
    # the device finished playing a TTS_AUDIO frame — its mic may open again
    "playback_done",
    "error",
}
# 서버 → 디바이스로 내려보내는 이벤트 이름 목록(허용 목록).
SERVER_EVENTS = {
    "hello_ack",
    "capture_request",
    "speech_state",
    # what STT heard, and whether it starts a hint flow — a hands-free device
    # needs this to know an utterance produced no response and it may listen again
    "transcript",
    # what the tutor says, and the turn-taking state, for devices whose VAD runs
    # on the server (browser client): LISTENING/USER_SPEAKING/PROCESSING/AGENT_SPEAKING
    "tutor_says",
    # A generated hint is committed word-by-word after its rolling leak guard.
    # start/delta/done keeps one chat bubble and one TTS utterance across those
    # commits; tutor_says remains the complete-line path for fixed reactions.
    "tutor_stream_start",
    "tutor_stream_delta",
    "tutor_stream_done",
    "turn_state",
    "hint_issued",
    "error",
}


# EVENT 텍스트 프레임의 공통 봉투 구조.
class Event(BaseModel):
    type: str = "EVENT"
    event: str
    data: dict[str, Any] = {}
    ts: int = 0


# 이벤트 이름 + 데이터를 타임스탬프까지 붙여 JSON 문자열로 만든다.
def make_event(event: str, data: dict[str, Any] | None = None) -> str:
    return Event(event=event, data=data or {}, ts=int(time.time() * 1000)).model_dump_json()


# 받은 텍스트 프레임을 Event로 검증·파싱한다(형식이 어긋나면 ProtocolError).
def parse_event(text: str) -> Event:
    try:
        ev = Event.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ProtocolError(f"bad EVENT frame: {e}") from e
    if ev.type != "EVENT":
        raise ProtocolError(f"unexpected text frame type {ev.type!r}")
    return ev
