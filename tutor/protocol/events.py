"""EVENT messages: text frames on the device WebSocket.

Envelope: {"type": "EVENT", "event": "<name>", "data": {...}, "ts": <unix ms>}
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from tutor.protocol.frames import ProtocolError

DEVICE_EVENTS = {"hello", "hint_request", "capture_failed", "error"}
SERVER_EVENTS = {
    "hello_ack",
    "capture_request",
    "speech_state",
    # what STT heard, and whether it starts a hint flow — a hands-free device
    # needs this to know an utterance produced no response and it may listen again
    "transcript",
    "hint_issued",
    "error",
}


class Event(BaseModel):
    type: str = "EVENT"
    event: str
    data: dict[str, Any] = {}
    ts: int = 0


def make_event(event: str, data: dict[str, Any] | None = None) -> str:
    return Event(event=event, data=data or {}, ts=int(time.time() * 1000)).model_dump_json()


def parse_event(text: str) -> Event:
    try:
        ev = Event.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise ProtocolError(f"bad EVENT frame: {e}") from e
    if ev.type != "EVENT":
        raise ProtocolError(f"unexpected text frame type {ev.type!r}")
    return ev
