"""Binary framing for the single device WebSocket.

Layout: [1 type byte][uint32 BE header length][UTF-8 JSON header][payload]
Type 0x01 = IMAGE (payload: one JPEG), 0x02 = AUDIO (payload: raw 16-bit LE mono PCM),
0x03 = TTS_AUDIO (server → device, payload: encoded speech for a device that has
the speaker — the browser client; a local-mic device plays through the laptop).
Text frames are EVENT JSON and never pass through this module.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

IMAGE_TYPE = 0x01
AUDIO_TYPE = 0x02
TTS_AUDIO_TYPE = 0x03

_HEADER_LEN_STRUCT = struct.Struct(">I")


class ProtocolError(Exception):
    pass


class ImageHeader(BaseModel):
    capture_id: str
    format: str = "jpeg"
    width: int | None = None
    height: int | None = None
    seq: int = 0


class AudioHeader(BaseModel):
    stream_id: str
    sample_rate: int = 16000
    bits: int = 16
    channels: int = 1
    seq: int = 0
    last: bool = False


@dataclass(frozen=True)
class ImageFrame:
    header: ImageHeader
    jpeg: bytes


@dataclass(frozen=True)
class AudioFrame:
    header: AudioHeader
    pcm: bytes


class TtsAudioHeader(BaseModel):
    utterance_id: str = ""
    format: str = "mp3"  # what the tutor's TTS returned; the device plays it as-is


@dataclass(frozen=True)
class TtsAudioFrame:
    header: TtsAudioHeader
    audio: bytes


Frame = ImageFrame | AudioFrame | TtsAudioFrame


def _encode(frame_type: int, header: BaseModel, payload: bytes) -> bytes:
    header_bytes = header.model_dump_json().encode("utf-8")
    return bytes([frame_type]) + _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload


def encode_image(jpeg: bytes, header: ImageHeader) -> bytes:
    if header.format != "jpeg":
        raise ProtocolError(f"IMAGE format must be 'jpeg', got {header.format!r}")
    return _encode(IMAGE_TYPE, header, jpeg)


def encode_audio(pcm: bytes, header: AudioHeader) -> bytes:
    return _encode(AUDIO_TYPE, header, pcm)


def encode_tts_audio(audio: bytes, header: TtsAudioHeader) -> bytes:
    return _encode(TTS_AUDIO_TYPE, header, audio)


def decode(frame: bytes) -> Frame:
    if len(frame) < 5:
        raise ProtocolError(f"frame too short ({len(frame)} bytes)")
    frame_type = frame[0]
    (header_len,) = _HEADER_LEN_STRUCT.unpack_from(frame, 1)
    if len(frame) < 5 + header_len:
        raise ProtocolError("frame truncated: header length exceeds frame size")
    try:
        header_json = json.loads(frame[5 : 5 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"bad frame header: {e}") from e
    payload = frame[5 + header_len :]
    try:
        if frame_type == IMAGE_TYPE:
            header = ImageHeader.model_validate(header_json)
            if header.format != "jpeg":
                raise ProtocolError(f"IMAGE format must be 'jpeg', got {header.format!r}")
            return ImageFrame(header=header, jpeg=payload)
        if frame_type == AUDIO_TYPE:
            return AudioFrame(header=AudioHeader.model_validate(header_json), pcm=payload)
        if frame_type == TTS_AUDIO_TYPE:
            return TtsAudioFrame(
                header=TtsAudioHeader.model_validate(header_json), audio=payload
            )
    except ValidationError as e:
        raise ProtocolError(f"invalid frame header: {e}") from e
    raise ProtocolError(f"unknown frame type 0x{frame_type:02x}")
