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

# 프레임 첫 바이트에 들어가는 타입 코드.
IMAGE_TYPE = 0x01
AUDIO_TYPE = 0x02
TTS_AUDIO_TYPE = 0x03

# 헤더 길이를 담는 4바이트 big-endian 정수.
_HEADER_LEN_STRUCT = struct.Struct(">I")


# 프레임 규격을 벗어났을 때 던지는 예외.
class ProtocolError(Exception):
    pass


# IMAGE 프레임 헤더: 어떤 촬영 요청(capture_id)에 대한 JPEG인지 알려준다.
class ImageHeader(BaseModel):
    capture_id: str
    format: str = "jpeg"
    width: int | None = None
    height: int | None = None
    seq: int = 0


# AUDIO 프레임 헤더: 마이크 PCM 스트림의 포맷·순번·마지막 조각 여부.
class AudioHeader(BaseModel):
    stream_id: str
    sample_rate: int = 16000
    bits: int = 16
    channels: int = 1
    seq: int = 0
    last: bool = False


# 디코딩된 IMAGE 프레임(헤더 + JPEG 바이트).
@dataclass(frozen=True)
class ImageFrame:
    header: ImageHeader
    jpeg: bytes


# 디코딩된 AUDIO 프레임(헤더 + 16bit LE 모노 PCM).
@dataclass(frozen=True)
class AudioFrame:
    header: AudioHeader
    pcm: bytes


# TTS_AUDIO 프레임 헤더: 서버 → 디바이스로 내려보내는 합성 음성 조각.
class TtsAudioHeader(BaseModel):
    utterance_id: str = ""
    format: str = "mp3"  # what the tutor's TTS returned; the device plays it as-is
    # Streaming: one utterance may arrive as many frames. The defaults make a
    # sender that ships whole files valid unchanged — a single frame IS a
    # complete stream — so old captures and simple devices keep working.
    seq: int = 0
    last: bool = True


# 디코딩된 TTS_AUDIO 프레임(헤더 + 인코딩된 음성 바이트).
@dataclass(frozen=True)
class TtsAudioFrame:
    header: TtsAudioHeader
    audio: bytes


# decode()가 돌려줄 수 있는 프레임 종류.
Frame = ImageFrame | AudioFrame | TtsAudioFrame


# 공통 인코더: [타입 1B][헤더 길이 4B][JSON 헤더][페이로드] 순으로 조립한다.
def _encode(frame_type: int, header: BaseModel, payload: bytes) -> bytes:
    header_bytes = header.model_dump_json().encode("utf-8")
    return bytes([frame_type]) + _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + payload


# 카메라 JPEG을 IMAGE 프레임으로 포장(jpeg가 아니면 거부).
def encode_image(jpeg: bytes, header: ImageHeader) -> bytes:
    if header.format != "jpeg":
        raise ProtocolError(f"IMAGE format must be 'jpeg', got {header.format!r}")
    return _encode(IMAGE_TYPE, header, jpeg)


# 마이크 PCM을 AUDIO 프레임으로 포장.
def encode_audio(pcm: bytes, header: AudioHeader) -> bytes:
    return _encode(AUDIO_TYPE, header, pcm)


# 합성된 음성 바이트를 TTS_AUDIO 프레임으로 포장.
def encode_tts_audio(audio: bytes, header: TtsAudioHeader) -> bytes:
    return _encode(TTS_AUDIO_TYPE, header, audio)


# 바이너리 프레임 하나를 타입에 맞는 Image/Audio/TtsAudio 프레임으로 해석한다.
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
