import pytest

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import (
    AudioFrame,
    AudioHeader,
    ImageFrame,
    ImageHeader,
    ProtocolError,
    decode,
    encode_audio,
    encode_image,
)


def test_image_round_trip():
    jpeg = b"\xff\xd8fakejpegdata"
    frame = decode(encode_image(jpeg, ImageHeader(capture_id="cap-1", width=960, height=640)))
    assert isinstance(frame, ImageFrame)
    assert frame.header.capture_id == "cap-1"
    assert frame.jpeg == jpeg


def test_audio_round_trip():
    pcm = b"\x00\x01" * 160
    frame = decode(encode_audio(pcm, AudioHeader(stream_id="utt-1", seq=3, last=True)))
    assert isinstance(frame, AudioFrame)
    assert frame.header.last is True
    assert frame.header.sample_rate == 16000
    assert frame.pcm == pcm


def test_non_jpeg_rejected_on_encode_and_decode():
    with pytest.raises(ProtocolError):
        encode_image(b"x", ImageHeader(capture_id="c", format="png"))
    # hand-craft a png-format frame
    import json
    import struct

    header = json.dumps({"capture_id": "c", "format": "png"}).encode()
    raw = bytes([0x01]) + struct.pack(">I", len(header)) + header + b"data"
    with pytest.raises(ProtocolError):
        decode(raw)


def test_truncated_and_garbage_frames():
    with pytest.raises(ProtocolError):
        decode(b"\x01\x00")
    good = encode_image(b"payload", ImageHeader(capture_id="c"))
    with pytest.raises(ProtocolError):
        decode(good[:8])
    with pytest.raises(ProtocolError):
        decode(b"\x09\x00\x00\x00\x02{}")  # unknown type


def test_event_round_trip():
    ev = parse_event(make_event("capture_request", {"capture_id": "cap-9"}))
    assert ev.event == "capture_request"
    assert ev.data["capture_id"] == "cap-9"
    assert ev.ts > 0


def test_bad_event():
    with pytest.raises(ProtocolError):
        parse_event("not json")
    with pytest.raises(ProtocolError):
        parse_event('{"type": "OTHER", "event": "x"}')
