"""Where the tutor's voice comes out — and saying so when it cannot.

On a headless SSH host ffplay exits 0 while ALSA has no card, so a failed
playback looked exactly like a successful one: the tutor "spoke" and the
student heard nothing, with nothing in the log to explain it.
"""

import re

import httpx
import pytest

from tutor.config import Settings
from tutor.speech import tts
from tutor.speech.tts import XaiSpeaker, has_local_audio_output


class TestAudioDetection:
    def test_no_alsa_card_means_no_output(self, monkeypatch, tmp_path):
        monkeypatch.setattr(tts.sys, "platform", "linux")
        monkeypatch.delenv("PULSE_SERVER", raising=False)
        monkeypatch.setattr(tts, "Path", lambda p: tmp_path / "missing")
        assert has_local_audio_output() is False

    def test_pulse_server_counts_as_output(self, monkeypatch):
        monkeypatch.setattr(tts.sys, "platform", "linux")
        monkeypatch.setenv("PULSE_SERVER", "unix:/run/pulse")
        assert has_local_audio_output() is True

    def test_non_linux_assumes_a_default_device(self, monkeypatch):
        monkeypatch.setattr(tts.sys, "platform", "darwin")
        assert has_local_audio_output() is True


class TestSilentFailureIsLoud:
    def _speaker(self, monkeypatch, can_play: bool) -> XaiSpeaker:
        monkeypatch.setattr(tts.shutil, "which", lambda name: "/usr/bin/ffplay")
        monkeypatch.setattr(tts, "has_local_audio_output", lambda: can_play)
        return XaiSpeaker(Settings(xai_api_key="test"))

    def test_playback_without_a_device_logs_an_actionable_error(
        self, monkeypatch, caplog
    ):
        speaker = self._speaker(monkeypatch, can_play=False)
        ran = []
        monkeypatch.setattr(tts.subprocess, "run", lambda *a, **k: ran.append(a))

        with caplog.at_level("ERROR"):
            speaker.play(b"fake-mp3")

        assert ran == []  # no pointless subprocess, no ALSA spew
        assert caplog.records, "silence with no explanation is the bug"
        message = caplog.records[-1].getMessage()
        assert "브라우저" in message and "8765" in message  # tells you what to do

    def test_playback_with_a_device_still_plays(self, monkeypatch):
        speaker = self._speaker(monkeypatch, can_play=True)
        ran = []
        monkeypatch.setattr(tts.subprocess, "run", lambda *a, **k: ran.append(a[0]))
        speaker.play(b"fake-mp3")
        assert ran and ran[0][0] == "/usr/bin/ffplay"

    def test_synthesize_is_independent_of_local_playback(self, monkeypatch):
        """The browser client needs the bytes even though the host is mute."""
        speaker = self._speaker(monkeypatch, can_play=False)
        monkeypatch.setattr(
            tts.httpx,
            "post",
            lambda *a, **k: httpx.Response(200, content=b"ID3-audio",
                                           request=httpx.Request("POST", "http://x")),
        )
        assert speaker.synthesize("안녕하세요") == b"ID3-audio"


class FakeWs:
    """The xAI TTS websocket, scripted: frames out, sent messages recorded."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []
        self.closed = False

    def send(self, data):
        import json

        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        import json

        frame = self.frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return json.dumps(frame)

    def close(self):
        self.closed = True


def b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


class TestWebSocketTts:
    """TTS_TRANSPORT=ws: one held-open socket, ~0.3s to first audio instead of
    the ~1.3s the per-request HTTP path pays in setup."""

    def _speaker(self, monkeypatch, frames) -> tuple[XaiSpeaker, FakeWs, list]:
        monkeypatch.setattr(tts.shutil, "which", lambda name: None)
        speaker = XaiSpeaker(Settings(xai_api_key="k", tts_transport="ws"))
        ws = FakeWs(frames)
        connects: list = []
        speaker._ws_connect = lambda url, **kw: (connects.append(url), ws)[1]
        return speaker, ws, connects

    def test_chunks_stream_in_order_and_the_socket_is_reused(self, monkeypatch):
        speaker, ws, connects = self._speaker(monkeypatch, [
            {"type": "audio.delta", "delta": b64(b"one")},
            {"type": "audio.delta", "delta": b64(b"two")},
            {"type": "audio.done"},
            {"type": "audio.delta", "delta": b64(b"three")},
            {"type": "audio.done"},
        ])
        assert list(speaker.synthesize_stream("첫 문장")) == [b"one", b"two"]
        assert list(speaker.synthesize_stream("둘째 문장")) == [b"three"]
        assert len(connects) == 1                    # ONE dial, many utterances
        assert "optimize_streaming_latency=2" in connects[0]
        assert [m["type"] for m in ws.sent] == [
            "text.delta", "text.done", "text.delta", "text.done",
        ]
        assert not ws.closed

    def test_live_text_deltas_feed_one_audio_utterance(self, monkeypatch):
        speaker, ws, _ = self._speaker(monkeypatch, [
            {"type": "audio.delta", "delta": b64(b"early")},
            {"type": "audio.done"},
        ])
        assert list(speaker.synthesize_text_stream(iter(["첫 어절 ", "둘째 어절"]))) == [
            b"early"
        ]
        assert ws.sent == [
            {"type": "text.delta", "delta": "첫 어절 "},
            {"type": "text.delta", "delta": "둘째 어절"},
            {"type": "text.done"},
        ]

    def test_a_dead_socket_falls_back_to_http_for_that_line(self, monkeypatch):
        monkeypatch.setattr(tts.shutil, "which", lambda name: None)
        speaker = XaiSpeaker(Settings(xai_api_key="k", tts_transport="ws"))
        speaker._ws_connect = lambda url, **kw: (_ for _ in ()).throw(OSError("refused"))

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def raise_for_status(self):
                return None

            def iter_bytes(self):
                yield b"http-audio"

        monkeypatch.setattr(tts.httpx, "stream", lambda *a, **k: FakeResp())
        assert list(speaker.synthesize_stream("안녕하세요")) == [b"http-audio"]

    def test_a_mid_utterance_death_truncates_instead_of_replaying(self, monkeypatch):
        """After audio has played, an HTTP retry would replay the line from the
        top over what was already heard: the line ends where the socket did."""
        speaker, ws, _ = self._speaker(monkeypatch, [
            {"type": "audio.delta", "delta": b64(b"heard")},
            OSError("connection reset"),
        ])
        monkeypatch.setattr(tts.httpx, "stream",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("no HTTP retry")))
        assert list(speaker.synthesize_stream("깨진 문장")) == [b"heard"]
        assert speaker._ws is None                   # thrown away, next line redials

    def test_a_barge_in_resets_the_socket(self, monkeypatch):
        """A dropped generator mid-reply leaves the stream unknowable: the
        next utterance must not read this one's tail."""
        speaker, ws, _ = self._speaker(monkeypatch, [
            {"type": "audio.delta", "delta": b64(b"start")},
            {"type": "audio.delta", "delta": b64(b"never-read")},
            {"type": "audio.done"},
        ])
        stream = speaker.synthesize_stream("끊길 문장")
        assert next(stream) == b"start"
        stream.close()                               # the barge-in
        assert ws.closed and speaker._ws is None


class TestBrowserPage:
    """The page is what actually makes sound on the laptop."""

    def test_unlock_clip_is_real_playable_silence(self):
        import base64
        import io
        import wave
        from pathlib import Path

        html = (Path(__file__).resolve().parent.parent / "tutor" / "web" / "index.html").read_text(
            encoding="utf-8"
        )
        block = re.search(r"const SILENCE =\s*(.*?);\n", html, re.S).group(1)
        uri = "".join(re.findall(r'"([^"]*)"', block))
        raw = base64.b64decode(uri.split(",", 1)[1])
        with wave.open(io.BytesIO(raw)) as w:
            assert 0 < w.getnframes() / w.getframerate() < 1  # short, but not empty
        assert set(raw[44:]) == {128}, "the unlock clip must be silent"

    def test_unlock_happens_before_the_socket_opens(self):
        from pathlib import Path

        html = (Path(__file__).resolve().parent.parent / "tutor" / "web" / "index.html").read_text(
            encoding="utf-8"
        )
        # inside the click handler, and before connect(): a later unlock would
        # be outside the user gesture and the browser would refuse to play
        assert html.index("unlockAudio()") < html.index("connect();")


class TestPortPreflight:
    """A busy port is the commonest restart mistake — say so, do not traceback."""

    def test_a_busy_port_is_detected(self):
        import socket

        from tutor.server.app import port_is_free

        with socket.socket() as taken:
            taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            port = taken.getsockname()[1]
            assert port_is_free("127.0.0.1", port) is False
        assert port_is_free("127.0.0.1", port) is True

    def test_the_message_names_the_way_out(self):
        from tutor.config import Settings
        from tutor.server.app import port_in_use_help

        help_text = port_in_use_help(Settings(ws_port=8765))
        assert "8765" in help_text
        assert "pkill" in help_text          # how to stop the old one
        assert "WS_PORT=8766" in help_text   # or how to avoid it
