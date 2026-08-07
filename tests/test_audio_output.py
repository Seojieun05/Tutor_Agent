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
