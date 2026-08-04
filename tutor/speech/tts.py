"""Text-to-speech via the xAI /v1/tts endpoint (JSON, no model name).

``speak()`` plays on the machine running the server (ffplay) — right for the
XIAO setup, where laptop and student share a room. ``synthesize()`` returns the
same audio instead, for devices that own the speaker: the browser client gets
the bytes over its WebSocket and plays them itself (so nothing is played on an
SSH host nobody is sitting at). Speakers with no audio return None.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import httpx

from tutor.config import Settings

log = logging.getLogger(__name__)


class XaiSpeaker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._player = shutil.which("ffplay")
        if self._player is None:
            log.warning("ffplay not found: TTS audio will not play")

    audio_format = "mp3"

    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        resp = httpx.post(
            f"{self.settings.xai_base_url.rstrip('/')}/tts",
            headers={"Authorization": f"Bearer {self.settings.xai_api_key}"},
            json={
                "text": text,
                "voice_id": self.settings.tts_voice,
                "language": self.settings.tutor_language,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.content

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio:
            self._play(audio)

    def _play(self, mp3: bytes) -> None:
        if self._player is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(mp3)
            path = f.name
        try:
            subprocess.run(
                [self._player, "-nodisp", "-autoexit", "-loglevel", "quiet", path],
                check=False,
            )
        finally:
            Path(path).unlink(missing_ok=True)


class EchoSpeaker:
    """No-key mode: print instead of speaking (no audio to hand out)."""

    audio_format = "mp3"

    def __init__(self, settings: Settings | None = None):
        pass

    def synthesize(self, text: str) -> bytes | None:
        self.speak(text)
        return None

    def speak(self, text: str) -> None:
        if text:
            print(f"[TUTOR 🔊] {text}", flush=True)


class NullSpeaker:
    """Test double. ``spoken`` is what was played HERE, ``synthesized`` what was
    handed to a device to play — the browser path must never touch ``spoken``."""

    audio_format = "mp3"

    def __init__(self, audio: bytes | None = None):
        self.spoken: list[str] = []
        self.synthesized: list[str] = []
        self.audio = audio  # what synthesize() hands back, if anything

    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        self.synthesized.append(text)
        return self.audio

    def speak(self, text: str) -> None:
        if text:
            self.spoken.append(text)
