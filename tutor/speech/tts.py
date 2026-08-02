"""Text-to-speech via the xAI /v1/tts endpoint (JSON, no model name), played
on the laptop speaker with ffplay."""

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

    def speak(self, text: str) -> None:
        if not text:
            return
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
        self._play(resp.content)

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
    """No-key mode: print instead of speaking."""

    def __init__(self, settings: Settings | None = None):
        pass

    def speak(self, text: str) -> None:
        if text:
            print(f"[TUTOR 🔊] {text}", flush=True)


class NullSpeaker:
    """Test double: records what would have been spoken."""

    def __init__(self):
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        if text:
            self.spoken.append(text)
