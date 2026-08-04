"""Text-to-speech via the xAI /v1/tts endpoint (JSON, no model name).

``speak()`` plays on the machine running the server (ffplay) — right for the
XIAO setup, where laptop and student share a room. ``synthesize()`` returns the
same audio instead, for devices that own the speaker: the browser client gets
the bytes over its WebSocket and plays them itself (so nothing is played on an
SSH host nobody is sitting at). Speakers with no audio return None.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from tutor.config import Settings

log = logging.getLogger(__name__)

NO_AUDIO_HINT = (
    "이 서버에는 재생할 오디오 장치가 없습니다 (헤드리스/SSH 호스트). "
    "노트북에서 소리를 들으려면 브라우저 클라이언트를 쓰세요: "
    "ssh -N -L 8765:localhost:8765 <user>@<host> 후 http://localhost:8765/ 접속"
)


def has_local_audio_output() -> bool:
    """Can this machine actually make a sound?

    ffplay exits 0 even when ALSA has no card, so a failed playback is
    indistinguishable from a successful one — the tutor would appear to speak
    while the student hears nothing. Check for a device up front instead.
    """
    if sys.platform != "linux":
        return True  # macOS/Windows: assume the default device works
    if os.environ.get("PULSE_SERVER"):
        return True
    try:
        if Path(f"/run/user/{os.getuid()}/pulse/native").exists():
            return True
    except OSError:
        pass
    try:
        cards = Path("/proc/asound/cards").read_text()
    except OSError:
        return False  # no /proc/asound at all: no ALSA card
    return any(line.strip() and "no soundcards" not in line.lower()
               for line in cards.splitlines())


class XaiSpeaker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._player = shutil.which("ffplay")
        self._can_play = self._player is not None and has_local_audio_output()
        if self._player is None:
            log.warning("ffplay not found: TTS audio cannot play here. %s", NO_AUDIO_HINT)
        elif not self._can_play:
            log.warning("no audio output device. %s", NO_AUDIO_HINT)

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
        if not self._can_play:
            # Say so every time: silence with no explanation is the worst
            # failure mode this system has.
            log.error("TTS audio was generated but cannot be played. %s", NO_AUDIO_HINT)
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
