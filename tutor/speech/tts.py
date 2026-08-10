"""Text-to-speech via the xAI /v1/tts endpoint (JSON, no model name).

``speak()`` plays on the machine running the server (ffplay) — right for the
local-mic setup, where laptop and student share a room. ``synthesize()`` returns
same audio instead, for devices that own the speaker: the browser client gets
the bytes over its WebSocket and plays them itself (so nothing is played on an
SSH host nobody is sitting at). Speakers with no audio return None.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

from tutor.console import say

from tutor.config import Settings

log = logging.getLogger(__name__)

NO_AUDIO_HINT = (
    "이 기기에서는 튜터 음성을 재생할 수 없습니다. "
    "브라우저 클라이언트를 열면 그 기기의 스피커로 나옵니다: http://localhost:8765/ "
    "(서버가 원격이면 ssh -N -L 8765:localhost:8765 <user>@<host> 로 터널을 먼저 여세요)"
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


def can_play_locally() -> bool:
    """Both halves are needed: a player binary AND somewhere to play it."""
    return shutil.which("ffplay") is not None and has_local_audio_output()


class XaiSpeaker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._player = shutil.which("ffplay")
        self._can_play = self._player is not None and has_local_audio_output()
        if self._player is None:
            log.warning("ffplay not found: TTS audio cannot play here. %s", NO_AUDIO_HINT)
        elif not self._can_play:
            log.warning("no audio output device. %s", NO_AUDIO_HINT)
        # TTS_TRANSPORT=ws: ONE bidirectional socket, held open and reused per
        # utterance. The ~1s the HTTP path pays per request is mostly TLS and
        # request setup; over a live socket the first audio arrives in ~0.3s.
        self._ws = None
        self._ws_lock = threading.Lock()
        self._ws_connect = None  # lazily imported; tests inject a fake

    audio_format = "mp3"

    def _request(self, text: str) -> dict:
        return {
            "url": f"{self.settings.xai_base_url.rstrip('/')}/tts",
            "headers": {"Authorization": f"Bearer {self.settings.xai_api_key}"},
            "json": {
                "text": text,
                "voice_id": self.settings.tts_voice,
                "language": self.settings.tutor_language,
            },
        }

    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        req = self._request(text)
        resp = httpx.post(req["url"], headers=req["headers"], json=req["json"], timeout=60)
        resp.raise_for_status()
        return resp.content

    def synthesize_stream(self, text: str):
        """Yield the utterance as MP3 chunks while xAI is still rendering it.

        Two transports, same shape. HTTP: the /tts endpoint answers with
        Transfer-Encoding: chunked — first chunk ~1.3s, full file ~2.5s.
        WS (TTS_TRANSPORT=ws): a held-open socket skips the per-request
        setup, first chunk ~0.3s. A websocket failure BEFORE any audio falls
        back to HTTP for that line — the student hears it either way; a
        failure mid-utterance cannot (replaying from the top would stutter),
        so the line ends where the socket did.
        """
        if not text:
            return
        if self.settings.tts_transport == "ws":
            spoke = False
            try:
                for chunk in self._stream_ws(text):
                    spoke = True
                    yield chunk
                return
            except GeneratorExit:
                raise  # barge-in: _stream_ws already reset the socket
            except Exception as e:  # noqa: BLE001 — any ws trouble → HTTP
                if spoke:
                    log.warning("TTS websocket died mid-utterance (%s); line truncated", e)
                    return
                log.warning("TTS websocket failed (%s); HTTP fallback for this line", e)
        req = self._request(text)
        with httpx.stream(
            "POST", req["url"], headers=req["headers"], json=req["json"], timeout=60
        ) as resp:
            resp.raise_for_status()
            yield from resp.iter_bytes()

    # --- the websocket transport ---------------------------------------------

    def _ws_url(self) -> str:
        host = (self.settings.xai_base_url.rstrip("/")
                .replace("https://", "wss://").replace("http://", "ws://"))
        return (f"{host}/tts?language={quote(self.settings.tutor_language)}"
                f"&voice={quote(self.settings.tts_voice)}&codec=mp3")

    def _ws_open(self):
        if self._ws is None:
            if self._ws_connect is None:
                from websockets.sync.client import connect
                self._ws_connect = connect
            self._ws = self._ws_connect(
                self._ws_url(),
                additional_headers={
                    "Authorization": f"Bearer {self.settings.xai_api_key}"
                },
                open_timeout=10,
            )
            log.info("TTS websocket open (reused across utterances)")
        return self._ws

    def _ws_reset(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 — it is already being discarded
                pass

    def _stream_ws(self, text: str):
        """One utterance over the shared socket: text in, MP3 chunks out.

        The lock serializes utterances — the protocol interleaves nothing, so
        a second speaker thread must wait, exactly like the audio queue does.
        Any exit that is not a clean audio.done leaves the stream state
        unknowable (a barge-in dropped us mid-reply, the socket broke), so the
        connection is thrown away and the next utterance redials.
        """
        with self._ws_lock:
            try:
                ws = self._ws_open()
                ws.send(json.dumps({"type": "text.delta", "delta": text}))
                ws.send(json.dumps({"type": "text.done"}))
                while True:
                    msg = json.loads(ws.recv(timeout=30))
                    kind = msg.get("type")
                    if kind == "audio.delta":
                        audio = base64.b64decode(msg.get("delta") or "")
                        if audio:
                            yield audio
                    elif kind == "audio.done":
                        return
                    elif kind == "error":
                        raise RuntimeError(msg.get("message") or "tts websocket error")
                    # anything else (audio.clear, future frames): keep reading
            except BaseException:
                self._ws_reset()
                raise

    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio:
            self.play(audio)

    def play(self, mp3: bytes) -> None:
        """Play audio we already have — a cached phrase costs no TTS call."""
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
            # say(), not print(): a cp949 console cannot encode the emoji, and
            # echo mode dying on its own output makes the whole mode useless
            say(f"[TUTOR 🔊] {text}")

    def synthesize_stream(self, text: str):
        """No audio to stream: print, yield nothing — the text-only path."""
        self.speak(text)
        return iter(())

    def play(self, audio: bytes) -> None:
        pass  # echo mode has no audio to play


class NullSpeaker:
    """Test double. ``spoken`` is what was played HERE, ``synthesized`` what was
    handed to a device to play — the browser path must never touch ``spoken``."""

    audio_format = "mp3"

    def __init__(self, audio: bytes | None = None, stream_chunks: int = 1):
        self.spoken: list[str] = []
        self.synthesized: list[str] = []
        self.played: list[bytes] = []
        self.audio = audio  # what synthesize() hands back, if anything
        self.stream_chunks = stream_chunks  # >1 simulates chunked TTS arrival

    def play(self, audio: bytes) -> None:
        self.played.append(audio)

    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        self.synthesized.append(text)
        return self.audio

    def synthesize_stream(self, text: str):
        audio = self.synthesize(text)
        if not audio:
            return
        n = max(1, self.stream_chunks)
        size = max(1, -(-len(audio) // n))  # ceil division: n pieces, none empty
        for i in range(0, len(audio), size):
            yield audio[i : i + size]

    def speak(self, text: str) -> None:
        if text:
            self.spoken.append(text)
