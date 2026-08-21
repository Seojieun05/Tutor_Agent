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

# 이 기계에서 소리를 낼 수 없을 때 안내할 문구.
NO_AUDIO_HINT = (
    "이 기기에서는 튜터 음성을 재생할 수 없습니다. "
    "브라우저 클라이언트를 열면 그 기기의 스피커로 나옵니다: http://localhost:8765/ "
    "(서버가 원격이면 ssh -N -L 8765:localhost:8765 <user>@<host> 로 터널을 먼저 여세요)"
)


# 이 기계에 실제로 소리를 낼 장치가 있는지.
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
    getuid = getattr(os, "getuid", None)
    if callable(getuid):
        try:
            if Path(f"/run/user/{getuid()}/pulse/native").exists():
                return True
        except OSError:
            pass
    try:
        cards = Path("/proc/asound/cards").read_text()
    except OSError:
        return False  # no /proc/asound at all: no ALSA card
    return any(line.strip() and "no soundcards" not in line.lower()
               for line in cards.splitlines())


# 재생기(ffplay)와 출력 장치가 둘 다 있어야 여기서 소리가 난다.
def can_play_locally() -> bool:
    """Both halves are needed: a player binary AND somewhere to play it."""
    return shutil.which("ffplay") is not None and has_local_audio_output()


# xAI /v1/tts를 쓰는 음성 합성기. 서버에서 직접 재생하거나, 바이트만 만들어 기기로 보낸다.
class XaiSpeaker:
    # 재생 가능 여부를 확인하고, ws 전송을 쓸 경우를 대비해 소켓 자리를 준비한다.
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

    # TTS 요청 한 건의 주소·헤더·본문.
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

    # 문장 하나를 음성 바이트로(완성본을 한 번에).
    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        req = self._request(text)
        resp = httpx.post(req["url"], headers=req["headers"], json=req["json"], timeout=60)
        resp.raise_for_status()
        return resp.content

    # 합성이 끝나기 전부터 오디오 조각을 흘려보낸다(HTTP 또는 웹소켓).
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

    # 아직 완성되지 않은 '텍스트'를 흘려 넣고 오디오를 받는다 — 실시간 힌트가 쓰는 최단 지연 경로.
    def synthesize_text_stream(self, chunks):
        """Stream not-yet-complete text in and stream generated MP3 back out.

        This is the latency path used by live LLM hints: safe word units arrive
        over ``chunks`` while Grok is still writing, and xAI begins synthesis
        once it has enough context instead of waiting for ``text.done``.
        """
        if self.settings.tts_transport != "ws":
            text = "".join(chunks)
            yield from self.synthesize_stream(text)
            return
        spoke = False
        try:
            for chunk in self._stream_ws_text(chunks):
                spoke = True
                yield chunk
        except GeneratorExit:
            raise
        except Exception as e:  # noqa: BLE001
            # Unlike the complete-text path, the iterator may already have
            # been consumed by the sender thread. Replaying a partial spoken
            # line would stutter, so fail closed and let the next turn redial.
            log.warning(
                "streaming-input TTS websocket failed%s: %s",
                " mid-utterance" if spoke else " before audio", e,
            )

    # --- the websocket transport ---------------------------------------------

    # TTS 웹소켓 주소(언어·목소리·코덱·지연 모드 포함).
    def _ws_url(self) -> str:
        host = (self.settings.xai_base_url.rstrip("/")
                .replace("https://", "wss://").replace("http://", "ws://"))
        return (f"{host}/tts?language={quote(self.settings.tutor_language)}"
                f"&voice={quote(self.settings.tts_voice)}&codec=mp3"
                f"&optimize_streaming_latency={self.settings.tts_streaming_latency}")

    # 웹소켓을 열거나 이미 열린 것을 재사용.
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

    # 소켓을 닫고 다음 발화에서 새로 연결하게 한다.
    def _ws_reset(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                ws.close()
            except Exception:  # noqa: BLE001 — it is already being discarded
                pass

    # 완성된 문장을 웹소켓으로 보내고 오디오 조각을 받는다.
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

    # 텍스트 조각을 보내면서 동시에 오디오를 받는다(보내기는 별도 스레드).
    def _stream_ws_text(self, chunks):
        """Send text deltas and receive audio concurrently on the shared WS."""
        with self._ws_lock:
            feeder_error: list[BaseException] = []
            feeder_done = threading.Event()
            clean = False
            try:
                ws = self._ws_open()

                # 텍스트 델타를 순서대로 보내고 마지막에 종료 신호.
                def feed() -> None:
                    try:
                        for delta in chunks:
                            if delta:
                                ws.send(json.dumps({"type": "text.delta", "delta": delta}))
                        ws.send(json.dumps({"type": "text.done"}))
                    except BaseException as e:  # carried back to the receiver thread
                        feeder_error.append(e)
                        self._ws_reset()
                    finally:
                        feeder_done.set()

                feeder = threading.Thread(target=feed, name="tts-text-feed", daemon=True)
                feeder.start()
                while True:
                    msg = json.loads(ws.recv(timeout=30))
                    kind = msg.get("type")
                    if kind == "audio.delta":
                        audio = base64.b64decode(msg.get("delta") or "")
                        if audio:
                            yield audio
                    elif kind == "audio.done":
                        clean = True
                        feeder.join(timeout=1)
                        if feeder_error:
                            raise feeder_error[0]
                        return
                    elif kind == "error":
                        raise RuntimeError(msg.get("message") or "tts websocket error")
            except BaseException:
                self._ws_reset()
                raise
            finally:
                if not clean:
                    self._ws_reset()

    # 서버가 도는 기계의 스피커로 재생한다.
    def speak(self, text: str) -> None:
        audio = self.synthesize(text)
        if audio:
            self.play(audio)

    # 이미 만들어진 오디오 바이트를 ffplay로 재생.
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


# 키 없는 모드: 말하는 대신 콘솔에 출력한다.
class EchoSpeaker:
    """No-key mode: print instead of speaking (no audio to hand out)."""

    audio_format = "mp3"

    # 설정은 받지만 쓰지 않는다.
    def __init__(self, settings: Settings | None = None):
        pass

    # 출력만 하고 오디오는 없다.
    def synthesize(self, text: str) -> bytes | None:
        self.speak(text)
        return None

    # 콘솔에 튜터 대사를 찍는다.
    def speak(self, text: str) -> None:
        if text:
            # say(), not print(): a cp949 console cannot encode the emoji, and
            # echo mode dying on its own output makes the whole mode useless
            say(f"[TUTOR 🔊] {text}")

    # 낼 소리가 없으므로 출력만 하고 빈 스트림.
    def synthesize_stream(self, text: str):
        """No audio to stream: print, yield nothing — the text-only path."""
        self.speak(text)
        return iter(())

    # 조각을 모아 한 번 출력.
    def synthesize_text_stream(self, chunks):
        self.speak("".join(chunks))
        return iter(())

    # 재생할 오디오가 없다.
    def play(self, audio: bytes) -> None:
        pass  # echo mode has no audio to play


# 테스트용 대역. 여기서 재생된 것과 기기로 넘긴 것을 따로 기록한다.
class NullSpeaker:
    """Test double. ``spoken`` is what was played HERE, ``synthesized`` what was
    handed to a device to play — the browser path must never touch ``spoken``."""

    audio_format = "mp3"

    # 기록용 리스트와, 돌려줄 가짜 오디오를 준비.
    def __init__(self, audio: bytes | None = None, stream_chunks: int = 1):
        self.spoken: list[str] = []
        self.synthesized: list[str] = []
        self.played: list[bytes] = []
        self.audio = audio  # what synthesize() hands back, if anything
        self.stream_chunks = stream_chunks  # >1 simulates chunked TTS arrival

    # 재생된 오디오를 기록.
    def play(self, audio: bytes) -> None:
        self.played.append(audio)

    # 합성 요청을 기록하고 가짜 오디오를 돌려준다.
    def synthesize(self, text: str) -> bytes | None:
        if not text:
            return None
        self.synthesized.append(text)
        return self.audio

    # 가짜 오디오를 조각내어 스트리밍처럼 돌려준다.
    def synthesize_stream(self, text: str):
        audio = self.synthesize(text)
        if not audio:
            return
        n = max(1, self.stream_chunks)
        size = max(1, -(-len(audio) // n))  # ceil division: n pieces, none empty
        for i in range(0, len(audio), size):
            yield audio[i : i + size]

    # 텍스트 조각을 모아 위와 같이 처리.
    def synthesize_text_stream(self, chunks):
        text = "".join(chunks)
        yield from self.synthesize_stream(text)

    # 여기서 말한 것으로 기록.
    def speak(self, text: str) -> None:
        if text:
            self.spoken.append(text)
