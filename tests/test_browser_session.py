"""Browser client against a real in-process server, with a scripted VAD.

Covers the SSH topology end to end: PCM in over the socket → server-side Silero
seam → endpointed utterance → pipeline → TTS bytes back out → playback_done →
LISTENING again, several turns, no button anywhere.
"""

import asyncio
import math

import httpx
import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import (
    AudioHeader,
    ImageHeader,
    TtsAudioFrame,
    decode,
    encode_audio,
    encode_image,
)
from tutor.server.app import serve_static
from tutor.server.browser import BrowserSession
from tutor.server.session import Deps
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.speech.turn import TurnConfig
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

WORKSHEET = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "student_work": ["3*x = 20 + 5"],
    "confidence": 0.95,
}
MP3 = b"ID3\x04\x00fake-mp3-bytes"
JPEG = b"\xff\xd8" + b"\x00" * 4096  # what the page's file picker would hold

SETTINGS = Settings(capture_timeout_s=2.0, vad_tail_guard_ms=20)
CFG = TurnConfig.from_settings(SETTINGS)
SILENCE_FRAMES = math.ceil(CFG.silence_ms / CFG.frame_ms)


class ScriptedVAD:
    def __init__(self):
        self.script: list[bool] = []
        self.seen = 0

    def is_speech(self, frame, sample_rate=None) -> bool:
        self.seen += 1
        return self.script.pop(0) if self.script else False

    def reset(self) -> None:
        pass


class FakeBrowser:
    """What tutor/web/index.html does, minus the audio hardware."""

    def __init__(self, url: str, vad: ScriptedVAD, worksheet: bytes | None = None):
        self.url = url
        self.vad = vad
        self.worksheet = worksheet  # what the page's file picker holds, if anything
        self.states: list[str] = []
        self.heard: list[str] = []
        self.tutor_said: list[str] = []
        self.audio: list[TtsAudioFrame] = []
        self.gated = True  # like the page: stop sending unless LISTENING/USER_SPEAKING
        self.state = ""
        self._listening = asyncio.Event()

    async def __aenter__(self):
        self._ws = await connect(self.url, max_size=16 * 1024 * 1024).__aenter__()
        self._reader = asyncio.create_task(self._read())
        await self._ws.send(
            make_event("hello", {"device_id": "browser", "caps": ["mic", "speaker"]})
        )
        await asyncio.wait_for(self._listening.wait(), timeout=10)
        return self

    async def __aexit__(self, *exc):
        self._reader.cancel()
        await self._ws.close()

    async def _read(self) -> None:
        async for raw in self._ws:
            if not isinstance(raw, str):
                frame = decode(raw)
                assert isinstance(frame, TtsAudioFrame)
                self.audio.append(frame)
                await self._ws.send(make_event("playback_done"))  # "finished playing"
                continue
            ev = parse_event(raw)
            if ev.event == "turn_state":
                self.state = ev.data["state"]
                self.states.append(self.state)
                if self.state == "LISTENING":
                    self._listening.set()
                else:
                    self._listening.clear()
            elif ev.event == "transcript":
                self.heard.append(ev.data["text"])
            elif ev.event == "tutor_says":
                self.tutor_said.append(ev.data["text"])
            elif ev.event == "capture_request":
                capture_id = ev.data["capture_id"]
                if self.worksheet is None:
                    await self._ws.send(make_event("capture_failed", {"capture_id": capture_id}))
                else:
                    await self._ws.send(
                        encode_image(self.worksheet, ImageHeader(capture_id=capture_id))
                    )

    def _mic_open(self) -> bool:
        return self.state in ("LISTENING", "USER_SPEAKING")

    async def wait_until(self, predicate, timeout: float = 15.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() > deadline:
                raise AssertionError(f"timed out; state={self.state}")
            await asyncio.sleep(0.005)

    async def send_frames(self, flags: list[bool]) -> None:
        """One 512-sample frame per flag, as the page's onSamples() does."""
        for flag in flags:
            if self.gated and not self._mic_open():
                continue
            self.vad.script.append(flag)
            await self._ws.send(
                encode_audio(
                    bytes(CFG.frame_bytes),
                    AudioHeader(stream_id="browser", sample_rate=CFG.sample_rate),
                )
            )
            await asyncio.sleep(0)

    async def speak_a_turn(self) -> None:
        await self.send_frames([True] * 30 + [False] * SILENCE_FRAMES)
        await asyncio.wait_for(self._listening.wait(), timeout=15)


@pytest.fixture
def deps(db):
    llm = EchoLLMClient({"recognize": [WORKSHEET] * 5})
    return Deps(
        settings=SETTINGS,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(audio=MP3),
        store=SessionStore(),
    )


async def browser_server(deps, vad):
    async def handler(ws):
        await BrowserSession(ws, deps, vad=vad).run()

    return serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024,
                 process_request=serve_static)


async def test_one_turn_streams_pcm_in_and_audio_out(deps):
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            await browser.speak_a_turn()

    # the endpointed utterance went through STT → pipeline → hint
    assert browser.heard == ["힌트 주세요"]
    assert len(browser.tutor_said) == 1 and browser.tutor_said[0]
    assert deps.store.get_history()[0].level == 1
    # ...and came back as audio for the laptop to play
    assert [f.audio for f in browser.audio] == [MP3]
    assert browser.audio[0].header.format == "mp3"
    # every state, in order, with no button
    assert browser.states == [
        "LISTENING",
        "USER_SPEAKING",
        "PROCESSING",
        "AGENT_SPEAKING",
        "LISTENING",
    ]


async def test_server_speaker_is_never_used(deps):
    """On an SSH host nobody is listening: audio must go to the browser only."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser.speak_a_turn()

    assert deps.speaker.spoken == []  # nothing played on the server
    assert len(deps.speaker.synthesized) == 1  # handed to the device instead
    assert len(browser.audio) == 1


async def test_three_turns_hands_free(deps):
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            for _ in range(3):
                await browser.speak_a_turn()

    assert len(browser.audio) == 3
    assert [h.level for h in deps.store.get_history()] == [1, 2, 3]  # same session
    assert browser.states[-1] == "LISTENING"


async def test_without_a_worksheet_the_tutor_asks_for_one(deps):
    """No camera on a laptop: the pipeline degrades to ASK_RECAPTURE, not silence."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser.speak_a_turn()

    assert deps.store.get_history()[0].action == "ASK_RECAPTURE"
    assert len(browser.audio) == 1  # still spoken, still conversational
    assert browser.states[-1] == "LISTENING"


async def test_vad_ignores_mic_while_the_tutor_answers(deps):
    """An ungated client keeps streaming speech; the server must not hear it."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            browser.gated = False  # worst case: the page's own gate is broken
            await browser.send_frames([True] * 30 + [False] * SILENCE_FRAMES)
            # only PROCESSING proves the endpoint frame itself was consumed
            await browser.wait_until(
                lambda: browser.state in ("PROCESSING", "AGENT_SPEAKING")
            )
            seen_after_turn = vad.seen
            # the tutor's own voice, straight back into the mic
            await browser.send_frames([True] * 60)
            await browser.wait_until(lambda: browser.state == "LISTENING")

    assert vad.seen == seen_after_turn  # not one frame reached the model
    assert len(browser.audio) == 1  # and no second turn was triggered
    assert browser.states[-1] == "LISTENING"


async def test_no_tts_audio_still_returns_to_listening(deps):
    """Echo mode (no key, no audio): text only, conversation continues."""
    deps.speaker = NullSpeaker(audio=None)
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser.speak_a_turn()

    assert browser.audio == []
    assert len(browser.tutor_said) == 1
    assert browser.states[-1] == "LISTENING"


async def test_page_and_worklet_are_served_on_the_same_port(deps):
    """One SSH tunnel has to be enough: HTTP and WS share the port."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
            page = await http.get("/")
            worklet = await http.get("/worklet.js")
            missing = await http.get("/../etc/passwd")

    assert page.status_code == 200 and "pcm-capture" in page.text
    assert "text/html" in page.headers["content-type"]
    assert worklet.status_code == 200 and "registerProcessor" in worklet.text
    assert missing.status_code == 404


async def test_wrong_sample_rate_is_rejected(deps):
    """8 kHz would silently mis-frame Silero; fail loudly instead."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser._ws.send(
                encode_audio(
                    bytes(CFG.frame_bytes),
                    AudioHeader(stream_id="browser", sample_rate=8000),
                )
            )
            await asyncio.sleep(0.2)
            assert browser.state == "LISTENING"  # session survives a bad frame
            await browser.speak_a_turn()  # and still works

    assert len(browser.audio) == 1
