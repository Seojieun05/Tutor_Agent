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
        self.captures = 0
        self.gated = True  # like the page: stop sending unless LISTENING/USER_SPEAKING
        self.state = ""
        self._listening = asyncio.Event()
        # Barge-in bookkeeping, mirroring the page's audio queue.
        self.barge_in = False       # told by the server's `config` event
        self.input_mode = None      # same event: is this page an eye at all?
        self.hold_playback = False  # True = the clip is still playing
        self.queue: list = []       # clips received but not finished
        self.interruptions = 0
        self.dropped = 0

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
                self.queue.append(frame)
                if not frame.header.last:
                    continue  # a stream chunk: playback ends only after `last`
                if self.hold_playback:
                    continue  # still playing: the server waits for playback_done
                self.queue.clear()
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
            elif ev.event == "config":
                self.barge_in = bool(ev.data.get("barge_in"))
                self.input_mode = ev.data.get("input_mode")
            elif ev.event == "barge_in":
                # stopSpeaking(): drop what is playing and everything queued,
                # and deliberately send no playback_done.
                self.interruptions += 1
                self.dropped += len(self.queue)
                self.queue.clear()
            elif ev.event == "capture_request":
                self.captures += 1
                capture_id = ev.data["capture_id"]
                if self.worksheet is None:
                    await self._ws.send(make_event("capture_failed", {"capture_id": capture_id}))
                else:
                    await self._ws.send(
                        encode_image(self.worksheet, ImageHeader(capture_id=capture_id))
                    )

    def _mic_open(self) -> bool:
        if self.state in ("LISTENING", "USER_SPEAKING"):
            return True
        # With barge-in the page keeps the mic open while the tutor talks.
        return self.barge_in and self.state == "AGENT_SPEAKING"

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
    # every state, in order, with no button. The second PROCESSING is the
    # multi-utterance turn contract: after ANY utterance the floor goes back to
    # the TURN (a reaction may be followed by its hint), and only the turn's
    # end — not the utterance's — reopens the mic.
    assert browser.states == [
        "LISTENING",
        "USER_SPEAKING",
        "PROCESSING",
        "AGENT_SPEAKING",
        "PROCESSING",
        "LISTENING",
    ]


async def test_streamed_speech_arrives_in_ordered_chunks(db):
    """One utterance, many frames: seq in order, exactly one `last`, ONE
    playback_done — and the first frame goes out before the tail exists,
    which is the entire reason the streaming path was built."""
    llm = EchoLLMClient({"recognize": [WORKSHEET] * 5})
    deps = Deps(
        settings=SETTINGS,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(audio=MP3 * 4, stream_chunks=3),
        store=SessionStore(),
    )
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            await browser.speak_a_turn()

    frames = browser.audio
    assert len(frames) == 3                              # chunked, not one file
    assert [f.header.seq for f in frames] == [0, 1, 2]   # in order
    assert [f.header.last for f in frames] == [False, False, True]
    assert len({f.header.utterance_id for f in frames}) == 1
    assert b"".join(f.audio for f in frames) == MP3 * 4  # nothing lost or reordered
    assert deps.store.get_history()[0].level == 1        # and the turn completed


async def test_the_ear_hears_korean_and_the_eye_sees_notation(deps):
    """One line, two renderings at the _say boundary: the TTS synthesizes
    "f 프라임 1 … 2 x 제곱", the transcript panel shows f'(1) and 2·x².
    Showing the ear's text on screen was a shipped bug — the panel read
    like a phonetics guide instead of like math."""
    import json as _json

    class RecorderWS:
        def __init__(self):
            self.events = []

        async def send(self, raw):
            if isinstance(raw, str):
                self.events.append(_json.loads(raw))

    deps.speaker.audio = None  # text-only path: no playback future to wait out
    ws = RecorderWS()
    session = BrowserSession(ws, deps, vad=ScriptedVAD())

    await session._say("f'(1) = 2*x**2 - x 부터 봅시다")

    ear = deps.speaker.synthesized[0]
    assert "프라임" in ear and "제곱" in ear and "**" not in ear
    eye = next(e["data"]["text"] for e in ws.events if e["event"] == "tutor_says")
    assert "f'(1)" in eye and "2·x²" in eye
    assert "프라임" not in eye and "제곱" not in eye


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


async def test_the_page_is_told_whether_it_is_an_eye(deps):
    """The page hides its upload box unless this server takes photos from it.
    With a phone on /camera the box would be an invitation to nowhere, so the
    mode travels in the same config event as barge-in."""
    from dataclasses import replace

    vad = ScriptedVAD()
    deps.settings = replace(SETTINGS, input_mode="camera")
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser.wait_until(lambda: browser.input_mode is not None)
            assert browser.input_mode == "camera"

    deps.settings = replace(SETTINGS, input_mode="upload")
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad) as browser:
            await browser.wait_until(lambda: browser.input_mode is not None)
            assert browser.input_mode == "upload"


async def test_without_barge_in_the_mic_is_ignored_while_the_tutor_answers(deps):
    """BARGE_IN=0 restores the old guarantee. An ungated client keeps streaming
    speech; the server must not hear a frame of it."""
    from dataclasses import replace

    deps.settings = replace(SETTINGS, barge_in=False)
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
    assert browser.interruptions == 0


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


class ScriptedTranscriber:
    """STT that says something different each turn."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)

    def transcribe(self, pcm, sample_rate=16000):
        from tutor.speech.stt import Transcript

        return Transcript(text=self.texts.pop(0) if self.texts else "", language="ko")


async def test_spoken_answer_advances_without_a_second_capture(db):
    """The reported flow: ask → answer out loud → next step's L1.

    The answer turn must not re-open the camera or re-run recognition; that
    round trip is what made a reply take ~30s and re-recognition noise is what
    reset the state back to L1 every time.
    """
    from tutor.state.answer import AnswerEvaluator

    llm = EchoLLMClient(
        {
            "recognize": [WORKSHEET] * 5,
            "evaluate": [
                {"verdict": "CORRECT", "feedback": "맞아요!", "misconception": None,
                 "status": "CORRECT"}
            ],
        }
    )
    deps = Deps(
        settings=SETTINGS,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(["힌트 주세요", "양변에서 5를 빼요"]),
        speaker=NullSpeaker(audio=MP3),
        evaluator=AnswerEvaluator(llm, db),
        store=SessionStore(),
    )
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            await browser.speak_a_turn()  # "힌트 주세요"  → L1 at step 1
            first_captures = browser.captures
            await browser.speak_a_turn()  # answers it     → L1 at step 2

    history = deps.store.get_history()
    assert [(h.step, h.level) for h in history] == [(1, 1), (2, 1)]
    assert history[0].effective is True  # the answer proved the hint worked
    assert browser.captures == first_captures == 1  # camera used once, not twice
    assert llm.calls.count("recognize") == 1
    assert llm.calls.count("evaluate") == 1
    # the reaction now arrives as its own utterance, BEFORE the hint finishes
    # generating — that split is the answer turn's perceived-latency win
    assert browser.tutor_said[1] == "맞아요!"
    assert browser.tutor_said[2] != browser.tutor_said[0]  # not the same question again


async def test_filler_utterance_is_answered_and_the_mic_reopens(db):
    """Hands-free: a hesitation gets the gentle prompt, not the pipeline, and
    the turn still completes so the student can keep talking."""
    llm = EchoLLMClient({"recognize": [WORKSHEET] * 5})
    deps = Deps(
        settings=SETTINGS,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=ScriptedTranscriber(["음... 어...", "힌트 주세요"]),
        speaker=NullSpeaker(audio=MP3),
        store=SessionStore(),
    )
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            await browser.speak_a_turn()  # hesitation
            assert browser.tutor_said == ["괜찮아요, 이어서 말해 줄래요?"]
            assert browser.captures == 0  # nothing ran: no camera, no pipeline
            assert deps.store.get_history() == []  # not a hint

            await browser.speak_a_turn()  # then a real request, same session
            assert deps.store.get_history()[0].level == 1

    assert browser.states[-1] == "LISTENING"  # never left deaf


# --- barge-in ----------------------------------------------------------------
#
# The student cuts in mid-sentence. Three things have to happen, and the order
# matters: the audio stops, the queued audio is dropped, and the words that did
# the interrupting become the next turn.


async def _cut_in(deps, *, then=None):
    """Let the tutor start speaking, hold the audio as if it were playing, then
    talk over it."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            browser.hold_playback = True  # the clip keeps playing until we say
            await browser.send_frames([True] * 30 + [False] * SILENCE_FRAMES)
            await browser.wait_until(lambda: browser.state == "AGENT_SPEAKING")
            assert browser.barge_in, "the server never advertised barge-in"

            # Talk over it. More than barge_in_frames, then a pause to endpoint.
            await browser.send_frames(
                [True] * (CFG.barge_in_frames + 25) + [False] * SILENCE_FRAMES
            )
            if then is not None:
                await then(browser)
            return browser


async def test_the_tutor_stops_when_the_student_cuts_in(deps):
    browser = await _cut_in(
        deps, then=lambda b: b.wait_until(lambda: b.interruptions > 0)
    )
    assert browser.interruptions == 1


async def test_the_queued_audio_is_dropped(deps):
    """A turn can be a filler then a hint. Stopping the clip that is playing is
    not enough if the next one is already sitting in the queue."""
    async def wait(browser):
        await browser.wait_until(lambda: browser.interruptions > 0)

    browser = await _cut_in(deps, then=wait)
    assert browser.queue == [], "audio survived the interruption"


async def test_the_interrupting_question_is_handled(deps):
    """The whole point. Cutting in and then being ignored is worse than not
    being able to cut in at all."""
    async def wait(browser):
        await browser.wait_until(lambda: len(browser.heard) >= 2, timeout=20)

    browser = await _cut_in(deps, then=wait)
    assert len(browser.heard) >= 2, browser.heard


async def test_the_tutor_does_not_finish_its_sentence_afterwards(deps):
    """Everything left in the abandoned turn stays silent — no closing praise
    spoken over the student who just took the floor."""
    async def wait(browser):
        await browser.wait_until(lambda: browser.interruptions > 0)
        said_at_cut = len(browser.tutor_said)
        await asyncio.sleep(0.2)
        browser.said_at_cut = said_at_cut

    browser = await _cut_in(deps, then=wait)
    # The next turn may speak; the abandoned one may not add to what it said.
    assert browser.tutor_said[browser.said_at_cut:] in ([], browser.tutor_said[-1:])


async def test_a_cough_does_not_take_the_floor(deps):
    """Short bursts are what echo cancellation leaks. The bar is higher than a
    normal onset for exactly this."""
    vad = ScriptedVAD()
    async with await browser_server(deps, vad) as server:
        port = server.sockets[0].getsockname()[1]
        async with FakeBrowser(f"ws://127.0.0.1:{port}/browser", vad, JPEG) as browser:
            browser.hold_playback = True
            await browser.send_frames([True] * 30 + [False] * SILENCE_FRAMES)
            await browser.wait_until(lambda: browser.state == "AGENT_SPEAKING")
            # enough to start a turn in silence, not enough to take the floor
            await browser.send_frames([True] * CFG.onset_frames)
            await asyncio.sleep(0.15)

    assert browser.interruptions == 0
    assert browser.state == "AGENT_SPEAKING"
