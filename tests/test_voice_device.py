"""Hands-free device against a real in-process server, no audio hardware.

The mic is replaced by a scripted VAD and a fake clock; everything else — the
wire protocol, Session, STT (echo), the hint pipeline, the speech_state events
that mute the VAD — is the production path.
"""

import asyncio
import math

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from simulator.voice_device import VoiceDevice
from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.events import make_event, parse_event
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.speech.turn import TurnConfig, TurnState
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

WORKSHEET = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "student_work": ["3*x = 20 + 5"],
    "confidence": 0.95,
}

CFG = TurnConfig()
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


class ScriptedVoiceDevice(VoiceDevice):
    """Feeds scripted frames instead of a microphone, on a fake clock."""

    def __init__(self, server, images, turns: int, vad: ScriptedVAD):
        super().__init__(server, images, CFG, vad=vad)
        self.vad = vad
        self.turns = turns
        self.states: list[TurnState] = []
        self.clock_ms = 0.0
        self.speech_heard_by_vad_while_muted = 0

    def now_ms(self) -> float:
        return self.clock_ms

    async def _tick(self, ws, is_speech: bool) -> None:
        """One 32 ms frame; the clock only moves here, as with a real mic."""
        self.clock_ms += self.config.frame_ms
        muted = not self.taker.listening
        seen_before = self.vad.seen
        self.vad.script.append(is_speech)
        await self.on_frame(ws, bytes(self.config.frame_bytes))
        if muted:
            self.vad.script.clear()  # unconsumed: the frame never reached the model
            # (a frame that expires the tail guard is listening again by now —
            # it is allowed to reach the VAD)
            if not self.taker.listening:
                self.speech_heard_by_vad_while_muted += self.vad.seen - seen_before
        self.states.append(self.taker.state)
        await asyncio.sleep(0)  # let the reader task drain server events

    async def _repl(self, ws) -> None:
        for _ in range(self.turns):
            for flag in [True] * 30 + [False] * SILENCE_FRAMES:
                await self._tick(ws, flag)
            # keep the mic running while the tutor answers — including speech,
            # which stands in for the tutor's own voice reaching the mic
            for _ in range(400):
                if self.taker.state is TurnState.LISTENING:
                    break
                await self._tick(ws, True)
                await asyncio.sleep(0.002)
            else:
                pytest.fail(f"never returned to LISTENING (stuck in {self.taker.state})")


@pytest.fixture
def deps(db):
    llm = EchoLLMClient({"recognize": [WORKSHEET] * 4})
    return Deps(
        settings=Settings(capture_timeout_s=2.0),
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(),
        store=SessionStore(),
    )


@pytest.fixture
def worksheet(tmp_path):
    path = tmp_path / "worksheet.jpg"
    path.write_bytes(b"\xff\xd8" + b"\x00" * 1024)
    return path


async def run_device(deps, images, turns) -> ScriptedVoiceDevice:
    async def handler(ws):
        await Session(ws, deps).run()

    async with serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        port = server.sockets[0].getsockname()[1]
        device = ScriptedVoiceDevice(
            f"ws://127.0.0.1:{port}", images, turns, ScriptedVAD()
        )
        await asyncio.wait_for(device.run(), timeout=30)
        return device


async def test_one_spoken_turn_reaches_the_pipeline(deps, worksheet):
    device = await run_device(deps, [worksheet], turns=1)

    # the utterance was transcribed, diagnosed, and answered out loud
    assert len(deps.speaker.spoken) == 1
    assert deps.store.get_history()[0].level == 1
    # ...with no button: every state was reached, in order
    assert [s for i, s in enumerate(device.states) if i == 0 or s is not device.states[i - 1]] == [
        TurnState.LISTENING,  # onset debounce: not speaking yet
        TurnState.USER_SPEAKING,
        TurnState.PROCESSING,
        TurnState.AGENT_SPEAKING,
        TurnState.LISTENING,
    ]


async def test_agent_never_hears_itself(deps, worksheet):
    device = await run_device(deps, [worksheet], turns=1)
    # frames kept arriving (and were "speech") for the whole response, yet the
    # VAD was never consulted while the tutor was thinking or speaking
    assert device.speech_heard_by_vad_while_muted == 0
    assert device.states.count(TurnState.AGENT_SPEAKING) > 0
    assert len(deps.speaker.spoken) == 1  # exactly one turn, no self-triggered second


async def test_three_turns_without_a_button(deps, worksheet):
    device = await run_device(deps, [worksheet], turns=3)
    assert len(deps.speaker.spoken) == 3
    # escalation across turns proves the same session handled all three
    assert [h.level for h in deps.store.get_history()] == [1, 2, 3]
    assert device.states[-1] is TurnState.LISTENING


async def test_device_without_camera_still_completes_a_turn(deps):
    """No --images: the device answers capture_failed and stays conversational."""
    device = await run_device(deps, [], turns=1)
    assert len(deps.speaker.spoken) == 1
    assert device.states[-1] is TurnState.LISTENING


async def test_utterance_only_is_sent_not_the_whole_stream(deps, worksheet):
    """The server must receive one bounded utterance per turn, not open audio."""
    received: list[int] = []

    async def handler(ws):
        session = Session(ws, deps)
        original = session._handle_utterance

        async def spy(pcm, rate):
            received.append(len(pcm))
            await original(pcm, rate)

        session._handle_utterance = spy
        await session.run()

    async with serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024) as server:
        port = server.sockets[0].getsockname()[1]
        device = ScriptedVoiceDevice(
            f"ws://127.0.0.1:{port}", [worksheet], 1, ScriptedVAD()
        )
        await asyncio.wait_for(device.run(), timeout=30)

    assert len(received) == 1
    spoken_ms = 30 * CFG.frame_ms
    got_ms = received[0] / 2 / CFG.sample_rate * 1000
    # speech + prefix padding + the trailing silence window, and nothing more
    assert spoken_ms + CFG.prefix_ms <= got_ms <= spoken_ms + CFG.prefix_ms + CFG.silence_ms + 2 * CFG.frame_ms
    assert device.states[-1] is TurnState.LISTENING
