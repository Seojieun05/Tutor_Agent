"""A camera on its own socket, borrowed by the voice session.

The XIAO is the eyes and the laptop is the ears: two WebSockets, one hint
request. These tests drive the real server code with the camera simulator's
protocol, so they cover everything except the firmware itself.
"""

import asyncio

import pytest
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from tutor.config import Settings
from tutor.hints.generator import HintGenerator
from tutor.knowledge.matching import Matcher
from tutor.llm.echo import EchoLLMClient
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import ImageHeader, encode_image
from tutor.server.camera import CameraConnection, CameraHub
from tutor.server.session import Deps, Session
from tutor.solver.grok_solver import GrokSolver
from tutor.speech.stt import EchoTranscriber
from tutor.speech.tts import NullSpeaker
from tutor.state.estimator import StudentStateEstimator
from tutor.store.session_store import SessionStore
from tutor.vision.recognizer import Recognizer

WORKSHEET = {
    "problem_text": "다음 일차방정식을 푸시오: 3x + 5 = 20",
    "equations": ["3*x + 5 = 20"],
    "student_work": ["3*x = 20 + 5"],
    "confidence": 0.95,
}
XIAO_JPEG = b"\xff\xd8" + b"xiao" * 2048     # what the board would send
LOCAL_JPEG = b"\xff\xd8" + b"local" * 2048   # what the browser attached


@pytest.fixture
def deps(db):
    llm = EchoLLMClient({"recognize": [WORKSHEET] * 5})
    # input_mode="camera": borrowing a XIAO is opt-in now that the default
    # worksheet source is the browser's file picker. Wired through to the
    # generator exactly as tutor/server/app.py does it.
    settings = Settings(capture_timeout_s=2.0, input_mode="camera")
    return Deps(
        settings=settings,
        recognizer=Recognizer(llm),
        matcher=Matcher(db),
        solver=GrokSolver(llm, db),
        estimator=StudentStateEstimator(llm, db),
        hint_gen=HintGenerator(llm, db, settings.input_mode),
        transcriber=EchoTranscriber(),
        speaker=NullSpeaker(),
        cameras=CameraHub(),
        store=SessionStore(),
    )


async def tutor_server(deps):
    """One port, two roles — exactly as tutor/server/app.py routes them."""

    async def handler(ws):
        if ws.request.path.rstrip("/") == "/camera":
            await CameraConnection(ws, deps.cameras).run()
        else:
            await Session(ws, deps).run()

    return serve(handler, "127.0.0.1", 0, max_size=16 * 1024 * 1024)


class FakeCamera:
    """The firmware's half of the conversation."""

    def __init__(self, url: str, jpeg: bytes | None = XIAO_JPEG):
        self.url = url + "/camera"
        self.jpeg = jpeg
        self.requests = 0
        self.ready = asyncio.Event()

    async def __aenter__(self):
        self._ws = await connect(self.url, max_size=16 * 1024 * 1024).__aenter__()
        self._task = asyncio.create_task(self._run())
        await self._ws.send(
            make_event("hello", {"device_id": "xiao-1", "caps": ["camera"]})
        )
        await asyncio.wait_for(self.ready.wait(), timeout=5)
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        await self._ws.close()

    async def _run(self):
        async for raw in self._ws:
            if not isinstance(raw, str):
                continue
            ev = parse_event(raw)
            if ev.event == "hello_ack":
                self.ready.set()
            elif ev.event == "capture_request":
                self.requests += 1
                capture_id = ev.data["capture_id"]
                if self.jpeg is None:
                    await self._ws.send(make_event("capture_failed", {"capture_id": capture_id}))
                else:
                    await self._ws.send(
                        encode_image(self.jpeg, ImageHeader(capture_id=capture_id))
                    )


class VoiceClient:
    """A laptop-side session with no camera of its own, unless given one."""

    def __init__(self, url: str, worksheet: bytes | None = None):
        self.url = url
        self.worksheet = worksheet
        self.hints: list[dict] = []
        self.issued = asyncio.Event()

    async def __aenter__(self):
        self._ws = await connect(self.url, max_size=16 * 1024 * 1024).__aenter__()
        self._task = asyncio.create_task(self._run())
        await self._ws.send(make_event("hello", {"device_id": "laptop"}))
        return self

    async def __aexit__(self, *exc):
        self._task.cancel()
        await self._ws.close()

    async def _run(self):
        async for raw in self._ws:
            if not isinstance(raw, str):
                continue
            ev = parse_event(raw)
            if ev.event == "capture_request":
                capture_id = ev.data["capture_id"]
                if self.worksheet is None:
                    await self._ws.send(make_event("capture_failed", {"capture_id": capture_id}))
                else:
                    await self._ws.send(
                        encode_image(self.worksheet, ImageHeader(capture_id=capture_id))
                    )
            elif ev.event == "hint_issued":
                self.hints.append(ev.data)
                self.issued.set()

    async def ask_for_a_hint(self):
        self.issued.clear()
        await self._ws.send(make_event("hint_request", {}))
        await asyncio.wait_for(self.issued.wait(), timeout=15)


async def test_a_cameraless_session_borrows_the_xiao(deps):
    """The point of the whole thing: talk to the laptop, see through the board."""
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url) as camera, VoiceClient(url) as laptop:
            assert deps.cameras.count == 1
            await laptop.ask_for_a_hint()

    assert camera.requests == 1              # the board was asked
    assert laptop.hints[0]["level"] == 1     # and a real hint came back
    assert deps.store.get_history()[0].level == 1


async def test_upload_mode_never_borrows_a_connected_board(deps):
    """INPUT_MODE=upload means the picture is the one the student chose.

    A XIAO left plugged in from an earlier experiment must not quietly become
    the tutor's eyes: it would answer with a photo of whatever the board happens
    to be pointed at, and the student would be told about a different problem
    from the one on their screen.
    """
    deps.settings.input_mode = "upload"
    deps.hint_gen = HintGenerator(deps.hint_gen.llm, deps.hint_gen.db, "upload")
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url) as camera, VoiceClient(url) as laptop:
            assert deps.cameras.count == 1   # connected, and still not used
            await laptop.ask_for_a_hint()

    assert camera.requests == 0
    assert laptop.hints[0]["action"] == "ASK_RECAPTURE"
    assert "사진" in deps.speaker.spoken[0], deps.speaker.spoken


async def test_the_sessions_own_camera_wins(deps):
    """An attached worksheet is a deliberate choice: do not override it."""
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url) as camera, VoiceClient(url, LOCAL_JPEG) as laptop:
            await laptop.ask_for_a_hint()

    assert camera.requests == 0              # never bothered
    assert laptop.hints[0]["level"] == 1


async def test_a_camera_that_cannot_shoot_does_not_hang_the_hint(deps):
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url, jpeg=None) as camera, VoiceClient(url) as laptop:
            await laptop.ask_for_a_hint()

    assert camera.requests == 1
    # nothing to look at → the tutor asks to be shown the worksheet, it does not stall
    assert laptop.hints[0]["action"] == "ASK_RECAPTURE"


async def test_disconnecting_the_camera_leaves_no_ghost(deps):
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url):
            assert deps.cameras.count == 1
        await asyncio.sleep(0.1)
        assert deps.cameras.count == 0

        async with VoiceClient(url) as laptop:   # still usable with no camera at all
            await laptop.ask_for_a_hint()
        assert laptop.hints[0]["action"] == "ASK_RECAPTURE"


async def test_the_newest_camera_is_asked_first(deps):
    async with await tutor_server(deps) as server:
        url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        async with FakeCamera(url) as old, FakeCamera(url) as new, VoiceClient(url) as laptop:
            await laptop.ask_for_a_hint()

    assert (new.requests, old.requests) == (1, 0)  # the one just plugged in


class TestFirmwareWireContract:
    """The board builds frames by hand in C; the server must keep accepting them.

    These bytes are the layout tutor_xiao_camera.ino::sendImage() writes:
        [0x01][uint32 BE header length][JSON header][JPEG]
    Verified once by compiling those exact C lines and decoding the output with
    tutor.protocol.frames; kept here so a change to the framing breaks loudly
    instead of bricking a flashed board.
    """

    @staticmethod
    def firmware_frame(capture_id: str, jpeg: bytes, width=1600, height=1200) -> bytes:
        header = (
            f'{{"capture_id":"{capture_id}","format":"jpeg",'
            f'"width":{width},"height":{height},"seq":0}}'
        ).encode()
        n = len(header)
        return (
            bytes([0x01, (n >> 24) & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF])
            + header
            + jpeg
        )

    def test_the_server_decodes_what_the_board_writes(self):
        from tutor.protocol.frames import ImageFrame, decode

        frame = decode(self.firmware_frame("cam-7", XIAO_JPEG))
        assert isinstance(frame, ImageFrame)
        assert frame.header.capture_id == "cam-7"
        assert (frame.header.width, frame.header.height) == (1600, 1200)
        assert frame.jpeg == XIAO_JPEG

    async def test_a_board_shaped_frame_serves_a_real_hint(self, deps):
        """End to end with bytes laid out exactly as the firmware lays them."""

        class RawCamera(FakeCamera):
            async def _run(self):
                async for raw in self._ws:
                    if not isinstance(raw, str):
                        continue
                    ev = parse_event(raw)
                    if ev.event == "hello_ack":
                        self.ready.set()
                    elif ev.event == "capture_request":
                        self.requests += 1
                        await self._ws.send(
                            TestFirmwareWireContract.firmware_frame(
                                ev.data["capture_id"], XIAO_JPEG
                            )
                        )

        async with await tutor_server(deps) as server:
            url = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            async with RawCamera(url) as camera, VoiceClient(url) as laptop:
                await laptop.ask_for_a_hint()

        assert camera.requests == 1
        assert laptop.hints[0]["level"] == 1
