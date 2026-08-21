"""Laptop microphone as a continuous 16 kHz mono PCM frame source.

PortAudio delivers frames on its own thread; they are handed to the event loop
through a bounded queue. Bounded on purpose: if the consumer ever stalls, the
right failure is to drop the oldest audio and keep the conversation live rather
than to fall further and further behind.

``sounddevice`` is imported lazily so nothing else in the project needs the
``voice`` extra.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

log = logging.getLogger(__name__)

# sounddevice가 없을 때 보여 줄 안내.
INSTALL_HINT = (
    'microphone capture needs sounddevice — pip install -e ".[voice]" '
    "(and the PortAudio system library, e.g. apt install libportaudio2)"
)


# 로컬 마이크에서 고정 크기 PCM 프레임을 뽑아 주는 스트림.
class MicStream:
    """Async iterator over fixed-size int16 mono PCM frames."""

    # 샘플레이트·프레임 크기·장치를 잡고 입력 스트림을 연다.
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        device: int | str | None = None,
        max_queue: int = 64,
    ):
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.device = device
        self.max_queue = max_queue
        self.dropped = 0
        self._queue: asyncio.Queue[bytes] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream = None

    # 프레임 하나의 바이트 수.
    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * 2

    async def __aenter__(self) -> "MicStream":
        try:
            import sounddevice as sd
        except (ImportError, OSError) as e:  # OSError: PortAudio not installed
            raise RuntimeError(INSTALL_HINT) from e

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=self.max_queue)
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            device=self.device,
            dtype="int16",
            channels=1,
            callback=self._on_audio,
        )
        self._stream.start()
        log.info(
            "microphone open: %d Hz mono, %d-sample frames (%s)",
            self.sample_rate,
            self.frame_samples,
            sd.query_devices(self._stream.device)["name"],
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self.dropped:
            log.warning("dropped %d mic frame(s): consumer could not keep up", self.dropped)

    # 사운드 장치 콜백: 들어온 오디오를 버퍼에 넣는다.
    def _on_audio(self, indata, frames, time_info, status) -> None:
        # PortAudio thread: hand off, never block.
        if status:
            log.debug("mic status: %s", status)
        data = bytes(indata)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._push, data)

    # 프레임 크기에 맞춰 잘라 큐에 넣는다.
    def _push(self, data: bytes) -> None:
        assert self._queue is not None
        try:
            self._queue.put_nowait(data)
        except asyncio.QueueFull:
            self.dropped += 1
            try:  # keep the newest audio: drop the oldest frame instead
                self._queue.get_nowait()
                self._queue.put_nowait(data)
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                pass

    async def frames(self) -> AsyncIterator[bytes]:
        if self._queue is None:
            raise RuntimeError("MicStream must be used as an async context manager")
        while True:
            yield await self._queue.get()
