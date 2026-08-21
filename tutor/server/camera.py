"""Camera devices as their own connection, shared with the voice session.

The phone is the eyes; the laptop (browser page or local mic) is
the ears and the mouth. They are two WebSockets, so a hint request arriving on
the voice session has to be able to reach the camera sitting on the other one:

    phone ──ws /camera──►  CameraHub  ◄──borrowed by──  Session (/browser, /)

A camera connection is passive. It says hello, then waits: on capture_request
it answers with one IMAGE frame (or capture_failed). It never drives the tutor,
so nothing about the pedagogy changes when hardware appears or disappears.
"""

from __future__ import annotations

import asyncio
import logging

from websockets.exceptions import ConnectionClosed

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import ImageFrame, ProtocolError, decode

log = logging.getLogger(__name__)


# 연결된 카메라 기기 하나. 스스로 튜터를 움직이지 않고, 촬영 요청에만 응답하는 수동적 존재.
class CameraConnection:
    """One connected camera device."""

    # 소켓과 허브, 대기 중인 촬영 요청표를 든다.
    def __init__(self, ws, hub: "CameraHub", device_id: str = "camera"):
        self.ws = ws
        self.hub = hub
        self.device_id = device_id
        self._pending: dict[str, asyncio.Future] = {}
        self._seq = 0

    # 이 카메라에 JPEG 한 장을 요청한다. 실패하거나 응답이 없으면 None.
    async def capture(self, timeout: float) -> bytes | None:
        """Ask this camera for one JPEG. None if it fails or does not answer."""
        self._seq += 1
        capture_id = f"cam-{self._seq}"
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[capture_id] = future
        try:
            await self.ws.send(
                make_event("capture_request", {"capture_id": capture_id, "quality": "high"})
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except (asyncio.TimeoutError, Exception) as e:
            log.warning("camera %s did not deliver %s: %r", self.device_id, capture_id, e)
            return None
        finally:
            self._pending.pop(capture_id, None)

    # 카메라 연결 수신 루프. 허브에 등록했다가 끊기면 해제한다.
    async def run(self) -> None:
        self.hub.register(self)
        try:
            async for raw in self.ws:
                try:
                    await self._on_frame(raw)
                except ProtocolError as e:
                    log.warning("camera protocol error: %s", e)
                except Exception:
                    log.exception("camera frame handling failed; connection continues")
        except (ConnectionClosed, OSError) as e:
            # A phone locking its screen or hopping Wi-Fi is a TCP reset, not
            # a server bug: one quiet line, not an ERROR with a traceback.
            log.info("camera %s connection dropped (%s)", self.device_id,
                     type(e).__name__)
        finally:
            self.hub.unregister(self)
            for future in self._pending.values():
                if not future.done():
                    future.set_result(None)  # never leave a hint request hanging

    # 프레임 처리: IMAGE면 기다리던 촬영 요청에 채워 주고, 실패 이벤트면 None으로 닫는다.
    async def _on_frame(self, raw: bytes | str) -> None:
        if isinstance(raw, str):
            ev = parse_event(raw)
            if ev.event == "hello":
                self.device_id = ev.data.get("device_id", self.device_id)
                log.info("camera connected: %s", self.device_id)
                await self.ws.send(make_event("hello_ack", {"proto": 1, "role": "camera"}))
            elif ev.event == "capture_failed":
                self._resolve(ev.data.get("capture_id", ""), None)
            elif ev.event == "error":
                log.warning("camera %s error: %s", self.device_id, ev.data)
            return
        frame = decode(raw)
        if isinstance(frame, ImageFrame):
            log.info(
                "camera %s sent %s (%d bytes)",
                self.device_id,
                frame.header.capture_id,
                len(frame.jpeg),
            )
            self._resolve(frame.header.capture_id, frame.jpeg)

    # 대기 중인 촬영 요청에 결과를 넣는다.
    def _resolve(self, capture_id: str, jpeg: bytes | None) -> None:
        future = self._pending.get(capture_id)
        if future is not None and not future.done():
            future.set_result(jpeg)
        else:
            log.warning("camera sent an unexpected frame for %r", capture_id)


# 연결된 카메라들의 목록. 음성 세션은 여기서 눈을 빌려 쓴다.
class CameraHub:
    """Every connected camera; the most recent one is asked first."""

    # 빈 목록으로 시작.
    def __init__(self) -> None:
        self._cameras: list[CameraConnection] = []

    # 카메라 등록.
    def register(self, camera: CameraConnection) -> None:
        self._cameras.append(camera)

    # 카메라 해제.
    def unregister(self, camera: CameraConnection) -> None:
        if camera in self._cameras:
            self._cameras.remove(camera)
            log.info("camera disconnected: %s", camera.device_id)

    # 붙어 있는 카메라가 있는지.
    def __bool__(self) -> bool:
        return bool(self._cameras)

    # 연결된 카메라 수.
    @property
    def count(self) -> int:
        return len(self._cameras)

    # 가장 최근에 붙은 카메라에 촬영을 요청한다.
    async def capture(self, timeout: float) -> bytes | None:
        """First camera that answers wins; a dead one falls through to the next."""
        for camera in reversed(list(self._cameras)):
            jpeg = await camera.capture(timeout)
            if jpeg:
                return jpeg
        return None
