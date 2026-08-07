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

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import ImageFrame, ProtocolError, decode

log = logging.getLogger(__name__)


class CameraConnection:
    """One connected camera device."""

    def __init__(self, ws, hub: "CameraHub", device_id: str = "camera"):
        self.ws = ws
        self.hub = hub
        self.device_id = device_id
        self._pending: dict[str, asyncio.Future] = {}
        self._seq = 0

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
        finally:
            self.hub.unregister(self)
            for future in self._pending.values():
                if not future.done():
                    future.set_result(None)  # never leave a hint request hanging

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

    def _resolve(self, capture_id: str, jpeg: bytes | None) -> None:
        future = self._pending.get(capture_id)
        if future is not None and not future.done():
            future.set_result(jpeg)
        else:
            log.warning("camera sent an unexpected frame for %r", capture_id)


class CameraHub:
    """Every connected camera; the most recent one is asked first."""

    def __init__(self) -> None:
        self._cameras: list[CameraConnection] = []

    def register(self, camera: CameraConnection) -> None:
        self._cameras.append(camera)

    def unregister(self, camera: CameraConnection) -> None:
        if camera in self._cameras:
            self._cameras.remove(camera)
            log.info("camera disconnected: %s", camera.device_id)

    def __bool__(self) -> bool:
        return bool(self._cameras)

    @property
    def count(self) -> int:
        return len(self._cameras)

    async def capture(self, timeout: float) -> bytes | None:
        """First camera that answers wins; a dead one falls through to the next."""
        for camera in reversed(list(self._cameras)):
            jpeg = await camera.capture(timeout)
            if jpeg:
                return jpeg
        return None
