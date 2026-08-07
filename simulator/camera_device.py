"""Stand-in for the phone camera: same wire protocol, no phone.

    python -m simulator.camera_device --server ws://localhost:8765 \
        --images simulator/assets/lin_001_wrong_sign.jpg

Connects to /camera and answers every capture_request with the current image,
exactly as tutor/web/phone.html does. Use it to check the pairing (voice on the
eyes elsewhere) before flashing anything, and to debug the server side without
a board on the desk. Press Enter to advance to the next image.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from websockets.asyncio.client import connect

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import ImageHeader, encode_image


class CameraDevice:
    def __init__(self, server: str, images: list[Path], device_id: str = "sim-camera"):
        self.server = server.rstrip("/") + "/camera"
        self.images = images
        self.device_id = device_id
        self.index = 0

    @property
    def current(self) -> Path:
        return self.images[min(self.index, len(self.images) - 1)]

    async def run(self) -> None:
        async with connect(self.server, max_size=16 * 1024 * 1024) as ws:
            await ws.send(
                make_event(
                    "hello",
                    {"device_id": self.device_id, "fw_version": "sim", "proto": 1,
                     "caps": ["camera"]},
                )
            )
            print(f"연결됨: {self.server}  (현재 이미지: {self.current.name})")
            reader = asyncio.create_task(self._reader(ws))
            try:
                await self._advance_on_enter()
            finally:
                reader.cancel()

    async def _reader(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            ev = parse_event(raw)
            if ev.event == "capture_request":
                capture_id = ev.data.get("capture_id", "cam-?")
                jpeg = self.current.read_bytes()
                await ws.send(
                    encode_image(jpeg, ImageHeader(capture_id=capture_id, format="jpeg"))
                )
                print(f"  → {self.current.name} 전송 ({len(jpeg) // 1024} KB) for {capture_id}")
            elif ev.event == "hello_ack":
                print("  hello_ack — 서버가 카메라로 등록했습니다")
            elif ev.event == "error":
                print(f"  [server error] {ev.data}")

    async def _advance_on_enter(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await loop.run_in_executor(None, input, "다음 이미지로 넘기려면 Enter (Ctrl+C 종료)\n")
            if self.index < len(self.images) - 1:
                self.index += 1
            print(f"  현재 이미지: {self.current.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://localhost:8765")
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--device-id", default="sim-camera")
    args = parser.parse_args()
    for path in args.images:
        if not path.exists():
            sys.exit(f"이미지 없음: {path} (python -m tutor.scripts.gen_assets 로 생성)")
    try:
        asyncio.run(CameraDevice(args.server, args.images, args.device_id).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
