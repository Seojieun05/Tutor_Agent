"""Ask the camera for one photo and say exactly what came back.

The tutor answering "카메라에 다시 보여 줄래요?" forever has two completely
different causes that look identical from the outside:

    the frame never arrived    → the VLM was never called
    the frame arrived unread   → the VLM was called and gave up

This settles it in one run. It serves only /camera — no STT, no TTS, no
knowledge DB, no pedagogy — so nothing else can be blamed:

    python -m tutor.scripts.camera_check                 capture + recognize
    python -m tutor.scripts.camera_check --no-recognize  transport only
    python -m tutor.scripts.camera_check --repeat 5      is it flaky or dead?

The saved JPEG is the other half of the answer: open it and you can see
whether the sensor is pointed at the worksheet, in focus, and right way up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from websockets.asyncio.server import serve

from tutor.config import PROJECT_ROOT, load_settings
from tutor.server.camera import CameraConnection, CameraHub

log = logging.getLogger("camera_check")

SAVE_DIR = PROJECT_ROOT / "data" / "captures"


def describe_jpeg(jpeg: bytes) -> str:
    """Dimensions if Pillow is around; never fail the diagnostic over it."""
    try:
        import io

        from PIL import Image

        with Image.open(io.BytesIO(jpeg)) as im:
            return f"{im.width}x{im.height} {im.format}"
    except Exception:  # noqa: BLE001 — a missing Pillow must not hide the byte count
        return "크기 불명 (pillow 없음)"


async def wait_for_camera(hub: CameraHub, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if hub:
            return True
        await asyncio.sleep(0.2)
    return False


async def one_capture(hub: CameraHub, timeout: float, index: int, budget: float) -> bytes | None:
    print(f"\n[{index}] capture_request 전송 (최대 {timeout:.0f}초 대기)...", flush=True)
    started = time.monotonic()
    jpeg = await hub.capture(timeout)
    elapsed = (time.monotonic() - started) * 1000

    if not jpeg:
        print(f"[{index}] ✗ {elapsed:.0f} ms 안에 사진이 오지 않았습니다 "
              f"(실제 수업의 제한은 {budget:.0f}초).")
        print("      → 보드의 시리얼 모니터를 보세요:")
        print("        '전송' 이 찍혔다면 Wi-Fi가 5초 안에 다 못 보낸 것 (전송 시간 확인),")
        print("        '촬영 실패' 라면 센서/리본 케이블,")
        print("        아무것도 없다면 capture_request가 도착하지 않은 것입니다.")
        return None

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    path = SAVE_DIR / f"camera-{int(time.time())}-{index}.jpg"
    path.write_bytes(jpeg)
    rate = len(jpeg) / 1024 / max(elapsed / 1000, 1e-6)
    print(f"[{index}] ✓ {len(jpeg):,} bytes · {describe_jpeg(jpeg)} · "
          f"{elapsed:.0f} ms ({rate:.0f} KB/s)")
    print(f"      저장: {path}")
    if elapsed > budget * 1000:
        print(f"      ⚠ 서버가 기다리는 시간({budget:.0f}초)보다 오래 걸렸습니다 — 실제 수업에서는 "
              "매번 실패합니다. CAPTURE_TIMEOUT_S를 올리거나 프레임 크기를 줄이세요.")
    return jpeg


def recognize(jpeg: bytes) -> None:
    """The real Recognizer, on the real photo. No DB: `recognize` has no tools."""
    settings = load_settings()
    if settings.echo_mode:
        print("      (XAI_API_KEY 없음 → 인식 건너뜀)")
        return
    from tutor.llm.client import GrokClient
    from tutor.vision.recognizer import Recognizer

    print("      VLM에 보내는 중...", flush=True)
    started = time.monotonic()
    try:
        # registry=None on purpose: complete_json never touches it, and building
        # a real one would load the 62 MB knowledge DB and the embedding index
        # into a diagnostic that has nothing to do with either.
        rec = Recognizer(GrokClient(settings, None)).recognize(jpeg)
    except Exception as exc:  # noqa: BLE001 — the failure IS the result here
        print(f"      ✗ 인식 호출 실패 ({time.monotonic() - started:.1f}초): "
              f"{type(exc).__name__}: {exc}")
        return
    elapsed = time.monotonic() - started
    print(f"      ✓ {settings.chat_model} 응답 ({elapsed:.1f}초):")
    print(json.dumps(rec.model_dump(), ensure_ascii=False, indent=2))
    verdict = (
        "문제도 풀이도 못 읽었습니다 — 카메라가 워크시트를 향하고 있는지 확인하세요"
        if not rec.problem_text and not rec.student_work
        else f"confidence {rec.confidence} < {settings.recog_conf_threshold} → "
             "UNCERTAIN → 다시 보여 달라는 말이 나옵니다"
        if rec.confidence < settings.recog_conf_threshold
        else "인식은 정상입니다 — 원인은 인식 뒤쪽에 있습니다"
    )
    print(f"      판정: {verdict}")


async def amain(args) -> int:
    settings = load_settings()
    hub = CameraHub()

    async def handler(ws):
        path = ws.request.path.split("?", 1)[0]
        if path.rstrip("/") != "/camera":
            await ws.close(1008, "this diagnostic serves /camera only")
            return
        await CameraConnection(ws, hub).run()

    from tutor.scripts.live_demo import lan_ip

    async with serve(handler, settings.ws_host, args.port, max_size=16 * 1024 * 1024):
        print(f"카메라 진단 서버: ws://{lan_ip()}:{args.port}/camera")
        print("보드의 SERVER_HOST/SERVER_PORT가 위 주소와 같은지 확인하세요.")
        print(f"\n카메라 연결 대기 중... (최대 {args.wait:.0f}초)", flush=True)
        if not await wait_for_camera(hub, args.wait):
            print("✗ 카메라가 연결되지 않았습니다.")
            print("  → 보드 시리얼에 '[ws] 연결됨'이 찍히는지, SERVER_HOST가 이 노트북의")
            print("    LAN IP인지, 둘이 같은 Wi-Fi(2.4GHz)인지 확인하세요.")
            return 1
        print(f"✓ 카메라 {hub.count}대 연결됨")

        ok = 0
        for i in range(1, args.repeat + 1):
            jpeg = await one_capture(hub, args.timeout, i, settings.capture_timeout_s)
            if jpeg:
                ok += 1
                if args.recognize:
                    await asyncio.to_thread(recognize, jpeg)
            if i < args.repeat:
                await asyncio.sleep(args.interval)

        print(f"\n결과: {ok}/{args.repeat} 성공")
        if ok == 0:
            print("→ 사진이 한 장도 오지 않았습니다. 인식 이전에 전송 문제입니다.")
        elif ok < args.repeat:
            print("→ 간헐적입니다. Wi-Fi 신호나 타임아웃 여유를 의심하세요.")
        return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="카메라 사진 한 장 받아보기")
    parser.add_argument("--port", type=int, default=load_settings().ws_port)
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="한 장을 기다리는 시간(초). 서버 기본값은 5초입니다")
    parser.add_argument("--wait", type=float, default=60.0, help="연결 대기 시간(초)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--no-recognize", dest="recognize", action="store_false",
                        help="전송만 확인하고 VLM은 부르지 않습니다")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    try:
        raise SystemExit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        print("\nbye")
        raise SystemExit(130) from None


if __name__ == "__main__":
    sys.exit(main())
