"""Device simulator: speaks the exact device wire protocol over one WebSocket.

Usage:
    python -m simulator.device_sim --server ws://localhost:8765 \
        --images simulator/assets/lin_001_wrong_sign.jpg simulator/assets/lin_001_step1_ok.jpg \
        [--wav simulator/assets/hint.wav | --mic]

Keys: h = hint request, n = next image ("the student wrote more"),
      a = send audio utterance, q = quit.
On capture_request the CURRENT image is sent — the --images list is an
ordered script of the student's evolving worksheet.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import wave
from pathlib import Path

from websockets.asyncio.client import connect

from tutor.console import soften_stdout
from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioHeader, ImageHeader, encode_audio, encode_image

# 오디오를 이 길이로 잘라 보낸다.
CHUNK_MS = 320


# 하드웨어 없이 디바이스 프로토콜을 그대로 흉내 내는 시뮬레이터.
class DeviceSim:
    # 서버 주소, 순서대로 보여 줄 문제지 사진들, 보낼 음성 파일을 받는다.
    def __init__(self, server: str, images: list[Path], wav: Path | None, use_mic: bool):
        self.server = server
        self.images = images
        self.wav = wav
        self.use_mic = use_mic
        self.image_idx = 0
        self.audio_seq = 0
        self.utterance = 0

    # 지금 촬영 요청에 답할 사진(= 학생 풀이의 현재 단계).
    @property
    def current_image(self) -> Path:
        return self.images[min(self.image_idx, len(self.images) - 1)]

    # 접속해서 hello를 보내고, 수신 루프와 키 입력 루프를 함께 돌린다.
    async def run(self) -> None:
        async with connect(self.server, max_size=16 * 1024 * 1024) as ws:
            await ws.send(
                make_event(
                    "hello",
                    {"device_id": "sim-1", "fw_version": "sim", "proto": 1,
                     "caps": ["camera", "mic"]},
                )
            )
            reader = asyncio.create_task(self._reader(ws))
            try:
                await self._repl(ws)
            finally:
                reader.cancel()

    # 서버 → 디바이스 방향 처리: 촬영 요청이 오면 현재 사진을 IMAGE 프레임으로 보낸다.
    async def _reader(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            ev = parse_event(raw)
            if ev.event == "capture_request":
                capture_id = ev.data.get("capture_id", "cap-?")
                if not self.images:  # no camera on this device variant
                    await ws.send(make_event("capture_failed", {"capture_id": capture_id}))
                    print(f"  -> no image available for {capture_id}")
                    continue
                jpeg = self.current_image.read_bytes()
                await ws.send(
                    encode_image(jpeg, ImageHeader(capture_id=capture_id, format="jpeg"))
                )
                print(f"  -> sent {self.current_image.name} for {capture_id}")
            elif ev.event == "speech_state":
                state = ev.data.get("state")
                print(f"  [speaker] {'🔊 speaking...' if state == 'speaking' else 'idle'}")
            elif ev.event == "hint_issued":
                print(f"  [hint] level L{ev.data.get('level')} action {ev.data.get('action')}")
            elif ev.event == "hello_ack":
                print("  connected (hello_ack)")
            elif ev.event == "transcript":
                print(f"  [heard] {ev.data.get('text')!r}")
            elif ev.event == "error":
                print(f"  [server error] {ev.data}")
            self.on_server_event(ev)

    # 서버 이벤트를 화면에 표시(하위 클래스가 확장한다).
    def on_server_event(self, ev) -> None:
        """Hook for device variants that react to server events (see
        simulator/voice_device.py, which drives turn taking from them)."""

    # 키 입력: h 힌트 요청 · n 다음 사진 · a 음성 전송 · q 종료.
    async def _repl(self, ws) -> None:
        loop = asyncio.get_running_loop()
        print("keys: [h]int  [n]ext image  [a]udio  [q]uit")
        while True:
            cmd = (await loop.run_in_executor(None, input, "> ")).strip().lower()
            if cmd == "h":
                await ws.send(make_event("hint_request", {"source": "button"}))
            elif cmd == "n":
                if self.image_idx < len(self.images) - 1:
                    self.image_idx += 1
                print(f"  current image: {self.current_image.name}")
            elif cmd == "a":
                await self._send_audio(ws)
            elif cmd == "q":
                return

    # PCM을 조각내어 AUDIO 프레임으로 보낸다(마지막 조각에 last 표시).
    async def _send_audio(self, ws, pcm: bytes | None = None, rate: int | None = None) -> None:
        if pcm is None:
            pcm, rate = self._get_audio()
        rate = rate or 16000
        if not pcm:
            print("  no audio source (--wav or --mic)")
            return
        self.utterance += 1
        stream_id = f"utt-{self.utterance}"
        chunk_bytes = int(rate * CHUNK_MS / 1000) * 2
        chunks = [pcm[i : i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)] or [b""]
        for i, chunk in enumerate(chunks):
            self.audio_seq += 1
            await ws.send(
                encode_audio(
                    chunk,
                    AudioHeader(
                        stream_id=stream_id,
                        sample_rate=rate,
                        seq=self.audio_seq,
                        last=(i == len(chunks) - 1),
                    ),
                )
            )
        print(f"  -> sent {len(pcm)} bytes of PCM as {len(chunks)} chunks")

    # WAV 파일을 읽거나 마이크로 몇 초 녹음해 PCM을 만든다.
    def _get_audio(self) -> tuple[bytes, int]:
        if self.wav is not None:
            with wave.open(str(self.wav), "rb") as w:
                if w.getsampwidth() != 2 or w.getnchannels() != 1:
                    print("  wav must be 16-bit mono")
                    return b"", 16000
                return w.readframes(w.getnframes()), w.getframerate()
        if self.use_mic:
            try:
                import sounddevice as sd
            except ImportError:
                print("  pip install sounddevice for --mic")
                return b"", 16000
            rate = 16000
            print("  recording 3s ...")
            audio = sd.rec(int(3 * rate), samplerate=rate, channels=1, dtype="int16")
            sd.wait()
            return audio.tobytes(), rate
        return b"", 16000


# 커맨드라인 진입점.
def main() -> None:
    soften_stdout()  # this docstring becomes --help, and cp949 cannot hold it
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://localhost:8765")
    parser.add_argument("--images", nargs="+", required=True, type=Path)
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--mic", action="store_true")
    args = parser.parse_args()
    for p in args.images:
        if not p.exists():
            sys.exit(f"image not found: {p} (run: python -m tutor.scripts.gen_assets)")
    try:
        asyncio.run(DeviceSim(args.server, args.images, args.wav, args.mic).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
