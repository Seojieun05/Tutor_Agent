"""Device simulator: speaks the exact XIAO wire protocol over one WebSocket.

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

from tutor.protocol.events import make_event, parse_event
from tutor.protocol.frames import AudioHeader, ImageHeader, encode_audio, encode_image

CHUNK_MS = 320


class DeviceSim:
    def __init__(self, server: str, images: list[Path], wav: Path | None, use_mic: bool):
        self.server = server
        self.images = images
        self.wav = wav
        self.use_mic = use_mic
        self.image_idx = 0
        self.audio_seq = 0
        self.utterance = 0

    @property
    def current_image(self) -> Path:
        return self.images[min(self.image_idx, len(self.images) - 1)]

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

    async def _reader(self, ws) -> None:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            ev = parse_event(raw)
            if ev.event == "capture_request":
                capture_id = ev.data.get("capture_id", "cap-?")
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
            elif ev.event == "error":
                print(f"  [server error] {ev.data}")

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

    async def _send_audio(self, ws) -> None:
        pcm, rate = self._get_audio()
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


def main() -> None:
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
