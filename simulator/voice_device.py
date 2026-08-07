"""Laptop mic + speaker as the device: hands-free, multi-turn, no buttons.

Usage:
    python -m simulator.voice_device --server ws://localhost:8765 \
        [--images simulator/assets/lin_001_wrong_sign.jpg ...] [--input-device 3]

Speak; stop speaking; the tutor answers; it listens again. Nothing else.

    mic 16 kHz mono PCM → Silero VAD turn detection → the same AUDIO frames the
    device sends → server STT → tutor pipeline → server TTS on the laptop speaker

Only the endpointed utterance is sent, so STT and Grok never see room noise.
The four states (LISTENING / USER_SPEAKING / PROCESSING / AGENT_SPEAKING) live
in tutor/speech/turn.py; this module only wires them to the wire protocol:
``speech_state`` drives AGENT_SPEAKING, and the VAD stays muted for that whole
span so the tutor cannot transcribe its own voice.

Camera frames are optional here — ``--images`` replays a worksheet exactly like
device_sim; without it the device answers capture_failed and the server falls
back to its recognition-failed hint path.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from simulator.device_sim import DeviceSim
from tutor.config import load_settings
from tutor.speech.mic import MicStream
from tutor.speech.turn import TurnConfig, TurnDetector, TurnState, TurnTaker

log = logging.getLogger("voice_device")

STATE_ICON = {
    TurnState.LISTENING: "🎙  listening",
    TurnState.USER_SPEAKING: "🗣  you are speaking",
    TurnState.PROCESSING: "⏳ thinking",
    TurnState.AGENT_SPEAKING: "🔊 tutor speaking (mic muted)",
}


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class VoiceDevice(DeviceSim):
    def __init__(
        self,
        server: str,
        images: list[Path],
        config: TurnConfig,
        input_device: int | str | None = None,
        processing_timeout_s: float = 90.0,
        vad=None,
    ):
        super().__init__(server, images, wav=None, use_mic=False)
        self.config = config
        self.input_device = input_device
        self.processing_timeout_ms = processing_timeout_s * 1000
        # vad=None builds the real Silero model; tests inject a scripted stub
        self.taker = TurnTaker(TurnDetector(vad=vad, config=config), on_change=self._show_state)
        self._processing_since = 0.0

    def now_ms(self) -> float:
        """Monotonic clock for the tail guard and the response watchdog
        (overridden in tests to make the state machine deterministic)."""
        return _now_ms()

    @staticmethod
    def _show_state(state: TurnState) -> None:
        print(f"  [{STATE_ICON[state]}]", flush=True)

    # --- server events drive the non-audio half of the state machine ---------

    def on_server_event(self, ev) -> None:
        if ev.event == "speech_state":
            if ev.data.get("state") == "speaking":
                self.taker.agent_speaking()
            else:
                # tail guard first: the room is still decaying
                self.taker.agent_finished(self.now_ms())
        elif ev.event == "transcript":
            # heard, but no hint flow follows — nothing more is coming.
            # (Only from PROCESSING: never cut a playback short.)
            if not ev.data.get("wants_hint") and self.taker.state is TurnState.PROCESSING:
                self.taker.listen()
        elif ev.event == "hint_issued":
            # a hint with no spoken text; after TTS the tail guard already runs
            if self.taker.state is TurnState.PROCESSING:
                self.taker.listen()
        elif ev.event == "error" and self.taker.state is TurnState.PROCESSING:
            self.taker.listen()

    # --- the audio half: replaces device_sim's keyboard REPL -----------------

    async def _repl(self, ws) -> None:
        cfg = self.config
        print(
            f"hands-free: just talk (prefix {cfg.prefix_ms}ms, min speech "
            f"{cfg.min_speech_ms}ms, endpoint {cfg.silence_ms}ms silence). Ctrl-C to quit."
        )
        self._show_state(self.taker.state)
        async with MicStream(
            sample_rate=cfg.sample_rate,
            frame_samples=cfg.frame_samples,
            device=self.input_device,
        ) as mic:
            async for frame in mic.frames():
                if len(frame) != cfg.frame_bytes:  # short block on device stop
                    continue
                await self.on_frame(ws, frame)

    async def on_frame(self, ws, frame: bytes) -> None:
        """One mic frame through the gate; sends the turn when it ends."""
        now = self.now_ms()
        utterance = self.taker.feed(frame, now_ms=now)
        if utterance is not None:
            self._processing_since = now
            await self._send_audio(ws, utterance, self.config.sample_rate)
        elif (
            self.taker.state is TurnState.PROCESSING
            and now - self._processing_since > self.processing_timeout_ms
        ):
            # never stay deaf because a response went missing
            print("  [!] no response in time; listening again")
            self.taker.listen()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://localhost:8765")
    parser.add_argument(
        "--images",
        nargs="*",
        type=Path,
        default=[],
        help="worksheet images answered on capture_request (as in device_sim)",
    )
    parser.add_argument("--input-device", default=None, help="mic index or name substring")
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    parser.add_argument(
        "--timeout", type=float, default=90.0, help="seconds to wait for a response"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    for p in args.images:
        if not p.exists():
            sys.exit(f"image not found: {p} (run: python -m tutor.scripts.gen_assets)")

    device = args.input_device
    if isinstance(device, str) and device.isdigit():
        device = int(device)

    config = TurnConfig.from_settings(load_settings())
    try:
        voice = VoiceDevice(args.server, args.images, config, device, args.timeout)
        asyncio.run(voice.run())
    except KeyboardInterrupt:
        print("\nbye")
    except RuntimeError as e:  # missing silero-vad, sounddevice, or PortAudio
        sys.exit(str(e))
    except OSError as e:  # server not up / connection lost
        sys.exit(f"connection to {args.server} failed: {e}")


if __name__ == "__main__":
    main()
