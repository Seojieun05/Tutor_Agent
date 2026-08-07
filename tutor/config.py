"""Settings loaded from .env; echo mode when no XAI_API_KEY is present."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    xai_api_key: str = ""
    xai_base_url: str = "https://api.x.ai/v1"
    chat_model: str = "grok-4.5"
    # Reading the worksheet is the one job that can be moved to another model
    # without touching the pedagogy: "grok" (default) or "gemini".
    vision_provider: str = "grok"
    google_api_key: str = ""
    gemini_vision_model: str = "gemini-3.6-flash"
    tts_voice: str = "eve"
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    # HTTPS/WSS, for the phone camera only. getUserMedia runs in a secure
    # context or not at all, and a phone reaching the laptop by LAN IP has
    # none — `navigator.mediaDevices` is simply undefined there. Off unless a
    # cert is given, and it never replaces the plain port: localhost is already
    # a secure context, so the browser client keeps working as it is.
    tls_cert: Path | None = None
    tls_key: Path | None = None
    tls_port: int = 0  # 0 → ws_port + 1
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "knowledge.db")
    tutor_language: str = "ko"
    recog_conf_threshold: float = 0.6
    # A phone sends a multi-megapixel JPEG over Wi-Fi after letting autofocus
    # settle. Five seconds is enough on a good link and never on a busy one —
    # and a capture that times out is indistinguishable, to the student, from a
    # camera that cannot see the page.
    capture_timeout_s: float = 15.0
    # Where to drop every received frame, for when the tutor says it cannot read
    # the worksheet and you want to know what it was actually looking at.
    save_captures_dir: Path | None = None
    audio_sample_rate: int = 16000
    # Reasoning effort for the conversational LLM calls (see llm.client
    # FAST_PURPOSES). Empty string = the model's default.
    llm_reasoning_effort: str = "low"
    # Hands-free turn taking (laptop mic; see tutor/speech/turn.py)
    vad_threshold: float = 0.5
    vad_prefix_ms: int = 300
    vad_min_speech_ms: int = 250
    vad_silence_ms: int = 800
    vad_tail_guard_ms: int = 250

    @property
    def echo_mode(self) -> bool:
        return not self.xai_api_key

    @property
    def tls_enabled(self) -> bool:
        return self.tls_cert is not None and self.tls_key is not None

    @property
    def tls_listen_port(self) -> int:
        return self.tls_port or self.ws_port + 1


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    s = Settings()
    s.xai_api_key = os.getenv("XAI_API_KEY", "").strip()
    s.xai_base_url = os.getenv("XAI_BASE_URL", s.xai_base_url).strip() or s.xai_base_url
    s.chat_model = os.getenv("CHAT_MODEL", s.chat_model).strip() or s.chat_model
    s.vision_provider = (
        os.getenv("VISION_PROVIDER", s.vision_provider).strip().lower() or s.vision_provider
    )
    s.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    s.gemini_vision_model = (
        os.getenv("GEMINI_VISION_MODEL", s.gemini_vision_model).strip()
        or s.gemini_vision_model
    )
    s.tts_voice = os.getenv("TTS_VOICE", s.tts_voice).strip() or s.tts_voice
    s.ws_host = os.getenv("WS_HOST", s.ws_host).strip() or s.ws_host
    s.ws_port = int(os.getenv("WS_PORT", str(s.ws_port)))
    cert, key = os.getenv("TLS_CERT", "").strip(), os.getenv("TLS_KEY", "").strip()
    # both or neither: half a pair would start a listener that cannot handshake
    if cert and key:
        s.tls_cert, s.tls_key = Path(cert), Path(key)
    s.tls_port = int(os.getenv("TLS_PORT", str(s.tls_port)))
    s.db_path = Path(os.getenv("DB_PATH", str(s.db_path)))
    s.tutor_language = os.getenv("TUTOR_LANGUAGE", s.tutor_language).strip() or s.tutor_language
    s.recog_conf_threshold = float(os.getenv("RECOG_CONF_THRESHOLD", str(s.recog_conf_threshold)))
    s.capture_timeout_s = float(os.getenv("CAPTURE_TIMEOUT_S", str(s.capture_timeout_s)))
    captures = os.getenv("SAVE_CAPTURES_DIR", "").strip()
    s.save_captures_dir = Path(captures) if captures else None
    s.llm_reasoning_effort = os.getenv(
        "LLM_REASONING_EFFORT", s.llm_reasoning_effort
    ).strip()
    s.vad_threshold = float(os.getenv("VAD_THRESHOLD", str(s.vad_threshold)))
    s.vad_prefix_ms = int(os.getenv("VAD_PREFIX_MS", str(s.vad_prefix_ms)))
    s.vad_min_speech_ms = int(os.getenv("VAD_MIN_SPEECH_MS", str(s.vad_min_speech_ms)))
    s.vad_silence_ms = int(os.getenv("VAD_SILENCE_MS", str(s.vad_silence_ms)))
    s.vad_tail_guard_ms = int(os.getenv("VAD_TAIL_GUARD_MS", str(s.vad_tail_guard_ms)))
    return s
