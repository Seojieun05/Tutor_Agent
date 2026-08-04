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
    tts_voice: str = "eve"
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "knowledge.db")
    tutor_language: str = "ko"
    recog_conf_threshold: float = 0.6
    capture_timeout_s: float = 5.0
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


def load_settings(env_file: Path | None = None) -> Settings:
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    s = Settings()
    s.xai_api_key = os.getenv("XAI_API_KEY", "").strip()
    s.xai_base_url = os.getenv("XAI_BASE_URL", s.xai_base_url).strip() or s.xai_base_url
    s.chat_model = os.getenv("CHAT_MODEL", s.chat_model).strip() or s.chat_model
    s.tts_voice = os.getenv("TTS_VOICE", s.tts_voice).strip() or s.tts_voice
    s.ws_host = os.getenv("WS_HOST", s.ws_host).strip() or s.ws_host
    s.ws_port = int(os.getenv("WS_PORT", str(s.ws_port)))
    s.db_path = Path(os.getenv("DB_PATH", str(s.db_path)))
    s.tutor_language = os.getenv("TUTOR_LANGUAGE", s.tutor_language).strip() or s.tutor_language
    s.recog_conf_threshold = float(os.getenv("RECOG_CONF_THRESHOLD", str(s.recog_conf_threshold)))
    s.llm_reasoning_effort = os.getenv(
        "LLM_REASONING_EFFORT", s.llm_reasoning_effort
    ).strip()
    s.vad_threshold = float(os.getenv("VAD_THRESHOLD", str(s.vad_threshold)))
    s.vad_prefix_ms = int(os.getenv("VAD_PREFIX_MS", str(s.vad_prefix_ms)))
    s.vad_min_speech_ms = int(os.getenv("VAD_MIN_SPEECH_MS", str(s.vad_min_speech_ms)))
    s.vad_silence_ms = int(os.getenv("VAD_SILENCE_MS", str(s.vad_silence_ms)))
    s.vad_tail_guard_ms = int(os.getenv("VAD_TAIL_GUARD_MS", str(s.vad_tail_guard_ms)))
    return s
