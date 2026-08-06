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
    # What the tutor SAYS is the other job worth moving. Gemini carries Google's
    # LearnLM fine-tuning, which is specifically about following pedagogical
    # system instructions — "nudge, not the answer" as behaviour rather than as
    # a style note. The policy that picks the hint level and the leak guard that
    # checks the result are deterministic and do not move with it.
    hint_provider: str = "grok"
    google_api_key: str = ""
    gemini_vision_model: str = "gemini-3.6-flash"
    # Pro, not flash: phrasing a hint that teaches without leaking is the
    # hardest judgement in the pipeline, and it happens once per turn.
    gemini_hint_model: str = "gemini-3.1-pro-preview"
    tts_voice: str = "eve"
    ws_host: str = "0.0.0.0"
    ws_port: int = 8765
    db_path: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "knowledge.db")
    tutor_language: str = "ko"
    recog_conf_threshold: float = 0.6
    # A XIAO sends ~300 KB of UXGA JPEG over 2.4 GHz Wi-Fi, and it grabs two
    # throwaway frames first so the exposure is right. Five seconds is enough
    # on a good link and never on a busy one — and a capture that times out is
    # indistinguishable, to the student, from a camera that cannot see the page.
    capture_timeout_s: float = 15.0
    # Where to drop every received frame, for when the tutor says it cannot read
    # the worksheet and you want to know what it was actually looking at.
    save_captures_dir: Path | None = None
    # Something to say while the pipeline thinks. The delay is the whole design:
    # a turn that answers sooner than this stays silent, so the filler only ever
    # replaces a wait the student would have had anyway.
    filler_enabled: bool = True
    filler_delay_ms: int = 400
    # Pre-rendered phrases live here between runs; a warm cache means the filler
    # itself costs no TTS round trip.
    tts_cache_dir: Path | None = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "tts_cache"
    )
    # Where the worksheet picture comes from. "upload" is the browser page's
    # file picker (paste, drag or choose); "camera" borrows a XIAO on /camera.
    # Upload is the default because it is the one that always works: a 2 MP
    # fixed-focus sensor on a desk mount gave ~65 DPI across an A4 page, and
    # handwriting OCR wants 150+.
    input_mode: str = "upload"
    # A camera on a desk photographs mostly desk. Cropping to the page before
    # the reading model sees it is worth 2-3x in effective resolution, because
    # the model's tile budget stops being spent on an empty table.
    # WORKSHEET_ROI="x,y,w,h" (fractions) is exact; auto-crop is best-effort and
    # returns the whole frame whenever it is unsure. See tutor/vision/framing.py.
    worksheet_roi: tuple[float, float, float, float] | None = None
    auto_crop: bool = True
    crop_target_px: int = 1024
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
    s.vision_provider = (
        os.getenv("VISION_PROVIDER", s.vision_provider).strip().lower() or s.vision_provider
    )
    s.hint_provider = (
        os.getenv("HINT_PROVIDER", s.hint_provider).strip().lower() or s.hint_provider
    )
    s.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    s.gemini_hint_model = (
        os.getenv("GEMINI_HINT_MODEL", s.gemini_hint_model).strip() or s.gemini_hint_model
    )
    s.gemini_vision_model = (
        os.getenv("GEMINI_VISION_MODEL", s.gemini_vision_model).strip()
        or s.gemini_vision_model
    )
    s.tts_voice = os.getenv("TTS_VOICE", s.tts_voice).strip() or s.tts_voice
    s.ws_host = os.getenv("WS_HOST", s.ws_host).strip() or s.ws_host
    s.ws_port = int(os.getenv("WS_PORT", str(s.ws_port)))
    s.db_path = Path(os.getenv("DB_PATH", str(s.db_path)))
    s.tutor_language = os.getenv("TUTOR_LANGUAGE", s.tutor_language).strip() or s.tutor_language
    s.recog_conf_threshold = float(os.getenv("RECOG_CONF_THRESHOLD", str(s.recog_conf_threshold)))
    s.capture_timeout_s = float(os.getenv("CAPTURE_TIMEOUT_S", str(s.capture_timeout_s)))
    captures = os.getenv("SAVE_CAPTURES_DIR", "").strip()
    s.save_captures_dir = Path(captures) if captures else None
    s.filler_enabled = os.getenv("FILLER", "1").strip().lower() not in {"0", "false", "no"}
    s.filler_delay_ms = int(os.getenv("FILLER_DELAY_MS", str(s.filler_delay_ms)))
    cache = os.getenv("TTS_CACHE_DIR", str(s.tts_cache_dir or "")).strip()
    s.tts_cache_dir = Path(cache) if cache else None
    mode = os.getenv("INPUT_MODE", s.input_mode).strip().lower()
    s.input_mode = mode if mode in {"upload", "camera"} else s.input_mode
    from tutor.vision.framing import parse_roi

    s.worksheet_roi = parse_roi(os.getenv("WORKSHEET_ROI", ""))
    s.auto_crop = os.getenv("AUTO_CROP", "1").strip().lower() not in {"0", "false", "no"}
    s.crop_target_px = int(os.getenv("CROP_TARGET_PX", str(s.crop_target_px)))
    s.llm_reasoning_effort = os.getenv(
        "LLM_REASONING_EFFORT", s.llm_reasoning_effort
    ).strip()
    s.vad_threshold = float(os.getenv("VAD_THRESHOLD", str(s.vad_threshold)))
    s.vad_prefix_ms = int(os.getenv("VAD_PREFIX_MS", str(s.vad_prefix_ms)))
    s.vad_min_speech_ms = int(os.getenv("VAD_MIN_SPEECH_MS", str(s.vad_min_speech_ms)))
    s.vad_silence_ms = int(os.getenv("VAD_SILENCE_MS", str(s.vad_silence_ms)))
    s.vad_tail_guard_ms = int(os.getenv("VAD_TAIL_GUARD_MS", str(s.vad_tail_guard_ms)))
    return s
