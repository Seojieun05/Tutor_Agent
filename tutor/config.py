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
    # Set VERTEX_PROJECT to reach the same models through Google Cloud instead
    # of an AI Studio key — different billing, and a Cloud project can spend its
    # own credits. Auth is ADC (`gcloud auth application-default login`).
    # `global` is not a default to be tidy about: gemini-3.1-pro-preview is only
    # published there, and answers 404 in us-central1.
    vertex_project: str = ""
    vertex_location: str = "global"
    gemini_vision_model: str = "gemini-3.6-flash"
    # Flash, not pro. Pro phrased hints noticeably better — it reached for an
    # everyday analogy ("등식은 양팔저울과 같아서") where flash restated the rule —
    # but it took 8-12s per hint against flash's 4-6s, pushing a turn from 11s
    # to 18s. A tutor the student is waiting on is a worse tutor. Set
    # GEMINI_HINT_MODEL=gemini-3.1-pro-preview to trade back.
    gemini_hint_model: str = "gemini-3.6-flash"
    # Grading a spoken answer (evaluate) is the gate of the ANSWER turn — the
    # one that has to feel like conversation. EVAL_PROVIDER=gemini moves it to
    # flash; the verdict feeds the same deterministic policy either way, and a
    # FallbackLLM keeps Grok on standby exactly as with hints and vision.
    eval_provider: str = "grok"
    gemini_eval_model: str = "gemini-3.6-flash"
    # Consulted only before telling a student they are wrong, so its cost
    # lands on the turn that can afford it. Measured on an ordinary wrong
    # answer: this model adds ~4.5s where the chat model adds ~9s, and both
    # catch what the small grader missed — the cheaper second look wins.
    gemini_eval_second_model: str = "gemini-3.6-flash"
    # Diagnosing written work (estimate) sits on the WORK_CHECK critical path:
    # the student asked a yes/no question and is waiting on this call for the
    # verdict. ESTIMATE_PROVIDER=gemini moves it off grok; the estimate feeds
    # the same deterministic policy and the same sympy arithmetic check either
    # way, and a FallbackLLM keeps Grok on standby exactly as with eval.
    estimate_provider: str = "grok"
    gemini_estimate_model: str = "gemini-3.6-flash"
    # Writing the reference solution is the longest single call in the system
    # and the student waits on it twice: a work check needs it for the verdict,
    # and a spoken answer cannot be graded without it. Measured on three real
    # 4점 problems, two runs each: grok-4.5 27.5s and 6/6 right, flash 14.5s
    # and 6/6 right, flash-lite 8.0s and 1/6 — fast and wrong is not a trade,
    # so the small model is not offered here. Flash cuts the solution coarser
    # (6 steps where grok wrote 17), which is the real thing being traded: the
    # hint ladder walks these steps, so each one is a slightly bigger ask.
    solve_provider: str = "grok"
    gemini_solve_model: str = "gemini-3.6-flash"
    # The drawing hand runs while the tutor speaks, so its only deadline is
    # the end of the sentence: it has to land inside ~8s of speech, and a
    # short hint gives it half that. Measured on the same sketch, flash-lite
    # returns the same function, window and caption in 1.3s where flash takes
    # 4.0s — a narrow structured job the small model does just as well.
    illustrate_provider: str = "grok"
    gemini_illustrate_model: str = "gemini-3.5-flash-lite"
    # The intent gate is the FIRST model call of every ambiguous turn, and it
    # runs before any filler or stage line exists to cover it: the student has
    # stopped talking and nothing on screen moves until this returns. Measured
    # live on grok-4.5 it took 9.0s to decide the tutor should say NOTHING —
    # nine seconds of dead air for a one-word label. It is the same shape of
    # narrow classification the illustrator runs on flash-lite in 1.3s.
    intent_provider: str = "grok"
    gemini_intent_model: str = "gemini-3.5-flash-lite"
    tts_voice: str = "eve"
    # "ws" holds one bidirectional TTS socket open and reuses it per utterance
    # (~0.3s to first audio instead of ~1.3s per-request HTTP). Any websocket
    # failure falls back to the HTTP stream for that line, loudly.
    tts_transport: str = "http"
    # xAI's websocket TTS can trade a little prosody/lookahead for a shorter
    # time-to-first-audio. 2 is the documented aggressive/lowest-latency mode;
    # it affects only the streaming transport and leaves HTTP unchanged.
    tts_streaming_latency: int = 2
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
    # Barge-in: let the student cut the tutor off mid-sentence. Needs the
    # browser's echo cancellation to be doing its job — without it the tutor
    # interrupts itself. BARGE_IN_FRAMES is how much sustained speech it takes
    # (frames of 32 ms), and it is the knob to raise if that happens anyway.
    barge_in: bool = True
    barge_in_frames: int = 8

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
    s.hint_provider = (
        os.getenv("HINT_PROVIDER", s.hint_provider).strip().lower() or s.hint_provider
    )
    s.google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    s.vertex_project = os.getenv("VERTEX_PROJECT", "").strip()
    s.vertex_location = (
        os.getenv("VERTEX_LOCATION", s.vertex_location).strip() or s.vertex_location
    )
    s.gemini_hint_model = (
        os.getenv("GEMINI_HINT_MODEL", s.gemini_hint_model).strip() or s.gemini_hint_model
    )
    s.eval_provider = (
        os.getenv("EVAL_PROVIDER", s.eval_provider).strip().lower() or s.eval_provider
    )
    s.gemini_eval_model = (
        os.getenv("GEMINI_EVAL_MODEL", s.gemini_eval_model).strip() or s.gemini_eval_model
    )
    s.gemini_eval_second_model = (
        os.getenv("GEMINI_EVAL_SECOND_MODEL", s.gemini_eval_second_model).strip()
        or s.gemini_eval_second_model
    )
    s.estimate_provider = (
        os.getenv("ESTIMATE_PROVIDER", s.estimate_provider).strip().lower()
        or s.estimate_provider
    )
    s.gemini_estimate_model = (
        os.getenv("GEMINI_ESTIMATE_MODEL", s.gemini_estimate_model).strip()
        or s.gemini_estimate_model
    )
    s.solve_provider = (
        os.getenv("SOLVE_PROVIDER", s.solve_provider).strip().lower() or s.solve_provider
    )
    s.gemini_solve_model = (
        os.getenv("GEMINI_SOLVE_MODEL", s.gemini_solve_model).strip()
        or s.gemini_solve_model
    )
    s.illustrate_provider = (
        os.getenv("ILLUSTRATE_PROVIDER", s.illustrate_provider).strip().lower()
        or s.illustrate_provider
    )
    s.gemini_illustrate_model = (
        os.getenv("GEMINI_ILLUSTRATE_MODEL", s.gemini_illustrate_model).strip()
        or s.gemini_illustrate_model
    )
    s.intent_provider = (
        os.getenv("INTENT_PROVIDER", s.intent_provider).strip().lower()
        or s.intent_provider
    )
    s.gemini_intent_model = (
        os.getenv("GEMINI_INTENT_MODEL", s.gemini_intent_model).strip()
        or s.gemini_intent_model
    )
    s.tts_transport = (
        os.getenv("TTS_TRANSPORT", s.tts_transport).strip().lower() or s.tts_transport
    )
    s.tts_streaming_latency = max(0, min(2, int(os.getenv(
        "TTS_STREAMING_LATENCY", str(s.tts_streaming_latency)
    ))))
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
    s.barge_in = os.getenv("BARGE_IN", "1").strip().lower() not in {"0", "false", "no"}
    s.barge_in_frames = int(os.getenv("BARGE_IN_FRAMES", str(s.barge_in_frames)))
    return s
